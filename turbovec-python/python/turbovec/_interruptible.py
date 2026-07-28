"""Interruptible long search/add via Python-side batch chunking (#216).

The Rust kernels already release the GIL (#186), but Python delivers
signals on the *main* thread — the thread parked inside the Rust call.
So a Ctrl-C pressed during a long batch ``search`` / ``add`` is queued and
not acted on until the kernel returns (measured on the originating issue:
a 12.9 s delay on a ~14 s search, 7.7 s on a ~9 s add). Painful in
notebooks and servers.

The cheap, core-agnostic fix: split a large batch into row-slices and call
the raw kernel once per slice. Control returns to Python between slices,
where a pending ``KeyboardInterrupt`` is serviced — so a queued Ctrl-C now
fires within roughly *one slice* rather than at the end of the whole
batch. Measured in the research pass: at ``chunk_size≈1000`` the Ctrl-C
latency dropped from ~13 s to ~57 ms for ~+5 % throughput.

These wrappers are installed over the native ``search`` / ``add`` /
``add_with_ids`` methods at import time. They are deliberately transparent:

* **Fall-through.** Anything too small to chunk, or that is not a plain
  C-contiguous 2-D ``float32`` ndarray (ids: 1-D ``uint64``), is passed
  straight to the raw kernel — every existing result and error path is
  preserved bit-for-bit.
* **Result equivalence.** For a batch that chunks, the per-slice results
  are concatenated in order, so the returned ``(scores, ids)`` are
  identical to a single unchunked call (per-query results are
  deterministic and ``effective_k`` depends only on ``k`` / index size /
  mask, not on how the queries are sliced).
* **Atomicity.** ``add`` is cumulative. Before chunking an add we
  pre-validate the whole batch the same way the core does its up-front
  check (finite, ``|x| < 1e16``; ids unique and length-matched); if the
  batch would be rejected, we hand the *entire* array to the raw kernel so
  it raises atomically with nothing committed — exactly as today. Only a
  genuine mid-batch *cancel* (KeyboardInterrupt) then leaves state behind,
  and it leaves it consistent: the completed slices are committed and
  queryable, and the exception propagates. That completed-slices-committed
  outcome is the *defined* behavior of a cancelled add — distinct from a
  torn *within-encode* write, which the core already guards (#89/#118).
  No such torn/partial-slice state is ever produced here.

Known blind spots (documented, narrow in practice):

* A *single* huge operation cannot be chunked. A one-query search
  (``nq == 1``) or a one-vector add is one indivisible kernel call and
  stays deaf to Ctrl-C for its duration. In practice a single query only
  approaches multi-second latency around ~10^8 indexed vectors, so this is
  not a concern for normal use.
* The *first* add into an empty index is not chunked. That first add fits
  and locks the index's TQ+ calibration from its batch, and every later
  add reuses it — so a later add is sliced with bit-identical results, but
  slicing the first add would fit calibration on only the first slice and
  silently change the whole index's quantization. Preserving exact
  equivalence therefore requires the initial bulk-load add to run whole;
  it stays deaf to Ctrl-C like a single huge op. Adds after it, and all
  searches, are chunked.

Making these single indivisible calls interruptible needs the core
cancellation poll (``PyErr::CheckSignals`` inside the hot loops) — the
deferred follow-up to this change.

Concurrency note: a chunked call takes the index lock once *per slice*,
not once for the whole batch, so it is not atomic with respect to other
threads mutating the *same* index concurrently (an unchunked call is).
Finish a batch before adding to the same index from another thread, or
pass ``chunk_size=0`` to opt a specific call out of chunking.
"""

from __future__ import annotations

import numpy as np

from ._turbovec import IdMapIndex, TurboQuantIndex

#: Default batch slice size (number of queries / vectors per raw kernel
#: call). Batches with more rows than this are chunked; smaller ones run
#: as a single call. Tunable globally via ``turbovec.BATCH_CHUNK_SIZE`` or
#: per call via the ``chunk_size=`` keyword. ~1000 keeps Ctrl-C latency to
#: tens of milliseconds for ~+5 % throughput.
BATCH_CHUNK_SIZE = 1000

# Core rejection bound, mirrored so a would-be-rejected batch is delegated
# whole (atomic error) instead of chunked. Compared in float64, which is
# strictly *more* conservative than the core's float32 `|x| < 1e16` at the
# exact boundary, so we never deem "clean" a value the core would reject.
_MAX_MAGNITUDE = 1e16


def _default_chunk_size() -> int:
    # Read the live module default so `turbovec.BATCH_CHUNK_SIZE = n` takes
    # effect without re-importing. `import turbovec` is a cached dict hit.
    import turbovec

    return turbovec.BATCH_CHUNK_SIZE


def _finite_ok(a: np.ndarray) -> bool:
    # Mirror the core's `first_invalid_coord` predicate `!(|x| < 1e16)`
    # (rejects NaN, ±Inf, and magnitudes >= 1e16). Empty arrays are clean.
    return bool(np.all(np.abs(a) < _MAX_MAGNITUDE))


def _chunkable_2d(a: object, cs: int) -> bool:
    return (
        cs > 0
        and isinstance(a, np.ndarray)
        and a.ndim == 2
        and a.dtype == np.float32
        and a.shape[0] > cs
    )


def _make_search(raw):
    def search(self, queries, k, *, chunk_size=None, **kwargs):
        cs = _default_chunk_size() if chunk_size is None else int(chunk_size)
        # Chunk only a plain 2-D float32 ndarray with more rows than one
        # slice; anything else (and a would-be-rejected non-finite batch)
        # falls through to the raw kernel, so every existing result and
        # error path — including the exact bad-value row index — is
        # preserved.
        if _chunkable_2d(queries, cs):
            # Snapshot the whole batch once, up front, so every slice reads
            # one coherent version of the query array even though each slice
            # is a separate kernel call that snapshots its own view at its
            # own time. Without this, another thread stomping the source
            # array mid-search could split the result across old and new
            # query rows — weakening the "search sees one version of the
            # buffer" data-integrity guarantee (#108). The rows are never
            # torn regardless (each kernel snapshot is atomic); this keeps
            # the whole batch coherent too.
            snap = np.array(queries)
            if _finite_ok(snap):
                n = snap.shape[0]
                scores_parts = []
                ids_parts = []
                for start in range(0, n, cs):
                    # A signal queued on the main thread is serviced here,
                    # between slices — a KeyboardInterrupt fires within one
                    # slice rather than at the end of the batch.
                    s, i = raw(self, snap[start : start + cs], k, **kwargs)
                    scores_parts.append(s)
                    ids_parts.append(i)
                return (
                    np.concatenate(scores_parts, axis=0),
                    np.concatenate(ids_parts, axis=0),
                )
        return raw(self, queries, k, **kwargs)

    search.__doc__ = raw.__doc__
    search.__wrapped__ = raw
    search._tv_chunk_wrapper = True
    return search


def _make_add(raw):
    def add(self, vectors, *, chunk_size=None):
        cs = _default_chunk_size() if chunk_size is None else int(chunk_size)
        # Chunk only once the index is non-empty. The *first* add into an
        # empty index fits and locks the TQ+ calibration from that batch;
        # every later add reuses it, so a later add encodes each vector
        # identically no matter how it is sliced — but chunking the first
        # add would fit calibration on only the first slice and silently
        # change the whole index's quantization. So the first (calibrating)
        # add runs whole (and stays deaf to Ctrl-C, like a single huge op);
        # incremental adds afterward chunk with bit-identical results.
        #
        # Pre-validate so a batch the core would reject is added atomically
        # (whole array, nothing committed) rather than committing earlier
        # slices before a later slice's value trips the check.
        if not (len(self) > 0 and _chunkable_2d(vectors, cs) and _finite_ok(vectors)):
            return raw(self, vectors)
        n = vectors.shape[0]
        for start in range(0, n, cs):
            raw(self, vectors[start : start + cs])
        return None

    add.__doc__ = raw.__doc__
    add.__wrapped__ = raw
    add._tv_chunk_wrapper = True
    return add


def _make_add_with_ids(raw):
    def add_with_ids(self, vectors, ids, *, chunk_size=None):
        cs = _default_chunk_size() if chunk_size is None else int(chunk_size)
        # Chunk only a non-empty index (the first add locks calibration —
        # see `_make_add`) with a clean, canonically-typed batch: finite
        # values, ids a 1-D uint64 array of matching length with no
        # duplicates. Anything else (empty index, bad values,
        # duplicate/pre-existing ids, length mismatch) is delegated whole
        # so the core's up-front validation rejects it atomically — and no
        # earlier slice is committed first.
        if (
            len(self) > 0
            and _chunkable_2d(vectors, cs)
            and isinstance(ids, np.ndarray)
            and ids.ndim == 1
            and ids.dtype == np.uint64
            and ids.shape[0] == vectors.shape[0]
            and _finite_ok(vectors)
            and np.unique(ids).size == ids.shape[0]
        ):
            n = vectors.shape[0]
            for start in range(0, n, cs):
                raw(self, vectors[start : start + cs], ids[start : start + cs])
            return None
        return raw(self, vectors, ids)

    add_with_ids.__doc__ = raw.__doc__
    add_with_ids.__wrapped__ = raw
    add_with_ids._tv_chunk_wrapper = True
    return add_with_ids


def install() -> None:
    """Install the chunking wrappers over the native kernel methods.

    Idempotent: a wrapper already installed is left in place (so a second
    import cannot double-wrap and treat a wrapper as the raw kernel).
    """
    if not getattr(TurboQuantIndex.search, "_tv_chunk_wrapper", False):
        TurboQuantIndex.search = _make_search(TurboQuantIndex.search)
    if not getattr(TurboQuantIndex.add, "_tv_chunk_wrapper", False):
        TurboQuantIndex.add = _make_add(TurboQuantIndex.add)
    if not getattr(IdMapIndex.search, "_tv_chunk_wrapper", False):
        IdMapIndex.search = _make_search(IdMapIndex.search)
    if not getattr(IdMapIndex.add_with_ids, "_tv_chunk_wrapper", False):
        IdMapIndex.add_with_ids = _make_add_with_ids(IdMapIndex.add_with_ids)
