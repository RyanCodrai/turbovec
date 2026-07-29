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
batch. At ``chunk_size≈1000`` the Ctrl-C latency drops from seconds to
tens of milliseconds.

Throughput cost is not symmetric:

* **search** — negligible (~0 %). The extra work (one query-batch
  snapshot, per-slice dispatch) is tiny next to scanning the index.
* **add / add_with_ids** — a real multiplier on wall time, but a small
  absolute cost. Each chunked add pays a full input snapshot, per-slice
  validation, per-slice kernel dispatch, and (``add_with_ids``) an O(n)
  pre-existing-id check. At the default ``chunk_size=1000`` this measured
  roughly **2–7× the unchunked wall time** (the base add is fast, so the
  fixed per-slice Python overhead dominates the *ratio*; the exact figure
  swings with dim, batch size, and machine). In absolute terms the
  overhead is only on the order of ~1–10 µs per vector, and it buys
  interruptibility. For a throughput-critical one-shot bulk load where
  interruptibility does not matter, pass ``chunk_size=0`` to run the add
  whole at full speed. (An add into a still-warming-up index already runs
  whole regardless — see the blind spots below.)

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
The slice size comes from ``chunk_size=`` (or the ``turbovec.BATCH_CHUNK_SIZE``
default). It is coerced with ``int(...)``, so a float is truncated; any value
``<= 0`` (including a negative) simply disables chunking and runs the call
whole — the same as ``chunk_size=0``.

* **Atomicity.** ``add`` is cumulative. Before chunking an add we
  pre-validate the whole batch the same way the core does its up-front
  check (finite, ``|x| < 1e16``; ids unique, length-matched, and none
  already present in the index); if the
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
* Adds into an index that is still warming up (fewer than 1000 vectors
  seen, ``index.calibration_state == "warming_up"``) are not chunked. The
  add that carries the index past that threshold fits and locks the TQ+
  calibration from its batch, and every later add reuses it — so a later
  add is sliced with bit-identical results, but slicing the calibrating
  add would fit calibration on only the first slice and silently change
  the whole index's quantization. Preserving exact equivalence therefore
  requires the initial bulk-load add to run whole; it stays deaf to
  Ctrl-C like a single huge op. Adds after it, and all searches, are
  chunked.

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

# The default slice size is the single public constant `turbovec.BATCH_CHUNK_SIZE`
# (defined in the package __init__); `_default_chunk_size` reads it live so a
# reassignment takes effect without re-importing.

# Core rejection bound, mirrored so a would-be-rejected batch is delegated
# whole (atomic error) instead of chunked. Stored as float32 so the numpy
# compare below matches the core's `|x| < 1e16` in f32 exactly: under NEP50
# (numpy >= 2) an f32 array compared to this f32 scalar stays in f32; under
# legacy promotion it also stays f32 (same dtype). Either way we never deem
# "clean" a value the core would reject.
_MAX_MAGNITUDE = np.float32(1e16)


def _default_chunk_size() -> int:
    # Read the live package constant so `turbovec.BATCH_CHUNK_SIZE = n` takes
    # effect without re-importing. `import turbovec` is a cached dict hit.
    import turbovec

    return turbovec.BATCH_CHUNK_SIZE


def _finite_ok(a: np.ndarray) -> bool:
    # Mirror the core's `first_invalid_coord` predicate `!(|x| < 1e16)`
    # (rejects NaN, ±Inf, and magnitudes >= 1e16). Empty arrays are clean.
    return bool(np.all(np.abs(a) < _MAX_MAGNITUDE))


def _calibration_settled(index) -> bool:
    """Whether slicing an add is safe: the index already holds a locked TQ+
    calibration, so each vector encodes identically no matter how the batch
    is sliced. While the index is warming up the calibrating add must run
    whole (see the module docstring)."""
    return index.calibration_state != "warming_up"


def _chunkable_2d(a: object, cs: int) -> bool:
    # C-contiguity is required: a row-slice of a non-contiguous array is
    # rejected by the raw kernel, and snapshotting one would silently
    # launder it into an acceptable copy — so a non-contiguous batch must
    # fall through to the kernel (which raises) instead of being chunked.
    return (
        cs > 0
        and isinstance(a, np.ndarray)
        and a.ndim == 2
        and a.dtype == np.float32
        and a.flags.c_contiguous
        and a.shape[0] > cs
    )


def _snapshot_kwargs(kwargs):
    """Snapshot ndarray-valued kwargs (``mask`` / ``allowlist``) once so
    every slice reads one coherent version, mirroring the query snapshot
    (#108). Returns ``(snapped, ok)``; ``ok`` is ``False`` if any ndarray
    kwarg is non-C-contiguous — the raw kernel rejects those, and copying
    one would launder it, so the caller must fall through so the kernel
    raises the contiguity error unchanged.
    """
    snapped = {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            if not value.flags.c_contiguous:
                return kwargs, False
            snapped[key] = np.array(value)
        else:
            snapped[key] = value
    return snapped, True


def _any_id_present(index, ids: np.ndarray) -> bool:
    # Mirror the core's up-front "id already in the index" rejection
    # (id_map.rs) so a batch colliding with an existing id is delegated
    # whole and fails atomically — not committed slice-by-slice up to the
    # collision. `__contains__` is O(1) per id. Iterate the array lazily
    # (one numpy scalar at a time, converted to a Python int) rather than
    # materializing a full `.tolist()` — at hundreds of thousands of ids
    # the transient list would be a multi-MB memory spike.
    return any(int(handle) in index for handle in ids)


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
            # the whole batch coherent too. The same coherence must cover a
            # ``mask`` / ``allowlist`` array, so snapshot those alongside.
            snap = np.array(queries)
            snap_kwargs, kwargs_ok = _snapshot_kwargs(kwargs)
            if kwargs_ok and _finite_ok(snap):
                n = snap.shape[0]
                scores_parts = []
                ids_parts = []
                for start in range(0, n, cs):
                    # A signal queued on the main thread is serviced here,
                    # between slices — a KeyboardInterrupt fires within one
                    # slice rather than at the end of the batch.
                    s, i = raw(self, snap[start : start + cs], k, **snap_kwargs)
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
        # Chunk only once the index's TQ+ calibration is settled. While
        # the index is warming up, the add that carries it past the
        # 1000-vector sample threshold fits the calibration every stored
        # vector is then encoded in — and it fits it from the batch it is
        # handed, so slicing that add would calibrate on a prefix and
        # silently change the whole index's quantization. So the
        # calibrating add runs whole (and stays deaf to Ctrl-C, like a
        # single huge op); adds afterward reuse the locked calibration and
        # chunk with bit-identical results.
        #
        # Snapshot once, up front, then validate and slice the snapshot —
        # so a thread mutating the source array mid-batch cannot make
        # pre-validation pass and then feed a later slice different (or
        # invalid) data after earlier slices have committed (the add-side
        # analogue of the query snapshot, #108). Pre-validating the
        # snapshot means a batch the core would reject is added atomically
        # (whole array via the raw kernel, nothing committed) rather than
        # committing earlier slices before a later value trips the check.
        # When not chunking, fall through with the ORIGINAL array so the
        # kernel's error paths stay byte-identical.
        if _calibration_settled(self) and _chunkable_2d(vectors, cs):
            snap = np.array(vectors)
            if _finite_ok(snap):
                n = snap.shape[0]
                for start in range(0, n, cs):
                    raw(self, snap[start : start + cs])
                return None
        return raw(self, vectors)

    add.__doc__ = raw.__doc__
    add.__wrapped__ = raw
    add._tv_chunk_wrapper = True
    return add


def _make_add_with_ids(raw):
    def add_with_ids(self, vectors, ids, *, chunk_size=None):
        cs = _default_chunk_size() if chunk_size is None else int(chunk_size)
        # Chunk only an index whose calibration is settled (the
        # calibrating add must not be sliced — see `_make_add`) with a
        # clean, canonically-typed batch: a
        # C-contiguous 1-D uint64 id array of matching length, finite
        # vectors, no duplicate ids within the batch, AND no id already
        # present in the index. The core validates ALL of these up front
        # (id_map.rs) so a rejected batch commits nothing; chunking must
        # reproduce that, so any failing condition delegates the whole
        # batch to the raw kernel (with the ORIGINAL arrays, keeping error
        # paths byte-identical). In particular the pre-existing-id check is
        # essential: without it, a batch whose colliding id sits in a late
        # slice would commit the earlier slices before raising.
        #
        # Vectors and ids are snapshotted once, up front, before validation
        # (the add-side analogue of the query snapshot, #108): a thread
        # mutating a source array mid-batch cannot make validation pass and
        # then feed a later slice different/invalid data after earlier
        # slices committed.
        if (
            _calibration_settled(self)
            and _chunkable_2d(vectors, cs)
            and isinstance(ids, np.ndarray)
            and ids.ndim == 1
            and ids.dtype == np.uint64
            and ids.flags.c_contiguous
            and ids.shape[0] == vectors.shape[0]
        ):
            v_snap = np.array(vectors)
            i_snap = np.array(ids)
            if (
                _finite_ok(v_snap)
                and np.unique(i_snap).size == i_snap.shape[0]
                and not _any_id_present(self, i_snap)
            ):
                n = v_snap.shape[0]
                for start in range(0, n, cs):
                    raw(self, v_snap[start : start + cs], i_snap[start : start + cs])
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
