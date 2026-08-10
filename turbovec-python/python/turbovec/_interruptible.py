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
batch. At a slice size around 1000 rows the Ctrl-C latency drops from seconds to
tens of milliseconds.

Throughput cost is not symmetric:

* **search** — negligible (~0 %). The extra work (one query-batch
  snapshot, per-slice dispatch) is tiny next to scanning the index.
* **add / add_with_ids** — a real multiplier on wall time, but a small
  absolute cost. Each chunked add pays a full input snapshot, per-slice
  validation, per-slice kernel dispatch, and (``add_with_ids``) an O(n)
  pre-existing-id check. At ``chunk_size=1000`` this measured
  roughly **2–7× the unchunked wall time** (the base add is fast, so the
  fixed per-slice Python overhead dominates the *ratio*; the exact figure
  swings with dim, batch size, and machine). In absolute terms the
  overhead is only on the order of ~1–10 µs per vector, and it buys
  interruptibility. For a throughput-critical one-shot bulk load where
  interruptibility does not matter, pass ``chunk_size=0`` to run the add
  whole at full speed.

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

import inspect

import numpy as np

from ._turbovec import IdMapIndex, TurboQuantIndex, _all_finite

# The default slice size is the single public constant `turbovec.BATCH_CHUNK_SIZE`
# (defined in the package __init__); `_default_chunk_size` reads it live so a
# reassignment takes effect without re-importing.

def _default_chunk_size() -> int:
    # Read the live package constant so `turbovec.BATCH_CHUNK_SIZE = n` takes
    # effect without re-importing. `import turbovec` is a cached dict hit.
    import turbovec

    # Coerce here too, not just on an explicit `chunk_size=` argument
    # (#345): the constant is documented as a user-settable knob, so a
    # float assigned to it must truncate like a float argument does
    # rather than surface as a `TypeError` from inside `range()` on an
    # add that has nothing wrong with it.
    return int(turbovec.BATCH_CHUNK_SIZE)


def _finite_ok(a: np.ndarray) -> bool:
    # The core's own `first_invalid_coord` predicate (rejects NaN, ±Inf, and
    # magnitudes >= 1e16), not a numpy restatement of it. The old
    # `np.all(np.abs(a) < 1e16)` materialized an `abs` array and a bool array
    # of the batch — 1.5 GB of temporaries for a 200k x 768 add — and read
    # every coordinate even when the first one was already bad. The native
    # form is one parallel pass with no temporaries that stops at the first
    # violation, and it cannot drift from what the core accepts. Empty arrays
    # are clean either way.
    return _all_finite(a)


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


def _ids_addable(index, ids: np.ndarray) -> bool:
    # Both id preconditions the core validates up front (id_map.rs), in one
    # native pass: no duplicate within the batch, and no id already in the
    # index. Chunking must reproduce that check so a batch the core would
    # reject is delegated whole and fails atomically, rather than committing
    # earlier slices up to the collision.
    #
    # This replaces `np.unique(ids).size == ids.shape[0]` — a full sort of
    # the id array — plus a Python-level `any(int(h) in index for h in ids)`,
    # which paid a GIL round trip and an index lock per id: 200k of them for
    # a 200k-row batch, and the single largest cost in the wrapper. The
    # native form short-circuits on the first violation.
    return index._batch_addable(ids)


def _transparent(wrapper, raw):
    """Make ``wrapper`` present itself as the kernel method it wraps.

    ``__signature__`` is set explicitly rather than left to
    ``__wrapped__`` (#340). ``inspect.signature`` follows ``__wrapped__``
    by default, so it reported the *native* signature and the documented
    ``chunk_size`` kwarg was invisible: ``help()`` and IDE completion hid
    it, and every signature-driven caller — ``pydantic.validate_call``,
    framework tool-arg introspection, CLI adapters — rejected it or
    silently forwarded it into ``**kwargs``. The explicit signature is
    the native one plus keyword-only ``chunk_size``, which is exactly
    what the wrapper accepts. ``__wrapped__`` stays set so callers that
    want the unwrapped kernel can still reach it.

    ``__name__`` / ``__qualname__`` are copied and ``__module__`` is
    taken from the owning class, so ``help()`` renders the public method
    instead of the internal ``_make_search.<locals>.search`` closure.
    """
    wrapper.__doc__ = raw.__doc__
    wrapper.__wrapped__ = raw
    for attr in ("__name__", "__qualname__"):
        try:
            setattr(wrapper, attr, getattr(raw, attr))
        except AttributeError:
            # A native method need not expose either; a missing one just
            # leaves the closure's own value in place.
            pass
    # A native `method_descriptor` carries no `__module__` of its own, so
    # take the owning class's — otherwise `help()` attributes the public
    # method to this private module.
    owner = getattr(raw, "__objclass__", None)
    if owner is not None:
        wrapper.__module__ = owner.__module__
    try:
        base = inspect.signature(raw)
    except (TypeError, ValueError):
        # No introspectable native signature (no `__text_signature__`).
        # Leave `__signature__` unset: the wrapper's own signature is
        # still a better answer than a wrong one.
        pass
    else:
        params = list(base.parameters.values())
        if not any(p.name == "chunk_size" for p in params):
            chunk_size = inspect.Parameter(
                "chunk_size", inspect.Parameter.KEYWORD_ONLY, default=None
            )
            # Keyword-only parameters must precede `**kwargs`. The native
            # kernels take no `**kwargs`, but a test double or a
            # third-party wrapper handed to `_make_*` may.
            at = next(
                (
                    i
                    for i, p in enumerate(params)
                    if p.kind is inspect.Parameter.VAR_KEYWORD
                ),
                len(params),
            )
            params.insert(at, chunk_size)
        wrapper.__signature__ = base.replace(parameters=params)
    wrapper._tv_chunk_wrapper = True
    return wrapper


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

    return _transparent(search, raw)


def _make_add(raw):
    def add(self, vectors, *, chunk_size=None):
        cs = _default_chunk_size() if chunk_size is None else int(chunk_size)
        # Every add chunks, including the first into an empty index.
        # This used to be conditional: while an index was warming up, the
        # add that carried it past the 1000-vector sample threshold fit
        # the calibration from its own batch, so slicing it would have
        # calibrated on a prefix and changed the whole index's
        # quantization. #474 removed automatic warm-up calibration — an
        # index is now either explicitly calibrated or identity, and no
        # add ever fits one — so there is no add that must run whole.
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
        # Slicing an add is always exact: an add never fits a
        # calibration, so each vector encodes identically no matter how
        # the batch is sliced.
        if _chunkable_2d(vectors, cs):
            snap = np.array(vectors)
            if _finite_ok(snap):
                n = snap.shape[0]
                for start in range(0, n, cs):
                    raw(self, snap[start : start + cs])
                return None
        return raw(self, vectors)

    return _transparent(add, raw)


def _make_add_with_ids(raw):
    def add_with_ids(self, vectors, ids, *, chunk_size=None):
        cs = _default_chunk_size() if chunk_size is None else int(chunk_size)
        # Chunk only a clean, canonically-typed batch: a
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
            _chunkable_2d(vectors, cs)
            and isinstance(ids, np.ndarray)
            and ids.ndim == 1
            and ids.dtype == np.uint64
            and ids.flags.c_contiguous
            and ids.shape[0] == vectors.shape[0]
        ):
            v_snap = np.array(vectors)
            i_snap = np.array(ids)
            if _finite_ok(v_snap) and _ids_addable(self, i_snap):
                n = v_snap.shape[0]
                for start in range(0, n, cs):
                    raw(self, v_snap[start : start + cs], i_snap[start : start + cs])
                return None
        return raw(self, vectors, ids)

    return _transparent(add_with_ids, raw)


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
