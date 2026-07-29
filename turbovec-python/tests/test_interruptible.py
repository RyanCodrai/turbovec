"""Interruptible long search/add via Python-side batch chunking (#216).

The Rust kernels release the GIL, but Python services signals on the main
thread — the one parked inside the kernel — so a Ctrl-C during a long
batch search/add is queued until the whole call returns. These wrappers
split a large batch into row-slices and call the raw kernel per slice, so
control returns to Python (and a queued KeyboardInterrupt fires) between
slices.

Covered here:
  * chunked search/add/add_with_ids return results identical to a single
    unchunked call (concatenation preserves order, ids, and scores);
  * a cancel between slices is delivered within ~one slice, not at the end,
    and a cancelled add leaves the index committed at exactly the
    completed-slice count (consistent and queryable);
  * the transparent fall-through: bad or unchunkable input hits the raw
    kernel unchanged, so error paths and the single-huge-op blind spot
    behave exactly as before.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex, _interruptible

DIM = 64


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """`n` unit-norm float32 rows, C-contiguous."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return np.ascontiguousarray(v, dtype=np.float32)


# ---------------------------------------------------------------------------
# Result equivalence: chunked == unchunked, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cs", [1, 7, 100, 999, 1000])
def test_chunked_search_matches_unchunked_turboquant(cs):
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(3000, seed=0))
    q = _rand(2500, seed=1)  # > cs for the small cs values → really chunks
    s_ref, i_ref = idx.search(q, 10, chunk_size=0)  # single unchunked call
    s_chunk, i_chunk = idx.search(q, 10, chunk_size=cs)
    np.testing.assert_array_equal(s_ref, s_chunk)
    np.testing.assert_array_equal(i_ref, i_chunk)


def test_chunked_search_matches_unchunked_with_mask():
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(2000, seed=0))
    mask = np.zeros(len(idx), dtype=bool)
    mask[::3] = True  # a non-trivial subset
    q = _rand(1500, seed=1)
    s_ref, i_ref = idx.search(q, 8, mask=mask, chunk_size=0)
    s_chunk, i_chunk = idx.search(q, 8, mask=mask, chunk_size=250)
    np.testing.assert_array_equal(s_ref, s_chunk)
    np.testing.assert_array_equal(i_ref, i_chunk)


def test_chunked_search_matches_unchunked_idmap_with_allowlist():
    idx = IdMapIndex(DIM, 4)
    ids = np.arange(2000, dtype=np.uint64) + 100
    idx.add_with_ids(_rand(2000, seed=0), ids)
    allow = np.ascontiguousarray(ids[::2])  # half the ids
    q = _rand(1500, seed=1)
    s_ref, i_ref = idx.search(q, 8, allowlist=allow, chunk_size=0)
    s_chunk, i_chunk = idx.search(q, 8, allowlist=allow, chunk_size=200)
    np.testing.assert_array_equal(s_ref, s_chunk)
    np.testing.assert_array_equal(i_ref, i_chunk)


def test_chunked_add_matches_unchunked():
    # The first add locks the TQ+ calibration; the second add (the one we
    # chunk) reuses it, so slicing must be bit-identical to one call.
    seed = _rand(500, seed=1)
    data = _rand(2500, seed=5)

    ref = TurboQuantIndex(DIM, 4)
    ref.add(seed)
    ref.add(data, chunk_size=0)
    chunked = TurboQuantIndex(DIM, 4)
    chunked.add(seed)
    chunked.add(data, chunk_size=256)

    assert len(ref) == len(chunked) == 3000
    # Strongest form of equivalence: the two indices are byte-identical.
    assert ref.to_bytes() == chunked.to_bytes()


def test_chunked_add_with_ids_matches_unchunked():
    seed = _rand(500, seed=1)
    seed_ids = np.arange(500, dtype=np.uint64)
    data = _rand(2500, seed=5)
    data_ids = np.arange(2500, dtype=np.uint64) + 500

    ref = IdMapIndex(DIM, 4)
    ref.add_with_ids(seed, seed_ids)
    ref.add_with_ids(data, data_ids, chunk_size=0)
    chunked = IdMapIndex(DIM, 4)
    chunked.add_with_ids(seed, seed_ids)
    chunked.add_with_ids(data, data_ids, chunk_size=256)

    assert len(ref) == len(chunked) == 3000
    assert ref.to_bytes() == chunked.to_bytes()


def test_first_add_into_empty_index_is_not_chunked():
    """Documents the calibration blind spot: the first (calibrating) add
    runs whole; a later add is sliced. Observed through the raw-kernel
    call count."""
    from turbovec import _interruptible

    raw = TurboQuantIndex.add.__wrapped__
    calls = {"n": 0}

    def counting(self, vectors):
        calls["n"] += 1
        return raw(self, vectors)

    chunked_add = _interruptible._make_add(counting)
    idx = TurboQuantIndex(DIM, 4)
    big = _rand(2500, seed=0)

    # Empty index: first add runs as a single call (calibration).
    chunked_add(idx, big, chunk_size=500)
    assert calls["n"] == 1

    # Non-empty index: the next add of 2500 with cs=500 slices into 5.
    calls["n"] = 0
    chunked_add(idx, _rand(2500, seed=1), chunk_size=500)
    assert calls["n"] == 5


# ---------------------------------------------------------------------------
# Cancel between slices: delivered within ~one slice + add atomicity
# ---------------------------------------------------------------------------


def test_add_cancel_commits_completed_slices_and_stays_queryable():
    """Deterministic model of a queued Ctrl-C serviced at a slice boundary.

    A shim over the real kernel raises KeyboardInterrupt *before* committing
    the 4th slice — exactly what a signal serviced between slice 3 and 4
    does. The first three slices must be committed (and searchable); the
    loop must not run to the end.
    """
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(300, seed=7))  # first add locks calibration → later adds chunk
    base = len(idx)
    raw_add = TurboQuantIndex.add.__wrapped__  # the native kernel

    cs = 500
    cancel_at = 3
    calls = {"n": 0}

    def shim(self, vectors):
        if calls["n"] == cancel_at:
            raise KeyboardInterrupt  # queued signal serviced here
        calls["n"] += 1
        return raw_add(self, vectors)

    chunked_add = _interruptible._make_add(shim)
    data = _rand(10 * cs, seed=0)  # 10 slices available

    with pytest.raises(KeyboardInterrupt):
        chunked_add(idx, data, chunk_size=cs)

    # Cancelled within one slice of the signal — not at the end of 10.
    assert calls["n"] == cancel_at
    assert len(idx) == base + cancel_at * cs
    # The index is consistent and queryable at the partial count.
    scores, _ = idx.search(_rand(5, seed=1), 4, chunk_size=0)
    assert scores.shape == (5, 4)


def test_add_with_ids_cancel_commits_completed_slices():
    idx = IdMapIndex(DIM, 4)
    seed_ids = np.arange(300, dtype=np.uint64) + 10_000
    idx.add_with_ids(_rand(300, seed=7), seed_ids)  # locks calibration
    base = len(idx)
    raw = IdMapIndex.add_with_ids.__wrapped__
    cs = 400
    cancel_at = 2
    calls = {"n": 0}

    def shim(self, vectors, ids):
        if calls["n"] == cancel_at:
            raise KeyboardInterrupt
        calls["n"] += 1
        return raw(self, vectors, ids)

    chunked = _interruptible._make_add_with_ids(shim)
    n = 8 * cs
    data = _rand(n, seed=0)
    ids = np.arange(n, dtype=np.uint64)

    with pytest.raises(KeyboardInterrupt):
        chunked(idx, data, ids, chunk_size=cs)

    assert len(idx) == base + cancel_at * cs
    # The committed ids are exactly the first two slices' — queryable.
    assert 0 in idx and (cancel_at * cs - 1) in idx
    assert (cancel_at * cs) not in idx


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="delivering a real SIGINT from a helper thread is Unix-only; "
    "Windows only routes Ctrl-C (CTRL_C_EVENT) to the main thread. The "
    "chunking feature works cross-platform and is covered by the "
    "deterministic __wrapped__-based interrupt tests above.",
)
def test_real_sigint_interrupts_search_mid_batch():
    """A real SIGINT from a helper thread is delivered mid-batch, well
    before the full chunked search would have finished.

    Wall-clock timing is inherently jittery (CI load, scheduler), so this
    uses a generous margin (interrupt must land in the first ~80% of the
    batch, firing at ~20% in) and retries a few times before failing — a
    single unlucky scheduling hiccup should not flake the suite.
    """
    idx = TurboQuantIndex(128, 4)
    idx.add(_rand(100_000, 128, seed=0))
    idx.prepare()
    q = _rand(40_000, 128, seed=1)
    cs = 1000  # 40 slices

    # Baseline: uninterrupted full run (median of two to damp jitter).
    def _full():
        t = time.perf_counter()
        idx.search(q, 10, chunk_size=cs)
        return time.perf_counter() - t

    t_full = min(_full(), _full())

    last = None
    for _attempt in range(3):
        old = signal.signal(signal.SIGINT, signal.default_int_handler)
        fire_after = max(0.1, t_full * 0.2)
        try:
            def fire():
                time.sleep(fire_after)
                os.kill(os.getpid(), signal.SIGINT)

            th = threading.Thread(target=fire)
            th.start()
            interrupted = False
            t1 = time.perf_counter()
            try:
                idx.search(q, 10, chunk_size=cs)
            except KeyboardInterrupt:
                interrupted = True
            elapsed = time.perf_counter() - t1
            th.join()
        finally:
            signal.signal(signal.SIGINT, old)

        last = (interrupted, elapsed, t_full)
        # Delivered mid-batch, not queued to the end — generous 80% bound.
        if interrupted and elapsed < t_full * 0.8:
            break
    else:
        assert False, f"SIGINT not delivered mid-batch after retries: {last}"


# ---------------------------------------------------------------------------
# Transparent fall-through: unchunkable / bad input behaves as before
# ---------------------------------------------------------------------------


def test_disable_chunking_with_zero():
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(2000, seed=0))
    q = _rand(1500, seed=1)
    a = idx.search(q, 5, chunk_size=0)
    b = idx.search(q, 5, chunk_size=1000)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_global_default_is_live_and_tunable():
    import turbovec

    assert _interruptible._default_chunk_size() == turbovec.BATCH_CHUNK_SIZE
    old = turbovec.BATCH_CHUNK_SIZE
    try:
        turbovec.BATCH_CHUNK_SIZE = 123
        assert _interruptible._default_chunk_size() == 123
    finally:
        turbovec.BATCH_CHUNK_SIZE = old


def test_single_query_blind_spot_still_correct():
    """nq==1 cannot be chunked; it must still return the right result."""
    idx = TurboQuantIndex(DIM, 4)
    data = _rand(500, seed=0)
    idx.add(data)
    q = _rand(1, seed=1)
    scores, ids = idx.search(q, 5)
    assert scores.shape == (1, 5)
    # Top-1 is the true nearest by exact dot product over the raw data.
    exact_best = int(np.argmax(data @ q[0]))
    # Quantized search should rank the exact best highly; assert it is
    # among the returned top-5 (quantization is lossy but faithful here).
    assert exact_best in ids[0].tolist()


def test_bad_input_falls_through_to_raw_error():
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(10, seed=0))
    # A Python list is not a float32 ndarray → raw kernel raises, unchanged.
    with pytest.raises((TypeError, ValueError)):
        idx.search([[0.0] * DIM] * 5, 3)


def test_nan_add_is_atomic_nothing_committed():
    """A NaN deep in the batch must reject the whole add (0 committed),
    exactly like the unchunked kernel — chunking pre-validates so it does
    not commit earlier clean slices first."""
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(300, seed=7))  # non-empty → the bad add is chunk-eligible
    base = len(idx)
    data = _rand(2500, seed=0)
    data[2200, 3] = np.nan  # invalid value in what would be a later slice
    with pytest.raises(ValueError):
        idx.add(data, chunk_size=500)
    # Pre-validation delegated the whole batch → nothing from it committed.
    assert len(idx) == base


def test_duplicate_ids_add_is_atomic_nothing_committed():
    idx = IdMapIndex(DIM, 4)
    idx.add_with_ids(_rand(300, seed=7), np.arange(300, dtype=np.uint64) + 10_000)
    base = len(idx)
    data = _rand(2500, seed=0)
    ids = np.arange(2500, dtype=np.uint64)
    ids[2300] = ids[10]  # duplicate spanning two would-be slices
    with pytest.raises(ValueError):
        idx.add_with_ids(data, ids, chunk_size=500)
    assert len(idx) == base


# ---------------------------------------------------------------------------
# F1: an id colliding with one ALREADY in the index must reject atomically
# (within-batch uniqueness alone is not enough — the core also rejects
# pre-existing ids up front, so chunking must too).
# ---------------------------------------------------------------------------


def test_add_with_ids_colliding_with_existing_id_is_atomic():
    idx = IdMapIndex(DIM, 4)
    seed_ids = np.arange(300, dtype=np.uint64) + 10_000
    idx.add_with_ids(_rand(300, seed=7), seed_ids)
    base = len(idx)

    data = _rand(2500, seed=0)
    ids = np.arange(2500, dtype=np.uint64)  # 0..2499, disjoint from seed
    # A single id, in what would be a LATE slice, collides with the index.
    ids[2300] = 10_050  # already present (seed id)
    assert np.unique(ids).size == ids.size  # still unique *within* the batch

    with pytest.raises(ValueError):
        idx.add_with_ids(data, ids, chunk_size=500)
    # The whole batch was delegated and rejected — no earlier slice landed.
    assert len(idx) == base


# ---------------------------------------------------------------------------
# F2: add snapshots its inputs once, so a source-array mutation landing
# after validation cannot feed later slices different/invalid data.
# ---------------------------------------------------------------------------


def test_add_snapshots_source_against_concurrent_mutation():
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(300, seed=7))  # lock calibration → later add chunks

    data = _rand(2000, seed=0)
    original = data.copy()
    raw = TurboQuantIndex.add.__wrapped__

    def stomping(self, vectors):
        # A concurrent writer stomps the SOURCE array after the wrapper
        # snapshotted it; the slices must still see the pre-mutation data.
        data[:] = _rand(2000, seed=99)
        return raw(self, vectors)

    chunked_add = _interruptible._make_add(stomping)
    chunked_add(idx, data, chunk_size=500)

    ref = TurboQuantIndex(DIM, 4)
    ref.add(_rand(300, seed=7))
    ref.add(original, chunk_size=0)
    assert idx.to_bytes() == ref.to_bytes()


def test_add_with_ids_snapshots_source_against_concurrent_mutation():
    idx = IdMapIndex(DIM, 4)
    seed_ids = np.arange(300, dtype=np.uint64) + 10_000
    idx.add_with_ids(_rand(300, seed=7), seed_ids)

    data = _rand(2000, seed=0)
    ids = np.arange(2000, dtype=np.uint64)
    original_v, original_i = data.copy(), ids.copy()
    raw = IdMapIndex.add_with_ids.__wrapped__

    def stomping(self, vectors, the_ids):
        data[:] = _rand(2000, seed=99)
        ids[:] = np.arange(2000, dtype=np.uint64) + 500_000
        return raw(self, vectors, the_ids)

    chunked = _interruptible._make_add_with_ids(stomping)
    chunked(idx, data, ids, chunk_size=500)

    ref = IdMapIndex(DIM, 4)
    ref.add_with_ids(_rand(300, seed=7), seed_ids)
    ref.add_with_ids(original_v, original_i, chunk_size=0)
    assert idx.to_bytes() == ref.to_bytes()


# ---------------------------------------------------------------------------
# F3: a non-C-contiguous batch must fall through to the raw kernel (which
# rejects it) at BOTH sizes — the snapshot must not launder a strided view
# into an acceptable copy for large batches.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nrows", [2400, 100])  # would-chunk, and small
def test_noncontiguous_query_falls_through_both_sizes(nrows):
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(500, seed=0))
    view = _rand(nrows * 2, seed=1)[::2]  # strided → non-contiguous
    assert not view.flags.c_contiguous
    with pytest.raises((ValueError, TypeError)):
        idx.search(view, 5, chunk_size=1000)


@pytest.mark.parametrize("nrows", [2400, 100])
def test_noncontiguous_add_falls_through_both_sizes(nrows):
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(500, seed=0))  # non-empty → chunk-eligible
    base = len(idx)
    view = _rand(nrows * 2, seed=1)[::2]
    assert not view.flags.c_contiguous
    with pytest.raises((ValueError, TypeError)):
        idx.add(view, chunk_size=1000)
    assert len(idx) == base  # rejected whole, nothing committed


# ---------------------------------------------------------------------------
# F4: the query snapshot must also cover mask/allowlist, so a thread
# stomping the mask mid-batch cannot split it across versions.
# ---------------------------------------------------------------------------


def test_search_snapshots_mask_against_concurrent_mutation():
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(2000, seed=0))
    mask = np.zeros(2000, dtype=bool)
    mask[::3] = True
    original_mask = mask.copy()
    q = _rand(1500, seed=1)

    raw = TurboQuantIndex.search.__wrapped__
    ref = idx.search(q, 8, mask=original_mask, chunk_size=0)

    def stomping(self, queries, k, **kw):
        mask[:] = ~original_mask  # stomp the SOURCE mask mid-batch
        return raw(self, queries, k, **kw)

    chunked = _interruptible._make_search(stomping)
    scores, ids = chunked(idx, q, 8, mask=mask, chunk_size=250)
    # Snapshotted once → every slice used the original mask → == unchunked.
    np.testing.assert_array_equal(scores, ref[0])
    np.testing.assert_array_equal(ids, ref[1])


def test_search_noncontiguous_mask_falls_through():
    idx = IdMapIndex(DIM, 4)
    ids = np.arange(2000, dtype=np.uint64) + 100
    idx.add_with_ids(_rand(2000, seed=0), ids)
    allow = ids[::2]  # non-contiguous view
    assert not allow.flags.c_contiguous
    q = _rand(1500, seed=1)
    with pytest.raises((ValueError, TypeError)):
        idx.search(q, 8, allowlist=allow, chunk_size=250)
