"""Regression tests for GIL release in the pyo3 bindings (issue #121).

Long ``search`` / ``add`` / ``prepare`` / ``write`` / ``load`` calls
detach from the Python runtime (release the GIL) around the
compute-bound core call, so other Python threads keep running and
concurrent searches actually overlap. The bindings snapshot
numpy-borrowed buffers into owned Vecs *before* releasing the GIL, so a
Python thread mutating an input array mid-call cannot affect (or
corrupt) the running operation.

Thresholds are deliberately generous for slow CI machines: the
background-progress tests count iterations strictly inside the middle
of the call window (excluding the GIL hand-off slices at the call
boundaries, which occur even without the fix) and assert a small
absolute floor, not a throughput ratio.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex

N = 200_000
DIM = 256


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.standard_normal((N, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


@pytest.fixture(scope="module")
def index(vectors: np.ndarray) -> TurboQuantIndex:
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(vectors)
    idx.prepare()
    return idx


def _progress_during(fn):
    """Run ``fn()`` on this thread while a counter thread runs.

    Returns the number of counter iterations whose timestamps fall
    strictly inside the middle of the call window — the call-boundary
    GIL hand-off slices (a few ms each, present even when the callee
    never releases the GIL) are excluded, so without GIL release this
    is 0.
    """
    stop = threading.Event()
    samples: list[tuple[float, int]] = []

    def counter() -> None:
        n = 0
        while not stop.is_set():
            n += 1
            if n % 4096 == 0:
                samples.append((time.perf_counter(), n))

    t = threading.Thread(target=counter)
    t.start()
    time.sleep(0.05)  # let the counter thread reach its loop
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    stop.set()
    t.join()

    margin = (t1 - t0) * 0.25
    lo, hi = t0 + margin, t1 - margin
    inside = [n for (ts, n) in samples if lo < ts < hi]
    iters = (inside[-1] - inside[0]) if len(inside) >= 2 else 0
    return iters, t1 - t0


def test_background_thread_progresses_during_search(index, vectors):
    queries = vectors[:4096]
    iters, _ = _progress_during(lambda: index.search(queries, 100))
    assert iters > 1000, (
        f"background thread made only {iters} iterations mid-search: "
        "the binding did not release the GIL"
    )


def test_background_thread_progresses_during_add(vectors):
    idx = IdMapIndex(dim=DIM, bit_width=4)
    ids = np.arange(N, dtype=np.uint64)
    iters, _ = _progress_during(lambda: idx.add_with_ids(vectors, ids))
    assert iters > 1000, (
        f"background thread made only {iters} iterations mid-add: "
        "the binding did not release the GIL"
    )


def test_len_blocked_on_add_does_not_hold_gil(vectors):
    """len() landing during an in-flight add blocks until the write
    lock frees (writes serialize), but must wait with the GIL released
    so other Python threads keep running.

    The probe is retried up to 3 times: the add's internal rayon
    fan-out saturates every core, so the scheduler can starve the
    counter thread for the whole mid-window of a single attempt (~4%
    of runs on small machines), zeroing the measurement. A real
    regression (the blocked call holding the GIL) shows zero progress
    deterministically, so it fails every attempt.
    """
    ids = np.arange(N, dtype=np.uint64)
    blocked_attempts: list[tuple[int, float]] = []

    for _ in range(3):
        idx = IdMapIndex(dim=DIM, bit_width=4)
        errors: list[BaseException] = []
        started = threading.Event()

        def adder() -> None:
            try:
                started.set()
                idx.add_with_ids(vectors, ids)
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        t = threading.Thread(target=adder)
        t.start()
        started.wait()
        time.sleep(0.02)  # let the add reach the core kernel
        iters, dur = _progress_during(lambda: len(idx))
        t.join()

        assert not errors, f"add raised: {errors}"
        assert len(idx) == N
        if dur < 0.05:
            # add finished before len() could block; inconclusive
            continue
        if iters > 1000:
            return
        blocked_attempts.append((iters, dur))

    if not blocked_attempts:
        pytest.skip("add finished before len() could block; nothing to measure")
    details = ", ".join(
        f"{iters} iters in {dur * 1000:.0f} ms" for iters, dur in blocked_attempts
    )
    raise AssertionError(
        f"background thread made no mid-window progress while len() was "
        f"blocked, on every attempt ({details}): the blocked call is "
        "holding the GIL"
    )


def test_concurrent_searches_match_serial(index, vectors):
    queries = vectors[:512]
    expected_scores, expected_indices = index.search(queries, 10)

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    errors: list[BaseException] = []

    def worker(slot: int) -> None:
        try:
            for _ in range(5):
                results[slot] = index.search(queries, 10)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent search raised: {errors}"
    for slot in range(2):
        scores, indices = results[slot]
        np.testing.assert_array_equal(scores, expected_scores)
        np.testing.assert_array_equal(indices, expected_indices)


def test_search_while_add_both_succeed(vectors):
    """A search overlapping an add on the same index must not raise:
    the write lock serializes the add against the search (as the GIL
    did before GIL release) instead of surfacing pyo3's 'Already
    borrowed' RuntimeError."""
    idx = IdMapIndex(dim=DIM, bit_width=4)
    half = N // 2
    idx.add_with_ids(vectors[:half], np.arange(half, dtype=np.uint64))
    idx.prepare()
    queries = vectors[:512]
    errors: list[BaseException] = []
    started = threading.Event()

    def adder() -> None:
        try:
            started.set()
            idx.add_with_ids(
                vectors[half:], np.arange(half, N, dtype=np.uint64)
            )
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    t = threading.Thread(target=adder)
    t.start()
    started.wait()
    scores, ids = idx.search(queries, 10)  # overlaps the ~0.2 s add
    t.join()

    assert not errors, f"concurrent add raised: {errors}"
    assert scores.shape == (512, 10)
    assert np.all(np.isfinite(scores))
    assert len(idx) == N


def test_two_concurrent_adds_both_succeed(vectors):
    """Two adds racing on the same index must both land (writes
    serialize), not fail with a borrow error or lose data."""
    idx = IdMapIndex(dim=DIM, bit_width=4)
    half = N // 2
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def adder(lo: int, hi: int) -> None:
        try:
            barrier.wait()
            idx.add_with_ids(
                vectors[lo:hi], np.arange(lo, hi, dtype=np.uint64)
            )
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [
        threading.Thread(target=adder, args=(0, half)),
        threading.Thread(target=adder, args=(half, N)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent adds raised: {errors}"
    assert len(idx) == N
    assert 0 in idx and N - 1 in idx


def test_mutating_query_array_mid_search_is_safe(index, vectors):
    """A writable query array stomped mid-search must not crash or
    produce torn results: the binding snapshots the buffer under the
    GIL, so the result always corresponds to one coherent version of
    the array (whichever version the snapshot captured)."""
    queries = vectors[:4096].copy()  # writable
    mutant = np.ascontiguousarray(vectors[4096:8192])

    expected_original = index.search(queries.copy(), 10)
    expected_mutant = index.search(mutant.copy(), 10)

    started = threading.Event()

    def mutator() -> None:
        started.wait()
        time.sleep(0.01)  # land inside the (much longer) search
        np.copyto(queries, mutant)

    t = threading.Thread(target=mutator)
    t.start()
    started.set()
    scores, indices = index.search(queries, 10)
    t.join()

    matches_original = np.array_equal(
        scores, expected_original[0]
    ) and np.array_equal(indices, expected_original[1])
    matches_mutant = np.array_equal(scores, expected_mutant[0]) and np.array_equal(
        indices, expected_mutant[1]
    )
    assert matches_original or matches_mutant, (
        "search results match neither the original nor the mutated "
        "query array: the searched buffer was torn mid-call"
    )
