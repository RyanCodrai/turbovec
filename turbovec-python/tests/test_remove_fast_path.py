"""Removes on a *loaded* index must take the binding's fast path (#392).

``TurboQuantIndex.swap_remove`` maintains whichever code layout is
materialized in O(dim) lane ops and never triggers the lazy O(n·dim)
packed rebuild, so ``packed_ready()`` stays ``False`` for a loaded
index's entire lifetime. Gating the fast path on it therefore did not
select "first mutation after a load" — it selected *every* remove on
every loaded index, and routed each one through a ``py.detach`` plus a
rayon pool handoff costing far more than the removal itself.

Asserted as a loaded-vs-fresh ratio on the same process and the same
index contents, so machine speed cancels; the gate is deliberately far
above the honest ratio (a loaded index does pay real O(dim) lane work
where a fresh one memcpys a packed row) and far below the defect, which
is a fixed per-op overhead of ~3 µs single-threaded and ~18 µs with the
pool contended, against sub-microsecond removes.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex

N = 50_000
DIM = 64
OPS = 1_000
REPS = 3

# Honest ratios measured after the fix are 1.4x (IdMapIndex.remove) and
# 2.2x (swap_remove); the defect is 20-280x depending on pool contention.
MAX_RATIO = 8.0


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N, DIM)).astype(np.float32)


def _per_op_us(build, op) -> float:
    """Microseconds per ``op`` call, best of ``REPS`` fresh indexes."""
    best = float("inf")
    for _ in range(REPS):
        index = build()
        op(index, 0)  # warm up: materializes any lazy structure
        t0 = time.perf_counter()
        for i in range(1, OPS + 1):
            op(index, i)
        best = min(best, (time.perf_counter() - t0) / OPS * 1e6)
    return best


def _ratio(fresh_build, loaded_build, op) -> tuple[float, float, float]:
    fresh = _per_op_us(fresh_build, op)
    loaded = _per_op_us(loaded_build, op)
    return fresh, loaded, loaded / fresh


def test_id_map_remove_on_a_loaded_index_is_not_pool_bound(vectors: np.ndarray) -> None:
    ids = np.arange(1, N + 1, dtype=np.uint64)

    def fresh() -> IdMapIndex:
        index = IdMapIndex(dim=DIM, bit_width=4)
        index.add_with_ids(vectors, ids)
        return index

    def loaded() -> IdMapIndex:
        return IdMapIndex.from_bytes(fresh().to_bytes())

    f, l, ratio = _ratio(fresh, loaded, lambda ix, i: ix.remove(int(ids[i])))
    assert ratio < MAX_RATIO, (
        f"remove() on a loaded index costs {l:.2f} us vs {f:.2f} us on a fresh "
        f"one ({ratio:.1f}x) — it is going through the detach + pool slow path"
    )


def test_swap_remove_on_a_loaded_index_is_not_pool_bound(vectors: np.ndarray) -> None:
    def fresh() -> TurboQuantIndex:
        index = TurboQuantIndex(dim=DIM, bit_width=4)
        index.add(vectors)
        return index

    def loaded() -> TurboQuantIndex:
        return TurboQuantIndex.from_bytes(fresh().to_bytes())

    f, l, ratio = _ratio(fresh, loaded, lambda ix, _i: ix.swap_remove(0))
    assert ratio < MAX_RATIO, (
        f"swap_remove() on a loaded index costs {l:.2f} us vs {f:.2f} us on a "
        f"fresh one ({ratio:.1f}x) — it is going through the detach + pool "
        f"slow path"
    )
