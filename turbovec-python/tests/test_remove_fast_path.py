"""Mutations on a *loaded* index must take the binding's fast path (#392).

No mutation on a v6-loaded index materializes the packed bit-plane rows:
``add`` lazy-appends to the blocked cache, ``swap_remove`` patches it with
O(dim) lane ops, and neither sets the ``packed_codes`` OnceLock. So
``packed_ready()`` stays ``False`` for such an index's entire lifetime,
and a binding fast path gated on it did not select "first mutation after
a load" — it selected *every* mutation on *every* loaded index, routing
each through a ``py.detach`` plus a rayon pool handoff costing far more
than the operation itself.

Asserted as a loaded-vs-fresh ratio in the same process over the same
index contents, so machine speed cancels. The gates sit well above the
honest ratio — a loaded index does pay real lane work where a fresh one
appends to or memcpys a packed row — and well below the defect, which is
a fixed per-op overhead of ~3-45 µs against sub-microsecond to
low-microsecond operations.

Measured honest ratios on arm64 after the fix: 1.4x (``remove``), 2.2x
(``swap_remove``), 3.5x (``add``). The lane writes those paths do go
through a heavier branch on x86_64 (``pack::write_x86_code_byte`` vs a
plain byte store), so the honest ratios are expected to be somewhat
higher there; if a gate proves tight on x86 CI it should be widened, not
deleted — the defect it discriminates against is 20-280x.
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

# Per-op gates, each roughly the geometric mean of the honest ratio and
# the smallest defect ratio measured for that op (the single-threaded
# case, where the pool handoff is cheapest).
MAX_RATIO_REMOVE = 8.0  # honest 1.4x, defect >= 20.7x
MAX_RATIO_SWAP_REMOVE = 12.0  # honest 2.2x, defect >= 48.0x
MAX_RATIO_ADD = 12.0  # honest 3.5x, defect >= 28.0x


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
    assert ratio < MAX_RATIO_REMOVE, (
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
    assert ratio < MAX_RATIO_SWAP_REMOVE, (
        f"swap_remove() on a loaded index costs {l:.2f} us vs {f:.2f} us on a "
        f"fresh one ({ratio:.1f}x) — it is going through the detach + pool "
        f"slow path"
    )


def test_single_row_add_on_a_loaded_index_is_not_pool_bound(
    vectors: np.ndarray,
) -> None:
    """The sibling of the two above, on the hotter verb.

    The single-row add bypass was gated on the same permanently-false
    ``packed_ready()`` probe, so every post-load add paid the pool
    handoff for the index's whole lifetime.
    """
    row = vectors[:1].copy()

    def fresh() -> TurboQuantIndex:
        index = TurboQuantIndex(dim=DIM, bit_width=4)
        index.add(vectors)
        return index

    def loaded() -> TurboQuantIndex:
        return TurboQuantIndex.from_bytes(fresh().to_bytes())

    f, l, ratio = _ratio(fresh, loaded, lambda ix, _i: ix.add(row))
    assert ratio < MAX_RATIO_ADD, (
        f"a single-row add() on a loaded index costs {l:.2f} us vs {f:.2f} us "
        f"on a fresh one ({ratio:.1f}x) — it is going through the pool handoff"
    )
