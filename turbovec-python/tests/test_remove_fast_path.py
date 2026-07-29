"""Mutations on a *loaded* index must take the binding's fast path (#392).

No mutation on a v6-loaded index materializes the packed bit-plane rows:
``add`` lazy-appends to the blocked cache, ``swap_remove`` patches it with
O(dim) lane ops, and neither sets the ``packed_codes`` OnceLock. So
``packed_ready()`` stays ``False`` for such an index's entire lifetime,
and a binding fast path gated on it did not select "first mutation after
a load" — it selected *every* mutation on *every* loaded index, routing
each through a ``py.detach`` plus a rayon pool handoff costing far more
than the operation itself.

The two remove tests assert a loaded-vs-fresh ratio in the same process
over the same index contents, so machine speed cancels. Their fresh
baselines are 0.06-0.38 µs against a ~20 µs handoff, so the defect
separates from the honest ratio by 10-100x and a ratio gate is safe.

The add test does **not** use that shape, and deliberately. Its fresh
baseline is ~0.9 µs, so the same fixed handoff shows up as ~25x rather
than ~50-280x, and under memory pressure the honest and defect
distributions come within a few percent of each other — a 12.0x gate was
measured passing with the defect live at 12.64x. Widening such a gate
makes it unfalsifiable; the fix is to measure something else. So the add
test compares a one-row add against a **two-row** add on the same loaded
index: a two-row add is pooled under every version of this code
(``n_rows > 1``), so both arms pay identical lane-append work and differ
only by the handoff the one-row bypass is supposed to skip. Contention
inflates both arms together — measured separation widens from 6.5x idle
to 12x under load, in the safe direction.

Honest ratios on arm64 after the fix: 1.4x (``remove``), 2.2x
(``swap_remove``), 0.12 one-row/two-row (``add``). The lane writes these
paths do go through a heavier branch on x86_64
(``pack::write_x86_code_byte`` vs a plain byte store), so the two remove
ratios may sit higher there. If one proves tight, check first that the
defect range for that op is still an order of magnitude away before
touching the threshold — for a gate whose honest and defect
distributions overlap, widening hides the defect instead of catching it.
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

# Loaded-vs-fresh gates for the removes, each roughly the geometric mean
# of the honest ratio and the smallest defect ratio measured for that op
# (the single-threaded case, where the pool handoff is cheapest).
MAX_RATIO_REMOVE = 8.0  # honest 1.4x, defect >= 20.7x
MAX_RATIO_SWAP_REMOVE = 12.0  # honest 2.2x, defect >= 48.0x

# One-row / two-row gate for the add: honest 0.08-0.13, defect 0.81-1.29
# over 12 idle and 6 contended samples. Sits ~3x clear of both.
MAX_ONE_OVER_TWO_ROW_ADD = 0.4


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


def test_single_row_add_on_a_loaded_index_takes_the_inline_bypass(
    vectors: np.ndarray,
) -> None:
    """The sibling of the two above, on the hotter verb.

    The single-row add bypass was gated on the same permanently-false
    ``packed_ready()`` probe, so every post-load add paid the pool
    handoff for the index's whole lifetime.

    Measured against a two-row add on the same loaded index rather than
    against a fresh one: two rows are pooled under every version of this
    code, so the two arms differ only by the handoff the bypass skips,
    and machine load moves both together. See the module docstring for
    why the loaded-vs-fresh ratio does not work for this op.
    """

    def loaded(n_rows: int):
        def build() -> TurboQuantIndex:
            index = TurboQuantIndex(dim=DIM, bit_width=4)
            index.add(vectors)
            return TurboQuantIndex.from_bytes(index.to_bytes())

        batch = vectors[:n_rows].copy()
        return _per_op_us(build, lambda ix, _i: ix.add(batch))

    one, two = loaded(1), loaded(2)
    ratio = one / two
    assert ratio < MAX_ONE_OVER_TWO_ROW_ADD, (
        f"a one-row add() on a loaded index costs {one:.2f} us against "
        f"{two:.2f} us for a two-row add ({ratio:.2f} of it) — the one-row "
        f"inline bypass is not engaging, so it is paying the pool handoff"
    )
