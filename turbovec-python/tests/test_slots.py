"""Slots, holes, and the `len()` / `slot_capacity` split.

Calibration is fitted per block of rows, and a block keeps its slots
when one of them is removed rather than renumbering every later slot.
So a removal inside a filled block leaves a slot that is allocated,
addressable and empty — and `len(index)` stops being the slot count.

Everything slot-shaped has to be sized against `slot_capacity` from
that point on. These pin the two places that were not: the `mask=`
length check and the `swap_remove` bounds check.
"""
from __future__ import annotations

import numpy as np
import pytest

from turbovec import TurboQuantIndex


DIM = 64
# The default calibration block. A hole needs a removal from a block
# that has already filled, so the index has to be larger than one.
BLOCK = 8192


def unit_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


@pytest.fixture
def holed():
    """An index with one dead slot in its first, filled block."""
    v = unit_vectors(2 * BLOCK, seed=7)
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(v)
    assert len(idx) == idx.slot_capacity == 2 * BLOCK
    idx.swap_remove(0)
    assert len(idx) == 2 * BLOCK - 1
    assert idx.slot_capacity == 2 * BLOCK
    return idx, v


def test_capacity_equals_len_until_a_slot_dies():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    assert len(idx) == idx.slot_capacity == 0
    idx.add(unit_vectors(100, seed=1))
    assert len(idx) == idx.slot_capacity == 100
    # A removal from the open block frees its slot, so they stay equal.
    idx.swap_remove(0)
    assert len(idx) == idx.slot_capacity == 99


def test_slot_is_live_reports_the_hole(holed):
    idx, _ = holed
    assert idx.slot_is_live(0), "the filler landed in the vacated slot"
    assert not idx.slot_is_live(BLOCK - 1), "the vacated tail slot is still live"
    assert idx.slot_is_live(BLOCK), "a removal in one block killed a slot in another"
    assert not idx.slot_is_live(idx.slot_capacity), "past the end is not live"


def test_mask_is_sized_by_slot_capacity(holed):
    idx, v = holed
    mask = np.zeros(idx.slot_capacity, dtype=bool)
    mask[9000] = True
    _, ids = idx.search(v[9000:9001], 1, mask=mask)
    assert ids[0][0] == 9000


def test_a_len_sized_mask_is_a_value_error_not_a_panic(holed):
    # Both halves matter. The binding checked `len()` while the core
    # checked the slot count, so no length satisfied both — and the core
    # raised its objection as a panic, which reaches Python as
    # PanicException rather than anything catchable by contract.
    idx, _ = holed
    with pytest.raises(ValueError, match="slot capacity"):
        idx.search(unit_vectors(1, seed=3), 1, mask=np.ones(len(idx), dtype=bool))


def test_the_live_tail_slot_is_removable(holed):
    # Slots at or above `len()` are live here, and gating the bounds
    # check on `len()` made the tail permanently unremovable.
    idx, _ = holed
    tail = idx.slot_capacity - 1
    assert tail >= len(idx)
    assert idx.slot_is_live(tail)
    assert idx.swap_remove(tail) == tail


def test_removing_a_dead_slot_is_an_index_error(holed):
    idx, _ = holed
    dead = BLOCK - 1
    assert not idx.slot_is_live(dead)
    with pytest.raises(IndexError, match="not a live slot"):
        idx.swap_remove(dead)


def test_health_falls_when_a_slot_dies():
    v = unit_vectors(2 * BLOCK, seed=11)
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(v)
    before = idx.health()
    assert 0.99 < before <= 1.0
    for _ in range(100):
        idx.swap_remove(0)
    assert idx.health() < before
    # An index with nothing allocated wastes nothing.
    assert TurboQuantIndex(dim=DIM, bit_width=4).health() == 1.0
