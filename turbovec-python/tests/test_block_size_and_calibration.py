"""Constructor options for per-block calibration.

Both features were Rust-only until now: the Python constructor took
`(dim, bit_width)` and nothing else, so a caller could neither choose a
block size nor turn TQ+ off. The failure mode worth pinning is not that
they are absent but that they might be *silently ignored* — an argument
accepted and discarded is what the identity cliff was.
"""
from __future__ import annotations

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex

DIM = 64


def unit(n, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def test_calibration_is_on_by_default():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit(2000, seed=1))
    assert idx.calibration_state == "fitted"


def test_calibrate_false_gives_an_uncalibrated_index():
    idx = TurboQuantIndex(dim=DIM, bit_width=4, calibrate=False)
    assert idx.calibration_state == "identity"
    idx.add(unit(2000, seed=1))
    assert idx.calibration_state == "identity", "adding rows re-enabled calibration"


def test_calibrate_false_works_on_the_lazy_path():
    idx = TurboQuantIndex(bit_width=4, calibrate=False)
    idx.add(unit(2000, seed=2))
    assert idx.calibration_state == "identity"


def test_block_size_actually_takes_effect():
    # Not merely accepted: a different block size must produce a
    # different index. Same rows, two block sizes, different bytes.
    rows = unit(4096, seed=3)
    a = TurboQuantIndex(dim=DIM, bit_width=4, block_size=1024)
    b = TurboQuantIndex(dim=DIM, bit_width=4, block_size=2048)
    a.add(rows)
    b.add(rows)
    assert a.to_bytes() != b.to_bytes(), "block_size was accepted and ignored"

    # And the default is reachable by not passing it.
    d = TurboQuantIndex(dim=DIM, bit_width=4)
    d.add(rows)
    e = TurboQuantIndex(dim=DIM, bit_width=4, block_size=8192)
    e.add(rows)
    assert d.to_bytes() == e.to_bytes(), "the default is not DEFAULT_BLOCK_SIZE"


@pytest.mark.parametrize("bad", [0, 1, 63, 65, 100])
def test_an_invalid_block_size_raises(bad):
    with pytest.raises(ValueError, match="multiple of"):
        TurboQuantIndex(dim=DIM, bit_width=4, block_size=bad)


def test_id_map_takes_a_block_size():
    rows = unit(4096, seed=5)
    ids = np.arange(len(rows), dtype=np.uint64)
    a = IdMapIndex(dim=DIM, bit_width=4, block_size=1024)
    b = IdMapIndex(dim=DIM, bit_width=4, block_size=2048)
    a.add_with_ids(rows, ids)
    b.add_with_ids(rows, ids)
    assert a.to_bytes() != b.to_bytes(), "block_size was accepted and ignored"
    with pytest.raises(ValueError, match="multiple of"):
        IdMapIndex(dim=DIM, bit_width=4, block_size=100)


def test_block_size_on_a_lazy_index_is_refused_not_ignored():
    # A lazy index has no dim to build against, so it cannot take a
    # block size at construction. Refusing is the point: accepting and
    # discarding it is the failure this test exists for.
    with pytest.raises(ValueError, match="requires dim"):
        TurboQuantIndex(bit_width=4, block_size=1024)
    # An invalid one is still rejected as invalid.
    with pytest.raises(ValueError, match="multiple of"):
        TurboQuantIndex(bit_width=4, block_size=100)
