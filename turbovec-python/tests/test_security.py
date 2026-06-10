"""Security regression tests.

These guard against the classes of bug found in the security audit: an
untrusted/corrupt index file must be *rejected* at load with a typed error
rather than loading and later panicking, returning silently-wrong results,
or driving an unbounded allocation. Each test crafts a malformed file by
hand against the on-disk format documented in ``turbovec/src/io.rs``.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex


def _craft_tv(
    path,
    *,
    bit_width: int,
    dim: int,
    n_vectors: int,
    n_scales: int | None = None,
    codes: bytes = b"",
    n_calib: int = 0,
) -> None:
    """Write a v3 ``.tv`` file with fully attacker-controlled header fields."""
    if n_scales is None:
        n_scales = n_vectors
    with open(path, "wb") as f:
        f.write(b"TVPI")               # magic
        f.write(bytes([3]))            # version 3
        f.write(bytes([bit_width & 0xFF]))
        f.write(struct.pack("<I", dim))
        f.write(struct.pack("<I", n_vectors))
        f.write(codes)
        f.write(struct.pack("<f", 1.0) * n_scales)
        f.write(struct.pack("<I", n_calib))


@pytest.mark.parametrize("bit_width", [0, 1, 5, 6, 8, 255])
def test_load_rejects_out_of_range_bit_width(tmp_path, bit_width):
    # bit_width 0/>8 divide-by-zero'd in repack; 5..8 silently passed the
    # length check and returned wrong scores. Only 2/3/4 are valid.
    p = tmp_path / "bad_bitwidth.tv"
    _craft_tv(p, bit_width=bit_width, dim=8, n_vectors=1, codes=b"\x00" * 8)
    with pytest.raises((ValueError, OSError)):
        TurboQuantIndex.load(str(p))


@pytest.mark.parametrize("dim", [12, 7, 100])
def test_load_rejects_non_multiple_of_8_dim(tmp_path, dim):
    p = tmp_path / "bad_dim.tv"
    _craft_tv(p, bit_width=4, dim=dim, n_vectors=1, codes=b"\x00" * 8)
    with pytest.raises((ValueError, OSError)):
        TurboQuantIndex.load(str(p))


def test_load_rejects_dim_zero_with_vectors(tmp_path):
    # dim==0 is the lazy-index sentinel and is only valid with n_vectors==0.
    p = tmp_path / "bad_lazy.tv"
    _craft_tv(p, bit_width=4, dim=0, n_vectors=5)
    with pytest.raises((ValueError, OSError)):
        TurboQuantIndex.load(str(p))


def test_load_rejects_huge_n_vectors_without_allocating(tmp_path):
    # A tiny file declaring billions of vectors must fail on the truncated
    # data, NOT pre-allocate gigabytes. This completes quickly if the loader
    # reads incrementally; it would OOM/hang if it pre-sized from the header.
    p = tmp_path / "huge.tv"
    _craft_tv(p, bit_width=2, dim=8, n_vectors=0xFFFFFFFF, n_scales=0)
    with pytest.raises((ValueError, OSError)):
        TurboQuantIndex.load(str(p))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e17])
def test_turboquant_search_rejects_non_finite_query(bad):
    # A NaN/Inf/overflow query coord previously panicked inside the core,
    # surfacing as an uncatchable PanicException. It must raise ValueError.
    idx = TurboQuantIndex(dim=8, bit_width=4)
    idx.add(np.ones((4, 8), dtype=np.float32))
    q = np.ones((1, 8), dtype=np.float32)
    q[0, 0] = bad
    with pytest.raises(ValueError):
        idx.search(q, 2)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e17])
def test_idmap_search_rejects_non_finite_query(bad):
    idx = IdMapIndex(dim=8, bit_width=4)
    idx.add_with_ids(np.ones((4, 8), dtype=np.float32), np.arange(4, dtype=np.uint64))
    q = np.ones((1, 8), dtype=np.float32)
    q[0, 0] = bad
    with pytest.raises(ValueError):
        idx.search(q, 2)


def test_load_rejects_oversized_dim(tmp_path):
    # A tiny file declaring a huge dim passes the dim%8 check but would drive
    # a multi-GB dim x dim rotation-matrix allocation on first search. The
    # loader must reject dim > MAX_DIM (65536).
    p = tmp_path / "bigdim.tv"
    _craft_tv(p, bit_width=2, dim=70000, n_vectors=0, n_scales=0)
    with pytest.raises((ValueError, OSError)):
        TurboQuantIndex.load(str(p))


def test_construct_rejects_oversized_dim():
    with pytest.raises(ValueError):
        TurboQuantIndex(dim=70000, bit_width=4)


def test_lazy_add_rejects_zero_column_array():
    # A 0-column array to a lazy index slipped past the dim%8 check (0%8==0),
    # divided by zero in the core, and wedged the index at dim=0. Must raise
    # ValueError and leave the index uncommitted.
    idx = TurboQuantIndex()  # lazy: no dim
    with pytest.raises(ValueError):
        idx.add(np.ones((4, 0), dtype=np.float32))
    assert idx.dim is None
    # Not wedged: a normal add still works afterwards.
    idx.add(np.ones((3, 8), dtype=np.float32))
    assert len(idx) == 3 and idx.dim == 8


def test_valid_roundtrip_still_loads(tmp_path):
    # The hardening must not break legitimate files.
    p = tmp_path / "good.tv"
    idx = TurboQuantIndex(dim=8, bit_width=4)
    idx.add(np.ones((3, 8), dtype=np.float32))
    idx.write(str(p))
    loaded = TurboQuantIndex.load(str(p))
    assert len(loaded) == 3
