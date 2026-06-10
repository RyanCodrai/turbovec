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

from turbovec import TurboQuantIndex


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


def test_valid_roundtrip_still_loads(tmp_path):
    # The hardening must not break legitimate files.
    p = tmp_path / "good.tv"
    idx = TurboQuantIndex(dim=8, bit_width=4)
    idx.add(np.ones((3, 8), dtype=np.float32))
    idx.write(str(p))
    loaded = TurboQuantIndex.load(str(p))
    assert len(loaded) == 3
