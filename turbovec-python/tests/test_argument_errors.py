"""Argument-validation errors at the PyO3 boundary (#127, #128).

Wrong dtype/ndim array inputs must raise a TypeError that names the
argument and states expected vs got (not pyo3's opaque "'ndarray' object
cannot be cast as 'ndarray'"). Negative or out-of-range integers must
follow per-method semantics: IndexError for swap_remove, False for
membership tests, and a ValueError naming the argument for count/size
parameters (k, dim, bit_width). Wrong-dtype inputs are rejected, never
silently converted.
"""
from __future__ import annotations

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex

DIM = 16


def vectors(n: int = 4, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


@pytest.fixture
def idx() -> TurboQuantIndex:
    index = TurboQuantIndex(dim=DIM)
    index.add(vectors())
    return index


@pytest.fixture
def idmap() -> IdMapIndex:
    index = IdMapIndex(dim=DIM)
    index.add_with_ids(vectors(), np.array([1, 2, 3, 4], dtype=np.uint64))
    return index


# --- #127: wrong dtype / ndim name the argument and expected vs got ------


def test_add_float64_names_argument_and_dtypes(idx):
    with pytest.raises(
        TypeError, match=r"vectors must be a 2-D float32 array, got 2-D float64"
    ):
        idx.add(vectors().astype(np.float64))


def test_search_1d_query_names_ndim(idx):
    with pytest.raises(
        TypeError, match=r"queries must be a 2-D float32 array, got 1-D float32"
    ):
        idx.search(vectors()[0], 2)


def test_search_3d_query_names_ndim(idx):
    with pytest.raises(
        TypeError, match=r"queries must be a 2-D float32 array, got 3-D float32"
    ):
        idx.search(vectors().reshape(2, 2, DIM), 2)


def test_add_big_endian_rejected_with_dtype(idx):
    with pytest.raises(TypeError, match=r"vectors must be a 2-D float32 array"):
        idx.add(vectors().astype(">f4"))


def test_add_list_names_got_type(idx):
    with pytest.raises(
        TypeError, match=r"vectors must be a 2-D float32 array, got list"
    ):
        idx.add(vectors().tolist())


def test_add_with_ids_int64_ids_names_argument(idmap):
    with pytest.raises(
        TypeError, match=r"ids must be a 1-D uint64 array, got 1-D int64"
    ):
        idmap.add_with_ids(vectors(seed=1), np.array([11, 12, 13, 14]))


def test_add_with_ids_float64_vectors_names_argument():
    index = IdMapIndex(dim=DIM)
    with pytest.raises(
        TypeError, match=r"vectors must be a 2-D float32 array, got 2-D float64"
    ):
        index.add_with_ids(
            vectors().astype(np.float64),
            np.array([1, 2, 3, 4], dtype=np.uint64),
        )


def test_allowlist_int64_names_argument(idmap):
    with pytest.raises(
        TypeError, match=r"allowlist must be a 1-D uint64 array, got 1-D int64"
    ):
        idmap.search(vectors(1), 2, allowlist=np.array([1, 3]))


def test_mask_int_names_argument(idx):
    with pytest.raises(
        TypeError, match=r"mask must be a 1-D bool array, got 1-D int64"
    ):
        idx.search(vectors(1), 2, mask=np.ones(len(idx), dtype=np.int64))


def test_noncontiguous_float32_still_hits_contiguity_error(idx):
    fortran = np.asfortranarray(vectors())
    with pytest.raises(ValueError, match="contiguous"):
        idx.add(fortran)


# --- #128: negative / out-of-range integers, per-method semantics --------


def test_swap_remove_negative_raises_index_error(idx):
    with pytest.raises(IndexError, match=r"index -1 out of range"):
        idx.swap_remove(-1)


def test_swap_remove_huge_raises_index_error(idx):
    with pytest.raises(IndexError, match=r"out of range"):
        idx.swap_remove(2**64)


def test_contains_negative_is_false(idmap):
    assert (-1) not in idmap
    assert idmap.contains(-1) is False
    assert idmap.contains(np.int64(-1)) is False


def test_contains_above_u64_is_false(idmap):
    assert (2**64) not in idmap
    assert (2**200) not in idmap
    assert idmap.contains(2**64) is False


def test_contains_present_id_still_true(idmap):
    assert 1 in idmap
    assert idmap.contains(np.uint64(1)) is True


def test_remove_out_of_range_returns_false(idmap):
    assert idmap.remove(-1) is False
    assert idmap.remove(2**64) is False
    assert len(idmap) == 4


def test_negative_k_names_argument(idx):
    with pytest.raises(ValueError, match=r"k must be a non-negative integer, got -1"):
        idx.search(vectors(1), -1)


def test_negative_k_names_argument_idmap(idmap):
    with pytest.raises(ValueError, match=r"k must be a non-negative integer, got -1"):
        idmap.search(vectors(1), -1)


@pytest.mark.parametrize("cls", [TurboQuantIndex, IdMapIndex])
def test_negative_dim_names_argument(cls):
    with pytest.raises(
        ValueError, match=r"dim must be a non-negative integer, got -8"
    ):
        cls(dim=-8)


@pytest.mark.parametrize("cls", [TurboQuantIndex, IdMapIndex])
def test_negative_bit_width_names_argument(cls):
    with pytest.raises(
        ValueError, match=r"bit_width must be a non-negative integer, got -4"
    ):
        cls(dim=DIM, bit_width=-4)
