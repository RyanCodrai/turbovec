"""Tests for the `mask=` and `allowlist=` filtering kwargs."""
from __future__ import annotations

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex


DIM = 128


def unit_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


# ------------------- TurboQuantIndex.search(mask=...) -------------------

def test_mask_none_matches_unmasked():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(100, seed=1))
    queries = unit_vectors(3, seed=2)

    s1, i1 = idx.search(queries, 10)
    s2, i2 = idx.search(queries, 10, mask=None)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(i1, i2)


def test_mask_all_true_matches_unmasked():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(80, seed=3))
    queries = unit_vectors(2, seed=4)

    s1, i1 = idx.search(queries, 5)
    mask = np.ones(len(idx), dtype=bool)
    s2, i2 = idx.search(queries, 5, mask=mask)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(i1, i2)


def test_mask_restricts_returned_indices():
    n = 200
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=5))
    queries = unit_vectors(4, seed=6)

    mask = np.zeros(n, dtype=bool)
    allowed = [3, 7, 19, 42, 88, 121, 150, 175, 198]
    mask[allowed] = True

    scores, indices = idx.search(queries, 5, mask=mask)
    assert scores.shape == (4, 5)
    assert indices.shape == (4, 5)
    for row in indices:
        for slot in row:
            assert slot in allowed, f"kernel returned disallowed slot {slot}"
    # Scores descending per row.
    for row in scores:
        assert np.all(np.diff(row) <= 1e-6)


def test_mask_shrinks_effective_k():
    n = 100
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=7))
    queries = unit_vectors(2, seed=8)

    mask = np.zeros(n, dtype=bool)
    mask[[5, 10, 15]] = True
    scores, indices = idx.search(queries, 10, mask=mask)
    assert scores.shape == (2, 3), "effective_k should be popcount(mask) = 3"
    assert indices.shape == (2, 3)


def test_mask_all_false_returns_empty_columns():
    n = 50
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=9))
    queries = unit_vectors(2, seed=10)

    mask = np.zeros(n, dtype=bool)
    scores, indices = idx.search(queries, 5, mask=mask)
    assert scores.shape == (2, 0)
    assert indices.shape == (2, 0)


def test_mask_wrong_length_raises():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(50, seed=11))
    queries = unit_vectors(1, seed=12)

    with pytest.raises(ValueError, match="mask length"):
        idx.search(queries, 5, mask=np.ones(10, dtype=bool))


def test_mask_must_be_bool_dtype():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(50, seed=13))
    queries = unit_vectors(1, seed=14)

    with pytest.raises((TypeError, ValueError)):
        idx.search(queries, 5, mask=np.ones(50, dtype=np.uint8))


def test_mask_matches_post_hoc_filter():
    n = 256
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=15))
    queries = unit_vectors(5, seed=16)

    mask = np.zeros(n, dtype=bool)
    mask[::3] = True
    k = 7

    unfiltered_scores, unfiltered_idx = idx.search(queries, n)
    expected_indices = []
    expected_scores = []
    for qi in range(queries.shape[0]):
        row_idx = unfiltered_idx[qi]
        row_s = unfiltered_scores[qi]
        keep = mask[row_idx]
        expected_indices.append(row_idx[keep][:k])
        expected_scores.append(row_s[keep][:k])

    masked_scores, masked_idx = idx.search(queries, k, mask=mask)
    assert masked_idx.shape == (queries.shape[0], k)
    for qi in range(queries.shape[0]):
        np.testing.assert_array_equal(masked_idx[qi], expected_indices[qi])
        # Score parity is exact when both calls go through the same code
        # path (single batched call on both sides).
        np.testing.assert_allclose(masked_scores[qi], expected_scores[qi], rtol=1e-4, atol=1e-5)


# ------------------- IdMapIndex.search(allowlist=...) -------------------

def test_allowlist_none_matches_unfiltered():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    ids = np.arange(7000, 7100, dtype=np.uint64)
    idx.add_with_ids(unit_vectors(100, seed=20), ids)
    queries = unit_vectors(2, seed=21)

    s1, i1 = idx.search(queries, 10)
    s2, i2 = idx.search(queries, 10, allowlist=None)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(i1, i2)


def test_allowlist_restricts_returned_ids():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    ids = np.arange(1000, 1100, dtype=np.uint64)
    idx.add_with_ids(unit_vectors(100, seed=22), ids)
    queries = unit_vectors(3, seed=23)

    allowed = np.array([1003, 1010, 1042, 1077, 1099], dtype=np.uint64)
    scores, returned = idx.search(queries, 10, allowlist=allowed)
    assert scores.shape == (3, len(allowed))
    assert returned.shape == (3, len(allowed))
    for row in returned:
        for rid in row:
            assert rid in allowed


def test_allowlist_empty_raises_value_error():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    idx.add_with_ids(
        unit_vectors(5, seed=24),
        np.array([1, 2, 3, 4, 5], dtype=np.uint64),
    )
    queries = unit_vectors(1, seed=25)
    with pytest.raises(ValueError, match="allowlist is empty"):
        idx.search(queries, 3, allowlist=np.array([], dtype=np.uint64))


def test_allowlist_unknown_id_raises_key_error():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    idx.add_with_ids(
        unit_vectors(5, seed=26),
        np.array([1, 2, 3, 4, 5], dtype=np.uint64),
    )
    queries = unit_vectors(1, seed=27)
    with pytest.raises(KeyError, match="not present"):
        idx.search(queries, 3, allowlist=np.array([2, 999], dtype=np.uint64))


def test_allowlist_kwarg_is_keyword_only():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    ids = np.arange(100, 110, dtype=np.uint64)
    idx.add_with_ids(unit_vectors(10, seed=28), ids)
    queries = unit_vectors(1, seed=29)
    # Positional third arg should be rejected by PyO3 signature.
    with pytest.raises(TypeError):
        idx.search(queries, 3, ids)


def test_mask_kwarg_is_keyword_only():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(10, seed=30))
    queries = unit_vectors(1, seed=31)
    mask = np.ones(10, dtype=bool)
    with pytest.raises(TypeError):
        idx.search(queries, 3, mask)
