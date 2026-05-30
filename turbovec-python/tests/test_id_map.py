"""Tests for IdMapIndex — the stable-id wrapper around TurboQuantIndex."""
from __future__ import annotations

import numpy as np
import pytest

from turbovec import IdMapIndex


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def test_add_with_ids_updates_len_and_contains():
    idx = IdMapIndex(dim=128, bit_width=4)
    idx.add_with_ids(unit_vectors(5, 128), np.array([10, 20, 30, 40, 50], dtype=np.uint64))
    assert len(idx) == 5
    assert idx.contains(30)
    assert not idx.contains(99)
    # __contains__ sugar
    assert 30 in idx
    assert 99 not in idx


def test_search_returns_external_ids():
    idx = IdMapIndex(dim=256, bit_width=4)
    vectors = unit_vectors(10, 256, seed=0)
    ids = np.arange(1_000_000, 1_000_010, dtype=np.uint64)
    idx.add_with_ids(vectors, ids)

    _, got = idx.search(vectors, k=1)
    # Each self-query should return its own external id as top-1.
    np.testing.assert_array_equal(got[:, 0], ids)


def test_remove_returns_true_false_correctly():
    idx = IdMapIndex(dim=128, bit_width=4)
    idx.add_with_ids(unit_vectors(3, 128), np.array([1, 2, 3], dtype=np.uint64))
    assert idx.remove(2) is True
    assert len(idx) == 2
    assert idx.remove(2) is False  # already gone
    assert idx.remove(999) is False  # never existed


def test_remove_then_re_add_same_id():
    idx = IdMapIndex(dim=128, bit_width=4)
    idx.add_with_ids(unit_vectors(5, 128), np.array([1, 2, 3, 4, 5], dtype=np.uint64))
    assert idx.remove(3)
    new_vec = unit_vectors(1, 128, seed=42)
    idx.add_with_ids(new_vec, np.array([3], dtype=np.uint64))
    assert 3 in idx
    assert len(idx) == 5


def test_remaining_ids_self_query_after_removes():
    dim = 256
    idx = IdMapIndex(dim=dim, bit_width=4)
    vectors = unit_vectors(15, dim, seed=0)
    ids = np.array([i * 7 + 11 for i in range(15)], dtype=np.uint64)
    idx.add_with_ids(vectors, ids)

    # Remove a few positions.
    removed_positions = [5, 14, 0]
    for p in removed_positions:
        assert idx.remove(int(ids[p]))

    for i, id_val in enumerate(ids):
        if i in removed_positions:
            continue
        _, got = idx.search(vectors[i:i + 1], k=1)
        assert got[0, 0] == id_val, (
            f"id {id_val} (row {i}) didn't self-query after removes"
        )


def test_add_with_ids_rejects_duplicate_id():
    idx = IdMapIndex(dim=128, bit_width=4)
    idx.add_with_ids(unit_vectors(2, 128), np.array([1, 2], dtype=np.uint64))
    # Second call includes id=2 which is already present.
    with pytest.raises(BaseException):  # pyo3 surfaces Rust panic as PanicException/RuntimeError
        idx.add_with_ids(unit_vectors(1, 128, seed=1), np.array([2], dtype=np.uint64))


def test_write_and_load_round_trip(tmp_path):
    idx = IdMapIndex(dim=256, bit_width=4)
    vectors = unit_vectors(10, 256, seed=0)
    ids = np.arange(5000, 5010, dtype=np.uint64)
    idx.add_with_ids(vectors, ids)

    idx.remove(5004)
    idx.remove(5007)

    path = tmp_path / "idx.tvim"
    idx.write(str(path))

    restored = IdMapIndex.load(str(path))
    assert len(restored) == 8
    assert 5000 in restored
    assert 5004 not in restored
    assert 5007 not in restored

    for i, id_val in enumerate(ids):
        if id_val in (5004, 5007):
            continue
        _, got = restored.search(vectors[i:i + 1], k=1)
        assert got[0, 0] == id_val


def test_load_rejects_nonexistent_file():
    with pytest.raises(IOError):
        IdMapIndex.load("/nonexistent/path/does-not-exist.tvim")


@pytest.mark.parametrize("bad_bit_width", [0, 1, 5, 8])
def test_constructor_rejects_bad_bit_width(bad_bit_width):
    with pytest.raises(ValueError, match="bit_width"):
        IdMapIndex(dim=128, bit_width=bad_bit_width)


@pytest.mark.parametrize("bad_dim", [0, 1, 4, 7, 9])
def test_constructor_rejects_bad_dim(bad_dim):
    with pytest.raises(ValueError, match="dim"):
        IdMapIndex(dim=bad_dim, bit_width=4)


def test_search_on_empty_eager_index_returns_zero_effective_k():
    idx = IdMapIndex(dim=128, bit_width=4)
    q = unit_vectors(1, 128)
    _, ids = idx.search(q, k=3)
    assert ids.shape == (1, 0)


# ---- Wave 5: typed-exception hygiene + cross-class consistency ----

def test_search_empty_queries_returns_consistent_shape_across_index_types():
    # Cross-class invariant: passing an empty queries array with k>n
    # must produce the same shape on both index types. Previously,
    # IdMapIndex returned (0, k) (raw k) while TurboQuantIndex returned
    # (0, min(k, n_vectors)) — a silent divergence in result shape.
    from turbovec import TurboQuantIndex

    tq = TurboQuantIndex(dim=64, bit_width=4)
    im = IdMapIndex(dim=64, bit_width=4)
    tq.add(unit_vectors(3, 64))
    im.add_with_ids(unit_vectors(3, 64), np.array([1, 2, 3], dtype=np.uint64))

    empty_queries = np.empty((0, 64), dtype=np.float32)

    tq_scores, tq_indices = tq.search(empty_queries, k=5)
    im_scores, im_ids = im.search(empty_queries, k=5)

    assert tq_scores.shape == im_scores.shape
    assert tq_indices.shape == im_ids.shape
    # effective_k should be min(k=5, n_vectors=3) = 3.
    assert tq_scores.shape == (0, 3)


def test_search_query_dim_mismatch_raises_value_error():
    idx = IdMapIndex(dim=128, bit_width=4)
    idx.add_with_ids(unit_vectors(3, 128), np.array([1, 2, 3], dtype=np.uint64))
    wrong = unit_vectors(1, 64)
    with pytest.raises(ValueError, match="query dim"):
        idx.search(wrong, k=1)


def test_add_with_ids_noncontiguous_vectors_raises_value_error():
    idx = IdMapIndex(dim=128, bit_width=4)
    full = unit_vectors(2, 256)
    sliced = full[:, ::2]
    assert not sliced.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="contiguous"):
        idx.add_with_ids(sliced, np.array([1, 2], dtype=np.uint64))


def test_add_with_ids_rejects_nan_with_value_error():
    idx = IdMapIndex(dim=64, bit_width=4)
    data = unit_vectors(1, 64).copy()
    data[0, 5] = np.nan
    with pytest.raises(ValueError, match="invalid input value"):
        idx.add_with_ids(data, np.array([1], dtype=np.uint64))
