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
