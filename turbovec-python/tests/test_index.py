"""Tests for TurboQuantIndex — the pyo3 binding surface.

These exercise the public Python API exposed by the Rust extension
(add, search, save/load, prepare, len, dim, bit_width), independent of
the LangChain / LlamaIndex wrappers.
"""
from __future__ import annotations

import numpy as np
import pytest

from turbovec import TurboQuantIndex


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def test_new_reports_dim_and_bit_width():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    assert idx.dim == 128
    assert idx.bit_width == 4
    assert len(idx) == 0


@pytest.mark.parametrize("bit_width", [2, 4])
def test_bit_width_options(bit_width):
    idx = TurboQuantIndex(dim=128, bit_width=bit_width)
    assert idx.bit_width == bit_width
    idx.add(unit_vectors(20, 128))
    assert len(idx) == 20


def test_add_updates_length():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(50, 128))
    assert len(idx) == 50


def test_add_is_incremental():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(20, 128, seed=1))
    idx.add(unit_vectors(30, 128, seed=2))
    assert len(idx) == 50


def test_search_shape():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(100, 128))
    scores, indices = idx.search(unit_vectors(5, 128, seed=99), k=10)
    assert scores.shape == (5, 10)
    assert indices.shape == (5, 10)


def test_search_single_query():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(100, 128))
    scores, indices = idx.search(unit_vectors(1, 128, seed=99), k=5)
    assert scores.shape == (1, 5)
    assert indices.shape == (1, 5)


def test_self_query_recall_at_1():
    vectors = unit_vectors(100, 256, seed=42)
    idx = TurboQuantIndex(dim=256, bit_width=4)
    idx.add(vectors)

    hits = 0
    for i in range(20):
        _, indices = idx.search(vectors[i:i + 1], k=1)
        if indices[0, 0] == i:
            hits += 1
    assert hits == 20, f"recall@1 failed: {hits}/20"


def test_save_load_roundtrip(tmp_path):
    vectors = unit_vectors(80, 128, seed=7)
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(vectors)
    idx.prepare()

    path = str(tmp_path / "idx.tv")
    idx.write(path)
    loaded = TurboQuantIndex.load(path)

    assert len(loaded) == 80
    assert loaded.dim == 128
    assert loaded.bit_width == 4

    q = unit_vectors(3, 128, seed=8)
    s_orig, i_orig = idx.search(q, k=10)
    s_load, i_load = loaded.search(q, k=10)
    np.testing.assert_array_equal(i_orig, i_load)
    np.testing.assert_allclose(s_orig, s_load, rtol=1e-5)


def test_prepare_is_idempotent():
    idx = TurboQuantIndex(dim=64, bit_width=4)
    idx.add(unit_vectors(20, 64))
    idx.prepare()
    idx.prepare()
    assert len(idx) == 20


def test_batch_query_matches_individual():
    idx = TurboQuantIndex(dim=256, bit_width=4)
    vectors = unit_vectors(50, 256, seed=0)
    idx.add(vectors)

    queries = unit_vectors(5, 256, seed=99)
    _, batch_indices = idx.search(queries, k=3)

    for i in range(5):
        _, single_indices = idx.search(queries[i:i + 1], k=3)
        np.testing.assert_array_equal(
            batch_indices[i:i + 1], single_indices
        )


def test_noncontiguous_input_is_handled():
    # A strided slice of a larger array should still work.
    big = unit_vectors(100, 128)
    strided = big[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    idx = TurboQuantIndex(dim=128, bit_width=4)
    # PyO3 layer asserts contiguity; caller is expected to convert.
    # This test documents that behaviour: a contiguous copy works.
    idx.add(np.ascontiguousarray(strided))
    assert len(idx) == 50


def test_swap_remove_shrinks_length():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(10, 128))
    moved_from = idx.swap_remove(3)
    assert moved_from == 9
    assert len(idx) == 9


def test_swap_remove_last_is_no_swap():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(5, 128))
    assert idx.swap_remove(4) == 4
    assert len(idx) == 4


def test_search_after_swap_remove_reflects_new_layout():
    # Cache-invalidation regression: the vector that moves into the
    # deleted slot must be findable immediately after the delete.
    idx = TurboQuantIndex(dim=256, bit_width=4)
    vectors = unit_vectors(20, 256, seed=0)
    idx.add(vectors)

    # Prime the cache with a self-query.
    _, pre = idx.search(vectors[5:6], k=1)
    assert pre[0, 0] == 5

    # Delete slot 5 — the last vector (index 19) moves into slot 5.
    idx.swap_remove(5)
    assert len(idx) == 19

    _, post = idx.search(vectors[19:20], k=1)
    assert post[0, 0] == 5, "vector that moved into slot 5 not found there"
