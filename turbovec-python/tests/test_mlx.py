"""Phase 1 parity tests for the turbovec.mlx Apple/Metal backend.

These cover the Rust -> Python -> MLX bridge only:
  * The Rust-supplied rotation matrix is actually orthogonal.
  * MLX matmul agrees with numpy matmul on the same R and same input
    (to within fp32 reduction-order noise).
  * The Lloyd-Max codebook has the right shape and monotonic boundaries.

They do NOT cover end-to-end encode/search correctness — that arrives
with the Metal kernels in phases 2 and 3.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from turbovec._turbovec import codebook as rust_codebook
from turbovec._turbovec import make_rotation_matrix as rust_make_rotation_matrix
from turbovec.mlx import TurboQuantIndex


@pytest.mark.parametrize("dim", [64, 200, 384, 1536])
def test_rust_rotation_matrix_is_orthogonal(dim):
    R = rust_make_rotation_matrix(dim)
    assert R.shape == (dim, dim)
    assert R.dtype == np.float32

    identity = R @ R.T
    np.testing.assert_allclose(identity, np.eye(dim, dtype=np.float32), atol=1e-4)


@pytest.mark.parametrize("dim", [64, 200, 1536])
def test_mlx_rotation_matches_numpy(dim):
    R = rust_make_rotation_matrix(dim)
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((8, dim)).astype(np.float32)
    x_np /= np.linalg.norm(x_np, axis=1, keepdims=True)

    expected = x_np @ R.T

    index = TurboQuantIndex(dim=dim, bit_width=4)
    x_mx = mx.array(x_np)
    actual = np.asarray(index._rotate(x_mx))

    np.testing.assert_allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 1536])
def test_codebook_shapes_and_monotonic(bit_width, dim):
    boundaries, centroids = rust_codebook(bit_width, dim)
    n_levels = 1 << bit_width

    assert boundaries.shape == (n_levels - 1,)
    assert centroids.shape == (n_levels,)
    assert np.all(np.diff(boundaries) > 0), "boundaries must be strictly increasing"
    assert np.all(np.diff(centroids) > 0), "centroids must be strictly increasing"


def test_index_construction_rejects_bad_bit_width():
    with pytest.raises(ValueError):
        TurboQuantIndex(dim=64, bit_width=3)


def test_phase3_search_still_stubbed():
    index = TurboQuantIndex(dim=64, bit_width=4)
    with pytest.raises(NotImplementedError):
        index.search(np.zeros((1, 64), dtype=np.float32), k=10)


def _random_unit_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_add_norms_match_rust(dim, bit_width):
    from turbovec._turbovec import encode as rust_encode

    vectors = _random_unit_vectors(32, dim, seed=0)
    rust_packed, rust_norms = rust_encode(vectors, bit_width)

    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(vectors)
    mlx_norms = np.asarray(index._norms)

    np.testing.assert_allclose(mlx_norms, rust_norms, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_add_codes_match_rust(dim, bit_width):
    """Byte-exact parity vs the Rust CPU encode.

    In principle, tiny float drift between Accelerate's GEMM and MLX's
    Metal matmul could flip codes for coordinates within ~1e-6 of a
    Lloyd-Max boundary. In practice we observe zero drift across the
    configs tested here — assert bit-exact equality, and we'll relax if
    a future config ever flakes.
    """
    from turbovec._turbovec import encode as rust_encode

    vectors = _random_unit_vectors(32, dim, seed=1)
    rust_packed, _ = rust_encode(vectors, bit_width)

    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(vectors)
    mlx_packed = np.asarray(index._packed_codes)

    assert mlx_packed.shape == rust_packed.shape
    assert mlx_packed.dtype == rust_packed.dtype
    assert np.array_equal(mlx_packed, rust_packed), (
        f"byte-level parity failed: "
        f"{np.unpackbits(rust_packed ^ mlx_packed).sum()} bits differ"
    )


def test_add_accumulates_across_calls():
    dim, bit_width = 128, 4
    vectors_a = _random_unit_vectors(10, dim, seed=2)
    vectors_b = _random_unit_vectors(7, dim, seed=3)

    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(vectors_a)
    index.add(vectors_b)
    assert len(index) == 17
    assert index._packed_codes.shape == (17, bit_width * dim // 8)
    assert index._norms.shape == (17,)
