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


def test_search_empty_index_returns_empty():
    index = TurboQuantIndex(dim=64, bit_width=4)
    scores, indices = index.search(np.zeros((2, 64), dtype=np.float32), k=10)
    assert scores.shape == (2, 0)
    assert indices.shape == (2, 0)
    assert scores.dtype == np.float32
    assert indices.dtype == np.int64


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


def _numpy_scores(packed, centroids, norms, q_rot, bit_width, dim):
    """Pure-numpy reference scorer mirroring the MLX kernel.

    Decodes the bit-plane packed bytes back to integer codes, looks up
    centroid values, dot-products with the rotated query, and scales by
    norms. Used as a precision-independent oracle for the MLX scoring
    kernel — does not match the Rust u8-LUT path exactly.
    """
    n_db, bytes_per_vec = packed.shape
    plane_size = dim // 8
    codes = np.zeros((n_db, dim), dtype=np.uint8)
    bit_pos = 7 - (np.arange(dim) % 8)
    byte_pos = np.arange(dim) // 8
    for p in range(bit_width):
        plane = packed[:, p * plane_size:(p + 1) * plane_size]
        bits = (plane[:, byte_pos] >> bit_pos) & 1
        codes |= bits.astype(np.uint8) << p
    decoded = centroids[codes]                       # (n_db, dim)
    return (q_rot @ decoded.T) * norms[None, :]      # (nq, n_db)


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_search_matches_numpy_oracle(dim, bit_width):
    """The MLX scoring kernel matches a pure-numpy reference.

    Independent of the Rust path — proves the kernel computes
    ``sum_j q_rot[j] * centroids[code_j] * norms[v]`` correctly.
    """
    from turbovec._turbovec import codebook as rust_codebook
    from turbovec._turbovec import make_rotation_matrix as rust_make_rotation_matrix

    n_db, n_q, k = 64, 4, 8
    db = _random_unit_vectors(n_db, dim, seed=30)
    queries = _random_unit_vectors(n_q, dim, seed=31)

    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(db)
    mlx_scores, mlx_idx = index.search(queries, k=k)

    R = rust_make_rotation_matrix(dim)
    _, centroids = rust_codebook(bit_width, dim)
    q_rot = queries @ R.T
    packed = np.asarray(index._packed_codes)
    norms = np.asarray(index._norms)
    full_scores = _numpy_scores(packed, centroids, norms, q_rot, bit_width, dim)

    expected_idx = np.argsort(-full_scores, axis=1)[:, :k]
    expected_scores = np.take_along_axis(full_scores, expected_idx, axis=1)

    # Returned top-k set matches numpy oracle exactly (no precision
    # divergence — MLX uses the same fp32 dequantize-and-dot path).
    for q in range(n_q):
        assert set(mlx_idx[q].tolist()) == set(expected_idx[q].tolist())
    np.testing.assert_allclose(
        np.sort(mlx_scores, axis=1),
        np.sort(expected_scores, axis=1),
        atol=1e-3,
    )


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_search_recall_vs_rust(dim, bit_width):
    """Recall@k against the Rust CPU search.

    The Rust path runs u8-LUT FastScan; MLX runs straight fp32
    dequantize-and-dot. Both compute the same TurboQuant scoring
    function but with different precision, so the top-k boundary can
    swap when consecutive scores are tied within u8-LUT rounding.
    We require recall@k >= 0.9 (typically 1.0 in the interior and
    occasionally drops a single boundary slot).
    """
    from turbovec import TurboQuantIndex as RustIndex

    n_db, n_q, k = 256, 8, 10
    db = _random_unit_vectors(n_db, dim, seed=10)
    queries = _random_unit_vectors(n_q, dim, seed=11)

    rust = RustIndex(dim=dim, bit_width=bit_width)
    rust.add(db)
    _, rust_idx = rust.search(queries, k=k)

    mlx_index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    mlx_index.add(db)
    _, mlx_idx = mlx_index.search(queries, k=k)

    recalls = [
        len(set(rust_idx[q].tolist()) & set(mlx_idx[q].tolist())) / k
        for q in range(n_q)
    ]
    mean_recall = sum(recalls) / n_q
    assert mean_recall >= 0.9, f"mean recall@{k} = {mean_recall:.3f}, recalls={recalls}"


def test_search_returns_fewer_than_k_when_index_small():
    dim, bit_width = 64, 4
    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(_random_unit_vectors(3, dim, seed=20))
    scores, idx = index.search(_random_unit_vectors(2, dim, seed=21), k=10)
    assert scores.shape == (2, 3)
    assert idx.shape == (2, 3)


# --- .tv / .tvim round-trip ---

@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_tv_round_trip_cpu_to_mlx(tmp_path, dim, bit_width):
    """A `.tv` file written by the Rust CPU backend loads bit-exactly
    into the MLX backend."""
    from turbovec import TurboQuantIndex as RustIndex
    from turbovec.mlx import TurboQuantIndex as MlxIndex

    vectors = _random_unit_vectors(48, dim, seed=40)
    rust = RustIndex(dim=dim, bit_width=bit_width)
    rust.add(vectors)
    path = tmp_path / "cpu.tv"
    rust.write(str(path))

    loaded = MlxIndex.load(str(path))
    assert loaded.dim == dim
    assert loaded.bit_width == bit_width
    assert len(loaded) == 48
    # Bit-exact vs the Rust encode oracle: writing and re-reading is a
    # pure byte copy, so MLX-loaded values must equal Rust's originals
    # bit-for-bit (not just close).
    from turbovec._turbovec import encode as rust_encode

    rust_packed, rust_norms = rust_encode(vectors, bit_width)
    assert np.array_equal(np.asarray(loaded._packed_codes), rust_packed)
    assert np.array_equal(np.asarray(loaded._norms), rust_norms)


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_tv_round_trip_mlx_to_cpu(tmp_path, dim, bit_width):
    """A `.tv` file written by the MLX backend loads into the CPU
    backend and produces working search results."""
    from turbovec import TurboQuantIndex as RustIndex
    from turbovec.mlx import TurboQuantIndex as MlxIndex

    db = _random_unit_vectors(48, dim, seed=41)
    queries = _random_unit_vectors(4, dim, seed=42)

    mlx_idx = MlxIndex(dim=dim, bit_width=bit_width)
    mlx_idx.add(db)
    path = tmp_path / "mlx.tv"
    mlx_idx.write(str(path))

    rust_loaded = RustIndex.load(str(path))
    assert rust_loaded.dim == dim
    assert rust_loaded.bit_width == bit_width
    assert len(rust_loaded) == 48

    # CPU search on the loaded index should produce sensible top-k
    # (most are nonempty and within the valid index range).
    scores, idx = rust_loaded.search(queries, k=5)
    assert scores.shape == (4, 5)
    assert idx.shape == (4, 5)
    assert (idx >= 0).all() and (idx < 48).all()


def test_tv_round_trip_empty(tmp_path):
    """An empty `.tv` file (no vectors added) round-trips correctly."""
    from turbovec.mlx import TurboQuantIndex as MlxIndex

    path = tmp_path / "empty.tv"
    src = MlxIndex(dim=64, bit_width=4)
    src.write(str(path))
    loaded = MlxIndex.load(str(path))
    assert len(loaded) == 0
    assert loaded.dim == 64
    assert loaded.bit_width == 4


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128])
def test_tvim_round_trip_cpu_to_mlx(tmp_path, dim, bit_width):
    """A `.tvim` file written by the Rust CPU backend loads correctly
    on the MLX backend, ids round-trip, and search returns ids."""
    from turbovec import IdMapIndex as RustIdMap
    from turbovec.mlx import IdMapIndex as MlxIdMap

    vectors = _random_unit_vectors(16, dim, seed=50)
    ids = np.arange(1000, 1016, dtype=np.uint64)

    cpu = RustIdMap(dim=dim, bit_width=bit_width)
    cpu.add_with_ids(vectors, ids)
    path = tmp_path / "cpu.tvim"
    cpu.write(str(path))

    mlx_loaded = MlxIdMap.load(str(path))
    assert mlx_loaded.dim == dim
    assert mlx_loaded.bit_width == bit_width
    assert len(mlx_loaded) == 16
    for id_ in ids:
        assert int(id_) in mlx_loaded

    queries = _random_unit_vectors(3, dim, seed=51)
    scores, returned_ids = mlx_loaded.search(queries, k=4)
    assert scores.shape == (3, 4)
    assert returned_ids.shape == (3, 4)
    assert returned_ids.dtype == np.uint64
    assert set(returned_ids.flatten().tolist()) <= set(int(i) for i in ids)


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128])
def test_tvim_round_trip_mlx_to_cpu(tmp_path, dim, bit_width):
    """A `.tvim` file written by the MLX backend loads on the CPU
    backend and exposes the same ids."""
    from turbovec import IdMapIndex as RustIdMap
    from turbovec.mlx import IdMapIndex as MlxIdMap

    vectors = _random_unit_vectors(16, dim, seed=52)
    ids = np.array([7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71], dtype=np.uint64)

    mlx_idx = MlxIdMap(dim=dim, bit_width=bit_width)
    mlx_idx.add_with_ids(vectors, ids)
    path = tmp_path / "mlx.tvim"
    mlx_idx.write(str(path))

    cpu_loaded = RustIdMap.load(str(path))
    assert cpu_loaded.dim == dim
    assert cpu_loaded.bit_width == bit_width
    assert len(cpu_loaded) == 16
    for id_ in ids:
        assert int(id_) in cpu_loaded


def test_id_map_rejects_duplicate_ids():
    from turbovec.mlx import IdMapIndex as MlxIdMap

    dim = 64
    vectors = _random_unit_vectors(4, dim, seed=60)
    index = MlxIdMap(dim=dim, bit_width=4)
    index.add_with_ids(vectors, np.array([1, 2, 3, 4], dtype=np.uint64))
    # Duplicate vs existing
    with pytest.raises(ValueError):
        index.add_with_ids(vectors[:1], np.array([2], dtype=np.uint64))
    # Duplicate within the batch
    with pytest.raises(ValueError):
        MlxIdMap(dim=dim, bit_width=4).add_with_ids(
            vectors, np.array([5, 6, 5, 7], dtype=np.uint64)
        )


# --- swap_remove / remove ---


def test_swap_remove_middle_shifts_last_into_slot():
    dim = 64
    vectors = _random_unit_vectors(5, dim, seed=70)
    index = TurboQuantIndex(dim=dim, bit_width=4)
    index.add(vectors)
    before_codes = np.asarray(index._packed_codes).copy()
    before_norms = np.asarray(index._norms).copy()

    moved_from = index.swap_remove(1)
    assert moved_from == 4
    assert len(index) == 4

    after_codes = np.asarray(index._packed_codes)
    after_norms = np.asarray(index._norms)

    # Slot 0 unchanged, slot 1 = original slot 4, slots 2 and 3 unchanged.
    assert np.array_equal(after_codes[0], before_codes[0])
    assert np.array_equal(after_codes[1], before_codes[4])
    assert np.array_equal(after_codes[2], before_codes[2])
    assert np.array_equal(after_codes[3], before_codes[3])
    np.testing.assert_array_equal(after_norms[0], before_norms[0])
    np.testing.assert_array_equal(after_norms[1], before_norms[4])
    np.testing.assert_array_equal(after_norms[2], before_norms[2])
    np.testing.assert_array_equal(after_norms[3], before_norms[3])


def test_swap_remove_last_truncates_only():
    dim = 64
    vectors = _random_unit_vectors(3, dim, seed=71)
    index = TurboQuantIndex(dim=dim, bit_width=4)
    index.add(vectors)
    before_codes = np.asarray(index._packed_codes).copy()

    moved_from = index.swap_remove(2)
    assert moved_from == 2
    assert len(index) == 2

    after_codes = np.asarray(index._packed_codes)
    assert np.array_equal(after_codes, before_codes[:2])


def test_swap_remove_down_to_empty():
    dim = 64
    vectors = _random_unit_vectors(2, dim, seed=72)
    index = TurboQuantIndex(dim=dim, bit_width=4)
    index.add(vectors)
    index.swap_remove(0)
    index.swap_remove(0)
    assert len(index) == 0
    assert index._packed_codes is None
    assert index._norms is None


def test_swap_remove_raises_on_invalid_index():
    index = TurboQuantIndex(dim=64, bit_width=4)
    with pytest.raises(IndexError):
        index.swap_remove(0)  # empty

    index.add(_random_unit_vectors(2, 64, seed=73))
    with pytest.raises(IndexError):
        index.swap_remove(5)
    with pytest.raises(IndexError):
        index.swap_remove(-1)


def test_id_map_remove_returns_false_for_missing():
    from turbovec.mlx import IdMapIndex as MlxIdMap

    dim = 64
    index = MlxIdMap(dim=dim, bit_width=4)
    index.add_with_ids(
        _random_unit_vectors(3, dim, seed=74),
        np.array([10, 20, 30], dtype=np.uint64),
    )
    assert index.remove(99) is False
    assert len(index) == 3


def test_id_map_remove_updates_mappings_and_search():
    from turbovec.mlx import IdMapIndex as MlxIdMap

    dim = 64
    vectors = _random_unit_vectors(5, dim, seed=75)
    ids = np.array([100, 200, 300, 400, 500], dtype=np.uint64)
    index = MlxIdMap(dim=dim, bit_width=4)
    index.add_with_ids(vectors, ids)

    assert index.remove(300) is True
    assert len(index) == 4
    assert 300 not in index
    assert 100 in index and 200 in index and 400 in index and 500 in index

    # Search must not return the removed id, even when the query is the
    # removed vector itself.
    scores, returned_ids = index.search(vectors[2:3], k=4)
    assert 300 not in set(int(x) for x in returned_ids.flatten())
    # All returned ids are still in the index.
    for id_ in returned_ids.flatten():
        assert int(id_) in index


def test_id_map_remove_round_trips_through_file(tmp_path):
    from turbovec.mlx import IdMapIndex as MlxIdMap

    dim = 64
    vectors = _random_unit_vectors(5, dim, seed=76)
    ids = np.array([7, 11, 13, 17, 19], dtype=np.uint64)
    index = MlxIdMap(dim=dim, bit_width=4)
    index.add_with_ids(vectors, ids)
    index.remove(13)

    path = tmp_path / "after_remove.tvim"
    index.write(str(path))
    loaded = MlxIdMap.load(str(path))
    assert len(loaded) == 4
    assert 13 not in loaded
    for id_ in (7, 11, 17, 19):
        assert id_ in loaded


def test_prepare_is_safe_on_empty_index():
    from turbovec.mlx import IdMapIndex as MlxIdMap

    TurboQuantIndex(dim=64, bit_width=4).prepare()
    MlxIdMap(dim=64, bit_width=4).prepare()


def test_prepare_materializes_after_many_adds():
    """After repeated `add` calls leave a long concat graph,
    `prepare()` should evaluate it without changing the visible state.
    """
    dim = 128
    index = TurboQuantIndex(dim=dim, bit_width=4)
    for seed in range(8):
        index.add(_random_unit_vectors(4, dim, seed=seed))
    n = len(index)
    codes_before = np.asarray(index._packed_codes).copy()

    index.prepare()

    assert len(index) == n
    codes_after = np.asarray(index._packed_codes)
    assert np.array_equal(codes_before, codes_after)

    # Search still works after prepare.
    scores, idx = index.search(_random_unit_vectors(2, dim, seed=99), k=3)
    assert scores.shape == (2, 3)
    assert idx.shape == (2, 3)


def test_id_map_remove_matches_rust_post_search_set():
    """After the same removal sequence on both backends, the surviving
    id sets agree."""
    from turbovec import IdMapIndex as RustIdMap
    from turbovec.mlx import IdMapIndex as MlxIdMap

    dim = 64
    vectors = _random_unit_vectors(8, dim, seed=77)
    ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64)

    rust = RustIdMap(dim=dim, bit_width=4)
    rust.add_with_ids(vectors, ids)
    mlx_idx = MlxIdMap(dim=dim, bit_width=4)
    mlx_idx.add_with_ids(vectors, ids)

    for rm in (3, 7, 1):
        assert rust.remove(rm)
        assert mlx_idx.remove(rm)

    assert len(rust) == len(mlx_idx) == 5
    surviving = {2, 4, 5, 6, 8}
    for id_ in surviving:
        assert id_ in rust
        assert id_ in mlx_idx
    for removed in (1, 3, 7):
        assert removed not in rust
        assert removed not in mlx_idx
