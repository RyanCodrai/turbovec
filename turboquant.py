"""
py-turboquant — TurboQuant_mse vector search (arXiv:2504.19874)
"""

import struct

import numpy as np
from scipy.integrate import quad
from scipy.stats import beta as beta_dist

from cache import disk_cache

from py_turboquant import repack as _rust_repack, score as _rust_score, score_topk as _rust_score_topk, search as _rust_search

ROTATION_SEED = 42
HEADER_FORMAT = "<BII"  # bit_width(u8), dim(u32), n_vectors(u32) = 9 bytes
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


# =============================================================================
# Codebook
# =============================================================================


def _make_beta(dim):
    """Beta((d-1)/2, (d-1)/2) on [-1, 1] — the exact coordinate distribution
    of a uniformly random point on S^{d-1} after orthogonal rotation."""
    a = (dim - 1) / 2.0
    return beta_dist(a, a, loc=-1, scale=2)


@disk_cache
def make_codebook(bits, dim, max_iter=200, tol=1e-12):
    rv = _make_beta(dim)
    n_levels = 1 << bits
    # Initialize centroids within ±3 std devs of the distribution
    spread = 3.0 * rv.std()
    centroids = np.linspace(-spread, spread, n_levels)

    for _ in range(max_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        edges = np.concatenate([[-1.0], boundaries, [1.0]])
        new_centroids = np.zeros(n_levels)

        for i in range(n_levels):
            lo, hi = edges[i], edges[i + 1]
            prob = rv.cdf(hi) - rv.cdf(lo)
            if prob < 1e-15:
                new_centroids[i] = centroids[i]
            else:
                mean, _ = quad(lambda x: x * rv.pdf(x), lo, hi)
                new_centroids[i] = mean / prob

        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return boundaries, centroids


# =============================================================================
# Rotation
# =============================================================================


@disk_cache
def make_rotation_matrix(dim):
    rng = np.random.RandomState(ROTATION_SEED)
    G = rng.randn(dim, dim).astype(np.float32)
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs[None, :]
    return Q


# =============================================================================
# Bit-Packing
# =============================================================================


def pack_codes(codes, bits):
    codes = codes.astype(np.uint8)
    planes = []
    for i in range(bits):
        plane = ((codes >> i) & 1).astype(np.uint8)
        planes.append(np.packbits(plane, axis=1))
    return np.concatenate(planes, axis=1)


def unpack_codes(packed, bits, d, j0=0, j1=None):
    """Unpack bit-plane packed codes to uint8. Optionally unpack only
    dimensions j0:j1 to avoid allocating the full (n, d) array."""
    if j1 is None:
        j1 = d
    bytes_per_plane = d // 8
    b0, b1 = j0 // 8, j1 // 8
    chunk_d = j1 - j0
    codes = np.zeros((packed.shape[0], chunk_d), dtype=np.uint8)
    for i in range(bits):
        plane_offset = i * bytes_per_plane
        plane = np.unpackbits(
            packed[:, plane_offset + b0 : plane_offset + b1], axis=1
        )[:, :chunk_d]
        codes |= plane << i
    return codes


# =============================================================================
# TurboQuantIndex
# =============================================================================


class TurboQuantIndex:
    def __init__(self, dim, bit_width):
        self.dim = dim
        self.bit_width = bit_width
        self.n_vectors = 0
        self.packed_codes = np.empty((0, 0), dtype=np.uint8)
        self.norms = np.empty(0, dtype=np.float32)
        self._blocked = None
        self._n_blocks = 0
        self._centroids = None

    def _encode(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)

        # 1. Extract norms and normalize to unit vectors on the hypersphere
        norms = np.linalg.norm(vectors, axis=-1).astype(np.float32)
        unit_vectors = vectors / np.maximum(norms, 1e-10)[..., None]

        # 2. Rotate so each coordinate follows a known distribution
        Q = make_rotation_matrix(self.dim)
        rotated = unit_vectors @ Q.T

        # 3. Quantize each coordinate to a small integer bucket
        boundaries, _ = make_codebook(self.bit_width, self.dim)
        codes = np.zeros(rotated.shape, dtype=np.uint8)
        for b in boundaries:
            codes += (rotated > b).view(np.uint8)

        # 4. Bit-pack the bucket indices for storage
        packed = pack_codes(codes, self.bit_width)

        return packed, norms

    def add(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        packed, norms = self._encode(vectors)

        if self.n_vectors == 0:
            self.packed_codes = packed
            self.norms = norms
        else:
            self.packed_codes = np.concatenate([self.packed_codes, packed], axis=0)
            self.norms = np.concatenate([self.norms, norms])

        self.n_vectors += len(vectors)
        self._blocked = None

    def _ensure_cached(self):
        if self._centroids is None:
            _, c = make_codebook(self.bit_width, self.dim)
            self._centroids = np.asarray(c, dtype=np.float32)
        if self._blocked is None:
            self._blocked, self._n_blocks = _rust_repack(
                self.packed_codes, self.bit_width, self.dim)

    def search(self, queries, k=10):
        self._ensure_cached()
        Q = make_rotation_matrix(self.dim)
        return _rust_search(
            np.asarray(queries, dtype=np.float32), Q,
            self._blocked, self._centroids, self.norms,
            self.bit_width, self.dim, self.n_vectors,
            self._n_blocks, k)

    def write(self, path):
        header = struct.pack(HEADER_FORMAT, self.bit_width, self.dim, self.n_vectors)
        with open(path, "wb") as f:
            f.write(header)
            f.write(self.packed_codes.tobytes())
            f.write(self.norms.tobytes())

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            header = struct.unpack(HEADER_FORMAT, f.read(HEADER_SIZE))
            bit_width, dim, n_vectors = header
            packed_bytes = (dim // 8) * bit_width * n_vectors
            packed = np.frombuffer(f.read(packed_bytes), dtype=np.uint8)
            packed = packed.reshape(n_vectors, -1)
            norms = np.frombuffer(f.read(n_vectors * 4), dtype=np.float32).copy()
        index = cls(dim=dim, bit_width=bit_width)
        index.n_vectors = n_vectors
        index.packed_codes = packed
        index.norms = norms
        return index
