"""
py-turboquant — TurboQuant_mse vector search (arXiv:2504.19874)
"""

import struct

import numpy as np
from scipy.integrate import quad
from scipy.stats import beta as beta_dist

from cache import disk_cache

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


def unpack_codes(packed, bits, d):
    bytes_per_plane = d // 8
    codes = np.zeros((packed.shape[0], d), dtype=np.uint8)
    for i in range(bits):
        plane = np.unpackbits(
            packed[:, i * bytes_per_plane : (i + 1) * bytes_per_plane], axis=1
        )[:, :d]
        codes |= plane << i
    return codes


# =============================================================================
# TurboQuantIndex
# =============================================================================


class TurboQuantIndex:
    def __init__(self, dim, bit_width, n_vectors, packed_codes, norms):
        self.dim = dim
        self.bit_width = bit_width
        self.n_vectors = n_vectors
        self.packed_codes = packed_codes
        self.norms = norms
        self.codes = None  # lazily unpacked, invalidated on add_vectors

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
        codes = np.searchsorted(boundaries, rotated).astype(np.uint8)

        # 4. Bit-pack the bucket indices for storage
        packed = pack_codes(codes, self.bit_width)

        return packed, norms

    @classmethod
    def from_vectors(cls, vectors, bit_width=3):
        vectors = np.asarray(vectors, dtype=np.float32)
        n, dim = vectors.shape
        index = cls(dim=dim, bit_width=bit_width, n_vectors=0,
                    packed_codes=np.empty((0, 0), dtype=np.uint8),
                    norms=np.empty(0, dtype=np.float32))
        index.add_vectors(vectors)
        return index

    def add_vectors(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        packed, norms = self._encode(vectors)

        if self.n_vectors == 0:
            self.packed_codes = packed
            self.norms = norms
        else:
            self.packed_codes = np.concatenate([self.packed_codes, packed], axis=0)
            self.norms = np.concatenate([self.norms, norms])

        self.n_vectors += len(vectors)
        self.codes = None

    def search(self, queries, k=10):
        queries = np.asarray(queries, dtype=np.float32)
        _, centroids = make_codebook(self.bit_width, self.dim)
        centroids = np.asarray(centroids, dtype=np.float32)

        if self.codes is None:
            self.codes = unpack_codes(self.packed_codes, self.bit_width, self.dim)
        codes = self.codes

        Q = make_rotation_matrix(self.dim)
        q_rot = queries @ Q.T

        # LUT-based scoring: chunk along d to avoid materializing
        # the full (n_vectors, dim) float array of centroid values.
        # Each chunk expands only a slice of codes to floats, then
        # accumulates via BLAS matmul.
        scores = np.zeros((len(queries), self.n_vectors), dtype=np.float32)
        CHUNK = 256
        for j0 in range(0, self.dim, CHUNK):
            j1 = min(j0 + CHUNK, self.dim)
            chunk_vals = centroids[codes[:, j0:j1]]
            scores += q_rot[:, j0:j1] @ chunk_vals.T

        scores *= self.norms[None, :]

        k = min(k, self.n_vectors)
        top_idx = np.argpartition(-scores, k, axis=-1)[:, :k]
        top_scores = np.take_along_axis(scores, top_idx, axis=-1)
        order = np.argsort(-top_scores, axis=-1)
        top_idx = np.take_along_axis(top_idx, order, axis=-1)
        top_scores = np.take_along_axis(top_scores, order, axis=-1)
        return top_scores, top_idx

    def save(self, path):
        header = struct.pack(HEADER_FORMAT, self.bit_width, self.dim, self.n_vectors)
        with open(path, "wb") as f:
            f.write(header)
            f.write(self.packed_codes.tobytes())
            f.write(self.norms.tobytes())

    @classmethod
    def from_bin(cls, path):
        with open(path, "rb") as f:
            header = struct.unpack(HEADER_FORMAT, f.read(HEADER_SIZE))
            bit_width, dim, n_vectors = header
            packed_bytes = (dim // 8) * bit_width * n_vectors
            packed = np.frombuffer(f.read(packed_bytes), dtype=np.uint8)
            packed = packed.reshape(n_vectors, -1)
            norms = np.frombuffer(f.read(n_vectors * 4), dtype=np.float32).copy()
        return cls(
            dim=dim,
            bit_width=bit_width,
            n_vectors=n_vectors,
            packed_codes=packed,
            norms=norms,
        )
