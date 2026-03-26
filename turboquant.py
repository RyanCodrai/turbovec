"""
py-turboquant — TurboQuant_mse vector search (arXiv:2504.19874)
"""

import struct

import numpy as np
from scipy.stats import norm

ROTATION_SEED = 42
HEADER_FORMAT = "<BII"  # bit_width(u8), dim(u32), n_vectors(u32) = 9 bytes
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


# =============================================================================
# Codebook
# =============================================================================


class Codebook:
    def __init__(self, bits, max_iter=200, tol=1e-12):
        self.bits = bits
        n_levels = 1 << bits
        centroids = np.linspace(-3, 3, n_levels)

        for _ in range(max_iter):
            boundaries = (centroids[:-1] + centroids[1:]) / 2.0
            edges = np.concatenate([[-np.inf], boundaries, [np.inf]])
            new_centroids = np.zeros(n_levels)

            for i in range(n_levels):
                lo, hi = edges[i], edges[i + 1]
                prob = norm.cdf(hi) - norm.cdf(lo)
                if prob < 1e-15:
                    new_centroids[i] = centroids[i]
                else:
                    new_centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / prob

            if np.max(np.abs(new_centroids - centroids)) < tol:
                break
            centroids = new_centroids

        self.boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        self.centroids = centroids

    def scaled(self, dim):
        scale = 1.0 / np.sqrt(dim)
        return self.boundaries * scale, self.centroids * scale


# =============================================================================
# Rotation
# =============================================================================


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
        self.codebook = Codebook(bit_width)
        self._packed_codes = packed_codes
        self._norms = norms

    @classmethod
    def from_vectors(cls, vectors, bit_width=3):
        vectors = np.asarray(vectors, dtype=np.float32)
        n, dim = vectors.shape

        # 1. Extract norms and normalize to unit vectors on the hypersphere
        norms = np.linalg.norm(vectors, axis=-1).astype(np.float32)
        unit_vectors = vectors / np.maximum(norms, 1e-10)[..., None]

        # 2. Rotate so each coordinate follows a known distribution
        Q = make_rotation_matrix(dim)
        rotated = unit_vectors @ Q.T

        # 3. Quantize each coordinate to a small integer bucket
        boundaries, _ = Codebook(bit_width).scaled(dim)
        codes = np.searchsorted(boundaries, rotated).astype(np.uint8)

        # 4. Bit-pack the bucket indices for storage
        packed = pack_codes(codes, bit_width)

        return cls(
            dim=dim, bit_width=bit_width, n_vectors=n, packed_codes=packed, norms=norms
        )

    def search(self, queries, k=10):
        queries = np.asarray(queries, dtype=np.float32)
        _, centroids = self.codebook.scaled(self.dim)
        codes = unpack_codes(self._packed_codes, self.bit_width, self.dim)

        Q = make_rotation_matrix(self.dim)
        q_rot = queries @ Q.T

        centroid_vals = centroids[codes]
        scores = q_rot @ centroid_vals.T
        scores *= self._norms[None, :]

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
            f.write(self._packed_codes.tobytes())
            f.write(self._norms.tobytes())

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
