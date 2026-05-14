"""Metal kernel sources for the turbovec MLX backend."""
from __future__ import annotations

import mlx.core as mx


# Apple simdgroup size; also a convenient threadgroup width for our kernels.
_TG_SIZE_DEFAULT = 32


_QUANTIZE_PACK_SOURCE = r"""
    // One threadgroup per vector. TG_SIZE threads cooperate.
    //
    // Inputs:
    //   rotated:    (n, DIM) float32 — pre-rotated unit vectors.
    //   boundaries: (N_LEVELS - 1,) float32 — Lloyd-Max boundaries.
    // Output:
    //   packed:     (n, BYTES_PER_VEC) uint8 — bit-plane layout matching
    //               turbovec/src/encode.rs::pack_codes. Plane p occupies
    //               bytes [p*PLANE_SIZE, (p+1)*PLANE_SIZE); within each
    //               plane, byte k holds coords [k*8, k*8+8), MSB-first.

    uint v = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;

    threadgroup uchar codes_local[DIM];
    threadgroup float bnd_local[N_LEVELS - 1];

    if (tid < N_LEVELS - 1) {
        bnd_local[tid] = boundaries[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < DIM; j += TG_SIZE) {
        float r = rotated[v * DIM + j];
        uchar code = 0;
        for (int b = 0; b < N_LEVELS - 1; b++) {
            code += (r > bnd_local[b]) ? 1u : 0u;
        }
        codes_local[j] = code;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = tid; k < BYTES_PER_VEC; k += TG_SIZE) {
        uint p = k / PLANE_SIZE;
        uint bp = k % PLANE_SIZE;
        uchar byte_val = 0;
        for (uint i = 0; i < 8; i++) {
            uchar code = codes_local[bp * 8 + i];
            byte_val |= ((code >> p) & 1u) << (7 - i);
        }
        packed[v * BYTES_PER_VEC + k] = byte_val;
    }
"""


def build_quantize_pack_kernel(dim: int, bit_width: int, tg_size: int = 128):
    """Compile the fused quantize + bit-pack Metal kernel for one
    ``(dim, bit_width)``.

    Returns a callable ``(rotated, boundaries) -> packed`` where
    ``packed`` is ``(n, bit_width * dim / 8)`` ``uint8`` in the
    bit-plane layout used by ``.tv`` files.
    """
    if dim % 8 != 0:
        raise ValueError(f"dim must be a multiple of 8, got {dim}")
    if bit_width not in (2, 4):
        raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")

    n_levels = 1 << bit_width
    plane_size = dim // 8
    bytes_per_vec = bit_width * plane_size

    header = (
        f"#define DIM {dim}\n"
        f"#define BIT_WIDTH {bit_width}\n"
        f"#define N_LEVELS {n_levels}\n"
        f"#define PLANE_SIZE {plane_size}\n"
        f"#define BYTES_PER_VEC {bytes_per_vec}\n"
        f"#define TG_SIZE {tg_size}\n"
    )

    kernel = mx.fast.metal_kernel(
        name=f"turbovec_quantize_pack_d{dim}_b{bit_width}",
        input_names=["rotated", "boundaries"],
        output_names=["packed"],
        source=_QUANTIZE_PACK_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )

    def call(rotated: "mx.array", boundaries: "mx.array") -> "mx.array":
        n = rotated.shape[0]
        outputs = kernel(
            inputs=[rotated, boundaries],
            grid=(tg_size, n, 1),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(n, bytes_per_vec)],
            output_dtypes=[mx.uint8],
        )
        return outputs[0]

    return call


_SCORE_SOURCE = r"""
    // One threadgroup per (query, vector) pair. TG_SIZE threads cooperate
    // over the inner DIM loop and tree-reduce at the end.
    //
    // Inputs:
    //   q_rot:     (nq, DIM) float32 — rotated queries.
    //   packed:    (n_db, BYTES_PER_VEC) uint8 — bit-plane codes
    //              (same layout as the encode kernel).
    //   centroids: (N_LEVELS,) float32 — Lloyd-Max centroids.
    //   norms:     (n_db,) float32 — per-vector L2 norms.
    // Output:
    //   scores:    (nq, n_db) float32 — dot products vs the reconstructed
    //              database vectors.

    uint tid = thread_position_in_threadgroup.x;
    uint v = threadgroup_position_in_grid.y;
    uint q = threadgroup_position_in_grid.z;
    uint n_db = threadgroups_per_grid.y;

    threadgroup float partial[TG_SIZE];
    threadgroup float cent_local[N_LEVELS];

    if (tid < N_LEVELS) {
        cent_local[tid] = centroids[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accum = 0.0f;

    for (uint bp = tid; bp < PLANE_SIZE; bp += TG_SIZE) {
        uchar bits[BIT_WIDTH];
        for (uint p = 0; p < BIT_WIDTH; p++) {
            bits[p] = packed[v * BYTES_PER_VEC + p * PLANE_SIZE + bp];
        }
        for (uint i = 0; i < 8; i++) {
            uint j = bp * 8 + i;
            uint shift = 7 - i;
            uchar code = 0;
            for (uint p = 0; p < BIT_WIDTH; p++) {
                code |= ((bits[p] >> shift) & 1u) << p;
            }
            accum += q_rot[q * DIM + j] * cent_local[code];
        }
    }

    partial[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = TG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        scores[q * n_db + v] = partial[0] * norms[v];
    }
"""


_SCORE_BATCHED_SOURCE = r"""
    // Batched scoring: one threadgroup per (VB vectors, query-block of
    // QB queries). TG_SIZE threads cooperate; codes for VB vectors are
    // decoded once into threadgroup memory and dot-producted against
    // QB queries. Each vector's BYTES_PER_VEC bytes are read once per
    // QB queries; each query's slice of q_rot is read once per VB
    // vectors. TG_SIZE >= VB * QB so one thread can own one output.

    uint tid = thread_position_in_threadgroup.x;
    uint v_base = threadgroup_position_in_grid.y * VB;
    uint q_block = threadgroup_position_in_grid.z;
    uint q_base = q_block * QB;
    uint n_db = threadgroups_per_grid.y * VB;

    // Per-coord centroid value cached as half — fuses the
    // `codes_local[j] -> cent_local[code]` two-read dependent chain
    // into a single TG read in the inner loop, and halves the TG
    // memory traffic.
    threadgroup half cent_for_coord[VB][DIM];
    threadgroup float cent_local[N_LEVELS];

    if (tid < N_LEVELS) {
        cent_local[tid] = centroids[tid];
    }

    for (uint vi = 0; vi < VB; vi++) {
        uint v = v_base + vi;
        for (uint bp = tid; bp < PLANE_SIZE; bp += TG_SIZE) {
            uchar bits[BIT_WIDTH];
            for (uint p = 0; p < BIT_WIDTH; p++) {
                bits[p] = packed[v * BYTES_PER_VEC + p * PLANE_SIZE + bp];
            }
            for (uint i = 0; i < 8; i++) {
                uchar code = 0;
                for (uint p = 0; p < BIT_WIDTH; p++) {
                    code |= ((bits[p] >> (7 - i)) & 1u) << p;
                }
                cent_for_coord[vi][bp * 8 + i] = half(cent_local[code]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accum[VB][QB];
    for (uint vi = 0; vi < VB; vi++) {
        for (uint qi = 0; qi < QB; qi++) accum[vi][qi] = 0.0f;
    }

    for (uint j = tid; j < DIM; j += TG_SIZE) {
        for (uint vi = 0; vi < VB; vi++) {
            float cent_val = float(cent_for_coord[vi][j]);
            for (uint qi = 0; qi < QB; qi++) {
                accum[vi][qi] += float(q_rot[(q_base + qi) * DIM + j]) * cent_val;
            }
        }
    }

    for (uint vi = 0; vi < VB; vi++) {
        for (uint qi = 0; qi < QB; qi++) {
            accum[vi][qi] = simd_sum(accum[vi][qi]);
        }
    }

    // VB * QB outputs per threadgroup. After simd_sum, every thread
    // holds the same reduced value for each (vi, qi), so any thread
    // can write any output.
    for (uint out = tid; out < VB * QB; out += TG_SIZE) {
        uint vi = out / QB;
        uint qi = out % QB;
        scores[(q_base + qi) * n_db + (v_base + vi)] = accum[vi][qi] * norms[v_base + vi];
    }
"""


def build_score_batched_kernel(
    dim: int,
    bit_width: int,
    qb: int = 4,
    vb: int = 1,
    tg_size: int = _TG_SIZE_DEFAULT,
):
    """Compile the query-batched scoring Metal kernel.

    Each threadgroup processes ``vb`` vectors × ``qb`` queries. With
    ``vb > 1`` the per-query q_rot reads are amortized across vectors
    as well as the per-vector code reads being amortized across
    queries. Caller must pad ``q_rot`` and ``packed`` so the totals
    are multiples of ``qb`` / ``vb``.
    """
    if dim % 8 != 0:
        raise ValueError(f"dim must be a multiple of 8, got {dim}")
    if bit_width not in (2, 4):
        raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")
    if qb < 1 or qb & (qb - 1):
        raise ValueError(f"qb must be a power of 2, got {qb}")
    if vb < 1 or vb & (vb - 1):
        raise ValueError(f"vb must be a power of 2, got {vb}")
    if vb * qb > tg_size and (vb * qb) % tg_size != 0:
        raise ValueError(
            f"vb * qb ({vb * qb}) must be <= tg_size ({tg_size}) "
            f"or a multiple of it"
        )

    n_levels = 1 << bit_width
    plane_size = dim // 8
    bytes_per_vec = bit_width * plane_size

    header = (
        f"#define DIM {dim}\n"
        f"#define BIT_WIDTH {bit_width}\n"
        f"#define N_LEVELS {n_levels}\n"
        f"#define PLANE_SIZE {plane_size}\n"
        f"#define BYTES_PER_VEC {bytes_per_vec}\n"
        f"#define TG_SIZE {tg_size}\n"
        f"#define QB {qb}\n"
        f"#define VB {vb}\n"
    )

    kernel = mx.fast.metal_kernel(
        name=f"turbovec_score_qb{qb}_vb{vb}_d{dim}_b{bit_width}",
        input_names=["q_rot", "packed", "centroids", "norms"],
        output_names=["scores"],
        source=_SCORE_BATCHED_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )

    def call(
        q_rot: "mx.array",
        packed: "mx.array",
        centroids: "mx.array",
        norms: "mx.array",
    ) -> "mx.array":
        nq_padded = q_rot.shape[0]
        n_db = packed.shape[0]
        if nq_padded % qb != 0:
            raise ValueError(
                f"q_rot.shape[0] ({nq_padded}) must be a multiple of qb ({qb})"
            )
        if n_db % vb != 0:
            raise ValueError(
                f"packed.shape[0] ({n_db}) must be a multiple of vb ({vb})"
            )
        outputs = kernel(
            inputs=[q_rot, packed, centroids, norms],
            grid=(tg_size, n_db // vb, nq_padded // qb),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(nq_padded, n_db)],
            output_dtypes=[mx.float32],
        )
        return outputs[0]

    return call


def build_score_kernel(dim: int, bit_width: int, tg_size: int = _TG_SIZE_DEFAULT):
    """Compile the dequantize-and-dot scoring Metal kernel.

    Returns a callable ``(q_rot, packed, centroids, norms) -> scores``
    where ``scores`` is shape ``(nq, n_db)`` ``float32``.
    """
    if dim % 8 != 0:
        raise ValueError(f"dim must be a multiple of 8, got {dim}")
    if bit_width not in (2, 4):
        raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")

    n_levels = 1 << bit_width
    plane_size = dim // 8
    bytes_per_vec = bit_width * plane_size

    header = (
        f"#define DIM {dim}\n"
        f"#define BIT_WIDTH {bit_width}\n"
        f"#define N_LEVELS {n_levels}\n"
        f"#define PLANE_SIZE {plane_size}\n"
        f"#define BYTES_PER_VEC {bytes_per_vec}\n"
        f"#define TG_SIZE {tg_size}\n"
    )

    kernel = mx.fast.metal_kernel(
        name=f"turbovec_score_d{dim}_b{bit_width}",
        input_names=["q_rot", "packed", "centroids", "norms"],
        output_names=["scores"],
        source=_SCORE_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )

    def call(
        q_rot: "mx.array",
        packed: "mx.array",
        centroids: "mx.array",
        norms: "mx.array",
    ) -> "mx.array":
        nq = q_rot.shape[0]
        n_db = packed.shape[0]
        outputs = kernel(
            inputs=[q_rot, packed, centroids, norms],
            grid=(tg_size, n_db, nq),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(nq, n_db)],
            output_dtypes=[mx.float32],
        )
        return outputs[0]

    return call
