"""Metal kernel sources for the turbovec MLX backend."""
from __future__ import annotations

import mlx.core as mx


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
