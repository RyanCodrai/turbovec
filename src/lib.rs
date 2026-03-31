use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const BLOCK: usize = 32;

// Max byte groups per uint16 flush batch.
// With 7-bit LUT entries, hi+lo max = 254 per group.
// 256 * 254 = 65,024 < 65,535 (uint16 max).
const FLUSH_EVERY: usize = 256;

// Queries processed per rayon work unit.
// Larger = less scheduling overhead, but need enough units to fill all cores.
// 16 queries × 24KB LUT = 384KB (fits in L2).
const QUERY_BATCH: usize = 16;

/// Repack bit-plane codes into SIMD-blocked nibble layout.
fn repack_inner(
    packed_codes: ArrayView2<u8>,
    bits: usize,
    dim: usize,
) -> (Array1<u8>, usize) {
    let n = packed_codes.nrows();
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = (n + BLOCK - 1) / BLOCK;
    let blocked_size = n_blocks * n_byte_groups * BLOCK;

    let mut blocked = vec![0u8; blocked_size];

    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n {
                    continue;
                }

                let mut byte_val = 0u8;
                let dim_start = g * codes_per_byte;

                for c in 0..codes_per_byte {
                    let j = dim_start + c;
                    let byte_in_plane = j / 8;
                    let bit_in_byte = 7 - (j % 8);
                    let mask = 1u8 << bit_in_byte;

                    let mut code = 0u8;
                    for p in 0..bits {
                        let plane_byte =
                            packed_codes[[vec_idx, p * bytes_per_plane + byte_in_plane]];
                        if plane_byte & mask != 0 {
                            code |= 1 << p;
                        }
                    }

                    // Pack into nibble-aligned positions for NEON.
                    // For 2-bit (4 codes/byte) and 4-bit (2 codes/byte), bits*codes
                    // naturally aligns to nibbles. For 3-bit (2 codes/byte), we force
                    // nibble alignment: code0 → high nibble, code1 → low nibble.
                    let shift = if bits == 3 {
                        (codes_per_byte - 1 - c) * 4
                    } else {
                        (codes_per_byte - 1 - c) * bits
                    };
                    byte_val |= code << shift;
                }

                blocked[out_offset + lane] = byte_val;
            }
        }
    }

    (Array1::from_vec(blocked), n_blocks)
}

// =============================================================================
// 4-bit NEON scoring with periodic uint16→float flushing
// =============================================================================

// EVOLVE-BLOCK-START
#[cfg(target_arch = "aarch64")]
unsafe fn score_4bit_block_neon(
    blocked_codes: &[u8],
    uint8_luts: &[u8],
    block_offset: usize,
    n_byte_groups: usize,
    scale: f32,
    bias: f32,
    norms: &[f32],
    base_vec: usize,
    n_vectors: usize,
    row: &mut [f32],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let v_scale = vdupq_n_f32(scale);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    // Float accumulators in NEON registers (8 × float32x4 = 32 floats)
    let mut fa = [vdupq_n_f32(0.0); 8];

    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let luts_base = uint8_luts.as_ptr();

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
        let n_groups = g_end - g_start;

        let mut accum = [vdupq_n_u16(0); 4];

        // 4-group unrolled inner loop. Interleaves lookups to hide latency of vqtbl1q_u8
        let mut g = g_start;
        while g + 3 < g_end {
            let lp0 = luts_base.add(g * 32);
            let lp1 = luts_base.add((g + 1) * 32);
            let lp2 = luts_base.add((g + 2) * 32);
            let lp3 = luts_base.add((g + 3) * 32);
            let cp0 = codes_base.add(g * BLOCK);
            let cp1 = codes_base.add((g + 1) * BLOCK);
            let cp2 = codes_base.add((g + 2) * BLOCK);
            let cp3 = codes_base.add((g + 3) * BLOCK);

            for (lp, cp) in [(lp0, cp0), (lp1, cp1), (lp2, cp2), (lp3, cp3)] {
                let lut_hi = vld1q_u8(lp);
                let lut_lo = vld1q_u8(lp.add(16));
                let c0 = vld1q_u8(cp);
                let c1 = vld1q_u8(cp.add(16));
                let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)));
                let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)));
                accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
                accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
                accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
                accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            }
            g += 4;
        }

        // Handle remaining groups (0-3)
        while g < g_end {
            let lp = luts_base.add(g * 32);
            let lut_hi = vld1q_u8(lp);
            let lut_lo = vld1q_u8(lp.add(16));
            let cp = codes_base.add(g * BLOCK);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)),
                              vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)));
            let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)),
                              vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)));
            accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
            accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
            accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
            accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            g += 1;
        }

        // Flush: uint16 → float via NEON widening + fused multiply-add
        let v_bias = vdupq_n_f32(n_groups as f32 * 2.0 * bias);
        for i in 0..4 {
            // Split uint16x8 into two uint32x4, convert to float32x4
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum[i])));
            // fa += scale * val + bias
            fa[i * 2] = vaddq_f32(fa[i * 2], vfmaq_f32(v_bias, v_scale, lo));
            fa[i * 2 + 1] = vaddq_f32(fa[i * 2 + 1], vfmaq_f32(v_bias, v_scale, hi));
        }
    }

    // Write scores with norms (NEON multiply + store)
    let end = (base_vec + BLOCK).min(n_vectors);
    let row_ptr = row.as_mut_ptr().add(base_vec);
    let norms_ptr = norms.as_ptr().add(base_vec);

    if end - base_vec == BLOCK {
        // Full block: vectorized norm multiply
        for i in 0..8 {
            let n = vld1q_f32(norms_ptr.add(i * 4));
            vst1q_f32(row_ptr.add(i * 4), vmulq_f32(fa[i], n));
        }
    } else {
        // Partial block: scalar fallback
        let mut float_accum = [0.0f32; BLOCK];
        for i in 0..8 {
            vst1q_f32(float_accum.as_mut_ptr().add(i * 4), fa[i]);
        }
        for lane in 0..(end - base_vec) {
            *row_ptr.add(lane) = float_accum[lane] * *norms_ptr.add(lane);
        }
    }
}
// EVOLVE-BLOCK-END

/// Repack 3-bit codes into two blocked arrays:
/// - sub_codes: 2-bit nibble format from planes 0,1
/// - plane2: packed bits blocked by 32 vectors
fn repack_3bit_inner(
    packed_codes: ArrayView2<u8>,
    dim: usize,
) -> (Array1<u8>, Array1<u8>, usize) {
    let n = packed_codes.nrows();
    let bytes_per_plane = dim / 8;
    let n_blocks = (n + BLOCK - 1) / BLOCK;

    let sub_byte_groups = dim / 4;
    let mut sub_codes = vec![0u8; n_blocks * sub_byte_groups * BLOCK];

    let plane2_byte_groups = bytes_per_plane;
    let mut plane2_blocked = vec![0u8; n_blocks * plane2_byte_groups * BLOCK];

    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;

        for g in 0..sub_byte_groups {
            let out_offset = (block_idx * sub_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n { continue; }

                let mut byte_val = 0u8;
                let dim_start = g * 4;
                for c in 0..4usize {
                    let j = dim_start + c;
                    let byte_in_plane = j / 8;
                    let bit_in_byte = 7 - (j % 8);
                    let mask = 1u8 << bit_in_byte;

                    let mut code = 0u8;
                    for p in 0..2usize {
                        let plane_byte = packed_codes[[vec_idx, p * bytes_per_plane + byte_in_plane]];
                        if plane_byte & mask != 0 { code |= 1 << p; }
                    }
                    byte_val |= code << ((3 - c) * 2);
                }
                sub_codes[out_offset + lane] = byte_val;
            }
        }

        for g in 0..plane2_byte_groups {
            let out_offset = (block_idx * plane2_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n { continue; }
                plane2_blocked[out_offset + lane] = packed_codes[[vec_idx, 2 * bytes_per_plane + g]];
            }
        }
    }

    (Array1::from_vec(sub_codes), Array1::from_vec(plane2_blocked), n_blocks)
}

/// Score one block for TWO queries, sharing code loads.
/// Loads each code byte once, looks up in both queries' LUTs.
#[cfg(target_arch = "aarch64")]
unsafe fn score_2query_block_neon(
    blocked_codes: &[u8],
    luts_a: &[u8],
    luts_b: &[u8],
    block_offset: usize,
    n_byte_groups: usize,
    scale_a: f32,
    bias_a: f32,
    scale_b: f32,
    bias_b: f32,
    norms: &[f32],
    base_vec: usize,
    n_vectors: usize,
    row_a: &mut [f32],
    row_b: &mut [f32],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let v_scale_a = vdupq_n_f32(scale_a);
    let v_scale_b = vdupq_n_f32(scale_b);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    let mut fa_a = [vdupq_n_f32(0.0); 8];
    let mut fa_b = [vdupq_n_f32(0.0); 8];

    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let luts_a_base = luts_a.as_ptr();
    let luts_b_base = luts_b.as_ptr();

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
        let n_groups = g_end - g_start;

        let mut acc_a = [vdupq_n_u16(0); 4];
        let mut acc_b = [vdupq_n_u16(0); 4];

        for g in g_start..g_end {
            // Load codes ONCE — shared between both queries
            let cp = codes_base.add(g * BLOCK);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));

            // Split nibbles ONCE — shared
            let lo0 = vandq_u8(c0, mask);
            let lo1 = vandq_u8(c1, mask);
            let hi0 = vshrq_n_u8(c0, 4);
            let hi1 = vshrq_n_u8(c1, 4);

            // Query A lookups
            let lp_a = luts_a_base.add(g * 32);
            let lut_hi_a = vld1q_u8(lp_a);
            let lut_lo_a = vld1q_u8(lp_a.add(16));
            let s0_a = vaddq_u8(vqtbl1q_u8(lut_lo_a, lo0), vqtbl1q_u8(lut_hi_a, hi0));
            let s1_a = vaddq_u8(vqtbl1q_u8(lut_lo_a, lo1), vqtbl1q_u8(lut_hi_a, hi1));
            acc_a[0] = vaddw_u8(acc_a[0], vget_low_u8(s0_a));
            acc_a[1] = vaddw_u8(acc_a[1], vget_high_u8(s0_a));
            acc_a[2] = vaddw_u8(acc_a[2], vget_low_u8(s1_a));
            acc_a[3] = vaddw_u8(acc_a[3], vget_high_u8(s1_a));

            // Query B lookups (reuses same codes, split nibbles)
            let lp_b = luts_b_base.add(g * 32);
            let lut_hi_b = vld1q_u8(lp_b);
            let lut_lo_b = vld1q_u8(lp_b.add(16));
            let s0_b = vaddq_u8(vqtbl1q_u8(lut_lo_b, lo0), vqtbl1q_u8(lut_hi_b, hi0));
            let s1_b = vaddq_u8(vqtbl1q_u8(lut_lo_b, lo1), vqtbl1q_u8(lut_hi_b, hi1));
            acc_b[0] = vaddw_u8(acc_b[0], vget_low_u8(s0_b));
            acc_b[1] = vaddw_u8(acc_b[1], vget_high_u8(s0_b));
            acc_b[2] = vaddw_u8(acc_b[2], vget_low_u8(s1_b));
            acc_b[3] = vaddw_u8(acc_b[3], vget_high_u8(s1_b));
        }

        // Flush query A
        let v_bias_a = vdupq_n_f32(n_groups as f32 * 2.0 * bias_a);
        for i in 0..4 {
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc_a[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc_a[i])));
            fa_a[i * 2] = vaddq_f32(fa_a[i * 2], vfmaq_f32(v_bias_a, v_scale_a, lo));
            fa_a[i * 2 + 1] = vaddq_f32(fa_a[i * 2 + 1], vfmaq_f32(v_bias_a, v_scale_a, hi));
        }

        // Flush query B
        let v_bias_b = vdupq_n_f32(n_groups as f32 * 2.0 * bias_b);
        for i in 0..4 {
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc_b[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc_b[i])));
            fa_b[i * 2] = vaddq_f32(fa_b[i * 2], vfmaq_f32(v_bias_b, v_scale_b, lo));
            fa_b[i * 2 + 1] = vaddq_f32(fa_b[i * 2 + 1], vfmaq_f32(v_bias_b, v_scale_b, hi));
        }
    }

    // Write with norms
    let end = (base_vec + BLOCK).min(n_vectors);
    let row_a_ptr = row_a.as_mut_ptr().add(base_vec);
    let row_b_ptr = row_b.as_mut_ptr().add(base_vec);
    let norms_ptr = norms.as_ptr().add(base_vec);

    if end - base_vec == BLOCK {
        for i in 0..8 {
            let n = vld1q_f32(norms_ptr.add(i * 4));
            vst1q_f32(row_a_ptr.add(i * 4), vmulq_f32(fa_a[i], n));
            vst1q_f32(row_b_ptr.add(i * 4), vmulq_f32(fa_b[i], n));
        }
    } else {
        let mut buf_a = [0.0f32; BLOCK];
        let mut buf_b = [0.0f32; BLOCK];
        for i in 0..8 {
            vst1q_f32(buf_a.as_mut_ptr().add(i * 4), fa_a[i]);
            vst1q_f32(buf_b.as_mut_ptr().add(i * 4), fa_b[i]);
        }
        for lane in 0..(end - base_vec) {
            *row_a_ptr.add(lane) = buf_a[lane] * *norms_ptr.add(lane);
            *row_b_ptr.add(lane) = buf_b[lane] * *norms_ptr.add(lane);
        }
    }
}

/// Score one block for FOUR queries, sharing code loads and nibble splits.
/// Codes loaded once, nibbles split once, then looked up in 4 different LUTs.
#[cfg(target_arch = "aarch64")]
unsafe fn score_4query_block_neon(
    blocked_codes: &[u8],
    luts: [&[u8]; 4],
    block_offset: usize,
    n_byte_groups: usize,
    scales: [f32; 4],
    biases: [f32; 4],
    norms: &[f32],
    base_vec: usize,
    n_vectors: usize,
    rows: [*mut f32; 4],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    // Float accumulators on stack (flushed infrequently)
    let mut fa: [[float32x4_t; 8]; 4] = [[vdupq_n_f32(0.0); 8]; 4];

    let codes_base = blocked_codes.as_ptr().add(block_offset);

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
        let n_groups = g_end - g_start;

        let mut acc: [[uint16x8_t; 4]; 4] = [[vdupq_n_u16(0); 4]; 4];

        for g in g_start..g_end {
            // Load codes ONCE
            let cp = codes_base.add(g * BLOCK);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));

            // Split nibbles ONCE
            let lo0 = vandq_u8(c0, mask);
            let lo1 = vandq_u8(c1, mask);
            let hi0 = vshrq_n_u8(c0, 4);
            let hi1 = vshrq_n_u8(c1, 4);

            // Score 4 queries against the same nibbles
            for q in 0..4 {
                let lp = luts[q].as_ptr().add(g * 32);
                let lut_hi = vld1q_u8(lp);
                let lut_lo = vld1q_u8(lp.add(16));
                let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, lo0), vqtbl1q_u8(lut_hi, hi0));
                let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, lo1), vqtbl1q_u8(lut_hi, hi1));
                acc[q][0] = vaddw_u8(acc[q][0], vget_low_u8(s0));
                acc[q][1] = vaddw_u8(acc[q][1], vget_high_u8(s0));
                acc[q][2] = vaddw_u8(acc[q][2], vget_low_u8(s1));
                acc[q][3] = vaddw_u8(acc[q][3], vget_high_u8(s1));
            }
        }

        // Flush each query
        for q in 0..4 {
            let v_scale = vdupq_n_f32(scales[q]);
            let v_bias = vdupq_n_f32(n_groups as f32 * 2.0 * biases[q]);
            for i in 0..4 {
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc[q][i])));
                let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc[q][i])));
                fa[q][i * 2] = vaddq_f32(fa[q][i * 2], vfmaq_f32(v_bias, v_scale, lo));
                fa[q][i * 2 + 1] = vaddq_f32(fa[q][i * 2 + 1], vfmaq_f32(v_bias, v_scale, hi));
            }
        }
    }

    // Write with norms
    let end = (base_vec + BLOCK).min(n_vectors);
    let norms_ptr = norms.as_ptr().add(base_vec);

    for q in 0..4 {
        let rp = rows[q].add(base_vec);
        if end - base_vec == BLOCK {
            for i in 0..8 {
                let n = vld1q_f32(norms_ptr.add(i * 4));
                vst1q_f32(rp.add(i * 4), vmulq_f32(fa[q][i], n));
            }
        } else {
            let mut buf = [0.0f32; BLOCK];
            for i in 0..8 {
                vst1q_f32(buf.as_mut_ptr().add(i * 4), fa[q][i]);
            }
            for lane in 0..(end - base_vec) {
                *rp.add(lane) = buf[lane] * *norms_ptr.add(lane);
            }
        }
    }
}

/// Per-query nibble LUTs for NEON scoring (works for 2-bit and 4-bit).
struct QueryNeonLut {
    uint8_luts: Vec<u8>,  // n_byte_groups * 32 bytes: [hi_16 | lo_16] per group
    scale: f32,
    bias: f32,
}

/// Build nibble LUTs for NEON scoring.
///
/// For any bit width where codes_per_byte is even, split each byte into
/// two nibbles. Each nibble encodes codes_per_nibble codes of `bits` bits
/// each, giving 2^4 = 16 possible values — perfect for vqtbl1q_u8.
///
/// 2-bit: 4 codes/byte → 2 codes/nibble → nibble = code0<<2 | code1
/// 4-bit: 2 codes/byte → 1 code/nibble → nibble = code0
fn build_query_neon_lut(
    q_rot: ArrayView2<f32>,
    qi: usize,
    centroids: ArrayView1<f32>,
    bits: usize,
    dim: usize,
) -> QueryNeonLut {
    let codes_per_byte = 8 / bits;
    let codes_per_nibble = codes_per_byte / 2;
    let n_byte_groups = dim / codes_per_byte;
    let code_mask = (1u16 << bits) - 1;

    let mut uint8_luts = vec![0u8; n_byte_groups * 32];
    let mut float_vals = vec![0.0f32; n_byte_groups * 32]; // 16 hi + 16 lo per group
    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;

    for g in 0..n_byte_groups {
        let dim_start = g * codes_per_byte;

        // High nibble: first codes_per_nibble codes
        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot[[qi, dim_start + c]] * centroids[code as usize];
            }
            float_vals[g * 32 + nibble_val as usize] = s;
            global_min = global_min.min(s);
            global_max = global_max.max(s);
        }

        // Low nibble: last codes_per_nibble codes
        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot[[qi, dim_start + codes_per_nibble + c]] * centroids[code as usize];
            }
            float_vals[g * 32 + 16 + nibble_val as usize] = s;
            global_min = global_min.min(s);
            global_max = global_max.max(s);
        }
    }

    let range = global_max - global_min;
    let scale = if range > 1e-10 { range / 127.0 } else { 1.0 };
    let bias = global_min;

    for i in 0..float_vals.len() {
        uint8_luts[i] = ((float_vals[i] - bias) / scale).round().min(127.0) as u8;
    }

    QueryNeonLut { uint8_luts, scale, bias }
}

/// NEON-accelerated scoring for 2-bit and 4-bit codes.
fn score_neon_inner(
    q_rot: ArrayView2<f32>,
    blocked_codes: &[u8],
    centroids: ArrayView1<f32>,
    norms: ArrayView1<f32>,
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
) -> Array2<f32> {
    let nq = q_rot.nrows();
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let norms_slice = norms.as_slice().unwrap();

    // Prebuild all LUTs in parallel
    let query_luts: Vec<QueryNeonLut> = (0..nq)
        .into_par_iter()
        .map(|qi| build_query_neon_lut(q_rot, qi, centroids, bits, dim))
        .collect();

    // Block-parallel scoring: each thread owns a disjoint range of blocks.
    // For each block, score ALL queries. Each block's code data is loaded
    // once and reused across all queries — shared memory reads.
    // Threads write to non-overlapping vector ranges, so no synchronization needed.
    let mut flat_scores = vec![0.0f32; nq * n_vectors];

    // Shared pointer for parallel write to disjoint regions.
    // Safe because each block_idx writes to base_vec..base_vec+BLOCK,
    // which is non-overlapping across different block_idx values.
    let scores_base = std::sync::atomic::AtomicPtr::new(flat_scores.as_mut_ptr());

    (0..n_blocks).into_par_iter().for_each(|block_idx| {
        let base_vec = block_idx * BLOCK;
        let block_offset = block_idx * n_byte_groups * BLOCK;
        let ptr = scores_base.load(std::sync::atomic::Ordering::Relaxed);

        // Process queries in groups of 4 with shared code loads
        let mut qi = 0;
        while qi + 3 < nq {
            unsafe {

                #[cfg(target_arch = "aarch64")]
                score_4query_block_neon(
                    blocked_codes,
                    [&query_luts[qi].uint8_luts, &query_luts[qi+1].uint8_luts,
                     &query_luts[qi+2].uint8_luts, &query_luts[qi+3].uint8_luts],
                    block_offset, n_byte_groups,
                    [query_luts[qi].scale, query_luts[qi+1].scale,
                     query_luts[qi+2].scale, query_luts[qi+3].scale],
                    [query_luts[qi].bias, query_luts[qi+1].bias,
                     query_luts[qi+2].bias, query_luts[qi+3].bias],
                    norms_slice, base_vec, n_vectors,
                    [ptr.add(qi * n_vectors), ptr.add((qi+1) * n_vectors),
                     ptr.add((qi+2) * n_vectors), ptr.add((qi+3) * n_vectors)],
                );

                #[cfg(not(target_arch = "aarch64"))]
                { /* scalar fallback */ }
            }
            qi += 4;
        }

        // Handle remaining 1-3 queries with 2-query function
        while qi + 1 < nq {
            unsafe {
                #[cfg(target_arch = "aarch64")]
                score_2query_block_neon(
                    blocked_codes, &query_luts[qi].uint8_luts, &query_luts[qi+1].uint8_luts,
                    block_offset, n_byte_groups,
                    query_luts[qi].scale, query_luts[qi].bias,
                    query_luts[qi+1].scale, query_luts[qi+1].bias,
                    norms_slice, base_vec, n_vectors,
                    std::slice::from_raw_parts_mut(ptr.add(qi * n_vectors), n_vectors),
                    std::slice::from_raw_parts_mut(ptr.add((qi+1) * n_vectors), n_vectors),
                );
            }
            qi += 2;
        }

        if qi < nq {
            let qlut = &query_luts[qi];
            unsafe {
                let row = std::slice::from_raw_parts_mut(
                    ptr.add(qi * n_vectors),
                    n_vectors,
                );

                #[cfg(target_arch = "aarch64")]
                score_4bit_block_neon(
                    blocked_codes,
                    &qlut.uint8_luts,
                    block_offset,
                    n_byte_groups,
                    qlut.scale,
                    qlut.bias,
                    norms_slice,
                    base_vec,
                    n_vectors,
                    row,
                );

                #[cfg(not(target_arch = "aarch64"))]
                {
                    let mut block_scores = [0.0f32; BLOCK];
                    for g in 0..n_byte_groups {
                        let codes_start = block_offset + g * BLOCK;
                        let mut lut_hi = [0.0f32; 16];
                        let mut lut_lo = [0.0f32; 16];
                        for c in 0..16usize {
                            lut_hi[c] = qlut.scale * qlut.uint8_luts[g * 32 + c] as f32 + qlut.bias;
                            lut_lo[c] = qlut.scale * qlut.uint8_luts[g * 32 + 16 + c] as f32 + qlut.bias;
                        }
                        for lane in 0..BLOCK {
                            let byte_val = *blocked_codes.get_unchecked(codes_start + lane);
                            let hi = (byte_val >> 4) as usize;
                            let lo = (byte_val & 0x0F) as usize;
                            *block_scores.get_unchecked_mut(lane) +=
                                *lut_hi.get_unchecked(hi) + *lut_lo.get_unchecked(lo);
                        }
                    }
                    let end = (base_vec + BLOCK).min(n_vectors);
                    for lane in 0..(end - base_vec) {
                        *row.get_unchecked_mut(base_vec + lane) =
                            *block_scores.get_unchecked(lane) * *norms_slice.get_unchecked(base_vec + lane);
                    }
                }
            }
        }
    });

    let scores = Array2::from_shape_vec((nq, n_vectors), flat_scores).unwrap();

    scores
}

/// Generic scoring using byte-level LUTs (works for any bit width).
fn score_generic_inner(
    q_rot: ArrayView2<f32>,
    blocked_codes: &[u8],
    centroids: ArrayView1<f32>,
    norms: ArrayView1<f32>,
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
) -> Array2<f32> {
    let nq = q_rot.nrows();
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let shift_bits = bits;

    let rows: Vec<Vec<f32>> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let mut byte_lut = vec![0.0f32; n_byte_groups * 256];

            for g in 0..n_byte_groups {
                let dim_start = g * codes_per_byte;
                for v in 0u16..256 {
                    let mut s = 0.0f32;
                    for c in 0..codes_per_byte {
                        let shift = (codes_per_byte - 1 - c) * shift_bits;
                        let code = ((v as u8) >> shift) & ((1 << bits) - 1);
                        let j = dim_start + c;
                        s += q_rot[[qi, j]] * centroids[code as usize];
                    }
                    byte_lut[g * 256 + v as usize] = s;
                }
            }

            let mut row = vec![0.0f32; n_vectors];

            for block_idx in 0..n_blocks {
                let base_vec = block_idx * BLOCK;
                let block_offset = block_idx * n_byte_groups * BLOCK;
                let mut block_scores = [0.0f32; BLOCK];

                for g in 0..n_byte_groups {
                    let codes_start = block_offset + g * BLOCK;
                    let lut_base = &byte_lut[g * 256..g * 256 + 256];

                    for lane in 0..BLOCK {
                        unsafe {
                            let byte_val =
                                *blocked_codes.get_unchecked(codes_start + lane) as usize;
                            *block_scores.get_unchecked_mut(lane) +=
                                *lut_base.get_unchecked(byte_val);
                        }
                    }
                }

                let end = (base_vec + BLOCK).min(n_vectors);
                for lane in 0..(end - base_vec) {
                    row[base_vec + lane] = block_scores[lane] * norms[base_vec + lane];
                }
            }

            row
        })
        .collect();

    let mut scores = Array2::<f32>::zeros((nq, n_vectors));
    for (qi, row) in rows.into_iter().enumerate() {
        scores.row_mut(qi).as_slice_mut().unwrap().copy_from_slice(&row);
    }

    scores
}

#[pymodule]
mod py_turboquant {
    use super::*;

    #[pyfunction]
    fn repack<'py>(
        py: Python<'py>,
        packed_codes: PyReadonlyArray2<u8>,
        bits: usize,
        dim: usize,
    ) -> (Bound<'py, PyArray1<u8>>, usize) {
        let packed_codes = packed_codes.as_array();
        let (blocked, n_blocks) = repack_inner(packed_codes, bits, dim);
        (blocked.into_pyarray(py), n_blocks)
    }

    #[pyfunction]
    fn score<'py>(
        py: Python<'py>,
        q_rot: PyReadonlyArray2<f32>,
        blocked_codes: PyReadonlyArray1<u8>,
        centroids: PyReadonlyArray1<f32>,
        norms: PyReadonlyArray1<f32>,
        bits: usize,
        dim: usize,
        n_vectors: usize,
        n_blocks: usize,
    ) -> Bound<'py, PyArray2<f32>> {
        let q_rot = q_rot.as_array();
        let blocked_codes = blocked_codes.as_array();
        let centroids = centroids.as_array();
        let norms = norms.as_array();
        let codes_slice = blocked_codes.as_slice().unwrap();

        let result = if bits == 2 || bits == 3 || bits == 4 {
            score_neon_inner(
                q_rot, codes_slice, centroids, norms, bits, dim, n_vectors, n_blocks,
            )
        } else {
            score_generic_inner(
                q_rot, codes_slice, centroids, norms, bits, dim, n_vectors, n_blocks,
            )
        };

        result.into_pyarray(py)
    }
}