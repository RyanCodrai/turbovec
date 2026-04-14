//! SIMD-accelerated search pipeline.
//!
//! Scores queries against quantized database vectors using nibble-split
//! lookup tables with architecture-specific SIMD kernels:
//! - NEON on ARM (sequential code layout)
//! - AVX2 on x86 (FAISS-style perm0-interleaved layout)

use rayon::prelude::*;
use crate::{BLOCK, FLUSH_EVERY};

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
    out: &mut [f32; BLOCK],
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

    // Write 32 scores to output buffer, applying norms
    let end = (base_vec + BLOCK).min(n_vectors);
    let out_ptr = out.as_mut_ptr();
    let norms_ptr = norms.as_ptr().add(base_vec);

    if end - base_vec == BLOCK {
        for i in 0..8 {
            let n = vld1q_f32(norms_ptr.add(i * 4));
            vst1q_f32(out_ptr.add(i * 4), vmulq_f32(fa[i], n));
        }
    } else {
        let mut float_accum = [0.0f32; BLOCK];
        for i in 0..8 {
            vst1q_f32(float_accum.as_mut_ptr().add(i * 4), fa[i]);
        }
        for lane in 0..BLOCK {
            *out_ptr.add(lane) = if lane < end - base_vec {
                float_accum[lane] * *norms_ptr.add(lane)
            } else {
                f32::NEG_INFINITY
            };
        }
    }
}

// =============================================================================
// AVX2 scoring kernel for x86_64
// =============================================================================

/// Fused multi-query scoring + heap top-k. Processes NQ=4 queries per block,
/// sharing code loads. No score array materialization — heap updated per block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn search_multi_query_avx2(
    blocked_codes: &[u8],
    luts: &[&[u8]],
    scales: &[f32],
    biases: &[f32],
    n_byte_groups: usize,
    norms: &[f32],
    n_vectors: usize,
    nq: usize,
    k: usize,
    heap_scores: &mut [Vec<f32>],
    heap_indices: &mut [Vec<u32>],
    heap_sizes: &mut [usize],
    heap_mins: &mut [f32],
    heap_min_idxs: &mut [usize],
) {
    use std::arch::x86_64::*;

    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let mask = _mm256_set1_epi8(0x0F);
    let codes_base = blocked_codes.as_ptr();

    for b in 0..n_blocks {
        let base_vec = b * BLOCK;
        let mut accus = [[_mm256_setzero_si256(); 4]; 4];

        for g in 0..n_byte_groups {
            let cp = codes_base.add((b * n_byte_groups + g) * BLOCK);
            let codes_v = _mm256_loadu_si256(cp as *const __m256i);
            let clo = _mm256_and_si256(codes_v, mask);
            let chi = _mm256_and_si256(_mm256_srli_epi16(codes_v, 4), mask);

            for qi in 0..4 {
                let lut = _mm256_loadu_si256(luts[qi].as_ptr().add(g * 32) as *const __m256i);
                let res0 = _mm256_shuffle_epi8(lut, clo);
                let res1 = _mm256_shuffle_epi8(lut, chi);
                accus[qi][0] = _mm256_add_epi16(accus[qi][0], res0);
                accus[qi][1] = _mm256_add_epi16(accus[qi][1], _mm256_srli_epi16(res0, 8));
                accus[qi][2] = _mm256_add_epi16(accus[qi][2], res1);
                accus[qi][3] = _mm256_add_epi16(accus[qi][3], _mm256_srli_epi16(res1, 8));
            }
        }

        let end = (base_vec + BLOCK).min(n_vectors);
        let norms_ptr = norms.as_ptr().add(base_vec);

        for qi in 0..nq {
            let v_scale = _mm256_set1_ps(scales[qi]);
            let v_bias = _mm256_set1_ps(n_byte_groups as f32 * 2.0 * biases[qi]);

            accus[qi][0] = _mm256_sub_epi16(accus[qi][0], _mm256_slli_epi16(accus[qi][1], 8));
            accus[qi][2] = _mm256_sub_epi16(accus[qi][2], _mm256_slli_epi16(accus[qi][3], 8));

            let dis0 = _mm256_add_epi16(
                _mm256_permute2x128_si256(accus[qi][0], accus[qi][1], 0x21),
                _mm256_blend_epi32(accus[qi][0], accus[qi][1], 0xF0),
            );
            let dis1 = _mm256_add_epi16(
                _mm256_permute2x128_si256(accus[qi][2], accus[qi][3], 0x21),
                _mm256_blend_epi32(accus[qi][2], accus[qi][3], 0xF0),
            );

            let mut block_out = [0.0f32; BLOCK];
            let bp = block_out.as_mut_ptr();
            let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0)));
            let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0, 1)));
            let f2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1)));
            let f3 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1, 1)));

            if end - base_vec == BLOCK {
                for (i, f) in [f0, f1, f2, f3].iter().enumerate() {
                    let scored = _mm256_fmadd_ps(v_scale, *f, v_bias);
                    let n = _mm256_loadu_ps(norms_ptr.add(i * 8));
                    _mm256_storeu_ps(bp.add(i * 8), _mm256_mul_ps(scored, n));
                }
            } else {
                for (i, f) in [f0, f1, f2, f3].iter().enumerate() {
                    _mm256_storeu_ps(bp.add(i * 8), _mm256_fmadd_ps(v_scale, *f, v_bias));
                }
                for lane in 0..(end - base_vec) {
                    block_out[lane] *= *norms_ptr.add(lane);
                }
                for lane in (end - base_vec)..BLOCK {
                    block_out[lane] = f32::NEG_INFINITY;
                }
            }

            let hs = &mut heap_scores[qi];
            let hi = &mut heap_indices[qi];
            let sz = &mut heap_sizes[qi];
            let hmin = &mut heap_mins[qi];
            let hmi = &mut heap_min_idxs[qi];

            if *sz < k {
                for lane in 0..(end - base_vec) {
                    let score = block_out[lane];
                    if *sz < k {
                        hs[*sz] = score;
                        hi[*sz] = (base_vec + lane) as u32;
                        *sz += 1;
                        if *sz == k {
                            *hmin = hs[0]; *hmi = 0;
                            for h in 1..k {
                                if hs[h] < *hmin { *hmin = hs[h]; *hmi = h; }
                            }
                        }
                    } else if score > *hmin {
                        hs[*hmi] = score;
                        hi[*hmi] = (base_vec + lane) as u32;
                        *hmin = hs[0]; *hmi = 0;
                        for h in 1..k {
                            if hs[h] < *hmin { *hmin = hs[h]; *hmi = h; }
                        }
                    }
                }
            } else {
                let v_hmin = _mm256_set1_ps(*hmin);
                for chunk in 0..4 {
                    let chunk_start = chunk * 8;
                    if chunk_start >= end - base_vec { break; }
                    let scores_v = _mm256_loadu_ps(block_out.as_ptr().add(chunk_start));
                    let cmp = _mm256_cmp_ps(scores_v, v_hmin, _CMP_GT_OQ);
                    if _mm256_movemask_ps(cmp) == 0 { continue; }

                    let chunk_end = (chunk_start + 8).min(end - base_vec);
                    for lane in chunk_start..chunk_end {
                        let score = block_out[lane];
                        if score > *hmin {
                            hs[*hmi] = score;
                            hi[*hmi] = (base_vec + lane) as u32;
                            *hmi = 0;
                            for h in 1..k {
                                if hs[h] < hs[*hmi] { *hmi = h; }
                            }
                            *hmin = hs[*hmi];
                        }
                    }
                }
            }
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


/// Build nibble LUTs for NEON/AVX2 scoring from a flat query rotation row.
fn build_query_neon_lut_from_slice(
    q_rot_row: &[f32],
    centroids: &[f32],
    bits: usize,
    dim: usize,
) -> QueryNeonLut {
    let codes_per_byte = 8 / bits;
    let codes_per_nibble = codes_per_byte / 2;
    let n_byte_groups = dim / codes_per_byte;
    let code_mask = (1u16 << bits) - 1;

    let mut uint8_luts = vec![0u8; n_byte_groups * 32];
    let mut float_vals = vec![0.0f32; n_byte_groups * 32];
    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;

    for g in 0..n_byte_groups {
        let dim_start = g * codes_per_byte;

        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot_row[dim_start + c] * centroids[code as usize];
            }
            float_vals[g * 32 + nibble_val as usize] = s;
            global_min = global_min.min(s);
            global_max = global_max.max(s);
        }

        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot_row[dim_start + codes_per_nibble + c] * centroids[code as usize];
            }
            float_vals[g * 32 + 16 + nibble_val as usize] = s;
            global_min = global_min.min(s);
            global_max = global_max.max(s);
        }
    }

    let range = global_max - global_min;
    #[cfg(target_arch = "x86_64")]
    let max_lut = (65535.0 / (n_byte_groups as f64 * 2.0)).floor().min(127.0) as f32;
    #[cfg(not(target_arch = "x86_64"))]
    let max_lut = 127.0f32;

    let scale = if range > 1e-10 { range / max_lut } else { 1.0 };
    let bias = global_min;

    for i in 0..float_vals.len() {
        uint8_luts[i] = ((float_vals[i] - bias) / scale).round().min(max_lut) as u8;
    }

    QueryNeonLut { uint8_luts, scale, bias }
}

/// Full search: rotation + LUT build + scoring + heap top-k.
/// Returns (scores_flat, indices_flat) each of length nq * k.
pub fn search(
    queries: &[f32],    // (nq, dim) row-major
    nq: usize,
    rotation: &[f32],   // (dim, dim) row-major
    blocked_codes: &[u8],
    centroids: &[f32],
    norms: &[f32],
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
    k: usize,
) -> (Vec<f32>, Vec<i64>) {
    let k = k.min(n_vectors);
    let n_byte_groups = dim / (8 / bits);

    // Batched rotation: q_rot = queries @ rotation^T via a single GEMM.
    // Much faster than per-query matvec loops because it saturates FMA throughput
    // and reuses the rotation matrix across queries.
    let mut q_rot = vec![0.0f32; nq * dim];
    {
        let q_ref = faer::mat::from_row_major_slice::<f32, _, _>(queries, nq, dim);
        let r_ref = faer::mat::from_row_major_slice::<f32, _, _>(rotation, dim, dim);
        let out_mut = faer::mat::from_row_major_slice_mut::<f32, _, _>(&mut q_rot, nq, dim);
        faer::linalg::matmul::matmul(
            out_mut,
            q_ref,
            r_ref.transpose(),
            None,
            1.0_f32,
            faer::Parallelism::Rayon(0),
        );
    }

    // Build LUTs in parallel
    let query_luts: Vec<QueryNeonLut> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let row = &q_rot[qi * dim..(qi + 1) * dim];
            build_query_neon_lut_from_slice(row, centroids, bits, dim)
        })
        .collect();

    // Platform-specific scoring + top-k
    #[cfg(target_arch = "aarch64")]
    let results = {
        // ARM: 4-query fused scoring (shares code loads + nibble splits across queries)
        const QBS: usize = 4;
        let results: Vec<Vec<(Vec<f32>, Vec<i64>)>> = (0..nq)
            .step_by(QBS)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|qi_start| {
                let qi_end = (qi_start + QBS).min(nq);
                let batch_size = qi_end - qi_start;

                // Materialize per-query scores rows so the 4-query kernel can
                // write directly with offset = base_vec.
                let mut scores_flat = vec![f32::NEG_INFINITY; QBS * n_vectors];
                let rows: [*mut f32; QBS] = unsafe {
                    let p = scores_flat.as_mut_ptr();
                    [p, p.add(n_vectors), p.add(2 * n_vectors), p.add(3 * n_vectors)]
                };

                if batch_size == QBS {
                    // Fast path: 4-query fused kernel
                    let lut_refs: [&[u8]; QBS] = [
                        &query_luts[qi_start].uint8_luts,
                        &query_luts[qi_start + 1].uint8_luts,
                        &query_luts[qi_start + 2].uint8_luts,
                        &query_luts[qi_start + 3].uint8_luts,
                    ];
                    let scales: [f32; QBS] = [
                        query_luts[qi_start].scale,
                        query_luts[qi_start + 1].scale,
                        query_luts[qi_start + 2].scale,
                        query_luts[qi_start + 3].scale,
                    ];
                    let biases: [f32; QBS] = [
                        query_luts[qi_start].bias,
                        query_luts[qi_start + 1].bias,
                        query_luts[qi_start + 2].bias,
                        query_luts[qi_start + 3].bias,
                    ];
                    for block_idx in 0..n_blocks {
                        let base_vec = block_idx * BLOCK;
                        let block_offset = block_idx * n_byte_groups * BLOCK;
                        unsafe {
                            score_4query_block_neon(
                                blocked_codes, lut_refs, block_offset, n_byte_groups,
                                scales, biases, norms, base_vec, n_vectors, rows,
                            );
                        }
                    }
                } else {
                    // Tail path (batch_size < 4): single-query kernel per query
                    for qi_off in 0..batch_size {
                        let qi = qi_start + qi_off;
                        let qlut = &query_luts[qi];
                        let row_ptr = rows[qi_off];
                        for block_idx in 0..n_blocks {
                            let base_vec = block_idx * BLOCK;
                            let block_offset = block_idx * n_byte_groups * BLOCK;
                            let end = (base_vec + BLOCK).min(n_vectors);
                            let mut block_out = [0.0f32; BLOCK];
                            unsafe {
                                score_4bit_block_neon(
                                    blocked_codes, &qlut.uint8_luts, block_offset, n_byte_groups,
                                    qlut.scale, qlut.bias, norms, base_vec, n_vectors, &mut block_out,
                                );
                                for lane in 0..(end - base_vec) {
                                    *row_ptr.add(base_vec + lane) = block_out[lane];
                                }
                            }
                        }
                    }
                }

                // Per-query top-k scan over the materialized scores row.
                (0..batch_size)
                    .map(|qi_off| {
                        let row_start = qi_off * n_vectors;
                        let row = &scores_flat[row_start..row_start + n_vectors];
                        let mut heap_s = vec![f32::NEG_INFINITY; k];
                        let mut heap_i = vec![0u32; k];
                        let mut heap_sz = 0usize;
                        let mut heap_min = f32::NEG_INFINITY;
                        let mut heap_mi = 0usize;
                        for (i, &s) in row.iter().enumerate() {
                            if heap_sz < k {
                                heap_s[heap_sz] = s;
                                heap_i[heap_sz] = i as u32;
                                heap_sz += 1;
                                if heap_sz == k {
                                    heap_min = heap_s[0];
                                    heap_mi = 0;
                                    for h in 1..k {
                                        if heap_s[h] < heap_min {
                                            heap_min = heap_s[h];
                                            heap_mi = h;
                                        }
                                    }
                                }
                            } else if s > heap_min {
                                heap_s[heap_mi] = s;
                                heap_i[heap_mi] = i as u32;
                                heap_min = heap_s[0];
                                heap_mi = 0;
                                for h in 1..k {
                                    if heap_s[h] < heap_min {
                                        heap_min = heap_s[h];
                                        heap_mi = h;
                                    }
                                }
                            }
                        }
                        let mut pairs: Vec<(f32, u32)> = heap_s[..heap_sz].iter()
                            .zip(heap_i[..heap_sz].iter())
                            .map(|(&s, &i)| (s, i)).collect();
                        pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                        let s: Vec<f32> = pairs.iter().map(|p| p.0).collect();
                        let i: Vec<i64> = pairs.iter().map(|p| p.1 as i64).collect();
                        (s, i)
                    })
                    .collect()
            })
            .collect();
        results.into_iter().flatten().collect::<Vec<_>>()
    };

    #[cfg(target_arch = "x86_64")]
    let results = {
        const NQ_BATCH: usize = 4;
        let results: Vec<(Vec<f32>, Vec<i64>)> = (0..nq)
            .step_by(NQ_BATCH)
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(|qi_start| {
                let qi_end = (qi_start + NQ_BATCH).min(nq);
                let batch_nq = qi_end - qi_start;
                let pad_qi = qi_end - 1;
                let lut_refs: Vec<&[u8]> = (0..NQ_BATCH)
                    .map(|i| {
                        let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                        query_luts[qi].uint8_luts.as_slice()
                    }).collect();
                let scale_vals: Vec<f32> = (0..NQ_BATCH)
                    .map(|i| {
                        let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                        query_luts[qi].scale
                    }).collect();
                let bias_vals: Vec<f32> = (0..NQ_BATCH)
                    .map(|i| {
                        let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                        query_luts[qi].bias
                    }).collect();

                let mut heap_scores: Vec<Vec<f32>> = (0..batch_nq)
                    .map(|_| vec![f32::NEG_INFINITY; k]).collect();
                let mut heap_indices: Vec<Vec<u32>> = (0..batch_nq)
                    .map(|_| vec![0u32; k]).collect();
                let mut heap_sizes = vec![0usize; batch_nq];
                let mut heap_mins = vec![f32::NEG_INFINITY; batch_nq];
                let mut heap_min_idxs = vec![0usize; batch_nq];

                unsafe {
                    if is_x86_feature_detected!("avx2") {
                        search_multi_query_avx2(
                            blocked_codes, &lut_refs, &scale_vals, &bias_vals,
                            n_byte_groups, norms, n_vectors,
                            batch_nq, k,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    }
                }

                let mut batch_results = Vec::with_capacity(batch_nq);
                for qo in 0..batch_nq {
                    let sz = heap_sizes[qo];
                    let mut pairs: Vec<(f32, u32)> = heap_scores[qo][..sz].iter()
                        .zip(heap_indices[qo][..sz].iter())
                        .map(|(&s, &i)| (s, i)).collect();
                    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    batch_results.push((
                        pairs.iter().map(|p| p.0).collect::<Vec<f32>>(),
                        pairs.iter().map(|p| p.1 as i64).collect::<Vec<i64>>(),
                    ));
                }
                batch_results
            })
            .collect();
        results
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let results = {
        // Scalar fallback for other architectures
        let results: Vec<(Vec<f32>, Vec<i64>)> = (0..nq)
            .into_par_iter()
            .map(|qi| {
                let qlut = &query_luts[qi];
                let mut heap_s = vec![f32::NEG_INFINITY; k];
                let mut heap_i = vec![0u32; k];
                let mut heap_sz = 0usize;
                let mut heap_min = f32::NEG_INFINITY;
                let mut heap_mi = 0usize;

                for b in 0..n_blocks {
                    let base_vec = b * BLOCK;
                    let block_offset = b * n_byte_groups * BLOCK;
                    for lane in 0..BLOCK {
                        let vi = base_vec + lane;
                        if vi >= n_vectors { break; }
                        let mut score = 0.0f32;
                        for g in 0..n_byte_groups {
                            let byte_val = blocked_codes[block_offset + g * BLOCK + lane] as usize;
                            let hi = byte_val >> 4;
                            let lo = byte_val & 0x0F;
                            score += qlut.scale * qlut.uint8_luts[g * 32 + hi] as f32 + qlut.bias;
                            score += qlut.scale * qlut.uint8_luts[g * 32 + 16 + lo] as f32 + qlut.bias;
                        }
                        score *= norms[vi];
                        if heap_sz < k {
                            heap_s[heap_sz] = score; heap_i[heap_sz] = vi as u32; heap_sz += 1;
                            if heap_sz == k {
                                heap_min = heap_s[0]; heap_mi = 0;
                                for h in 1..k { if heap_s[h] < heap_min { heap_min = heap_s[h]; heap_mi = h; } }
                            }
                        } else if score > heap_min {
                            heap_s[heap_mi] = score; heap_i[heap_mi] = vi as u32;
                            heap_min = heap_s[0]; heap_mi = 0;
                            for h in 1..k { if heap_s[h] < heap_min { heap_min = heap_s[h]; heap_mi = h; } }
                        }
                    }
                }
                let mut pairs: Vec<(f32, u32)> = heap_s[..heap_sz].iter()
                    .zip(heap_i[..heap_sz].iter()).map(|(&s, &i)| (s, i)).collect();
                pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                (pairs.iter().map(|p| p.0).collect(), pairs.iter().map(|p| p.1 as i64).collect())
            })
            .collect();
        results
    };

    // Flatten into (scores, indices)
    let mut all_scores = Vec::with_capacity(nq * k);
    let mut all_indices = Vec::with_capacity(nq * k);
    for (s, i) in &results {
        let pad = k.saturating_sub(s.len());
        all_scores.extend_from_slice(s);
        all_scores.extend(std::iter::repeat(f32::NEG_INFINITY).take(pad));
        all_indices.extend_from_slice(i);
        all_indices.extend(std::iter::repeat(0i64).take(pad));
    }

    (all_scores, all_indices)
}
