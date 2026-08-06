//! SIMD-accelerated search pipeline.
//!
//! Scores queries against quantized database vectors using nibble-split
//! lookup tables with architecture-specific SIMD kernels:
//! - NEON on ARM (sequential code layout)
//! - AVX-512BW on x86 when available, with an AVX2 fallback
//!   (FAISS-style perm0-interleaved layout); selected at runtime via
//!   `is_x86_feature_detected!`
//! - a scalar fallback for any other target

use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

/// Block-count threshold above which a single unmasked query scans in
/// parallel. Bindings use [`single_query_parallelizes`] (which wraps
/// this) to decide when an nq=1 search must run inside the fork-safe
/// pool instead of inline.
///
/// Set to one full `MIN_TILE_BLOCKS`-sized tile: below that the batch
/// dispatch would not split the block axis either, so a single query
/// gains nothing from the pool and pays the `install` handoff plus a slot
/// in the process-wide pool queue (#336). The previous 256 fired at
/// n = 8192, where the handoff alone was larger than the whole scan.
/// Measured A/B interleaved (14-core arm64, dim=128, k=10, nq=1, inline
/// vs pooled, min-of-7 rounds): 0.64x at n=8192, 0.77x at 16384, 0.98x at
/// 32768, 1.34x at 65536 — inline is faster up to ~32k and the crossover
/// sits at 1024 blocks. At `RAYON_NUM_THREADS=1` inline is never slower
/// at any size.
pub const SINGLE_QUERY_PARALLEL_MIN_BLOCKS: usize = 1024;

/// Whether an nq=1 search over `n_vectors` is *large enough* to take the
/// block-parallel path.
///
/// This is the size half of the gate, and it is the whole gate on
/// aarch64. It is a **necessary but not sufficient** condition, not a
/// prediction: what it guarantees is the safe direction of the #147
/// invariant —
///
/// > `false` ⇒ the core never splits the block axis for that query, on
/// > every target.
///
/// The Python bindings consult it to decide whether an nq=1 search must
/// run inside the fork-safe rayon pool. Only the `false` direction has to
/// hold for that to be sound: routing a query into the pool that then
/// runs serially wastes an `install` handoff, whereas splitting outside
/// the pool would be a correctness bug.
///
/// `true` can still run serially, because each dispatch adds its own
/// terms after the size test:
///
/// * **aarch64** adds nothing — `nq == 1 && n_blocks >=
///   SINGLE_QUERY_PARALLEL_MIN_BLOCKS` is exactly the branch condition,
///   so here the predicate is exact.
/// * **x86_64** additionally requires runtime AVX2+FMA (or AVX-512BW +
///   AVX-512F + FMA). On a CPU without them the dedicated single-query
///   kernel is skipped and the batch dispatch is handed
///   `serial_required(.., simd_ok = false, ..) == true`, which pins
///   the block-range count at 1 — a fully serial scan at a size this
///   predicate calls parallel. That is the exact hardware
///   `score_query_into_heap` exists for.
///
/// Both halves are pinned by tests rather than by inspection:
/// `above_gate_single_query_does_split` and
/// `sub_gate_single_query_never_splits_the_block_axis` cover the size
/// term, and `each_term_of_the_serial_predicate_forces_serial_alone`
/// covers the x86 `simd_ok` term.
///
/// Neither dispatch calls this function directly: each re-tests
/// `SINGLE_QUERY_PARALLEL_MIN_BLOCKS` inline at its own branch, and
/// nothing makes those inline conditions agree with this function.
/// It is still reached on an nq=1 search, indirectly — when the inline
/// test sends a single query down the batch path, that path's
/// `n_block_ranges` tests `nq == 1 && !single_query_parallelizes(..)`
/// and clamps the block-range count to 1.
///
/// That clamp is a drift guard rather than a live safety mechanism.
/// While `SINGLE_QUERY_PARALLEL_MIN_BLOCKS == MIN_TILE_BLOCKS` (both
/// 1024) it is inert: a query it would clamp has fewer than
/// `MIN_TILE_BLOCKS` blocks, so the `n_blocks.div_ceil(min_tile_blocks)`
/// term already pins the count at 1 on its own. It only starts doing
/// work of its own if the two constants diverge — which is what makes
/// the threshold safe to move.
pub fn single_query_parallelizes(n_vectors: usize) -> bool {
    n_vectors.div_ceil(crate::BLOCK) >= SINGLE_QUERY_PARALLEL_MIN_BLOCKS
}

/// Smallest block-axis tile the batch dispatch will create. Below one
/// full tile the block axis is not split at all, so this is also the
/// floor for [`SINGLE_QUERY_PARALLEL_MIN_BLOCKS`] — a single query must
/// not be routed into the pool at a size where the same work, batched,
/// would not have been worth splitting (#336).
///
/// Hoisted from the two per-architecture dispatch bodies so the two
/// constants can be related in one place instead of drifting apart.
pub(crate) const MIN_TILE_BLOCKS: usize = 1024;


/// Whether the block axis must not be split at all, whatever the size.
///
/// Any one of these forces it: a mask (the allowlist walk is sequential),
/// no usable SIMD, or a caller-forced scalar path. Extracted from the
/// dispatch call site so the rule is unit-testable — inline it was a
/// three-term expression whose individual terms no test could reach, so
/// an `||` there could become `&&` unnoticed.
///
/// x86-only, because only the x86 dispatch derives `serial` from these
/// three terms — the aarch64 path passes a literal `false`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn serial_required(mask_present: bool, simd_ok: bool, force_scalar_any: bool) -> bool {
    mask_present || !simd_ok || force_scalar_any
}

/// Number of block-axis ranges the batch dispatch splits into.
///
/// Extracted so the gate is testable without timing and so both the
/// aarch64 and x86 dispatches share one rule. The `nq == 1` clamp is the
/// #147 invariant: [`single_query_parallelizes`] is what the Python
/// bindings consult to decide whether a search must run inside the
/// fork-safe pool, so a single query it calls *serial* must not reach
/// rayon here either.
#[inline]
fn n_block_ranges(
    nq: usize,
    n_quads: usize,
    n_blocks: usize,
    n_vectors: usize,
    k: usize,
    n_threads: usize,
    tiles_per_thread: usize,
    min_tile_blocks: usize,
    serial: bool,
) -> usize {
    if n_threads == 1 || serial || (nq == 1 && !single_query_parallelizes(n_vectors)) {
        return 1;
    }
    (n_threads * tiles_per_thread)
        .div_ceil(n_quads)
        .min(n_blocks.div_ceil(min_tile_blocks))
        .min(range_cap_for_k(n_vectors, k))
        .max(1)
}

/// How many tiles per worker the block-axis split aims to produce.
///
/// The tiles are the unit rayon load-balances over, and they are not
/// equal-cost, so the schedule is only as good as its granularity: with
/// too few, the final wave leaves most workers idle while the stragglers
/// finish. At nq=100 on 8 workers the old target of 4 gave 25 quads x 2
/// ranges = 50 tiles, i.e. 6.25 waves rounded up to 7 — an entire ragged
/// wave of waste. Raising the target lets the split run to the
/// `min_tile_blocks` cap (7 ranges, 175 tiles) where the tail is
/// amortized instead.
///
/// Swept interleaved at (200k, 768, 4-bit, nq=100, k=10), 5 rounds of
/// reps=15, medians, target = 4 / 16 / 32:
///
/// * arm (c4a-standard-8): 41.43 / 37.64 / **37.50** ms — x1.105
/// * x86 (c3-standard-8):  61.85 / 60.27 / **60.04** ms — x1.030
///
/// 32 beat 16 on both and the x86 samples do not overlap (60.02–60.06 vs
/// 60.23–60.29). Shapes where the other caps bind are unchanged: below
/// nq≈21 (at the benched 200k shape) the `min_tile_blocks` cap decided
/// the count under the old target too, and the count still falls to 1
/// once `n_quads` alone exceeds the target. Between nq≈21 and 64 the
/// count CAN rise to the block cap (e.g. nq=24: 6 → 7 ranges) — results
/// there are covered by the bitwise sweep below (nq=25 is in it), and
/// the cost delta is one more range's heap, bounded by the same cap
/// that binds at nq=100.
///
/// Results are unaffected — the cross-range merge is
/// (score desc, index asc) by construction, so the range count cannot
/// change what a search returns. Verified bitwise on both arches across
/// nq ∈ {1,4,25,100,257} x k ∈ {1,10,100} plus masked and tied-score
/// shapes (40 result arrays, identical at every count swept).
const TILES_PER_THREAD: usize = 32;

/// The same two knobs for the NEON dispatch, which wants a markedly finer
/// split than x86.
///
/// The difference is the per-range top-k, not the scan: splitting the block
/// axis gives every range its own `k`-entry heap, and the AVX-512 dispatch
/// pays more for that duplication than the finer schedule returns. Swept
/// block-major at (200k, 768, 4-bit, nq=100, k=10), medians against the
/// 7-range default:
///
/// | ranges | search-arm | search-x86 |
/// |---|---|---|
/// | 7 | 36.738 | **59.789** |
/// | 21 | **36.125** | 60.635 |
/// | 41 | 37.391 | 62.655 |
///
/// arm peaks at 21 ranges where x86 is already losing and keeps losing.
/// One shared constant cannot hold both, and the two dispatches are
/// separate `cfg` bodies with separate kernels anyway, so each carries its
/// own pair. Confirmed by an 8-round interleaved A/B: arm x1.0170
/// (36.738 -> 36.125, every candidate sample below every control sample),
/// x86 untouched by construction. See LOG_search.md H15.
#[cfg(target_arch = "aarch64")]
const TILES_PER_THREAD_NEON: usize = TILES_PER_THREAD * 2;

/// See [`TILES_PER_THREAD_NEON`]. Lower than [`MIN_TILE_BLOCKS`] so the
/// finer target can be reached; the
/// `SINGLE_QUERY_PARALLEL_MIN_BLOCKS >= MIN_TILE_BLOCKS` invariant relates
/// the single-query gate to the shared floor and is unaffected.
#[cfg(target_arch = "aarch64")]
const MIN_TILE_BLOCKS_NEON: usize = MIN_TILE_BLOCKS / 4;

/// TEMPORARY (H20): A/B hooks so the shipped NEON pair and the previous
/// one can be compared at several thread counts from a single build.
#[cfg(target_arch = "aarch64")]
fn neon_tpt() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("TV_NEON_MULT").ok().and_then(|v| v.parse().ok())
        .filter(|&v| v > 0).unwrap_or(TILES_PER_THREAD_NEON))
}
#[cfg(target_arch = "aarch64")]
fn neon_mtb() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("TV_NEON_CAP").ok().and_then(|v| v.parse().ok())
        .filter(|&v| v > 0).unwrap_or(MIN_TILE_BLOCKS_NEON))
}


/// Rescan a full top-k heap for its minimum. Ties on score resolve to
/// the LARGEST index — the eviction victim among tied minima — so that
/// sequential scans keep the lowest-index members of any tied cohort,
/// matching the block-parallel paths' index-ascending merges. This is
/// what makes top-k results identical across the batch, scalar, and
/// parallel single-query paths even for bitwise-tied scores (duplicate
/// vectors).
#[inline(always)]
fn rescan_min(hs: &[f32], hi: &[u64], k: usize) -> (f32, usize) {
    let mut mi = 0usize;
    for h in 1..k {
        if hs[h] < hs[mi] || (hs[h] == hs[mi] && hi[h] > hi[mi]) {
            mi = h;
        }
    }
    (hs[mi], mi)
}
/// Upper bound on block-range tiles for a given `k`.
///
/// Splitting the block axis duplicates the per-query top-k: each range
/// keeps its own `k`-entry heap (every replacement an O(k)
/// [`rescan_min`]) and the cross-range merge then sorts `n_ranges * k`
/// candidates. That cost grows with `k`, while the load-balancing
/// benefit of tiling does not — so past some point splitting is a net
/// loss. Bound the split by how many vectors each range would still
/// hold per unit of `k`. At the batch default (k=10, 200k vectors) this
/// never binds; at k=1000 it collapses to a single range, i.e. exactly
/// the untiled behavior.
#[inline]
fn range_cap_for_k(n_vectors: usize, k: usize) -> usize {
    // Swept on a c4a-standard-8 (200k x 768, 4-bit) over
    // k ∈ {10, 100, 200, 400, 1000} × nq ∈ {20, 100}: 256 left k=400 at
    // 1.2-1.5x, 1024 gave back the k=100 win; 512 holds every cell at or
    // below parity while preserving the k=10 win (0.69x / 0.84x).
    const MIN_VECTORS_PER_RANGE_PER_K: usize = 512;
    n_vectors
        .div_ceil(MIN_VECTORS_PER_RANGE_PER_K * k.max(1))
        .max(1)
}

/// Avoid the ragged schedule where the tile count lands just above the
/// worker count — one full round plus a long tail on mostly-idle
/// workers. When the caps push us into that zone, prefer a single round
/// instead: the duplicated per-range top-k is exactly the cost the `k`
/// cap exists to avoid, so paying it *and* getting a bad schedule is the
/// worst of both. (Measured: at nq=20 on 8 workers this is the whole
/// difference between 1.2x and 0.96x at k=400.)
#[inline]
fn smooth_tile_count(n_ranges: usize, n_quads: usize, n_threads: usize) -> usize {
    let tiles = n_quads * n_ranges;
    if tiles > n_threads && tiles < 2 * n_threads {
        (n_threads / n_quads).max(1)
    } else {
        n_ranges
    }
}

use crate::rotation::Rotation;
use crate::{BLOCK, FLUSH_EVERY};

/// Cumulative count of 32-vector blocks short-circuited by the mask
/// early-exit path, incremented by [`block_has_allowed`] and
/// [`block_pair_has_allowed`] — but **only** under the
/// `mask-skip-counter` feature. The per-skip atomic RMW landed on one
/// shared cache line in the masked hot loop, so counting every skip made
/// a more selective filter cost more (#294).
///
/// Deliberately not public: when counting is compiled out this reads
/// zero, which is indistinguishable from "nothing was skipped". Read it
/// through [`blocks_skipped_by_mask`], whose `Option` makes that
/// difference impossible to miss.
pub(crate) static BLOCKS_SKIPPED_BY_MASK: AtomicU64 = AtomicU64::new(0);

/// Test-only switch that forces the x86 dispatch to take the scalar
/// fallback even when AVX2/AVX-512 is available, so tests can exercise
/// `score_query_into_heap` on hardware that would otherwise always pick a
/// SIMD kernel. Compiled only under `cfg(test)` — zero cost in release.
#[cfg(test)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) static FORCE_SCALAR_FALLBACK: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Blocks short-circuited by the mask early-exit path since the last
/// [`reset_blocks_skipped_by_mask`], or `None` when the crate was built
/// without the `mask-skip-counter` feature.
///
/// The `Option` is the point: counting costs an atomic RMW per skipped
/// block on a shared cache line, so it is off by default (#294). Handing
/// a telemetry consumer a plain `0` in that case would be a silent lie —
/// "no blocks were skipped" and "this build does not count" are different
/// facts and must not share a representation (#368).
pub fn blocks_skipped_by_mask() -> Option<u64> {
    #[cfg(feature = "mask-skip-counter")]
    {
        Some(BLOCKS_SKIPPED_BY_MASK.load(Ordering::Relaxed))
    }
    #[cfg(not(feature = "mask-skip-counter"))]
    {
        None
    }
}

/// Reset the block-skip counter. Tests call this before issuing a
/// selective search to take a clean delta.
pub fn reset_blocks_skipped_by_mask() {
    BLOCKS_SKIPPED_BY_MASK.store(0, Ordering::Relaxed);
}

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn score_4bit_block_neon(
    blocked_codes: &[u8],
    uint8_luts: &[u8],
    block_offset: usize,
    n_byte_groups: usize,
    scale: f32,
    bias: f32,
    vec_scales: &[f32],
    base_vec: usize,
    n_vectors: usize,
    out: &mut [f32; BLOCK],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let v_scale = vdupq_n_f32(scale);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    // Float accumulators start at the total decode bias (sum of per-sub-table
    // mins). Flushes add `v_scale * acc` on top; the final values are the
    // calibrated per-vector scores (before norm multiplication).
    let mut fa = [vdupq_n_f32(bias); 8];

    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let luts_base = uint8_luts.as_ptr();

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);

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
        for i in 0..4 {
            // Split uint16x8 into two uint32x4, convert to float32x4
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum[i])));
            // fa += scale * val  (bias is added once after all flushes)
            fa[i * 2] = vfmaq_f32(fa[i * 2], v_scale, lo);
            fa[i * 2 + 1] = vfmaq_f32(fa[i * 2 + 1], v_scale, hi);
        }
    }

    // Write 32 scores to output buffer, applying vec_scales
    let end = (base_vec + BLOCK).min(n_vectors);
    let out_ptr = out.as_mut_ptr();
    let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);

    if end - base_vec == BLOCK {
        for i in 0..8 {
            let n = vld1q_f32(vec_scales_ptr.add(i * 4));
            vst1q_f32(out_ptr.add(i * 4), vmulq_f32(fa[i], n));
        }
    } else {
        let mut float_accum = [0.0f32; BLOCK];
        for i in 0..8 {
            vst1q_f32(float_accum.as_mut_ptr().add(i * 4), fa[i]);
        }
        for lane in 0..BLOCK {
            *out_ptr.add(lane) = if lane < end - base_vec {
                float_accum[lane] * *vec_scales_ptr.add(lane)
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
    vec_scales: &[f32],
    n_vectors: usize,
    nq: usize,
    k: usize,
    mask: Option<&[u64]>,
    heap_scores: &mut [Vec<f32>],
    heap_indices: &mut [Vec<u64>],
    heap_sizes: &mut [usize],
    heap_mins: &mut [f32],
    heap_min_idxs: &mut [usize],
) {
    use std::arch::x86_64::*;

    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    // SIMD nibble mask; named distinctly from the `mask: Option<&[u64]>`
    // function parameter (the slot allowlist) to avoid shadowing inside
    // the loops below where we test the slot mask.
    let nibble_mask = _mm256_set1_epi8(0x0F);
    let codes_base = blocked_codes.as_ptr();

    for b in 0..n_blocks {
        let base_vec = b * BLOCK;
        if !block_has_allowed(mask, base_vec) {
            continue;
        }

        // Per-query f32 score accumulators, seeded with the per-query bias so
        // the per-batch flush below is a single `fmadd(v_scale, partial, fa)`
        // — matching the operation sequence of `score_4bit_block_neon` on
        // ARM, which lets the two kernels produce bit-identical scores given
        // the same encoded LUTs.
        let v_scales: [__m256; 4] = [
            _mm256_set1_ps(scales[0]),
            _mm256_set1_ps(scales[1]),
            _mm256_set1_ps(scales[2]),
            _mm256_set1_ps(scales[3]),
        ];
        let v_biases: [__m256; 4] = [
            _mm256_set1_ps(biases[0]),
            _mm256_set1_ps(biases[1]),
            _mm256_set1_ps(biases[2]),
            _mm256_set1_ps(biases[3]),
        ];
        let mut fa = [
            [v_biases[0]; 4],
            [v_biases[1]; 4],
            [v_biases[2]; 4],
            [v_biases[3]; 4],
        ];

        // Batch the inner-group loop by FLUSH_EVERY so the per-half u16
        // accumulator can hold `FLUSH_EVERY * max_lut <= 65535` (256 * 127 =
        // 32512 ≪ 65535 with max_lut=127). Without this flush the AVX2
        // SUB-trick would require capping max_lut at 65535/n_byte_groups,
        // dropping LUT precision sharply at high dim — the source of the
        // historical ARM vs x86 recall gap.
        let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;
        for batch in 0..n_batches {
            let g_start = batch * FLUSH_EVERY;
            let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
            let mut accus = [[_mm256_setzero_si256(); 4]; 4];

            for g in g_start..g_end {
                let cp = codes_base.add((b * n_byte_groups + g) * BLOCK);
                let codes_v = _mm256_loadu_si256(cp as *const __m256i);
                let clo = _mm256_and_si256(codes_v, nibble_mask);
                let chi = _mm256_and_si256(_mm256_srli_epi16(codes_v, 4), nibble_mask);

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

            // Batch epilogue: SUB trick → combine → convert i16→f32 → FMA
            // into per-query f32 accumulator. fmadd(v_scale, partial, fa)
            // mirrors ARM's `vfmaq_f32(fa, v_scale, lo/hi)` per flush.
            for qi in 0..4 {
                let mut lo_a0 = accus[qi][0];
                let lo_a1 = accus[qi][1];
                let mut hi_a2 = accus[qi][2];
                let hi_a3 = accus[qi][3];
                lo_a0 = _mm256_sub_epi16(lo_a0, _mm256_slli_epi16(lo_a1, 8));
                hi_a2 = _mm256_sub_epi16(hi_a2, _mm256_slli_epi16(hi_a3, 8));

                let dis0 = _mm256_add_epi16(
                    _mm256_permute2x128_si256(lo_a0, lo_a1, 0x21),
                    _mm256_blend_epi32(lo_a0, lo_a1, 0xF0),
                );
                let dis1 = _mm256_add_epi16(
                    _mm256_permute2x128_si256(hi_a2, hi_a3, 0x21),
                    _mm256_blend_epi32(hi_a2, hi_a3, 0xF0),
                );

                let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0)));
                let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0, 1)));
                let f2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1)));
                let f3 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1, 1)));

                fa[qi][0] = _mm256_fmadd_ps(v_scales[qi], f0, fa[qi][0]);
                fa[qi][1] = _mm256_fmadd_ps(v_scales[qi], f1, fa[qi][1]);
                fa[qi][2] = _mm256_fmadd_ps(v_scales[qi], f2, fa[qi][2]);
                fa[qi][3] = _mm256_fmadd_ps(v_scales[qi], f3, fa[qi][3]);
            }
        }

        let end = (base_vec + BLOCK).min(n_vectors);
        let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);

        for qi in 0..nq {
            // fa already holds bias + Σ scale*partial — only vec_scales left.
            let f0 = fa[qi][0];
            let f1 = fa[qi][1];
            let f2 = fa[qi][2];
            let f3 = fa[qi][3];

            let mut block_out = [0.0f32; BLOCK];
            let bp = block_out.as_mut_ptr();

            if end - base_vec == BLOCK {
                for (i, f) in [f0, f1, f2, f3].iter().enumerate() {
                    let n = _mm256_loadu_ps(vec_scales_ptr.add(i * 8));
                    _mm256_storeu_ps(bp.add(i * 8), _mm256_mul_ps(*f, n));
                }
            } else {
                for (i, f) in [f0, f1, f2, f3].iter().enumerate() {
                    _mm256_storeu_ps(bp.add(i * 8), *f);
                }
                for lane in 0..(end - base_vec) {
                    block_out[lane] *= *vec_scales_ptr.add(lane);
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
                    if let Some(m) = mask {
                        if !mask_allows(m, base_vec + lane) { continue; }
                    }
                    let score = block_out[lane];
                    if *sz < k {
                        hs[*sz] = score;
                        hi[*sz] = (base_vec + lane) as u64;
                        *sz += 1;
                        if *sz == k {
                            let (m, mi) = rescan_min(hs, hi, k);
                            *hmin = m;
                            *hmi = mi;
                        }
                    } else if score > *hmin {
                        hs[*hmi] = score;
                        hi[*hmi] = (base_vec + lane) as u64;
                        let (m, mi) = rescan_min(hs, hi, k);
                        *hmin = m;
                        *hmi = mi;
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
                        if let Some(m) = mask {
                            if !mask_allows(m, base_vec + lane) { continue; }
                        }
                        let score = block_out[lane];
                        if score > *hmin {
                            hs[*hmi] = score;
                            hi[*hmi] = (base_vec + lane) as u64;
                            let (m, mi) = rescan_min(hs, hi, k);
                            *hmin = m;
                            *hmi = mi;
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// AVX-512BW scoring kernel for x86_64
// =============================================================================
//
// Processes pairs of consecutive BLOCK=32 blocks per inner-loop iteration,
// loading the two 32-byte code regions (which are NOT adjacent in the blocked
// layout — they're separated by the rest of block b's groups) into a single
// 512-bit register via `_mm512_inserti64x4`. The lane-local
// `_mm512_shuffle_epi8` then performs both blocks' lookups in one instruction
// pair (one for hi nibbles, one for lo). Re-uses the existing AVX2 pack
// layout and the existing 32-byte LUT format unchanged — the LUT is
// `_mm512_broadcast_i64x4`'d so both 256-bit halves see the same shuffle table.
//
// The lower 256 bits of each zmm accumulator hold block b's state and the
// upper 256 bits hold block b+1's. Periodically (every FLUSH_EVERY groups,
// to keep the u16 lane sums from overflowing) both halves are extracted
// into `__m256i` locals and folded into per-query f32 accumulators via
// `avx2_batch_flush_to_fa`; after the last batch a final
// `avx2_post_flush_heap_update` does the top-k heap insertion.
//
// Tail (when `n_blocks` is odd) processes the final unpaired block via an
// inlined AVX2 inner-loop body at the end. Avoids any masked AVX-512 logic.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "avx512f", enable = "avx512bw")]
unsafe fn search_multi_query_avx512bw(
    blocked_codes: &[u8],
    luts: &[&[u8]],
    scales: &[f32],
    biases: &[f32],
    n_byte_groups: usize,
    vec_scales: &[f32],
    n_vectors: usize,
    nq: usize,
    k: usize,
    mask: Option<&[u64]>,
    heap_scores: &mut [Vec<f32>],
    heap_indices: &mut [Vec<u64>],
    heap_sizes: &mut [usize],
    heap_mins: &mut [f32],
    heap_min_idxs: &mut [usize],
) {
    use std::arch::x86_64::*;

    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let n_block_pairs = n_blocks / 2;
    let mask512 = _mm512_set1_epi8(0x0F);
    let mask256 = _mm256_set1_epi8(0x0F);
    let codes_base = blocked_codes.as_ptr();

    // Per-query broadcast scales/biases shared across all batches in the
    // paired-block loop.
    let v_biases: [__m256; 4] = [
        _mm256_set1_ps(biases[0]),
        _mm256_set1_ps(biases[1]),
        _mm256_set1_ps(biases[2]),
        _mm256_set1_ps(biases[3]),
    ];
    let v_scales: [__m256; 4] = [
        _mm256_set1_ps(scales[0]),
        _mm256_set1_ps(scales[1]),
        _mm256_set1_ps(scales[2]),
        _mm256_set1_ps(scales[3]),
    ];

    // ----- Main loop: pairs of blocks ---------------------------------------
    for p in 0..n_block_pairs {
        let b0 = p * 2;
        let b1 = b0 + 1;

        // Pair-level early exit: each 64-vector pair aligns to a single
        // u64 mask word, so when the whole word is zero we can skip the
        // entire pair (no SIMD scoring, no epilogue) without disturbing
        // top-k correctness — masked slots never appear in results today.
        if !block_pair_has_allowed(mask, b0 * BLOCK) {
            continue;
        }

        // Per-query, per-block f32 accumulators (32 floats per block per
        // query). Seeded with the broadcast bias so each per-batch flush
        // becomes `fa = fmadd(v_scale, partial, fa)` — matches ARM's
        // `vfmaq_f32` per-flush sequence and the AVX2 kernel's flush path
        // bit-for-bit.
        let mut fa_b0: [[__m256; 4]; 4] = [
            [v_biases[0]; 4],
            [v_biases[1]; 4],
            [v_biases[2]; 4],
            [v_biases[3]; 4],
        ];
        let mut fa_b1 = fa_b0;

        // Batch the inner loop by FLUSH_EVERY=256 byte-groups, exactly as the
        // NEON and AVX2 kernels do, so the f32 fmadd flush boundaries — and
        // therefore the rounding — are identical across architectures. The
        // inner loop consumes two byte-groups per iteration for ILP; because
        // FLUSH_EVERY is even, every batch starts on an even group index and
        // the odd-group tail below can only fire on the final batch.
        debug_assert!(FLUSH_EVERY % 2 == 0);
        let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

        for batch in 0..n_batches {
            let g_start = batch * FLUSH_EVERY;
            let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);

            // Each zmm holds 32 u16 values: lower 256 bits = block b0's state,
            // upper 256 bits = block b1's. Reset per batch.
            let mut accus = [[_mm512_setzero_si512(); 4]; 4];

            let mut g_pair = g_start;
            while g_pair + 1 < g_end {
                let g0 = g_pair;
                let g1 = g0 + 1;

                let cp0_a = codes_base.add((b0 * n_byte_groups + g0) * BLOCK);
                let cp1_a = codes_base.add((b1 * n_byte_groups + g0) * BLOCK);
                let codes_a = _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm256_loadu_si256(cp0_a as *const __m256i)),
                    _mm256_loadu_si256(cp1_a as *const __m256i),
                    1,
                );

                let cp0_b = codes_base.add((b0 * n_byte_groups + g1) * BLOCK);
                let cp1_b = codes_base.add((b1 * n_byte_groups + g1) * BLOCK);
                let codes_b = _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm256_loadu_si256(cp0_b as *const __m256i)),
                    _mm256_loadu_si256(cp1_b as *const __m256i),
                    1,
                );

                let clo_a = _mm512_and_si512(codes_a, mask512);
                let chi_a = _mm512_and_si512(_mm512_srli_epi16(codes_a, 4), mask512);
                let clo_b = _mm512_and_si512(codes_b, mask512);
                let chi_b = _mm512_and_si512(_mm512_srli_epi16(codes_b, 4), mask512);

                for qi in 0..4 {
                    let lut_a = _mm512_broadcast_i64x4(
                        _mm256_loadu_si256(luts[qi].as_ptr().add(g0 * 32) as *const __m256i),
                    );
                    let lut_b = _mm512_broadcast_i64x4(
                        _mm256_loadu_si256(luts[qi].as_ptr().add(g1 * 32) as *const __m256i),
                    );

                    let res0_a = _mm512_shuffle_epi8(lut_a, clo_a);
                    let res1_a = _mm512_shuffle_epi8(lut_a, chi_a);
                    let res0_b = _mm512_shuffle_epi8(lut_b, clo_b);
                    let res1_b = _mm512_shuffle_epi8(lut_b, chi_b);

                    accus[qi][0] = _mm512_add_epi16(accus[qi][0], _mm512_add_epi16(res0_a, res0_b));
                    accus[qi][1] = _mm512_add_epi16(
                        accus[qi][1],
                        _mm512_add_epi16(_mm512_srli_epi16(res0_a, 8), _mm512_srli_epi16(res0_b, 8)),
                    );
                    accus[qi][2] = _mm512_add_epi16(accus[qi][2], _mm512_add_epi16(res1_a, res1_b));
                    accus[qi][3] = _mm512_add_epi16(
                        accus[qi][3],
                        _mm512_add_epi16(_mm512_srli_epi16(res1_a, 8), _mm512_srli_epi16(res1_b, 8)),
                    );
                }

                g_pair += 2;
            }

            // Tail: the odd last byte-group of this batch, when the batch holds
            // an odd number of groups. Only reachable on the final batch (see
            // the FLUSH_EVERY parity note above); current codebook shapes
            // always produce even n_byte_groups so this is defensive only.
            for g in g_pair..g_end {
                let cp0 = codes_base.add((b0 * n_byte_groups + g) * BLOCK);
                let cp1 = codes_base.add((b1 * n_byte_groups + g) * BLOCK);
                let codes_low = _mm256_loadu_si256(cp0 as *const __m256i);
                let codes_high = _mm256_loadu_si256(cp1 as *const __m256i);
                let codes_v = _mm512_inserti64x4(
                    _mm512_castsi256_si512(codes_low),
                    codes_high,
                    1,
                );
                let clo = _mm512_and_si512(codes_v, mask512);
                let chi = _mm512_and_si512(_mm512_srli_epi16(codes_v, 4), mask512);

                for qi in 0..4 {
                    let lut_low =
                        _mm256_loadu_si256(luts[qi].as_ptr().add(g * 32) as *const __m256i);
                    let lut = _mm512_broadcast_i64x4(lut_low);
                    let res0 = _mm512_shuffle_epi8(lut, clo);
                    let res1 = _mm512_shuffle_epi8(lut, chi);
                    accus[qi][0] = _mm512_add_epi16(accus[qi][0], res0);
                    accus[qi][1] = _mm512_add_epi16(accus[qi][1], _mm512_srli_epi16(res0, 8));
                    accus[qi][2] = _mm512_add_epi16(accus[qi][2], res1);
                    accus[qi][3] = _mm512_add_epi16(accus[qi][3], _mm512_srli_epi16(res1, 8));
                }
            }

            // Per-batch mini-epilogue: extract both 256-bit halves from each
            // zmm accumulator and flush them via the shared AVX2 helper.
            for qi in 0..4 {
                let block_accus_b0: [__m256i; 4] = [
                    _mm512_castsi512_si256(accus[qi][0]),
                    _mm512_castsi512_si256(accus[qi][1]),
                    _mm512_castsi512_si256(accus[qi][2]),
                    _mm512_castsi512_si256(accus[qi][3]),
                ];
                avx2_batch_flush_to_fa(block_accus_b0, v_scales[qi], &mut fa_b0[qi]);

                let block_accus_b1: [__m256i; 4] = [
                    _mm512_extracti64x4_epi64(accus[qi][0], 1),
                    _mm512_extracti64x4_epi64(accus[qi][1], 1),
                    _mm512_extracti64x4_epi64(accus[qi][2], 1),
                    _mm512_extracti64x4_epi64(accus[qi][3], 1),
                ];
                avx2_batch_flush_to_fa(block_accus_b1, v_scales[qi], &mut fa_b1[qi]);
            }
        }

        // ----- Final epilogue: per block, vec_scales + heap update ----------
        for which_block in 0..2usize {
            let b = b0 + which_block;
            let base_vec = b * BLOCK;
            if base_vec >= n_vectors { break; }
            if !block_has_allowed(mask, base_vec) { continue; }
            let end = (base_vec + BLOCK).min(n_vectors);
            let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);

            let fa = if which_block == 0 { &fa_b0 } else { &fa_b1 };
            for qi in 0..nq {
                avx2_post_flush_heap_update(
                    &fa[qi],
                    base_vec,
                    end,
                    vec_scales_ptr,
                    qi,
                    k,
                    mask,
                    heap_scores,
                    heap_indices,
                    heap_sizes,
                    heap_mins,
                    heap_min_idxs,
                );
            }
        }
    }

    // ----- Tail: any remaining unpaired block via the AVX2 flush body -------
    let bulk_blocks = n_block_pairs * 2;
    if bulk_blocks < n_blocks {
        let b = bulk_blocks;
        let base_vec = b * BLOCK;
        if !block_has_allowed(mask, base_vec) {
            return;
        }

        // Same flush structure as `search_multi_query_avx2`: per-query f32
        // accumulators seeded with bias, batched i16 accumulation with
        // periodic fmadd into fa.
        let mut fa: [[__m256; 4]; 4] = [
            [v_biases[0]; 4],
            [v_biases[1]; 4],
            [v_biases[2]; 4],
            [v_biases[3]; 4],
        ];

        let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;
        for batch in 0..n_batches {
            let g_start = batch * FLUSH_EVERY;
            let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
            let mut accus = [[_mm256_setzero_si256(); 4]; 4];

            for g in g_start..g_end {
                let cp = codes_base.add((b * n_byte_groups + g) * BLOCK);
                let codes_v = _mm256_loadu_si256(cp as *const __m256i);
                let clo = _mm256_and_si256(codes_v, mask256);
                let chi = _mm256_and_si256(_mm256_srli_epi16(codes_v, 4), mask256);

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

            for qi in 0..4 {
                avx2_batch_flush_to_fa(
                    [accus[qi][0], accus[qi][1], accus[qi][2], accus[qi][3]],
                    v_scales[qi],
                    &mut fa[qi],
                );
            }
        }

        let end = (base_vec + BLOCK).min(n_vectors);
        let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);
        for qi in 0..nq {
            avx2_post_flush_heap_update(
                &fa[qi],
                base_vec,
                end,
                vec_scales_ptr,
                qi,
                k,
                mask,
                heap_scores,
                heap_indices,
                heap_sizes,
                heap_mins,
                heap_min_idxs,
            );
        }
    }
}

/// Per-batch mini-epilogue: takes one block's 4×4 i16 accumulator matrix for
/// ONE query, runs the SUB trick + permute+blend combine + cvt-to-f32, then
/// FMAs `v_scale * partial` into the running f32 accumulators `fa`. Mirrors
/// the per-flush fmadd sequence used by `score_4bit_block_neon` on ARM so
/// scores across arches differ only by tied-rank f32 swaps.
// =============================================================================
// Vector-major VNNI scoring kernel for x86_64 (AVX-512 VBMI + VNNI)
// =============================================================================
//
// Scores a block of 32 vectors against 4 queries using `vpermb` for the table
// lookup and `vpdpbusd` for the accumulation, over codes in the vector-major
// layout (see `pack::vector_major_chunk`).
//
// Why a dot-product instruction is legal here at all: in the classic layout
// adjacent code bytes belong to different database vectors, so `vpdpbusd`'s
// 4-byte reduction would mix them. Vector-major puts one vector's codes for
// four consecutive byte-groups in each aligned 4-byte group, so the reduction
// sums four byte-groups *for that vector* and lands in that vector's dword
// lane. `vpermb`'s 6-bit index then lets byte position `j` select a different
// 16-entry sub-table from a 64-byte concatenation — which `vpshufb` cannot do,
// since it applies one 16-byte table per 128-bit lane.
//
// Two consequences beyond the op count: accumulation is u32, so there is no
// u16 overflow and hence no periodic f32 flush (the score rounds once at the
// end rather than every FLUSH_EVERY groups — more accurate, but NOT
// bit-identical to the classic kernel); and one accumulator per 16 vectors
// per query replaces the classic kernel's 16 live zmm.
//
// Measured x1.52 on the inner sequence and x1.388 over a full 73 MB scan; see
// `benchmarks/hillclimb/LOG_search.md` (P11-P13).
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vbmi",
    enable = "avx512vnni"
)]
#[allow(clippy::too_many_arguments)]
unsafe fn search_multi_query_vnni(
    blocked_codes: &[u8],
    split_luts: &[&[u8]],
    scales: &[f32],
    biases: &[f32],
    n_byte_groups: usize,
    vec_scales: &[f32],
    n_vectors: usize,
    nq: usize,
    k: usize,
    mask: Option<&[u64]>,
    heap_scores: &mut [Vec<f32>],
    heap_indices: &mut [Vec<u64>],
    heap_sizes: &mut [usize],
    heap_mins: &mut [f32],
    heap_min_idxs: &mut [usize],
) {
    use std::arch::x86_64::*;

    let n_blocks = n_vectors.div_ceil(BLOCK);
    let m0f = _mm512_set1_epi8(0x0F);
    // Per 32-bit lane: byte j of each vector's 4-byte group gets (j << 4), so
    // the permute index becomes (j << 4) | code.
    let kpos = _mm512_set1_epi32(0x3020_1000u32 as i32);
    let ones = _mm512_set1_epi8(1);
    let quads = n_byte_groups / 4;
    let block_bytes = n_byte_groups * BLOCK;

    for b in 0..n_blocks {
        let base_vec = b * BLOCK;
        if !block_has_allowed(mask, base_vec) {
            continue;
        }
        let block_base = b * block_bytes;
        // acc[q][h]: 16 u32 lanes = 16 vectors, halves h = vectors 0-15, 16-31.
        // Up to 8 queries: 16 zmm live, against the classic kernel's 16 at
        // only 4 queries.
        let mut acc = [[_mm512_setzero_si512(); 2]; 8];

        for q4 in 0..quads {
            for h in 0..2 {
                let c = _mm512_loadu_si512(
                    blocked_codes.as_ptr().add(block_base + q4 * 128 + h * 64) as *const __m512i,
                );
                let ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), kpos);
                let ihi = _mm512_or_si512(
                    _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f),
                    kpos,
                );
                for qi in 0..nq.min(8) {
                    let tp = split_luts[qi].as_ptr().add(q4 * 128);
                    let tlo = _mm512_loadu_si512(tp as *const __m512i);
                    let thi = _mm512_loadu_si512(tp.add(64) as *const __m512i);
                    acc[qi][h] = _mm512_dpbusd_epi32(
                        acc[qi][h],
                        _mm512_permutexvar_epi8(ilo, tlo),
                        ones,
                    );
                    acc[qi][h] = _mm512_dpbusd_epi32(
                        acc[qi][h],
                        _mm512_permutexvar_epi8(ihi, thi),
                        ones,
                    );
                }
            }
        }

        let end = (base_vec + BLOCK).min(n_vectors);
        for qi in 0..nq.min(8) {
            // score = acc * qscale + qbias, in the same lane order the classic
            // kernel's `fa` uses: 4 x 8 lanes covering vectors 0-7, 8-15,
            // 16-23, 24-31. Rounds once here rather than per flush.
            let vs = _mm256_set1_ps(scales[qi]);
            let vb = _mm256_set1_ps(biases[qi]);
            // The extract index must be a literal, so the four halves are
            // written out rather than looped.
            let cvt = |h: __m256i| _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(h), vs), vb);
            let fa = [
                cvt(_mm512_extracti32x8_epi32(acc[qi][0], 0)),
                cvt(_mm512_extracti32x8_epi32(acc[qi][0], 1)),
                cvt(_mm512_extracti32x8_epi32(acc[qi][1], 0)),
                cvt(_mm512_extracti32x8_epi32(acc[qi][1], 1)),
            ];
            avx2_post_flush_heap_update(
                &fa,
                base_vec,
                end,
                vec_scales.as_ptr().add(base_vec),
                qi,
                k,
                mask,
                heap_scores,
                heap_indices,
                heap_sizes,
                heap_mins,
                heap_min_idxs,
            );
        }
    }
}

/// Permute-dot scan over the vector-major layout (4-bit codes).
///
/// Same shape as [`search_multi_query_vnni`], with the per-query `vpermb`
/// replaced by one `vpshufb` pair *shared* across every query in the batch:
/// the nibble -> level map is the codebook itself, which no query depends
/// on. Per 64 bytes of codes, per query, that turns
///
///   2 x 64-byte LUT load + 2 `vpermb` + 2 `vpdpbusd`
///
/// into a 4-byte broadcast and 2 `vpdpbusd` — the query-side table traffic
/// drops from 128 bytes per group to 8, and the two shuffles amortize across
/// the batch instead of repeating per query.
///
/// No accumulator flush: `vpdpbusd` reduces into i32 lanes and the widest
/// possible sum over 768 dimensions is `768 * 255 * 127` ≈ 2.5e7, three
/// orders of magnitude inside i32. That removes the `FLUSH_EVERY` cadence
/// and, with it, the 7-bit table cap the flush imposed.
///
/// See P18 in `benchmarks/hillclimb/LOG_search.md`.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vnni"
)]
#[allow(clippy::too_many_arguments)]
unsafe fn search_multi_query_permute_dot(
    blocked_codes: &[u8],
    pds: &[&QueryPermuteDot],
    n_byte_groups: usize,
    vec_scales: &[f32],
    n_vectors: usize,
    nq: usize,
    k: usize,
    mask: Option<&[u64]>,
    heap_scores: &mut [Vec<f32>],
    heap_indices: &mut [Vec<u64>],
    heap_sizes: &mut [usize],
    heap_mins: &mut [f32],
    heap_min_idxs: &mut [usize],
) {
    use std::arch::x86_64::*;

    let n_blocks = n_vectors.div_ceil(BLOCK);
    let m0f = _mm512_set1_epi8(0x0F);
    let quads = n_byte_groups / 4;
    let block_bytes = n_byte_groups * BLOCK;
    let nqm = nq.min(8);

    // `vpshufb` indexes within each 128-bit lane, so the same 16 entries are
    // broadcast to all four. Every query in the batch shares this table —
    // they are all scoring against one index, hence one codebook.
    //
    // XOR by 0x80 is the +128 bias that puts the signed level table into
    // `vpdpbusd`'s unsigned operand range; `zero` cancels it. Hoisted out of
    // the block loop, so it is one instruction for the whole scan.
    let levels = _mm512_xor_si512(
        _mm512_broadcast_i32x4(_mm_loadu_si128(pds[0].levels.as_ptr() as *const __m128i)),
        _mm512_set1_epi8(0x80u8 as i8),
    );

    for b in 0..n_blocks {
        let base_vec = b * BLOCK;
        if !block_has_allowed(mask, base_vec) {
            continue;
        }
        let block_base = b * block_bytes;
        // acc[q][h]: 16 i32 lanes = 16 vectors, halves h = vectors 0-15,
        // 16-31. Seeded with the +128 cancellation rather than zero.
        let mut acc = [[_mm512_setzero_si512(); 2]; 8];
        for (a, pd) in acc.iter_mut().zip(pds.iter()).take(nqm) {
            let z = _mm512_set1_epi32(pd.zero);
            a[0] = z;
            a[1] = z;
        }

        for q4 in 0..quads {
            for h in 0..2 {
                let c = _mm512_loadu_si512(
                    blocked_codes.as_ptr().add(block_base + q4 * 128 + h * 64) as *const __m512i,
                );
                // Shared across the whole batch — this is the whole point.
                let vlo = _mm512_shuffle_epi8(levels, _mm512_and_si512(c, m0f));
                let vhi =
                    _mm512_shuffle_epi8(levels, _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f));
                for qi in 0..nqm {
                    // Four dimensions of query weights, broadcast: every
                    // dword lane holds a different database vector but the
                    // same four byte-groups. `weights` is `Vec<i8>`, so the
                    // 4-byte read is unaligned by construction.
                    let wp = pds[qi].weights.as_ptr().add(q4 * 8);
                    let wlo = _mm512_set1_epi32((wp as *const i32).read_unaligned());
                    let whi = _mm512_set1_epi32((wp.add(4) as *const i32).read_unaligned());
                    acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], vlo, wlo);
                    acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], vhi, whi);
                }
            }
        }

        let end = (base_vec + BLOCK).min(n_vectors);
        for qi in 0..nqm {
            let vs = _mm256_set1_ps(pds[qi].scale);
            let vb = _mm256_set1_ps(pds[qi].bias);
            let cvt = |h: __m256i| _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(h), vs), vb);
            let fa = [
                cvt(_mm512_extracti32x8_epi32(acc[qi][0], 0)),
                cvt(_mm512_extracti32x8_epi32(acc[qi][0], 1)),
                cvt(_mm512_extracti32x8_epi32(acc[qi][1], 0)),
                cvt(_mm512_extracti32x8_epi32(acc[qi][1], 1)),
            ];
            avx2_post_flush_heap_update(
                &fa,
                base_vec,
                end,
                vec_scales.as_ptr().add(base_vec),
                qi,
                k,
                mask,
                heap_scores,
                heap_indices,
                heap_sizes,
                heap_mins,
                heap_min_idxs,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_batch_flush_to_fa(
    accus: [std::arch::x86_64::__m256i; 4],
    v_scale: std::arch::x86_64::__m256,
    fa: &mut [std::arch::x86_64::__m256; 4],
) {
    use std::arch::x86_64::*;
    let a0 = _mm256_sub_epi16(accus[0], _mm256_slli_epi16(accus[1], 8));
    let a1 = accus[1];
    let a2 = _mm256_sub_epi16(accus[2], _mm256_slli_epi16(accus[3], 8));
    let a3 = accus[3];

    let dis0 = _mm256_add_epi16(
        _mm256_permute2x128_si256(a0, a1, 0x21),
        _mm256_blend_epi32(a0, a1, 0xF0),
    );
    let dis1 = _mm256_add_epi16(
        _mm256_permute2x128_si256(a2, a3, 0x21),
        _mm256_blend_epi32(a2, a3, 0xF0),
    );

    let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0)));
    let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0, 1)));
    let f2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1)));
    let f3 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1, 1)));

    fa[0] = _mm256_fmadd_ps(v_scale, f0, fa[0]);
    fa[1] = _mm256_fmadd_ps(v_scale, f1, fa[1]);
    fa[2] = _mm256_fmadd_ps(v_scale, f2, fa[2]);
    fa[3] = _mm256_fmadd_ps(v_scale, f3, fa[3]);
}

/// Final epilogue: takes per-query f32 accumulators `fa` (already containing
/// `bias + Σ scale*partial`), applies the per-vector `vec_scales` multiplier,
/// then runs the in-register-threshold-prune + heap-update logic for one block.
/// Used by both the AVX2 and AVX-512BW kernels after their flush loops.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_post_flush_heap_update(
    fa: &[std::arch::x86_64::__m256; 4],
    base_vec: usize,
    end: usize,
    vec_scales_ptr: *const f32,
    qi: usize,
    k: usize,
    mask: Option<&[u64]>,
    heap_scores: &mut [Vec<f32>],
    heap_indices: &mut [Vec<u64>],
    heap_sizes: &mut [usize],
    heap_mins: &mut [f32],
    heap_min_idxs: &mut [usize],
) {
    use std::arch::x86_64::*;

    let end_lane = end - base_vec;
    let (s0, s1, s2, s3) = if end_lane == BLOCK {
        (
            _mm256_mul_ps(fa[0], _mm256_loadu_ps(vec_scales_ptr)),
            _mm256_mul_ps(fa[1], _mm256_loadu_ps(vec_scales_ptr.add(8))),
            _mm256_mul_ps(fa[2], _mm256_loadu_ps(vec_scales_ptr.add(16))),
            _mm256_mul_ps(fa[3], _mm256_loadu_ps(vec_scales_ptr.add(24))),
        )
    } else {
        (fa[0], fa[1], fa[2], fa[3])
    };

    let hs = &mut heap_scores[qi];
    let hi = &mut heap_indices[qi];
    let sz = &mut heap_sizes[qi];
    let hmin = &mut heap_mins[qi];
    let hmi = &mut heap_min_idxs[qi];

    if *sz >= k && end_lane == BLOCK {
        let thr = _mm256_set1_ps(*hmin);
        let m0 = _mm256_movemask_ps(_mm256_cmp_ps(s0, thr, _CMP_GT_OQ)) as u32;
        let m1 = _mm256_movemask_ps(_mm256_cmp_ps(s1, thr, _CMP_GT_OQ)) as u32;
        let m2 = _mm256_movemask_ps(_mm256_cmp_ps(s2, thr, _CMP_GT_OQ)) as u32;
        let m3 = _mm256_movemask_ps(_mm256_cmp_ps(s3, thr, _CMP_GT_OQ)) as u32;
        if (m0 | m1 | m2 | m3) == 0 {
            return;
        }
        let mut block_out = [0.0f32; BLOCK];
        let bp = block_out.as_mut_ptr();
        if m0 != 0 { _mm256_storeu_ps(bp, s0); }
        if m1 != 0 { _mm256_storeu_ps(bp.add(8), s1); }
        if m2 != 0 { _mm256_storeu_ps(bp.add(16), s2); }
        if m3 != 0 { _mm256_storeu_ps(bp.add(24), s3); }

        for (chunk, &mask0) in [m0, m1, m2, m3].iter().enumerate() {
            let mut m = mask0;
            while m != 0 {
                let bit = m.trailing_zeros() as usize;
                m &= m - 1;
                let lane = chunk * 8 + bit;
                if let Some(am) = mask {
                    if !mask_allows(am, base_vec + lane) { continue; }
                }
                let score = block_out[lane];
                if score > *hmin {
                    hs[*hmi] = score;
                    hi[*hmi] = (base_vec + lane) as u64;
                    let (m, mi) = rescan_min(hs, hi, k);
                    *hmin = m;
                    *hmi = mi;
                }
            }
        }
        return;
    }

    let mut block_out = [0.0f32; BLOCK];
    let bp = block_out.as_mut_ptr();
    _mm256_storeu_ps(bp, s0);
    _mm256_storeu_ps(bp.add(8), s1);
    _mm256_storeu_ps(bp.add(16), s2);
    _mm256_storeu_ps(bp.add(24), s3);

    if end_lane != BLOCK {
        for lane in 0..end_lane {
            block_out[lane] *= *vec_scales_ptr.add(lane);
        }
        for lane in end_lane..BLOCK {
            block_out[lane] = f32::NEG_INFINITY;
        }
    }

    if *sz < k {
        for lane in 0..end_lane {
            if let Some(am) = mask {
                if !mask_allows(am, base_vec + lane) { continue; }
            }
            let score = block_out[lane];
            if *sz < k {
                hs[*sz] = score;
                hi[*sz] = (base_vec + lane) as u64;
                *sz += 1;
                if *sz == k {
                    let (m, mi) = rescan_min(hs, hi, k);
                    *hmin = m;
                    *hmi = mi;
                }
            } else if score > *hmin {
                hs[*hmi] = score;
                hi[*hmi] = (base_vec + lane) as u64;
                let (m, mi) = rescan_min(hs, hi, k);
                *hmin = m;
                *hmi = mi;
            }
        }
    } else {
        let v_hmin = _mm256_set1_ps(*hmin);
        for chunk in 0..4 {
            let chunk_start = chunk * 8;
            if chunk_start >= end_lane { break; }
            let scores_v = _mm256_loadu_ps(block_out.as_ptr().add(chunk_start));
            let cmp = _mm256_cmp_ps(scores_v, v_hmin, _CMP_GT_OQ);
            if _mm256_movemask_ps(cmp) == 0 { continue; }

            let chunk_end = (chunk_start + 8).min(end_lane);
            for lane in chunk_start..chunk_end {
                if let Some(am) = mask {
                    if !mask_allows(am, base_vec + lane) { continue; }
                }
                let score = block_out[lane];
                if score > *hmin {
                    hs[*hmi] = score;
                    hi[*hmi] = (base_vec + lane) as u64;
                    let (m, mi) = rescan_min(hs, hi, k);
                    *hmin = m;
                    *hmi = mi;
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
    vec_scales: &[f32],
    base_vec: usize,
    n_vectors: usize,
    out: &mut [[f32; BLOCK]; 4],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    // Float accumulators on stack, seeded with each query's decode bias so
    // flushes only need to add `v_scale * acc`. Final values are calibrated
    // per-vector scores (before norm multiplication).
    let mut fa: [[float32x4_t; 8]; 4] = [
        [vdupq_n_f32(biases[0]); 8],
        [vdupq_n_f32(biases[1]); 8],
        [vdupq_n_f32(biases[2]); 8],
        [vdupq_n_f32(biases[3]); 8],
    ];

    let codes_base = blocked_codes.as_ptr().add(block_offset);

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);

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

        // Flush each query (bias applied once below, after all batches)
        for q in 0..4 {
            let v_scale = vdupq_n_f32(scales[q]);
            for i in 0..4 {
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc[q][i])));
                let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc[q][i])));
                fa[q][i * 2] = vfmaq_f32(fa[q][i * 2], v_scale, lo);
                fa[q][i * 2 + 1] = vfmaq_f32(fa[q][i * 2 + 1], v_scale, hi);
            }
        }
    }

    // Write with vec_scales; padding lanes get NEG_INFINITY so callers can
    // take a whole-block max without seeing garbage.
    let end = (base_vec + BLOCK).min(n_vectors);
    let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);

    for q in 0..4 {
        let op = out[q].as_mut_ptr();
        if end - base_vec == BLOCK {
            for i in 0..8 {
                let n = vld1q_f32(vec_scales_ptr.add(i * 4));
                vst1q_f32(op.add(i * 4), vmulq_f32(fa[q][i], n));
            }
        } else {
            let mut buf = [0.0f32; BLOCK];
            for i in 0..8 {
                vst1q_f32(buf.as_mut_ptr().add(i * 4), fa[q][i]);
            }
            for lane in 0..BLOCK {
                *op.add(lane) = if lane < end - base_vec {
                    buf[lane] * *vec_scales_ptr.add(lane)
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
    }
}

/// One `SDOT`: `acc.4s += sum of four s8 x s8 products per 32-bit lane`.
///
/// Written as inline asm because `vdotq_s32` is still unstable on stable
/// Rust (rust-lang/rust#117224), and `.arch_extension dotprod` because the
/// assembler otherwise rejects the mnemonic on a baseline-v8 target. The
/// caller is `#[target_feature(enable = "dotprod")]`, and reaching it at all
/// requires `use_vector_major()` to have detected the feature at runtime.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut o = acc;
    std::arch::asm!(
        ".arch_extension dotprod",
        "sdot {o:v}.4s, {a:v}.16b, {b:v}.16b",
        o = inout(vreg) o,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    o
}

/// Permute-dot scan of one 32-vector block over the vector-major layout,
/// for `NQ` queries at once (4-bit codes).
///
/// The ARM half of P18. `SDOT` reduces four s8 x s8 products into one 32-bit
/// lane, exactly as `vpdpbusd` does on x86 — and it is signed by signed, so
/// unlike x86 it needs no `+128` bias on the level table and no `zero`
/// correction. The integers accumulated are identical on both arches.
///
/// The vector-major unit puts each vector's four consecutive byte-groups in
/// its own aligned 4-byte group, so one 16-byte register holds four vectors
/// and one `SDOT` advances all four by four dimensions. Register `i` covers
/// lanes `i*4 .. i*4+3`, matching the float accumulator order the classic
/// NEON kernel writes, so the epilogue and every caller are unchanged.
///
/// No flush: the widest possible sum over 768 dimensions is `768 * 127 *
/// 127` ~ 1.2e7, well inside i32, so the `FLUSH_EVERY` cadence and the u8
/// pre-add that capped the LUT at 127 both leave this path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_block_permute_dot_neon<const NQ: usize>(
    blocked_codes: &[u8],
    pds: &[&QueryPermuteDot; NQ],
    block_offset: usize,
    n_byte_groups: usize,
    vec_scales: &[f32],
    base_vec: usize,
    n_vectors: usize,
    out: &mut [[f32; BLOCK]; NQ],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    // One shared nibble -> level permute for every query and every
    // dimension, live in a register for the whole block.
    let levels = vld1q_s8(pds[0].levels.as_ptr());
    let quads = n_byte_groups / 4;
    let codes_base = blocked_codes.as_ptr().add(block_offset);

    // acc[q][i]: four vectors per register, i covering lanes i*4 .. i*4+3.
    let mut acc = [[vdupq_n_s32(0); 8]; NQ];

    for q4 in 0..quads {
        // Four dimensions of query weights duplicated across all four 32-bit
        // lanes: each lane is a different database vector, but every vector
        // needs the same four byte-groups' weights. `weights` is `Vec<i8>`,
        // so the 4-byte read is unaligned by construction.
        let mut w = [[vdupq_n_s8(0); 2]; NQ];
        for (wq, pd) in w.iter_mut().zip(pds.iter()) {
            let wp = pd.weights.as_ptr().add(q4 * 8);
            wq[0] = vreinterpretq_s8_s32(vdupq_n_s32((wp as *const i32).read_unaligned()));
            wq[1] =
                vreinterpretq_s8_s32(vdupq_n_s32((wp.add(4) as *const i32).read_unaligned()));
        }
        for i in 0..8 {
            let c = vld1q_u8(codes_base.add(q4 * 128 + i * 16));
            // Shared across the whole batch — this is the whole point.
            let vlo = vqtbl1q_s8(levels, vandq_u8(c, mask));
            let vhi = vqtbl1q_s8(levels, vshrq_n_u8(c, 4));
            for q in 0..NQ {
                acc[q][i] = sdot(acc[q][i], vlo, w[q][0]);
                acc[q][i] = sdot(acc[q][i], vhi, w[q][1]);
            }
        }
    }

    // Padding lanes get NEG_INFINITY so callers can take a whole-block max
    // without seeing garbage — same contract as the classic NEON kernel.
    let end = (base_vec + BLOCK).min(n_vectors);
    let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);
    for q in 0..NQ {
        let vs = vdupq_n_f32(pds[q].scale);
        let vb = vdupq_n_f32(pds[q].bias);
        let op = out[q].as_mut_ptr();
        if end - base_vec == BLOCK {
            for i in 0..8 {
                let f = vfmaq_f32(vb, vcvtq_f32_s32(acc[q][i]), vs);
                let n = vld1q_f32(vec_scales_ptr.add(i * 4));
                vst1q_f32(op.add(i * 4), vmulq_f32(f, n));
            }
        } else {
            let mut buf = [0.0f32; BLOCK];
            for i in 0..8 {
                let f = vfmaq_f32(vb, vcvtq_f32_s32(acc[q][i]), vs);
                vst1q_f32(buf.as_mut_ptr().add(i * 4), f);
            }
            for lane in 0..BLOCK {
                *op.add(lane) = if lane < end - base_vec {
                    buf[lane] * *vec_scales_ptr.add(lane)
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
    }
}

/// Fold one scored block into a query's running top-k — the ARM analogue
/// of the x86 post-flush heap update. Insertion order is lane-ascending
/// within block-ascending visits, so together with [`rescan_min`]'s
/// evict-largest-index tie-break the results are identical to a flat
/// index-order scan of a fully materialized score row.
///
/// `block_scores` must hold NEG_INFINITY in padding lanes (the kernels
/// guarantee this) so the whole-block max prune can read all 32 lanes.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_block_topk_update(
    block_scores: &[f32; BLOCK],
    base_vec: usize,
    end_lane: usize,
    mask: Option<&[u64]>,
    k: usize,
    hs: &mut [f32],
    hi: &mut [u64],
    sz: &mut usize,
    hmin: &mut f32,
    hmi: &mut usize,
) {
    use std::arch::aarch64::*;

    if *sz >= k {
        // Whole-block prune: skip the lane loop when nothing can beat the
        // current heap minimum (the overwhelmingly common case once the
        // heap is warm).
        let p = block_scores.as_ptr();
        let mut m = vld1q_f32(p);
        for i in 1..8 {
            m = vmaxq_f32(m, vld1q_f32(p.add(i * 4)));
        }
        if vmaxvq_f32(m) <= *hmin {
            return;
        }
    }
    for (lane, &s) in block_scores.iter().enumerate().take(end_lane) {
        if let Some(am) = mask {
            if !mask_allows(am, base_vec + lane) {
                continue;
            }
        }
        if *sz < k {
            hs[*sz] = s;
            hi[*sz] = (base_vec + lane) as u64;
            *sz += 1;
            if *sz == k {
                let (m, mi) = rescan_min(hs, hi, k);
                *hmin = m;
                *hmi = mi;
            }
        } else if s > *hmin {
            hs[*hmi] = s;
            hi[*hmi] = (base_vec + lane) as u64;
            let (m, mi) = rescan_min(hs, hi, k);
            *hmin = m;
            *hmi = mi;
        }
    }
}

/// Per-query nibble LUTs for NEON scoring (works for 2-bit and 4-bit).

pub(crate) struct QueryNeonLut {
    pub(crate) uint8_luts: Vec<u8>,  // n_byte_groups * 32 bytes: [hi_16 | lo_16] per group
    /// The same table reordered for `vpermb` (see [`split_lut_for_vnni`]).
    /// Empty unless this process and geometry use the vector-major layout;
    /// built once per query rather than per tile.
    #[cfg(target_arch = "x86_64")]
    pub(crate) split: Vec<u8>,
    /// Present instead of `split` when this geometry can use the permute-dot
    /// kernel (see [`QueryPermuteDot`]); the two are mutually exclusive.
    pub(crate) pd: Option<QueryPermuteDot>,
    pub(crate) scale: f32,
    /// Total decode bias = sum of per-sub-table mins. Added once to
    /// the accumulator at the end of scoring, not per lookup.
    pub(crate) bias: f32,
}


/// Rearrange a query's LUT for the `vpermb` kernel.
///
/// `vpermb`'s index is 6 bits, so `(j << 4) | code` selects from a 64-byte
/// table holding four consecutive 16-entry sub-tables. Per group of four
/// byte-groups this emits 64 bytes of the four *lo*-nibble sub-tables
/// followed by 64 bytes of the four *hi* — the same values as
/// `uint8_luts`, reordered so one permute serves four byte-groups at once.
///
/// Same total size as the source, built once per query.
#[cfg(target_arch = "x86_64")]
pub(crate) fn split_lut_for_vnni(uint8_luts: &[u8], n_byte_groups: usize) -> Vec<u8> {
    debug_assert_eq!(uint8_luts.len(), n_byte_groups * 32);
    debug_assert_eq!(n_byte_groups % 4, 0);
    let mut out = vec![0u8; n_byte_groups * 32];
    for g0 in (0..n_byte_groups).step_by(4) {
        let c = (g0 / 4) * 128;
        for j in 0..4 {
            let src = (g0 + j) * 32;
            // `uint8_luts` is [hi_16 | lo_16] per group.
            out[c + j * 16..c + j * 16 + 16].copy_from_slice(&uint8_luts[src + 16..src + 32]);
            out[c + 64 + j * 16..c + 64 + j * 16 + 16]
                .copy_from_slice(&uint8_luts[src..src + 16]);
        }
    }
    out
}

/// Per-query state for the permute-dot kernel.
///
/// The codebook is shared across every dimension, so nibble -> level is a
/// *fixed* 16-entry permute: query-independent, dimension-independent, and
/// register-resident for the whole scan. Applying it to nibbles that have to
/// be unpacked anyway turns the score into a plain integer dot product,
///
///   score = Σ_d q[d] * C[code[d]]
///
/// which is what `vpdpbusd` computes natively — with the full Lloyd-Max
/// codebook intact. That last part is what separates this from the uniform
/// codebook of P15/P17, which bought the same dot product at ~2 recall
/// points. Here recall goes *up*, because the query and the codebook are
/// quantized separately to 8 bits and their products accumulate exactly in
/// i32, rather than each (dimension, level) product being pre-rounded into a
/// 7-bit table entry.
///
/// Only built for 4-bit codes: at 2 bits a nibble spans two dimensions, so
/// its value depends on two query weights and the map stops being shared.
///
/// Shared by both arches. `vpdpbusd` and `SDOT` reduce four bytes into one
/// 32-bit lane in the same way, so the layout and the arithmetic are common.
/// The only difference is that `vpdpbusd` multiplies *unsigned* by signed and
/// needs the level table biased into u8 range, which `zero` then cancels
/// exactly; `SDOT` is signed by signed and takes the table as-is. Both
/// therefore accumulate the same integers and produce bit-identical scores.
///
/// See P18 in `benchmarks/hillclimb/LOG_search.md`.
pub(crate) struct QueryPermuteDot {
    /// The codebook as int8. x86 biases this by +128 in-register to feed
    /// `vpdpbusd`'s unsigned operand; `SDOT` reads it directly.
    pub(crate) levels: [i8; 16],
    /// One int8 weight per dimension, grouped to match the 4-byte reduction:
    /// bytes `q4*8 .. q4*8+4` hold the *low*-nibble dimensions of byte-groups
    /// `4*q4 .. 4*q4+4`, and bytes `q4*8+4 .. q4*8+8` the high-nibble ones —
    /// the same four byte-groups one instruction sums into a single vector's
    /// 32-bit lane.
    ///
    /// Note the packing order: [`crate::pack`] writes code `c` of a group at
    /// shift `(codes_per_byte - 1 - c) * bits`, so at 4 bits the *high*
    /// nibble carries the even dimension `2g` and the low nibble the odd
    /// `2g+1` — the reverse of the reading the field names suggest.
    pub(crate) weights: Vec<i8>,
    /// Accumulator seed for kernels that bias `levels` into unsigned range:
    /// `-128 * Σ_d w[d]`, which cancels that bias exactly. Seeding the
    /// accumulator rather than correcting afterwards keeps the cancellation
    /// in exact integers and costs nothing — the large offset never reaches
    /// the f32 conversion, where it would cost precision once the raw sum
    /// passed 2^24. Unused by `SDOT`, which needs no bias.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) zero: i32,
    pub(crate) scale: f32,
    pub(crate) bias: f32,
}

/// Quantize one query row and the shared codebook to int8 for
/// [`QueryPermuteDot`].
fn build_permute_dot(q_rot_row: &[f32], centroids: &[f32], dim: usize) -> QueryPermuteDot {
    debug_assert_eq!(centroids.len(), 16);
    let n_byte_groups = dim / 2;

    // Rounding a scale to int8 is 16 values of work, so it rides along with
    // the per-query build rather than being cached on the index.
    let cmax = centroids.iter().fold(0.0f32, |m, &c| m.max(c.abs()));
    let cs = if cmax > 0.0 { cmax / 127.0 } else { 1.0 };
    let mut levels = [0i8; 16];
    for (l, &c) in levels.iter_mut().zip(centroids.iter()) {
        *l = (c / cs).round().clamp(-127.0, 127.0) as i8;
    }

    // One scale for the whole query row. Like the LUT path this is linear in
    // the query magnitude, so scaling a query leaves the integer sums — and
    // hence the ranking — untouched, down to where `1.0 / qs` stops being
    // representable.
    let qmax = q_rot_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let qs = qmax / 127.0;
    let (qs, inv_qs) = if qs >= f32::MIN_POSITIVE { (qs, 1.0 / qs) } else { (1.0, 1.0) };

    let mut weights = vec![0i8; n_byte_groups * 2];
    let mut wsum: i32 = 0;
    for g in 0..n_byte_groups {
        let (q4, j) = (g / 4, g % 4);
        // Low nibble = odd dimension, high nibble = even. See `weights`.
        let lo = (q_rot_row[2 * g + 1] * inv_qs).round().clamp(-127.0, 127.0) as i8;
        let hi = (q_rot_row[2 * g] * inv_qs).round().clamp(-127.0, 127.0) as i8;
        weights[q4 * 8 + j] = lo;
        weights[q4 * 8 + 4 + j] = hi;
        wsum += lo as i32 + hi as i32;
    }

    QueryPermuteDot {
        levels,
        weights,
        zero: -128 * wsum,
        scale: cs * qs,
        bias: 0.0,
    }
}

/// Build nibble LUTs for NEON/AVX2 scoring from a flat query rotation row.
///
/// Uses FAISS-style per-sub-table quantization: each 16-entry nibble
/// LUT subtracts its own min before u8 rounding, with a single
/// shared `scale = max_span / max_lut`. This avoids the systematic
/// rounding bias that a single global min produces when sub-tables
/// have different value ranges (which they do for asymmetric-sign
/// products of `q_rot[coord] * centroid[code]`).
pub(crate) fn build_query_neon_lut_from_slice(
    q_rot_row: &[f32],
    centroids: &[f32],
    bits: usize,
    dim: usize,
) -> QueryNeonLut {
    let codes_per_byte = 8 / bits;
    let codes_per_nibble = codes_per_byte / 2;
    let n_byte_groups = dim / codes_per_byte;
    let code_mask = (1u16 << bits) - 1;
    let n_subs = n_byte_groups * 2; // lo + hi nibble sub-table per byte group

    let mut uint8_luts = vec![0u8; n_byte_groups * 32];
    let mut float_vals = vec![0.0f32; n_byte_groups * 32];
    let mut mins = vec![0.0f32; n_subs];
    let mut max_span = 0.0f32;
    let mut sum_spans = 0.0f32;
    let mut bias = 0.0f32;

    for g in 0..n_byte_groups {
        let dim_start = g * codes_per_byte;

        // lo nibble sub-table (16 entries)
        let mut lo_min = f32::MAX;
        let mut lo_max = f32::MIN;
        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot_row[dim_start + c] * centroids[code as usize];
            }
            float_vals[g * 32 + nibble_val as usize] = s;
            if s < lo_min { lo_min = s; }
            if s > lo_max { lo_max = s; }
        }

        // hi nibble sub-table (16 entries)
        let mut hi_min = f32::MAX;
        let mut hi_max = f32::MIN;
        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot_row[dim_start + codes_per_nibble + c] * centroids[code as usize];
            }
            float_vals[g * 32 + 16 + nibble_val as usize] = s;
            if s < hi_min { hi_min = s; }
            if s > hi_max { hi_max = s; }
        }

        mins[g * 2] = lo_min;
        mins[g * 2 + 1] = hi_min;
        bias += lo_min + hi_min;

        let lo_span = lo_max - lo_min;
        let hi_span = hi_max - hi_min;
        if lo_span > max_span { max_span = lo_span; }
        if hi_span > max_span { max_span = hi_span; }
        sum_spans += lo_span + hi_span;
    }

    // Per-query LUT cap. Both kernels flush their integer accumulators every
    // `FLUSH_EVERY = 256` byte-groups, so the per-flush u16 sum constraint is
    // `FLUSH_EVERY * max_lut <= 65535` ⇒ max_lut ≤ 255. That is not the
    // binding constraint on ARM:
    //
    // ARM: NEON adds the two nibble lookups with `vaddq_u8(lo, hi)` before
    // widening, so the *pair* sum must fit a u8: `2 * max_lut <= 255` ⇒
    // max_lut ≤ 127. That u8 pre-add, not the flush, is the binding
    // constraint, and 127 is exactly its ceiling (128 + 128 = 256 wraps).
    //
    // x86: AVX2 / AVX-512 accumulate u8 lookups directly into i16 lanes
    // via FAISS even/odd interleave + SUB-trick. With periodic flush, the
    // per-half u16 sum is bounded by `FLUSH_EVERY * max_lut`, allowing
    // max_lut up to ~255. We share 127 with ARM so codes encoded against
    // an x86-built index round identically to an ARM-built index — keeps
    // the kernel arches numerically equivalent. Raising x86 alone would
    // break that equivalence; raising ARM needs the u8 pre-add replaced by
    // two widening adds (see #332).
    let _ = sum_spans; // retained for the FAISS-style data-dependent path; not
                       // used now that both kernels flush.
    let max_lut: f32 = 127.0;

    // `float_vals`, `mins` and `max_span` are all linear in the query
    // magnitude, so the u8 LUT is magnitude-free and `scale` alone carries
    // it: multiplying a query by a positive constant then leaves the integer
    // sums — and hence the ranking — untouched. An *absolute* floor here
    // (previously `max_span > 1e-10`) broke that invariant by forcing
    // `scale = 1.0` for small queries, which rounds every LUT entry to 0 and
    // destroys the ranking (#335). The only real limit is representability:
    // `1.0 / scale` must stay finite, which holds until `scale` itself
    // underflows to a subnormal — far below the point where the f32 score
    // (an inner product, so it legitimately scales with the query) still has
    // usable precision.
    let scale = if max_span > 0.0 { max_span / max_lut } else { 1.0 };
    let (scale, inv_scale) = if scale >= f32::MIN_POSITIVE {
        (scale, 1.0 / scale)
    } else {
        (1.0, 1.0)
    };

    for g in 0..n_byte_groups {
        let lo_min = mins[g * 2];
        let hi_min = mins[g * 2 + 1];
        for i in 0..16 {
            let j_lo = g * 32 + i;
            let j_hi = g * 32 + 16 + i;
            uint8_luts[j_lo] =
                ((float_vals[j_lo] - lo_min) * inv_scale).round().clamp(0.0, max_lut) as u8;
            uint8_luts[j_hi] =
                ((float_vals[j_hi] - hi_min) * inv_scale).round().clamp(0.0, max_lut) as u8;
        }
    }

    // On the vector-major layout, 4-bit codes score through the permute-dot
    // kernel and 2-bit codes through the arch's classic one; the two need
    // different per-query tables and only one of them is ever built.
    let vm = crate::pack::vector_major_for(bits, n_byte_groups);
    let pd = if vm && bits == 4 {
        Some(build_permute_dot(q_rot_row, centroids, dim))
    } else {
        None
    };

    QueryNeonLut {
        #[cfg(target_arch = "x86_64")]
        split: if vm && pd.is_none() {
            split_lut_for_vnni(&uint8_luts, n_byte_groups)
        } else {
            Vec::new()
        },
        pd,
        uint8_luts,
        scale,
        bias,
    }
}

/// Slot-allowlist bitmask: packed little-endian, bit `i` set iff slot `i` is
/// allowed. Caller guarantees `len * 64 >= n_vectors`. Bits at index `>=
/// n_vectors` are ignored.
#[inline(always)]
pub(crate) fn mask_allows(mask: &[u64], slot: usize) -> bool {
    // Safety: caller validates mask length against n_vectors before reaching
    // any kernel; we never query past it in scoring loops.
    (mask[slot >> 6] >> (slot & 63)) & 1 != 0
}

/// Block-level early-exit predicate: true iff at least one slot in the
/// 32-vector block starting at `base_vec` is allowed by `mask`. Returns
/// true unconditionally when no mask is present, so the scoring kernel
/// only short-circuits when a mask is supplied.
///
/// `base_vec` is always a multiple of [`BLOCK`] (= 32) and the slot bitmap
/// is packed at 64 slots per `u64` word, so the relevant 32-bit window is
/// either the low or high half of a single word.
#[inline(always)]
pub(crate) fn block_has_allowed(mask: Option<&[u64]>, base_vec: usize) -> bool {
    match mask {
        None => true,
        Some(m) => {
            let word = m[base_vec >> 6];
            let bit_offset = base_vec & 63;
            let allowed = ((word >> bit_offset) & 0xFFFF_FFFF) != 0;
            #[cfg(feature = "mask-skip-counter")]
            if !allowed {
                BLOCKS_SKIPPED_BY_MASK.fetch_add(1, Ordering::Relaxed);
            }
            allowed
        }
    }
}

/// Blocks per rayon range for the single-query block-parallel paths.
///
/// Rounded up to an even count so every range starts on a 64-slot
/// boundary: that is exactly one `u64` mask word, which is what lets a
/// masked search hand each range a word-aligned sub-slice of the bitmap
/// and keep indexing it range-relative like the codes and scales.
#[inline]
pub(crate) fn block_range_stride(n_blocks: usize, n_threads: usize) -> usize {
    (n_blocks.div_ceil(n_threads)).max(64).next_multiple_of(2)
}

/// Pair-level early-exit predicate for the AVX-512BW kernel which scores
/// two adjacent 32-vector blocks per zmm iteration. The 64-vector pair
/// aligns to a single `u64` word, so a zero word means neither block has
/// allowed slots and the entire SIMD pair can be skipped.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn block_pair_has_allowed(mask: Option<&[u64]>, base_vec_pair: usize) -> bool {
    match mask {
        None => true,
        Some(m) => {
            let allowed = m[base_vec_pair >> 6] != 0;
            // A pair-level skip short-circuits two 32-vector blocks.
            #[cfg(feature = "mask-skip-counter")]
            if !allowed {
                BLOCKS_SKIPPED_BY_MASK.fetch_add(2, Ordering::Relaxed);
            }
            allowed
        }
    }
}

/// Per-query scalar scoring writing into caller-provided heap arrays.
/// Used by the non-x86_64 / non-aarch64 scalar fallback at the bottom
/// of `search`, AND as the x86_64 fallback inside the SIMD-dispatch
/// `unsafe` block when neither AVX-512 BW nor AVX2 is detected at
/// runtime (e.g. running a turbovec binary built without the cargo
/// config's `target-cpu=x86-64-v3` on a pre-Haswell CPU, or under a
/// VM / emulator that doesn't expose AVX2 to userspace). Without this
/// fallback, pre-AVX2 x86_64 silently returned empty top-k results
/// instead of falling back to a slower-but-correct kernel.
///
/// Not compiled on aarch64, where the NEON kernel is always available and
/// this scalar path is never reached (it would warn as dead code).
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn score_query_into_heap(
    qlut_uint8: &[u8],
    qlut_scale: f32,
    qlut_bias: f32,
    blocked_codes: &[u8],
    vec_scales: &[f32],
    bits: usize,
    n_byte_groups: usize,
    n_vectors: usize,
    n_blocks: usize,
    mask: Option<&[u64]>,
    k: usize,
    heap_s: &mut [f32],
    // u64, not u32: the on-disk format's count field is u64 (format v4),
    // so vector indices can legitimately exceed u32::MAX; a u32 heap
    // slot would silently truncate them.
    heap_i: &mut [u64],
    heap_sz: &mut usize,
    heap_min: &mut f32,
    heap_mi: &mut usize,
) {
    for b in 0..n_blocks {
        let base_vec = b * BLOCK;
        if !block_has_allowed(mask, base_vec) {
            continue;
        }
        let block_offset = b * n_byte_groups * BLOCK;
        for lane in 0..BLOCK {
            let vi = base_vec + lane;
            if vi >= n_vectors {
                break;
            }
            if let Some(m) = mask {
                if !mask_allows(m, vi) {
                    continue;
                }
            }
            let mut score = qlut_bias;
            for g in 0..n_byte_groups {
                // x86 has two possible native layouts (perm0-interleaved
                // nibble planes, or vector-major for the VNNI kernel), so go
                // through the shared accessor rather than assuming either;
                // every other target stores the sequential layout directly.
                // Reading the wrong one here would silently mis-score
                // (issue #106 is the original perm0 instance of this).
                let byte_val =
                    crate::pack::read_code(blocked_codes, bits, n_byte_groups, b, g, lane) as usize;
                let hi = byte_val >> 4;
                let lo = byte_val & 0x0F;
                score += qlut_scale * qlut_uint8[g * 32 + hi] as f32;
                score += qlut_scale * qlut_uint8[g * 32 + 16 + lo] as f32;
            }
            score *= vec_scales[vi];
            if *heap_sz < k {
                heap_s[*heap_sz] = score;
                heap_i[*heap_sz] = vi as u64;
                *heap_sz += 1;
                if *heap_sz == k {
                    let (m, mi) = rescan_min(heap_s, heap_i, k);
                    *heap_min = m;
                    *heap_mi = mi;
                }
            } else if score > *heap_min {
                heap_s[*heap_mi] = score;
                heap_i[*heap_mi] = vi as u64;
                let (m, mi) = rescan_min(heap_s, heap_i, k);
                *heap_min = m;
                *heap_mi = mi;
            }
        }
    }
}

/// Apply TQ+ per-coord (shift, scale) calibration to a batch of rotated
/// queries. Returns the calibrated queries and a per-query bias correction
/// (the search kernel folds this into the per-query bias). When the index
/// has no calibration (v2 file, lazy index with no add), returns the
/// queries unchanged and zero bias corrections.
fn calibrate_queries(
    q_rot: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    nq: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    if tqplus_shift.is_empty() {
        debug_assert!(tqplus_scale.is_empty());
        return (q_rot.to_vec(), vec![0.0f32; nq]);
    }
    debug_assert_eq!(tqplus_shift.len(), dim);
    debug_assert_eq!(tqplus_scale.len(), dim);

    let mut q_calib = vec![0.0f32; nq * dim];
    let mut bias_corrs = vec![0.0f32; nq];

    q_calib
        .par_chunks_mut(dim)
        .zip(bias_corrs.par_iter_mut())
        .enumerate()
        .for_each(|(qi, (calib_row, bias))| {
            let q_row = &q_rot[qi * dim..(qi + 1) * dim];
            let mut bc = 0.0f64;
            for d in 0..dim {
                calib_row[d] = q_row[d] / tqplus_scale[d];
                bc -= (q_row[d] as f64) * (tqplus_shift[d] as f64);
            }
            *bias = bc as f32;
        });

    (q_calib, bias_corrs)
}

/// Full search: rotation + LUT build + scoring + heap top-k.
///
/// `mask`: optional packed bitset over slots (one bit per vector,
/// little-endian within each u64). When `Some`, only slots with their bit set
/// contribute to the top-k. The returned per-query result count is
/// `min(k, popcount(mask))`.
///
/// Returns (scores_flat, indices_flat) each of length nq * effective_k.
///
/// Crate-internal (soundness-critical). The unsafe SIMD kernels index
/// `blocked_codes` using the caller-supplied `n_vectors`/`n_blocks` scalars
/// with no consistency check, so passing a `blocked_codes` buffer that does
/// not match those scalars causes an out-of-bounds read (silent info
/// disclosure) or a SIGBUS — undefined behaviour from otherwise-safe code.
/// Every field here is established by [`TurboQuantIndex::search_with_mask`]
/// from an index whose parts were validated at construction
/// ([`from_parts`](crate::TurboQuantIndex::from_parts)); it is not exposed
/// publicly for that reason.
pub(crate) fn search(
    queries: &[f32],    // (nq, dim) row-major
    nq: usize,
    rotation: &Rotation,
    blocked_codes: &[u8],
    centroids: &[f32],
    vec_scales: &[f32],
    tqplus_shift: &[f32],     // empty for v2 indexes (identity calibration)
    tqplus_scale: &[f32],     // empty for v2 indexes (identity calibration)
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
    k: usize,
    mask: Option<&[u64]>,
) -> (Vec<f32>, Vec<i64>) {
    let n_allowed = match mask {
        Some(m) => m.iter().map(|w| w.count_ones() as usize).sum::<usize>(),
        None => n_vectors,
    };
    let k = k.min(n_allowed);
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    let n_byte_groups = dim / (8 / bits);

    // Rotate each query row in place with the same deterministic
    // block-Hadamard transform the encode path applies to the database, so
    // query and database vectors live in the same rotated space by
    // construction (one shared rotation, no GEMM, no BLAS). Reduction-free
    // per row, so the result does not depend on the thread count.
    let mut q_rot = queries.to_vec();
    q_rot
        .par_chunks_mut(dim)
        .for_each_init(|| vec![0.0f32; dim], |scratch, row| {
            rotation.apply_with_scratch(row, scratch)
        });

    // TQ+ per-coord (shift, scale) was applied to the database at encode
    // time. At search time we apply the inverse to the query:
    //   q_calibrated[d] = q_rot[d] / scale_tq[d]
    //   bias_corr_q     = - sum_d q_rot[d] * shift[d]
    // The LUT build then runs against q_calibrated; bias_corr_q is folded
    // into the per-query bias the kernel adds to every score. The SIMD
    // kernel itself is unchanged.
    let (q_for_lut, bias_corrs) =
        calibrate_queries(&q_rot, tqplus_shift, tqplus_scale, nq, dim);

    // Build LUTs in parallel; fold the TQ+ bias correction into each lut's
    // bias so the kernel doesn't need to know TQ+ exists.
    let query_luts: Vec<QueryNeonLut> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let row = &q_for_lut[qi * dim..(qi + 1) * dim];
            let mut lut = build_query_neon_lut_from_slice(row, centroids, bits, dim);
            lut.bias += bias_corrs[qi];
            if let Some(pd) = lut.pd.as_mut() {
                // The permute-dot kernel carries its own scale/bias, so the
                // TQ+ correction has to land there too.
                pd.bias += bias_corrs[qi];
            }
            lut
        })
        .collect();

    // Platform-specific scoring + top-k
    // Single-query fast path (aarch64) — mirror of the x86 version: one
    // query on a large index partitions the block range across pool
    // workers; each range scores blocks with the single-query NEON
    // kernel straight into a local top-k (no full scores row), then
    // ranges merge deterministically.
    //
    // A mask rides along by slicing the bitmap at the range's first
    // word: `blocks_per_range` is rounded to an even number of 32-vector
    // blocks so every range starts on a 64-slot boundary, which is
    // exactly one `u64` word. The slice is then indexed range-relative
    // like the codes and scales.
    /// One rayon range's worth of blocks, scored straight into a local
    /// top-k. `MASKED` is a const parameter rather than a runtime check
    /// so the unmasked instantiation carries no mask code at all.
    /// Indices are range-relative; the caller rebases them.
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::too_many_arguments)]
    fn scan_range_neon<const MASKED: bool>(
        codes: &[u8],
        lut: &QueryNeonLut,
        n_byte_groups: usize,
        scales_slice: &[f32],
        block_bytes: usize,
        range_blocks: usize,
        range_vecs: usize,
        k: usize,
        mask: Option<&[u64]>,
    ) -> Vec<(f32, u64)> {
        let mut heap: Vec<(f32, u64)> = Vec::with_capacity(k);
        let mut heap_min = f32::NEG_INFINITY;
        let mut heap_mi = 0usize;
        // One row, so the single-query and 4-query permute-dot kernels can
        // share a signature.
        let mut out = [[0.0f32; BLOCK]; 1];
        for b in 0..range_blocks {
            let base = b * BLOCK;
            let end = (base + BLOCK).min(range_vecs);
            if MASKED && !block_has_allowed(mask, base) {
                continue;
            }
            // SAFETY: NEON is baseline on aarch64; the permute-dot kernel
            // additionally needs `dotprod`, which is exactly what put the
            // codes in the vector-major layout `lut.pd` implies. Slices are
            // range-relative and consistent.
            unsafe {
                if let Some(pd) = lut.pd.as_ref() {
                    score_block_permute_dot_neon::<1>(
                        codes, &[pd], b * block_bytes, n_byte_groups,
                        scales_slice, base, range_vecs, &mut out,
                    );
                } else {
                    score_4bit_block_neon(
                        codes, &lut.uint8_luts, b * block_bytes, n_byte_groups,
                        lut.scale, lut.bias, scales_slice, base, range_vecs, &mut out[0],
                    );
                }
            }
            for (lane, &s) in out[0][..end - base].iter().enumerate() {
                if MASKED && !mask_allows(mask.expect("MASKED implies a mask"), base + lane) {
                    continue;
                }
                if heap.len() < k {
                    heap.push((s, (base + lane) as u64));
                    if heap.len() == k {
                        heap_mi = 0;
                        for (h, &(hs, hix)) in heap.iter().enumerate().skip(1) {
                            if hs < heap[heap_mi].0
                                || (hs == heap[heap_mi].0 && hix > heap[heap_mi].1)
                            {
                                heap_mi = h;
                            }
                        }
                        heap_min = heap[heap_mi].0;
                    }
                } else if s > heap_min {
                    heap[heap_mi] = (s, (base + lane) as u64);
                    heap_mi = 0;
                    for (h, &(hs, hix)) in heap.iter().enumerate().skip(1) {
                        if hs < heap[heap_mi].0 || (hs == heap[heap_mi].0 && hix > heap[heap_mi].1)
                        {
                            heap_mi = h;
                        }
                    }
                    heap_min = heap[heap_mi].0;
                }
            }
        }
        heap
    }

    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::too_many_arguments)]
    fn search_single_query_block_parallel_neon(
        blocked_codes: &[u8],
        lut: &QueryNeonLut,
        n_byte_groups: usize,
        vec_scales: &[f32],
        n_vectors: usize,
        n_blocks: usize,
        k: usize,
        mask: Option<&[u64]>,
    ) -> (Vec<f32>, Vec<i64>) {
        let n_threads = rayon::current_num_threads().max(1);
        let blocks_per_range = block_range_stride(n_blocks, n_threads);
        let ranges: Vec<usize> = (0..n_blocks).step_by(blocks_per_range).collect();
        let block_bytes = n_byte_groups * BLOCK;
        let mut candidates: Vec<(f32, u64)> = ranges
            .into_par_iter()
            .flat_map(|block_start| {
                let range_blocks = blocks_per_range.min(n_blocks - block_start);
                let vec_start = block_start * BLOCK;
                let range_vecs = (range_blocks * BLOCK).min(n_vectors - vec_start);
                let codes = &blocked_codes
                    [block_start * block_bytes..(block_start + range_blocks) * block_bytes];
                let scales_slice = &vec_scales[vec_start..vec_start + range_vecs];
                let mask_slice = mask.map(|m| &m[vec_start / 64..]);
                // Monomorphized on mask presence: the unmasked path must
                // compile to the same lane loop it did before the mask
                // was threaded through, with no per-lane branch and
                // nothing inhibiting the loop's unrolling. Sharing one
                // loop with a loop-invariant `Option` check measured ~18%
                // slower unmasked at one thread.
                let heap = if mask_slice.is_some() {
                    scan_range_neon::<true>(
                        codes, lut, n_byte_groups, scales_slice, block_bytes,
                        range_blocks, range_vecs, k, mask_slice,
                    )
                } else {
                    scan_range_neon::<false>(
                        codes, lut, n_byte_groups, scales_slice, block_bytes,
                        range_blocks, range_vecs, k, None,
                    )
                };
                heap.into_iter()
                    .map(|(s, i)| (s, i + vec_start as u64))
                    .collect::<Vec<_>>()
            })
            .collect();
        candidates.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        candidates.truncate(k);
        (
            candidates.iter().map(|p| p.0).collect(),
            candidates.iter().map(|p| p.1 as i64).collect(),
        )
    }

    #[cfg(target_arch = "aarch64")]
    let results = {
        if nq == 1 && n_blocks >= SINGLE_QUERY_PARALLEL_MIN_BLOCKS {
            vec![search_single_query_block_parallel_neon(
                blocked_codes, &query_luts[0], n_byte_groups, vec_scales,
                n_vectors, n_blocks, k, mask,
            )]
        } else {
        // ARM: 4-query fused scoring (shares code loads + nibble splits
        // across queries), parallelized over 2D (query-quad × block-range)
        // tiles. 1D quad partitioning gives ~nq/4 ragged tasks; with ~8-10
        // workers the tail round idles much of the pool. Splitting the
        // block axis smooths the schedule. Each tile's per-query top-k
        // candidates merge with the same (score desc, index asc) order the
        // single-query block-parallel path uses, so results are identical
        // to a serial scan. A 1-thread pool gets exactly one range —
        // identical work and visit order to the serial scan.
        const QBS: usize = 4;
        // `.max(1)`: an empty query batch (nq == 0) is a legal no-op —
        // main returns empty results for it — but it would otherwise be
        // the divisor below and panic with a divide-by-zero. The tile
        // loop is empty at nq == 0 either way, so the merge yields the
        // same empty result.
        let n_quads = nq.div_ceil(QBS).max(1);
        let n_threads = rayon::current_num_threads().max(1);
        let n_ranges = n_block_ranges(
            nq, n_quads, n_blocks, n_vectors, k, n_threads,
            neon_tpt(), neon_mtb(), false,
        );
        let n_ranges = smooth_tile_count(n_ranges, n_quads, n_threads);
        let blocks_per_range = n_blocks.div_ceil(n_ranges).max(1);
        // Block-range-major, not query-quad-major. Same tile set either
        // way — only the order rayon draws them in — but quad-major puts
        // the tiles in flight at any moment in *different* block ranges,
        // so the workers stream disjoint slices of the code array at
        // once. Block-major keeps them inside one range, sharing those
        // bytes in cache. Worth x1.019 arm / x1.004 x86 at nq=100
        // (see benchmarks/hillclimb/LOG_search.md, H7/H9).
        let tiles: Vec<(usize, usize)> = (0..n_blocks.max(1))
            .step_by(blocks_per_range)
            .flat_map(|b| (0..nq).step_by(QBS).map(move |q| (q, b)))
            .collect();

        let tile_results: Vec<(usize, Vec<Vec<(f32, u64)>>)> = tiles
            .into_par_iter()
            .map(|(qi_start, block_start)| {
                let block_end = (block_start + blocks_per_range).min(n_blocks);
                let qi_end = (qi_start + QBS).min(nq);
                let batch_size = qi_end - qi_start;

                // Fused scoring + top-k: no per-quad score matrix. Each block's
                // 32 scores live on the stack and fold straight into the
                // per-query heaps (block-ascending, lane-ascending — the same
                // visit order as the old flat scan, so results are identical).
                let mut heap_s = vec![vec![f32::NEG_INFINITY; k]; batch_size];
                let mut heap_i = vec![vec![0u64; k]; batch_size];
                let mut heap_sz = [0usize; QBS];
                let mut heap_min = [f32::NEG_INFINITY; QBS];
                let mut heap_mi = [0usize; QBS];

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
                    // `Some` iff this geometry scores through permute-dot;
                    // every query's lut is built the same way, so one check
                    // covers the batch.
                    let pd_refs: Option<[&QueryPermuteDot; QBS]> =
                        query_luts[qi_start].pd.as_ref().map(|_| {
                            std::array::from_fn(|i| {
                                query_luts[qi_start + i]
                                    .pd
                                    .as_ref()
                                    .expect("pd built for every query")
                            })
                        });
                    let mut block_out = [[0.0f32; BLOCK]; QBS];
                    for block_idx in block_start..block_end {
                        let base_vec = block_idx * BLOCK;
                        if !block_has_allowed(mask, base_vec) {
                            // No allowed slot in the block: skipping it inserts
                            // nothing, exactly like the old flat scan which left
                            // NEG_INFINITY rows and mask-skipped every lane.
                            continue;
                        }
                        let block_offset = block_idx * n_byte_groups * BLOCK;
                        let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
                        unsafe {
                            if let Some(pds) = pd_refs {
                                score_block_permute_dot_neon::<QBS>(
                                    blocked_codes, &pds, block_offset, n_byte_groups,
                                    vec_scales, base_vec, n_vectors, &mut block_out,
                                );
                            } else {
                                score_4query_block_neon(
                                    blocked_codes, lut_refs, block_offset, n_byte_groups,
                                    scales, biases, vec_scales, base_vec, n_vectors,
                                    &mut block_out,
                                );
                            }
                            for q in 0..QBS {
                                neon_block_topk_update(
                                    &block_out[q], base_vec, end_lane, mask, k,
                                    &mut heap_s[q], &mut heap_i[q], &mut heap_sz[q],
                                    &mut heap_min[q], &mut heap_mi[q],
                                );
                            }
                        }
                    }
                } else {
                    // Tail path (batch_size < 4): single-query kernel per query
                    for qi_off in 0..batch_size {
                        let qi = qi_start + qi_off;
                        let qlut = &query_luts[qi];
                        for block_idx in block_start..block_end {
                            let base_vec = block_idx * BLOCK;
                            if !block_has_allowed(mask, base_vec) {
                                continue;
                            }
                            let block_offset = block_idx * n_byte_groups * BLOCK;
                            let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
                            let mut block_out = [[0.0f32; BLOCK]; 1];
                            unsafe {
                                if let Some(pd) = qlut.pd.as_ref() {
                                    score_block_permute_dot_neon::<1>(
                                        blocked_codes, &[pd], block_offset, n_byte_groups,
                                        vec_scales, base_vec, n_vectors, &mut block_out,
                                    );
                                } else {
                                    score_4bit_block_neon(
                                        blocked_codes, &qlut.uint8_luts, block_offset, n_byte_groups,
                                        qlut.scale, qlut.bias, vec_scales, base_vec, n_vectors,
                                        &mut block_out[0],
                                    );
                                }
                                neon_block_topk_update(
                                    &block_out[0], base_vec, end_lane, mask, k,
                                    &mut heap_s[qi_off], &mut heap_i[qi_off],
                                    &mut heap_sz[qi_off], &mut heap_min[qi_off],
                                    &mut heap_mi[qi_off],
                                );
                            }
                        }
                    }
                }

                // Hand back each query's raw candidates; the merge below
                // sorts across ranges.
                let cands: Vec<Vec<(f32, u64)>> = (0..batch_size)
                    .map(|qi_off| {
                        let sz = heap_sz[qi_off];
                        heap_s[qi_off][..sz]
                            .iter()
                            .zip(heap_i[qi_off][..sz].iter())
                            .map(|(&s, &i)| (s, i))
                            .collect()
                    })
                    .collect();
                (qi_start, cands)
            })
            .collect();

        // Merge each query's per-range candidates: (score desc, index asc),
        // truncate to k — the same deterministic order the heaps maintain,
        // so tiled and serial results are identical even for tied scores.
        let mut merged: Vec<Vec<(f32, u64)>> = vec![Vec::new(); nq];
        for (qi_start, cands) in tile_results {
            for (off, c) in cands.into_iter().enumerate() {
                merged[qi_start + off].extend(c);
            }
        }
        merged
            .into_iter()
            .map(|mut pairs| {
                pairs.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                });
                pairs.truncate(k);
                let s: Vec<f32> = pairs.iter().map(|p| p.0).collect();
                let i: Vec<i64> = pairs.iter().map(|p| p.1 as i64).collect();
                (s, i)
            })
            .collect::<Vec<_>>()
        }
    };

    // Single-query fast path (x86): one query scanning a large index is
    // memory-bandwidth-bound on one core, so partition the block range
    // across rayon workers — each range runs the existing SIMD kernel on
    // its sub-slices (kernels index relative to the slices they are
    // given), producing a local top-k; ranges then merge. A mask rides
    // along as a word-aligned sub-slice of the bitmap — see
    // [`block_range_stride`].
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    fn search_single_query_block_parallel(
        blocked_codes: &[u8],
        lut: &QueryNeonLut,
        n_byte_groups: usize,
        vec_scales: &[f32],
        n_vectors: usize,
        n_blocks: usize,
        k: usize,
        use_avx512: bool,
        mask: Option<&[u64]>,
    ) -> (Vec<f32>, Vec<i64>) {
        let n_threads = rayon::current_num_threads().max(1);
        // Whole blocks per range, at least 64 blocks (2k vectors) each,
        // an even count so each range is mask-word aligned.
        let blocks_per_range = block_range_stride(n_blocks, n_threads);
        let ranges: Vec<usize> = (0..n_blocks).step_by(blocks_per_range).collect();
        let block_bytes = n_byte_groups * BLOCK;
        let mut candidates: Vec<(f32, u64)> = ranges
            .into_par_iter()
            .flat_map(|block_start| {
                let range_blocks = blocks_per_range.min(n_blocks - block_start);
                let vec_start = block_start * BLOCK;
                let range_vecs = (range_blocks * BLOCK).min(n_vectors - vec_start);
                let codes =
                    &blocked_codes[block_start * block_bytes..(block_start + range_blocks) * block_bytes];
                let scales_slice = &vec_scales[vec_start..vec_start + range_vecs];
                let mask_slice = mask.map(|m| &m[vec_start / 64..]);
                let lut_refs = [lut.uint8_luts.as_slice(); 4];
                let scale_vals = [lut.scale; 4];
                let bias_vals = [lut.bias; 4];
                let mut heap_scores = vec![vec![f32::NEG_INFINITY; k]];
                let mut heap_indices = vec![vec![0u64; k]];
                let mut heap_sizes = vec![0usize];
                let mut heap_mins = vec![f32::NEG_INFINITY];
                let mut heap_min_idxs = vec![0usize];
                // SAFETY: feature presence checked by the caller once.
                unsafe {
                    if let Some(pd) = lut.pd.as_ref() {
                        let pd_refs = [pd; 4];
                        search_multi_query_permute_dot(
                            codes, &pd_refs,
                            n_byte_groups, scales_slice, range_vecs,
                            1, k, mask_slice,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    } else if !lut.split.is_empty() {
                        let split_refs = [lut.split.as_slice(); 4];
                        search_multi_query_vnni(
                            codes, &split_refs, &scale_vals, &bias_vals,
                            n_byte_groups, scales_slice, range_vecs,
                            1, k, mask_slice,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    } else if use_avx512 {
                        search_multi_query_avx512bw(
                            codes, &lut_refs, &scale_vals, &bias_vals,
                            n_byte_groups, scales_slice, range_vecs,
                            1, k, mask_slice,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    } else {
                        search_multi_query_avx2(
                            codes, &lut_refs, &scale_vals, &bias_vals,
                            n_byte_groups, scales_slice, range_vecs,
                            1, k, mask_slice,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    }
                }
                let sz = heap_sizes[0];
                heap_scores[0][..sz]
                    .iter()
                    .zip(heap_indices[0][..sz].iter())
                    .map(|(&s, &i)| (s, i + vec_start as u64))
                    .collect::<Vec<_>>()
            })
            .collect();
        // Deterministic merge: score desc, index asc on ties.
        candidates.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        candidates.truncate(k);
        (
            candidates.iter().map(|p| p.0).collect(),
            candidates.iter().map(|p| p.1 as i64).collect(),
        )
    }

    #[cfg(target_arch = "x86_64")]
    let results = {
        #[cfg(test)]
        let force_scalar_single =
            FORCE_SCALAR_FALLBACK.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let force_scalar_single = false;
        // Every AVX2 kernel (and the AVX-512 kernel's 256-bit epilogue)
        // declares and executes FMA, so the runtime gate must test it
        // too — declaring an unchecked feature "would be a lie the
        // compiler is entitled to act on" (see rotation.rs) and SIGILLs
        // on avx2-without-fma CPU models (#291).
        let avx2_fma_ok =
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        let use_avx512 = is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f")
            && avx2_fma_ok;
        let simd_ok = use_avx512 || avx2_fma_ok;
        if nq == 1
            && n_blocks >= SINGLE_QUERY_PARALLEL_MIN_BLOCKS
            && simd_ok
            && !force_scalar_single
        {
            vec![search_single_query_block_parallel(
                blocked_codes, &query_luts[0], n_byte_groups, vec_scales,
                n_vectors, n_blocks, k, use_avx512, mask,
            )]
        } else {
        // 4, on both kernels. The VNNI kernel *can* carry 8 queries per pass
        // — its u32 accumulators leave the registers free where the classic
        // kernel's u16 pairs did not (H12) — but doing so halves the quad
        // count and therefore the tile count, and x86 has no way to buy that
        // granularity back: H6 showed it degrades with more block ranges.
        // Measured at parity (x0.997), so the simpler batch width stands.
        // See LOG_search.md H23.
        let nq_batch: usize = 4;
        // 2D tiles (query-quad × block-range), mirroring the ARM path:
        // 1D quad partitioning leaves a ragged tail round on the pool.
        // Only when unmasked and SIMD — the mask bitmap is absolute-indexed
        // and the scalar fallback is unsliced, so those keep one range
        // (identical behavior to before). A 1-thread pool also keeps one
        // range: identical work and visit order to the serial scan.
        #[cfg(test)]
        let force_scalar_any = FORCE_SCALAR_FALLBACK.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let force_scalar_any = false;
        // `.max(1)`: an empty query batch (nq == 0) is a legal no-op —
        // main returns empty results for it — but it would otherwise be
        // the divisor below and panic with a divide-by-zero. The tile
        // loop is empty at nq == 0 either way, so the merge yields the
        // same empty result.
        let n_quads = nq.div_ceil(nq_batch).max(1);
        let n_threads = rayon::current_num_threads().max(1);
        let n_ranges = n_block_ranges(
            nq,
            n_quads,
            n_blocks,
            n_vectors,
            k,
            n_threads,
            TILES_PER_THREAD,
            MIN_TILE_BLOCKS,
            serial_required(mask.is_some(), simd_ok, force_scalar_any),
        );
        let n_ranges = smooth_tile_count(n_ranges, n_quads, n_threads);
        let blocks_per_range = n_blocks.div_ceil(n_ranges).max(1);
        let block_bytes = n_byte_groups * BLOCK;
        // Block-range-major, not query-quad-major. Same tile set either
        // way — only the order rayon draws them in — but quad-major puts
        // the tiles in flight at any moment in *different* block ranges,
        // so the workers stream disjoint slices of the code array at
        // once. Block-major keeps them inside one range, sharing those
        // bytes in cache. Worth x1.019 arm / x1.004 x86 at nq=100
        // (see benchmarks/hillclimb/LOG_search.md, H7/H9).
        let tiles: Vec<(usize, usize)> = (0..n_blocks.max(1))
            .step_by(blocks_per_range)
            .flat_map(move |b| (0..nq).step_by(nq_batch).map(move |q| (q, b)))
            .collect();

        let tile_results: Vec<(usize, Vec<Vec<(f32, u64)>>)> = tiles
            .into_par_iter()
            .map(|(qi_start, block_start)| {
                let range_blocks = blocks_per_range.min(n_blocks - block_start);
                let vec_start = block_start * BLOCK;
                let range_vecs = (range_blocks * BLOCK).min(n_vectors - vec_start);
                let codes = &blocked_codes
                    [block_start * block_bytes..(block_start + range_blocks) * block_bytes];
                let scales_slice = &vec_scales[vec_start..vec_start + range_vecs];
                let qi_end = (qi_start + nq_batch).min(nq);
                let batch_nq = qi_end - qi_start;
                let pad_qi = qi_end - 1;
                let lut_refs: Vec<&[u8]> = (0..nq_batch)
                    .map(|i| {
                        let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                        query_luts[qi].uint8_luts.as_slice()
                    }).collect();
                let scale_vals: Vec<f32> = (0..nq_batch)
                    .map(|i| {
                        let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                        query_luts[qi].scale
                    }).collect();
                let bias_vals: Vec<f32> = (0..nq_batch)
                    .map(|i| {
                        let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                        query_luts[qi].bias
                    }).collect();

                let mut heap_scores: Vec<Vec<f32>> = (0..batch_nq)
                    .map(|_| vec![f32::NEG_INFINITY; k]).collect();
                let mut heap_indices: Vec<Vec<u64>> = (0..batch_nq)
                    .map(|_| vec![0u64; k]).collect();
                let mut heap_sizes = vec![0usize; batch_nq];
                let mut heap_mins = vec![f32::NEG_INFINITY; batch_nq];
                let mut heap_min_idxs = vec![0usize; batch_nq];

                #[cfg(test)]
                let force_scalar =
                    FORCE_SCALAR_FALLBACK.load(std::sync::atomic::Ordering::Relaxed);
                #[cfg(not(test))]
                let force_scalar = false;

                unsafe {
                    // avx2+fma too: the AVX-512 kernel executes 256-bit
                    // AVX2/FMA instructions (loads, epilogue helpers),
                    // and the AVX2 kernel uses _mm256_fmadd_ps — gates
                    // must match the kernels' declared features (#291).
                    if !force_scalar && query_luts[pad_qi].pd.is_some() {
                        // 4-bit vector-major: one shared nibble -> level
                        // permute for the whole batch, then a straight
                        // integer dot product per query.
                        let pd_refs: Vec<&QueryPermuteDot> = (0..nq_batch)
                            .map(|i| {
                                let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                                query_luts[qi].pd.as_ref().expect("pd built for every query")
                            })
                            .collect();
                        search_multi_query_permute_dot(
                            codes, &pd_refs,
                            n_byte_groups, scales_slice, range_vecs,
                            batch_nq, k, mask,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    } else if !force_scalar && !query_luts[pad_qi].split.is_empty() {
                        // Vector-major layout in memory: only this kernel can
                        // read it, so the choice is made by the layout, not by
                        // feature detection alone.
                        let split_refs: Vec<&[u8]> = (0..nq_batch)
                            .map(|i| {
                                let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                                query_luts[qi].split.as_slice()
                            })
                            .collect();
                        search_multi_query_vnni(
                            codes, &split_refs, &scale_vals, &bias_vals,
                            n_byte_groups, scales_slice, range_vecs,
                            batch_nq, k, mask,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    } else if !force_scalar
                        && is_x86_feature_detected!("avx512bw")
                        && is_x86_feature_detected!("avx512f")
                        && is_x86_feature_detected!("avx2")
                        && is_x86_feature_detected!("fma")
                    {
                        search_multi_query_avx512bw(
                            codes, &lut_refs, &scale_vals, &bias_vals,
                            n_byte_groups, scales_slice, range_vecs,
                            batch_nq, k, mask,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    } else if !force_scalar
                        && is_x86_feature_detected!("avx2")
                        && is_x86_feature_detected!("fma")
                    {
                        search_multi_query_avx2(
                            codes, &lut_refs, &scale_vals, &bias_vals,
                            n_byte_groups, scales_slice, range_vecs,
                            batch_nq, k, mask,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    } else {
                        // Neither AVX-512 BW nor AVX2 detected at runtime on
                        // this x86_64 CPU. Previously this fell through to
                        // an empty `unsafe { }` block and `heap_sizes` stayed
                        // at 0 — `search` then returned empty top-k results
                        // for every query with no error signal. Fall back to
                        // per-query scalar scoring instead.
                        // Only reachable with n_ranges == 1 (see the tiling
                        // gate), so the unsliced buffers are the full index.
                        for qo in 0..batch_nq {
                            score_query_into_heap(
                                lut_refs[qo],
                                scale_vals[qo],
                                bias_vals[qo],
                                blocked_codes,
                                vec_scales,
                                bits,
                                n_byte_groups,
                                n_vectors,
                                n_blocks,
                                mask,
                                k,
                                &mut heap_scores[qo],
                                &mut heap_indices[qo],
                                &mut heap_sizes[qo],
                                &mut heap_mins[qo],
                                &mut heap_min_idxs[qo],
                            );
                        }
                    }
                }

                // Raw candidates with indices remapped to absolute; the
                // merge below sorts across ranges.
                let cands: Vec<Vec<(f32, u64)>> = (0..batch_nq)
                    .map(|qo| {
                        let sz = heap_sizes[qo];
                        heap_scores[qo][..sz]
                            .iter()
                            .zip(heap_indices[qo][..sz].iter())
                            .map(|(&s, &i)| (s, i + vec_start as u64))
                            .collect()
                    })
                    .collect();
                (qi_start, cands)
            })
            .collect();

        // Merge each query's per-range candidates: (score desc, index asc),
        // truncate to k — identical selection to the serial heap.
        let mut merged: Vec<Vec<(f32, u64)>> = vec![Vec::new(); nq];
        for (qi_start, cands) in tile_results {
            for (off, c) in cands.into_iter().enumerate() {
                merged[qi_start + off].extend(c);
            }
        }
        merged
            .into_iter()
            .map(|mut pairs| {
                pairs.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                });
                pairs.truncate(k);
                let s: Vec<f32> = pairs.iter().map(|p| p.0).collect();
                let i: Vec<i64> = pairs.iter().map(|p| p.1 as i64).collect();
                (s, i)
            })
            .collect::<Vec<_>>()
        }
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let results = {
        // Scalar fallback for architectures without a SIMD kernel.
        let results: Vec<(Vec<f32>, Vec<i64>)> = (0..nq)
            .into_par_iter()
            .map(|qi| {
                let qlut = &query_luts[qi];
                let mut heap_s = vec![f32::NEG_INFINITY; k];
                let mut heap_i = vec![0u64; k];
                let mut heap_sz = 0usize;
                let mut heap_min = f32::NEG_INFINITY;
                let mut heap_mi = 0usize;
                score_query_into_heap(
                    &qlut.uint8_luts,
                    qlut.scale,
                    qlut.bias,
                    blocked_codes,
                    vec_scales,
                    bits,
                    n_byte_groups,
                    n_vectors,
                    n_blocks,
                    mask,
                    k,
                    &mut heap_s,
                    &mut heap_i,
                    &mut heap_sz,
                    &mut heap_min,
                    &mut heap_mi,
                );
                let mut pairs: Vec<(f32, u64)> = heap_s[..heap_sz].iter()
                    .zip(heap_i[..heap_sz].iter()).map(|(&s, &i)| (s, i)).collect();
                pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.cmp(&b.1)));
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

#[cfg(test)]
mod gate_tests {
    use super::*;

    /// The single-query pool gate must never fire below the granularity
    /// at which the batch dispatch itself splits the block axis.
    ///
    /// This is the rule the threshold is chosen by (#336): routing an
    /// nq=1 search into the process-wide fork-safe pool costs an
    /// `install` handoff *and* a slot in a queue shared by every other
    /// caller, so it must only happen where the block axis carries at
    /// least one full tile. At the old value (256 blocks = 8192 vectors)
    /// the gate fired four tile-widths early: the handoff was larger
    /// than the entire scan, and every concurrent caller of an 8k-32k
    /// index was serialized behind the pool for nothing.
    ///
    /// A structural invariant rather than a latency assertion on
    /// purpose: the honest and defective distributions of a ~20 µs
    /// handoff overlap completely on a loaded CI box.
    #[test]
    fn single_query_gate_is_at_least_one_tile_wide() {
        assert!(
            SINGLE_QUERY_PARALLEL_MIN_BLOCKS >= MIN_TILE_BLOCKS,
            "single-query pool gate ({SINGLE_QUERY_PARALLEL_MIN_BLOCKS} blocks) fires below \
             the batch dispatch's own tile granularity ({MIN_TILE_BLOCKS} blocks): an nq=1 \
             search would enter the shared pool at a size where the work is not worth \
             splitting (#336)",
        );
    }

    /// `single_query_parallelizes` is the predicate the Python bindings
    /// use to decide whether a search must run inside the fork-safe
    /// pool, so a single query it reports as *serial* must not reach
    /// rayon in the batch dispatch either — whatever the tile
    /// granularity (#147). The clamp is what makes the threshold safe to
    /// move; without it, raising the gate past `MIN_TILE_BLOCKS` would
    /// split the block axis outside the pool.
    #[test]
    fn sub_gate_single_query_never_splits_the_block_axis() {
        let n_vectors = (SINGLE_QUERY_PARALLEL_MIN_BLOCKS - 1) * BLOCK;
        let n_blocks = n_vectors.div_ceil(BLOCK);
        assert!(!single_query_parallelizes(n_vectors));
        for &min_tile in &[1usize, 8, 64, MIN_TILE_BLOCKS] {
            assert_eq!(
                n_block_ranges(1, 1, n_blocks, n_vectors, 10, 16, TILES_PER_THREAD, min_tile, false),
                1,
                "nq=1 below the pool gate split the block axis at min_tile={min_tile}",
            );
        }
        // The clamp is specific to nq == 1: a real batch at the same
        // size still tiles.
        assert!(n_block_ranges(64, 16, n_blocks, n_vectors, 10, 16, TILES_PER_THREAD, 1, false) > 1);
    }

    /// Above the gate a single query does split, so routing it through
    /// the pool is the correct call — the two halves of the rule have to
    /// agree or the gate is either useless or unsafe.
    #[test]
    fn above_gate_single_query_does_split() {
        let n_vectors = SINGLE_QUERY_PARALLEL_MIN_BLOCKS * BLOCK * 4;
        assert!(single_query_parallelizes(n_vectors));
        assert!(
            n_block_ranges(
                1,
                1,
                n_vectors.div_ceil(BLOCK),
                n_vectors,
                10,
                16,
                TILES_PER_THREAD,
                MIN_TILE_BLOCKS,
                false
            ) > 1
        );
    }

    /// Each of the three conditions that forces `n_block_ranges` to 1
    /// must do so ON ITS OWN. The tests above only ever vary the third
    /// disjunct (`nq == 1` below the pool gate), which left the `||`
    /// joining `n_threads == 1` and `serial` unpinned: turned into `&&`
    /// it reads `(n_threads == 1 && serial) || (nq == 1 && ..)`, so a
    /// single-threaded pool would start splitting the block axis and a
    /// masked or scalar search would too. Both are #147 violations.
    ///
    /// The size is chosen above the pool gate so that "all three false"
    /// genuinely splits — otherwise every row would return 1 for the
    /// wrong reason and the table could not fail.
    /// Pin the tile target where it is the binding term: enough blocks
    /// that the block cap clears, k small enough that the k-cap clears,
    /// n_quads equal to n_threads so the target divides out exactly.
    /// `(16 * TILES_PER_THREAD).div_ceil(16) = TILES_PER_THREAD` — this
    /// is the assertion the baseline test above cannot make (its block
    /// cap binds first), and the one that distinguishes 32 from the old
    /// 4 (or any arithmetic slip in the target term).
    #[test]
    fn the_tile_target_binds_when_the_caps_do_not() {
        let n_blocks = 100 * MIN_TILE_BLOCKS; // block cap = 100 >> target
        let n_vectors = n_blocks * BLOCK;
        assert_eq!(
            n_block_ranges(64, 16, n_blocks, n_vectors, 1, 16, TILES_PER_THREAD, MIN_TILE_BLOCKS, false),
            TILES_PER_THREAD,
            "with both caps clear, the range count IS the per-worker tile target",
        );
        // And the NEON pair keeps its documented relation to the shared
        // constants (H15: 2x finer target, 4x finer block floor).
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(TILES_PER_THREAD_NEON, TILES_PER_THREAD * 2);
            assert_eq!(MIN_TILE_BLOCKS_NEON, MIN_TILE_BLOCKS / 4);
        }
    }

    #[test]
    fn each_serial_condition_forces_one_range_on_its_own() {
        let n_vectors = SINGLE_QUERY_PARALLEL_MIN_BLOCKS * BLOCK * 4;
        let n_blocks = n_vectors.div_ceil(BLOCK);
        assert!(
            single_query_parallelizes(n_vectors),
            "fixture must sit above the pool gate or the table is vacuous",
        );

        // Baseline: nothing forces serial, so the axis does split.
        //
        // Pinned to the exact count, not just `> 1`. The three rows below
        // only prove the guard fires; nothing else pins the arithmetic
        // *under* it, and `> 1` is too loose to notice a change there —
        // e.g. `(n_threads * 4)` becoming `(n_threads + 4)` yields 2,
        // which still satisfies `> 1` while halving the parallelism on
        // every batch search. For this tuple the three terms are
        // `(16 * 32).div_ceil(16) = 32`, `n_blocks.div_ceil(MIN_TILE_BLOCKS)
        // = 4096/1024 = 4`, and `range_cap_for_k(131072, 10) = 26`, so
        // the min is the BLOCK cap, 4 — the tile target does not bind
        // here; `the_tile_target_binds_when_the_caps_do_not` pins it.
        // Update this number deliberately if a cap moves.
        assert_eq!(
            n_block_ranges(64, 16, n_blocks, n_vectors, 10, 16, TILES_PER_THREAD, MIN_TILE_BLOCKS, false),
            4,
            "baseline range count changed; the rows below prove only that \
             the guard fires, so this is the one place the arithmetic \
             beneath it is pinned",
        );

        // n_threads == 1 alone. `n_quads` must be 1 here, not 16: with
        // 16 the arithmetic below the guard yields `(1*4).div_ceil(16)
        // == 1` anyway, so the row would pass whether or not the guard
        // exists and would pin nothing. This is the disjunct that fires
        // in production — the bindings pin the global pool to a
        // 1-thread sentinel, so the inline nq==1 path sees
        // `rayon::current_num_threads() == 1`.
        assert_eq!(
            n_block_ranges(64, 1, n_blocks, n_vectors, 10, 1, TILES_PER_THREAD, MIN_TILE_BLOCKS, false),
            1,
            "a single-threaded pool must not split the block axis",
        );

        // serial alone.
        assert_eq!(
            n_block_ranges(64, 16, n_blocks, n_vectors, 10, 16, TILES_PER_THREAD, MIN_TILE_BLOCKS, true),
            1,
            "an explicitly serial call must not split the block axis",
        );

        // nq == 1 below the gate alone. `min_tile` must be 1 here, not
        // MIN_TILE_BLOCKS: this fixture is one block short of the gate
        // (n_blocks = MIN_TILE_BLOCKS - 1), so the below-guard cap
        // `n_blocks.div_ceil(min_tile_blocks)` would be
        // `1023.div_ceil(1024) == 1` and force the whole `.min()` chain
        // to 1 whether or not the guard exists — the same vacuity the
        // n_threads row above had.
        let small = (SINGLE_QUERY_PARALLEL_MIN_BLOCKS - 1) * BLOCK;
        assert_eq!(
            n_block_ranges(1, 1, small.div_ceil(BLOCK), small, 10, 16, TILES_PER_THREAD, 1, false),
            1,
            "nq=1 below the pool gate must not split the block axis (#147)",
        );
    }

    /// `serial_required` is the dispatch's three-term serial predicate.
    /// Each term must force serial on its own: a mask makes the walk
    /// sequential, absent SIMD leaves nothing to tile, and a forced
    /// scalar path is a caller instruction. Inline at the call site these
    /// terms were unreachable from any test, so an `||` could silently
    /// become `&&` — which would let a masked search split the block
    /// axis outside the fork-safe pool (#147).
    ///
    /// Gated to x86 with the function it tests: the aarch64 dispatch
    /// passes a literal `false`, so there is no predicate there to pin.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn each_term_of_the_serial_predicate_forces_serial_alone() {
        // All false is the only combination that may run parallel.
        assert!(!serial_required(false, true, false));

        assert!(serial_required(true, true, false), "a mask alone must force serial");
        assert!(serial_required(false, false, false), "absent SIMD alone must force serial");
        assert!(serial_required(false, true, true), "forced scalar alone must force serial");

        // And any combination stays serial.
        assert!(serial_required(true, false, true));
    }
}
