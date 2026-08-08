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
const MIN_TILE_BLOCKS_NEON: usize = MIN_TILE_BLOCKS / 2;

/// The x86 dispatch's own block floor, 3x the shared one.
///
/// `MIN_TILE_BLOCKS` cannot simply move: it is also the single-query pool
/// gate (`SINGLE_QUERY_PARALLEL_MIN_BLOCKS >= MIN_TILE_BLOCKS`) and the base
/// for the NEON floor, so raising it would break the invariant and undo H69.
/// x86 gets its own constant, exactly as aarch64 already does.
///
/// 3072 gives 3 block ranges at N=200k where 1024 gave 7. Swept after
/// H54/H59/H62/H65 changed x86's memory behaviour — H37 had found 7 optimal
/// against the pre-H34 kernel. nq=100 MT medians: 18.755 / 18.711 / **18.004**
/// / 18.902 ms for 1024 / 2048 / 3072 / 4096, a x1.042 win with no overlap,
/// and 4096 turning back up marks it as a knee rather than a trend. Same
/// direction as H69 on arm: a faster, streaming kernel wants fewer and
/// longer contiguous runs. See H70.
#[cfg(target_arch = "x86_64")]
const MIN_TILE_BLOCKS_X86: usize = MIN_TILE_BLOCKS * 3;
// Was `/ 4` (256 blocks, 24 ranges at N=200k), tuned in H15 against a
// kernel that has since gained SMMLA (H33), the vm8 layout (H41), the H45
// restructure and prefetch (H67). `/ 2` gives 12 ranges and is worth x1.048
// on nq=100 MT, winning all five interleaved rounds without overlap. Fewer,
// larger ranges suit a kernel that now streams: each worker keeps a longer
// contiguous run, and the per-range top-k duplication that argued for a
// finer split costs the same as it always did while the scan got 2.5x
// cheaper. See H69.

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
/// Picks the prefetching instantiation for a single-query sweep and the
/// bare one for a batch (H6).
///
/// The depth-8 lookahead is worth +16% at nq=1 ST and costs ~5% at nq=100,
/// because a 2-bit block is half a 4-bit block: H62's distance runs two
/// thirds of a block ahead here and evicts what the next pass re-reads. A
/// batch re-reads; a single sweep does not.
///
/// # Safety
/// Same contract as [`search_multi_query_vnni`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni", enable = "avx512vl", enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn search_multi_query_vnni_dispatch(
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
    if nq == 1 {
        search_single_query_vnni_blk2(
            blocked_codes, split_luts, scales, biases, n_byte_groups, vec_scales,
            n_vectors, k, mask, heap_scores, heap_indices, heap_sizes,
            heap_mins, heap_min_idxs,
        )
    } else {
        search_multi_query_vnni::<false>(
            blocked_codes, split_luts, scales, biases, n_byte_groups, vec_scales,
            n_vectors, nq, k, mask, heap_scores, heap_indices, heap_sizes,
            heap_mins, heap_min_idxs,
        )
    }
}

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
unsafe fn search_multi_query_vnni<const PF: bool>(
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

    // 8-wide by construction: `acc` is `[[__m512i; 2]; 8]` and both the
    // accumulation and epilogue loops clamp at `nq.min(8)`, so a wider
    // batch would be silently truncated, not scored. The batch dispatch
    // widens `nq_batch` past 8 only when the 10-lane permute-dot kernel
    // is the one taking the batch (see the width gate there).
    debug_assert!(nq <= 8, "search_multi_query_vnni is 8-wide; got nq={nq}");

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
                // H4: this kernel had no prefetch at all, while the 4-bit
                // permute-dot path has had one since H59/H62 — the same
                // multi-sweep re-streaming, and at 2 bits nq=100 sweeps 13
                // times over 38.4 MB. Depth follows H62's measured knee: 32
                // quads when the batch re-reads the codes, 8 at nq=1 where a
                // single sweep overshoots.
                // H5: only the single-query sweep wants it here. At nq=1 the
                // depth-8 lookahead is worth +10.8% ST; at nq=100 the same
                // idea costs ~5%, because a 2-bit block is half a 4-bit block,
                // so H62's 32-quad depth runs two thirds of a block ahead
                // instead of one third and evicts what the next pass is about
                // to re-read. The batch re-reads; the single sweep does not.
                // `PF` is const so the batched instantiation emits no branch
                // at all: nq=100 must be machine-identical to main, or the
                // no-regression gate is measuring a test it cannot see.
                if PF {
                    let pf = block_base + (q4 + 8) * 128 + h * 64;
                    if pf + 64 <= blocked_codes.len() {
                        _mm_prefetch(blocked_codes.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                    }
                }
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
            // H11: convert and bias at full width and hand two __m512 to the
            // 512-bit epilogue, exactly as the 4-bit permute-dot path has
            // done since H111 (+5.9% MT / +7.7% ST there). This kernel was
            // still splitting into four __m256 for the AVX2 epilogue — P6
            // priced the shipped cell 25% under the inner loop's roofline,
            // and this per-(block, query) code is where that gap lives.
            let vs = _mm512_set1_ps(scales[qi]);
            let vb = _mm512_set1_ps(biases[qi]);
            let f0 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][0]), vs), vb);
            let f1 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][1]), vs), vb);
            avx512_post_flush_heap_update(
                f0,
                f1,
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

/// Two-block interleaved single-query scan (H34, re-opening H13).
///
/// The 4-bit climb's H54 found x1.29 at nq=1 ST on x86 by giving the scan
/// several independent block streams: one stream leaves the core waiting on a
/// single miss chain and the fill buffers can hold far more. The 2-bit kernel
/// never got it, and H13 closed the cell on a roofline that P18 has since
/// shown was measured with a probe slower than the kernel it bounded.
///
/// Two blocks in flight, one query: 4 zmm of accumulator, and each quad's
/// table load is shared between them, so per-block table traffic halves too.
///
/// # Safety
/// Same contract as [`search_multi_query_vnni`]; `nq` must be 1.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "avx512f",
    enable = "avx512bw",
    // vbmi is what makes `vpermb` a real instruction. Without it LLVM
    // emulates `_mm512_permutexvar_epi8` and the kernel runs 3x slower —
    // the first two builds of this hypothesis measured exactly that, and
    // the feature list, not the loop structure, was the defect.
    enable = "avx512vbmi",
    enable = "avx512vnni"
)]
#[allow(clippy::too_many_arguments)]
unsafe fn search_single_query_vnni_blk2(
    blocked_codes: &[u8],
    split_luts: &[&[u8]],
    scales: &[f32],
    biases: &[f32],
    n_byte_groups: usize,
    vec_scales: &[f32],
    n_vectors: usize,
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
    let kpos = _mm512_set1_epi32(0x3020_1000u32 as i32);
    let ones = _mm512_set1_epi8(1);
    let quads = n_byte_groups / 4;
    let block_bytes = n_byte_groups * BLOCK;

    // Pairs are unrolled at compile time. `pair` as a runtime bound made
    // `acc[i][h]` a runtime index, which LLVM cannot hold in registers — it
    // spilled every accumulator to the stack and the first build measured
    // x0.34. Same trap H34 documented in the 4-bit log for runtime batch
    // widths; the odd tail block goes through its own straight-line copy.
    // Two streams, not four: H35 measured BLK=4 at 1.37 ms against this
    // shape's 1.30 on the same box. Two blocks is enough to cover the miss
    // latency and four doubles the live accumulators and code registers for
    // nothing.
    let n_pairs = n_blocks / 2;
    for pb in 0..n_pairs {
        let b = pb * 2;
        if !block_has_allowed(mask, b * BLOCK) && !block_has_allowed(mask, (b + 1) * BLOCK) {
            continue;
        }
        let base0 = b * block_bytes;
        let base1 = base0 + block_bytes;
        let mut a0 = [_mm512_setzero_si512(); 2];
        let mut a1 = [_mm512_setzero_si512(); 2];

        for q4 in 0..quads {
            for h in 0..2 {
                let pf = base0 + (q4 + 8) * 128 + h * 64;
                if pf + 64 <= blocked_codes.len() {
                    _mm_prefetch(blocked_codes.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                }
                let tp = split_luts[0].as_ptr().add(q4 * 128);
                let tlo = _mm512_loadu_si512(tp as *const __m512i);
                let thi = _mm512_loadu_si512(tp.add(64) as *const __m512i);
                let c0 = _mm512_loadu_si512(
                    blocked_codes.as_ptr().add(base0 + q4 * 128 + h * 64) as *const __m512i);
                let c1 = _mm512_loadu_si512(
                    blocked_codes.as_ptr().add(base1 + q4 * 128 + h * 64) as *const __m512i);
                let i0lo = _mm512_or_si512(_mm512_and_si512(c0, m0f), kpos);
                let i0hi = _mm512_or_si512(
                    _mm512_and_si512(_mm512_srli_epi16(c0, 4), m0f), kpos);
                let i1lo = _mm512_or_si512(_mm512_and_si512(c1, m0f), kpos);
                let i1hi = _mm512_or_si512(
                    _mm512_and_si512(_mm512_srli_epi16(c1, 4), m0f), kpos);
                a0[h] = _mm512_dpbusd_epi32(a0[h], _mm512_permutexvar_epi8(i0lo, tlo), ones);
                a0[h] = _mm512_dpbusd_epi32(a0[h], _mm512_permutexvar_epi8(i0hi, thi), ones);
                a1[h] = _mm512_dpbusd_epi32(a1[h], _mm512_permutexvar_epi8(i1lo, tlo), ones);
                a1[h] = _mm512_dpbusd_epi32(a1[h], _mm512_permutexvar_epi8(i1hi, thi), ones);
            }
        }

        let vs = _mm512_set1_ps(scales[0]);
        let vb = _mm512_set1_ps(biases[0]);
        for (i, a) in [a0, a1].iter().enumerate() {
            let base_vec = (b + i) * BLOCK;
            let end = (base_vec + BLOCK).min(n_vectors);
            let f0 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[0]), vs), vb);
            let f1 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[1]), vs), vb);
            avx512_post_flush_heap_update(
                f0, f1, base_vec, end, vec_scales.as_ptr().add(base_vec), 0, k, mask,
                heap_scores, heap_indices, heap_sizes, heap_mins, heap_min_idxs,
            );
        }
    }

    // Tail blocks, single-stream.
    for b in (n_pairs * 2)..n_blocks {
        if block_has_allowed(mask, b * BLOCK) {
            let base = b * block_bytes;
            let mut a = [_mm512_setzero_si512(); 2];
            for q4 in 0..quads {
                for h in 0..2 {
                    let tp = split_luts[0].as_ptr().add(q4 * 128);
                    let tlo = _mm512_loadu_si512(tp as *const __m512i);
                    let thi = _mm512_loadu_si512(tp.add(64) as *const __m512i);
                    let c = _mm512_loadu_si512(
                        blocked_codes.as_ptr().add(base + q4 * 128 + h * 64) as *const __m512i);
                    let ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), kpos);
                    let ihi = _mm512_or_si512(
                        _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), kpos);
                    a[h] = _mm512_dpbusd_epi32(a[h], _mm512_permutexvar_epi8(ilo, tlo), ones);
                    a[h] = _mm512_dpbusd_epi32(a[h], _mm512_permutexvar_epi8(ihi, thi), ones);
                }
            }
            let base_vec = b * BLOCK;
            let end = (base_vec + BLOCK).min(n_vectors);
            let vs = _mm512_set1_ps(scales[0]);
            let vb = _mm512_set1_ps(biases[0]);
            let f0 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[0]), vs), vb);
            let f1 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[1]), vs), vb);
            avx512_post_flush_heap_update(
                f0, f1, base_vec, end, vec_scales.as_ptr().add(base_vec), 0, k, mask,
                heap_scores, heap_indices, heap_sizes, heap_mins, heap_min_idxs,
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
/// `NQ` is a const generic, and that is load-bearing rather than tidiness.
/// With a runtime batch width `acc[qi]` is a runtime index into an array,
/// which LLVM cannot hold in registers: it spilled all 16 accumulators to
/// the stack and wrapped every pair of `vpdpbusd` in a 64-byte reload and
/// 64-byte store, plus two pointer chases and a bounds branch, because the
/// query loop never unrolled either. A fixed-size array makes every index a
/// constant, the loop unrolls, and the accumulators stay in `zmm`. See H34.
///
/// Callers pass a batch padded to `NQ` and the real count in `nq`; the
/// padding lanes are scored and discarded, which costs nothing measurable
/// (the epilogue is ~2% of runtime) and keeps the hot loop branch-free.
///
/// See P18 in `benchmarks/hillclimb/LOG_search.md`.
/// Whether this core has GFNI, cached after the first probe.
///
/// AVX-512 VNNI and GFNI are not the same generation: Cascade Lake has VNNI
/// without GFNI, so the affine-shift kernel cannot be selected by the same
/// gate as permute-dot itself.
///
/// `TURBOVEC_NO_GFNI` forces the baseline kernel. Without it the fallback is
/// unreachable on any machine this is developed or benchmarked on — both
/// bench boxes and every current dev machine have GFNI — so the escape hatch
/// is what keeps that path testable rather than merely present. Same role as
/// `TURBOVEC_NO_I8MM` on aarch64.
#[cfg(target_arch = "x86_64")]
pub(crate) fn have_gfni() -> bool {
    static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        std::env::var_os("TURBOVEC_NO_GFNI").is_none() && is_x86_feature_detected!("gfni")
    })
}

#[cfg(target_arch = "x86_64")]
macro_rules! define_permute_dot {
    ($name:ident, [$($feature:literal),*], |$c:ident, $mask:ident| $hi:expr) => {

        #[target_feature($(enable = $feature),*)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name<const NQ: usize, const BLK: usize>(
    blocked_codes: &[u8],
    pds: &[&QueryPermuteDot; NQ],
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
    let nqm = nq.min(NQ);

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

    for b in (0..n_blocks).step_by(BLK) {
        let base_vec = b * BLOCK;
        // The skip must clear the WHOLE interleaved group: H54 steps this
        // loop by BLK blocks, and testing only the head block's mask word
        // silently dropped allowed vectors in blocks 2..BLK of a group
        // whose head was fully masked (caught by the masked filtering
        // suite on AVX-512 hardware — short results padded with the
        // heap prefill, surfacing as disallowed slot 0).
        let group_blocks = BLK.min(n_blocks - b);
        if !(0..group_blocks).any(|s| block_has_allowed(mask, base_vec + s * BLOCK)) {
            continue;
        }
        let block_base = b * block_bytes;
        // acc[q][h]: 16 i32 lanes = 16 vectors, halves h = vectors 0-15,
        // 16-31. Seeded with the +128 cancellation rather than zero.
        // `BLK` blocks interleaved inside the quad loop, so the kernel walks
        // BLK independent sequential streams instead of one.
        //
        // P24 measured x86 nq=1 as memory-bound once the array leaves cache
        // (14.5 -> 31.5 ns/vec, 38 MB -> 154 MB) while arm stays flat. H43
        // (prefetch) and H49 (more accumulator chains) both failed there, and
        // for the same unstated reason: neither adds a memory *stream*. One
        // query walks one stream, so outstanding misses are capped at what a
        // single stream sustains. At NQ=8 the registers are full and BLK=1;
        // at NQ=1 only two accumulators are live and there is room. See H54.
        let nb = BLK.min(n_blocks - b);
        let mut acc = [[[_mm512_setzero_si512(); 2]; NQ]; BLK];
        for sub in 0..nb {
            for (a, pd) in acc[sub].iter_mut().zip(pds.iter()) {
                let z = _mm512_set1_epi32(pd.zero);
                a[0] = z;
                a[1] = z;
            }
        }

        for q4 in 0..quads {
            for h in 0..2 {
              for sub in 0..nb {
                let block_base = block_base + sub * block_bytes;
                let acc = &mut acc[sub];
                // One line per 64-byte load, a fixed distance ahead — the
                // competent shape from H43, which measured neutral at
                // nq=100 when that cell was compute-bound. P28 showed it no
                // longer is: crossing out of cache now costs 34% there,
                // against 4% when H43 was taken. See H59.
                // Lookahead depends on the batch width, because the two
                // widths make different use of the array. At nq=100 the
                // scan sweeps it 12.5 times and a deep 4 KB lookahead is
                // worth +25% ST; at nq=1 it sweeps once and the same depth
                // overshoots, costing 6%. Measured: 32 quads is the knee at
                // NQ=8 (64/96/128 are indistinguishable) and 8 at NQ=1.
                // See H62.
                let pf_quads = if NQ == 1 { 8 } else { 32 };
                {
                    let pf = block_base + (q4 + pf_quads) * 128 + h * 64;
                    if pf + 64 <= blocked_codes.len() {
                        _mm_prefetch(blocked_codes.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                    }
                }
                let c = _mm512_loadu_si512(
                    blocked_codes.as_ptr().add(block_base + q4 * 128 + h * 64) as *const __m512i,
                );
                // Shared across the whole batch — this is the whole point.
                let vlo = _mm512_shuffle_epi8(levels, _mm512_and_si512(c, m0f));
                let vhi = _mm512_shuffle_epi8(levels, { let ($c, $mask) = (c, m0f); $hi });
                for qi in 0..NQ {
                    // Four dimensions of query weights, broadcast: every
                    // dword lane holds a different database vector but the
                    // same four byte-groups. `weights` is `Vec<i8>`, so the
                    // 4-byte read is unaligned by construction. LLVM folds
                    // this into `vpdpbusd`'s embedded `{1to16}` broadcast
                    // operand, so it costs no separate instruction.
                    let wp = pds[qi].weights.as_ptr().add(q4 * 8);
                    let wlo = _mm512_set1_epi32((wp as *const i32).read_unaligned());
                    let whi = _mm512_set1_epi32((wp.add(4) as *const i32).read_unaligned());
                    acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], vlo, wlo);
                    acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], vhi, whi);
                }
              }
            }
        }

        for sub in 0..nb {
        let base_vec = base_vec + sub * BLOCK;
        let acc = &acc[sub];
        let end = (base_vec + BLOCK).min(n_vectors);
        for qi in 0..nqm {
            // H111: stay 512-bit. Two converts and two FMAs replace four
            // converts, four multiplies, four adds and four 256-bit extracts.
            let vs = _mm512_set1_ps(pds[qi].scale);
            let vb = _mm512_set1_ps(pds[qi].bias);
            let f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[qi][0]), vs, vb);
            let f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[qi][1]), vs, vb);
            avx512_post_flush_heap_update(
                f0,
                f1,
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
}
    };
}

#[cfg(target_arch = "x86_64")]
define_permute_dot!(
    search_multi_query_permute_dot,
    ["avx2", "fma", "avx512f", "avx512bw", "avx512vnni"],
    |c, m0f| _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f)
);

#[cfg(target_arch = "x86_64")]
define_permute_dot!(
    search_multi_query_permute_dot_gfni,
    ["avx2", "fma", "avx512f", "avx512bw", "avx512vnni", "gfni"],
    // `c >> 4` per byte in one instruction. A logical shift is linear over
    // GF(2), so `vgf2p8affineqb` computes it as an 8x8 bit-matrix product,
    // and unlike `vpsrlw` it works a byte at a time — no follow-up AND to
    // clear the neighbouring bits a 16-bit shift drags in. Two ops become
    // one, worth x1.039 MT. The matrix is the identity
    // (0x0102040810204080, since output bit `i` is `parity(A[7-i] & x)`)
    // keeping only the rows that map input bits 4..7 to output bits 0..3.
    // See H38.
    |c, _m0f| _mm512_gf2p8affine_epi64_epi8(c, _mm512_set1_epi64(0x1020408000000000u64 as i64), 0)
);

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

/// The same per-block epilogue at 512 bits, for callers that already have
/// AVX-512 (H111).
///
/// H110 measured 5.3% at x86 nq=100 ST sitting in code the `x86-64-v2`
/// baseline compiles for pre-AVX2 CPUs — the whole gap is the v3 -> v4 step,
/// so it is feature availability and not scheduling. The baseline itself
/// cannot move (#137: the dispatch prologue runs before feature detection),
/// but a `#[target_feature]` variant reached from the AVX-512 kernel can,
/// and needs no new runtime check because that kernel already declares the
/// features.
///
/// A block that beats nothing is the overwhelmingly common case at
/// nq=100 over 200k vectors, so the path that matters is the early exit:
/// two scale multiplies, two compares and one mask test, against the AVX2
/// version's four of each. The caller also saves half its convert-and-bias
/// work by handing over two `__m512` instead of four `__m256`.
///
/// Selection itself is untouched — P44 priced that at ~free for k=10, and
/// this changes only the arithmetic that runs over all 32 lanes regardless.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn avx512_post_flush_heap_update(
    f0: std::arch::x86_64::__m512,
    f1: std::arch::x86_64::__m512,
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
    let sz_now = heap_sizes[qi];

    // Fast path: a full block with a filled heap. Everything else falls back
    // to the AVX2 routine rather than being duplicated — those paths run once
    // per scan (heap filling) or once per index (the ragged tail), so a second
    // copy of them would be all risk and no measurable gain.
    if sz_now >= k && end_lane == BLOCK {
        let s0 = _mm512_mul_ps(f0, _mm512_loadu_ps(vec_scales_ptr));
        let s1 = _mm512_mul_ps(f1, _mm512_loadu_ps(vec_scales_ptr.add(16)));
        let thr = _mm512_set1_ps(heap_mins[qi]);
        let m0 = _mm512_cmp_ps_mask(s0, thr, _CMP_GT_OQ) as u32;
        let m1 = _mm512_cmp_ps_mask(s1, thr, _CMP_GT_OQ) as u32;
        if (m0 | m1) == 0 {
            return;
        }

        let mut block_out = [0.0f32; BLOCK];
        let bp = block_out.as_mut_ptr();
        if m0 != 0 {
            _mm512_storeu_ps(bp, s0);
        }
        if m1 != 0 {
            _mm512_storeu_ps(bp.add(16), s1);
        }

        let hs = &mut heap_scores[qi];
        let hi = &mut heap_indices[qi];
        let hmin = &mut heap_mins[qi];
        let hmi = &mut heap_min_idxs[qi];
        for (half, &mask0) in [m0, m1].iter().enumerate() {
            let mut m = mask0;
            while m != 0 {
                let bit = m.trailing_zeros() as usize;
                m &= m - 1;
                let lane = half * 16 + bit;
                if let Some(am) = mask {
                    if !mask_allows(am, base_vec + lane) {
                        continue;
                    }
                }
                let score = block_out[lane];
                // Re-checked because `hmin` rises as this loop inserts, so a
                // lane that cleared the vector threshold may no longer clear
                // the current one. Same guard the AVX2 path uses.
                if score > *hmin {
                    hs[*hmi] = score;
                    hi[*hmi] = (base_vec + lane) as u64;
                    let (m2, mi) = rescan_min(hs, hi, k);
                    *hmin = m2;
                    *hmi = mi;
                }
            }
        }
        return;
    }

    let fa = [
        _mm512_extractf32x8_ps(f0, 0),
        _mm512_extractf32x8_ps(f0, 1),
        _mm512_extractf32x8_ps(f1, 0),
        _mm512_extractf32x8_ps(f1, 1),
    ];
    avx2_post_flush_heap_update(
        &fa, base_vec, end, vec_scales_ptr, qi, k, mask,
        heap_scores, heap_indices, heap_sizes, heap_mins, heap_min_idxs,
    );
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
            // H6: no prefetch here. H4/H5 measured one at 32 units — +2.8% at
            // nq=100 ST and -1.8% at nq=100 MT, because eight workers sharing
            // L2/L3 pay for a lookahead that one worker profits from. The two
            // cancel and the MT side breaks the no-regression gate, so the arm
            // batched kernel keeps the hardware prefetcher it already had.
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

/// `SDOT` by element: `acc.4s += sum of four s8 x s8 products per 32-bit
/// lane`, taking the second operand from one 4-byte group of `b` selected by
/// a compile-time index.
///
/// Lets a single register carry two byte-group quads' worth of query
/// weights instead of one quad needing two broadcast registers — halving
/// both the weight registers live in the loop and the instructions that
/// fill them. `IDX` is an assembler immediate, hence the const parameter.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_lane<const IDX: i32>(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut o = acc;
    std::arch::asm!(
        ".arch_extension dotprod",
        "sdot {o:v}.4s, {a:v}.16b, {b:v}.4b[{idx}]",
        o = inout(vreg) o,
        a = in(vreg) a,
        b = in(vreg) b,
        idx = const IDX,
        options(pure, nomem, nostack),
    );
    o
}

/// One `SMMLA`: a 2x8 by 8x2 signed int8 matrix product accumulated into a
/// 2x2 i32 tile. 32 MACs per instruction against `SDOT`'s 16.
///
/// `a` is the 2x8 operand row-major — bytes 0..8 are row 0, 8..16 row 1.
/// `b` is the 8x2 operand *column*-major — bytes 0..8 are column 0. The
/// destination is the 2x2 product row-major: lane 0 = row0.col0, lane 1 =
/// row0.col1, lane 2 = row1.col0, lane 3 = row1.col1.
///
/// Inline asm for the same reason as [`sdot`] — `vmmlaq_s32` is unstable —
/// plus `.arch_extension i8mm`, which is an optional v8.6 feature and so
/// must be runtime-detected by the caller ([`have_i8mm`]).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn smmla(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut o = acc;
    std::arch::asm!(
        ".arch_extension i8mm",
        "smmla {o:v}.4s, {a:v}.16b, {b:v}.16b",
        o = inout(vreg) o,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    o
}

/// Layout-facing i8mm probe, callable from any arch so [`crate::pack`] can
/// ask without `cfg` gymnastics. Always false off aarch64.
pub(crate) fn have_i8mm_layout() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        have_i8mm()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Whether this core has the i8mm extension, cached after the first probe.
///
/// `TURBOVEC_NO_I8MM` forces the `SDOT` kernel instead, which lets the two
/// be A/B'd from a single binary — the same escape hatch [`crate::pack`]
/// gives the vector-major layout, and for the same reason: swapping builds
/// between arms measures the compiler alongside the kernel.
#[cfg(target_arch = "aarch64")]
pub(crate) fn have_i8mm() -> bool {
    static I8MM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *I8MM.get_or_init(|| {
        std::env::var_os("TURBOVEC_NO_I8MM").is_none()
            && std::arch::is_aarch64_feature_detected!("i8mm")
    })
}

/// Reshape a batch's query weights into `SMMLA` A-operands, once per tile.
///
/// Each 16-byte entry is one query *pair* over one byte-group quad: bytes
/// 0..8 are the even query's eight dimensions `8*q4 .. 8*q4+8` in dimension
/// order, bytes 8..16 the odd query's. That is exactly the 2x8 row-major
/// operand, so the inner loop loads it with a single `LDR` instead of
/// rebuilding it 4x per block (once per quarter) from the interleaved
/// [`QueryPermuteDot::weights`] layout.
///
/// Indexed quad-major so the pairs a single `q4` iteration wants are
/// contiguous. Costs `NQ * dim` bytes of shuffling per tile and is reused
/// across every block in that tile.
#[cfg(target_arch = "aarch64")]
fn build_smmla_a<const NQ: usize>(pds: &[&QueryPermuteDot; NQ], quads: usize) -> Vec<i8> {
    let pairs = NQ / 2;
    let mut a = vec![0i8; quads * pairs * 16];
    for q4 in 0..quads {
        for p in 0..pairs {
            let dst = (q4 * pairs + p) * 16;
            for (r, pd) in pds[2 * p..2 * p + 2].iter().enumerate() {
                let w = &pd.weights[q4 * 8..q4 * 8 + 8];
                for j in 0..4 {
                    // weights holds [4 lo][4 hi] per quad, and the high
                    // nibble is the *even* dimension, so dimension order
                    // within the quad is hi(0), lo(0), hi(1), lo(1), ...
                    a[dst + r * 8 + 2 * j] = w[4 + j];
                    a[dst + r * 8 + 2 * j + 1] = w[j];
                }
            }
        }
    }
    a
}

/// `SMMLA` A-operands for the `vm8` layout, once per tile.
///
/// `vm8` makes the TBL output the B operand directly, but the dimensions it
/// carries are the eight *even* ones of a byte-group run (from the high
/// nibbles) or the eight *odd* ones (low nibbles) — not eight consecutive.
/// `SMMLA` sums over whatever pairing A and B agree on, so the fix is to
/// build A in that same order rather than to reorder B, which is the whole
/// point: the ZIPs go away.
///
/// Entry `(q8 * pairs + p) * 32` holds the even-dim operand for query pair
/// `p` over byte-groups `8*q8 .. 8*q8+8` in its first 16 bytes and the
/// odd-dim operand in its second 16.
#[cfg(target_arch = "aarch64")]
fn build_smmla_a_vm8<const NQ: usize>(pds: &[&QueryPermuteDot; NQ], octs: usize) -> Vec<i8> {
    let pairs = NQ / 2;
    let mut a = vec![0i8; octs * pairs * 32];
    for q8 in 0..octs {
        for p in 0..pairs {
            let dst = (q8 * pairs + p) * 32;
            for (r, pd) in pds[2 * p..2 * p + 2].iter().enumerate() {
                for j in 0..8 {
                    // Byte-group 8*q8+j lives in quad (8*q8+j)/4 at slot
                    // (8*q8+j)%4 of `weights`, which stores [4 lo][4 hi].
                    let g = 8 * q8 + j;
                    let (q4, slot) = (g / 4, g % 4);
                    // High nibble is the even dim, low the odd.
                    a[dst + r * 8 + j] = pd.weights[q4 * 8 + 4 + slot];
                    a[dst + 16 + r * 8 + j] = pd.weights[q4 * 8 + slot];
                }
            }
        }
    }
    a
}

/// One-query `vm8` scan, whole block, 16 accumulators live.
///
/// A lone query rides SMMLA as a duplicated pair — lanes 2/3 of every tile
/// repeat lanes 0/1 — so half the MACs are thrown away. That waste is not
/// what costs: the batched kernel and this one issue the same instructions
/// per byte of codes.
///
/// What costs is **instruction-level parallelism**. SMMLA has latency 3 and
/// needs several independent accumulator chains to stay fed. The batched
/// kernel holds 16 (4 query pairs x 4 vector pairs) and saturates; routing
/// one query through it at `NQ=2, NP=1` leaves **two**, and the loop goes
/// latency-bound. Measured: 6.19 ms against the classic LUT kernel's 4.15 at
/// nq=1, despite an identical instruction count. See H42.
///
/// With a single query pair the register budget is nearly empty, so this
/// keeps a separate accumulator for all 16 vector pairs of the block —
/// 16 chains, the same ILP the batched kernel gets, and 18 of 32 registers.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_block_vm8_single(
    blocked_codes: &[u8],
    pd: &QueryPermuteDot,
    a_buf: &[i8],
    block_offset: usize,
    n_byte_groups: usize,
    vec_scales: &[f32],
    base_vec: usize,
    n_vectors: usize,
    out: &mut [[f32; BLOCK]; 1],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let levels = vld1q_s8(pd.levels.as_ptr());
    let octs = n_byte_groups / 8;
    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let a_base = a_buf.as_ptr();

    // acc[r] covers block lanes 2r, 2r+1 — the whole block, so every chain
    // is independent for the entire scan.
    let mut acc = [vdupq_n_s32(0); 16];
    for q8 in 0..octs {
        let ap = a_base.add(q8 * 32);
        let ae = vld1q_s8(ap);
        let ao = vld1q_s8(ap.add(16));
        // Four loads in flight, not sixteen. All 16 accumulators stay live
        // — H44 showed eight chains stall, since SMMLA is latency 3 on four
        // pipes and wants ~12 — but LLVM was hoisting all sixteen code
        // loads alongside them and spilling (13 `stp` per loop). Consuming
        // each group of four before the next is loaded caps the transient
        // pressure at 16 + 4 rather than 16 + 16. See H45.
        for g in 0..4 {
            let mut cs = [vdupq_n_u8(0); 4];
            for (j, c) in cs.iter_mut().enumerate() {
                *c = vld1q_u8(codes_base.add(q8 * 256 + (g * 4 + j) * 16));
            }
            for (j, &c) in cs.iter().enumerate() {
                let r = g * 4 + j;
                let bo = vqtbl1q_s8(levels, vandq_u8(c, mask));
                let be = vqtbl1q_s8(levels, vshrq_n_u8(c, 4));
                acc[r] = smmla(acc[r], ae, be);
                acc[r] = smmla(acc[r], ao, bo);
            }
        }
    }

    // A operand is the query duplicated, so lane 0 is this query against
    // vector 2r and lane 1 against 2r+1; lanes 2/3 repeat them.
    let vs = vdupq_n_f32(pd.scale);
    let vb = vdupq_n_f32(pd.bias);
    let end = (base_vec + BLOCK).min(n_vectors);
    let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);
    let mut raw = [0.0f32; BLOCK];
    for i in 0..8 {
        let t = vcombine_s32(vget_low_s32(acc[i * 2]), vget_low_s32(acc[i * 2 + 1]));
        vst1q_f32(raw.as_mut_ptr().add(i * 4), vfmaq_f32(vb, vcvtq_f32_s32(t), vs));
    }
    let op = out[0].as_mut_ptr();
    if end - base_vec == BLOCK {
        for i in 0..8 {
            let f = vld1q_f32(raw.as_ptr().add(i * 4));
            let n = vld1q_f32(vec_scales_ptr.add(i * 4));
            vst1q_f32(op.add(i * 4), vmulq_f32(f, n));
        }
    } else {
        for lane in 0..BLOCK {
            *op.add(lane) = if lane < end - base_vec {
                raw[lane] * *vec_scales_ptr.add(lane)
            } else {
                f32::NEG_INFINITY
            };
        }
    }
}

/// `SMMLA` scan of one 32-vector block over the **`vm8`** layout — the
/// ZIP-free variant (H41).
///
/// Identical arithmetic to [`score_block_permute_smmla_neon`], two fewer
/// instructions per code register. `vm8` puts two vectors x eight
/// byte-groups in each 16-byte register, so after the nibble split the TBL
/// output *is* an `SMMLA` B operand: bytes 0-7 are vector `2r`'s eight even
/// dimensions and 8-15 are vector `2r+1`'s. No reorder, because the operand
/// arrived in operand shape.
///
/// The dimensions are the even eight (high nibbles) and the odd eight (low
/// nibbles) rather than eight consecutive; `build_smmla_a_vm8` builds A to
/// match. P23 priced the two ZIPs at x1.12.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_block_smmla_vm8<const NQ: usize, const NP: usize>(
    blocked_codes: &[u8],
    pds: &[&QueryPermuteDot; NQ],
    a_buf: &[i8],
    block_offset: usize,
    n_byte_groups: usize,
    vec_scales: &[f32],
    base_vec: usize,
    n_vectors: usize,
    out: &mut [[f32; BLOCK]; NQ],
) {
    use std::arch::aarch64::*;
    const { assert!(NQ == 2 * NP, "SMMLA tiles queries in pairs") };

    let pairs = NP;
    let mask = vdupq_n_u8(0x0F);
    let levels = vld1q_s8(pds[0].levels.as_ptr());
    let octs = n_byte_groups / 8;
    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let a_base = a_buf.as_ptr();

    let mut raw = [[0.0f32; BLOCK]; NQ];
    // Eighth-blocks, not quarters. `vm8` needs *two* A operands per query
    // pair — the even-dim and odd-dim halves — where the 4-group kernel
    // needed one, so at NQ=8 the A registers go from 4 to 8. Quarter-blocks
    // would hold 16 accumulators on top of that and spill: the first cut of
    // this kernel did exactly that, and `objdump` showed two accumulators
    // going to the stack every iteration, which ate the win the ZIPs paid
    // for. Four lanes per part halves the accumulators to 8, and 8 + 8 plus
    // the level table, mask and three transients fits. See H41.
    for part in 0..8 {
        // acc[p][r]: query pair `p` against the vector pair in register
        // `part*2 + r`, i.e. block lanes `part*4 + r*2` and `+1`.
        let mut acc = [[vdupq_n_s32(0); 2]; NP];

        for q8 in 0..octs {
            let ap = a_base.add(q8 * pairs * 32);
            let mut ae = [vdupq_n_s8(0); NP];
            let mut ao = [vdupq_n_s8(0); NP];
            for p in 0..pairs {
                ae[p] = vld1q_s8(ap.add(p * 32));
                ao[p] = vld1q_s8(ap.add(p * 32 + 16));
            }
            // Deep prefetch for the batched (nq=100) pattern. H48 refuted
            // arm prefetch at **nq=1 only**, and P29 closed nq=100 on a
            // compute-bound reading that P35 has since undercut: the PMU
            // attributes 18.4% of nq=100 cycles to memory stalls. H62 showed
            // a 32-unit lookahead is worth +25% on x86's 12.5-sweep pattern,
            // which arm nq=100 shares. See H67.
            {
                let pf = q8 * 256 + 32 * 256;
                if block_offset + pf + 64 <= blocked_codes.len() {
                    std::arch::asm!(
                        "prfm pldl1keep, [{p}]",
                        p = in(reg) codes_base.add(pf),
                        options(nostack, readonly, preserves_flags),
                    );
                }
            }
            for r in 0..2 {
                let c = vld1q_u8(codes_base.add(q8 * 256 + (part * 2 + r) * 16));
                // Already B operands: no ZIP.
                let bo = vqtbl1q_s8(levels, vandq_u8(c, mask));
                let be = vqtbl1q_s8(levels, vshrq_n_u8(c, 4));
                for p in 0..pairs {
                    acc[p][r] = smmla(acc[p][r], ae[p], be);
                    acc[p][r] = smmla(acc[p][r], ao[p], bo);
                }
            }
        }

        // Same 2x2 scatter as the classic kernel: lanes 0/1 of a tile belong
        // to the even query of the pair, 2/3 to the odd.
        for p in 0..pairs {
            for (r, q) in [2 * p, 2 * p + 1].into_iter().enumerate() {
                let vs = vdupq_n_f32(pds[q].scale);
                let vb = vdupq_n_f32(pds[q].bias);
                let (x, y) = (acc[p][0], acc[p][1]);
                let t = if r == 0 {
                    vcombine_s32(vget_low_s32(x), vget_low_s32(y))
                } else {
                    vcombine_s32(vget_high_s32(x), vget_high_s32(y))
                };
                let f = vfmaq_f32(vb, vcvtq_f32_s32(t), vs);
                vst1q_f32(raw[q].as_mut_ptr().add(part * 4), f);
            }
        }
    }

    let end = (base_vec + BLOCK).min(n_vectors);
    let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);
    for q in 0..NQ {
        let op = out[q].as_mut_ptr();
        if end - base_vec == BLOCK {
            for i in 0..8 {
                let f = vld1q_f32(raw[q].as_ptr().add(i * 4));
                let n = vld1q_f32(vec_scales_ptr.add(i * 4));
                vst1q_f32(op.add(i * 4), vmulq_f32(f, n));
            }
        } else {
            for lane in 0..BLOCK {
                *op.add(lane) = if lane < end - base_vec {
                    raw[q][lane] * *vec_scales_ptr.add(lane)
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
    }
}

/// Permute-dot scan of one 32-vector block using `SMMLA`, for `NQ` queries
/// at once (4-bit codes, `NQ` even).
///
/// The i8mm half of the permute-dot family (H33). Everything up to the two
/// TBLs is shared with [`score_block_permute_dot_neon`]; the difference is
/// what consumes them. `SDOT` reduces four products into one lane, so it
/// takes one query against four vectors. `SMMLA` reduces eight, arranged as
/// a 2x2 outer tile, so it takes *two* queries against *two* vectors —
/// twice the MACs for the same issue slot.
///
/// The unpacked nibbles fall into the 8x2 operand for free. `vlo`/`vhi`
/// byte `4u+v` is vector `4k+u`'s dimension `8*q4 + 2v+1` / `2v`, so
/// `vzip1q_s8(vhi, vlo)` lays vectors `4k` and `4k+1` down as eight
/// consecutive dimensions each — column-major, which is what `SMMLA`
/// wants — and `vzip2q_s8` does the same for `4k+2` and `4k+3`. Two extra
/// ZIPs per code register buy 8 SMMLA in place of 16 SDOT at NQ=8.
///
/// Register budget on a quarter-block at NQ=8: 16 accumulators (4 query
/// pairs x 4 vector pairs), 4 A-operands, the level table and mask, and
/// five transients — about 26 of 32. That ceiling is what sank H29 and
/// shaped H30/H32, so it is checked before the kernel, not after.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_block_permute_smmla_neon<const NQ: usize, const NP: usize>(
    blocked_codes: &[u8],
    pds: &[&QueryPermuteDot; NQ],
    a_buf: &[i8],
    block_offset: usize,
    n_byte_groups: usize,
    vec_scales: &[f32],
    base_vec: usize,
    n_vectors: usize,
    out: &mut [[f32; BLOCK]; NQ],
) {
    use std::arch::aarch64::*;
    // `NP` is `NQ / 2` spelled out: Rust will not let a const generic be
    // divided in an array length, and the accumulator tile must stay a
    // compile-time constant or it lands on the stack instead of in
    // registers.
    const { assert!(NQ == 2 * NP, "SMMLA tiles queries in pairs") };

    let pairs = NP;
    let mask = vdupq_n_u8(0x0F);
    let levels = vld1q_s8(pds[0].levels.as_ptr());
    let quads = n_byte_groups / 4;
    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let a_base = a_buf.as_ptr();

    let mut raw = [[0.0f32; BLOCK]; NQ];
    for part in 0..4 {
        // acc[p][w]: query pair `p` against vector pair `w`, where `w`
        // covers block lanes `part*8 + w*2` and `+1`.
        let mut acc = [[vdupq_n_s32(0); 4]; NP];

        for q4 in 0..quads {
            let ap = a_base.add(q4 * pairs * 16);
            let mut a = [vdupq_n_s8(0); NP];
            for (p, aq) in a.iter_mut().enumerate() {
                *aq = vld1q_s8(ap.add(p * 16));
            }
            for i in 0..2 {
                let c = vld1q_u8(codes_base.add(q4 * 128 + (part * 2 + i) * 16));
                let vlo = vqtbl1q_s8(levels, vandq_u8(c, mask));
                let vhi = vqtbl1q_s8(levels, vshrq_n_u8(c, 4));
                // Dimension-ordered columns: vectors 4k,4k+1 then 4k+2,4k+3.
                let b0 = vzip1q_s8(vhi, vlo);
                let b1 = vzip2q_s8(vhi, vlo);
                for p in 0..pairs {
                    acc[p][i * 2] = smmla(acc[p][i * 2], a[p], b0);
                    acc[p][i * 2 + 1] = smmla(acc[p][i * 2 + 1], a[p], b1);
                }
            }
        }

        // Scatter the 2x2 tiles. Lanes 0/1 of a tile belong to the even
        // query of the pair and lanes 2/3 to the odd one, so pulling the
        // low halves of two adjacent tiles together rebuilds four
        // consecutive block lanes for one query.
        for p in 0..pairs {
            for (r, q) in [2 * p, 2 * p + 1].into_iter().enumerate() {
                let vs = vdupq_n_f32(pds[q].scale);
                let vb = vdupq_n_f32(pds[q].bias);
                for h in 0..2 {
                    let (x, y) = (acc[p][h * 2], acc[p][h * 2 + 1]);
                    let t = if r == 0 {
                        vcombine_s32(vget_low_s32(x), vget_low_s32(y))
                    } else {
                        vcombine_s32(vget_high_s32(x), vget_high_s32(y))
                    };
                    let f = vfmaq_f32(vb, vcvtq_f32_s32(t), vs);
                    vst1q_f32(raw[q].as_mut_ptr().add(part * 8 + h * 4), f);
                }
            }
        }
    }

    let end = (base_vec + BLOCK).min(n_vectors);
    let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);
    for q in 0..NQ {
        let op = out[q].as_mut_ptr();
        if end - base_vec == BLOCK {
            for i in 0..8 {
                let f = vld1q_f32(raw[q].as_ptr().add(i * 4));
                let n = vld1q_f32(vec_scales_ptr.add(i * 4));
                vst1q_f32(op.add(i * 4), vmulq_f32(f, n));
            }
        } else {
            for lane in 0..BLOCK {
                *op.add(lane) = if lane < end - base_vec {
                    raw[q][lane] * *vec_scales_ptr.add(lane)
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
    }
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

    // The block is scored in quarters, 8 vectors at a time.
    //
    // A whole block needs 8 accumulator registers per query, which at 4
    // queries is 32 — the entire NEON register file, before the level table,
    // the mask, the code register, the two TBL results and 8 weight
    // registers. That spills, and the spills cost more than the fused batch
    // saves: the kernel measured 1.60 SDOT/cycle against a 3.11/cycle
    // independent-issue ceiling. Half a block is 4 accumulators per query,
    // 16 at NQ=4, which leaves room for the rest of the working set.
    //
    // The two halves read disjoint 64-byte runs of each 128-byte
    // vector-major unit, so each byte of the block is still read exactly
    // once across the pair. See LOG_search.md H30.
    let mut raw = [[0.0f32; BLOCK]; NQ];
    for part in 0..4 {
        // acc[q][i]: four vectors per register, covering block lanes
        // (half*4 + i)*4 .. +4.
        let mut acc = [[vdupq_n_s32(0); 2]; NQ];

        // Two byte-group quads per iteration: their weights are 16
        // contiguous bytes, so one register per query holds all four 4-byte
        // groups and `SDOT` selects between them by index. That is 4 weight
        // registers at NQ=4 instead of 8, and one load instead of four.
        // Lane order within the register follows `weights`: lo(q4),
        // hi(q4), lo(q4+1), hi(q4+1).
        let mut q4 = 0usize;
        while q4 < quads {
            let paired = q4 + 1 < quads;
            let mut w = [vdupq_n_s8(0); NQ];
            for (wq, pd) in w.iter_mut().zip(pds.iter()) {
                let wp = pd.weights.as_ptr().add(q4 * 8);
                *wq = if paired {
                    vld1q_s8(wp)
                } else {
                    // Odd final quad: only its own 8 bytes exist, and lanes
                    // 2/3 are never indexed below.
                    vcombine_s8(vld1_s8(wp), vdup_n_s8(0))
                };
            }
            for i in 0..2 {
                let c = vld1q_u8(codes_base.add(q4 * 128 + (part * 2 + i) * 16));
                // Shared across the whole batch — this is the whole point.
                let vlo = vqtbl1q_s8(levels, vandq_u8(c, mask));
                let vhi = vqtbl1q_s8(levels, vshrq_n_u8(c, 4));
                for q in 0..NQ {
                    acc[q][i] = sdot_lane::<0>(acc[q][i], vlo, w[q]);
                    acc[q][i] = sdot_lane::<1>(acc[q][i], vhi, w[q]);
                }
            }
            if paired {
                for i in 0..2 {
                    let c = vld1q_u8(codes_base.add((q4 + 1) * 128 + (part * 2 + i) * 16));
                    let vlo = vqtbl1q_s8(levels, vandq_u8(c, mask));
                    let vhi = vqtbl1q_s8(levels, vshrq_n_u8(c, 4));
                    for q in 0..NQ {
                        acc[q][i] = sdot_lane::<2>(acc[q][i], vlo, w[q]);
                        acc[q][i] = sdot_lane::<3>(acc[q][i], vhi, w[q]);
                    }
                }
            }
            q4 += 2;
        }

        // Convert this half out of the register file before the next one
        // claims it. `vec_scales` and padding are applied once at the end,
        // over both halves together.
        for q in 0..NQ {
            let vs = vdupq_n_f32(pds[q].scale);
            let vb = vdupq_n_f32(pds[q].bias);
            for i in 0..2 {
                let f = vfmaq_f32(vb, vcvtq_f32_s32(acc[q][i]), vs);
                vst1q_f32(raw[q].as_mut_ptr().add((part * 2 + i) * 4), f);
            }
        }
    }

    // Padding lanes get NEG_INFINITY so callers can take a whole-block max
    // without seeing garbage — same contract as the classic NEON kernel.
    let end = (base_vec + BLOCK).min(n_vectors);
    let vec_scales_ptr = vec_scales.as_ptr().add(base_vec);
    for q in 0..NQ {
        let op = out[q].as_mut_ptr();
        if end - base_vec == BLOCK {
            for i in 0..8 {
                let f = vld1q_f32(raw[q].as_ptr().add(i * 4));
                let n = vld1q_f32(vec_scales_ptr.add(i * 4));
                vst1q_f32(op.add(i * 4), vmulq_f32(f, n));
            }
        } else {
            for lane in 0..BLOCK {
                *op.add(lane) = if lane < end - base_vec {
                    raw[q][lane] * *vec_scales_ptr.add(lane)
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
        // Built once per scan, not per block: the reshape is O(dim) while the
        // loop below is O(n_blocks * dim). `None` means this index is not in
        // the vm8 layout, so the classic kernel applies.
        let vm8_single: Option<Vec<i8>> = lut.pd.as_ref().and_then(|pd| {
            // `pd` is built only at 4 bits, which is the width this helper
            // sees; `bits` is not in scope in this nested fn.
            crate::pack::vm8_for(4, n_byte_groups)
                .then(|| build_smmla_a_vm8::<2>(&[pd, pd], n_byte_groups / 8))
        });
        // H101 tried a `prfm` lookahead here, on the grounds that x86's nq=1
        // path has had one since H59/H62 and aarch64 never did, and that this
        // cell runs at 137.3 GB/s against 192.5 available. It does not help:
        // flat at nq=1 MT and ~3% worse at nq=1 ST at every depth tried. The
        // hardware prefetcher already has this stream; the 29% gap is
        // something else.
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
                    if let Some(a) = vm8_single.as_deref() {
                        score_block_vm8_single(
                            codes, pd, a, b * block_bytes, n_byte_groups,
                            scales_slice, base, range_vecs, &mut out,
                        );
                    } else {
                        score_block_permute_dot_neon::<1>(
                            codes, &[pd], b * block_bytes, n_byte_groups,
                            scales_slice, base, range_vecs, &mut out,
                        );
                    }
                } else {
                    score_4bit_block_neon(
                        codes, &lut.uint8_luts, b * block_bytes, n_byte_groups,
                        lut.scale, lut.bias, scales_slice, base, range_vecs, &mut out[0],
                    );
                }
            }
            // No whole-block prune here, unlike `neon_block_topk_update`, and
            // H116 measured that the omission costs nothing: adding one was
            // x1.009 at nq=1 ST and x0.987 at MT. This cell is memory-bound
            // (P42: 95% of the single-core streaming roofline), so the scalar
            // lane loop runs inside memory latency that is being paid anyway.
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
        // One range per thread, and H103 measured that this is right rather
        // than merely inherited. Giving rayon 4 or 8 ranges per thread to
        // steal from makes nq=1 MT monotonically *worse* (x0.95, x0.88): each
        // range costs a heap allocation and a `collect`, and shortens the
        // sequential stream the prefetcher is riding. The cell's 9% scaling
        // loss is not steal-starvation.
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
        // Batch width. Permute-dot shares the whole per-register unpack
        // (load, mask, shift, two TBL) across the batch and adds only two
        // SDOT per query, so a wider batch raises the fraction of the SIMD
        // stream doing useful MACs. H29 tried 8 and lost to register
        // spills; quarter-block accumulators (2 per query) and indexed
        // weights (1 register per query per two quads) bring the working
        // set to 24 of 32, which is what makes 8 fit. See LOG_search.md H32.
        let pd_batched = query_luts.first().is_some_and(|l| l.pd.is_some());
        // H84: batch width re-test. H40 refuted 12 and 16 on the 4-group
        // SMMLA kernel, where NQ=12 needed 24 accumulators. vm8's
        // eighth-blocks (H41) need NP*2 = 12, and 100/12 = 9 sweeps over
        // the code array against 12.5 — 28% less traffic, against the 14.2%
        // of cycles P39 still attributes to memory stalls.
        // Batch width, stepped down to a width the dispatch actually has
        // an arm for. `tiles` chunks queries by `qbs`, and any chunk whose
        // size has no `pd_scan!` arm falls to the per-query tail — so a
        // fixed 12 sends nq=10 through as one unbatched chunk of ten.
        // Measured: nq=10 cost 39.83 ms that way against 18.06 ms when the
        // width was 8 (chunks of 8 + 2). Stepping down keeps every nq on a
        // batched path for all but its remainder. See H90.
        let qbs: usize = if !pd_batched {
            4
        } else if nq >= 12 {
            12
        } else if nq >= 8 {
            8
        } else {
            4
        };
        /// Widest batch any path here takes; sizes the per-batch scratch.
        const QBS_MAX: usize = 12;
        /// The LUT kernel's fixed width.
        const QBS_LUT: usize = 4;
        // `.max(1)`: an empty query batch (nq == 0) is a legal no-op —
        // main returns empty results for it — but it would otherwise be
        // the divisor below and panic with a divide-by-zero. The tile
        // loop is empty at nq == 0 either way, so the merge yields the
        // same empty result.
        let n_quads = nq.div_ceil(qbs).max(1);
        let n_threads = rayon::current_num_threads().max(1);
        let n_ranges = n_block_ranges(
            nq, n_quads, n_blocks, n_vectors, k, n_threads,
            TILES_PER_THREAD_NEON,
            // H14: at 2 bits a block is half its 4-bit bytes, so H69's floor
            // of 512 makes ranges too small for the trade it was balancing —
            // the swept optimum is 1024 (18.08 -> 17.70 ms at nq=100 MT, with
            // 256 and 2048 both worse). 4-bit keeps its own measured 512.
            if bits == 2 { MIN_TILE_BLOCKS_NEON * 2 } else { MIN_TILE_BLOCKS_NEON },
            false,
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
            .flat_map(|b| (0..nq).step_by(qbs).map(move |q| (q, b)))
            .collect();

        let tile_results: Vec<(usize, Vec<Vec<(f32, u64)>>)> = tiles
            .into_par_iter()
            .map(|(qi_start, block_start)| {
                let block_end = (block_start + blocks_per_range).min(n_blocks);
                let qi_end = (qi_start + qbs).min(nq);
                let batch_size = qi_end - qi_start;

                // Fused scoring + top-k: no per-quad score matrix. Each block's
                // 32 scores live on the stack and fold straight into the
                // per-query heaps (block-ascending, lane-ascending — the same
                // visit order as the old flat scan, so results are identical).
                let mut heap_s = vec![vec![f32::NEG_INFINITY; k]; batch_size];
                let mut heap_i = vec![vec![0u64; k]; batch_size];
                let mut heap_sz = [0usize; QBS_MAX];
                let mut heap_min = [f32::NEG_INFINITY; QBS_MAX];
                let mut heap_mi = [0usize; QBS_MAX];

                // One fused scan over this tile's blocks for a whole batch
                // of queries. `$n` is a literal so the kernel's accumulator
                // count stays a compile-time constant.
                macro_rules! pd_scan {
                    ($n:literal, $np:literal) => {{
                        let pds: [&QueryPermuteDot; $n] = std::array::from_fn(|i| {
                            query_luts[qi_start + i]
                                .pd
                                .as_ref()
                                .expect("pd built for every query")
                        });
                        // One reshape of the batch's weights, reused by
                        // every block below. Which reshape depends on the
                        // layout in memory, which pack.rs decided at load
                        // or encode time — the two must not disagree.
                        let vm8 = crate::pack::vm8_for(bits, n_byte_groups);
                        let a_buf = if vm8 {
                            build_smmla_a_vm8::<$n>(&pds, n_byte_groups / 8)
                        } else if have_i8mm() {
                            build_smmla_a::<$n>(&pds, n_byte_groups / 4)
                        } else {
                            Vec::new()
                        };
                        let mut block_out = [[0.0f32; BLOCK]; $n];
                        for block_idx in block_start..block_end {
                            let base_vec = block_idx * BLOCK;
                            if !block_has_allowed(mask, base_vec) {
                                continue;
                            }
                            let block_offset = block_idx * n_byte_groups * BLOCK;
                            let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
                            unsafe {
                                if vm8 {
                                    score_block_smmla_vm8::<$n, $np>(
                                        blocked_codes, &pds, &a_buf, block_offset,
                                        n_byte_groups, vec_scales, base_vec, n_vectors,
                                        &mut block_out,
                                    );
                                } else if a_buf.is_empty() {
                                    score_block_permute_dot_neon::<$n>(
                                        blocked_codes, &pds, block_offset, n_byte_groups,
                                        vec_scales, base_vec, n_vectors, &mut block_out,
                                    );
                                } else {
                                    score_block_permute_smmla_neon::<$n, $np>(
                                        blocked_codes, &pds, &a_buf, block_offset,
                                        n_byte_groups, vec_scales, base_vec, n_vectors,
                                        &mut block_out,
                                    );
                                }
                                for q in 0..$n {
                                    neon_block_topk_update(
                                        &block_out[q], base_vec, end_lane, mask, k,
                                        &mut heap_s[q], &mut heap_i[q], &mut heap_sz[q],
                                        &mut heap_min[q], &mut heap_mi[q],
                                    );
                                }
                            }
                        }
                    }};
                }

                if pd_batched && batch_size == 12 {
                    pd_scan!(12, 6)
                } else if pd_batched && batch_size == 8 {
                    pd_scan!(8, 4)
                } else if pd_batched && batch_size == 4 {
                    // A tail landing on the narrower width still gets a
                    // fused scan rather than one query at a time.
                    pd_scan!(4, 2)
                } else if !pd_batched && batch_size == QBS_LUT {
                    // Fast path: 4-query fused LUT kernel
                    let lut_refs: [&[u8]; QBS_LUT] = [
                        &query_luts[qi_start].uint8_luts,
                        &query_luts[qi_start + 1].uint8_luts,
                        &query_luts[qi_start + 2].uint8_luts,
                        &query_luts[qi_start + 3].uint8_luts,
                    ];
                    let scales: [f32; QBS_LUT] = [
                        query_luts[qi_start].scale,
                        query_luts[qi_start + 1].scale,
                        query_luts[qi_start + 2].scale,
                        query_luts[qi_start + 3].scale,
                    ];
                    let biases: [f32; QBS_LUT] = [
                        query_luts[qi_start].bias,
                        query_luts[qi_start + 1].bias,
                        query_luts[qi_start + 2].bias,
                        query_luts[qi_start + 3].bias,
                    ];
                    let mut block_out = [[0.0f32; BLOCK]; QBS_LUT];
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
                            score_4query_block_neon(
                                blocked_codes, lut_refs, block_offset, n_byte_groups,
                                scales, biases, vec_scales, base_vec, n_vectors,
                                &mut block_out,
                            );
                            for q in 0..QBS_LUT {
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
                        let vm8_single: Option<Vec<i8>> = qlut.pd.as_ref().and_then(|pd| {
                            crate::pack::vm8_for(bits, n_byte_groups)
                                .then(|| build_smmla_a_vm8::<2>(&[pd, pd], n_byte_groups / 8))
                        });
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
                                    if let Some(a) = vm8_single.as_deref() {
                                        score_block_vm8_single(
                                            blocked_codes, pd, a, block_offset, n_byte_groups,
                                            vec_scales, base_vec, n_vectors, &mut block_out,
                                        );
                                    } else {
                                        score_block_permute_dot_neon::<1>(
                                            blocked_codes, &[pd], block_offset, n_byte_groups,
                                            vec_scales, base_vec, n_vectors, &mut block_out,
                                        );
                                    }
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
                        let pd_refs = [pd; 1];
                        let args = (
                            codes, &pd_refs,
                            n_byte_groups, scales_slice, range_vecs,
                            1, k, mask_slice,
                        );
                        if have_gfni() {
                            search_multi_query_permute_dot_gfni::<1, 8>(
                                args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7,
                                &mut heap_scores, &mut heap_indices,
                                &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                            );
                        } else {
                            search_multi_query_permute_dot::<1, 8>(
                                args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7,
                                &mut heap_scores, &mut heap_indices,
                                &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                            );
                        }
                    } else if !lut.split.is_empty() {
                        let split_refs = [lut.split.as_slice(); 4];
                        search_multi_query_vnni_dispatch(
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
        //
        // H23 measured 8 at parity (x0.997) and kept 4. Permute-dot voided
        // that result rather than confirming it: H23 was run against the
        // `vpermb` kernel, where each extra query in a batch cost a 128-byte
        // LUT load per byte-group, so widening the batch bought fewer passes
        // at the price of proportionally more table traffic. Permute-dot's
        // per-query cost inside a batch is an 8-byte broadcast, so the
        // passes are now nearly free to amortize. Re-measured at 8:
        // x1.433 single-threaded, x1.116 multi-threaded (H28).
        // Batch width, chosen so the permute-dot kernel's accumulators
        // stay in registers — see H34 and the note on
        // `search_multi_query_permute_dot`.
        // H124: the width is thread-dependent, because the trade it makes is.
        // A wider batch buys fewer passes over the code array and pays more
        // live state per tile. Fewer passes is a single-thread win; more live
        // state is a multi-thread loss, since every worker pays it into a
        // shared cache. H123 measured both halves at once — `NQ_BATCH = 10`
        // was x1.055 at nq=100 ST and x0.966 at MT — so one constant cannot be
        // right for both, which is why H97's sweep found 8 and H121's ARM
        // sweep found 12 with neither able to move.
        //
        // 10 also divides the nq=100 operating point exactly: 10 passes
        // against 8's 13.
        // Gated on 10 actually *saving a pass*, not merely on `nq >= 10`
        // (H135). The wider batch pays for itself only through the pass it
        // removes: a batch scores all its lanes and reports only the queries
        // that exist, so where both widths need the same number of passes the
        // wider one just pads more lanes. H135 measured that directly —
        // nq=16 (2 passes either way) x0.882, nq=24 (3/3) x0.863, nq=32 (4/4)
        // x0.827, against nq=50 (7 passes at 8, 5 at 10) x1.147 and nq=100
        // (13/10) x1.072. The predicate below is exactly the sign of that
        // ratio.
        //
        // Only when the permute-dot kernel is the one taking the batch:
        // it is the only x86 kernel with 10 query lanes. The VNNI kernel
        // that scores 2/3-bit vector-major indexes is 8-wide (`acc` is
        // `[[__m512i; 2]; 8]` and its loops clamp at 8), so a 10-wide
        // batch there scored lanes 0..8 and silently dropped queries 8
        // and 9 — heap sizes stayed 0 and they came back as empty
        // results. `pd` depends only on bits and geometry, so it is
        // uniform across the batch and the first query decides for all.
        // The classic BW/AVX2 arms chunk in 4s and the scalar arm loops
        // per query, so 8 remains safe everywhere else.
        let wide_batch_kernel = query_luts.first().is_some_and(|q| q.pd.is_some());
        let nq_batch: usize = if wide_batch_kernel
            && rayon::current_num_threads().max(1) == 1
            && nq.div_ceil(10) < nq.div_ceil(8)
        {
            10
        } else {
            8
        };
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
            MIN_TILE_BLOCKS_X86,
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
                        // `$n` is a literal so the kernel's accumulator count
                        // stays a compile-time constant; the runtime choice is
                        // only which instantiation to enter.
                        macro_rules! pd_dispatch {
                            ($n:literal) => {{
                                let pd_refs: [&QueryPermuteDot; $n] =
                                    std::array::from_fn(|i| {
                                        let qi = if qi_start + i < qi_end {
                                            qi_start + i
                                        } else {
                                            pad_qi
                                        };
                                        query_luts[qi]
                                            .pd
                                            .as_ref()
                                            .expect("pd built for every query")
                                    });
                                if have_gfni() {
                                    search_multi_query_permute_dot_gfni::<$n, 1>(
                                        codes, &pd_refs,
                                        n_byte_groups, scales_slice, range_vecs,
                                        batch_nq, k, mask,
                                        &mut heap_scores, &mut heap_indices,
                                        &mut heap_sizes, &mut heap_mins,
                                        &mut heap_min_idxs,
                                    );
                                } else {
                                    search_multi_query_permute_dot::<$n, 1>(
                                        codes, &pd_refs,
                                        n_byte_groups, scales_slice, range_vecs,
                                        batch_nq, k, mask,
                                        &mut heap_scores, &mut heap_indices,
                                        &mut heap_sizes, &mut heap_mins,
                                        &mut heap_min_idxs,
                                    );
                                }
                            }};
                        }
                        if nq_batch == 10 {
                            pd_dispatch!(10)
                        } else {
                            pd_dispatch!(8)
                        }
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
                        search_multi_query_vnni_dispatch(
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
                        // Like the AVX2 arm below: the classic BW kernel
                        // holds four queries of state, so the 8-wide batch
                        // is consumed in 4-query chunks (reachable on BW-
                        // without-VNNI parts, e.g. Cascade Lake).
                        let mut cs = 0;
                        while cs < batch_nq {
                            let ce = (cs + 4).min(batch_nq);
                            // The kernel's prologue reads 4 lanes of
                            // luts/scales/biases unconditionally (its
                            // historical callers padded); pad the chunk
                            // back to 4 — the epilogue writes only
                            // `0..nq`, so the padding lanes are inert.
                            let mut ch_luts = [lut_refs[cs]; 4];
                            let mut ch_scales = [scale_vals[cs]; 4];
                            let mut ch_biases = [bias_vals[cs]; 4];
                            let len = ce - cs;
                            ch_luts[..len].copy_from_slice(&lut_refs[cs..ce]);
                            ch_scales[..len].copy_from_slice(&scale_vals[cs..ce]);
                            ch_biases[..len].copy_from_slice(&bias_vals[cs..ce]);
                            search_multi_query_avx512bw(
                                codes, &ch_luts, &ch_scales, &ch_biases,
                                n_byte_groups, scales_slice, range_vecs,
                                ce - cs, k, mask,
                                &mut heap_scores[cs..ce], &mut heap_indices[cs..ce],
                                &mut heap_sizes[cs..ce], &mut heap_mins[cs..ce],
                                &mut heap_min_idxs[cs..ce],
                            );
                            cs = ce;
                        }
                    } else if !force_scalar
                        && is_x86_feature_detected!("avx2")
                        && is_x86_feature_detected!("fma")
                    {
                        // The AVX2 kernel holds FOUR queries of state
                        // (`fa: [[__m256; 4]; 4]`); NQ_BATCH grew to 8 for
                        // the wide AVX-512 kernels (H28), so on AVX2-only
                        // CPUs the batch is consumed in 4-query chunks —
                        // handing it the full batch indexed out of bounds
                        // (caught on hardware without AVX-512).
                        let mut cs = 0;
                        while cs < batch_nq {
                            let ce = (cs + 4).min(batch_nq);
                            // The kernel's prologue reads 4 lanes of
                            // luts/scales/biases unconditionally (its
                            // historical callers padded); pad the chunk
                            // back to 4 — the epilogue writes only
                            // `0..nq`, so the padding lanes are inert.
                            let mut ch_luts = [lut_refs[cs]; 4];
                            let mut ch_scales = [scale_vals[cs]; 4];
                            let mut ch_biases = [bias_vals[cs]; 4];
                            let len = ce - cs;
                            ch_luts[..len].copy_from_slice(&lut_refs[cs..ce]);
                            ch_scales[..len].copy_from_slice(&scale_vals[cs..ce]);
                            ch_biases[..len].copy_from_slice(&bias_vals[cs..ce]);
                            search_multi_query_avx2(
                                codes, &ch_luts, &ch_scales, &ch_biases,
                                n_byte_groups, scales_slice, range_vecs,
                                ce - cs, k, mask,
                                &mut heap_scores[cs..ce], &mut heap_indices[cs..ce],
                                &mut heap_sizes[cs..ce], &mut heap_mins[cs..ce],
                                &mut heap_min_idxs[cs..ce],
                            );
                            cs = ce;
                        }
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
        // constants: 2x finer target (H15), 2x finer block floor (H69 —
        // was 4x until the kernel got fast enough that fewer, longer
        // contiguous runs beat a finer split).
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(TILES_PER_THREAD_NEON, TILES_PER_THREAD * 2);
            assert_eq!(MIN_TILE_BLOCKS_NEON, MIN_TILE_BLOCKS / 2);
        }
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

        // n_threads == 1 alone. `n_quads` is 1 so the guard is the only
        // thing that can force 1: without it the arithmetic yields
        // `(1 * TILES_PER_THREAD).div_ceil(1) = 32`, capped to 4 by the
        // block cap — either way well above 1. (At the old target of 4
        // a 16-quad row was vacuous, `(1*4).div_ceil(16) == 1` with or
        // without the guard; at 32 it no longer is, but 1 quad keeps
        // the row's margin the widest.) This is the disjunct that fires
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

    /// `split_lut_for_vnni` rearranges [hi_16 | lo_16] per group into the
    /// 128-byte concatenations `vpermb` indexes: all four hi halves of a
    /// group-quad first, then all four lo halves. Pinned byte-for-byte
    /// against an asymmetric ramp over two quads, so any slip in the quad
    /// stride, the half offset, or the per-group base breaks a specific
    /// byte rather than surviving as a shuffle of equal values.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn split_lut_for_vnni_is_the_documented_byte_map() {
        let n_byte_groups = 8usize;
        let src: Vec<u8> = (0..n_byte_groups * 32).map(|i| (i % 253) as u8).collect();
        let out = split_lut_for_vnni(&src, n_byte_groups);
        for g in 0..n_byte_groups {
            let c = (g / 4) * 128;
            let j = g % 4;
            let s = g * 32;
            assert_eq!(
                &out[c + j * 16..c + j * 16 + 16],
                &src[s + 16..s + 32],
                "hi half of group {g}",
            );
            assert_eq!(
                &out[c + 64 + j * 16..c + 64 + j * 16 + 16],
                &src[s..s + 16],
                "lo half of group {g}",
            );
        }
    }

    /// Pin `build_permute_dot`'s weight placement and its accumulator seed.
    /// The seed must be exactly `-128 * Σ w` — that is the whole basis of
    /// the +128 bias cancellation — and each dimension's int8 weight must
    /// land at the documented slot: low nibble (odd dim) at `q4*8+j`, high
    /// (even dim) at `q4*8+4+j`.
    #[test]
    fn permute_dot_weights_land_at_their_slots_and_seed_the_bias() {
        let dim = 16usize; // 8 byte-groups, 2 quads
        let n_byte_groups = dim / 2;
        // Asymmetric, sign-mixed row so every slot is distinct and the
        // quantizer's scale is exercised away from 1.
        let q_rot_row: Vec<f32> = (0..dim).map(|d| (d as f32 - 5.5) * 0.11).collect();
        let centroids: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) / 8.0).collect();
        let pd = build_permute_dot(&q_rot_row, &centroids, dim);

        let qmax = q_rot_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let inv_qs = 1.0 / (qmax / 127.0); // the impl's own expression, ulp-exact
        for g in 0..n_byte_groups {
            let (q4, j) = (g / 4, g % 4);
            let lo = (q_rot_row[2 * g + 1] * inv_qs).round().clamp(-127.0, 127.0) as i8;
            let hi = (q_rot_row[2 * g] * inv_qs).round().clamp(-127.0, 127.0) as i8;
            assert_eq!(pd.weights[q4 * 8 + j], lo, "low-nibble weight of group {g}");
            assert_eq!(pd.weights[q4 * 8 + 4 + j], hi, "high-nibble weight of group {g}");
        }
        let wsum: i32 = pd.weights.iter().map(|&w| w as i32).sum();
        assert_eq!(pd.zero, -128 * wsum, "accumulator seed must cancel the +128 bias");
    }

    /// The SMMLA A-operand builders are pure shuffles of
    /// [`QueryPermuteDot::weights`]; pin them against the documented maps.
    /// aarch64-only because the kernels that read them are.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn smmla_a_operands_match_the_weight_maps() {
        let dim = 32usize; // 16 byte-groups: 4 quads, 2 octs
        let q0: Vec<f32> = (0..dim).map(|d| (d as f32 - 9.5) * 0.07).collect();
        let q1: Vec<f32> = (0..dim).map(|d| (14.5 - d as f32) * 0.05).collect();
        let centroids: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) / 8.0).collect();
        let pd0 = build_permute_dot(&q0, &centroids, dim);
        let pd1 = build_permute_dot(&q1, &centroids, dim);
        let pds: [&QueryPermuteDot; 2] = [&pd0, &pd1];

        let quads = dim / 8;
        let a = build_smmla_a::<2>(&pds, quads);
        for q4 in 0..quads {
            for (r, pd) in pds.iter().enumerate() {
                let dst = q4 * 16; // one pair
                let w = &pd.weights[q4 * 8..q4 * 8 + 8];
                for j in 0..4 {
                    assert_eq!(a[dst + r * 8 + 2 * j], w[4 + j], "even dim, quad {q4} row {r} j {j}");
                    assert_eq!(a[dst + r * 8 + 2 * j + 1], w[j], "odd dim, quad {q4} row {r} j {j}");
                }
            }
        }

        let octs = dim / 16;
        let a8 = build_smmla_a_vm8::<2>(&pds, octs);
        for q8 in 0..octs {
            for (r, pd) in pds.iter().enumerate() {
                let dst = q8 * 32; // one pair
                for j in 0..8 {
                    let g = 8 * q8 + j;
                    let (q4, slot) = (g / 4, g % 4);
                    assert_eq!(a8[dst + r * 8 + j], pd.weights[q4 * 8 + 4 + slot], "even dim, oct {q8} row {r} j {j}");
                    assert_eq!(a8[dst + 16 + r * 8 + j], pd.weights[q4 * 8 + slot], "odd dim, oct {q8} row {r} j {j}");
                }
            }
        }
    }
}
