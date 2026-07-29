//! Differential simulation of the AVX-512BW paired-block kernel's
//! batch/tail structure against the NEON reference (issue #314).
//!
//! AVX-512 cannot be executed on the CI/dev ARM hosts, and CI has no AVX-512
//! leg at all, so these tests model the *semantics* of the two kernels'
//! accumulate/flush schedules in scalar arithmetic rather than running them.
//! What matters for correctness is which byte-groups land in which u16
//! accumulation batch, because every batch boundary is an f32 `fmadd` flush:
//! a schedule that drops groups scores nothing, and a schedule that merely
//! flushes at different points diverges from ARM in the last f32 ulp.
//!
//! The plan builders below mirror the loop bounds in
//! `search.rs::score_4bit_block_neon` and
//! `search.rs::search_multi_query_avx512bw` line for line. If those bounds
//! change, these must change with them.

const FLUSH_EVERY: usize = 256;

/// Groups visited per accumulation batch by `score_4bit_block_neon`, which is
/// also the AVX2 kernel's schedule and (post-#314) the AVX-512 kernel's.
fn neon_plan(n_byte_groups: usize) -> Vec<Vec<usize>> {
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;
    (0..n_batches)
        .map(|batch| {
            let g_start = batch * FLUSH_EVERY;
            let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
            (g_start..g_end).collect()
        })
        .collect()
}

/// Post-fix AVX-512 schedule: batch over byte-groups by FLUSH_EVERY, consume
/// two groups per inner iteration, and take any odd trailing group of *this*
/// batch as a single-group tail.
fn avx512_plan(n_byte_groups: usize) -> Vec<Vec<usize>> {
    assert_eq!(FLUSH_EVERY % 2, 0, "pair-stepped inner loop needs even batches");
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;
    (0..n_batches)
        .map(|batch| {
            let g_start = batch * FLUSH_EVERY;
            let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
            let mut groups = Vec::new();
            let mut g_pair = g_start;
            while g_pair + 1 < g_end {
                groups.push(g_pair);
                groups.push(g_pair + 1);
                g_pair += 2;
            }
            groups.extend(g_pair..g_end);
            groups
        })
        .collect()
}

/// Pre-fix AVX-512 schedule, retained so the tests below demonstrate that the
/// two defects in #314 are actually detected rather than merely asserted away.
/// Batching was driven by *group pairs*, and the odd tail was appended to
/// whichever batch happened to be last.
fn avx512_plan_prefix(n_byte_groups: usize) -> Vec<Vec<usize>> {
    let n_group_pairs_inner = n_byte_groups / 2;
    let n_pairs_per_flush = FLUSH_EVERY / 2;
    let n_batches = if n_group_pairs_inner == 0 {
        0
    } else {
        (n_group_pairs_inner + n_pairs_per_flush - 1) / n_pairs_per_flush
    };
    (0..n_batches)
        .map(|batch| {
            let gp_start = batch * n_pairs_per_flush;
            let gp_end = ((batch + 1) * n_pairs_per_flush).min(n_group_pairs_inner);
            let mut groups: Vec<usize> = (gp_start..gp_end).flat_map(|gp| [gp * 2, gp * 2 + 1]).collect();
            if batch == n_batches - 1 {
                groups.extend(n_group_pairs_inner * 2..n_byte_groups);
            }
            groups
        })
        .collect()
}

/// One output lane of a scored block. `luts[g]` is the u8 the group-`g` table
/// lookup produces for this lane; the kernels sum those into a u16 lane
/// accumulator, then flush with `fa = fmadd(scale, acc as f32, fa)` at every
/// batch boundary. Reproduces the arithmetic exactly, including f32 rounding.
fn simulate(plan: &[Vec<usize>], luts: &[u8], scale: f32, bias: f32) -> f32 {
    let mut fa = bias;
    for batch in plan {
        let mut acc: u16 = 0;
        for &g in batch {
            acc = acc.wrapping_add(luts[g] as u16);
        }
        fa = scale.mul_add(acc as f32, fa);
    }
    fa
}

#[test]
fn avx512_batch_schedule_matches_neon() {
    // Legal geometries (n_byte_groups = dim / (8 / bits)) plus every shape
    // that stresses a batch boundary, an odd group count, or both.
    for ng in [
        1usize, 2, 3, 4, 5, 7, 8, 16, 32, 64, 128, 254, 255, 256, 257, 258, 511, 512, 513, 767, 768,
        769, 1024, 1025, 2048,
    ] {
        assert_eq!(
            avx512_plan(ng),
            neon_plan(ng),
            "AVX-512 batch schedule diverges from NEON at n_byte_groups={ng}"
        );
        // Every group is scored exactly once, in order.
        let visited: Vec<usize> = avx512_plan(ng).into_iter().flatten().collect();
        assert_eq!(visited, (0..ng).collect::<Vec<_>>(), "group coverage at ng={ng}");
    }
}

/// #314(1): with `n_byte_groups == 1` the pre-fix kernel produced zero
/// batches, so the paired-block path scored every vector as `bias` alone.
#[test]
fn single_byte_group_is_scored_not_dropped() {
    let luts = [33u8];
    let (scale, bias) = (0.001f32, 0.0f32);

    let neon = simulate(&neon_plan(1), &luts, scale, bias);
    let fixed = simulate(&avx512_plan(1), &luts, scale, bias);
    let prefix = simulate(&avx512_plan_prefix(1), &luts, scale, bias);

    assert_eq!(neon, 0.033);
    assert_eq!(fixed, neon);
    // The defect this guards against: no batches at all, so score == bias.
    assert_eq!(prefix, 0.0);
}

/// #314(2): when `ng % 512 == 1` the last pre-fix batch already held 256
/// groups and the tail appended a 257th, moving the fmadd flush boundary off
/// NEON's and producing a cross-arch f32 tie-break divergence.
#[test]
fn odd_tail_does_not_overfill_a_full_batch() {
    let ng = 257;
    let luts = vec![3u8; ng];
    let (scale, bias) = (5e-5f32, 0.0f32);

    let neon = simulate(&neon_plan(ng), &luts, scale, bias);
    let fixed = simulate(&avx512_plan(ng), &luts, scale, bias);
    let prefix = simulate(&avx512_plan_prefix(ng), &luts, scale, bias);

    // One f32 ulp apart: NEON flushes 256 groups then 1, the pre-fix AVX-512
    // schedule flushed all 257 at once.
    assert_eq!(neon, 0.038549997);
    assert_eq!(fixed, neon);
    assert_eq!(prefix, 0.03855);

    // The pre-fix batch boundary, not an overflow: 257 * 254 < u16::MAX.
    assert_eq!(avx512_plan_prefix(ng)[0].len(), 257);
    assert_eq!(avx512_plan(ng)[0].len(), 256);
    assert_eq!(avx512_plan(ng)[1].len(), 1);
}

/// Randomized differential sweep: for legal and adversarial group counts, the
/// fixed AVX-512 schedule must be bit-identical to NEON on every lane, while
/// the pre-fix schedule must be observably wrong on at least one.
#[test]
fn randomized_differential_against_neon() {
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut prefix_divergences = 0;
    for ng in [1usize, 2, 3, 5, 255, 256, 257, 511, 512, 513, 769, 1024] {
        for _ in 0..64 {
            // max_lut = 127 per nibble, two nibbles per group.
            let luts: Vec<u8> = (0..ng).map(|_| (rng() % 255) as u8).collect();
            let scale = (rng() % 100_000) as f32 * 1e-7 + 1e-7;
            let bias = -((rng() % 1000) as f32) * 1e-3;

            let neon = simulate(&neon_plan(ng), &luts, scale, bias);
            assert_eq!(
                simulate(&avx512_plan(ng), &luts, scale, bias),
                neon,
                "AVX-512 diverges from NEON at ng={ng}"
            );
            if simulate(&avx512_plan_prefix(ng), &luts, scale, bias) != neon {
                prefix_divergences += 1;
            }
        }
    }
    assert!(
        prefix_divergences > 0,
        "pre-fix schedule should be detectably wrong somewhere"
    );
}

/// A u16 lane accumulator must not wrap within a batch. `max_lut = 127` and
/// two nibbles per group give at most 254 per group per lane, so any schedule
/// that keeps batches at FLUSH_EVERY groups stays under 65535 — the argument
/// that justifies the cap in `encode.rs`.
#[test]
fn no_batch_can_overflow_the_u16_lane_accumulator() {
    for ng in [1usize, 255, 256, 257, 512, 1024, 2048, 4096] {
        for batch in avx512_plan(ng) {
            assert!(
                batch.len() * 254 <= u16::MAX as usize,
                "batch of {} groups can overflow u16 at ng={ng}",
                batch.len()
            );
        }
    }
}
