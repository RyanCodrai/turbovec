//! P12 — price arm batch widths with *faithful LUT streaming*.
//!
//! P5 hoisted each query's table into one register, which is the exact
//! simplification that made H12 a surprise: the real kernel loads 32 B of
//! per-query LUT per byte-group, a 6 KB working set per query. This probe
//! keeps that pattern and compares:
//!   qbs4   — four queries, all 192 groups per block pass (shipped shape)
//!   qbs8   — eight queries, all 192 groups (H12's shape, 48 KB LUT set)
//!   qbs8db — eight queries, two 96-group half passes (24 KB per pass —
//!            the audit's dimension-blocking; u16 partials carried in
//!            registers across halves within a block)
//!
//! All three walk the same blocked code array (200k x 192 B) block-major,
//! per-block accumulate, black_box sink. nq=8 total queries throughout, so
//! qbs4 makes two passes over the codes and the qbs8 variants make one —
//! the pass-count trade is part of what is being priced.
//!
//! Usage: probe_2bit_lutstream [reps]

#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "aarch64")]
fn main() {
    use std::arch::aarch64::*;
    use std::hint::black_box;
    use std::time::Instant;

    let reps: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(12);

    const N_BLOCKS: usize = 6250; // 200k vectors
    const GROUPS: usize = 192;    // dim 768 at 2 bits
    const BLOCK: usize = 32;
    let block_bytes = GROUPS * BLOCK;

    let mut codes = vec![0u8; N_BLOCKS * block_bytes];
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for b in codes.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }
    // 8 queries x 192 groups x 32 B of LUT, the real footprint.
    let luts: Vec<Vec<u8>> = (0..8)
        .map(|q| (0..GROUPS * 32).map(|i| ((i * 7 + q * 13) % 127) as u8).collect())
        .collect();

    // One block for `nq` queries over groups [g0, g1); accumulators u16.
    #[inline(always)]
    unsafe fn scan_block<const NQ: usize>(
        codes: *const u8,
        luts: &[Vec<u8>],
        qoff: usize,
        g0: usize,
        g1: usize,
        acc: &mut [[uint16x8_t; 4]; NQ],
    ) {
        let mask = vdupq_n_u8(0x0F);
        for g in g0..g1 {
            let cp = codes.add(g * 32);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            let lo0 = vandq_u8(c0, mask);
            let hi0 = vshrq_n_u8(c0, 4);
            let lo1 = vandq_u8(c1, mask);
            let hi1 = vshrq_n_u8(c1, 4);
            for q in 0..NQ {
                let lp = luts[qoff + q].as_ptr().add(g * 32);
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
    }

    let run = |name: &str, f: &dyn Fn() -> u64| {
        black_box(f());
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            black_box(f());
            best = best.min(t0.elapsed().as_secs_f64());
        }
        // 8 queries x 200k vectors x 768 dims per full evaluation
        let work = 8.0 * 200_000.0 * 768.0;
        println!("{name}: {:.2} ms  {:.2} G(q.dim)/s", best * 1e3, work / best / 1e9);
    };

    let codes_ref = &codes;
    let luts_ref = &luts;

    run("qbs4  ", &|| unsafe {
        let mut sink = 0u64;
        for pass in 0..2 {
            for b in 0..N_BLOCKS {
                let mut acc = [[vdupq_n_u16(0); 4]; 4];
                scan_block::<4>(codes_ref.as_ptr().add(b * GROUPS * 32), luts_ref, pass * 4, 0, GROUPS, &mut acc);
                for q in 0..4 { for i in 0..4 { sink = sink.wrapping_add(vaddvq_u16(acc[q][i]) as u64); } }
            }
        }
        sink
    });
    run("qbs8  ", &|| unsafe {
        let mut sink = 0u64;
        for b in 0..N_BLOCKS {
            let mut acc = [[vdupq_n_u16(0); 4]; 8];
            scan_block::<8>(codes_ref.as_ptr().add(b * GROUPS * 32), luts_ref, 0, 0, GROUPS, &mut acc);
            for q in 0..8 { for i in 0..4 { sink = sink.wrapping_add(vaddvq_u16(acc[q][i]) as u64); } }
        }
        sink
    });
    run("qbs8db", &|| unsafe {
        let mut sink = 0u64;
        for b in 0..N_BLOCKS {
            let mut acc = [[vdupq_n_u16(0); 4]; 8];
            scan_block::<8>(codes_ref.as_ptr().add(b * GROUPS * 32), luts_ref, 0, 0, GROUPS / 2, &mut acc);
            scan_block::<8>(codes_ref.as_ptr().add(b * GROUPS * 32), luts_ref, 0, GROUPS / 2, GROUPS, &mut acc);
            for q in 0..8 { for i in 0..4 { sink = sink.wrapping_add(vaddvq_u16(acc[q][i]) as u64); } }
        }
        sink
    });

    // qbs4x2: four queries, two blocks per LUT load. The shipped kernel
    // re-reads each query's 32 B table for every block, so one load serves
    // 32 vectors; pairing blocks makes it serve 64. Accumulators stay at 16
    // registers by scoring each block's 16-lane halves in turn (H30's
    // structure), so this does not re-run into H29's spill wall.
    run("qbs4x2", &|| unsafe {
        let mut sink = 0u64;
        let mask = vdupq_n_u8(0x0F);
        for pass in 0..2 {
            for bb in (0..N_BLOCKS).step_by(2) {
                let mut acc = [[vdupq_n_u16(0); 4]; 4];
                let p0 = codes_ref.as_ptr().add(bb * GROUPS * 32);
                let p1 = codes_ref.as_ptr().add((bb + 1).min(N_BLOCKS - 1) * GROUPS * 32);
                for g in 0..GROUPS {
                    let c0a = vld1q_u8(p0.add(g * 32));
                    let c0b = vld1q_u8(p0.add(g * 32 + 16));
                    let c1a = vld1q_u8(p1.add(g * 32));
                    let c1b = vld1q_u8(p1.add(g * 32 + 16));
                    let (lo0, hi0) = (vandq_u8(c0a, mask), vshrq_n_u8(c0a, 4));
                    let (lo1, hi1) = (vandq_u8(c0b, mask), vshrq_n_u8(c0b, 4));
                    let (lo2, hi2) = (vandq_u8(c1a, mask), vshrq_n_u8(c1a, 4));
                    let (lo3, hi3) = (vandq_u8(c1b, mask), vshrq_n_u8(c1b, 4));
                    for q in 0..4 {
                        // one table load, four blocks-halves of work
                        let lp = luts_ref[pass * 4 + q].as_ptr().add(g * 32);
                        let lut_hi = vld1q_u8(lp);
                        let lut_lo = vld1q_u8(lp.add(16));
                        let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, lo0), vqtbl1q_u8(lut_hi, hi0));
                        let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, lo1), vqtbl1q_u8(lut_hi, hi1));
                        let s2 = vaddq_u8(vqtbl1q_u8(lut_lo, lo2), vqtbl1q_u8(lut_hi, hi2));
                        let s3 = vaddq_u8(vqtbl1q_u8(lut_lo, lo3), vqtbl1q_u8(lut_hi, hi3));
                        acc[q][0] = vaddw_u8(acc[q][0], vget_low_u8(s0));
                        acc[q][1] = vaddw_u8(acc[q][1], vget_high_u8(s1));
                        acc[q][2] = vaddw_u8(acc[q][2], vget_low_u8(s2));
                        acc[q][3] = vaddw_u8(acc[q][3], vget_high_u8(s3));
                    }
                }
                for q in 0..4 { for i in 0..4 { sink = sink.wrapping_add(vaddvq_u16(acc[q][i]) as u64); } }
            }
        }
        sink
    });
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("aarch64 only");
}
#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("aarch64 only");
}
