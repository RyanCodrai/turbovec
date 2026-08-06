//! The aarch64 counterpart to `kernel_roofline`: what rate can this machine
//! sustain on the NEON kernel's inner pattern, and how close is the real
//! kernel?
//!
//! P8 measured the AVX-512 kernel at 79% of its achievable issue rate, and
//! found that the *paper* port figure (1 shuffle/cycle) was not the ceiling —
//! the surrounding loads and adds were. Deriving arm's headroom from the same
//! kind of paper number (NEON 2 `tbl`/cycle) would repeat exactly that
//! mistake, so measure it instead.
//!
//! Mirrors `score_4query_block_neon`: load 32 B of codes for one block-group,
//! split nibbles once, then per query load the two 16-byte sub-tables, do
//! four `vqtbl1q_u8` lookups, and widen-accumulate into u16.
//!
//! Run: cargo run --release --example kernel_roofline_neon

fn main() {
    #[cfg(not(target_arch = "aarch64"))]
    println!("aarch64 only");

    #[cfg(target_arch = "aarch64")]
    unsafe {
        run();
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn run() {
    use std::arch::aarch64::*;
    use std::time::Instant;

    // L1-resident so only issue rate is under test.
    const GROUPS: usize = 64; // 64 groups x 32 B codes = 2 KB
    const NQ: usize = 4;
    // `big` sizes the code buffer like the real index (77 MB) so the same
    // sequence is measured while streaming rather than L1-resident; the
    // difference between the two is what the real kernel's residual gap is
    // made of.
    let big = std::env::args().any(|a| a == "big");
    let reps = if big { 77 * 1024 * 1024 / (GROUPS * 32) } else { 1 };
    let codes = vec![0x5Au8; GROUPS * 32 * reps];
    let luts: Vec<Vec<u8>> = (0..NQ)
        .map(|q| vec![(q as u8).wrapping_add(3); GROUPS * 32])
        .collect();

    let mask = vdupq_n_u8(0x0F);
    let iters: u64 = 200_000;

    let mut slab = 0usize;
    let t0 = Instant::now();
    let mut sink: u64 = 0;
    for _ in 0..iters {
        let mut acc: [[uint16x8_t; 4]; NQ] = [[vdupq_n_u16(0); 4]; NQ];
        for g in 0..GROUPS {
            let cp = codes.as_ptr().add(slab * GROUPS * 32 + g * 32);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            let lo0 = vandq_u8(c0, mask);
            let lo1 = vandq_u8(c1, mask);
            let hi0 = vshrq_n_u8(c0, 4);
            let hi1 = vshrq_n_u8(c1, 4);
            for q in 0..NQ {
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
        slab = (slab + 1) % reps;
        for a in acc.iter() {
            for v in a.iter() {
                sink = sink.wrapping_add(vaddvq_u16(*v) as u64);
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();

    // 4 tbl lookups per query per group (2 nibbles x 2 half-blocks).
    let tbls = iters * GROUPS as u64 * NQ as u64 * 4;
    println!("sink {sink}");
    println!("elapsed      {dt:.3} s");
    println!("tbl ops      {:.1} M", tbls as f64 / 1e6);
    println!("tbl/sec      {:.2} G/s   ({})", tbls as f64 / dt / 1e9,
             if big { "streaming 77 MB" } else { "L1-resident" });
    println!();
    let d = deferred(iters, GROUPS);
    println!("deferred     {:.2} G/s  (u8 accumulate, widen every 4 groups)",
             tbls as f64 / d / 1e9);
    println!("  vs current x{:.3}", dt / d);
    println!();
    println!("Real kernel, nq=100 dim=768 N=200k:");
    println!("  tbls = 100 * 384 groups * 6250 blocks * 4 = 960.0 M");
    println!("  divide by (search_seconds * n_cores) for per-core G/s");
}

/// Deferred widening: with a smaller LUT cap the per-group u8 sums can be
/// accumulated in u8 for G groups before a single widening round, replacing
/// 4 `vaddw_u8` per group with 2 u8 adds plus 4 `vaddw_u8` per G groups.
/// Cap 31 gives lo+hi <= 62, so G=4 sums to 248 < 255 with no saturation.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn deferred(iters: u64, groups: usize) -> f64 {
    use std::arch::aarch64::*;
    use std::time::Instant;
    const NQ: usize = 4;
    const G: usize = 4;
    let codes = vec![0x5Au8; groups * 32];
    let luts: Vec<Vec<u8>> = (0..NQ).map(|q| vec![(q as u8) + 3; groups * 32]).collect();
    let mask = vdupq_n_u8(0x0F);
    let mut sink: u64 = 0;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut acc: [[uint16x8_t; 4]; NQ] = [[vdupq_n_u16(0); 4]; NQ];
        let mut a8: [[uint8x16_t; 2]; NQ] = [[vdupq_n_u8(0); 2]; NQ];
        for g in 0..groups {
            let cp = codes.as_ptr().add(g * 32);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            let lo0 = vandq_u8(c0, mask);
            let lo1 = vandq_u8(c1, mask);
            let hi0 = vshrq_n_u8(c0, 4);
            let hi1 = vshrq_n_u8(c1, 4);
            for q in 0..NQ {
                let lp = luts[q].as_ptr().add(g * 32);
                let lut_hi = vld1q_u8(lp);
                let lut_lo = vld1q_u8(lp.add(16));
                let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, lo0), vqtbl1q_u8(lut_hi, hi0));
                let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, lo1), vqtbl1q_u8(lut_hi, hi1));
                a8[q][0] = vaddq_u8(a8[q][0], s0);
                a8[q][1] = vaddq_u8(a8[q][1], s1);
            }
            if g % G == G - 1 {
                for q in 0..NQ {
                    acc[q][0] = vaddw_u8(acc[q][0], vget_low_u8(a8[q][0]));
                    acc[q][1] = vaddw_u8(acc[q][1], vget_high_u8(a8[q][0]));
                    acc[q][2] = vaddw_u8(acc[q][2], vget_low_u8(a8[q][1]));
                    acc[q][3] = vaddw_u8(acc[q][3], vget_high_u8(a8[q][1]));
                    a8[q][0] = vdupq_n_u8(0);
                    a8[q][1] = vdupq_n_u8(0);
                }
            }
        }
        for a in acc.iter() {
            for v in a.iter() {
                sink = sink.wrapping_add(vaddvq_u16(*v) as u64);
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    dt
}
