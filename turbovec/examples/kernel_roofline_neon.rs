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
    let codes = vec![0x5Au8; GROUPS * 32];
    let luts: Vec<Vec<u8>> = (0..NQ)
        .map(|q| vec![(q as u8).wrapping_add(3); GROUPS * 32])
        .collect();

    let mask = vdupq_n_u8(0x0F);
    let iters: u64 = 200_000;

    let t0 = Instant::now();
    let mut sink: u64 = 0;
    for _ in 0..iters {
        let mut acc: [[uint16x8_t; 4]; NQ] = [[vdupq_n_u16(0); 4]; NQ];
        for g in 0..GROUPS {
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
                acc[q][0] = vaddw_u8(acc[q][0], vget_low_u8(s0));
                acc[q][1] = vaddw_u8(acc[q][1], vget_high_u8(s0));
                acc[q][2] = vaddw_u8(acc[q][2], vget_low_u8(s1));
                acc[q][3] = vaddw_u8(acc[q][3], vget_high_u8(s1));
            }
        }
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
    println!("tbl/sec      {:.2} G/s   (single thread, L1-resident)", tbls as f64 / dt / 1e9);
    println!();
    println!("Real kernel, nq=100 dim=768 N=200k:");
    println!("  tbls = 100 * 384 groups * 6250 blocks * 4 = 960.0 M");
    println!("  divide by (search_seconds * n_cores) for per-core G/s");
}
