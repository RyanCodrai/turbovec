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
    let x = tbx_variant::<false>(iters, GROUPS, reps, &codes, &luts);
    println!("tbx/zero     {:.2} G/s  (vqtbx1q_u8, zero fallback)",
             tbls as f64 / x / 1e9);
    println!("  vs current x{:.3}", dt / x);
    let x2 = tbx_variant::<true>(iters, GROUPS, reps, &codes, &luts);
    println!("tbx/self     {:.2} G/s  (vqtbx1q_u8, table as its own fallback)",
             tbls as f64 / x2 / 1e9);
    println!("  vs current x{:.3}", dt / x2);
    println!();
    let u = uadalp_variant(iters, GROUPS, reps, &codes, &luts);
    println!("uadalp       {:.2} G/s  (paired layout, vqtbl2q + vpadalq_u8)",
             tbls as f64 / u / 1e9);
    println!("  vs current x{:.3}", dt / u);
    println!();
    println!("Real kernel, nq=100 dim=768 N=200k:");
    println!("  tbls = 100 * 384 groups * 6250 blocks * 4 = 960.0 M");
    println!("  divide by (search_seconds * n_cores) for per-core G/s");
}

/// The same sequence with `vqtbx1q_u8` substituted for `vqtbl1q_u8`.
///
/// The Neoverse V2 optimisation guide gives ASIMD `TBL` with 1-2 table
/// registers throughput 2 on pipes **V01**, but `TBX` with 1 table register
/// throughput **4** on pipes **V** — all four. The two differ only in what
/// they do with an out-of-range index: `TBL` writes zero, `TBX` leaves the
/// destination byte unchanged. Our indices are nibbles, always 0-15, so they
/// are never out of range and the two are semantically identical here.
///
/// P9 concluded the loop is bound by its loads, ANDs and widening adds rather
/// than by the lookup unit, which reads as "a faster TBL cannot help". But
/// that misses the mechanism: those ANDs and adds also need V pipes, and TBL
/// monopolises V01. If TBX spreads the four lookups across all four pipes it
/// relieves the exact port pressure P9 identified as binding. Whether the
/// scheduler actually converts that freedom into throughput is the question,
/// and it is one only measurement answers.
///
/// The fallback operand is a loop-invariant zero: TBX reads its destination,
/// so passing a value that never changes keeps the dependency on a constant
/// register rather than creating a serial chain through the accumulators.
#[cfg(target_arch = "aarch64")]
unsafe fn tbx_variant<const REUSE_TABLE_AS_FALLBACK: bool>(
    iters: u64,
    groups: usize,
    reps: usize,
    codes: &[u8],
    luts: &[Vec<u8>],
) -> f64 {
    use std::arch::aarch64::*;
    use std::time::Instant;
    const NQ: usize = 4;
    let mask = vdupq_n_u8(0x0F);
    let zero = vdupq_n_u8(0);
    let mut sink: u64 = 0;
    let mut slab = 0usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut acc: [[uint16x8_t; 4]; NQ] = [[vdupq_n_u16(0); 4]; NQ];
        for g in 0..groups {
            let cp = codes.as_ptr().add(slab * groups * 32 + g * 32);
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
                // Fallback = the table register itself. Its value is
                // irrelevant (no index is ever out of range), and it is
                // already materialised, so this asks for no `movi`.
                let (f0, f1) = if REUSE_TABLE_AS_FALLBACK {
                    (lut_lo, lut_hi)
                } else {
                    (zero, zero)
                };
                let s0 = vaddq_u8(
                    vqtbx1q_u8(f0, lut_lo, lo0),
                    vqtbx1q_u8(f1, lut_hi, hi0),
                );
                let s1 = vaddq_u8(
                    vqtbx1q_u8(f0, lut_lo, lo1),
                    vqtbx1q_u8(f1, lut_hi, hi1),
                );
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
    std::hint::black_box(sink);
    dt
}

/// A paired layout that makes `vpadalq_u8` (UADALP) the accumulate step.
///
/// The accumulate chain is the critical path here — P9 found the loop bound
/// by its ANDs, adds and accumulator pressure rather than by the lookup
/// unit, and Faiss's NEON kernel spends 4 ops per 16 lanes on it where ours
/// spends 3 (`vaddq_u8` + two `vaddw_u8`). `UADALP` would spend 1: it adds
/// adjacent byte lanes pairwise and accumulates straight into u16.
///
/// It is semantically wrong under the current layout, where adjacent byte
/// lanes are different database vectors — folding them sums two vectors'
/// scores together. It becomes exactly right under a layout where byte lane
/// 2i and lane 2i+1 hold vector i's codes for two *consecutive byte-groups*,
/// because then the pairwise fold is summing two dimensions of one vector,
/// which is precisely what the score wants. That is the same manoeuvre the
/// x86 kernel already makes for `vpdpbusd`.
///
/// The obstacle is that the two lanes now need two different 16-entry LUTs,
/// and `vqtbl1q_u8` has one table for the whole register. `vqtbl2q_u8`
/// solves it: a 32-entry table holding both groups' LUTs, indexed by
/// `(parity << 4) | code`. The V2 optimisation guide prices ASIMD TBL with
/// "1 or 2 table" registers identically — latency 2, throughput 2, V01 — so
/// the wider table costs nothing.
///
/// Accounting per 32 bytes of codes, per query: the current shape spends
/// 4 TBL + 2 `vaddq_u8` + 4 `vaddw_u8` = 10 ops to cover 32 vectors x 1
/// group. This shape spends 4 TBL2 + 2 `vaddq_u8` + 2 `vpadalq_u8` = 8 ops
/// to cover 16 vectors x 2 groups — the same total work for 20% fewer
/// per-query ops. The shared nibble split costs 2 extra `ORR` to fold the
/// parity bit into the indices, amortised across the four queries, putting
/// the whole unit at 38 ops against 44.
///
/// This measures the instruction mix only. The layout it assumes is a real
/// change to the packed format, so it is priced here before any of that
/// work is started — the same order the x86 change was taken in.
#[cfg(target_arch = "aarch64")]
unsafe fn uadalp_variant(
    iters: u64,
    groups: usize,
    reps: usize,
    codes: &[u8],
    luts: &[Vec<u8>],
) -> f64 {
    use std::arch::aarch64::*;
    use std::time::Instant;
    const NQ: usize = 4;
    let mask = vdupq_n_u8(0x0F);
    // Parity bit selecting which of the two 16-entry sub-tables a lane reads.
    let parity: uint8x16_t = vld1q_u8(
        [0u8, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16].as_ptr(),
    );
    let mut sink: u64 = 0;
    let mut slab = 0usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        // 8 vectors per u16 register, 32 vectors per block => 4 per query,
        // the same accumulator count as the current kernel.
        let mut acc: [[uint16x8_t; 4]; NQ] = [[vdupq_n_u16(0); 4]; NQ];
        for g in 0..groups {
            let cp = codes.as_ptr().add(slab * groups * 32 + g * 32);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            // Same nibble split, plus the parity bit that turns each index
            // into a 5-bit selector over the 32-entry paired table.
            let lo0 = vorrq_u8(vandq_u8(c0, mask), parity);
            let lo1 = vorrq_u8(vandq_u8(c1, mask), parity);
            let hi0 = vorrq_u8(vshrq_n_u8(c0, 4), parity);
            let hi1 = vorrq_u8(vshrq_n_u8(c1, 4), parity);
            for q in 0..NQ {
                let lp = luts[q].as_ptr().add(g * 32);
                // One 32-entry table: the two consecutive groups' LUTs.
                let tl = uint8x16x2_t(vld1q_u8(lp.add(16)), vld1q_u8(lp.add(16)));
                let th = uint8x16x2_t(vld1q_u8(lp), vld1q_u8(lp));
                let s0 = vaddq_u8(vqtbl2q_u8(tl, lo0), vqtbl2q_u8(th, hi0));
                let s1 = vaddq_u8(vqtbl2q_u8(tl, lo1), vqtbl2q_u8(th, hi1));
                // The pairwise fold: lane 2i + lane 2i+1 are two byte-groups
                // of one vector, widened and accumulated in a single op.
                acc[q][0] = vpadalq_u8(acc[q][0], s0);
                acc[q][1] = vpadalq_u8(acc[q][1], s1);
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
    std::hint::black_box(sink);
    dt
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
