//! What rate can this machine actually sustain on the search kernel's inner
//! pattern, and how close does the real kernel get?
//!
//! The previous climb concluded the kernels are port-saturated. That was
//! never checked on this hardware, and the arithmetic disagrees: at nq=100
//! the AVX-512 kernel issues ~240M `vpshufb` against a port-5 budget several
//! times that. Either the kernel stalls (headroom exists) or the machine
//! cannot issue shuffles as fast as the paper number says (it does not).
//!
//! This runs the kernel's exact inner sequence — load codes, split nibbles,
//! shuffle two LUTs, accumulate into u16 — over a buffer small enough to sit
//! in L1, so nothing but issue rate is being measured. Compare its
//! shuffles/second against the same figure derived from a real search.
//!
//! Run: cargo run --release --example kernel_roofline

fn main() {
    #[cfg(not(target_arch = "x86_64"))]
    println!("x86_64 only");

    #[cfg(target_arch = "x86_64")]
    unsafe {
        if !(is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f")) {
            println!("no avx512bw/f");
            return;
        }
        run();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "avx512f", enable = "avx512bw")]
unsafe fn run() {
    use std::arch::x86_64::*;
    use std::time::Instant;

    // Small enough that codes and LUTs are L1-resident: this measures issue
    // rate, not the memory system.
    const GROUPS: usize = 64; // 64 groups x 64 B of codes = 4 KB
    const NQ: usize = 4; // same query batch the real kernel uses
    let codes = vec![0x5Au8; GROUPS * 64];
    let luts: Vec<Vec<u8>> = (0..NQ).map(|q| vec![(q as u8).wrapping_add(3); GROUPS * 32]).collect();

    let mask512 = _mm512_set1_epi8(0x0F);
    let iters: u64 = 200_000;

    let t0 = Instant::now();
    let mut sink = 0i32;
    for _ in 0..iters {
        let mut accus = [[_mm512_setzero_si512(); 4]; NQ];
        let mut g = 0;
        while g + 1 < GROUPS {
            let ca = _mm512_loadu_si512(codes.as_ptr().add(g * 64) as *const __m512i);
            let cb = _mm512_loadu_si512(codes.as_ptr().add((g + 1) * 64) as *const __m512i);
            let clo_a = _mm512_and_si512(ca, mask512);
            let chi_a = _mm512_and_si512(_mm512_srli_epi16(ca, 4), mask512);
            let clo_b = _mm512_and_si512(cb, mask512);
            let chi_b = _mm512_and_si512(_mm512_srli_epi16(cb, 4), mask512);
            for qi in 0..NQ {
                let lut_a = _mm512_broadcast_i64x4(_mm256_loadu_si256(
                    luts[qi].as_ptr().add(g * 32) as *const __m256i,
                ));
                let lut_b = _mm512_broadcast_i64x4(_mm256_loadu_si256(
                    luts[qi].as_ptr().add((g + 1) * 32) as *const __m256i,
                ));
                let r0a = _mm512_shuffle_epi8(lut_a, clo_a);
                let r1a = _mm512_shuffle_epi8(lut_a, chi_a);
                let r0b = _mm512_shuffle_epi8(lut_b, clo_b);
                let r1b = _mm512_shuffle_epi8(lut_b, chi_b);
                accus[qi][0] = _mm512_add_epi16(accus[qi][0], _mm512_add_epi16(r0a, r0b));
                accus[qi][1] = _mm512_add_epi16(
                    accus[qi][1],
                    _mm512_add_epi16(_mm512_srli_epi16(r0a, 8), _mm512_srli_epi16(r0b, 8)),
                );
                accus[qi][2] = _mm512_add_epi16(accus[qi][2], _mm512_add_epi16(r1a, r1b));
                accus[qi][3] = _mm512_add_epi16(
                    accus[qi][3],
                    _mm512_add_epi16(_mm512_srli_epi16(r1a, 8), _mm512_srli_epi16(r1b, 8)),
                );
            }
            g += 2;
        }
        // Keep the accumulators live so nothing is optimized away.
        for a in accus.iter() {
            for v in a.iter() {
                sink = sink.wrapping_add(_mm512_reduce_add_epi32(*v));
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();

    // 4 shuffles per query per group-pair.
    let pairs = (GROUPS / 2) as u64;
    let shuffles = iters * pairs * NQ as u64 * 4;
    println!("sink {sink}");
    println!("elapsed        {dt:.3} s");
    println!("shuffles       {:.1} M", shuffles as f64 / 1e6);
    println!("shuffles/sec   {:.2} G/s   (single thread, L1-resident)", shuffles as f64 / dt / 1e9);
    println!();
    println!("For comparison, derive the real kernel's rate from a search:");
    println!("  shuffles = nq * (dim/2 groups / 2) * (n_vectors/64 block-pairs) * 4");
    println!("  at nq=100, dim=768, N=200k: 100 * 192 * 3125 * 4 = 240.0 M");
    println!("  divide by (search_seconds * n_physical_cores) for per-core G/s");
}
