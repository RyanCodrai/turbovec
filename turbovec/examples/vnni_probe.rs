//! Does a vector-major layout with `vpermb` + `vpdpbusd` really beat the
//! current `vpshufb` + widening-add sequence on x86?
//!
//! The current kernel cannot use a dot-product instruction because adjacent
//! code bytes belong to different database vectors, so `vpdpbusd`'s 4-byte
//! reduction would mix them. That is a property of the LAYOUT, not the
//! algorithm. If instead each aligned 4-byte group holds four subquantizer
//! codes for ONE vector, the reduction becomes exactly "four subquantizer
//! contributions for that vector" and the per-lane semantics hold at dword
//! granularity.
//!
//! `vpermb` is what makes it affordable: its index is 6 bits, so
//! `(j << 4) | code` selects from a 64-byte table holding four consecutive
//! 16-entry LUTs — i.e. the existing LUT array, unchanged. `vpshufb` cannot
//! do this because it applies one 16-byte table per 128-bit lane.
//!
//! Both loops below process the same number of code bytes for the same
//! number of queries, L1-resident, so this measures issue rate only.
//!
//! Run: cargo run --release --example vnni_probe

fn main() {
    #[cfg(not(target_arch = "x86_64"))]
    println!("x86_64 only");

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let vbmi = is_x86_feature_detected!("avx512vbmi");
        let vnni = is_x86_feature_detected!("avx512vnni");
        println!("avx512vbmi={vbmi} avx512vnni={vnni}");
        if !(vbmi && vnni && is_x86_feature_detected!("avx512bw")) {
            println!("missing required features");
            return;
        }
        let a = current();
        let b = proposed();
        let c = deferred();
        let d = proposed8();
        println!();
        println!("current  (vpshufb + widen)      {a:8.3} s");
        println!("deferred (u8 accum, widen /4)   {c:8.3} s   x{:.3}", a / c);
        println!("proposed (vpermb + vpdpbusd)    {b:8.3} s   x{:.3}", a / b);
        println!("proposed, 8 queries/pass       {d:8.3} s   x{:.3} vs 4q",
                 (b * 2.0) / d);
    }
}

#[cfg(target_arch = "x86_64")]
const GROUPS: usize = 64; // 64 iterations of 64 code bytes, L1-resident
#[cfg(target_arch = "x86_64")]
const NQ: usize = 4;
#[cfg(target_arch = "x86_64")]
const ITERS: u64 = 200_000;

/// Today's sequence: shared nibble split, then per query two `vpshufb`, a u8
/// add, and two widening adds into u16 accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "avx512f", enable = "avx512bw")]
unsafe fn current() -> f64 {
    use std::arch::x86_64::*;
    use std::time::Instant;

    let codes = vec![0x5Au8; GROUPS * 64];
    let luts: Vec<Vec<u8>> = (0..NQ).map(|q| vec![(q as u8) + 3; GROUPS * 64]).collect();
    let m0f = _mm512_set1_epi8(0x0F);
    let mut sink = 0i32;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut acc = [[_mm512_setzero_si512(); 2]; NQ];
        for g in 0..GROUPS {
            let c = _mm512_loadu_si512(codes.as_ptr().add(g * 64) as *const __m512i);
            let lo = _mm512_and_si512(c, m0f);
            let hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f);
            for q in 0..NQ {
                let lut = _mm512_loadu_si512(luts[q].as_ptr().add(g * 64) as *const __m512i);
                let a = _mm512_shuffle_epi8(lut, lo);
                let b = _mm512_shuffle_epi8(lut, hi);
                let s = _mm512_add_epi8(a, b);
                // widen u8 -> u16 in two halves
                acc[q][0] = _mm512_add_epi16(acc[q][0], _mm512_and_si512(s, _mm512_set1_epi16(0x00FF)));
                acc[q][1] = _mm512_add_epi16(acc[q][1], _mm512_srli_epi16(s, 8));
            }
        }
        for a in acc.iter() {
            for v in a.iter() {
                sink = sink.wrapping_add(_mm512_reduce_add_epi32(*v));
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    dt
}

/// Proposed: vector-major codes, `vpermb` against a 64-byte concatenated LUT
/// (index = (j << 4) | code), and `vpdpbusd` reducing each vector's four
/// subquantizer contributions into its own u32 lane. One accumulator per
/// query instead of two, and no u16 flush at all.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vbmi",
    enable = "avx512vnni"
)]
unsafe fn proposed() -> f64 {
    use std::arch::x86_64::*;
    use std::time::Instant;

    let codes = vec![0x5Au8; GROUPS * 64];
    let luts: Vec<Vec<u8>> = (0..NQ).map(|q| vec![(q as u8) + 3; GROUPS * 64]).collect();
    let m0f = _mm512_set1_epi8(0x0F);
    // per-lane (j << 4) for j = byte position within each vector's 4-byte group
    let k = _mm512_set1_epi32(0x3020_1000u32 as i32);
    let ones = _mm512_set1_epi8(1);
    let mut sink = 0i32;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut acc = [_mm512_setzero_si512(); NQ];
        for g in 0..GROUPS {
            let c = _mm512_loadu_si512(codes.as_ptr().add(g * 64) as *const __m512i);
            // shared: build both permute indices
            let ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), k);
            let ihi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), k);
            for q in 0..NQ {
                let lut = _mm512_loadu_si512(luts[q].as_ptr().add(g * 64) as *const __m512i);
                let a = _mm512_permutexvar_epi8(ilo, lut);
                let b = _mm512_permutexvar_epi8(ihi, lut);
                acc[q] = _mm512_dpbusd_epi32(acc[q], a, ones);
                acc[q] = _mm512_dpbusd_epi32(acc[q], b, ones);
            }
        }
        for v in acc.iter() {
            sink = sink.wrapping_add(_mm512_reduce_add_epi32(*v));
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    dt
}

/// Deferred widening on x86: with a smaller LUT cap the per-group u8 sums
/// accumulate in u8 for 4 groups before one widening round, replacing the
/// per-group and/shift/2x-vpaddw with a single vpaddb.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "avx512f", enable = "avx512bw")]
unsafe fn deferred() -> f64 {
    use std::arch::x86_64::*;
    use std::time::Instant;

    let codes = vec![0x5Au8; GROUPS * 64];
    let luts: Vec<Vec<u8>> = (0..NQ).map(|q| vec![(q as u8) + 3; GROUPS * 64]).collect();
    let m0f = _mm512_set1_epi8(0x0F);
    let mut sink = 0i32;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut acc = [[_mm512_setzero_si512(); 2]; NQ];
        let mut a8 = [_mm512_setzero_si512(); NQ];
        for g in 0..GROUPS {
            let c = _mm512_loadu_si512(codes.as_ptr().add(g * 64) as *const __m512i);
            let lo = _mm512_and_si512(c, m0f);
            let hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f);
            for q in 0..NQ {
                let lut = _mm512_loadu_si512(luts[q].as_ptr().add(g * 64) as *const __m512i);
                let a = _mm512_shuffle_epi8(lut, lo);
                let b = _mm512_shuffle_epi8(lut, hi);
                a8[q] = _mm512_add_epi8(a8[q], _mm512_add_epi8(a, b));
            }
            if g % 4 == 3 {
                for q in 0..NQ {
                    acc[q][0] = _mm512_add_epi16(
                        acc[q][0], _mm512_and_si512(a8[q], _mm512_set1_epi16(0x00FF)));
                    acc[q][1] = _mm512_add_epi16(acc[q][1], _mm512_srli_epi16(a8[q], 8));
                    a8[q] = _mm512_setzero_si512();
                }
            }
        }
        for a in acc.iter() {
            for v in a.iter() {
                sink = sink.wrapping_add(_mm512_reduce_add_epi32(*v));
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    dt
}

/// The same kernel at 8 queries per pass instead of 4. Halves the number of
/// passes over the code array, so the shared per-pass work (one load plus the
/// index computation) is amortised over twice as many queries. Only feasible
/// because u32 accumulation needs one register per query per 16 vectors --
/// at 4 queries that is 8 zmm, at 8 queries 16, where the classic kernel
/// already held 16 at 4 queries and could not go wider (H12).
///
/// Timed over the same total query-work as `proposed`, so the printed ratio
/// compares like with like: this runs half the iterations with twice the
/// queries.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vbmi",
    enable = "avx512vnni"
)]
unsafe fn proposed8() -> f64 {
    use std::arch::x86_64::*;
    use std::time::Instant;
    const NQ8: usize = 8;

    let codes = vec![0x5Au8; GROUPS * 64];
    let luts: Vec<Vec<u8>> = (0..NQ8).map(|q| vec![(q as u8) + 3; GROUPS * 64]).collect();
    let m0f = _mm512_set1_epi8(0x0F);
    let k = _mm512_set1_epi32(0x3020_1000u32 as i32);
    let ones = _mm512_set1_epi8(1);
    let mut sink = 0i32;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut acc = [_mm512_setzero_si512(); NQ8];
        for g in 0..GROUPS {
            let c = _mm512_loadu_si512(codes.as_ptr().add(g * 64) as *const __m512i);
            let ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), k);
            let ihi = _mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), k);
            for q in 0..NQ8 {
                let lut = _mm512_loadu_si512(luts[q].as_ptr().add(g * 64) as *const __m512i);
                acc[q] = _mm512_dpbusd_epi32(acc[q], _mm512_permutexvar_epi8(ilo, lut), ones);
                acc[q] = _mm512_dpbusd_epi32(acc[q], _mm512_permutexvar_epi8(ihi, lut), ones);
            }
        }
        for v in acc.iter() {
            sink = sink.wrapping_add(_mm512_reduce_add_epi32(*v));
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    dt
}
