//! Does the constant-permute + dot-product scheme help x86 too?
//!
//! P18 measured x1.187 on the arm inner loop for replacing the per-query
//! table lookup with a *shared* nibble->level permute followed by a dot
//! product. x86 should gain more, for a structural reason: the shipping
//! VNNI kernel's `vpermb` is per-query, because the table it permutes is
//! that query's LUT. Under permute-dot the permute is query-independent, so
//! it leaves the per-query path entirely.
//!
//! Per 64 bytes of codes, per query:
//!
//!   control    2 x 64-byte LUT load + 2 `vpermb` + 2 `vpdpbusd`
//!   candidate  1 x 4-byte broadcast + 2 `vpdpbusd`
//!
//! so the LUT traffic per query per group falls from 128 bytes to 8, and
//! two shuffles per query become two shuffles shared across all four.
//!
//! Both shapes read the same vector-major layout the index already uses
//! from H21, so unlike the arm side this needs no format change to test.
//!
//! One wrinkle: `vpdpbusd` multiplies unsigned by signed, and both our
//! operands are signed. Offsetting the levels by +128 makes them unsigned,
//! and the resulting `128 * sum_d q[d]` term is a per-query constant —
//! independent of which database vector is being scored — so it shifts
//! every score equally and cannot change a ranking.
//!
//! Run: cargo run --release --example x86_permute_dot

fn main() {
    #[cfg(not(target_arch = "x86_64"))]
    println!("x86_64 only");

    #[cfg(target_arch = "x86_64")]
    unsafe {
        if !(is_x86_feature_detected!("avx512vbmi") && is_x86_feature_detected!("avx512vnni")) {
            println!("needs avx512vbmi + avx512vnni");
            return;
        }
        run();
    }
}

#[cfg(target_arch = "x86_64")]
const GROUPS: usize = 384;
#[cfg(target_arch = "x86_64")]
const NQ: usize = 4;

#[cfg(target_arch = "x86_64")]
unsafe fn run() {
    let big = std::env::args().any(|a| a == "big");
    // L1-resident, or sized like the real index so the comparison is made
    // while streaming rather than out of cache.
    let unit = GROUPS * 32;
    let reps = if big { 77 * 1024 * 1024 / unit } else { 1 };

    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    let codes: Vec<u8> = (0..unit * reps).map(|_| (next() & 0xFF) as u8).collect();
    // Control reads a 32-byte-per-group split LUT per query; candidate reads
    // 16 bytes of int8 query weights per group.
    let luts: Vec<Vec<u8>> = (0..NQ)
        .map(|_| (0..GROUPS * 32).map(|_| (next() % 128) as u8).collect())
        .collect();
    let qw: Vec<Vec<i8>> = (0..NQ)
        .map(|_| (0..GROUPS * 16).map(|_| (next() % 127) as i8).collect())
        .collect();

    // Each iteration processes one 12 KB slab and advances, so `iters` has
    // to be a multiple of `reps` for the streaming case to actually sweep
    // the 77 MB buffer rather than touch a corner of it.
    let iters: u64 = if big { reps as u64 * 3 } else { 20_000 };

    let mut c = Vec::new();
    let mut p = Vec::new();
    for _ in 0..5 {
        c.push(control(iters, reps, &codes, &luts));
        p.push(permute_dot(iters, reps, &codes, &qw));
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (cm, pm) = (med(&mut c), med(&mut p));

    println!("{}", if big { "streaming 77 MB" } else { "L1-resident" });
    println!("interleaved, 5 rounds, medians:");
    println!("  control (vpermb + vpdpbusd)   {cm:.4} s");
    println!("  permute-dot (shared vpshufb)  {pm:.4} s");
    println!("  speedup                       x{:.3}", cm / pm);
}

/// The shipping VNNI shape: per query, permute that query's split LUT by the
/// code nibbles, then reduce four byte-groups per dword with `vpdpbusd`.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vbmi",
    enable = "avx512vnni"
)]
unsafe fn control(iters: u64, reps: usize, codes: &[u8], luts: &[Vec<u8>]) -> f64 {
    use std::arch::x86_64::*;
    use std::time::Instant;
    let m0f = _mm512_set1_epi8(0x0F);
    let k = _mm512_set1_epi32(0x3020_1000u32 as i32);
    let ones = _mm512_set1_epi8(1);
    let mut sink = 0u64;
    let mut slab = 0usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut acc = [[_mm512_setzero_si512(); 2]; NQ];
        for q4 in 0..GROUPS / 4 {
            for h in 0..2 {
                let c = _mm512_loadu_si512(
                    codes.as_ptr().add(slab * GROUPS * 32 + q4 * 128 + h * 64) as *const __m512i,
                );
                let ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), k);
                let ihi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), k);
                for q in 0..NQ {
                    let tp = luts[q].as_ptr().add(q4 * 128);
                    let tlo = _mm512_loadu_si512(tp as *const __m512i);
                    let thi = _mm512_loadu_si512(tp.add(64) as *const __m512i);
                    acc[q][h] = _mm512_dpbusd_epi32(
                        acc[q][h],
                        _mm512_permutexvar_epi8(ilo, tlo),
                        ones,
                    );
                    acc[q][h] = _mm512_dpbusd_epi32(
                        acc[q][h],
                        _mm512_permutexvar_epi8(ihi, thi),
                        ones,
                    );
                }
            }
        }
        slab = (slab + 1) % reps;
        for a in acc.iter() {
            for v in a.iter() {
                sink = sink.wrapping_add(_mm512_reduce_add_epi32(*v) as u64);
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    dt
}

/// Constant nibble->level permute, shared across queries, then `vpdpbusd`
/// straight against the query's int8 weights.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vbmi",
    enable = "avx512vnni"
)]
unsafe fn permute_dot(iters: u64, reps: usize, codes: &[u8], qw: &[Vec<i8>]) -> f64 {
    use std::arch::x86_64::*;
    use std::time::Instant;
    let m0f = _mm512_set1_epi8(0x0F);
    // The codebook as unsigned int8 (levels + 128), one fixed 16-entry table
    // for every dimension and every query. `vpshufb` indexes within each
    // 128-bit lane, so the same 16 entries are broadcast to all four lanes.
    let lv: [u8; 16] = [27, 51, 67, 80, 92, 103, 113, 123, 133, 143, 153, 164, 176, 189, 205, 229];
    let levels = _mm512_broadcast_i32x4(_mm_loadu_si128(lv.as_ptr() as *const __m128i));
    let mut sink = 0u64;
    let mut slab = 0usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut acc = [[_mm512_setzero_si512(); 2]; NQ];
        for q4 in 0..GROUPS / 4 {
            for h in 0..2 {
                let c = _mm512_loadu_si512(
                    codes.as_ptr().add(slab * GROUPS * 32 + q4 * 128 + h * 64) as *const __m512i,
                );
                // Shared across all queries — this is the whole point.
                let vlo = _mm512_shuffle_epi8(levels, _mm512_and_si512(c, m0f));
                let vhi =
                    _mm512_shuffle_epi8(levels, _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f));
                for q in 0..NQ {
                    // Four dimensions of query weights, broadcast: every
                    // dword lane holds a different database vector but the
                    // same four dimensions.
                    let wp = qw[q].as_ptr().add(q4 * 16 + h * 8);
                    let wlo = _mm512_set1_epi32(*(wp as *const i32));
                    let whi = _mm512_set1_epi32(*(wp.add(4) as *const i32));
                    acc[q][h] = _mm512_dpbusd_epi32(acc[q][h], vlo, wlo);
                    acc[q][h] = _mm512_dpbusd_epi32(acc[q][h], vhi, whi);
                }
            }
        }
        slab = (slab + 1) % reps;
        for a in acc.iter() {
            for v in a.iter() {
                sink = sink.wrapping_add(_mm512_reduce_add_epi32(*v) as u64);
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    dt
}
