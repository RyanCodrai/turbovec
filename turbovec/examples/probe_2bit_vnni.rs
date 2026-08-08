//! P6 — price the x86 2-bit formulations: shipped vpermb-LUT against a
//! shared-decode + vpdpbusd variant (H10).
//!
//! The shipped `search_multi_query_vnni` spends, per 64-byte chunk per query,
//! 2 `vpermb` (p5-only on Sapphire Rapids) + 2 `vpdpbusd` (p05). The
//! shared-decode variant expands the 2-bit fields to i8 levels once per chunk
//! (shifts + masks + `vpshufb`, mostly p0-capable) and then spends 4
//! `vpdpbusd` per query — twice the MACs, none of the per-query p5 permutes.
//! Which side of that trade wins is the probe question; the SimSIMD rewrite
//! bets on shared-decode, the shipped kernel on the LUT.
//!
//! Streams the real 37 MB code volume, black_box sink, no top-k. A probe,
//! not a kernel.
//!
//! Usage: probe_2bit_vnni [nq] [reps]   (defaults 8, 30)

#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "x86_64")]
fn main() {
    use std::arch::x86_64::*;
    use std::hint::black_box;
    use std::time::Instant;

    if !is_x86_feature_detected!("avx512vnni") || !is_x86_feature_detected!("avx512vbmi") {
        eprintln!("needs avx512vnni + vbmi");
        return;
    }

    let args: Vec<String> = std::env::args().collect();
    let nq: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(8);
    let reps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(30);

    const N: usize = 200_000;
    const DIM: usize = 768;
    let total = N * DIM / 4;

    let mut codes = vec![0u8; total];
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for b in codes.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }

    // Per-query 128-byte split LUT (the shipped shape: 64 lo + 64 hi) and
    // per-query u8 weight rows for the dpbusd side.
    let split_luts: Vec<[u8; 128]> = (0..nq)
        .map(|q| std::array::from_fn(|i| (i * 5 + q * 11) as u8))
        .collect();
    let weights: Vec<u8> = (0..nq * DIM).map(|i| ((i * 37) % 251) as u8).collect();

    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni", enable = "avx512vbmi")]
    unsafe fn lut_pass(codes: &[u8], split_luts: &[[u8; 128]], nq: usize) -> u64 {
        use std::arch::x86_64::*;
        let m0f = _mm512_set1_epi8(0x0F);
        let kpos = _mm512_set1_epi8(0x40);
        let mut acc = vec![_mm512_setzero_si512(); nq];
        let ones = _mm512_set1_epi8(1);
        for chunk in codes.chunks_exact(64) {
            let c = _mm512_loadu_si512(chunk.as_ptr() as *const _);
            let ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), kpos);
            let ihi = _mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f),
                kpos,
            );
            for q in 0..nq {
                let tlo = _mm512_loadu_si512(split_luts[q].as_ptr() as *const _);
                let thi = _mm512_loadu_si512(split_luts[q].as_ptr().add(64) as *const _);
                let vlo = _mm512_permutexvar_epi8(ilo, tlo);
                let vhi = _mm512_permutexvar_epi8(ihi, thi);
                acc[q] = _mm512_dpbusd_epi32(acc[q], vlo, ones);
                acc[q] = _mm512_dpbusd_epi32(acc[q], vhi, ones);
            }
        }
        let mut sink = 0u64;
        for q in 0..nq {
            sink = sink.wrapping_add(_mm512_reduce_add_epi32(acc[q]) as u64);
        }
        sink
    }

    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
    unsafe fn decode_pass(codes: &[u8], weights: &[u8], nq: usize) -> u64 {
        use std::arch::x86_64::*;
        // Shared per 64-byte chunk (256 dims): 3 shifts + 4 masks + 4 vpshufb
        // through the 4-entry level table -> 4 zmm of i8 levels. Then per
        // query: 4 vpdpbusd (u8 weights x s8 levels).
        let m3 = _mm512_set1_epi8(0x03);
        let ltab = {
            let mut t = [0i8; 64];
            for i in 0..64 {
                t[i] = [-100i8, -33, 33, 100][i % 4];
            }
            _mm512_loadu_si512(t.as_ptr() as *const _)
        };
        let mut acc = vec![_mm512_setzero_si512(); nq];
        // weight registers hoisted, as the arm probe does
        let w: Vec<[__m512i; 4]> = (0..nq)
            .map(|q| {
                let wp = weights.as_ptr().add(q * DIM);
                [
                    _mm512_loadu_si512(wp as *const _),
                    _mm512_loadu_si512(wp.add(64) as *const _),
                    _mm512_loadu_si512(wp.add(128) as *const _),
                    _mm512_loadu_si512(wp.add(192) as *const _),
                ]
            })
            .collect();
        for chunk in codes.chunks_exact(64) {
            let c = _mm512_loadu_si512(chunk.as_ptr() as *const _);
            let f0 = _mm512_shuffle_epi8(ltab, _mm512_and_si512(c, m3));
            let f1 = _mm512_shuffle_epi8(ltab, _mm512_and_si512(_mm512_srli_epi16(c, 2), m3));
            let f2 = _mm512_shuffle_epi8(ltab, _mm512_and_si512(_mm512_srli_epi16(c, 4), m3));
            let f3 = _mm512_shuffle_epi8(ltab, _mm512_and_si512(_mm512_srli_epi16(c, 6), m3));
            for q in 0..nq {
                acc[q] = _mm512_dpbusd_epi32(acc[q], w[q][0], f0);
                acc[q] = _mm512_dpbusd_epi32(acc[q], w[q][1], f1);
                acc[q] = _mm512_dpbusd_epi32(acc[q], w[q][2], f2);
                acc[q] = _mm512_dpbusd_epi32(acc[q], w[q][3], f3);
            }
        }
        let mut sink = 0u64;
        for q in 0..nq {
            sink = sink.wrapping_add(_mm512_reduce_add_epi32(acc[q]) as u64);
        }
        sink
    }

    let work = (total as f64) * 4.0 * (nq as f64);
    unsafe {
        for name in ["lut", "decode"] {
            let run = |c: &[u8]| -> u64 {
                match name {
                    "lut" => lut_pass(c, &split_luts, nq),
                    _ => decode_pass(c, &weights, nq),
                }
            };
            black_box(run(&codes));
            let mut best = f64::MAX;
            for _ in 0..reps {
                let t0 = Instant::now();
                black_box(run(black_box(&codes)));
                best = best.min(t0.elapsed().as_secs_f64());
            }
            println!("{name} nq={nq}: {:.3} ms/pass  {:.2} G(q.dim)/s", best * 1e3, work / best / 1e9);
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("x86_64 only");
}
