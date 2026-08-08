//! P5 — price the 2-bit expand+SDOT formulation against the shipped LUT scan
//! before building a kernel (H3).
//!
//! The 4-bit climb's permute-dot won at nq=100 because the unpack is shared
//! across the batch and the per-query work drops to dot-product MACs, which
//! Neoverse V2 issues on all four vector pipes where TBL gets two. The 2-bit
//! question is whether the same shape wins when a byte carries four
//! dimensions: the KleidiAI qsu2cxp expansion is 3 USHR + 3 AND + 4 TBL per
//! 16 code bytes (64 dims, shared), then 4 SDOT-by-element per query.
//!
//! Both loops stream the same ~37 MB code array (200k x 768 dims at 2 bits),
//! deliberately DRAM-resident like the real cell — P10/P13/H23 recorded four
//! L1-resident probes overstating the cell by 10-25%.
//!
//! This is a probe, not a kernel: scores go to a black_box sink, no top-k, no
//! scale epilogue. It prices the inner loop and nothing else.
//!
//! Usage: probe_2bit_sdot [nq] [reps]   (defaults 4, 30)

#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "aarch64")]
fn main() {
    use std::arch::aarch64::*;
    use std::hint::black_box;
    use std::time::Instant;

    // vdotq_s32 is unstable in std::arch; the crate uses inline asm for the
    // same reason (see search.rs sdot_lane).
    #[inline(always)]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut o = acc;
        std::arch::asm!(
            ".arch_extension dotprod",
            "sdot {o:v}.4s, {a:v}.16b, {b:v}.16b",
            o = inout(vreg) o, a = in(vreg) a, b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        o
    }

    // SMMLA: 2x8 by 8x2 -> 2x2 i32 tile, 32 MACs per instruction.
    #[inline(always)]
    unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut o = acc;
        std::arch::asm!(
            ".arch_extension i8mm",
            "smmla {o:v}.4s, {a:v}.16b, {b:v}.16b",
            o = inout(vreg) o, a = in(vreg) a, b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        o
    }

    let args: Vec<String> = std::env::args().collect();
    let nq: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(4);
    let reps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(30);

    const N: usize = 200_000;
    const DIM: usize = 768;
    let bytes_per_vec = DIM / 4; // 2-bit: 4 dims per byte
    let total = N * bytes_per_vec;

    // Deterministic pseudo-random codes; content is irrelevant to timing.
    let mut codes = vec![0u8; total];
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for b in codes.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }

    // Per-query nibble LUT (the shipped kernel's shape: 16 entries covering
    // two dims of four levels each) and per-query i8 weights for SDOT.
    let luts: Vec<[u8; 16]> = (0..nq)
        .map(|q| std::array::from_fn(|i| (i * 7 + q * 13) as u8))
        .collect();
    let weights: Vec<i8> = (0..nq * DIM).map(|i| ((i * 37) % 255) as i8 - 127).collect();
    let levels: [i8; 4] = [-100, -33, 33, 100];

    // ---------- A: shipped-shape LUT scan ----------
    // Per byte-group of 32 code bytes (128 dims x 32 lanes is the real
    // blocked layout; here a flat stream prices the same instruction mix):
    // shared nibble split, then per query 2 TBL + widening adds.
    let lut_pass = |codes: &[u8]| -> u64 {
        let mask = unsafe { vdupq_n_u8(0x0F) };
        let mut sink = 0u64;
        unsafe {
            let mut acc = vec![vdupq_n_u16(0); nq];
            for chunk in codes.chunks_exact(32) {
                let c0 = vld1q_u8(chunk.as_ptr());
                let c1 = vld1q_u8(chunk.as_ptr().add(16));
                let lo0 = vandq_u8(c0, mask);
                let hi0 = vshrq_n_u8(c0, 4);
                let lo1 = vandq_u8(c1, mask);
                let hi1 = vshrq_n_u8(c1, 4);
                for q in 0..nq {
                    let t = vld1q_u8(luts[q].as_ptr());
                    let s0 = vaddq_u8(vqtbl1q_u8(t, lo0), vqtbl1q_u8(t, hi0));
                    let s1 = vaddq_u8(vqtbl1q_u8(t, lo1), vqtbl1q_u8(t, hi1));
                    acc[q] = vaddw_u8(acc[q], vget_low_u8(s0));
                    acc[q] = vaddw_u8(acc[q], vget_high_u8(s1));
                }
            }
            for q in 0..nq {
                sink = sink.wrapping_add(vaddvq_u16(acc[q]) as u64);
            }
        }
        sink
    };

    // ---------- B: expand + SDOT (KleidiAI qsu2cxp shape) ----------
    // Shared per 16 code bytes: 3 USHR + 3 AND + 4 TBL -> 64 i8 levels.
    // Per query: 4 SDOT against the query's i8 weights.
    let sdot_pass = |codes: &[u8]| -> u64 {
        let m3 = unsafe { vdupq_n_u8(0x03) };
        let ltab = unsafe { vld1q_s8([levels[0], levels[1], levels[2], levels[3],
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr()) };
        let mut sink = 0u64;
        unsafe {
            let mut acc = vec![vdupq_n_s32(0); nq];
            // Weights hoisted: a pricing probe wants the loop's port mix, and
            // the real kernel keeps its weight registers live across the
            // block anyway. (The first version computed a modulo per query
            // per chunk and priced integer division instead of SDOT.)
            let w: Vec<[int8x16_t; 4]> = (0..nq)
                .map(|q| {
                    let wp = weights.as_ptr().add(q * DIM);
                    [vld1q_s8(wp), vld1q_s8(wp.add(16)),
                     vld1q_s8(wp.add(32)), vld1q_s8(wp.add(48))]
                })
                .collect();
            for chunk in codes.chunks_exact(16) {
                let c = vld1q_u8(chunk.as_ptr());
                let f0 = vqtbl1q_s8(ltab, vandq_u8(c, m3));
                let f1 = vqtbl1q_s8(ltab, vandq_u8(vshrq_n_u8(c, 2), m3));
                let f2 = vqtbl1q_s8(ltab, vandq_u8(vshrq_n_u8(c, 4), m3));
                let f3 = vqtbl1q_s8(ltab, vshrq_n_u8(c, 6));
                for q in 0..nq {
                    acc[q] = sdot(acc[q], f0, w[q][0]);
                    acc[q] = sdot(acc[q], f1, w[q][1]);
                    acc[q] = sdot(acc[q], f2, w[q][2]);
                    acc[q] = sdot(acc[q], f3, w[q][3]);
                }
            }
            for q in 0..nq {
                sink = sink.wrapping_add(vaddvq_s32(acc[q]) as u64);
            }
        }
        sink
    };


    // ---------- C: expand + SMMLA ----------
    // Same shared expansion; queries in pairs, each SMMLA scoring 8 dims for
    // 2 queries x 2 "columns" (two 8-dim groups of the stream stand in for
    // the two vectors — the real kernel bakes that into the layout, vm8
    // style, so no in-loop ZIPs are priced here).
    let smmla_pass = |codes: &[u8]| -> u64 {
        let m3 = unsafe { vdupq_n_u8(0x03) };
        let ltab = unsafe { vld1q_s8([levels[0], levels[1], levels[2], levels[3],
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr()) };
        let pairs = nq.div_ceil(2);
        let mut sink = 0u64;
        unsafe {
            let mut acc = vec![vdupq_n_s32(0); pairs];
            // 2x8 A operand per pair: query q's 8 weights then q+1's.
            let w: Vec<[int8x16_t; 4]> = (0..pairs)
                .map(|pr| {
                    let mut a = [[0i8; 16]; 4];
                    for (g, ag) in a.iter_mut().enumerate() {
                        for j in 0..8 {
                            ag[j] = weights[(2 * pr) * DIM + g * 8 + j];
                            ag[8 + j] = weights[(2 * pr + 1).min(nq - 1) * DIM + g * 8 + j];
                        }
                    }
                    [vld1q_s8(a[0].as_ptr()), vld1q_s8(a[1].as_ptr()),
                     vld1q_s8(a[2].as_ptr()), vld1q_s8(a[3].as_ptr())]
                })
                .collect();
            for chunk in codes.chunks_exact(16) {
                let c = vld1q_u8(chunk.as_ptr());
                let f0 = vqtbl1q_s8(ltab, vandq_u8(c, m3));
                let f1 = vqtbl1q_s8(ltab, vandq_u8(vshrq_n_u8(c, 2), m3));
                let f2 = vqtbl1q_s8(ltab, vandq_u8(vshrq_n_u8(c, 4), m3));
                let f3 = vqtbl1q_s8(ltab, vshrq_n_u8(c, 6));
                for pr in 0..pairs {
                    acc[pr] = smmla(acc[pr], w[pr][0], f0);
                    acc[pr] = smmla(acc[pr], w[pr][1], f1);
                    acc[pr] = smmla(acc[pr], w[pr][2], f2);
                    acc[pr] = smmla(acc[pr], w[pr][3], f3);
                }
            }
            for pr in 0..pairs {
                sink = sink.wrapping_add(vaddvq_s32(acc[pr]) as u64);
            }
        }
        sink
    };

    // dims scored per pass = total bytes * 4 dims/byte * nq queries
    let work = (total as f64) * 4.0 * (nq as f64);
    for (name, f) in [("lut", &lut_pass as &dyn Fn(&[u8]) -> u64), ("sdot", &sdot_pass), ("smmla", &smmla_pass)] {
        black_box(f(&codes)); // warm
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            black_box(f(black_box(&codes)));
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!("{name} nq={nq}: {:.3} ms/pass  {:.2} G(q.dim)/s", best * 1e3, work / best / 1e9);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("aarch64 only");
}
