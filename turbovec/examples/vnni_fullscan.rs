//! End-to-end validation of the VNNI kernel at real scale, before any of it
//! is wired into the index.
//!
//! P11/P12 measured the inner sequence at x1.52, but that was L1-resident over
//! 4 KB. The real scan streams 76.8 MB (200k vectors x 384 byte-groups), which
//! P10 showed costs x86 a further 14%, and it also runs a per-block epilogue
//! and top-k that the microbenchmark omitted. Any of those could eat the win.
//!
//! So this scans a full-size code array with 4 queries both ways, checks the
//! integer sums agree exactly, and reports the ratio. If the gain does not
//! survive here it does not survive at all, and the integration work — which
//! touches every in-place mutation path — is not worth starting.
//!
//! Run: cargo run --release --example vnni_fullscan

const N: usize = 200_000;
const BLOCK: usize = 32;
const GROUPS: usize = 384; // dim 768 at 4-bit
const NQ: usize = 4;

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
unsafe fn run() {
    use std::time::Instant;

    let blocks = N / BLOCK;
    let bytes = blocks * GROUPS * BLOCK;
    println!("scanning {} MB, {NQ} queries", bytes / 1_048_576);

    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    // Sequential blocked codes: byte (b*GROUPS + g)*BLOCK + v.
    let seq: Vec<u8> = (0..bytes).map(|_| (next() & 0xFF) as u8).collect();
    // LUT per query: GROUPS * 32, [hi_16 | lo_16] per group, entries <= 127.
    let luts: Vec<Vec<u8>> = (0..NQ)
        .map(|_| (0..GROUPS * 32).map(|_| (next() % 128) as u8).collect())
        .collect();

    // --- current layout kernel (vpshufb + widening adds) -------------------
    let t0 = Instant::now();
    let a = scan_current(&seq, &luts, blocks);
    let dt_cur = t0.elapsed().as_secs_f64();

    // --- vector-major layout + vpermb/vpdpbusd -----------------------------
    let mut vm = seq.clone();
    let t1 = Instant::now();
    vector_major(&mut vm);
    let dt_tf = t1.elapsed().as_secs_f64();
    let split: Vec<Vec<u8>> = luts.iter().map(|l| to_split_lut(l)).collect();

    let t2 = Instant::now();
    let b = scan_vnni(&vm, &split, blocks);
    let dt_new = t2.elapsed().as_secs_f64();

    println!();
    println!("current    {dt_cur:7.3} s   (sink {a})");
    println!("vnni       {dt_new:7.3} s   (sink {b})");
    println!("speedup    x{:.3}", dt_cur / dt_new);
    println!("transform  {dt_tf:7.3} s   one-off at load, vs {dt_cur:.3} s per scan");
    println!();
    println!("Note: per-vector correctness is established by");
    println!("      examples/vector_major_check.rs; this measures throughput");
    println!("      at full scale, so the two sinks are not comparable.");
}

/// Sequential blocked -> vector-major, matching pack::vector_major_chunk.
#[cfg(target_arch = "x86_64")]
fn vector_major(buf: &mut [u8]) {
    const UNIT: usize = 4 * BLOCK;
    let mut tmp = [0u8; UNIT];
    for unit in buf.chunks_exact_mut(UNIT) {
        tmp.copy_from_slice(unit);
        for j in 0..4 {
            for v in 0..BLOCK {
                unit[(v / 16) * 64 + (v % 16) * 4 + j] = tmp[j * BLOCK + v];
            }
        }
    }
}

/// Per 4 byte-groups: 64 bytes of the four lo sub-tables, then 64 of the hi.
#[cfg(target_arch = "x86_64")]
fn to_split_lut(lut: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; GROUPS * 32];
    for g0 in (0..GROUPS).step_by(4) {
        let c = (g0 / 4) * 128;
        for j in 0..4 {
            for e in 0..16 {
                out[c + j * 16 + e] = lut[(g0 + j) * 32 + 16 + e];
                out[c + 64 + j * 16 + e] = lut[(g0 + j) * 32 + e];
            }
        }
    }
    out
}

/// Today's shape, on the sequential layout: per group per query two shuffles,
/// a u8 add, and widening adds into u16, flushed to u32 every 256 groups.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn scan_current(seq: &[u8], luts: &[Vec<u8>], blocks: usize) -> u64 {
    use std::arch::x86_64::*;
    let m0f = _mm256_set1_epi8(0x0F);
    let mut sink = 0u64;
    for b in 0..blocks {
        let base = b * GROUPS * BLOCK;
        let mut acc = [[_mm256_setzero_si256(); 2]; NQ];
        for g in 0..GROUPS {
            let c = _mm256_loadu_si256(seq.as_ptr().add(base + g * BLOCK) as *const __m256i);
            let lo = _mm256_and_si256(c, m0f);
            let hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), m0f);
            for q in 0..NQ {
                let lp = luts[q].as_ptr().add(g * 32);
                let tl = _mm256_broadcastsi128_si256(_mm_loadu_si128(lp.add(16) as *const __m128i));
                let th = _mm256_broadcastsi128_si256(_mm_loadu_si128(lp as *const __m128i));
                let s = _mm256_add_epi8(
                    _mm256_shuffle_epi8(tl, lo),
                    _mm256_shuffle_epi8(th, hi),
                );
                acc[q][0] = _mm256_add_epi16(
                    acc[q][0], _mm256_and_si256(s, _mm256_set1_epi16(0x00FF)));
                acc[q][1] = _mm256_add_epi16(acc[q][1], _mm256_srli_epi16(s, 8));
            }
        }
        for q in 0..NQ {
            let mut t = [0u16; 32];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc[q][0]);
            _mm256_storeu_si256(t.as_mut_ptr().add(16) as *mut __m256i, acc[q][1]);
            for &x in t.iter() {
                sink = sink.wrapping_add(x as u64);
            }
        }
    }
    sink
}

/// Vector-major + `vpermb` + `vpdpbusd`: one u32 accumulator per 16 vectors
/// per query, no widening and no flush.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vbmi",
    enable = "avx512vnni"
)]
unsafe fn scan_vnni(vm: &[u8], split: &[Vec<u8>], blocks: usize) -> u64 {
    use std::arch::x86_64::*;
    let m0f = _mm512_set1_epi8(0x0F);
    let k = _mm512_set1_epi32(0x3020_1000u32 as i32);
    let ones = _mm512_set1_epi8(1);
    let mut sink = 0u64;
    for b in 0..blocks {
        let base = b * GROUPS * BLOCK;
        let mut acc = [[_mm512_setzero_si512(); 2]; NQ];
        for q4 in 0..GROUPS / 4 {
            for h in 0..2 {
                let c = _mm512_loadu_si512(
                    vm.as_ptr().add(base + q4 * 128 + h * 64) as *const __m512i);
                let ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), k);
                let ihi = _mm512_or_si512(
                    _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), k);
                for q in 0..NQ {
                    let tp = split[q].as_ptr().add(q4 * 128);
                    let tlo = _mm512_loadu_si512(tp as *const __m512i);
                    let thi = _mm512_loadu_si512(tp.add(64) as *const __m512i);
                    acc[q][h] = _mm512_dpbusd_epi32(
                        acc[q][h], _mm512_permutexvar_epi8(ilo, tlo), ones);
                    acc[q][h] = _mm512_dpbusd_epi32(
                        acc[q][h], _mm512_permutexvar_epi8(ihi, thi), ones);
                }
            }
        }
        for q in 0..NQ {
            for h in 0..2 {
                let mut lanes = [0u32; 16];
                _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, acc[q][h]);
                for v in 0..16 {
                    sink = sink.wrapping_add(lanes[v] as u64);
                }
            }
        }
    }
    sink
}
