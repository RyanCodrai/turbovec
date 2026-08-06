//! Correctness harness for the vector-major / `vpermb` + `vpdpbusd` kernel.
//!
//! P11/P12 measured the instruction sequence at x1.52, but a fast sequence is
//! worthless if the layout arithmetic is wrong, and that arithmetic is the
//! risky part: a 6-bit permute index built from `(j << 4) | code` against a
//! 64-byte table of four concatenated 16-entry LUTs, over codes rearranged so
//! each aligned 4-byte group belongs to one vector.
//!
//! So before any of this touches the dispatch, this computes the same integer
//! sums two ways — a scalar reference over the current layout, and the new
//! kernel over the transformed layout — and checks they agree exactly.
//!
//! Run: cargo run --release --example vector_major_check

const VECS: usize = 16; // vectors per accumulator group (16 u32 lanes in a zmm)
const GROUPS: usize = 32; // byte-groups (each = 2 dims at 4-bit); multiple of 4

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

/// Codes in the *current* shape: `codes[g * VECS + v]` is vector `v`'s byte for
/// byte-group `g` (low nibble = dim 2g, high nibble = dim 2g+1).
///
/// LUT in the *current* shape: `lut[g * 32 .. ]` = 16 hi entries then 16 lo.
#[cfg(target_arch = "x86_64")]
fn reference(codes: &[u8], lut: &[u8]) -> Vec<u32> {
    let mut out = vec![0u32; VECS];
    for g in 0..GROUPS {
        for v in 0..VECS {
            let c = codes[g * VECS + v];
            let lo = (c & 0x0F) as usize;
            let hi = (c >> 4) as usize;
            out[v] += lut[g * 32 + 16 + lo] as u32; // lo sub-table
            out[v] += lut[g * 32 + hi] as u32; // hi sub-table
        }
    }
    out
}

/// Rearrange codes so that byte `v * 4 + j` of each 64-byte chunk holds vector
/// `v`'s code byte for byte-group `g0 + j`. One chunk covers 16 vectors x 4
/// byte-groups; `vpdpbusd` then reduces each vector's 4 bytes into its own
/// dword lane.
#[cfg(target_arch = "x86_64")]
fn to_vector_major(codes: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; GROUPS * VECS];
    for g0 in (0..GROUPS).step_by(4) {
        let chunk = (g0 / 4) * 64;
        for v in 0..VECS {
            for j in 0..4 {
                out[chunk + v * 4 + j] = codes[(g0 + j) * VECS + v];
            }
        }
    }
    out
}

/// Rearrange the LUT into, per 4 byte-groups, a 64-byte table of the four
/// lo sub-tables followed by a 64-byte table of the four hi sub-tables — so a
/// single `vpermb` with index `(j << 4) | code` selects the right one.
#[cfg(target_arch = "x86_64")]
fn to_split_lut(lut: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; GROUPS * 32];
    for g0 in (0..GROUPS).step_by(4) {
        let chunk = (g0 / 4) * 128;
        for j in 0..4 {
            for e in 0..16 {
                out[chunk + j * 16 + e] = lut[(g0 + j) * 32 + 16 + e]; // lo
                out[chunk + 64 + j * 16 + e] = lut[(g0 + j) * 32 + e]; // hi
            }
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vbmi",
    enable = "avx512vnni"
)]
unsafe fn kernel(codes_vm: &[u8], lut_split: &[u8]) -> Vec<u32> {
    use std::arch::x86_64::*;
    let m0f = _mm512_set1_epi8(0x0F);
    let k = _mm512_set1_epi32(0x3020_1000u32 as i32); // per-lane (j << 4)
    let ones = _mm512_set1_epi8(1);
    let mut acc = _mm512_setzero_si512();

    for chunk in 0..GROUPS / 4 {
        let c = _mm512_loadu_si512(codes_vm.as_ptr().add(chunk * 64) as *const __m512i);
        let ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), k);
        let ihi = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f),
            k,
        );
        let tlo = _mm512_loadu_si512(lut_split.as_ptr().add(chunk * 128) as *const __m512i);
        let thi = _mm512_loadu_si512(lut_split.as_ptr().add(chunk * 128 + 64) as *const __m512i);
        acc = _mm512_dpbusd_epi32(acc, _mm512_permutexvar_epi8(ilo, tlo), ones);
        acc = _mm512_dpbusd_epi32(acc, _mm512_permutexvar_epi8(ihi, thi), ones);
    }

    let mut out = vec![0u32; VECS];
    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
    out
}

#[cfg(target_arch = "x86_64")]
unsafe fn run() {
    // Deterministic pseudo-random codes and LUT entries. LUT values stay <= 127
    // to match the current cap, though this kernel could take the full 255.
    let mut st = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    let codes: Vec<u8> = (0..GROUPS * VECS).map(|_| (next() & 0xFF) as u8).collect();
    let lut: Vec<u8> = (0..GROUPS * 32).map(|_| (next() % 128) as u8).collect();

    let want = reference(&codes, &lut);
    let got = kernel(&to_vector_major(&codes), &to_split_lut(&lut));

    let ok = want == got;
    println!("reference {:?}", &want[..8]);
    println!("kernel    {:?}", &got[..8]);
    println!();
    if ok {
        println!("MATCH — vector-major layout + vpermb/vpdpbusd reproduces the");
        println!("        scalar sums exactly for all {VECS} vectors");
    } else {
        println!("MISMATCH");
        for v in 0..VECS {
            if want[v] != got[v] {
                println!("  vector {v}: want {} got {}", want[v], got[v]);
            }
        }
        std::process::exit(1);
    }
}
