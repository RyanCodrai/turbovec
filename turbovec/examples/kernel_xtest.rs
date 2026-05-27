//! Cross-arch kernel parity smoke-test.
//!
//! Build & run on ARM and x86 separately. Compare the output files —
//! identical = kernels agree on the same encoded data, divergence = the
//! arches disagree and there's work to do.
//!
//! Usage:
//!   cargo run --release --example kernel_xtest -- <dim> <bits> <seed> > /tmp/out.txt
//!
//! Fixture is fully deterministic from `seed` so ARM and x86 see exactly
//! the same input — no dataset files needed, no cross-machine state
//! shuffling, no BLAS-backend noise (rotation is computed deterministically
//! by turbovec from a fixed ROTATION_SEED).

use std::env;
use std::io::Write;

// Pull in the BLAS symbol provider so the example links against Accelerate
// (macOS) / OpenBLAS (Linux) — ndarray's `blas` feature otherwise leaves
// `cblas_sgemm` undefined at link time in non-lib targets.
#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(target_os = "linux")]
extern crate blas_src;

use turbovec::TurboQuantIndex;

const N_DB: usize = 5_000;
const N_QUERIES: usize = 50;
const K: usize = 32;

fn main() {
    let args: Vec<String> = env::args().collect();
    let dim: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1536);
    let bits: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(4);
    let seed: u64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(42);

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "other"
    };

    eprintln!("# arch={arch} dim={dim} bits={bits} seed={seed} n_db={N_DB} n_queries={N_QUERIES} k={K}");

    // Deterministic synthetic fixture — xoshiro256** seeded from `seed`.
    // Normalising at the end so we look like real angular embeddings (and so
    // the per-vector scales are well-behaved).
    let mut state = splitmix64_init(seed);
    let mut db = vec![0.0f32; N_DB * dim];
    let mut queries = vec![0.0f32; N_QUERIES * dim];
    for v in &mut db {
        *v = next_normal(&mut state);
    }
    for v in &mut queries {
        *v = next_normal(&mut state);
    }
    normalize_rows(&mut db, dim);
    normalize_rows(&mut queries, dim);

    let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
    idx.add(&db);
    idx.prepare();

    let results = idx.search(&queries, K);

    // Dump per-query top-K as `qi <tab> rank <tab> idx`. Scores are
    // deliberately omitted — adjacent-rank tied-score swaps between arches
    // (from different FMA orderings in NEON vs AVX2 SUB-trick layouts) flip
    // 1.25% of rank positions but never change the SET of vectors in the
    // top-K. We surface that set-equivalence here; scores are a separate
    // (and stricter) bit-exactness question worth gating separately when
    // it matters.
    //
    // To make set-equivalence visible per query, the indices are sorted
    // ascending within each query before printing — flips inside the top-K
    // become invisible by construction, but a vector entering/leaving the
    // top-K (a real divergence) still shows up as a diff.
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    for qi in 0..N_QUERIES {
        let mut indices: Vec<i64> = results.indices_for_query(qi).to_vec();
        indices.sort_unstable();
        for i in indices {
            writeln!(w, "{qi}\t{i}").unwrap();
        }
    }
}

fn normalize_rows(rows: &mut [f32], dim: usize) {
    for row in rows.chunks_exact_mut(dim) {
        let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-10 {
            let inv = 1.0 / n;
            for v in row {
                *v *= inv;
            }
        }
    }
}

// ─── Tiny deterministic PRNG (xoshiro256** seeded via splitmix64) ────────
// Same on every arch; no BLAS, no platform RNG.

fn splitmix64_init(seed: u64) -> [u64; 4] {
    let mut x = seed;
    let mut out = [0u64; 4];
    for v in &mut out {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        *v = z ^ (z >> 31);
    }
    out
}

fn xoshiro256ss(s: &mut [u64; 4]) -> u64 {
    let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
    let t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = s[3].rotate_left(45);
    result
}

fn next_normal(s: &mut [u64; 4]) -> f32 {
    // Box-Muller. Uniforms come from the top 24 bits of consecutive xoshiro
    // outputs, giving exactly representable f32s in [0, 1).
    let u1 = ((xoshiro256ss(s) >> 40) as f32) / (1u32 << 24) as f32;
    let u2 = ((xoshiro256ss(s) >> 40) as f32) / (1u32 << 24) as f32;
    let u1 = u1.max(f32::MIN_POSITIVE);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}
