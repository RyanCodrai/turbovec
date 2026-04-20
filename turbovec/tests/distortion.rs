//! Statistical validation of quantizer distortion.
//!
//! Validates two layers of the paper's correctness claim:
//!
//! 1. **Codebook** (analytical): the Lloyd-Max codebook's MSE
//!    against the Beta((d-1)/2, (d-1)/2) distribution matches the
//!    paper's Theorem 1 value divided by d. Uses numerical
//!    integration, no sampling.
//! 2. **Pipeline** (empirical): the full encode -> pack -> SIMD-score
//!    path reproduces Theorem 1 distortion on random unit vectors.
//!    For self-queries in the asymmetric IP setting, the expected
//!    score deficit `E[1 - <Rv, dequantize(C_v)>]` equals
//!    `d * MSE_per_coord ≈ Theorem1(b)`, so we can measure
//!    reconstruction quality through public search scores without
//!    exposing a dequantize API.

extern crate blas_src;

use statrs::distribution::{Beta, Continuous};
use turbovec::codebook::codebook;
use turbovec::TurboQuantIndex;

/// Lloyd-Max MSE for an N(0, 1) source at b bits per dimension,
/// from the TurboQuant paper's Theorem 1 (equivalently, Max 1960).
/// Values match pyturboquant's `PAPER_MSE_VALUES` for cross-library
/// consistency.
const PAPER_MSE: &[(usize, f64)] = &[
    (2, 0.1175),
    (3, 0.03454),
    (4, 0.009497),
];

#[test]
fn codebook_mse_matches_paper_at_high_dim() {
    // At d=1536 the Beta((d-1)/2, (d-1)/2) distribution is very close
    // to N(0, 1/d), so the empirical Lloyd-Max MSE should equal
    // Theorem1(b) / d to within ~5%.
    let dim = 1536;

    for &(bits, paper_val) in PAPER_MSE {
        let (boundaries, centroids) = codebook(bits, dim);
        let mse = compute_codebook_mse(&boundaries, &centroids, dim);
        let expected = paper_val / dim as f64;
        let rel_err = (mse - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "bits={}, dim={}: codebook MSE={:.3e} vs Theorem1/d={:.3e} (rel_err={:.3})",
            bits,
            dim,
            mse,
            expected,
            rel_err,
        );
    }
}

#[test]
fn codebook_mse_within_shannon_factor() {
    // The paper claims distortion within ~2.7x of the Shannon lower
    // bound 2^{-2b}. Validate the ratio stays below 3.0 across every
    // (bits, dim) pair turbovec cares about.
    for &bits in &[2usize, 3, 4] {
        for &dim in &[256usize, 768, 1536] {
            let (boundaries, centroids) = codebook(bits, dim);
            let mse = compute_codebook_mse(&boundaries, &centroids, dim);
            let shannon_bound = 2f64.powi(-2 * bits as i32) / dim as f64;
            let ratio = mse / shannon_bound;
            assert!(
                ratio < 3.0,
                "bits={}, dim={}: MSE/Shannon = {:.3} exceeds 3x paper bound",
                bits,
                dim,
                ratio,
            );
            // Sanity: MSE must be above the bound, not below (would
            // indicate a broken test or a result too good to be true).
            assert!(
                ratio > 1.0,
                "bits={}, dim={}: MSE/Shannon = {:.3} below Shannon lower bound",
                bits,
                dim,
                ratio,
            );
        }
    }
}

/// Analytical MSE of the given (boundaries, centroids) scalar
/// quantizer against Beta((d-1)/2, (d-1)/2) on [-1, 1].
///
/// For each region [b_{i-1}, b_i] with centroid c_i:
///     contribution = integral_{b_{i-1}}^{b_i} (x - c_i)^2 * p(x) dx
/// where p is the Beta PDF shifted from [0,1] to [-1,1].
fn compute_codebook_mse(boundaries: &[f32], centroids: &[f32], dim: usize) -> f64 {
    let a = (dim as f64 - 1.0) / 2.0;
    let beta = Beta::new(a, a).unwrap();

    let n = centroids.len();
    let mut edges = Vec::with_capacity(n + 1);
    edges.push(-1.0f64);
    edges.extend(boundaries.iter().map(|&b| b as f64));
    edges.push(1.0);

    let mut mse = 0.0f64;
    for i in 0..n {
        let lo = edges[i];
        let hi = edges[i + 1];
        let c = centroids[i] as f64;
        // pdf on [-1, 1]: beta.pdf((x + 1) / 2) / 2
        mse += simpson(
            |x: f64| (x - c).powi(2) * beta.pdf((x + 1.0) / 2.0) / 2.0,
            lo,
            hi,
            4000,
        );
    }
    mse
}

/// Composite Simpson's rule over `n` intervals. `n` is rounded down
/// to even; 4000 is enough for 8+ digits on the integrands here.
fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = n & !1;
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

#[test]
fn pipeline_deficit_matches_theorem1() {
    // End-to-end pipeline validation: encode -> pack -> SIMD-score.
    // For a unit vector v indexed and then self-queried under the
    // asymmetric IP scoring turbovec uses:
    //     E[score] = 1 - d * MSE_per_coord
    //              ≈ 1 - Theorem1(b)
    // so `1 - mean_score` is a direct empirical estimate of the
    // paper's theoretical distortion. The kernel's per-sub-table LUT
    // calibration keeps scores within ~1e-4 of the straight IP, so
    // this assertion holds at every supported bit width.
    let dim = 1536;
    let n = 500;
    let vectors = unit_sphere_vectors(n, dim, 42);

    for &(bits, paper_mse) in PAPER_MSE {
        let stats = self_score_stats(&vectors, dim, bits);
        let deficit = 1.0 - stats.mean;
        let rel_err = (deficit - paper_mse).abs() / paper_mse;
        assert!(
            rel_err < 0.10,
            "bits={}: empirical (1 - mean self-score) = {:.5} vs Theorem1 = {:.5} (rel_err = {:.3})",
            bits,
            deficit,
            paper_mse,
            rel_err,
        );
    }
}

#[test]
fn self_query_variance_tightens_with_more_bits() {
    // Even though the mean self-query score is calibrated to 1.0
    // regardless of bit width, the variance around that mean reflects
    // the quantization noise. Higher bit widths must produce tighter
    // distributions. Catches pipeline wiring where `bits` is accepted
    // but silently ignored downstream.
    let dim = 512;
    let n = 200;
    let vectors = unit_sphere_vectors(n, dim, 0);

    let s2 = self_score_stats(&vectors, dim, 2);
    let s4 = self_score_stats(&vectors, dim, 4);

    assert!(
        s4.stddev < s2.stddev,
        "4-bit self-score stddev {:.4} not tighter than 2-bit {:.4} — bits may not be plumbed through",
        s4.stddev,
        s2.stddev,
    );
}

#[test]
fn self_query_recall_at_1() {
    // Pipeline end-to-end: at 4-bit on d=512, a vector queried
    // against an index containing itself must come back at rank 1
    // every time.
    let dim = 512;
    let n = 200;
    let vectors = unit_sphere_vectors(n, dim, 0);

    let mut index = TurboQuantIndex::new(dim, 4);
    index.add(&vectors);
    index.prepare();

    let mut hits = 0;
    for i in 0..n {
        let q = &vectors[i * dim..(i + 1) * dim];
        let results = index.search(q, 1);
        if results.indices_for_query(0)[0] as usize == i {
            hits += 1;
        }
    }
    let recall = hits as f64 / n as f64;
    assert!(
        recall >= 0.99,
        "recall@1 = {:.3} below 0.99 threshold",
        recall,
    );
}

struct ScoreStats {
    mean: f64,
    stddev: f64,
}

/// Mean and stddev of top-1 self-query scores.
fn self_score_stats(vectors: &[f32], dim: usize, bits: usize) -> ScoreStats {
    let n = vectors.len() / dim;
    let mut index = TurboQuantIndex::new(dim, bits);
    index.add(vectors);
    index.prepare();

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let q = &vectors[i * dim..(i + 1) * dim];
        let results = index.search(q, 1);
        scores.push(results.scores_for_query(0)[0] as f64);
    }

    let mean = scores.iter().sum::<f64>() / n as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64;
    ScoreStats { mean, stddev: variance.sqrt() }
}

/// Deterministic unit vectors uniformly distributed on S^{d-1}.
/// Box-Muller from a seeded xorshift stream, then L2-normalize each
/// row. Uniform sphere sampling is required so that the rotated
/// coordinates follow Beta((d-1)/2, (d-1)/2), which is what the
/// Theorem 1 prediction depends on.
fn unit_sphere_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut next_u = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (((state >> 32) as u32) & 0x007FFFFF) | 0x3F800000;
        f32::from_bits(bits) - 1.0
    };

    let mut out = vec![0.0f32; n * dim];
    let mut idx = 0;
    while idx < out.len() {
        // Box-Muller: two uniform -> two standard normals.
        let u1 = next_u().max(1e-30);
        let u2 = next_u();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out[idx] = r * theta.cos();
        idx += 1;
        if idx < out.len() {
            out[idx] = r * theta.sin();
            idx += 1;
        }
    }

    for i in 0..n {
        let row = &mut out[i * dim..(i + 1) * dim];
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    }
    out
}
