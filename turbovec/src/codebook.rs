//! Lloyd-Max scalar quantizer for the Beta distribution.
//!
//! After orthogonal rotation, each coordinate of a unit vector on S^{d-1}
//! follows Beta((d-1)/2, (d-1)/2) on [-1, 1]. This module computes optimal
//! quantization boundaries and centroids for that distribution.

use statrs::distribution::{Beta, ContinuousCDF, Continuous};

/// Returns (boundaries, centroids) for the given bit width and dimension.
///
/// Crate-internal: trusts `bits ∈ {2,3,4}` and `dim >= 2`. Callers reach it
/// only after those bounds are enforced (`TurboQuantIndex::new` /
/// `from_parts`). Exposing it publicly would let a caller pass `bits` in the
/// ~32..63 range and drive an unbounded `1 << bits` allocation (DoS), or
/// `dim` 0/1 and hit a `Beta::new` panic — see the validated
/// [`from_parts`](crate::TurboQuantIndex::from_parts) boundary instead.
pub(crate) fn codebook(bits: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    lloyd_max(bits, dim, 200, 1e-12)
}

fn lloyd_max(bits: usize, dim: usize, max_iter: usize, tol: f64) -> (Vec<f32>, Vec<f32>) {
    let a = (dim as f64 - 1.0) / 2.0;
    // Beta(a, a) on [0, 1], shifted to [-1, 1] via loc=-1, scale=2
    let beta = Beta::new(a, a).unwrap();

    let n_levels = 1usize << bits;

    // Initialize centroids within +/- 3 std devs
    let std_dev = (2.0 * a / ((2.0 * a + 1.0) * 4.0 * a)).sqrt(); // std of Beta on [-1,1]
    let spread = 3.0 * std_dev;
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| -spread + 2.0 * spread * i as f64 / (n_levels as f64 - 1.0))
        .collect();

    for _ in 0..max_iter {
        // Boundaries = midpoints between consecutive centroids
        let boundaries: Vec<f64> = (0..n_levels - 1)
            .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
            .collect();

        let mut edges = Vec::with_capacity(n_levels + 1);
        edges.push(-1.0);
        edges.extend_from_slice(&boundaries);
        edges.push(1.0);

        let mut new_centroids = vec![0.0f64; n_levels];

        for i in 0..n_levels {
            let lo = edges[i];
            let hi = edges[i + 1];

            // CDF on [-1, 1]: transform to [0, 1] for Beta
            let cdf_lo = beta.cdf((lo + 1.0) / 2.0);
            let cdf_hi = beta.cdf((hi + 1.0) / 2.0);
            let prob = cdf_hi - cdf_lo;

            if prob < 1e-15 {
                new_centroids[i] = centroids[i];
            } else {
                // Conditional mean = integral(x * pdf(x), lo, hi) / prob
                // where pdf is on [-1, 1]: pdf_shifted(x) = beta.pdf((x+1)/2) / 2
                let mean = adaptive_simpson(
                    |x| {
                        let t = (x + 1.0) / 2.0;
                        x * beta.pdf(t) / 2.0
                    },
                    lo,
                    hi,
                    1e-14,
                    50,
                );
                new_centroids[i] = mean / prob;
            }
        }

        let max_change = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        centroids = new_centroids;

        if max_change < tol {
            break;
        }
    }

    let boundaries: Vec<f32> = (0..n_levels - 1)
        .map(|i| ((centroids[i] + centroids[i + 1]) / 2.0) as f32)
        .collect();
    let centroids_f32: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();

    (boundaries, centroids_f32)
}

/// Adaptive Simpson's rule for numerical integration.
fn adaptive_simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64, max_depth: usize) -> f64 {
    let mid = (a + b) / 2.0;
    let fa = f(a);
    let fb = f(b);
    let fm = f(mid);
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    adaptive_simpson_rec(&f, a, b, fa, fb, fm, whole, tol, max_depth)
}

fn adaptive_simpson_rec<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    depth: usize,
) -> f64 {
    let mid = (a + b) / 2.0;
    let m1 = (a + mid) / 2.0;
    let m2 = (mid + b) / 2.0;
    let fm1 = f(m1);
    let fm2 = f(m2);
    let left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
    let right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
    let refined = left + right;

    if depth == 0 || (refined - whole).abs() < 15.0 * tol {
        refined + (refined - whole) / 15.0
    } else {
        adaptive_simpson_rec(f, a, mid, fa, fm, fm1, left, tol / 2.0, depth - 1)
            + adaptive_simpson_rec(f, mid, b, fm, fb, fm2, right, tol / 2.0, depth - 1)
    }
}
