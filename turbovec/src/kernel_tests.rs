//! In-crate relocation of the kernel-level correctness tests.
//!
//! These previously lived in `tests/codebook.rs`, `tests/encode.rs`,
//! `tests/distortion.rs`, and `tests/core_encode_hardening.rs` and reached
//! the low-level `codebook` / `encode` / `pack` functions directly. Those
//! functions are now `pub(crate)` (they trust their caller's invariants and
//! are no longer part of the public API), so the tests moved in-crate. The
//! assertions are unchanged — only the import paths (`turbovec::` →
//! `crate::`) and the module framing differ.

/// From `tests/codebook.rs` — Lloyd-Max codebook structural invariants.
mod codebook_correctness {
    use crate::codebook::codebook;

    #[test]
    fn centroids_strictly_ascending() {
        for &bits in &[2usize, 3, 4] {
            for &dim in &[256usize, 768, 1536] {
                let (_, centroids) = codebook(bits, dim);
                for i in 0..centroids.len() - 1 {
                    assert!(
                        centroids[i] < centroids[i + 1],
                        "centroids not ascending at bits={}, dim={}: c[{}]={} >= c[{}]={}",
                        bits,
                        dim,
                        i,
                        centroids[i],
                        i + 1,
                        centroids[i + 1]
                    );
                }
            }
        }
    }

    #[test]
    fn boundaries_strictly_between_centroids() {
        for &bits in &[2usize, 3, 4] {
            for &dim in &[256usize, 1536] {
                let (boundaries, centroids) = codebook(bits, dim);
                assert_eq!(boundaries.len(), centroids.len() - 1);
                for i in 0..boundaries.len() {
                    assert!(
                        boundaries[i] > centroids[i],
                        "boundary[{}] = {} not > centroid[{}] = {} (bits={}, dim={})",
                        i,
                        boundaries[i],
                        i,
                        centroids[i],
                        bits,
                        dim
                    );
                    assert!(
                        boundaries[i] < centroids[i + 1],
                        "boundary[{}] = {} not < centroid[{}] = {} (bits={}, dim={})",
                        i,
                        boundaries[i],
                        i + 1,
                        centroids[i + 1],
                        bits,
                        dim
                    );
                }
            }
        }
    }

    #[test]
    fn level_counts_correct() {
        for &bits in &[2usize, 3, 4] {
            let (boundaries, centroids) = codebook(bits, 1536);
            assert_eq!(
                centroids.len(),
                1 << bits,
                "expected 2^{} = {} centroids, got {}",
                bits,
                1 << bits,
                centroids.len()
            );
            assert_eq!(
                boundaries.len(),
                (1 << bits) - 1,
                "expected 2^{} - 1 = {} boundaries, got {}",
                bits,
                (1 << bits) - 1,
                boundaries.len()
            );
        }
    }

    #[test]
    fn symmetric_about_zero() {
        for &bits in &[2usize, 3, 4] {
            for &dim in &[768usize, 1536] {
                let (_, centroids) = codebook(bits, dim);
                let n = centroids.len();
                for i in 0..n / 2 {
                    let lo = centroids[i];
                    let hi = centroids[n - 1 - i];
                    assert!(
                        (lo + hi).abs() < 1e-4,
                        "asymmetric: c[{}]={} c[{}]={} (bits={}, dim={})",
                        i,
                        lo,
                        n - 1 - i,
                        hi,
                        bits,
                        dim
                    );
                }
            }
        }
    }

    #[test]
    fn deterministic_for_same_params() {
        let (b1, c1) = codebook(4, 1536);
        let (b2, c2) = codebook(4, 1536);
        assert_eq!(b1, b2);
        assert_eq!(c1, c2);
    }

    #[test]
    fn centroids_within_unit_interval() {
        for &bits in &[2usize, 3, 4] {
            let (_, centroids) = codebook(bits, 1536);
            for (i, &c) in centroids.iter().enumerate() {
                assert!(
                    c > -1.0 && c < 1.0,
                    "centroid[{}] = {} outside (-1, 1) (bits={})",
                    i,
                    c,
                    bits
                );
            }
        }
    }
}

/// From `tests/encode.rs` — encoding-pipeline shape/scale correctness.
mod encode_pipeline {
    use crate::codebook::codebook;
    /// Test shim preserving the old owned-return shape. `encode` no
    /// longer fits — a calibration comes only from an explicit
    /// `calibrate` call — so the trailing pair is whatever the caller
    /// passed in, echoed back for the tests that assert against it.
    #[allow(clippy::too_many_arguments)]
    fn encode_owned(
        vectors: &[f32],
        n: usize,
        dim: usize,
        rotation: &crate::rotation::Rotation,
        boundaries: &[f32],
        centroids: &[f32],
        bit_width: usize,
        existing: Option<(&[f32], &[f32])>,
    ) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        crate::encode::encode(
            vectors, n, dim, rotation, boundaries, centroids, bit_width, existing,
            &mut Vec::new(), &mut packed, &mut scales,
        );
        let (shift, scale_tq) = match existing {
            Some((s, sc)) => (s.to_vec(), sc.to_vec()),
            None => (Vec::new(), Vec::new()),
        };
        (packed, scales, shift, scale_tq)
    }

    use crate::rotation::Rotation;

    fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..(n * dim) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (((state >> 32) as u32) & 0x007FFFFF) | 0x3F800000;
            let uniform = f32::from_bits(bits) - 1.0;
            out.push(uniform * 2.0 - 1.0);
        }
        out
    }

    #[test]
    fn produces_expected_shape_for_bit_width_three() {
        let dim = 128;
        let n = 17;
        let rotation = Rotation::new(dim);
        let (boundaries, centroids) = codebook(3, dim);
        let vectors = make_vectors(n, dim, 0);

        let (packed, scales, _, _) = encode_owned(
            &vectors, n, dim, &rotation, &boundaries, &centroids, 3, None
        );

        let bytes_per_row = 3 * (dim / 8);
        assert_eq!(packed.len(), n * bytes_per_row);
        assert_eq!(scales.len(), n);
    }

    #[test]
    fn produces_expected_shape() {
        for &bit_width in &[2usize, 4] {
            let dim = 128;
            let n = 17;
            let rotation = Rotation::new(dim);
            let (boundaries, centroids) = codebook(bit_width, dim);
            let vectors = make_vectors(n, dim, 0);

            let (packed, scales, _, _) = encode_owned(
                &vectors, n, dim, &rotation, &boundaries, &centroids, bit_width, None
            );

            let bytes_per_row = bit_width * (dim / 8);
            assert_eq!(
                packed.len(),
                n * bytes_per_row,
                "wrong packed length for bits={}, dim={}",
                bit_width,
                dim
            );
            assert_eq!(scales.len(), n);
        }
    }

    #[test]
    fn scales_satisfy_rabitq_identity() {
        let dim = 128;
        let n = 10;
        let rotation = Rotation::new(dim);
        let (boundaries, centroids) = codebook(4, dim);
        let vectors = make_vectors(n, dim, 0);

        let (_, scales, _, _) =
            encode_owned(&vectors, n, dim, &rotation, &boundaries, &centroids, 4, None);

        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv_norm = 1.0 / norm;

            // Rotate the unit vector exactly as encode does: normalize,
            // then apply the block-Hadamard transform in place.
            let mut u_rot: Vec<f32> = row.iter().map(|&x| x * inv_norm).collect();
            rotation.apply(&mut u_rot);

            let mut inner = 0.0f64;
            for k in 0..dim {
                let mut code: usize = 0;
                for &b in &boundaries {
                    if u_rot[k] > b {
                        code += 1;
                    }
                }
                inner += (u_rot[k] as f64) * (centroids[code] as f64);
            }
            let expected_scale = norm as f64 / inner.max(1e-10);

            let rel_err =
                (scales[i] as f64 - expected_scale).abs() / expected_scale.abs().max(1e-10);
            assert!(
                rel_err < 1e-4,
                "scale identity broken at i={}: stored={}, expected={}, rel_err={}",
                i,
                scales[i],
                expected_scale,
                rel_err,
            );
        }
    }

    #[test]
    fn deterministic_output() {
        let dim = 128;
        let n = 5;
        let rotation = Rotation::new(dim);
        let (boundaries, centroids) = codebook(4, dim);
        let vectors = make_vectors(n, dim, 0);

        let (p1, s1, _, _) =
            encode_owned(&vectors, n, dim, &rotation, &boundaries, &centroids, 4, None);
        let (p2, s2, _, _) =
            encode_owned(&vectors, n, dim, &rotation, &boundaries, &centroids, 4, None);

        assert_eq!(p1, p2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn handles_zero_vector() {
        let dim = 128;
        let rotation = Rotation::new(dim);
        let (boundaries, centroids) = codebook(4, dim);
        let zeros = vec![0.0f32; dim];

        let (packed, scales, _, _) =
            encode_owned(&zeros, 1, dim, &rotation, &boundaries, &centroids, 4, None);

        assert_eq!(scales[0], 0.0);
        assert!(scales[0].is_finite());
        let bytes_per_row = 4 * (dim / 8);
        assert_eq!(packed.len(), bytes_per_row);
    }
}

/// From `tests/distortion.rs` — statistical validation of quantizer
/// distortion against the paper's Theorem 1.
mod distortion {
    use crate::codebook::codebook;
    use crate::TurboQuantIndex;
    use statrs::distribution::{Beta, Continuous};

    const PAPER_MSE: &[(usize, f64)] = &[(2, 0.1175), (3, 0.03454), (4, 0.009497)];

    #[test]
    fn codebook_mse_matches_paper_at_high_dim() {
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
            mse += simpson(
                |x: f64| (x - c).powi(2) * beta.pdf((x + 1.0) / 2.0) / 2.0,
                lo,
                hi,
                4000,
            );
        }
        mse
    }

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
    fn pipeline_self_score_is_unbiased() {
        let dim = 1536;
        let n = 500;
        let vectors = unit_sphere_vectors(n, dim, 42);

        for &(bits, _) in PAPER_MSE {
            let stats = self_score_stats(&vectors, dim, bits);
            let deficit = (1.0 - stats.mean).abs();
            assert!(
                deficit < 0.005,
                "bits={}: corrected self-score mean = {:.5}, deficit from 1.0 = {:.5} \
                 (correction should make this ~0 at all bit widths)",
                bits,
                stats.mean,
                deficit,
            );
        }
    }

    #[test]
    fn cross_query_variance_tightens_with_more_bits() {
        let dim = 512;
        let n = 200;
        let db = unit_sphere_vectors(n, dim, 0);
        let queries = unit_sphere_vectors(n, dim, 1);

        let s2 = cross_score_stats(&db, &queries, dim, 2);
        let s4 = cross_score_stats(&db, &queries, dim, 4);

        assert!(
            s4.stddev < s2.stddev,
            "4-bit cross-score stddev {:.4} not tighter than 2-bit {:.4} — bits may not be plumbed through",
            s4.stddev,
            s2.stddev,
        );
    }

    #[test]
    fn self_query_recall_at_1() {
        let dim = 512;
        let n = 200;
        let vectors = unit_sphere_vectors(n, dim, 0);

        let mut index = TurboQuantIndex::new(dim, 4).unwrap();
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
        assert!(recall >= 0.99, "recall@1 = {:.3} below 0.99 threshold", recall);
    }

    struct ScoreStats {
        mean: f64,
        stddev: f64,
    }

    fn self_score_stats(vectors: &[f32], dim: usize, bits: usize) -> ScoreStats {
        let n = vectors.len() / dim;
        let mut index = TurboQuantIndex::new(dim, bits).unwrap();
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

    fn cross_score_stats(database: &[f32], queries: &[f32], dim: usize, bits: usize) -> ScoreStats {
        let n_q = queries.len() / dim;
        let mut index = TurboQuantIndex::new(dim, bits).unwrap();
        index.add(database);
        index.prepare();

        let mut scores = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let q = &queries[i * dim..(i + 1) * dim];
            let results = index.search(q, 1);
            scores.push(results.scores_for_query(0)[0] as f64);
        }

        let mean = scores.iter().sum::<f64>() / n_q as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n_q as f64;
        ScoreStats { mean, stddev: variance.sqrt() }
    }

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
}

/// From `tests/core_encode_hardening.rs` — regression tests for #116/#117/#129.
mod core_encode_hardening {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::codebook::codebook;
    /// Test shim preserving the old owned-return shape. `encode` no
    /// longer fits — a calibration comes only from an explicit
    /// `calibrate` call — so the trailing pair is whatever the caller
    /// passed in, echoed back for the tests that assert against it.
    #[allow(clippy::too_many_arguments)]
    fn encode_owned(
        vectors: &[f32],
        n: usize,
        dim: usize,
        rotation: &crate::rotation::Rotation,
        boundaries: &[f32],
        centroids: &[f32],
        bit_width: usize,
        existing: Option<(&[f32], &[f32])>,
    ) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        crate::encode::encode(
            vectors, n, dim, rotation, boundaries, centroids, bit_width, existing,
            &mut Vec::new(), &mut packed, &mut scales,
        );
        let (shift, scale_tq) = match existing {
            Some((s, sc)) => (s.to_vec(), sc.to_vec()),
            None => (Vec::new(), Vec::new()),
        };
        (packed, scales, shift, scale_tq)
    }

    use crate::rotation::Rotation;
    use crate::{AddError, IdMapIndex, TurboQuantIndex};

    fn noise(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let raw = (*state >> 40) as u32;
        raw as f32 / (1u32 << 23) as f32 - 1.0
    }

    #[test]
    fn ood_vector_under_frozen_calibration_does_not_dominate_topk() {
        let dim = 64;
        let n = 1200;
        let mut index = TurboQuantIndex::new(dim, 4).unwrap();

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut vectors = vec![0.0f32; n * dim];
        for row in vectors.chunks_mut(dim) {
            row[0] = 1.0;
            for coord in row.iter_mut().skip(1) {
                *coord = 0.01 * noise(&mut state);
            }
        }
        index.add(&vectors);

        let mut ood = vec![0.0f32; dim];
        ood[0] = -1.0;
        index.add(&ood);
        let ood_slot = n as i64;

        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let results = index.search(&query, 5);

        let top_indices = results.indices_for_query(0);
        let top_scores = results.scores_for_query(0);
        assert!(
            !top_indices.contains(&ood_slot),
            "out-of-distribution vector (slot {ood_slot}) reached the top-5: \
             indices {top_indices:?}, scores {top_scores:?}",
        );
        for &s in top_scores {
            assert!(
                s.is_finite() && s.abs() < 10.0,
                "top-5 score {s} is outside the plausible inner-product range \
                 (scale explosion): scores {top_scores:?}",
            );
        }
    }

    #[test]
    fn degenerate_reconstruction_scale_is_zero_not_exploded() {
        let dim = 64;
        let n = 1200;
        let mut state = 0xdead_beef_dead_beefu64;
        let mut vectors = vec![0.0f32; n * dim];
        for row in vectors.chunks_mut(dim) {
            row[0] = 1.0;
            for coord in row.iter_mut().skip(1) {
                *coord = 0.01 * noise(&mut state);
            }
        }

        let rotation = Rotation::new(dim);
        let (boundaries, centroids) = codebook(4, dim);
        // Fit the frozen pair the way the index now does: explicitly,
        // from the cluster as the sample.
        let (shift, scale_tq) = crate::encode::fit_calibration(
            &vectors, n, dim, &rotation, &centroids, &mut Vec::new(),
        );

        let mut ood = vec![0.0f32; dim];
        ood[0] = -1.0;
        let (_, scales, _, _) = encode_owned(
            &ood, 1, dim, &rotation, &boundaries, &centroids, 4,
            Some((&shift, &scale_tq))
        );
        assert!(
            scales[0].abs() < 10.0,
            "degenerate reconstruction produced exploded scale {}",
            scales[0],
        );

        let zero = vec![0.0f32; dim];
        let (_, zero_scales, _, _) = encode_owned(
            &zero, 1, dim, &rotation, &boundaries, &centroids, 4,
            Some((&shift, &scale_tq))
        );
        assert_eq!(zero_scales[0], 0.0, "zero vector must keep scale 0");

        let mut nan_vec = vec![0.5f32; dim];
        nan_vec[3] = f32::NAN;
        let (_, nan_scales, _, _) = encode_owned(
            &nan_vec, 1, dim, &rotation, &boundaries, &centroids, 4,
            Some((&shift, &scale_tq))
        );
        assert_eq!(
            nan_scales[0], 0.0,
            "NaN input must store scale 0, got {}",
            nan_scales[0],
        );
    }

    #[test]
    fn near_orthogonal_window_under_frozen_calibration_is_bounded() {
        let dim = 64;
        let n = 1200;
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut cluster = vec![0.0f32; n * dim];
        for row in cluster.chunks_mut(dim) {
            row[0] = 1.0;
            for coord in row.iter_mut().skip(1) {
                *coord = 0.01 * noise(&mut state);
            }
        }

        let rotation = Rotation::new(dim);
        let (boundaries, centroids) = codebook(4, dim);
        let (shift, scale_tq) = crate::encode::fit_calibration(
            &cluster, n, dim, &rotation, &centroids, &mut Vec::new(),
        );
        let steps = 720;
        let mut sweep = vec![0.0f32; steps * dim];
        for (t, row) in sweep.chunks_mut(dim).enumerate() {
            let theta = std::f32::consts::PI * (t as f32 + 0.5) / steps as f32;
            row[0] = theta.cos();
            row[1] = theta.sin();
        }
        let (_, sweep_scales, _, _) = encode_owned(
            &sweep, steps, dim, &rotation, &boundaries, &centroids, 4,
            Some((&shift, &scale_tq))
        );
        for (t, &s) in sweep_scales.iter().enumerate() {
            assert!(
                s.is_finite() && (0.0..=10.0).contains(&s),
                "sweep step {t}: stored scale {s} escapes the [0, 1/EPS] bound",
            );
        }

        let mut index = TurboQuantIndex::new(dim, 4).unwrap();
        index.calibrate(&cluster).unwrap();
        index.add(&cluster);
        for theta in [1.6275f32, 1.629281051794] {
            let mut v = vec![0.0f32; dim];
            v[0] = theta.cos();
            v[1] = theta.sin();
            index.add(&v);
        }
        let pathological: Vec<i64> = vec![n as i64, n as i64 + 1];

        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let results = index.search(&query, 5);
        let top_indices = results.indices_for_query(0);
        let top_scores = results.scores_for_query(0);
        for slot in &pathological {
            assert!(
                !top_indices.contains(slot),
                "near-orthogonal vector (slot {slot}) reached the top-5: \
                 indices {top_indices:?}, scores {top_scores:?}",
            );
        }
        for &s in top_scores {
            assert!(
                s.is_finite() && s.abs() < 10.0,
                "top-5 score {s} is outside the plausible range: {top_scores:?}",
            );
        }
    }

    #[test]
    #[should_panic(expected = "multiple of 8")]
    fn encode_rejects_dim_not_multiple_of_8() {
        let dim = 12;
        let n = 4;
        // encode asserts `dim % 8 == 0` at its top, before it ever touches
        // the rotation — so this fires encode's own guard. (A dim=12
        // rotation can't be built anyway; `Rotation::new` enforces the
        // same rule.)
        let rotation = Rotation::new(8);
        let (boundaries, centroids) = codebook(2, dim);
        let vectors = vec![0.25f32; n * dim];
        let _ = encode_owned(
            &vectors, n, dim, &rotation, &boundaries, &centroids, 2, None
        );
    }

    #[test]
    fn lazy_add_2d_length_panic_does_not_commit_dim() {
        let mut index = TurboQuantIndex::new_lazy(4).unwrap();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = index.add_2d(&vec![0.5f32; 100], 64);
        }));
        assert!(result.is_err(), "add_2d must panic on non-multiple length");

        assert_eq!(
            index.dim_opt(),
            None,
            "failed add_2d left the lazy index wedged with a committed dim",
        );
        index
            .add_2d(&vec![0.5f32; 16], 8)
            .expect("fresh add with a different dim must succeed after the failed add");
        assert_eq!(index.dim_opt(), Some(8));
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn lazy_add_with_ids_2d_length_error_does_not_commit_dim() {
        let mut index = IdMapIndex::new_lazy(4).unwrap();
        let err = index
            .add_with_ids_2d(&vec![0.5f32; 100], 64, &[1, 2])
            .unwrap_err();
        assert!(matches!(err, AddError::VectorBufferNotMultipleOfDim { .. }));
        assert_eq!(index.dim_opt(), None);

        index
            .add_with_ids_2d(&vec![0.5f32; 16], 8, &[1, 2])
            .expect("fresh add with a different dim must succeed after the failed add");
        assert_eq!(index.dim_opt(), Some(8));
    }
}

/// Query-side LUT quantization invariants (#332, #335). Reaches
/// `build_query_neon_lut_from_slice` directly because the u8 LUT is the
/// object under test, not the score it eventually produces.
mod query_lut_quantization {
    use crate::codebook::codebook;
    use crate::search::build_query_neon_lut_from_slice;

    fn q_row(dim: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..dim)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((s >> 33) as f32 / (1u32 << 31) as f32) - 1.0) * scale
            })
            .collect()
    }

    /// The NEON kernel adds the two nibble lookups in u8 space
    /// (`vaddq_u8`) before widening, so any *pair* of entries must sum to
    /// at most 255. That caps the LUT at 127 — 128 + 128 would wrap. The
    /// x86 kernels accumulate into i16 and could carry 255, but are held
    /// at 127 so both arches round identically. Raising the cap therefore
    /// requires changing the NEON kernel first, not just this constant.
    #[test]
    fn lut_entries_never_exceed_the_neon_u8_pair_bound() {
        for &bits in &[2usize, 4] {
            for &dim in &[256usize, 768, 1536] {
                let (_, centroids) = codebook(bits, dim);
                for seed in 0..8u64 {
                    let lut = build_query_neon_lut_from_slice(
                        &q_row(dim, 0xBEEF + seed, 1.0),
                        &centroids,
                        bits,
                        dim,
                    );
                    let max = *lut.uint8_luts.iter().max().unwrap();
                    assert!(max <= 127, "LUT entry {max} > 127 at bits={bits} dim={dim}");
                    let n_groups = dim / (8 / bits);
                    for g in 0..n_groups {
                        for hi in 0..16 {
                            for lo in 0..16 {
                                let sum = lut.uint8_luts[g * 32 + hi] as u16
                                    + lut.uint8_luts[g * 32 + 16 + lo] as u16;
                                assert!(sum <= 255, "nibble pair sum {sum} overflows u8");
                            }
                        }
                    }
                }
            }
        }
    }

    /// A power-of-two rescale of the query is exact in f32, so the u8 LUT
    /// must come out byte-identical and `scale`/`bias` must carry the
    /// factor. Spans 1e30 down to 1e-30 (#335).
    #[test]
    fn lut_bytes_are_invariant_to_power_of_two_query_scaling() {
        for &bits in &[2usize, 4] {
            let dim = 768usize;
            let (_, centroids) = codebook(bits, dim);
            let base_row = q_row(dim, 0xF00D, 1.0);
            let base = build_query_neon_lut_from_slice(&base_row, &centroids, bits, dim);
            for e in -100i32..=100 {
                let c = f32::powi(2.0, e);
                let row: Vec<f32> = base_row.iter().map(|v| v * c).collect();
                let lut = build_query_neon_lut_from_slice(&row, &centroids, bits, dim);
                assert_eq!(lut.uint8_luts, base.uint8_luts, "LUT bytes changed at 2^{e}");
                assert_eq!(lut.scale, base.scale * c, "scale not proportional at 2^{e}");
                assert_eq!(lut.bias, base.bias * c, "bias not proportional at 2^{e}");
            }
        }
    }

    /// A query that rotates to all-zero has no span; the LUT is all zeros
    /// and `scale` must stay finite and non-zero so downstream scoring
    /// produces 0.0, not NaN.
    #[test]
    fn zero_span_query_yields_finite_scale() {
        let (dim, bits) = (256usize, 4usize);
        let (_, centroids) = codebook(bits, dim);
        let lut = build_query_neon_lut_from_slice(&vec![0.0f32; dim], &centroids, bits, dim);
        assert!(lut.uint8_luts.iter().all(|&b| b == 0));
        assert!(lut.scale.is_finite() && lut.scale > 0.0);
        assert!(lut.bias.is_finite());
    }

}

/// #307(2): the NEON partial-block tail clamp at `search.rs:171`.
///
/// The kernel writes `BLOCK` scores per block, so on a final partial block
/// it must clamp at `n_vectors` and pad the remaining lanes with
/// `NEG_INFINITY` (the invariant documented at `search.rs:1016`). Dropping
/// the `.min(n_vectors)` takes the full-block fast path instead, which both
/// reads past the end of `vec_scales` and fills the pad lanes with real
/// products. Nothing downstream notices, because `neon_block_topk_update`
/// clamps independently — hence this direct assertion on the kernel output.
/// `vec_scales` is sized to exactly `n_vectors` so the over-read is also a
/// genuine heap overflow under ASAN.
#[cfg(target_arch = "aarch64")]
mod neon_tail_clamp {
    use crate::search::score_4bit_block_neon;
    use crate::BLOCK;

    #[test]
    fn partial_block_pads_with_neg_infinity() {
        let n_byte_groups = 4;
        let n_vectors = 20;
        assert!(n_vectors % BLOCK != 0, "test needs a partial final block");

        let codes: Vec<u8> = (0..n_byte_groups * BLOCK).map(|i| (i * 37 % 256) as u8).collect();
        let luts: Vec<u8> = (0..n_byte_groups * 32).map(|i| (i * 13 % 128) as u8).collect();
        let vec_scales: Vec<f32> = (0..n_vectors).map(|i| 1.0 + i as f32 * 0.01).collect();

        let mut out = [0.0f32; BLOCK];
        unsafe {
            score_4bit_block_neon(
                &codes,
                &luts,
                0,
                n_byte_groups,
                0.01,
                -1.0,
                &vec_scales,
                0,
                n_vectors,
                &mut out,
            );
        }

        for (lane, &v) in out.iter().enumerate() {
            if lane < n_vectors {
                assert!(v.is_finite(), "lane {lane} should hold a real score, got {v}");
            } else {
                assert_eq!(
                    v,
                    f32::NEG_INFINITY,
                    "pad lane {lane} past n_vectors={n_vectors} must be NEG_INFINITY"
                );
            }
        }
    }
}


/// The test-only encode panic switch must be scoped to the calling
/// thread (#373): `cargo test` runs a binary's tests in parallel
/// threads, and a process-global one-shot armed by one test can be
/// consumed by a concurrent `add` in another.
mod encode_panic_switch {
    use crate::TurboQuantIndex;

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * dim];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        }
        v
    }

    /// Arming the switch must not leak into any other test in this
    /// binary: `cargo test` runs them in parallel threads, and this one
    /// does full validation plus `packed()` before the check, so a
    /// process-global flag could be consumed by a concurrent `add`
    /// instead (#373). Spawning adds on other threads while armed pins
    /// that the scoping is thread-local.
    #[test]
    fn the_panic_switch_does_not_leak_to_other_threads() {
        let dim = 32;
        TurboQuantIndex::force_encode_panic(true);
        let handles: Vec<_> = (0..4)
            .map(|k| {
                std::thread::spawn(move || {
                    let mut other = TurboQuantIndex::new(dim, 4).unwrap();
                    other.add_2d(&rows(50, dim, 100 + k), dim).unwrap();
                    other.len()
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().expect("a concurrent add consumed the switch"), 50);
        }
        // Still armed for THIS thread.
        let mut mine = TurboQuantIndex::new(dim, 4).unwrap();
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mine.add_2d(&rows(10, dim, 1), dim)
        }));
        assert!(failed.is_err(), "the switch was consumed by another thread");
    }
}

/// `encode_and_append`'s unwind guard: an append to a calibrated index.
///
/// The guard is the *only* thing standing between a panicking `encode` and
/// an index whose `packed_codes` / `scales` have been moved out of `self`
/// and never put back — `n_vectors` still counting rows whose codes are
/// gone. That state does not surface as an error: `len()` still reports
/// the old count, so the loss is silent until a search or a save reads the
/// missing codes.
#[cfg(test)]
mod settled_append_unwind {
    use crate::{CalibrationState, TurboQuantIndex};

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * dim];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        }
        v
    }

    #[test]
    fn a_panicking_append_to_a_settled_index_loses_nothing() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        // Calibrate explicitly (adds never fit), then add: every later
        // add takes the plain append path under the committed pair.
        idx.calibrate_2d(&rows(1024, dim, 9), dim).unwrap();
        idx.add_2d(&rows(1200, dim, 1), dim).unwrap();
        assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
        let before = idx.to_bytes();

        TurboQuantIndex::force_encode_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(300, dim, 2), dim)
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");

        // Nothing about the index changed — not the row count, not the
        // calibration, and not a single stored byte. The byte comparison
        // is what catches the silent form: `len()` alone still reads
        // 1200 even when the codes have been moved out of `self`.
        assert_eq!(idx.len(), 1200, "a panicking append changed the row count");
        assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
        assert_eq!(
            idx.to_bytes(),
            before,
            "a panicking append changed the index's serialized state"
        );

        // Still searchable, and self-recall is intact: row 7 is its own
        // nearest neighbour, which it cannot be if its codes were lost.
        let probe = &rows(1200, dim, 1)[7 * dim..8 * dim];
        let res = idx.search(probe, 5);
        assert_eq!(res.indices[0], 7, "self-recall broken after a caught panic");

        // And the index still accepts work afterwards.
        idx.add_2d(&rows(300, dim, 2), dim).unwrap();
        assert_eq!(idx.len(), 1500);
        let res = idx.search(probe, 5);
        assert_eq!(res.indices[0], 7);
    }

    /// The guard's `truncate` calls, which the test above cannot reach.
    ///
    /// `force_encode_panic` fires *before* `encode` runs, so at unwind
    /// time both buffers are still at their pre-call lengths and both
    /// `truncate`s are no-ops — deleting either one keeps that test (and
    /// the whole suite) green. `force_encode_panic_after_append` unwinds
    /// from inside `encode` with this batch already appended, which is
    /// the shape the guard was written for: buffers longer than the
    /// caller left them, `n_vectors` not yet incremented.
    #[test]
    fn a_panic_after_a_partial_append_truncates_both_buffers() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.calibrate_2d(&rows(1024, dim, 9), dim).unwrap();
        idx.add_2d(&rows(1200, dim, 1), dim).unwrap();
        assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
        let packed_len = idx.packed().len();
        let scales_len = idx.scales.len();
        let before = idx.to_bytes();

        TurboQuantIndex::force_encode_panic_after_append(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(300, dim, 2), dim)
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");

        // The 300 appended rows must be gone from BOTH buffers. Left in
        // place they outrun `n_vectors`, so every later append writes
        // over rows the index believes it never accepted.
        assert_eq!(
            idx.scales.len(),
            scales_len,
            "the scales buffer kept the failed batch's rows",
        );
        assert_eq!(
            idx.packed().len(),
            packed_len,
            "the packed buffer kept the failed batch's rows",
        );
        assert_eq!(idx.len(), 1200);
        assert_eq!(idx.to_bytes(), before, "a caught partial append changed the index");

        // The next append lands where the failed one would have, and the
        // buffers stay in step with the row count.
        idx.add_2d(&rows(300, dim, 2), dim).unwrap();
        assert_eq!(idx.len(), 1500);
        assert_eq!(idx.scales.len(), scales_len + 300);
        let probe = &rows(1200, dim, 1)[7 * dim..8 * dim];
        assert_eq!(idx.search(probe, 5).indices[0], 7);
    }

    /// The guard's *other* arm: the v6-load window, where the blocked
    /// cache is authoritative and `packed_codes` is deliberately left
    /// unset so the O(n·dim) materialization never runs.
    ///
    /// There, the taken buffer is a temp holding only the new rows, so
    /// restoring it under the `packed_codes` lock would publish an index
    /// whose packed rows are empty while `n_vectors` counts the loaded
    /// ones. Nothing else in the suite drives a panicking add in this
    /// window: mutating the guard's `if !lazy_append` to `if true`
    /// otherwise passes everything.
    #[test]
    fn a_panic_during_a_lazy_v6_append_leaves_the_blocked_cache_authoritative() {
        let dim = 64;
        let mut src = TurboQuantIndex::new(dim, 4).unwrap();
        src.add_2d(&rows(1200, dim, 1), dim).unwrap();
        let bytes = src.to_bytes();

        // A v6 load seeds the blocked cache from the file and leaves the
        // packed rows unmaterialized — the window `lazy_append` names.
        let mut idx = TurboQuantIndex::from_bytes(&bytes).unwrap();
        assert!(idx.packed_codes.get().is_none(), "v6 load should not materialize packed");
        assert!(idx.blocked.get().is_some(), "v6 load should seed the blocked cache");
        let before = idx.to_bytes();

        TurboQuantIndex::force_encode_panic_after_append(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(300, dim, 2), dim)
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");

        assert!(
            idx.packed_codes.get().is_none(),
            "the guard published the lazy temp — packed rows are now the new batch's only",
        );
        assert_eq!(idx.len(), 1200);
        assert_eq!(idx.to_bytes(), before, "a caught lazy append changed the index");
        let probe = &rows(1200, dim, 1)[7 * dim..8 * dim];
        assert_eq!(idx.search(probe, 5).indices[0], 7, "self-recall broken after a caught panic");

        // And the retry still appends correctly through the lazy path.
        let mut idx = TurboQuantIndex::from_bytes(&bytes).unwrap();
        idx.add_2d(&rows(300, dim, 2), dim).unwrap();
        assert_eq!(idx.len(), 1500);
        assert_eq!(idx.search(probe, 5).indices[0], 7);
    }
}

/// #380: state committed before the work that has to succeed for it to
/// be true.
///
/// Two sites, one shape, but they differ in how live they are.
/// `add_2d` committing a lazy index's dim before the encode is a real
/// defect with a real unwind behind it. `IdMapIndex::remove` mutating
/// its tables before the inner `swap_remove` is ordering hardening:
/// `swap_remove` has no unwind reachable from that caller today — its
/// one documented panic is the `idx < n_vectors` assert, and the slot
/// comes from the id table (see `force_swap_remove_panic`). That test
/// pins the statement order against a future fallible inner removal
/// rather than reproducing a bug reachable from the public API.
mod state_before_fallible_work {
    use crate::{AddError, IdMapIndex, TurboQuantIndex};

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * dim];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        }
        v
    }

    /// A panic in the inner removal must leave the id tables untouched:
    /// the id still resolves, the tables still agree with the inner
    /// index, and a retry removes exactly one vector.
    ///
    /// The switch fires before `swap_remove` touches anything, so this
    /// pins the caller's statement order and only that. It deliberately
    /// does not claim `remove` is atomic: a panic partway through
    /// `swap_remove` would leave the inner index short against full
    /// tables, which the ordering cannot address.
    #[test]
    fn a_panicking_inner_removal_leaves_the_id_tables_intact() {
        let dim = 64;
        let mut idx = IdMapIndex::new(dim, 4).unwrap();
        let ids: Vec<u64> = (0..1200).collect();
        idx.add_with_ids_2d(&rows(1200, dim, 1), dim, &ids).unwrap();

        TurboQuantIndex::force_swap_remove_panic(true);
        let failed =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| idx.remove(7)));
        assert!(failed.is_err(), "the forced swap_remove panic should have propagated");

        // `IdMapIndex::len()` is `slot_to_id.len()`, so it pins the table
        // length only. The stored row count is a separate fact: assert it
        // through the effective k a search clamps to, which comes from
        // the inner index's length.
        assert_eq!(idx.len(), 1200, "a caught panic changed the id table length");
        let (_, all) = idx.search(&rows(1, dim, 3), 5000);
        assert_eq!(all.len(), 1200, "a caught panic changed the stored row count");
        assert!(
            idx.contains(7),
            "a caught panic dropped the id from the map while its vector is still stored",
        );
        // The desync the reorder prevents: `slot_to_id` one longer than
        // the inner index, so every later remove computes `last` off the
        // wrong length. An allowlist search over every id is the cheapest
        // observable proof both tables still agree with `inner`.
        let (_, got) = idx.search_with_allowlist(&rows(1, dim, 3), 1200, Some(&ids)).unwrap();
        assert_eq!(got.len(), 1200, "the allowlist lost an id after a caught panic");

        // And a retry removes exactly the one vector, no more.
        assert!(idx.remove(7));
        assert_eq!(idx.len(), 1199);
        assert!(!idx.contains(7));
        let probe = &rows(1200, dim, 1)[9 * dim..10 * dim];
        assert_eq!(idx.search(probe, 5).1[0], 9, "self-recall broken after the retry");
    }

    /// A panic in the first add of a lazy index must leave it lazy: the
    /// dim is not committed, so a follow-up `add_2d` at a *different*
    /// dim gets the fresh start #129 established rather than a
    /// `DimMismatch` naming a dim the index never actually stored.
    #[test]
    fn a_panicking_first_add_leaves_a_lazy_index_lazy() {
        let mut idx = TurboQuantIndex::new_lazy(4).unwrap();
        assert_eq!(idx.dim_opt(), None);

        TurboQuantIndex::force_encode_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(10, 64, 1), 64)
        }));
        assert!(failed.is_err(), "the forced encode panic should have propagated");

        assert_eq!(idx.len(), 0, "a caught panic left rows behind");
        assert_eq!(
            idx.dim_opt(),
            None,
            "a caught panic wedged the lazy index at a committed dim with no vectors",
        );

        // The user-visible consequence: retrying at a different dim.
        idx.add_2d(&rows(10, 128, 2), 128).expect("a lazy index should still accept a new dim");
        assert_eq!(idx.dim_opt(), Some(128));
        assert_eq!(idx.len(), 10);
        // Rolling back the dim alone is not enough, and the `add_2d`
        // above is what proves it: with the rotation cache left behind it
        // panics ("rotation input row must have length dim, left: 128,
        // right: 64") instead of returning, so the `.expect` fires. That
        // failure is loud, not silent. The recall check below covers the
        // case the rotation assert hides — a stale `boundaries`/
        // `centroids` pair for the old dim is length-compatible, so it
        // would be accepted and mis-quantize every row rather than panic.
        let probe = &rows(10, 128, 2)[3 * 128..4 * 128];
        assert_eq!(idx.search(probe, 3).indices[0], 3, "self-recall broken at the new dim");

        // The committed dim is now real: a mismatched add is rejected.
        assert!(matches!(
            idx.add_2d(&rows(1, 64, 3), 64),
            Err(AddError::DimMismatch { existing: 128, got: 64 })
        ));
    }
}

/// #388: the eager `add` path must not publish codes or scales before the
/// blocked-cache repack, which can panic.
///
/// The `perf/op-hillclimb` merge moved `n_vectors` to last so a repack
/// panic could not leave the count ahead of the cache — but it published
/// `packed_codes` and `scales` *before* the repack, so a caught panic left
/// those holding the new rows while `n_vectors` still read the old count.
/// The next add then addresses past the orphans: silent slot corruption
/// rather than a detectable inconsistency.
mod eager_add_unwind {
    use crate::TurboQuantIndex;

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * dim];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        }
        v
    }

    #[test]
    fn a_panicking_cache_repack_leaves_the_index_at_its_pre_call_state() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.add_2d(&rows(1200, dim, 1), dim).unwrap();
        // The patch branch needs BOTH layouts live. Since #475 an add
        // converges the index to blocked-only, so the cache alone is not
        // enough — the next add would take `lazy_append` and never reach
        // the repack. Materializing the packed rows through the public
        // accessor restores the both-live state the branch is defined for.
        idx.prepare();
        let _ = idx.packed_codes();
        let before_len = idx.len();
        let before_bytes = idx.to_bytes();

        TurboQuantIndex::force_repack_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(100, dim, 2), dim)
        }));
        assert!(failed.is_err(), "the forced repack panic should have propagated");

        assert_eq!(idx.len(), before_len, "a caught repack panic changed the row count");
        assert_eq!(
            idx.to_bytes(),
            before_bytes,
            "a caught repack panic left codes or scales holding the failed batch"
        );

        // And the index is still usable: a retry appends cleanly.
        idx.add_2d(&rows(100, dim, 2), dim).unwrap();
        assert_eq!(idx.len(), before_len + 100);
        let res = idx.search(&rows(1, dim, 3), 10);
        assert!(
            res.indices.iter().all(|&i| (i as usize) < idx.len()),
            "search returned a slot past the end after the retry"
        );
    }
}

/// The refit's two unvalidated mechanics: unwind safety mid-refit, and
/// degenerate rows surviving one.
mod refit_hardening {
    use crate::{CalibrationState, TurboQuantIndex};

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * dim];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        }
        v
    }

    /// A panic inside the refit's re-encode must leave the index exactly
    /// as it was: the re-encode writes only into locals and the commit
    /// block runs after every fallible step, so an unwind discards the
    /// half-built buffers and keeps the old pair. Forced with the
    /// post-append switch, which fires inside `encode_prerotated` — the
    /// kernel the refit drives — with the first chunk already appended
    /// to the (local) output buffers.
    #[test]
    fn a_panicking_refit_leaves_the_index_untouched() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.calibrate(&rows(1024, dim, 1)).unwrap();
        idx.add(&rows(1200, dim, 2));
        let before_bytes = idx.to_bytes();
        let before_pair = idx.tqplus_shift().to_vec();

        crate::encode::force_panic_after_append(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.calibrate(&rows(1024, dim, 3))
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");

        assert_eq!(idx.len(), 1200, "a failed refit changed the row count");
        assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
        assert_eq!(
            idx.tqplus_shift(),
            &before_pair[..],
            "a failed refit committed the new pair"
        );
        assert_eq!(
            idx.to_bytes(),
            before_bytes,
            "a failed refit changed the serialized state"
        );
        let probe = &rows(1200, dim, 2)[7 * dim..8 * dim];
        assert_eq!(idx.search(probe, 1).indices[0], 7);

        // And the index still accepts the same refit afterwards.
        idx.calibrate(&rows(1024, dim, 3)).unwrap();
        assert_ne!(idx.tqplus_shift(), &before_pair[..]);
        assert_eq!(idx.search(probe, 1).indices[0], 7);
    }

    /// A degenerate row — stored scale exactly 0.0 — stays degenerate
    /// through a refit: its effective norm is `0.0 * <x,x> = 0.0`, so
    /// the kernel stores 0.0 again, and it keeps scoring zero rather
    /// than exploding under the new pair.
    #[test]
    fn a_degenerate_row_survives_a_refit_as_degenerate() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.calibrate(&rows(1024, dim, 4)).unwrap();
        idx.add(&rows(20, dim, 5));
        // A zero row is finite, so `add` accepts it; its norm is zero
        // and it stores scale 0.0 (the degenerate marker).
        idx.add(&vec![0.0f32; dim]);
        assert_eq!(idx.scales()[20], 0.0, "zero row must store scale 0.0");

        idx.calibrate(&rows(1024, dim, 6)).unwrap();

        assert_eq!(
            idx.scales()[20],
            0.0,
            "the degenerate row's scale must stay 0.0 through a refit"
        );
        assert!(
            idx.scales()[..20].iter().all(|&s| s > 0.0),
            "live rows must keep positive scales"
        );
        let probe = &rows(20, dim, 5)[3 * dim..4 * dim];
        assert_eq!(idx.search(probe, 1).indices[0], 3);
    }
}

/// `calibrate`'s fit-unwind arm: a panic inside the fit must leave the
/// index untouched, and on a lazy index it must also reset the
/// dim-shaped caches seeded before the fit — a retry at a *different*
/// dim must not reuse a rotation or codebook built for the first one.
mod calibrate_fit_unwind {
    use crate::{CalibrationState, TurboQuantIndex};

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * dim];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        }
        v
    }

    #[test]
    fn a_panicking_fit_on_a_lazy_index_resets_the_seeded_caches() {
        let mut idx = TurboQuantIndex::new_lazy(4).unwrap();
        TurboQuantIndex::force_fit_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.calibrate_2d(&rows(64, 64, 1), 64)
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");
        assert_eq!(idx.dim_opt(), None, "a failed fit committed the dim");
        assert_eq!(idx.calibration_state(), CalibrationState::Uncalibrated);

        // The whole point of the cache reset: a retry at a different dim
        // must fit and work, which it cannot if the dim-64 rotation or
        // codebook survived.
        idx.calibrate_2d(&rows(64, 128, 2), 128).unwrap();
        assert_eq!(idx.dim_opt(), Some(128));
        let data = rows(50, 128, 3);
        idx.add_2d(&data, 128).unwrap();
        assert_eq!(idx.search(&data[..128], 1).indices[0], 0);
    }

    #[test]
    fn a_panicking_fit_on_a_populated_index_changes_nothing() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.calibrate(&rows(1024, dim, 4)).unwrap();
        idx.add(&rows(200, dim, 5));
        let pair = idx.tqplus_shift().to_vec();
        let bytes = idx.to_bytes();

        TurboQuantIndex::force_fit_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.calibrate(&rows(1024, dim, 6))
        }));
        assert!(failed.is_err());
        assert_eq!(idx.tqplus_shift(), &pair[..]);
        assert_eq!(idx.to_bytes(), bytes);

        // And the same call succeeds once the switch is gone.
        idx.calibrate(&rows(1024, dim, 6)).unwrap();
        assert_ne!(idx.tqplus_shift(), &pair[..]);
    }
}
