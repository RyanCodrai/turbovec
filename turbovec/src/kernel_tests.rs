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
    /// Test shim preserving the old owned-return encode signature.
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
        let (shift, scale_tq) = crate::encode::encode(
            vectors, n, dim, rotation, boundaries, centroids, bit_width, existing,
            &mut Vec::new(), &mut packed, &mut scales,
        );
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
    /// Test shim preserving the old owned-return encode signature.
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
        let (shift, scale_tq) = crate::encode::encode(
            vectors, n, dim, rotation, boundaries, centroids, bit_width, existing,
            &mut Vec::new(), &mut packed, &mut scales,
        );
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
        let (_, _, shift, scale_tq) = encode_owned(
            &vectors, n, dim, &rotation, &boundaries, &centroids, 4, None
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
        let (_, _, shift, scale_tq) = encode_owned(
            &cluster, n, dim, &rotation, &boundaries, &centroids, 4, None
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


/// #353: the warm-up buffer must only grow after a *successful* encode.
///
/// `encode_and_append`'s unwind guard restores the index without
/// incrementing `n_vectors`, so extending the buffer beforehand leaves
/// `warmup.len()/dim` permanently ahead of `n_vectors` — breaking the
/// documented "buffer row i is slot i" invariant and replaying the failed
/// batch's rows into the threshold re-encode, which resurrects rows the
/// index never accepted.
///
/// #361 generalizes that rule to the rest of the lifecycle: nothing that
/// has to stay in step with `n_vectors` — the stored codes, the committed
/// calibration, the warm-up buffer itself — may be left mutated after a
/// failed add. The threshold crossing has to commit before it re-encodes,
/// so it rolls the whole lot back on unwind instead.
mod warmup_unwind {
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

    #[test]
    fn a_panicking_add_does_not_grow_the_warmup_buffer() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.add_2d(&rows(200, dim, 1), dim).unwrap();
        assert_eq!(idx.len(), 200);

        // A batch that panics inside encode must leave the index exactly
        // as it was — including the warm-up buffer.
        TurboQuantIndex::force_encode_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(300, dim, 2), dim)
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");
        assert_eq!(idx.len(), 200, "a panicking add changed the row count");

        // Cross the threshold. If the buffer had grown by the failed
        // batch's 300 rows, the re-encode would replay 500 buffered rows
        // against 200 real slots and the total would overshoot.
        idx.add_2d(&rows(900, dim, 3), dim).unwrap();
        assert_eq!(
            idx.len(),
            200 + 900,
            "the failed batch's rows were resurrected by the re-encode"
        );
        let res = idx.search(&rows(1, dim, 4), 10);
        assert!(
            res.indices.iter().all(|&i| (i as usize) < idx.len()),
            "search returned a slot past the end of the index"
        );
    }

    /// A panic in the threshold crossing's re-encode must not empty the
    /// index. The crossing clears the stored codes and resets
    /// `n_vectors` before re-encoding, so without the unwind guard a
    /// caught panic leaves every previously-added row gone — and the
    /// index looks like a legitimately empty one.
    #[test]
    fn a_panicking_threshold_crossing_keeps_the_committed_rows() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        let warm = rows(999, dim, 11);
        idx.add_2d(&warm, dim).unwrap();
        let probe = &warm[0..dim];
        let before = idx.search(probe, 5).indices[0];

        // The fit succeeds; the first re-encode of the buffered rows
        // panics.
        TurboQuantIndex::force_encode_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(1, dim, 12), dim)
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");

        assert_eq!(idx.len(), 999, "the crossing lost the committed rows");
        assert_eq!(
            idx.calibration_state(),
            CalibrationState::WarmingUp,
            "a failed crossing left warm-up"
        );
        assert_eq!(
            idx.search(probe, 5).indices[0],
            before,
            "the stored codes no longer answer the query they did before"
        );

        // The index is still fully functional: a retry crosses cleanly.
        idx.add_2d(&rows(1, dim, 12), dim).unwrap();
        assert_eq!(idx.len(), 1000);
        assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
    }

    /// A panic in the calibration fit alone must not forfeit TQ+. The
    /// crossing takes the warm-up buffer before it fits, so without the
    /// unwind guard the index keeps all its rows but loses the buffer —
    /// every later add then reuses the committed identity calibration
    /// and the index can never fit one, with no error surface at all.
    #[test]
    fn a_panicking_calibration_fit_does_not_forfeit_tqplus() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.add_2d(&rows(999, dim, 21), dim).unwrap();

        TurboQuantIndex::force_fit_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(1, dim, 22), dim)
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");

        assert_eq!(idx.len(), 999, "a failed fit changed the row count");
        assert_eq!(
            idx.calibration_state(),
            CalibrationState::WarmingUp,
            "a failed fit dropped the warm-up buffer"
        );

        // The whole point: TQ+ is still reachable.
        idx.add_2d(&rows(1, dim, 22), dim).unwrap();
        assert_eq!(
            idx.calibration_state(),
            CalibrationState::Fitted,
            "the index can no longer fit a calibration"
        );
    }

    /// The crossing branch that has nothing buffered (a fresh index whose
    /// very first add clears the threshold) also leaves warm-up, and must
    /// do so only once the encode has succeeded.
    #[test]
    fn a_panicking_first_bulk_add_stays_in_warm_up() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();

        TurboQuantIndex::force_encode_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(1000, dim, 31), dim)
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");

        assert_eq!(idx.len(), 0);
        assert_eq!(
            idx.calibration_state(),
            CalibrationState::WarmingUp,
            "a failed first bulk add left warm-up"
        );

        idx.add_2d(&rows(1000, dim, 31), dim).unwrap();
        assert_eq!(idx.len(), 1000);
        assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
    }

    /// `add_parallelizes` is the fork-safety gate for the crossing
    /// (#364): it must be true for the single-row add that crosses the
    /// threshold, whose real work is the ~1000-row re-encode.
    #[test]
    fn add_parallelizes_flags_the_threshold_crossing() {
        let dim = 64;
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        assert!(!idx.add_parallelizes(1), "an empty index does not cross");
        idx.add_2d(&rows(999, dim, 41), dim).unwrap();
        assert!(
            idx.add_parallelizes(1),
            "the single-row add that crosses the threshold must pool"
        );
        idx.add_2d(&rows(1, dim, 42), dim).unwrap();
        assert!(
            !idx.add_parallelizes(1),
            "a fitted index has no buffer to re-encode"
        );
    }
}

/// `encode_and_append`'s unwind guard on the path the warm-up tests above
/// never reach: an append to an index whose calibration is already
/// **settled**.
///
/// Every existing unwind test drives a warming-up index, where the guard's
/// job is bounded by the warm-up ordering rule (#353). Past the threshold
/// the guard is the *only* thing standing between a panicking `encode` and
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
        // One bulk add past TQPLUS_MIN_SAMPLES: calibration is fitted and
        // warm-up is over, so every later add takes the plain append path.
        idx.add_2d(&rows(1200, dim, 1), dim).unwrap();
        assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
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
        assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
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
        idx.add_2d(&rows(1200, dim, 1), dim).unwrap();
        assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
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

    /// The #380 guarantee has to hold for an *uncalibrated* lazy index
    /// too, and that is not free: `new_lazy_uncalibrated` commits a
    /// dim-shaped identity `(shift, scale)` and drops `warmup` before the
    /// fallible encode, so those three have to roll back with the dim.
    ///
    /// Without that, the unwound index reports `dim == None` and
    /// `len() == 0` — it looks lazy and fresh — while still holding a
    /// calibration shaped for the abandoned dim. The retry then dies on
    /// `existing shift length must equal dim` inside `encode`, and
    /// because the retry's panic re-enters the same rollback, every
    /// later add at that dim dies the same way. `to_bytes` panics too,
    /// on a dim-0 sentinel beside a length-`dim` pair.
    #[test]
    fn a_panicking_first_add_leaves_an_uncalibrated_lazy_index_lazy() {
        use crate::CalibrationState;
        let mut idx = TurboQuantIndex::new_lazy_uncalibrated(4).unwrap();
        assert_eq!(idx.dim_opt(), None);
        assert!(idx.tqplus_shift().is_empty(), "nothing is committed before the first add");

        TurboQuantIndex::force_encode_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(10, 64, 1), 64)
        }));
        assert!(failed.is_err(), "the forced encode panic should have propagated");

        assert_eq!(idx.len(), 0, "a caught panic left rows behind");
        assert_eq!(idx.dim_opt(), None, "a caught panic wedged the lazy index");
        // The three the #380 rollback did not originally cover.
        assert!(
            idx.tqplus_shift().is_empty() && idx.tqplus_scale().is_empty(),
            "a calibration shaped for the abandoned dim survived the unwind",
        );
        assert_eq!(
            idx.calibration_state(),
            CalibrationState::WarmingUp,
            "the warm-up buffer did not come back, so the index is not as lazy as it was",
        );

        // Serializing must not panic on a dim-0 sentinel beside a
        // length-dim pair — the `new_lazy_uncalibrated` docs promise an
        // index saved before its first add carries no calibration.
        let bytes = idx.to_bytes();
        assert!(!bytes.is_empty());

        // And the retry at a different dim gets the fresh start.
        idx.add_2d(&rows(10, 128, 2), 128)
            .expect("an uncalibrated lazy index should still accept a new dim");
        assert_eq!(idx.dim_opt(), Some(128));
        assert_eq!(idx.len(), 10);
        assert_eq!(
            idx.calibration_state(),
            CalibrationState::Identity,
            "the retry should have re-committed identity at the new dim",
        );
        assert_eq!(idx.tqplus_scale(), vec![1.0; 128]);
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
        // Materialize the blocked cache so the eager path takes the patch
        // branch at all.
        idx.prepare();
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

/// The lazy first add commits per calibration block, so its unwind
/// guard has to undo more than the dim.
///
/// `add` splits a batch at block boundaries and each chunk commits
/// durably before the next runs: `encode_and_append` publishes the
/// codes, scales and row count, `open_rows` grows, and a full block
/// pushes a `SealedBlock` carrying a pair shaped for the dim being
/// committed. A guard that restores only the dim, the caches and the
/// calibration triple leaves `dim_opt() == None` beside `len() > 0` —
/// the permanent wedge #380 exists to prevent, since `to_bytes` then
/// meets a dim-0 sentinel with rows behind it and a retry at another
/// dim addresses them at the wrong stride.
///
/// Not reachable through the public API: the injectors below are
/// `cfg(test)`. It is the invariant that is being pinned, not a live
/// break.
#[cfg(test)]
mod lazy_first_add_rollback_tests {
    use crate::{CalibrationState, TurboQuantIndex, DEFAULT_BLOCK_SIZE};

    const DIM: usize = 64;

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n * dim)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    /// Every piece of state a committed chunk leaves behind.
    fn assert_pristine_lazy(idx: &TurboQuantIndex, what: &str) {
        assert_eq!(idx.dim_opt(), None, "{what}: dim stayed committed");
        assert_eq!(idx.len(), 0, "{what}: rows survived the unwind");
        assert_eq!(idx.slot_capacity(), 0, "{what}: slots survived the unwind");
        assert_eq!(idx.sealed_blocks(), 0, "{what}: a sealed block survived");
        assert!(idx.packed_codes().is_empty(), "{what}: codes survived");
        assert!(idx.scales().is_empty(), "{what}: scales survived");
        assert!(
            idx.tqplus_shift().is_empty() && idx.tqplus_scale().is_empty(),
            "{what}: a calibration shaped for the abandoned dim survived",
        );
        assert_eq!(
            idx.calibration_state(),
            CalibrationState::WarmingUp,
            "{what}: the index did not go back to warming up",
        );
    }

    /// A retry at a *different* dim has to behave like a first add,
    /// which is the whole point of leaving the index lazy.
    fn assert_usable_at_another_dim(mut idx: TurboQuantIndex) {
        const OTHER: usize = 128;
        idx.add_2d(&rows(1200, OTHER, 99), OTHER)
            .expect("a retry at another dim must be a fresh start");
        assert_eq!(idx.dim_opt(), Some(OTHER));
        assert_eq!(idx.len(), 1200);
        let round_tripped = TurboQuantIndex::from_bytes(&idx.to_bytes()).unwrap();
        assert_eq!(round_tripped.len(), 1200, "the retried index does not serialize");
    }

    #[test]
    fn a_panic_on_a_later_chunk_leaves_the_index_lazy() {
        // Two chunks: the first fills a block and commits, the second
        // panics. Before the fix this left 8192 rows and a sealed block
        // behind a `None` dim.
        let n = DEFAULT_BLOCK_SIZE + 100;
        let mut idx = TurboQuantIndex::new_lazy(4).unwrap();
        TurboQuantIndex::force_encode_panic_at(2);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(n, DIM, 5), DIM)
        }));
        TurboQuantIndex::force_encode_panic_at(0);
        assert!(failed.is_err(), "the forced panic should have propagated");

        assert_pristine_lazy(&idx, "chunk 2 panic");
        assert_usable_at_another_dim(idx);
    }

    #[test]
    fn a_panic_sealing_the_first_block_leaves_the_index_lazy() {
        // Exactly one block: the chunk commits, then the seal refits
        // from the block's own rows and panics. This is the default
        // path — `new_lazy` has a block size — so it needs no unusual
        // construction to reach.
        let mut idx = TurboQuantIndex::new_lazy(4).unwrap();
        TurboQuantIndex::force_fit_panic(true);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(DEFAULT_BLOCK_SIZE, DIM, 6), DIM)
        }));
        assert!(failed.is_err(), "the forced panic should have propagated");

        assert_pristine_lazy(&idx, "seal panic");
        assert_usable_at_another_dim(idx);
    }
}

/// The eager `add` path must be atomic across the whole batch, not just
/// across each calibration block.
#[cfg(test)]
mod eager_multi_chunk_rollback_tests {
    use crate::{IdMapIndex, TurboQuantIndex, DEFAULT_BLOCK_SIZE};

    const DIM: usize = 64;

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n * dim)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn a_panic_on_a_later_chunk_leaves_an_eager_index_untouched() {
        let n = DEFAULT_BLOCK_SIZE + 100;
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        TurboQuantIndex::force_encode_panic_at(2);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_2d(&rows(n, DIM, 5), DIM)
        }));
        TurboQuantIndex::force_encode_panic_at(0);
        assert!(failed.is_err(), "the forced panic should have propagated");

        assert_eq!(idx.len(), 0, "chunk 1's rows survived the unwind");
        assert_eq!(idx.slot_capacity(), 0, "chunk 1's slots survived");
        assert_eq!(idx.sealed_blocks(), 0, "a block sealed inside a failed add");

        // And the index is still usable: a retry must land the rows once.
        idx.add_2d(&rows(n, DIM, 5), DIM).unwrap();
        assert_eq!(idx.len(), n);
    }

    #[test]
    fn an_id_map_survives_a_panic_partway_through_a_batch() {
        // The damaging half: the unwind skips `slot_to_id`'s extend, so
        // if the inner index keeps chunk 1 the two tables desync — and
        // the *next* add reports Ok while appending at the wrong offset.
        let n = DEFAULT_BLOCK_SIZE + 100;
        let mut im = IdMapIndex::new(DIM, 4).unwrap();
        let ids: Vec<u64> = (0..n as u64).collect();
        TurboQuantIndex::force_encode_panic_at(2);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            im.add_with_ids(&rows(n, DIM, 9), &ids)
        }));
        TurboQuantIndex::force_encode_panic_at(0);
        assert!(failed.is_err(), "the forced panic should have propagated");
        assert_eq!(im.len(), 0, "the inner index kept rows the id table has no ids for");

        // The retry must succeed and leave a searchable, consistent index.
        im.add_with_ids(&rows(n, DIM, 9), &ids).unwrap();
        assert_eq!(im.len(), n);
        let (_, got) = im.search(&rows(1, DIM, 9)[..DIM], 5);
        assert_eq!(got.len(), 5, "search after the retry did not return k results");
    }
}

/// `design.md`'s determinism criterion: the same vectors and the same
/// block size must produce byte-identical output however the adds were
/// batched, and whatever the thread count.
///
/// This is the justification for blocks being a fixed size rather than
/// the caller's batch shape (#206, #259). Batch-shaped blocks would make
/// the encoded bytes depend on how a caller chunked their `add` calls,
/// which is exactly what this pins against.
///
/// In-crate rather than in `tests/` because varying the thread count
/// needs `rayon`, which is a dependency of the library and not of the
/// integration-test target.
#[cfg(test)]
mod determinism_tests {
    use crate::{TurboQuantIndex, DEFAULT_BLOCK_SIZE, MIN_BLOCK_SIZE};

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        let mut out = vec![0.0f32; n * dim];
        for (i, row) in out.chunks_mut(dim).enumerate() {
            // A drifting mean, so the per-block fits genuinely differ
            // and a block boundary in the wrong place would show.
            let t = i as f32 / n as f32;
            for (d, x) in row.iter_mut().enumerate() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
                *x = v * 0.4 + if d % 2 == 0 { t } else { 1.0 - t };
            }
            let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in row.iter_mut() {
                *x /= nrm + 1e-12;
            }
        }
        out
    }

    /// Mirrors `new_uncalibrated`, which has no block-size form.
    fn index(dim: usize, bits: usize, bs: usize, calibrated: bool) -> TurboQuantIndex {
        let mut ix = TurboQuantIndex::with_block_size(dim, bits, bs).unwrap();
        if !calibrated {
            ix.calibration_enabled = false;
            ix.commit_identity_calibration(dim);
        }
        ix
    }

    /// Build with `add` called in the given chunk sizes, cycling.
    fn build(dim: usize, bits: usize, bs: usize, calibrated: bool, data: &[f32], chunks: &[usize]) -> Vec<u8> {
        let n = data.len() / dim;
        let mut ix = index(dim, bits, bs, calibrated);
        let mut off = 0usize;
        let mut k = 0usize;
        while off < n {
            let take = chunks[k % chunks.len()].min(n - off);
            ix.add(&data[off * dim..(off + take) * dim]);
            off += take;
            k += 1;
        }
        assert_eq!(ix.len(), n);
        ix.to_bytes()
    }

    #[test]
    fn batching_and_thread_count_do_not_change_a_byte() {
        let dim = 64;
        // (block size, rows, calibrated). Each row count ends mid-block
        // so the open block — not only sealed ones — is serialized.
        //
        // Every case seals on a real fit, whatever its block size: since
        // a1ca929 the floor applies to fitting from a *batch*, not from a
        // block, so `MIN_BLOCK_SIZE` is no longer an uncalibrated path.
        // What the three sizes vary is the number of blocks the same
        // rows are cut into, and how big a sealed block is.
        let cases = [
            (MIN_BLOCK_SIZE, 3 * MIN_BLOCK_SIZE + 37, true),
            (MIN_BLOCK_SIZE, 3 * MIN_BLOCK_SIZE + 37, false),
            (1024usize, 2 * 1024 + 37, true),
            (1024, 2 * 1024 + 37, false),
            // The shipping default, and the only size a user gets
            // without going looking.
            (DEFAULT_BLOCK_SIZE, 2 * DEFAULT_BLOCK_SIZE + 37, true),
            (DEFAULT_BLOCK_SIZE, 2 * DEFAULT_BLOCK_SIZE + 37, false),
        ];
        // Whole batch; many small adds; exactly one block at a time; and
        // uneven sizes that straddle boundaries in both directions.
        let batchings: [&[usize]; 5] = [
            &[usize::MAX],
            &[1],
            &[7],
            &[MIN_BLOCK_SIZE],
            &[1, 1023, 3, 1025, 511, 2],
        ];
        // At the default block size the one-row and seven-row batchings
        // are dropped: `&[1]` alone would be 16421 separate `add` calls
        // per thread count and bit width, and what it exercises —
        // per-add bookkeeping — is already covered at 64 and 1024. What
        // is distinctive about this size is boundary behaviour when a
        // sealed block is large, which the remaining three cover.
        let big_batchings: [&[usize]; 3] = [
            &[usize::MAX],
            &[DEFAULT_BLOCK_SIZE],
            &[1, 1023, 3, 1025, 511, 2],
        ];

        for (bs, n, calibrated) in cases {
            let selected: &[&[usize]] = if bs == DEFAULT_BLOCK_SIZE {
                &big_batchings
            } else {
                &batchings
            };
            for bits in [2usize, 4] {
                let data = rows(n, dim, 0xD37E_2000 + bs as u64);
                let mut reference: Option<Vec<u8>> = None;
                for threads in [1usize, 2, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .expect("test pool");
                    for chunks in selected.iter().copied() {
                        let got = pool.install(|| build(dim, bits, bs, calibrated, &data, chunks));
                        match &reference {
                            None => reference = Some(got),
                            Some(want) => assert!(
                                &got == want,
                                "bytes differ: bs={bs} n={n} bits={bits} \
                                 calibrated={calibrated} threads={threads} chunks={chunks:?} \
                                 ({} vs {} bytes)",
                                got.len(),
                                want.len(),
                            ),
                        }
                    }
                }
            }
        }
    }

    /// The block size is part of the index's identity, so two indexes
    /// over the same rows with different block sizes must *not* agree —
    /// otherwise the test above would pass on an implementation that
    /// ignored blocks entirely.
    #[test]
    fn a_different_block_size_is_a_different_index() {
        let dim = 64;
        let n = 2 * 1024 + 37;
        let data = rows(n, dim, 0xD1FF);
        let a = build(dim, 4, 1024, true, &data, &[usize::MAX]);
        let b = build(dim, 4, 2048, true, &data, &[usize::MAX]);
        assert_ne!(a, b, "block size did not affect the encoded bytes");
    }
}
