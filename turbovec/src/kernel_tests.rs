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
    use crate::encode::encode;
    use crate::rotation::make_rotation_matrix;

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
        let rotation = make_rotation_matrix(dim);
        let (boundaries, centroids) = codebook(3, dim);
        let vectors = make_vectors(n, dim, 0);

        let (packed, scales, _, _) = encode(
            &vectors, n, dim, &rotation, &boundaries, &centroids, 3, None,
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
            let rotation = make_rotation_matrix(dim);
            let (boundaries, centroids) = codebook(bit_width, dim);
            let vectors = make_vectors(n, dim, 0);

            let (packed, scales, _, _) = encode(
                &vectors, n, dim, &rotation, &boundaries, &centroids, bit_width, None,
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
        let rotation = make_rotation_matrix(dim);
        let (boundaries, centroids) = codebook(4, dim);
        let vectors = make_vectors(n, dim, 0);

        let (_, scales, _, _) =
            encode(&vectors, n, dim, &rotation, &boundaries, &centroids, 4, None);

        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv_norm = 1.0 / norm;

            let mut u_rot = vec![0.0f32; dim];
            for k in 0..dim {
                let mut acc = 0.0f32;
                for j in 0..dim {
                    acc += rotation[k * dim + j] * row[j] * inv_norm;
                }
                u_rot[k] = acc;
            }

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
        let rotation = make_rotation_matrix(dim);
        let (boundaries, centroids) = codebook(4, dim);
        let vectors = make_vectors(n, dim, 0);

        let (p1, s1, _, _) =
            encode(&vectors, n, dim, &rotation, &boundaries, &centroids, 4, None);
        let (p2, s2, _, _) =
            encode(&vectors, n, dim, &rotation, &boundaries, &centroids, 4, None);

        assert_eq!(p1, p2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn handles_zero_vector() {
        let dim = 128;
        let rotation = make_rotation_matrix(dim);
        let (boundaries, centroids) = codebook(4, dim);
        let zeros = vec![0.0f32; dim];

        let (packed, scales, _, _) =
            encode(&zeros, 1, dim, &rotation, &boundaries, &centroids, 4, None);

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
    use crate::encode::encode;
    use crate::rotation::make_rotation_matrix;
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

        let rotation = make_rotation_matrix(dim);
        let (boundaries, centroids) = codebook(4, dim);
        let (_, _, shift, scale_tq) = encode(
            &vectors, n, dim, &rotation, &boundaries, &centroids, 4, None,
        );

        let mut ood = vec![0.0f32; dim];
        ood[0] = -1.0;
        let (_, scales, _, _) = encode(
            &ood, 1, dim, &rotation, &boundaries, &centroids, 4,
            Some((&shift, &scale_tq)),
        );
        assert!(
            scales[0].abs() < 10.0,
            "degenerate reconstruction produced exploded scale {}",
            scales[0],
        );

        let zero = vec![0.0f32; dim];
        let (_, zero_scales, _, _) = encode(
            &zero, 1, dim, &rotation, &boundaries, &centroids, 4,
            Some((&shift, &scale_tq)),
        );
        assert_eq!(zero_scales[0], 0.0, "zero vector must keep scale 0");

        let mut nan_vec = vec![0.5f32; dim];
        nan_vec[3] = f32::NAN;
        let (_, nan_scales, _, _) = encode(
            &nan_vec, 1, dim, &rotation, &boundaries, &centroids, 4,
            Some((&shift, &scale_tq)),
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

        let rotation = make_rotation_matrix(dim);
        let (boundaries, centroids) = codebook(4, dim);
        let (_, _, shift, scale_tq) = encode(
            &cluster, n, dim, &rotation, &boundaries, &centroids, 4, None,
        );
        let steps = 720;
        let mut sweep = vec![0.0f32; steps * dim];
        for (t, row) in sweep.chunks_mut(dim).enumerate() {
            let theta = std::f32::consts::PI * (t as f32 + 0.5) / steps as f32;
            row[0] = theta.cos();
            row[1] = theta.sin();
        }
        let (_, sweep_scales, _, _) = encode(
            &sweep, steps, dim, &rotation, &boundaries, &centroids, 4,
            Some((&shift, &scale_tq)),
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
        let rotation = make_rotation_matrix(dim);
        let (boundaries, centroids) = codebook(2, dim);
        let vectors = vec![0.25f32; n * dim];
        let _ = encode(
            &vectors, n, dim, &rotation, &boundaries, &centroids, 2, None,
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
