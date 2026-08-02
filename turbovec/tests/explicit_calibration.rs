//! Explicit TQ+ calibration: `calibrate_2d` / `calibrate`.
//!
//! The implicit fit takes whatever rows the index happens to see first.
//! That is unbiased when the corpus arrives in one bulk add, or in
//! random order — and catastrophic when it arrives sorted or clustered,
//! because the leading rows are then a tiny, tight slice of the
//! population and the fitted quantiles are far too narrow. Every
//! subsequent row clips against the outer codebook centroids.
//!
//! These tests pin the API that lets a caller supply the fit sample
//! themselves, and the two properties that make it safe: it refuses an
//! index that already holds rows (those rows could not be re-encoded —
//! the float32 originals are gone), and it subsamples a large caller
//! slab at random rather than by prefix, so the one bias the API cannot
//! detect is destroyed rather than preserved.

use turbovec::{CalibrateError, CalibrationState, IdMapIndex, TurboQuantIndex};

// ---------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------

/// Cheap deterministic uniform stream. Tests must not depend on the
/// crate's own RNG, and `rand` is not a dev-dependency of the
/// integration tests.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn l2_normalize(rows: &mut [f32], dim: usize) {
    for row in rows.chunks_mut(dim) {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    }
}

/// A clustered corpus, returned **grouped by cluster** — i.e. exactly
/// the row order a corpus sorted by label, by source file, or by any
/// upstream clustering arrives in. Rows are L2-normalised, so inner
/// product is cosine and the index's ranking is directly comparable to
/// exact float32 top-k.
///
/// Each cluster sits around a sharpened centre — a handful of
/// coordinates carry most of its energy — with enough per-row noise that
/// retrieval within a cluster is still a real problem (a corpus of near
/// duplicates would make top-10 recall meaningless whatever the
/// calibration). What makes a leading slice unrepresentative is the
/// centre offset: over a single cluster, each rotated coordinate is
/// concentrated around that cluster's own rotated centre, so quantiles
/// fitted from it are both shifted and far too narrow, and every later
/// cluster clips against the outer codebook centroids.
fn clustered_corpus(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let mut centres = vec![0.0f32; clusters * dim];
    for c in centres.iter_mut() {
        *c = rng.normal();
    }
    // Sharpen each centre: a handful of coordinates carry most of its
    // energy, so different clusters occupy genuinely different
    // subspaces rather than all looking like isotropic noise.
    for centre in centres.chunks_mut(dim) {
        for (d, v) in centre.iter_mut().enumerate() {
            if d % 7 != 0 {
                *v *= 0.05;
            } else {
                *v *= 6.0;
            }
        }
    }
    let mut data = vec![0.0f32; n * dim];
    let per = n.div_ceil(clusters);
    for (i, row) in data.chunks_mut(dim).enumerate() {
        let c = (i / per).min(clusters - 1);
        let centre = &centres[c * dim..(c + 1) * dim];
        for (d, x) in row.iter_mut().enumerate() {
            *x = centre[d] + rng.normal();
        }
    }
    l2_normalize(&mut data, dim);
    data
}

/// Uniform random draw of `k` rows, without replacement — what the
/// `calibrate_2d` docs tell the caller to hand over.
fn random_sample(corpus: &[f32], dim: usize, k: usize, seed: u64) -> Vec<f32> {
    let n = corpus.len() / dim;
    let mut rng = Rng(seed | 1);
    let mut order: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.below(n - i);
        order.swap(i, j);
    }
    let mut out = Vec::with_capacity(k * dim);
    for &row in &order[..k] {
        out.extend_from_slice(&corpus[row * dim..(row + 1) * dim]);
    }
    out
}

/// Exact float32 top-`k` by inner product, for the recall baseline.
fn exact_topk(corpus: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = corpus
        .chunks(dim)
        .enumerate()
        .map(|(i, row)| {
            let s: f32 = row.iter().zip(query).map(|(a, b)| a * b).sum();
            (s, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    scored[..k].iter().map(|&(_, i)| i).collect()
}

/// R@k of the index against the exact float32 top-k. Rows are
/// L2-normalised, so the index's inner-product ranking and the exact
/// cosine ranking are the same objective — anything else fakes a recall
/// collapse that has nothing to do with the calibration.
fn recall_at_k(
    index: &TurboQuantIndex,
    corpus: &[f32],
    dim: usize,
    queries: &[f32],
    k: usize,
) -> f64 {
    use rayon::prelude::*;
    let nq = queries.len() / dim;
    let results = index.search(queries, k);
    let hit: usize = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let truth = exact_topk(corpus, dim, &queries[qi * dim..(qi + 1) * dim], k);
            let got = results.indices_for_query(qi);
            truth
                .into_iter()
                .filter(|&t| got.contains(&(t as i64)))
                .count()
        })
        .sum();
    hit as f64 / (nq * k) as f64
}

/// Add the corpus in fixed-size chunks — a streaming ingest. This is
/// what makes the implicit fit see only the leading rows: the chunk that
/// crosses the warm-up threshold is the whole fit sample.
fn add_in_chunks(index: &mut TurboQuantIndex, corpus: &[f32], dim: usize, chunk_rows: usize) {
    for chunk in corpus.chunks(chunk_rows * dim) {
        index.add_2d(chunk, dim).unwrap();
    }
}

// ---------------------------------------------------------------------
// The point of the feature
// ---------------------------------------------------------------------

/// An unbiased sample recovers the recall that a sorted ingest loses.
///
/// Both indexes see the identical corpus in the identical (clustered)
/// order and differ only in where the calibration came from: the first
/// 1000 rows for the implicit fit, a uniform random draw for the
/// explicit one. Without `calibrate_2d` there is no way to express the
/// second, which is what makes this test fail without the feature.
#[test]
fn unbiased_sample_recovers_recall_on_a_clustered_ingest() {
    let dim = 128;
    let n = 20_000;
    let corpus = clustered_corpus(n, dim, 20, 0xA11C_E501);
    let queries = random_sample(&corpus, dim, 100, 0x5EED_0002);

    let mut implicit = TurboQuantIndex::new(dim, 2).unwrap();
    add_in_chunks(&mut implicit, &corpus, dim, 1000);
    assert_eq!(implicit.calibration_state(), CalibrationState::Fitted);
    let implicit_recall = recall_at_k(&implicit, &corpus, dim, &queries, 10);

    let mut explicit = TurboQuantIndex::new(dim, 2).unwrap();
    let sample = random_sample(&corpus, dim, 1024, 0x5EED_0001);
    explicit.calibrate_2d(&sample, dim).unwrap();
    assert_eq!(explicit.calibration_state(), CalibrationState::Fitted);
    add_in_chunks(&mut explicit, &corpus, dim, 1000);
    let explicit_recall = recall_at_k(&explicit, &corpus, dim, &queries, 10);

    // The ceiling: one bulk add of the whole corpus, whose implicit fit
    // sample *is* the whole corpus and so is unbiased by construction.
    let mut bulk = TurboQuantIndex::new(dim, 2).unwrap();
    bulk.add_2d(&corpus, dim).unwrap();
    let bulk_recall = recall_at_k(&bulk, &corpus, dim, &queries, 10);

    assert!(
        explicit_recall > implicit_recall + 0.15,
        "an unbiased 1024-row sample should recover the recall a \
         clustered ingest loses to its own leading rows: explicit \
         {explicit_recall:.4} vs implicit {implicit_recall:.4}"
    );
    assert!(
        explicit_recall > bulk_recall - 0.05,
        "a 1024-row random draw should fit about as well as the whole \
         corpus does: explicit {explicit_recall:.4} vs whole-corpus \
         {bulk_recall:.4}"
    );
}

/// The subsample is a random draw, not a prefix: handing over the whole
/// clustered corpus in its biased order still fits a usable
/// calibration, because `calibrate_2d` chooses the rows itself.
///
/// This is the property that makes the API safe against the one mistake
/// it cannot detect. A `[..CALIBRATION_FIT_ROWS]` slice here would
/// reproduce the implicit fit's collapse exactly.
#[test]
fn large_ordered_sample_is_subsampled_at_random_not_by_prefix() {
    let dim = 128;
    let n = 20_000;
    let corpus = clustered_corpus(n, dim, 20, 0xA11C_E501);
    let queries = random_sample(&corpus, dim, 100, 0x5EED_0002);

    let mut implicit = TurboQuantIndex::new(dim, 2).unwrap();
    add_in_chunks(&mut implicit, &corpus, dim, 1000);
    let implicit_recall = recall_at_k(&implicit, &corpus, dim, &queries, 10);

    // The caller hands over the corpus exactly as it is stored —
    // clustered, unshuffled, and far larger than the fit needs.
    let mut explicit = TurboQuantIndex::new(dim, 2).unwrap();
    explicit.calibrate_2d(&corpus, dim).unwrap();
    add_in_chunks(&mut explicit, &corpus, dim, 1000);
    let explicit_recall = recall_at_k(&explicit, &corpus, dim, &queries, 10);

    assert!(
        explicit_recall > implicit_recall + 0.15,
        "a large ordered sample must be subsampled at random, not by \
         prefix: explicit {explicit_recall:.4} vs implicit \
         {implicit_recall:.4}"
    );
}

// ---------------------------------------------------------------------
// The ordering contract
// ---------------------------------------------------------------------

#[test]
fn calibrating_a_non_empty_index_is_an_error() {
    let dim = 64;
    let rows = clustered_corpus(2000, dim, 4, 0x0BAD_0001);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add_2d(&rows, dim).unwrap();

    let before_shift = idx.tqplus_shift().to_vec();
    let before_scale = idx.tqplus_scale().to_vec();
    let before_codes = idx.to_bytes();

    let sample = clustered_corpus(1500, dim, 4, 0x0BAD_0002);
    let err = idx.calibrate_2d(&sample, dim).unwrap_err();
    assert_eq!(err, CalibrateError::IndexNotEmpty { len: 2000 });

    // Not a partial application and not a silent no-op: nothing moved.
    assert_eq!(idx.tqplus_shift(), before_shift.as_slice());
    assert_eq!(idx.tqplus_scale(), before_scale.as_slice());
    assert_eq!(idx.to_bytes(), before_codes);
    assert!(err.to_string().contains("already stores vectors"));
}

/// Even a *warming-up* index — one holding sub-threshold rows under
/// identity — is non-empty, and its rows are just as unre-encodable.
#[test]
fn calibrating_a_warming_up_index_is_an_error() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add_2d(&clustered_corpus(50, dim, 2, 0x0BAD_0003), dim)
        .unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::WarmingUp);

    let sample = clustered_corpus(1500, dim, 4, 0x0BAD_0004);
    assert_eq!(
        idx.calibrate_2d(&sample, dim).unwrap_err(),
        CalibrateError::IndexNotEmpty { len: 50 }
    );
    assert_eq!(idx.calibration_state(), CalibrationState::WarmingUp);
}

/// An index drained back to empty may be calibrated: `swap_remove`
/// leaves nothing that would need re-encoding.
#[test]
fn a_drained_index_can_be_calibrated() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add_2d(&clustered_corpus(3, dim, 1, 0x0BAD_0005), dim)
        .unwrap();
    for i in (0..3).rev() {
        idx.swap_remove(i);
    }
    assert_eq!(idx.len(), 0);

    idx.calibrate_2d(&clustered_corpus(1500, dim, 4, 0x0BAD_0006), dim)
        .unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
}

// ---------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------

#[test]
fn sample_below_the_floor_is_rejected_rather_than_committing_identity() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let rows = turbovec::MIN_CALIBRATION_ROWS - 1;
    let sample = clustered_corpus(rows, dim, 4, 0x0BAD_0007);
    assert_eq!(
        idx.calibrate_2d(&sample, dim).unwrap_err(),
        CalibrateError::SampleTooSmall {
            rows,
            min: turbovec::MIN_CALIBRATION_ROWS
        }
    );
    // Still able to fit implicitly: the refused call committed nothing.
    assert_eq!(idx.calibration_state(), CalibrationState::WarmingUp);
    assert!(idx.tqplus_shift().is_empty());
}

#[test]
fn exactly_the_floor_is_accepted() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let sample = clustered_corpus(turbovec::MIN_CALIBRATION_ROWS, dim, 4, 0x0BAD_0008);
    idx.calibrate_2d(&sample, dim).unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
}

#[test]
fn dim_and_shape_errors() {
    let dim = 64;
    let sample = clustered_corpus(1200, dim, 4, 0x0BAD_0009);

    let mut eager = TurboQuantIndex::new(dim, 4).unwrap();
    assert_eq!(
        eager.calibrate_2d(&sample, 128).unwrap_err(),
        CalibrateError::DimMismatch {
            existing: 64,
            got: 128
        }
    );

    let mut lazy = TurboQuantIndex::new_lazy(4).unwrap();
    assert_eq!(
        lazy.calibrate_2d(&sample, 0).unwrap_err(),
        CalibrateError::ZeroDim
    );
    assert_eq!(
        lazy.calibrate_2d(&sample, 63).unwrap_err(),
        CalibrateError::DimNotMultipleOf8(63)
    );
    assert_eq!(
        lazy.calibrate_2d(&sample, turbovec::MAX_DIM + 8)
            .unwrap_err(),
        CalibrateError::DimTooLarge {
            dim: turbovec::MAX_DIM + 8,
            max: turbovec::MAX_DIM
        }
    );
    // A ragged buffer is a typed error here, not a panic.
    assert_eq!(
        lazy.calibrate_2d(&sample[..sample.len() - 1], dim)
            .unwrap_err(),
        CalibrateError::SampleBufferNotMultipleOfDim {
            sample_len: sample.len() - 1,
            dim
        }
    );
    // None of the rejections committed the lazy dim.
    assert_eq!(lazy.dim_opt(), None);
}

#[test]
fn non_finite_sample_is_rejected() {
    let dim = 64;
    let mut sample = clustered_corpus(1200, dim, 4, 0x0BAD_000A);
    sample[5 * dim + 3] = f32::NAN;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    match idx.calibrate_2d(&sample, dim).unwrap_err() {
        CalibrateError::InvalidInputValue {
            vector_index,
            coord_index,
            value,
        } => {
            assert_eq!((vector_index, coord_index), (5, 3));
            assert!(value.is_nan());
        }
        other => panic!("expected InvalidInputValue, got {other:?}"),
    }
    assert_eq!(idx.calibration_state(), CalibrationState::WarmingUp);
}

#[test]
fn calibrating_a_lazy_index_locks_the_dim() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new_lazy(4).unwrap();
    assert_eq!(idx.dim_opt(), None);
    idx.calibrate_2d(&clustered_corpus(1200, dim, 4, 0x0BAD_000B), dim)
        .unwrap();
    assert_eq!(idx.dim_opt(), Some(dim));
    assert_eq!(
        idx.add_2d(&clustered_corpus(4, 128, 1, 0x0BAD_000C), 128)
            .unwrap_err(),
        turbovec::AddError::DimMismatch {
            existing: 64,
            got: 128
        }
    );
}

#[test]
fn calibrate_without_dim_arg_uses_the_committed_dim() {
    let dim = 64;
    let sample = clustered_corpus(1200, dim, 4, 0x0BAD_000D);
    let mut a = TurboQuantIndex::new(dim, 4).unwrap();
    a.calibrate(&sample).unwrap();
    let mut b = TurboQuantIndex::new(dim, 4).unwrap();
    b.calibrate_2d(&sample, dim).unwrap();
    assert_eq!(a.tqplus_shift(), b.tqplus_shift());
    assert_eq!(a.tqplus_scale(), b.tqplus_scale());
}

// ---------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------

/// The internal subsample is a seeded draw, so the same oversized sample
/// selects the same rows every time — on every thread count. Encoded
/// bytes are a hard determinism invariant of this crate, and the
/// calibration is baked into every one of them.
#[test]
fn subsampled_fit_is_deterministic_across_runs_and_thread_counts() {
    let dim = 64;
    // Comfortably above CALIBRATION_FIT_ROWS so the draw actually runs.
    let sample = clustered_corpus(12_000, dim, 16, 0x0BAD_000E);
    let rows = clustered_corpus(1200, dim, 4, 0x0BAD_000F);

    let build = || {
        let mut idx = TurboQuantIndex::new(dim, 2).unwrap();
        idx.calibrate_2d(&sample, dim).unwrap();
        idx.add_2d(&rows, dim).unwrap();
        idx.to_bytes()
    };

    let baseline = build();
    assert_eq!(build(), baseline, "same input, same bytes across runs");

    for threads in [1usize, 2, 3, 7] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let bytes = pool.install(build);
        assert_eq!(
            bytes, baseline,
            "calibration subsample and fit must not depend on the rayon \
             worker count (threads={threads})"
        );
    }

    // And it is a real subsample: fitting from the first
    // CALIBRATION_FIT_ROWS rows of the same slab is a different
    // calibration, so the draw is not silently a prefix.
    let mut prefix_idx = TurboQuantIndex::new(dim, 2).unwrap();
    prefix_idx
        .calibrate_2d(&sample[..turbovec::CALIBRATION_FIT_ROWS * dim], dim)
        .unwrap();
    let mut drawn_idx = TurboQuantIndex::new(dim, 2).unwrap();
    drawn_idx.calibrate_2d(&sample, dim).unwrap();
    assert_ne!(
        prefix_idx.tqplus_shift(),
        drawn_idx.tqplus_shift(),
        "the subsample must not be the leading CALIBRATION_FIT_ROWS rows"
    );
}

// ---------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------

#[test]
fn explicit_calibration_round_trips_through_bytes() {
    let dim = 64;
    let sample = clustered_corpus(1500, dim, 4, 0x0BAD_0010);
    let rows = clustered_corpus(300, dim, 4, 0x0BAD_0011);
    let queries = clustered_corpus(16, dim, 4, 0x0BAD_0012);

    let mut idx = TurboQuantIndex::new(dim, 2).unwrap();
    idx.calibrate_2d(&sample, dim).unwrap();
    idx.add_2d(&rows, dim).unwrap();

    let restored = TurboQuantIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(restored.calibration_state(), CalibrationState::Fitted);
    assert_eq!(restored.tqplus_shift(), idx.tqplus_shift());
    assert_eq!(restored.tqplus_scale(), idx.tqplus_scale());
    assert_eq!(
        restored.search(&queries, 5).indices,
        idx.search(&queries, 5).indices
    );
}

/// A calibrated but still *empty* index round-trips as calibrated. The
/// exact-identity normalization that sends an empty index back to
/// warm-up (#418) must not swallow a real fit — otherwise "calibrate,
/// save, load, add" silently loses TQ+.
#[test]
fn calibrated_empty_index_round_trips_as_fitted() {
    let dim = 64;
    let sample = clustered_corpus(1500, dim, 4, 0x0BAD_0013);
    let mut idx = TurboQuantIndex::new(dim, 2).unwrap();
    idx.calibrate_2d(&sample, dim).unwrap();
    assert_eq!(idx.len(), 0);

    let restored = TurboQuantIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(restored.calibration_state(), CalibrationState::Fitted);
    assert_eq!(restored.tqplus_shift(), idx.tqplus_shift());
    assert_eq!(restored.tqplus_scale(), idx.tqplus_scale());

    // And adding after the reload reuses that calibration rather than
    // fitting a fresh one.
    let rows = clustered_corpus(2000, dim, 4, 0x0BAD_0014);
    let mut a = restored;
    a.add_2d(&rows, dim).unwrap();
    let mut b = TurboQuantIndex::new(dim, 2).unwrap();
    b.calibrate_2d(&sample, dim).unwrap();
    b.add_2d(&rows, dim).unwrap();
    assert_eq!(a.to_bytes(), b.to_bytes());
}

#[test]
fn id_map_calibration_round_trips_through_tvim() {
    let dim = 64;
    let sample = clustered_corpus(1500, dim, 4, 0x0BAD_0015);
    let rows = clustered_corpus(400, dim, 4, 0x0BAD_0016);
    let ids: Vec<u64> = (0..400u64).map(|i| i * 7 + 3).collect();

    let mut idx = IdMapIndex::new(dim, 2).unwrap();
    idx.calibrate_2d(&sample, dim).unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
    idx.add_with_ids_2d(&rows, dim, &ids).unwrap();

    let restored = IdMapIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(restored.calibration_state(), CalibrationState::Fitted);
    assert_eq!(restored.len(), 400);

    let queries = clustered_corpus(8, dim, 4, 0x0BAD_0017);
    assert_eq!(restored.search(&queries, 5).1, idx.search(&queries, 5).1);

    // File path too.
    let dir = std::env::temp_dir().join(format!("turbovec-explicit-calib-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("calibrated.tvim");
    idx.write(&path).unwrap();
    let loaded = IdMapIndex::load(&path).unwrap();
    assert_eq!(loaded.calibration_state(), CalibrationState::Fitted);
    assert_eq!(loaded.search(&queries, 5).1, idx.search(&queries, 5).1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn id_map_calibrate_rejects_a_non_empty_index() {
    let dim = 64;
    let rows = clustered_corpus(20, dim, 2, 0x0BAD_0018);
    let ids: Vec<u64> = (0..20u64).collect();
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids_2d(&rows, dim, &ids).unwrap();
    assert_eq!(
        idx.calibrate_2d(&clustered_corpus(1200, dim, 4, 0x0BAD_0019), dim)
            .unwrap_err(),
        CalibrateError::IndexNotEmpty { len: 20 }
    );
}
