//! Explicit TQ+ calibration: `calibrate_2d` / `calibrate`.
//!
//! A calibration comes from exactly one place: the sample the caller
//! hands to `calibrate`. The index never fits one on its own — an index
//! that is never calibrated is plain TurboQuant, order-independent by
//! construction.
//!
//! The fit uses every row given, so what the calibration describes is
//! whatever the caller's sample describes. A uniform random draw of
//! ~1024 rows matches a fit on the whole corpus; a sorted or clustered
//! prefix of the same size fits quantiles that are shifted and far too
//! narrow, and every row outside that slice clips against the outer
//! codebook centroids. Choosing a representative sample is the
//! caller's responsibility — these tests pin both sides of that
//! bargain, plus the refit contract: calibrating a populated index
//! re-encodes every stored row into the new coordinate system.

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

/// Add the corpus in fixed-size chunks — a streaming ingest.
fn add_in_chunks(index: &mut TurboQuantIndex, corpus: &[f32], dim: usize, chunk_rows: usize) {
    for chunk in corpus.chunks(chunk_rows * dim) {
        index.add_2d(chunk, dim).unwrap();
    }
}

// ---------------------------------------------------------------------
// The point of the feature
// ---------------------------------------------------------------------

/// The bias contrast that motivates the whole API: the same corpus, the
/// same sample size, and a wide recall gap between a random draw and a
/// sorted prefix. The fit uses every row the caller gives it, so the
/// difference below is entirely the caller's choice of rows — which is
/// the responsibility the docs assign them.
#[test]
fn a_random_sample_calibrates_and_a_sorted_prefix_destroys() {
    let dim = 128;
    let n = 20_000;
    let corpus = clustered_corpus(n, dim, 20, 0xA11C_E501);
    let queries = random_sample(&corpus, dim, 100, 0x5EED_0002);

    // The mistake: the first 1024 rows of a clustered corpus.
    let mut biased = TurboQuantIndex::new(dim, 2).unwrap();
    biased.calibrate_2d(&corpus[..1024 * dim], dim).unwrap();
    add_in_chunks(&mut biased, &corpus, dim, 1000);
    let biased_recall = recall_at_k(&biased, &corpus, dim, &queries, 10);

    // The documented input: a uniform random draw of the same size.
    let mut drawn = TurboQuantIndex::new(dim, 2).unwrap();
    let sample = random_sample(&corpus, dim, 1024, 0x5EED_0001);
    drawn.calibrate_2d(&sample, dim).unwrap();
    assert_eq!(drawn.calibration_state(), CalibrationState::Calibrated);
    add_in_chunks(&mut drawn, &corpus, dim, 1000);
    let drawn_recall = recall_at_k(&drawn, &corpus, dim, &queries, 10);

    // The ceiling: the entire corpus as the sample — any row count is
    // accepted, and the whole population is the one sample that cannot
    // be unrepresentative.
    let mut full = TurboQuantIndex::new(dim, 2).unwrap();
    full.calibrate_2d(&corpus, dim).unwrap();
    add_in_chunks(&mut full, &corpus, dim, 1000);
    let full_recall = recall_at_k(&full, &corpus, dim, &queries, 10);

    assert!(
        drawn_recall > biased_recall + 0.15,
        "a random 1024-row draw must beat a sorted 1024-row prefix by a \
         wide margin: drawn {drawn_recall:.4} vs prefix {biased_recall:.4}"
    );
    assert!(
        drawn_recall > full_recall - 0.05,
        "a 1024-row random draw should fit about as well as the whole \
         corpus does: drawn {drawn_recall:.4} vs whole-corpus \
         {full_recall:.4}"
    );
}

/// An index that is never calibrated is plain TurboQuant: no fitted
/// state, and insertion order cannot matter. The same corpus added in
/// one bulk add, in 1000-row chunks, and row by row produces the same
/// bytes.
#[test]
fn an_uncalibrated_index_is_order_independent() {
    let dim = 64;
    let corpus = clustered_corpus(3000, dim, 6, 0xA11C_E502);

    let mut bulk = TurboQuantIndex::new(dim, 2).unwrap();
    bulk.add_2d(&corpus, dim).unwrap();
    assert_eq!(bulk.calibration_state(), CalibrationState::Uncalibrated);
    assert!(bulk.tqplus_shift().is_empty(), "no fitted state anywhere");

    let mut chunked = TurboQuantIndex::new(dim, 2).unwrap();
    add_in_chunks(&mut chunked, &corpus, dim, 1000);
    let mut trickled = TurboQuantIndex::new(dim, 2).unwrap();
    add_in_chunks(&mut trickled, &corpus, dim, 1);

    assert_eq!(bulk.to_bytes(), chunked.to_bytes());
    assert_eq!(bulk.to_bytes(), trickled.to_bytes());
}

// ---------------------------------------------------------------------
// Refit: calibrating a populated index
// ---------------------------------------------------------------------

/// Refit's hard limit, pinned so the docs can never oversell it: a
/// *badly* biased calibration destroys information at encode time — a
/// too-narrow fit clips most coordinates to the outer centroids — and
/// no later refit can recover what clipping threw away. Measured here:
/// broken 0.041, "rescued" 0.037. The only repair for a bad
/// calibration is a rebuild from the source vectors, and that is
/// exactly what the `calibrate` docs say.
///
/// (Refit between *close* pairs is the benign case: at 2–3 bits the
/// codes reach an exact fixed point — see
/// `refit_under_the_committed_pair_reproduces_the_codes`.)
#[test]
fn refit_cannot_rescue_codes_a_biased_fit_clipped() {
    let dim = 128;
    let n = 20_000;
    let corpus = clustered_corpus(n, dim, 20, 0xA11C_E501);
    let queries = random_sample(&corpus, dim, 100, 0x5EED_0002);
    let sample = random_sample(&corpus, dim, 1024, 0x5EED_0001);

    let mut idx = TurboQuantIndex::new(dim, 2).unwrap();
    idx.calibrate_2d(&corpus[..1024 * dim], dim).unwrap();
    add_in_chunks(&mut idx, &corpus, dim, 1000);
    let broken = recall_at_k(&idx, &corpus, dim, &queries, 10);

    idx.calibrate_2d(&sample, dim).unwrap();
    assert_eq!(idx.len(), n);
    let refit = recall_at_k(&idx, &corpus, dim, &queries, 10);

    // Mechanically sound (rows stay searchable, state is Calibrated) —
    // but no recovery. If this assertion ever *fails upward*, refit has
    // started genuinely rescuing clipped codes and the documentation's
    // "rebuild from source" guidance should be revisited.
    assert!(
        (refit - broken).abs() < 0.10,
        "refit after a biased fit moved recall dramatically ({broken:.4} \
         -> {refit:.4}); the documented no-rescue contract is stale"
    );
    assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
}

/// Refit's honest cost sheet, measured rather than assumed. Calibrating
/// after the rows arrived re-quantizes each stored row from its own
/// coarse reconstruction, and that second quantization loses real
/// recall against calibrating before the adds — several points at these
/// bit widths on this corpus. The bound here is a *ceiling* on that
/// loss; the guidance it enforces is the one the docs give: calibrate
/// before adding when you can, refit to repair, rebuild for the last
/// word in quality.
#[test]
fn refit_cost_against_calibrate_first_is_bounded() {
    let dim = 128;
    let n = 20_000;
    let corpus = clustered_corpus(n, dim, 20, 0xA11C_E501);
    let queries = random_sample(&corpus, dim, 100, 0x5EED_0002);
    let sample = random_sample(&corpus, dim, 1024, 0x5EED_0001);

    for bits in [2usize, 3, 4] {
        let mut reference = TurboQuantIndex::new(dim, bits).unwrap();
        reference.calibrate_2d(&sample, dim).unwrap();
        add_in_chunks(&mut reference, &corpus, dim, 1000);
        let reference_recall = recall_at_k(&reference, &corpus, dim, &queries, 10);

        let mut refit = TurboQuantIndex::new(dim, bits).unwrap();
        add_in_chunks(&mut refit, &corpus, dim, 1000);
        assert_eq!(refit.calibration_state(), CalibrationState::Uncalibrated);
        refit.calibrate_2d(&sample, dim).unwrap();
        assert_eq!(refit.calibration_state(), CalibrationState::Calibrated);
        assert_eq!(refit.len(), n);
        let refit_recall = recall_at_k(&refit, &corpus, dim, &queries, 10);

        // Measured on this corpus: -6.5pp at 2 bits, -8.1pp at 3,
        // -7.2pp at 4 (uncalibrated -> fitted, the maximally divergent
        // refit). The 0.10 ceiling catches a regression that widens the
        // loss, not the loss itself.
        assert!(
            refit_recall > reference_recall - 0.10,
            "bits={bits}: refit fell more than the documented \
             re-quantization loss below calibrate-first: refit \
             {refit_recall:.4} vs reference {reference_recall:.4}"
        );
    }
}

/// Refitting under the committed pair reproduces the stored codes
/// bit-identically: the reconstruction of a code is an exact centroid
/// value, and re-quantizing a centroid lands in its own cell. This is
/// the fixed point the refit rests on, and it pins the code decode to
/// the pack layout — a wrong decode cannot reproduce the bytes.
#[test]
fn refit_under_the_committed_pair_reproduces_the_codes() {
    let dim = 64;
    let sample = clustered_corpus(1024, dim, 4, 0x0BAD_0002);
    for bits in [2usize, 3, 4] {
        let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
        idx.calibrate_2d(&sample, dim).unwrap();
        idx.add_2d(&clustered_corpus(2000, dim, 4, 0x0BAD_0001), dim)
            .unwrap();
        let codes_before = idx.packed_codes().to_vec();
        let scales_before = idx.scales().to_vec();

        idx.calibrate_2d(&sample, dim).unwrap();

        assert_eq!(
            idx.packed_codes(),
            codes_before.as_slice(),
            "refit under the same pair changed the stored codes (bits={bits})"
        );
        for (i, (a, b)) in idx.scales().iter().zip(&scales_before).enumerate() {
            assert!(
                (a - b).abs() <= b.abs() * 1e-5,
                "scale {i} moved more than rounding under a same-pair \
                 refit (bits={bits}): {a} vs {b}"
            );
        }
    }
}

/// A refit to a *different* pair keeps every row searchable: self-recall
/// holds before and after, and the pair actually changed.
#[test]
fn refit_to_a_new_pair_keeps_rows_searchable() {
    let dim = 64;
    let rows = clustered_corpus(2000, dim, 4, 0x0BAD_0001);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.calibrate_2d(&clustered_corpus(1024, dim, 4, 0x0BAD_0002), dim)
        .unwrap();
    idx.add_2d(&rows, dim).unwrap();
    let pair_before = idx.tqplus_shift().to_vec();

    idx.calibrate_2d(&clustered_corpus(1024, dim, 8, 0x0BAD_0003), dim)
        .unwrap();
    assert_ne!(
        idx.tqplus_shift(),
        pair_before.as_slice(),
        "a different sample must fit a different pair"
    );
    assert_eq!(idx.len(), 2000);
    for probe_row in [0usize, 7, 999, 1999] {
        let probe = &rows[probe_row * dim..(probe_row + 1) * dim];
        assert_eq!(
            idx.search(probe, 1).indices[0] as usize,
            probe_row,
            "row {probe_row} no longer finds itself after a refit"
        );
    }
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
    assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
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
    // The refused call committed nothing.
    assert_eq!(idx.calibration_state(), CalibrationState::Uncalibrated);
    assert!(idx.tqplus_shift().is_empty());
}

#[test]
fn exactly_the_floor_is_accepted() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let sample = clustered_corpus(turbovec::MIN_CALIBRATION_ROWS, dim, 4, 0x0BAD_0008);
    idx.calibrate_2d(&sample, dim).unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
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
    assert_eq!(idx.calibration_state(), CalibrationState::Uncalibrated);
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

/// The fit is a pure function of the sample: same rows, same pair, same
/// encoded bytes — on every thread count. Encoded bytes are a hard
/// determinism invariant of this crate, and the calibration is baked
/// into every one of them.
#[test]
fn the_fit_is_deterministic_across_runs_and_thread_counts() {
    let dim = 64;
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
    assert_eq!(restored.calibration_state(), CalibrationState::Calibrated);
    assert_eq!(restored.tqplus_shift(), idx.tqplus_shift());
    assert_eq!(restored.tqplus_scale(), idx.tqplus_scale());
    assert_eq!(
        restored.search(&queries, 5).indices,
        idx.search(&queries, 5).indices
    );
}

/// A calibrated but still *empty* index round-trips as calibrated —
/// otherwise "calibrate, save, load, add" silently loses TQ+.
#[test]
fn calibrated_empty_index_round_trips_as_calibrated() {
    let dim = 64;
    let sample = clustered_corpus(1500, dim, 4, 0x0BAD_0013);
    let mut idx = TurboQuantIndex::new(dim, 2).unwrap();
    idx.calibrate_2d(&sample, dim).unwrap();
    assert_eq!(idx.len(), 0);

    let restored = TurboQuantIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(restored.calibration_state(), CalibrationState::Calibrated);
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
    assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
    idx.add_with_ids_2d(&rows, dim, &ids).unwrap();

    let restored = IdMapIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(restored.calibration_state(), CalibrationState::Calibrated);
    assert_eq!(restored.len(), 400);

    let queries = clustered_corpus(8, dim, 4, 0x0BAD_0017);
    assert_eq!(restored.search(&queries, 5).1, idx.search(&queries, 5).1);

    // File path too.
    let dir = std::env::temp_dir().join(format!("turbovec-explicit-calib-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("calibrated.tvim");
    idx.write(&path).unwrap();
    let loaded = IdMapIndex::load(&path).unwrap();
    assert_eq!(loaded.calibration_state(), CalibrationState::Calibrated);
    assert_eq!(loaded.search(&queries, 5).1, idx.search(&queries, 5).1);
    std::fs::remove_dir_all(&dir).ok();
}

/// Refit through the id map: the mapping survives, and every id still
/// resolves to its own row.
#[test]
fn id_map_refit_keeps_ids_resolving() {
    let dim = 64;
    let rows = clustered_corpus(500, dim, 4, 0x0BAD_0018);
    let ids: Vec<u64> = (0..500u64).map(|i| i * 7 + 3).collect();
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids_2d(&rows, dim, &ids).unwrap();

    idx.calibrate_2d(&clustered_corpus(1200, dim, 4, 0x0BAD_0019), dim)
        .unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
    assert_eq!(idx.len(), 500);
    for probe_row in [0usize, 123, 499] {
        let probe = &rows[probe_row * dim..(probe_row + 1) * dim];
        let (_, found) = idx.search(probe, 1);
        assert_eq!(
            found[0], ids[probe_row],
            "id for row {probe_row} no longer resolves after refit"
        );
    }
}

/// Refit on a *file-loaded* index. After a load the packed rows are
/// unset and the blocked cache is authoritative; the refit has to
/// materialize them first (`packed()`), re-encode, and invalidate the
/// stale cache. This is the state-interaction shape that has bitten
/// before, so it gets its own end-to-end pin — through both formats.
#[test]
fn refit_works_on_a_loaded_index() {
    let dim = 64;
    let rows = clustered_corpus(1500, dim, 4, 0x0BAD_0020);
    let dir = std::env::temp_dir().join(format!("turbovec-refit-load-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // .tv path.
    let mut built = TurboQuantIndex::new(dim, 4).unwrap();
    built
        .calibrate_2d(&clustered_corpus(1024, dim, 4, 0x0BAD_0021), dim)
        .unwrap();
    built.add_2d(&rows, dim).unwrap();
    let path = dir.join("refit.tv");
    built.write(&path).unwrap();

    let mut loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.calibration_state(), CalibrationState::Calibrated);
    loaded
        .calibrate_2d(&clustered_corpus(1024, dim, 8, 0x0BAD_0022), dim)
        .unwrap();
    assert_eq!(loaded.len(), 1500);
    for probe_row in [0usize, 7, 1499] {
        let probe = &rows[probe_row * dim..(probe_row + 1) * dim];
        assert_eq!(
            loaded.search(probe, 1).indices[0] as usize,
            probe_row,
            "row {probe_row} lost after refit on a .tv-loaded index"
        );
    }
    // And the refitted state itself round-trips.
    let again = TurboQuantIndex::from_bytes(&loaded.to_bytes()).unwrap();
    assert_eq!(again.tqplus_shift(), loaded.tqplus_shift());

    // .tvim / from_bytes path, ids resolving afterwards.
    let ids: Vec<u64> = (0..1500u64).map(|i| i * 3 + 1).collect();
    let mut im = IdMapIndex::new(dim, 4).unwrap();
    im.calibrate_2d(&clustered_corpus(1024, dim, 4, 0x0BAD_0021), dim)
        .unwrap();
    im.add_with_ids_2d(&rows, dim, &ids).unwrap();
    let mut im_loaded = IdMapIndex::from_bytes(&im.to_bytes()).unwrap();
    im_loaded
        .calibrate_2d(&clustered_corpus(1024, dim, 8, 0x0BAD_0022), dim)
        .unwrap();
    for probe_row in [0usize, 750, 1499] {
        let probe = &rows[probe_row * dim..(probe_row + 1) * dim];
        let (_, found) = im_loaded.search(probe, 1);
        assert_eq!(
            found[0], ids[probe_row],
            "id for row {probe_row} lost after refit on a loaded IdMapIndex"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}
