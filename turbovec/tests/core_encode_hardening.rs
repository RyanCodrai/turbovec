//! Regression tests for the core-encode hardening fixes (#116, #117, #129).
//!
//! Bugs pinned here:
//!
//! 1. **#116 — frozen-calibration scale explosion.** The per-vector
//!    correction scale is `||v|| / <u_rot, x_hat>`. With a genuinely
//!    fitted (frozen) TQ+ calibration, a later-added out-of-distribution
//!    vector can reconstruct with a *negative* inner product; the old
//!    `inner.max(1e-10)` clamp turned that into a denominator of `1e-10`,
//!    producing a ~1e10 scale with a flipped sign — one garbage vector
//!    then falsely dominated every top-k. Degenerate reconstructions
//!    (`inner <= 1e-10`) must instead store scale 0 so the vector scores
//!    ~0 and ranks last.
//!
//! 2. **#117 — public `encode::encode` OOB write for dim % 8 != 0.**
//!    `bytes_per_plane = dim / 8` truncates; the tail-coordinate branch
//!    then wrote past the end of each row: the top bit-plane panicked
//!    with an index-out-of-bounds and lower planes silently corrupted
//!    the next plane's bytes. `encode` must reject non-multiple-of-8
//!    dims up front with a clear message, matching the index-level
//!    validation.
//!
//! 3. **#129 — `add_2d` on a lazy index committed dim before the
//!    length-validation panic.** `vectors.len() % dim` was only checked
//!    inside the inner `add()`, *after* `self.dim = Some(dim)`, so the
//!    panic left the lazy index wedged with a committed dim and zero
//!    vectors — a follow-up `add_2d` with a different dim saw a
//!    confusing `DimMismatch` instead of a fresh start. The length
//!    check must run before the dim commit.

use std::panic::{catch_unwind, AssertUnwindSafe};

use turbovec::codebook::codebook;
use turbovec::encode::encode;
use turbovec::rotation::make_rotation_matrix;
use turbovec::{AddError, IdMapIndex, TurboQuantIndex};

/// Deterministic small noise in [-1, 1] (xorshift).
fn noise(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let raw = (*state >> 40) as u32;
    raw as f32 / (1u32 << 23) as f32 - 1.0
}

// ─── #116: frozen-calibration scale explosion ───────────────────────────────

#[test]
fn ood_vector_under_frozen_calibration_does_not_dominate_topk() {
    let dim = 64;
    let n = 1200; // >= TQPLUS_MIN_SAMPLES so a real calibration is fitted
    let mut index = TurboQuantIndex::new(dim, 4).unwrap();

    // Cluster tightly around +e1 so the fitted calibration is frozen to a
    // distribution the later outlier does not belong to.
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut vectors = vec![0.0f32; n * dim];
    for row in vectors.chunks_mut(dim) {
        row[0] = 1.0;
        for coord in row.iter_mut().skip(1) {
            *coord = 0.01 * noise(&mut state);
        }
    }
    index.add(&vectors);

    // Out-of-distribution vector: -e1, the exact opposite of the cluster.
    let mut ood = vec![0.0f32; dim];
    ood[0] = -1.0;
    index.add(&ood);
    let ood_slot = n as i64;

    // Query +e1. True scores: cluster ≈ +1.0, OOD vector = -1.0 (worst
    // possible). The OOD vector must not appear in the top-k at all, and
    // no score may be inflated beyond the plausible inner-product range.
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
    // Same setup as above, but check the encode-level contract directly:
    // a vector whose quantized reconstruction points away from it must
    // get scale 0 (unrepresentable ⇒ scores ~0), never a huge negative-
    // inner-product blowup.
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

    // The zero vector keeps its historical behavior: norm 0 ⇒ scale 0.
    let zero = vec![0.0f32; dim];
    let (_, zero_scales, _, _) = encode(
        &zero, 1, dim, &rotation, &boundaries, &centroids, 4,
        Some((&shift, &scale_tq)),
    );
    assert_eq!(zero_scales[0], 0.0, "zero vector must keep scale 0");

    // NaN input through the *direct* encode path (the index-level API
    // rejects non-finite values before encoding) must land in the
    // degenerate branch and store scale 0, not a NaN/garbage scale. The
    // helper's comparison is written positively (`inner > EPS`) so NaN
    // fails it.
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
    // Adversarial-review follow-up on #116: zeroing only `inner <= 1e-10`
    // moved the explosion cliff instead of removing it — a unit vector at
    // theta ≈ 1.6293 rad from the cluster axis reconstructed with a tiny
    // *positive* inner (1.5e-4 .. 1.8e-3), storing scales of 559 .. 6.6e3
    // and again dominating the top-k with a hugely inflated score. The
    // degeneracy test is now `inner <= 0.1` (inner is unit-space /
    // cosine-like), so every stored scale for a unit vector is bounded by
    // 1 / 0.1 = 10 and the near-orthogonal window collapses to scale 0.
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

    // Encode-level sweep: 720 unit vectors cos(t)*e1 + sin(t)*e2 across
    // (0, pi), quantized under the calibration frozen from the cluster.
    // No stored scale may exceed the 1/EPS = 10 bound (or be non-finite).
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

    // Index-level: the two pathological angles found by the reviewer's
    // coarse sweep (theta = 1.6275, stored scale 559 pre-fix) and
    // bisection (theta = 1.629281051794, stored scale 6.6e3 pre-fix)
    // must never reach the top-k. Their true inner products with the +e1
    // query are ≈ -0.057 / -0.059 — worse than every cluster member.
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

// ─── #117: encode rejects dim not a multiple of 8 ───────────────────────────

#[test]
#[should_panic(expected = "multiple of 8")]
fn encode_rejects_dim_not_multiple_of_8() {
    // dim = 12 truncates bytes_per_plane to 1; before the fix the four
    // tail coordinates wrote out of bounds (top bit-plane panicked with
    // an index-out-of-bounds, lower planes silently corrupted the next
    // plane's bytes). encode must reject this up front with a clear
    // message instead.
    let dim = 12;
    let n = 4;
    let rotation = make_rotation_matrix(dim);
    let (boundaries, centroids) = codebook(2, dim);
    let vectors = vec![0.25f32; n * dim];
    let _ = encode(
        &vectors, n, dim, &rotation, &boundaries, &centroids, 2, None,
    );
}

// ─── #129: failed length validation leaves a lazy index uncommitted ─────────

#[test]
fn lazy_add_2d_length_panic_does_not_commit_dim() {
    let mut index = TurboQuantIndex::new_lazy(4).unwrap();

    // 100 is not a multiple of 64 → add_2d panics (documented contract).
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = index.add_2d(&vec![0.5f32; 100], 64);
    }));
    assert!(result.is_err(), "add_2d must panic on non-multiple length");

    // The failed add must NOT have committed the dim: the index stays
    // lazy/uncommitted and a fresh start with a different dim succeeds.
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
    // IdMapIndex::add_with_ids_2d validates length/dim up front and
    // returns a typed error without touching the inner index — pin that
    // the lazy index stays uncommitted there too.
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
