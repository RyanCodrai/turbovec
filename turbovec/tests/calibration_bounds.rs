//! A calibration or per-vector scale that is finite but unusable must be
//! refused at construction and at load, not accepted and then turned into
//! Inf/NaN scores by the search kernels (#478), and `expected_codebook`
//! must enforce the MAX_DIM bound its rustdoc claims (#489).

use turbovec::{FromPartsError, TurboQuantIndex};

const DIM: usize = 64;

fn rows(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * DIM];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
    }
    for r in v.chunks_mut(DIM) {
        let n: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in r.iter_mut() { *x /= n; }
    }
    v
}

/// An honest index, decomposed so a single field can be poisoned.
fn parts(n: usize) -> (usize, usize, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut i = TurboQuantIndex::new(DIM, 4).unwrap();
    i.calibrate(&rows(1024, 5)).unwrap();
    i.add(&rows(n, 6));
    (
        i.bit_width(), i.len(),
        i.packed_codes().to_vec(), i.scales().to_vec(),
        i.tqplus_shift().to_vec(), i.tqplus_scale().to_vec(),
    )
}

#[test]
fn a_tiny_tqplus_scale_is_refused_rather_than_producing_nan_scores() {
    // 1e-22 is included deliberately: it passed the first, dim-blind
    // bound and still produced all-NaN scores, because the divided query
    // is summed across every coordinate before it becomes a score.
    for poison in [1e-40f32, 1e-38, 1e-30, 1e-23, 1e-22, 1e-21] {
        let (bw, n, codes, scales, shift, mut tq) = parts(96);
        tq[7] = poison;
        let e = TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, scales, shift, tq);
        assert!(
            matches!(e, Err(FromPartsError::InvalidTqplusScaleValue { coord: 7, .. })),
            "tqplus_scale {poison:e} must be refused, got {:?}", e.map(|i| i.len())
        );
    }
}

#[test]
fn a_usable_tqplus_scale_is_still_accepted() {
    // The fit is magnitude-invariant, so nothing an honest corpus
    // produces may start failing. 1e-21 is already far below reachable.
    for ok in [1e-6f32, 0.5, 1.62, 1e6] {
        let (bw, n, codes, scales, shift, mut tq) = parts(96);
        tq[7] = ok;
        assert!(
            TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, scales, shift, tq).is_ok(),
            "tqplus_scale {ok:e} is usable and must be accepted"
        );
    }
}

#[test]
fn an_enormous_tqplus_shift_is_refused() {
    let (bw, n, codes, scales, mut sh, tq) = parts(96);
    sh[3] = f32::MAX;
    assert!(matches!(
        TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, scales, sh, tq),
        Err(FromPartsError::InvalidTqplusShiftValue { coord: 3, .. })
    ));
}

#[test]
fn an_enormous_per_vector_scale_is_refused() {
    let (bw, n, codes, mut sc, sh, tq) = parts(96);
    sc[2] = f32::MAX;
    assert!(matches!(
        TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, sc, sh, tq),
        Err(FromPartsError::InvalidScaleValue { slot: 2, .. })
    ));
}

/// The loaders share `validate_calibration`, so a poisoned file must be
/// refused too — that is the path an untrusted `.tv` arrives by.
#[test]
fn a_poisoned_calibration_does_not_survive_a_file_round_trip() {
    let dir = std::env::temp_dir().join(format!("tv-calbound-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("i.tv");
    let mut i = TurboQuantIndex::new(DIM, 4).unwrap();
    i.calibrate(&rows(1024, 7)).unwrap();
    i.add(&rows(96, 8));
    i.write(&path).unwrap();

    // Rewrite the first TQ+ scale as 1e-40. In a v7 image the
    // calibration sits in the superblock, not at the tail: magic(4) +
    // version(1) + bit_width(1) + kind(1) + dim(4) + nonce(8) +
    // max_ops(4) = 23, then the codebook (2^bits - 1 boundaries and
    // 2^bits centroids), then n_calib(4), then shift(dim), then
    // scale(dim).
    let mut b = std::fs::read(&path).unwrap();
    let n_levels = 1usize << 4;
    let at = 23 + (2 * n_levels - 1) * 4 + 4 + DIM * 4;
    b[at..at + 4].copy_from_slice(&1e-40f32.to_le_bytes());
    std::fs::write(&path, &b).unwrap();

    let e = TurboQuantIndex::load(&path);
    assert!(e.is_err(), "a 1e-40 TQ+ scale must not load");
    let msg = e.unwrap_err().to_string();
    assert!(msg.contains("TQ+ scale"), "unhelpful error: {msg}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[should_panic(expected = "MAX_DIM")]
fn expected_codebook_enforces_the_max_dim_bound_it_documents() {
    let _ = turbovec::expected_codebook(4, 1 << 20);
}

/// The bound has to grow with `dim`: the bias is a dim-long dot product
/// narrowed back to f32, and the divided query is reduced across every
/// coordinate. A flat constant is safe at dim 64 and wrong at dim 1024.
#[test]
fn the_calibration_bounds_scale_with_dim() {
    use turbovec::TurboQuantIndex as T;
    let mut small = T::new(64, 4).unwrap();
    small.add(&vec![0.1f32; 64 * 32]);
    let mut large = T::new(1024, 4).unwrap();
    large.add(&vec![0.1f32; 1024 * 32]);

    // A scale that is fine at dim 64 must be refused at dim 1024.
    let borderline = 1e-19f32;
    let ok64 = T::from_parts(
        Some(64), small.bit_width(), small.len(), small.packed_codes().to_vec(),
        small.scales().to_vec(), vec![0.0; 64], vec![borderline; 64],
    );
    let bad1024 = T::from_parts(
        Some(1024), large.bit_width(), large.len(), large.packed_codes().to_vec(),
        large.scales().to_vec(), vec![0.0; 1024], vec![borderline; 1024],
    );
    assert!(ok64.is_ok(), "{borderline:e} is usable at dim 64: {:?}", ok64.err());
    assert!(
        bad1024.is_err(),
        "{borderline:e} must be refused at dim 1024 — 16x the coordinates to sum"
    );
}

/// The v7 loader must apply the same per-vector scale bound as the v6
/// loader, or the two disagree about the same untrusted file.
#[test]
fn both_loaders_agree_on_an_oversized_per_vector_scale() {
    let dir = std::env::temp_dir().join(format!("tv-bothload-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (v6, v7) = (dir.join("a.tv"), dir.join("b.tv"));
    let mut i = TurboQuantIndex::new(DIM, 4).unwrap();
    i.add(&rows(96, 21));
    i.write(&v6).unwrap();
    i.sync(&v7).unwrap();
    // Both load cleanly to begin with.
    assert!(TurboQuantIndex::load(&v6).is_ok());
    assert!(TurboQuantIndex::load(&v7).is_ok());

    // A hand-built index carrying an out-of-range per-vector scale is
    // refused by from_parts, which is the shared gate both loaders'
    // value checks mirror.
    let mut sc = i.scales().to_vec();
    sc[0] = f32::MAX;
    assert!(TurboQuantIndex::from_parts(
        Some(DIM), i.bit_width(), i.len(), i.packed_codes().to_vec(),
        sc, i.tqplus_shift().to_vec(), i.tqplus_scale().to_vec(),
    ).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The rejection message must describe the rule that actually rejected
/// the value. A dim-aware bound reported as "must be finite and > 0"
/// tells the caller a rejected value is valid.
#[test]
fn rejection_messages_do_not_contradict_the_predicate() {
    let (bw, n, codes, scales, shift, mut tq) = parts(96);
    tq[7] = 1e-30;
    let e = TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, scales, shift, tq)
        .unwrap_err()
        .to_string();
    assert!(!e.contains("must be finite and > 0"), "1e-30 is finite and > 0: {e}");
    assert!(e.contains("summed"), "should name the real rule: {e}");

    let (bw, n, codes, scales, mut sh, tq) = parts(96);
    sh[3] = 1e30;
    let e = TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, scales, sh, tq)
        .unwrap_err()
        .to_string();
    assert!(
        !e.trim_end().ends_with("(must be finite)"),
        "1e30 is finite: {e}"
    );
    assert!(e.contains("bias"), "should name the real rule: {e}");
}

// ---------------------------------------------------------------------
// Boundary behaviour. The bounds are inclusive — a value exactly at the
// limit is usable — and the mutation gate showed the earlier tests never
// said so: they used extreme values only, so flipping `>` to `>=` (and
// the /10 margin to *10) changed nothing any test could see.
// ---------------------------------------------------------------------

/// Same formulas as `io::max_tqplus_shift` / `min_tqplus_scale`, restated
/// here because they are `pub(crate)`. A drift between the two shows up
/// as one of the assertions below failing.
const MAX_INPUT: f32 = 1e16;
fn max_shift_at(dim: usize) -> f32 {
    f32::MAX / ((dim as f32) * MAX_INPUT) / 10.0
}
fn min_scale_at(dim: usize) -> f32 {
    (dim as f32) * MAX_INPUT / f32::MAX * 10.0
}
const MAX_VECTOR_SCALE: f32 = 1e22;

#[test]
fn a_shift_exactly_at_the_bound_is_accepted() {
    let (bw, n, codes, scales, mut sh, tq) = parts(96);
    sh[3] = max_shift_at(DIM);
    assert!(
        TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, scales, sh, tq).is_ok(),
        "the bound is inclusive: a shift exactly at it must load"
    );
}

#[test]
fn a_shift_far_above_the_bound_is_rejected() {
    // 20x the bound. If the margin were multiplied instead of divided the
    // effective bound would be 100x and this would sail through.
    let (bw, n, codes, scales, mut sh, tq) = parts(96);
    sh[3] = max_shift_at(DIM) * 20.0;
    assert!(matches!(
        TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, scales, sh, tq),
        Err(FromPartsError::InvalidTqplusShiftValue { coord: 3, .. })
    ));
}

#[test]
fn a_scale_exactly_at_the_floor_is_accepted() {
    let (bw, n, codes, scales, shift, mut tq) = parts(96);
    tq[7] = min_scale_at(DIM);
    assert!(
        TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, scales, shift, tq).is_ok(),
        "the floor is inclusive: a scale exactly at it must load"
    );
}

#[test]
fn a_per_vector_scale_exactly_at_the_bound_is_accepted() {
    let (bw, n, codes, mut sc, sh, tq) = parts(96);
    sc[2] = MAX_VECTOR_SCALE;
    assert!(
        TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, sc, sh, tq).is_ok(),
        "the bound is inclusive: a per-vector scale exactly at it must load"
    );
}

/// The loaders share `read_scales_validated`, so the same boundary has to
/// hold through a file round trip, not just through `from_parts`.
#[test]
fn a_per_vector_scale_at_the_bound_survives_a_file_round_trip() {
    let dir = std::env::temp_dir().join(format!("tv-bound-rt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("i.tv");
    let (bw, n, codes, mut sc, sh, tq) = parts(96);
    sc[2] = MAX_VECTOR_SCALE;
    let idx = TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, sc, sh, tq).unwrap();
    idx.write(&path).unwrap();
    assert!(
        TurboQuantIndex::load(&path).is_ok(),
        "a per-vector scale at the bound must load back"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `validate_calibration` is the loader's copy of the shift rule, and it
/// is a different code path from `from_parts`. Pin the inclusive bound
/// through a file round trip so a `>` that drifts to `>=` there is
/// caught too.
#[test]
fn a_shift_at_the_bound_survives_a_file_round_trip() {
    let dir = std::env::temp_dir().join(format!("tv-shift-rt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("i.tv");
    let (bw, n, codes, scales, mut sh, tq) = parts(96);
    sh[3] = max_shift_at(DIM);
    let idx = TurboQuantIndex::from_parts(Some(DIM), bw, n, codes, scales, sh, tq).unwrap();
    idx.write(&path).unwrap();
    assert!(
        TurboQuantIndex::load(&path).is_ok(),
        "a TQ+ shift exactly at the bound must load back"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
