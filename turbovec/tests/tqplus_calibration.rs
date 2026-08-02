//! TQ+ calibration lifecycle.
//!
//! There are exactly two states — `Uncalibrated` and `Calibrated` — and
//! exactly one transition: an explicit [`TurboQuantIndex::calibrate`]
//! call. Adds never fit, removes never unfit, and both round-trip
//! whatever state they found. The predecessor of this file tested a
//! warm-up/threshold lifecycle (#353, #360, #361, #366, #418) that no
//! longer exists; what survives of it here is the invariant those
//! issues were ultimately about — no sequence of ordinary operations
//! may silently change what the index's calibration declares.

use turbovec::{CalibrationState, IdMapIndex, TurboQuantIndex};

fn gaussian_normalized(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5
    };
    let mut v: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    for row in v.chunks_mut(dim) {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    }
    v
}

/// Adds of any size leave the state alone. 1000 rows was the old
/// implicit-fit threshold; crossing it must no longer do anything.
#[test]
fn adds_never_change_the_calibration_state() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::Uncalibrated);

    idx.add(&gaussian_normalized(10, dim, 1));
    assert_eq!(idx.calibration_state(), CalibrationState::Uncalibrated);

    // Far past the old threshold, in one bulk add.
    idx.add(&gaussian_normalized(2000, dim, 2));
    assert_eq!(idx.calibration_state(), CalibrationState::Uncalibrated);
    assert!(
        idx.tqplus_shift().is_empty() && idx.tqplus_scale().is_empty(),
        "an uncalibrated index holds no fitted state at all"
    );
}

/// The one transition, and its persistence across every ordinary
/// operation: add, remove, drain to empty, serialize, reload.
#[test]
fn a_committed_calibration_survives_the_whole_lifecycle() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.calibrate(&gaussian_normalized(1024, dim, 3)).unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::Calibrated);
    let shift = idx.tqplus_shift().to_vec();
    let scale = idx.tqplus_scale().to_vec();

    idx.add(&gaussian_normalized(300, dim, 4));
    assert_eq!(idx.tqplus_shift(), &shift[..], "an add refit the pair");

    while idx.len() > 0 {
        idx.swap_remove(idx.len() - 1);
    }
    assert_eq!(
        idx.calibration_state(),
        CalibrationState::Calibrated,
        "draining to empty dropped the calibration"
    );

    let back = TurboQuantIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(back.len(), 0);
    assert_eq!(back.calibration_state(), CalibrationState::Calibrated);
    assert_eq!(back.tqplus_shift(), &shift[..]);
    assert_eq!(back.tqplus_scale(), &scale[..]);
}

/// An uncalibrated index round-trips as uncalibrated — populated or
/// empty — and keeps working after the reload.
#[test]
fn an_uncalibrated_index_round_trips_as_uncalibrated() {
    let dim = 64;
    let rows = gaussian_normalized(1500, dim, 5);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&rows);

    let mut back = TurboQuantIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(back.calibration_state(), CalibrationState::Uncalibrated);
    assert_eq!(back.len(), 1500);

    // Still fully functional: searchable, and appendable without any
    // state change.
    let probe = &rows[7 * dim..8 * dim];
    assert_eq!(back.search(probe, 1).indices[0], 7);
    back.add(&gaussian_normalized(100, dim, 6));
    assert_eq!(back.len(), 1600);
    assert_eq!(back.calibration_state(), CalibrationState::Uncalibrated);
}

/// `from_parts` with empty TQ+ arrays (the v2 wire shape) builds an
/// uncalibrated index, and an explicitly-identity pair collapses to the
/// same state: "declares nothing" has exactly one representation, so
/// `calibration_state` has exactly one test.
#[test]
fn from_parts_identity_shapes_collapse_to_uncalibrated() {
    let bit_width = 4usize;
    let dim = 128usize;
    let n_vectors = 3usize;
    let packed = vec![0u8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.0f32; n_vectors];

    let empty = TurboQuantIndex::from_parts(
        Some(dim),
        bit_width,
        n_vectors,
        packed.clone(),
        scales.clone(),
        Vec::new(),
        Vec::new(),
    )
    .expect("v2-shaped parts must construct");
    assert_eq!(empty.calibration_state(), CalibrationState::Uncalibrated);
    assert!(empty.tqplus_shift().is_empty());

    let identity = TurboQuantIndex::from_parts(
        Some(dim),
        bit_width,
        n_vectors,
        packed,
        scales,
        vec![0.0f32; dim],
        vec![1.0f32; dim],
    )
    .expect("an explicit identity pair must construct");
    assert_eq!(identity.calibration_state(), CalibrationState::Uncalibrated);
    assert!(
        identity.tqplus_shift().is_empty(),
        "an exact-identity pair must collapse to the empty representation"
    );
}

/// The id-map wrapper reports and round-trips the same two states.
#[test]
fn id_map_reports_and_round_trips_both_states() {
    let dim = 64;
    let rows = gaussian_normalized(200, dim, 7);
    let ids: Vec<u64> = (0..200u64).collect();

    let mut plain = IdMapIndex::new(dim, 4).unwrap();
    plain.add_with_ids(&rows, &ids).unwrap();
    assert_eq!(plain.calibration_state(), CalibrationState::Uncalibrated);
    let back = IdMapIndex::from_bytes(&plain.to_bytes()).unwrap();
    assert_eq!(back.calibration_state(), CalibrationState::Uncalibrated);

    let mut fitted = IdMapIndex::new(dim, 4).unwrap();
    fitted
        .calibrate(&gaussian_normalized(1024, dim, 8))
        .unwrap();
    fitted.add_with_ids(&rows, &ids).unwrap();
    assert_eq!(fitted.calibration_state(), CalibrationState::Calibrated);
    let back = IdMapIndex::from_bytes(&fitted.to_bytes()).unwrap();
    assert_eq!(back.calibration_state(), CalibrationState::Calibrated);
}
