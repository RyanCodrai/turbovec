//! TQ+ calibration is optional, and off means off for good (#455).
//!
//! The switch is expressed entirely through the committed `(shift,
//! scale)` pair — an explicit identity — so it needs no format change
//! and round-trips as `Identity`. These tests pin that, and pin the
//! thing an *empty* pair would do instead: make the next encode fit a
//! fresh calibration from its own batch, which is the opposite of what
//! an uncalibrated index asked for.

use turbovec::{CalibrationState, TurboQuantIndex};

const N: usize = 2 * 1000; // comfortably over the fit threshold

fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..n * dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f64 / u64::MAX as f64) as f32 - 0.5
        })
        .collect()
}

#[test]
fn an_uncalibrated_index_never_leaves_identity() {
    let dim = 64;
    let mut ix = TurboQuantIndex::new_uncalibrated(dim, 4).unwrap();
    assert!(!ix.calibration_enabled());
    // Identity from construction, before any row exists.
    assert_eq!(ix.calibration_state(), CalibrationState::Identity);
    assert_eq!(ix.tqplus_shift(), vec![0.0; dim]);
    assert_eq!(ix.tqplus_scale(), vec![1.0; dim]);

    // Well past the threshold that would normally trigger a fit.
    ix.add(&rows(N, dim, 1));
    assert_eq!(ix.calibration_state(), CalibrationState::Identity);
    assert_eq!(ix.tqplus_shift(), vec![0.0; dim]);
    assert_eq!(ix.tqplus_scale(), vec![1.0; dim]);

    // And a second add does not fit one either.
    ix.add(&rows(N, dim, 2));
    assert_eq!(ix.calibration_state(), CalibrationState::Identity);
    assert_eq!(ix.tqplus_scale(), vec![1.0; dim]);
}

#[test]
fn the_default_still_fits() {
    let dim = 64;
    let mut ix = TurboQuantIndex::new(dim, 4).unwrap();
    assert!(ix.calibration_enabled());
    ix.add(&rows(N, dim, 1));
    assert_eq!(ix.calibration_state(), CalibrationState::Fitted);
    assert_ne!(ix.tqplus_scale(), vec![1.0; dim]);
}

#[test]
fn uncalibrated_survives_a_round_trip() {
    let dim = 64;
    let mut ix = TurboQuantIndex::new_uncalibrated(dim, 4).unwrap();
    ix.add(&rows(N, dim, 7));
    let back = TurboQuantIndex::from_bytes(&ix.to_bytes()).unwrap();
    assert_eq!(back.calibration_state(), CalibrationState::Identity);
    assert_eq!(back.tqplus_scale(), vec![1.0; dim]);
    // Adding after the reload must not start fitting either: the
    // committed identity is what pins it, not the in-memory flag.
    let mut back = back;
    back.add(&rows(N, dim, 8));
    assert_eq!(back.calibration_state(), CalibrationState::Identity);
    assert_eq!(back.tqplus_scale(), vec![1.0; dim]);
}

#[test]
fn a_lazy_uncalibrated_index_commits_identity_on_the_first_add() {
    let dim = 64;
    let mut ix = TurboQuantIndex::new_lazy_uncalibrated(4).unwrap();
    assert!(!ix.calibration_enabled());
    // No dim yet, so nothing to commit; the pair is dim-shaped.
    assert!(ix.tqplus_shift().is_empty());
    ix.add_2d(&rows(N, dim, 3), dim).unwrap();
    assert_eq!(ix.calibration_state(), CalibrationState::Identity);
    assert_eq!(ix.tqplus_scale(), vec![1.0; dim]);
}

#[test]
fn uncalibrated_and_warming_up_are_different_states() {
    let dim = 64;
    // Under the threshold, a calibrated index is WarmingUp: it is
    // holding the rows to re-encode them later.
    let mut warming = TurboQuantIndex::new(dim, 4).unwrap();
    warming.add(&rows(10, dim, 4));
    assert_eq!(warming.calibration_state(), CalibrationState::WarmingUp);

    // An uncalibrated one is already done deciding.
    let mut off = TurboQuantIndex::new_uncalibrated(dim, 4).unwrap();
    off.add(&rows(10, dim, 4));
    assert_eq!(off.calibration_state(), CalibrationState::Identity);
}

#[test]
fn search_still_works_uncalibrated() {
    let dim = 64;
    let db = rows(N, dim, 11);
    let mut ix = TurboQuantIndex::new_uncalibrated(dim, 2).unwrap();
    ix.add(&db);
    // Query with a stored row: it must come back first.
    let q = &db[5 * dim..6 * dim];
    let r = ix.search(q, 3);
    assert_eq!(r.nq, 1);
    assert_eq!(r.indices_for_query(0)[0], 5);
}
