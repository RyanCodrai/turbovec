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

/// The case the review caught: an uncalibrated index that is *empty* at
/// save time used to lose its opt-out.
///
/// On disk, "opted out of calibration" and "warming up, then drained to
/// zero rows" are the same thing — no rows, full-length identity pair —
/// and `normalize_calibration`'s `declares_nothing` arm (#418) discards
/// the pair so a drained warming-up index can still fit later. That is
/// right for warming-up and wrong for uncalibrated, and the pair alone
/// cannot tell them apart. v7's trailer carries the flag that can.
#[test]
fn an_empty_uncalibrated_index_keeps_its_opt_out_across_a_reload() {
    let dim = 64;
    // Saved before any add.
    let ix = TurboQuantIndex::new_uncalibrated(dim, 4).unwrap();
    let mut back = TurboQuantIndex::from_bytes(&ix.to_bytes()).unwrap();
    assert!(!back.calibration_enabled(), "the opt-out did not survive the reload");
    back.add(&rows(N, dim, 21));
    assert_eq!(back.calibration_state(), CalibrationState::Identity);
    assert_eq!(back.tqplus_scale(), vec![1.0; dim]);
}

/// Same, via the fill-then-drain path — the deletion-heavy workload this
/// feature is for.
#[test]
fn a_drained_uncalibrated_index_keeps_its_opt_out_across_a_reload() {
    let dim = 64;
    let mut ix = TurboQuantIndex::new_uncalibrated(dim, 4).unwrap();
    ix.add(&rows(N, dim, 22));
    while !ix.is_empty() {
        ix.swap_remove(ix.len() - 1);
    }
    assert_eq!(ix.calibration_state(), CalibrationState::Identity);

    let mut back = TurboQuantIndex::from_bytes(&ix.to_bytes()).unwrap();
    assert!(!back.calibration_enabled(), "draining silently re-enabled calibration");
    back.add(&rows(N, dim, 23));
    assert_eq!(
        back.calibration_state(),
        CalibrationState::Identity,
        "the refilled index fitted a calibration it was told not to",
    );
    assert_eq!(back.tqplus_scale(), vec![1.0; dim]);
}

/// The #418 behaviour this must not break: a *calibrated* index drained
/// to zero still reloads as `WarmingUp`, free to fit later.
#[test]
fn a_drained_calibrated_index_still_reloads_as_warming_up() {
    let dim = 64;
    let mut ix = TurboQuantIndex::new(dim, 4).unwrap();
    ix.add(&rows(10, dim, 24)); // sub-threshold: commits explicit identity
    while !ix.is_empty() {
        ix.swap_remove(ix.len() - 1);
    }
    let mut back = TurboQuantIndex::from_bytes(&ix.to_bytes()).unwrap();
    assert!(back.calibration_enabled());
    assert_eq!(back.calibration_state(), CalibrationState::WarmingUp);
    back.add(&rows(N, dim, 25));
    assert_eq!(
        back.calibration_state(),
        CalibrationState::Fitted,
        "#418: a drained warming-up index must still be able to fit",
    );
}
