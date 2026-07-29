//! Wave-6 regression tests for the TQ+ calibration state machine.
//!
//! Two bugs surfaced by the audit:
//!
//! 1. **Empty first add silently froze identity calibration.** `add(&[])`
//!    hit the `n < TQPLUS_MIN_SAMPLES` branch in `encode`, returned
//!    `(zeros, ones)`, and the `n_vectors == 0` branch in `add` copied
//!    that identity into `self.tqplus_shift` / `self.tqplus_scale`.
//!    Every subsequent add — even a million-vector batch with rich
//!    distribution — then saw `existing = Some(identity)` and skipped
//!    fresh fitting, silently losing the TQ+ recall gain.
//!
//! 2. **v2-loaded index + add silently mis-encoded.** A v2 file (pre-TQ+)
//!    loads with empty `tqplus_shift`; on the next add, `existing` is
//!    `None`, so `encode` fits fresh calibration and bakes it into the
//!    packed codes. But the else branch (`n_vectors != 0`) only extends
//!    `packed_codes` / `scales`, never persisting the fitted shift /
//!    scale_tq. The new vectors end up encoded with calibration but
//!    searched with identity — silent score corruption.


use turbovec::{io, CalibrationState, IdMapIndex, TurboQuantIndex};

fn gaussian_normalized(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut uniform = || {
        let raw = (next() >> 40) as u32 | 1;
        raw as f32 / (1u32 << 24) as f32
    };
    let two_pi = 2.0_f32 * std::f32::consts::PI;
    let mut data = vec![0.0f32; n * dim];
    let mut i = 0;
    while i < data.len() {
        let u1 = uniform().max(1e-7);
        let u2 = uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = two_pi * u2;
        data[i] = r * theta.cos();
        if i + 1 < data.len() {
            data[i + 1] = r * theta.sin();
        }
        i += 2;
    }
    for row in data.chunks_mut(dim) {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    }
    data
}

#[test]
fn empty_first_add_does_not_freeze_identity_calibration() {
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();

    // Empty add — must be a true no-op, not silently lock identity
    // calibration on the index.
    idx.add(&[]);
    assert_eq!(idx.len(), 0);

    // Add a realistic batch big enough to trigger TQ+ fitting
    // (>= TQPLUS_MIN_SAMPLES, currently 1000).
    let data = gaussian_normalized(1500, dim, 0xC0FF_EE01);
    idx.add(&data);
    assert_eq!(idx.len(), 1500);

    // After the fix, the second add fits fresh calibration. Verify by
    // round-tripping through the format and inspecting the persisted
    // TQ+ trailer — at least one shift or scale value must differ from
    // identity (shift != 0 or scale != 1). Pre-fix, the trailer would
    // be exactly `(zeros, ones)` because identity was locked by the
    // empty add.
    let tmp = std::env::temp_dir().join(format!(
        "turbovec_empty_add_freeze_{}.tv",
        std::process::id()
    ));
    idx.write(&tmp).unwrap();
    let (_, _, _, _, _, shift, scale_tq) = io::load(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(shift.len(), dim);
    assert_eq!(scale_tq.len(), dim);

    let nontrivial_shift = shift.iter().any(|&x| x.abs() > 1e-6);
    let nontrivial_scale = scale_tq.iter().any(|&x| (x - 1.0).abs() > 1e-6);
    assert!(
        nontrivial_shift || nontrivial_scale,
        "TQ+ calibration is exactly identity after empty + 1500-vec add — \
         the empty first add likely locked identity, suppressing fresh \
         calibration on the real batch.",
    );
}

#[test]
fn empty_tqplus_parts_populate_identity_calibration() {
    // The v2-loaded-index concern (empty TQ+ trailer + n_vectors > 0,
    // then a follow-up add silently mis-encoding) now surfaces through
    // `from_parts`: it is the public path that accepts v2-shaped raw
    // parts (empty TQ+ arrays alongside stored vectors). v2 *files* are
    // no longer loadable after the v5 rotation break, but external
    // embedders can still hand `from_parts` this shape, so the
    // identity-population invariant must still hold.
    let bit_width = 4usize;
    let dim = 128usize;
    let n_vectors = 3usize;

    let packed = vec![0u8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.0f32; n_vectors];
    // Empty TQ+ arrays == the v2 wire shape.
    let mut idx = TurboQuantIndex::from_parts(
        Some(dim),
        bit_width,
        n_vectors,
        packed,
        scales,
        Vec::new(),
        Vec::new(),
    )
    .expect("v2-shaped parts must construct");
    assert_eq!(idx.len(), 3);
    assert_eq!(idx.dim_opt().unwrap(), dim);
    // from_parts fills identity so the next add sees `existing =
    // Some(identity)` rather than the lazy-first-add `None`.
    assert_eq!(idx.tqplus_shift(), &vec![0.0f32; dim][..]);
    assert_eq!(idx.tqplus_scale(), &vec![1.0f32; dim][..]);

    // Add a fresh batch big enough to make `encode` fit non-trivial
    // calibration if `existing` were `None` (the pre-fix path). After
    // the fix, `existing = Some(identity)` so encode does NOT fit, the
    // new vectors are encoded with identity, and writing back gives an
    // identity TQ+ trailer — round-trip-stable across the v2->v3 hop.
    let data = gaussian_normalized(1500, dim, 0x42EE_D101);
    idx.add(&data);
    assert_eq!(idx.len(), 1503);

    let tmp = std::env::temp_dir().join(format!(
        "turbovec_v2_load_then_add_out_{}.tv",
        std::process::id()
    ));
    idx.write(&tmp).unwrap();
    let (_, _, _, _, _, shift, scale_tq) = io::load(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(shift.len(), dim);
    assert_eq!(scale_tq.len(), dim);
    for &s in &shift {
        assert_eq!(s, 0.0, "v2-loaded + add must keep identity shift");
    }
    for &s in &scale_tq {
        assert_eq!(s, 1.0, "v2-loaded + add must keep identity scale");
    }
}

#[test]
fn small_first_add_then_large_add_still_fits_calibration() {
    // #285 / #107 / #317: a first add of 1-999 vectors used to lock
    // identity calibration for the index's lifetime. The rows added
    // below the sample threshold are buffered raw, so the add that
    // crosses it re-encodes them alongside the new batch under a
    // properly fitted calibration.
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&gaussian_normalized(500, dim, 0x5EED_0001));
    assert_eq!(idx.len(), 500);
    assert_eq!(idx.calibration_state(), CalibrationState::WarmingUp);

    idx.add(&gaussian_normalized(2000, dim, 0x5EED_0002));
    assert_eq!(idx.len(), 2500, "the buffered rows keep their slots");
    assert_eq!(idx.calibration_state(), CalibrationState::Fitted);

    let nontrivial = idx.tqplus_shift().iter().any(|&x| x.abs() > 1e-6)
        || idx.tqplus_scale().iter().any(|&x| (x - 1.0).abs() > 1e-6);
    assert!(
        nontrivial,
        "calibration is exactly identity after 500 + 2000 vectors — the \
         sub-threshold first add froze it again",
    );
}

#[test]
fn drip_fed_small_batches_reach_a_fitted_calibration() {
    // #317's ingestion pattern: batches of 500, never one add over the
    // threshold. Recall must match the single-bulk-add build.
    let dim = 128;
    let n = 3000;
    let data = gaussian_normalized(n, dim, 0xD21B_1234);
    let queries = gaussian_normalized(50, dim, 0xC0DE_0001);

    let mut bulk = TurboQuantIndex::new(dim, 4).unwrap();
    bulk.add(&data);

    let mut drip = TurboQuantIndex::new(dim, 4).unwrap();
    for chunk in data.chunks(500 * dim) {
        drip.add(chunk);
    }
    assert_eq!(drip.len(), bulk.len());
    assert_eq!(drip.calibration_state(), CalibrationState::Fitted);

    // The calibration is fitted from the first 1000 vectors rather than
    // all 3000, so the codes are not byte-identical to the bulk build —
    // but recall must land in the same place, not the identity-frozen
    // place.
    let recall = |idx: &TurboQuantIndex| -> f64 {
        let res = idx.search(&queries, 10);
        let mut hits = 0usize;
        for q in 0..res.nq {
            let exact = exact_topk(&queries[q * dim..(q + 1) * dim], &data, dim, 10);
            for got in res.indices_for_query(q) {
                if exact.contains(&(*got as usize)) {
                    hits += 1;
                }
            }
        }
        hits as f64 / (res.nq * 10) as f64
    };
    let bulk_recall = recall(&bulk);
    let drip_recall = recall(&drip);
    assert!(
        drip_recall >= bulk_recall - 0.05,
        "drip-fed recall {drip_recall:.3} is far below the bulk build's {bulk_recall:.3}",
    );
}

#[test]
fn drain_to_empty_then_add_keeps_the_fitted_calibration() {
    // #284: swap_remove down to zero left `tqplus_shift` populated, so
    // `encode` took the reuse path and returned empty calibration vecs
    // — which the old `n_vectors == 0` commit branch then wrote over
    // the fitted calibration, declaring identity for codes that were
    // not encoded that way.
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&gaussian_normalized(1500, dim, 0xDEAD_0001));
    assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
    let shift_before = idx.tqplus_shift().to_vec();
    let scale_before = idx.tqplus_scale().to_vec();

    while idx.len() > 0 {
        idx.swap_remove(idx.len() - 1);
    }
    idx.add(&gaussian_normalized(1500, dim, 0xDEAD_0002));

    assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
    assert_eq!(idx.tqplus_shift(), &shift_before[..]);
    assert_eq!(idx.tqplus_scale(), &scale_before[..]);

    // The written trailer must describe how the codes were encoded.
    let tmp = std::env::temp_dir().join(format!(
        "turbovec_drain_recal_{}.tv",
        std::process::id()
    ));
    idx.write(&tmp).unwrap();
    let (_, _, _, _, _, shift, scale_tq) = io::load(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(shift, shift_before, "drain-to-empty wiped the trailer");
    assert_eq!(scale_tq, scale_before);
}

#[test]
fn drain_to_empty_while_warming_up_keeps_the_buffer_aligned() {
    // The warm-up buffer mirrors swap_remove, so survivors keep their
    // slots and can still be re-encoded when the threshold is crossed.
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&gaussian_normalized(300, dim, 0xB0FF_0001));
    for _ in 0..100 {
        idx.swap_remove(0);
    }
    assert_eq!(idx.len(), 200);
    assert_eq!(idx.calibration_state(), CalibrationState::WarmingUp);
    idx.add(&gaussian_normalized(900, dim, 0xB0FF_0002));
    assert_eq!(idx.len(), 1100);
    assert_eq!(idx.calibration_state(), CalibrationState::Fitted);

    // Every slot must still be reachable: self-search returns each
    // vector's own slot most of the time, and never an out-of-range one.
    let res = idx.search(&gaussian_normalized(20, dim, 0xB0FF_0003), 5);
    for &i in &res.indices {
        assert!(i >= 0 && (i as usize) < idx.len(), "slot {i} out of range");
    }
}

#[test]
fn v6_load_with_empty_calibration_then_add_stays_reachable() {
    // #303: the v6 load arms built `Self { .. }` directly and skipped
    // the identity-population `from_parts` performs, so a file with an
    // empty TQ+ trailer (format-sanctioned, emitted by the public `io`
    // writers) plus a later add produced vectors that `len` counted but
    // search could never return.
    let dim = 64;
    let bit_width = 4usize;
    let seed = gaussian_normalized(500, dim, 0x0303_0001);
    let mut src = TurboQuantIndex::new(dim, bit_width).unwrap();
    src.add(&seed);

    // Write with an explicitly empty TQ+ trailer, the shape a
    // third-party v6 writer may legitimately produce.
    let (boundaries, centroids) = src.codebook_for_write();
    let mut bytes = Vec::new();
    io::write_to(
        &mut bytes,
        bit_width,
        dim,
        src.len(),
        &src.codes_blocked_seq(),
        &boundaries,
        &centroids,
        src.scales(),
        &[],
        &[],
    )
    .unwrap();

    let mut idx = TurboQuantIndex::from_bytes(&bytes).unwrap();
    // Stored rows always come with a declared calibration.
    assert_eq!(idx.tqplus_shift(), &vec![0.0f32; dim][..]);
    assert_eq!(idx.tqplus_scale(), &vec![1.0f32; dim][..]);
    assert_eq!(idx.calibration_state(), CalibrationState::Identity);

    let fresh = gaussian_normalized(2000, dim, 0x0303_0002);
    idx.add(&fresh);
    assert_eq!(idx.len(), 2500);

    // Self-recall on the newly added vectors: pre-fix this was 0.
    let probes = 100;
    let res = idx.search(&fresh[..probes * dim], 1);
    let hits = (0..probes)
        .filter(|&q| res.indices_for_query(q)[0] as usize == 500 + q)
        .count();
    assert!(
        hits > probes / 2,
        "only {hits}/{probes} newly added vectors are reachable — the v6 load \
         path dropped the calibration the add fitted",
    );
}

#[test]
fn calibration_state_reports_the_lifecycle() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::WarmingUp);
    idx.add(&gaussian_normalized(10, dim, 0xACCE_0001));
    assert_eq!(idx.calibration_state(), CalibrationState::WarmingUp);
    idx.add(&gaussian_normalized(1500, dim, 0xACCE_0002));
    assert_eq!(idx.calibration_state(), CalibrationState::Fitted);

    // A file carries no warm-up buffer: an index saved mid-warm-up
    // comes back committed to identity, and says so.
    let mut warm = TurboQuantIndex::new(dim, 4).unwrap();
    warm.add(&gaussian_normalized(100, dim, 0xACCE_0003));
    let reloaded = TurboQuantIndex::from_bytes(&warm.to_bytes()).unwrap();
    assert_eq!(reloaded.calibration_state(), CalibrationState::Identity);

    // Round-tripping a fitted index keeps it fitted.
    let round = TurboQuantIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(round.calibration_state(), CalibrationState::Fitted);

    // IdMapIndex exposes the same signal.
    let mut ids = IdMapIndex::new(dim, 4).unwrap();
    assert_eq!(ids.calibration_state(), CalibrationState::WarmingUp);
    let vecs = gaussian_normalized(1200, dim, 0xACCE_0004);
    let id_list: Vec<u64> = (0..1200u64).collect();
    ids.add_with_ids(&vecs, &id_list).unwrap();
    assert_eq!(ids.calibration_state(), CalibrationState::Fitted);
}

/// Brute-force exact top-k inner product, for recall comparisons.
fn exact_topk(query: &[f32], data: &[f32], dim: usize, k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = data
        .chunks(dim)
        .enumerate()
        .map(|(i, row)| {
            let dot: f32 = row.iter().zip(query).map(|(a, b)| a * b).sum();
            (dot, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}


/// #353: the warm-up buffer must never run ahead of `n_vectors`. The
/// documented invariant is "buffer row i is the index's slot i", and
/// `encode_and_append` has an unwind guard that restores the index
/// without incrementing `n_vectors` — so the buffer may only be extended
/// *after* a successful encode. If it is extended first, a caught panic
/// leaves the buffer longer than the index and the failed batch's rows
/// are resurrected into the threshold re-encode.
///
/// The panic path itself is not reachable through the public API (input
/// is validated before encoding), so this pins the observable half: a
/// rejected add must leave the buffer untouched, and every accepted add
/// must leave it exactly in step with the index.
#[test]
fn warmup_buffer_stays_in_step_with_n_vectors() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();

    // Four 100-row batches: 400 total, comfortably under the 1000-row
    // threshold so the index stays in warm-up.
    for batch in 1..=4u64 {
        idx.add_2d(&gaussian_normalized(100, dim, batch), dim).unwrap();
        assert_eq!(idx.calibration_state(), CalibrationState::WarmingUp);
    }
    let committed = idx.len();

    // A rejected batch must not move anything: NaN fails validation
    // before any encoding happens.
    let mut bad = gaussian_normalized(10, dim, 99);
    bad[5] = f32::NAN;
    assert!(idx.add_2d(&bad, dim).is_err());
    assert_eq!(idx.len(), committed, "a rejected add changed the index length");

    // Crossing the threshold re-encodes exactly the buffered rows plus
    // the new batch — if the buffer had drifted, the total would not
    // match and earlier rows would be unreachable.
    idx.add_2d(&gaussian_normalized(600, dim, 7), dim).unwrap();
    assert_eq!(idx.calibration_state(), CalibrationState::Fitted);
    assert_eq!(idx.len(), committed + 600, "re-encode changed the row count");

    // The phantom-vector symptom of #353 is a re-encode replaying more
    // rows than the index holds, so the post-crossing length above is the
    // load-bearing assertion. Also confirm the index stays coherent: no
    // returned slot may point past the end.
    let res = idx.search(&gaussian_normalized(1, dim, 7), 20);
    assert_eq!(res.indices.len(), 20);
    assert!(
        res.indices.iter().all(|&i| (i as usize) < idx.len()),
        "search returned a slot past the end of the index"
    );
}
