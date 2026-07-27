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


use turbovec::{io, TurboQuantIndex};

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
    assert_eq!(idx.dim(), dim);
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
