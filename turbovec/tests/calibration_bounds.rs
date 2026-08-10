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
    for poison in [1e-40f32, 1e-38, 1e-30, 1e-23] {
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
    for ok in [1e-21f32, 1e-6, 0.5, 1.62, 1e6] {
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

    // Rewrite the trailer's first TQ+ scale as 1e-40. The scales live at
    // the tail: [.. shift(dim) .. scale(dim)], so the first scale starts
    // `dim * 4` bytes before the end.
    let mut b = std::fs::read(&path).unwrap();
    let at = b.len() - DIM * 4;
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
