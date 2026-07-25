//! Tests for the validated public constructor `TurboQuantIndex::from_parts`
//! (issues #141, #142; delivers the low-level API requested in #70).
//!
//! `from_parts` is the single chokepoint where all structural invariants
//! are checked once, so an embedder constructing an index from
//! already-decoded bytes gets a named [`FromPartsError`] instead of the
//! panics / OOB reads / silently-wrong indexes that the raw per-module
//! kernels (`encode`, `pack`, `search`, `codebook`) would produce. Those
//! kernels are now `pub(crate)` and unreachable from here — this file only
//! touches the public surface.

use turbovec::{FromPartsError, TurboQuantIndex};

const DIM: usize = 64;
const BITS: usize = 4;
const N: usize = 40;

fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32 / u64::MAX as f32) * 2.0 - 1.0
    };
    let mut out = vec![0.0f32; n * dim];
    for v in out.chunks_mut(dim) {
        for x in v.iter_mut() {
            *x = next();
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    out
}

/// Build a normal index and return it plus its raw parts, exactly the
/// bytes an embedder would persist and later feed back to `from_parts`.
fn built_index() -> TurboQuantIndex {
    let mut index = TurboQuantIndex::new(DIM, BITS).unwrap();
    index.add(&unit_vectors(N, DIM, 7));
    index
}

// ─── happy path ─────────────────────────────────────────────────────────────

#[test]
fn from_parts_round_trip_matches_normal_build() {
    let src = built_index();

    let rebuilt = TurboQuantIndex::from_parts(
        src.dim_opt(),
        src.bit_width(),
        src.len(),
        src.packed_codes().to_vec(),
        src.scales().to_vec(),
        src.tqplus_shift().to_vec(),
        src.tqplus_scale().to_vec(),
    )
    .expect("consistent parts must construct");

    assert_eq!(rebuilt.len(), src.len());
    assert_eq!(rebuilt.dim(), src.dim());
    assert_eq!(rebuilt.bit_width(), src.bit_width());

    // Search both with the same queries: identical results.
    let queries = unit_vectors(6, DIM, 99);
    let a = src.search(&queries, 5);
    let b = rebuilt.search(&queries, 5);
    assert_eq!(a.nq, b.nq);
    assert_eq!(a.k, b.k);
    assert_eq!(a.indices, b.indices);
    assert_eq!(a.scores, b.scores);
}

#[test]
fn from_parts_persists_byte_identically() {
    let src = built_index();
    let rebuilt = TurboQuantIndex::from_parts(
        src.dim_opt(),
        src.bit_width(),
        src.len(),
        src.packed_codes().to_vec(),
        src.scales().to_vec(),
        src.tqplus_shift().to_vec(),
        src.tqplus_scale().to_vec(),
    )
    .unwrap();

    let dir = std::env::temp_dir();
    let fa = dir.join("turbovec_from_parts_a.tv");
    let fb = dir.join("turbovec_from_parts_b.tv");
    src.write(&fa).unwrap();
    rebuilt.write(&fb).unwrap();
    let ba = std::fs::read(&fa).unwrap();
    let bb = std::fs::read(&fb).unwrap();
    let _ = std::fs::remove_file(&fa);
    let _ = std::fs::remove_file(&fb);
    assert_eq!(ba, bb, "from_parts index must persist byte-identically");
}

#[test]
fn from_parts_accepts_lazy_uncommitted() {
    let idx = TurboQuantIndex::from_parts(None, BITS, 0, vec![], vec![], vec![], vec![])
        .expect("canonical lazy state");
    assert_eq!(idx.dim_opt(), None);
    assert_eq!(idx.len(), 0);
}

#[test]
fn from_parts_accepts_v2_shape_empty_tqplus() {
    // A v2 (pre-TQ+) payload arrives with empty calibration and positive
    // n_vectors; from_parts populates identity calibration internally.
    let src = built_index();
    let idx = TurboQuantIndex::from_parts(
        src.dim_opt(),
        src.bit_width(),
        src.len(),
        src.packed_codes().to_vec(),
        src.scales().to_vec(),
        vec![], // empty TQ+ shift
        vec![], // empty TQ+ scale
    )
    .expect("empty TQ+ is the valid v2 shape");
    assert_eq!(idx.len(), src.len());
}

// ─── every invariant → its named error ──────────────────────────────────────

#[test]
fn rejects_bit_width_zero() {
    let err = TurboQuantIndex::from_parts(Some(DIM), 0, 0, vec![], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::BitWidthOutOfRange(0));
}

#[test]
fn rejects_bit_width_five() {
    let err = TurboQuantIndex::from_parts(Some(DIM), 5, 0, vec![], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::BitWidthOutOfRange(5));
}

#[test]
fn rejects_bit_width_40_without_allocating() {
    // #142 codebook DoS: bits=40 fed to the raw codebook would try to
    // collect 2^40 levels (~8.8 TB) and hang. from_parts checks bit_width
    // FIRST — before any codebook is built — so this returns immediately
    // with a cheap named error and allocates nothing.
    let start = std::time::Instant::now();
    let err = TurboQuantIndex::from_parts(Some(DIM), 40, 0, vec![], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::BitWidthOutOfRange(40));
    assert!(
        start.elapsed().as_millis() < 500,
        "validation must reject bits=40 cheaply, not attempt the 2^40 allocation",
    );
}

#[test]
fn rejects_bit_width_64() {
    // #142: bits=64 is a debug shift-overflow panic / release silent-NaN
    // codebook via the raw path. Validation rejects it outright.
    let err = TurboQuantIndex::from_parts(Some(DIM), 64, 0, vec![], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::BitWidthOutOfRange(64));
}

#[test]
fn rejects_dim_zero() {
    // #142: dim=0 panics `Beta::new((dim-1)/2, ...)` in the raw codebook.
    let err = TurboQuantIndex::from_parts(Some(0), BITS, 0, vec![], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::DimNotPositiveMultipleOf8(0));
}

#[test]
fn rejects_dim_one() {
    // #142: dim=1 panics `Beta::new(0, 0)` in the raw codebook.
    let err = TurboQuantIndex::from_parts(Some(1), BITS, 0, vec![], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::DimNotPositiveMultipleOf8(1));
}

#[test]
fn rejects_dim_not_multiple_of_8() {
    let err = TurboQuantIndex::from_parts(Some(60), BITS, 0, vec![], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::DimNotPositiveMultipleOf8(60));
}

#[test]
fn rejects_dim_too_large() {
    let huge = turbovec::MAX_DIM + 8;
    let err = TurboQuantIndex::from_parts(Some(huge), BITS, 0, vec![], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(
        err,
        FromPartsError::DimTooLarge { dim: huge, max: turbovec::MAX_DIM }
    );
}

#[test]
fn rejects_packed_codes_too_short() {
    // dim=64, bits=4, n=2 → expected 2*64*4/8 = 64 bytes.
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 32], vec![1.0; 2], vec![], vec![],
    )
    .unwrap_err();
    assert_eq!(
        err,
        FromPartsError::PackedCodesLengthMismatch { expected: 64, got: 32 }
    );
}

#[test]
fn rejects_packed_codes_too_long() {
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 128], vec![1.0; 2], vec![], vec![],
    )
    .unwrap_err();
    assert_eq!(
        err,
        FromPartsError::PackedCodesLengthMismatch { expected: 64, got: 128 }
    );
}

#[test]
fn rejects_scales_too_short() {
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0; 1], vec![], vec![],
    )
    .unwrap_err();
    assert_eq!(err, FromPartsError::ScalesLengthMismatch { expected: 2, got: 1 });
}

#[test]
fn rejects_scales_too_long() {
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0; 5], vec![], vec![],
    )
    .unwrap_err();
    assert_eq!(err, FromPartsError::ScalesLengthMismatch { expected: 2, got: 5 });
}

#[test]
fn rejects_mismatched_tqplus_lengths() {
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0; 2], vec![0.0; 64], vec![1.0; 32],
    )
    .unwrap_err();
    assert_eq!(
        err,
        FromPartsError::TqplusLengthMismatch { shift_len: 64, scale_len: 32 }
    );
}

#[test]
fn rejects_tqplus_length_not_dim() {
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0; 2], vec![0.0; 48], vec![1.0; 48],
    )
    .unwrap_err();
    assert_eq!(err, FromPartsError::TqplusLengthNotDim { got: 48, dim: 64 });
}

#[test]
fn rejects_lazy_with_nonzero_n_vectors() {
    let err = TurboQuantIndex::from_parts(None, 4, 5, vec![], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::LazyMustHaveZeroVectors(5));
}

#[test]
fn rejects_lazy_with_nonempty_packed_codes() {
    let err = TurboQuantIndex::from_parts(None, 4, 0, vec![0u8; 8], vec![], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::LazyMustHaveEmptyPackedCodes(8));
}

#[test]
fn rejects_lazy_with_nonempty_scales() {
    let err = TurboQuantIndex::from_parts(None, 4, 0, vec![], vec![1.0; 3], vec![], vec![])
        .unwrap_err();
    assert_eq!(err, FromPartsError::LazyMustHaveEmptyScales(3));
}

#[test]
fn rejects_lazy_with_nonempty_tqplus() {
    let err = TurboQuantIndex::from_parts(None, 4, 0, vec![], vec![], vec![0.0; 4], vec![0.0; 4])
        .unwrap_err();
    assert_eq!(err, FromPartsError::LazyMustHaveEmptyTqplus(4));
}

// ─── error is a proper std::error::Error with a readable message ─────────────

#[test]
fn error_displays_readable_message() {
    let err = TurboQuantIndex::from_parts(Some(DIM), 7, 0, vec![], vec![], vec![], vec![])
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("bit_width"), "message was: {msg}");
    // Exercised as a boxed std::error::Error.
    let _boxed: Box<dyn std::error::Error> = Box::new(err);
}
