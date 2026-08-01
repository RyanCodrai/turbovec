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

use turbovec::{FromPartsError, ToPartsError, TurboQuantIndex};

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
    assert_eq!(rebuilt.dim_opt().unwrap(), src.dim_opt().unwrap());
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

// ─── overflow-safe size math ────────────────────────────────────────────────

#[test]
fn rejects_packed_size_overflow_with_named_error() {
    // n_vectors at 2^60 scale: (64/8) * 4 * 2^60 = 2^65 overflows usize.
    // Must return the named error — not debug-panic or release-wrap into a
    // bogus expected length.
    let n = 1usize << 60;
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, n, vec![], vec![], vec![], vec![],
    )
    .unwrap_err();
    assert_eq!(
        err,
        FromPartsError::PackedCodesSizeOverflow { n_vectors: n, dim: 64, bit_width: 4 }
    );
}

// ─── value validation: parity with the .tv loader ───────────────────────────
// The loader rejects non-finite/negative per-vector scales, non-finite TQ+
// shifts, and non-finite-or-<=0 TQ+ scales. from_parts applies exactly the
// same checks, so an accepted index always survives its own write → load
// round-trip (pinned by the property test below).

#[test]
fn rejects_nan_per_vector_scale() {
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0, f32::NAN], vec![], vec![],
    )
    .unwrap_err();
    assert!(
        matches!(err, FromPartsError::InvalidScaleValue { slot: 1, value } if value.is_nan()),
        "got {err:?}",
    );
}

#[test]
fn rejects_infinite_per_vector_scale() {
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![f32::INFINITY, 1.0], vec![], vec![],
    )
    .unwrap_err();
    assert_eq!(
        err,
        FromPartsError::InvalidScaleValue { slot: 0, value: f32::INFINITY }
    );
}

#[test]
fn rejects_negative_per_vector_scale() {
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0, -0.5], vec![], vec![],
    )
    .unwrap_err();
    assert_eq!(err, FromPartsError::InvalidScaleValue { slot: 1, value: -0.5 });
}

#[test]
fn accepts_zero_per_vector_scale() {
    // The encoder emits scale 0 for zero-norm vectors; 0 is valid.
    TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![0.0, 1.0], vec![], vec![],
    )
    .expect("zero scale is a legitimate encoder output");
}

#[test]
fn rejects_nan_tqplus_shift() {
    let mut shift = vec![0.0f32; 64];
    shift[3] = f32::NAN;
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0; 2], shift, vec![1.0; 64],
    )
    .unwrap_err();
    assert!(
        matches!(err, FromPartsError::InvalidTqplusShiftValue { coord: 3, value } if value.is_nan()),
        "got {err:?}",
    );
}

#[test]
fn rejects_zero_tqplus_scale() {
    let mut scale = vec![1.0f32; 64];
    scale[7] = 0.0;
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0; 2], vec![0.0; 64], scale,
    )
    .unwrap_err();
    assert_eq!(
        err,
        FromPartsError::InvalidTqplusScaleValue { coord: 7, value: 0.0 }
    );
}

#[test]
fn rejects_negative_tqplus_scale() {
    let mut scale = vec![1.0f32; 64];
    scale[0] = -1.0;
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0; 2], vec![0.0; 64], scale,
    )
    .unwrap_err();
    assert_eq!(
        err,
        FromPartsError::InvalidTqplusScaleValue { coord: 0, value: -1.0 }
    );
}

#[test]
fn rejects_infinite_tqplus_scale() {
    let mut scale = vec![1.0f32; 64];
    scale[63] = f32::INFINITY;
    let err = TurboQuantIndex::from_parts(
        Some(64), 4, 2, vec![0u8; 64], vec![1.0; 2], vec![0.0; 64], scale,
    )
    .unwrap_err();
    assert_eq!(
        err,
        FromPartsError::InvalidTqplusScaleValue { coord: 63, value: f32::INFINITY }
    );
}

// ─── property: accepted ⇒ survives its own write → load round-trip ──────────

/// Deterministic xorshift for the fuzz below.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn f32_maybe_bad(&mut self) -> f32 {
        match self.below(12) {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => -1.0,
            4 => 0.0,
            _ => (self.below(1000) as f32) / 250.0 + 0.001,
        }
    }
}

#[test]
fn fuzz_accepted_parts_always_round_trip_through_write_load() {
    // 400 random part tuples, some valid and some perturbed. The property:
    // whenever from_parts ACCEPTS a tuple, the resulting index must write
    // and re-load through the file loader without error (value-validation
    // parity), and the loaded index must expose identical parts.
    let mut rng = Rng(0x5eed_cafe_f00d_1234);
    let dir = std::env::temp_dir();
    let path = dir.join("turbovec_from_parts_fuzz.tv");
    let mut accepted = 0usize;

    for iter in 0..400 {
        let bit_width = rng.below(8) as usize; // 0..8, valid ⊂ {2,3,4}
        let lazy = rng.below(8) == 0;
        let dim = (rng.below(10) as usize) * 8; // 0..72, mult of 8 (0 invalid)
        // Mostly small n; occasionally 2^60-scale so the checked size math
        // is exercised under fuzz too (must reject via named error, never
        // debug-panic on overflow or release-wrap into a bogus length).
        let n = if rng.below(16) == 0 {
            (1usize << 60) + rng.below(4) as usize
        } else {
            rng.below(6) as usize
        };
        // Lengths: mostly consistent, sometimes perturbed.
        let jitter = |rng: &mut Rng, len: usize| -> usize {
            match rng.below(6) {
                0 => len + 1,
                1 => len.saturating_sub(1),
                _ => len,
            }
        };
        let (dim_opt, packed_len, scales_len, tq_len) = if lazy {
            (None, jitter(&mut rng, 0), jitter(&mut rng, 0), jitter(&mut rng, 0))
        } else {
            // u128 + cap: the huge-n draws would overflow this very
            // computation (and allocating a matching buffer is impossible
            // anyway); capped buffers simply get rejected by from_parts'
            // overflow/length checks, which is the point.
            let packed =
                ((n as u128 * dim as u128 * bit_width as u128) / 8).min(8192) as usize;
            let tq = if rng.below(2) == 0 { 0 } else { dim };
            (
                Some(dim),
                jitter(&mut rng, packed),
                jitter(&mut rng, n.min(8192)),
                jitter(&mut rng, tq),
            )
        };
        let n_eff = if lazy { 0 } else { n };
        let packed_codes: Vec<u8> = (0..packed_len).map(|_| rng.next() as u8).collect();
        let scales: Vec<f32> = (0..scales_len)
            .map(|_| rng.f32_maybe_bad().abs() * if rng.below(10) == 0 { -1.0 } else { 1.0 })
            .collect();
        let scales: Vec<f32> = if rng.below(4) == 0 {
            (0..scales_len).map(|_| rng.f32_maybe_bad()).collect()
        } else {
            scales
        };
        let tqplus_shift: Vec<f32> = (0..tq_len).map(|_| rng.f32_maybe_bad()).collect();
        let tqplus_scale: Vec<f32> = (0..tq_len).map(|_| rng.f32_maybe_bad()).collect();

        let built = TurboQuantIndex::from_parts(
            dim_opt,
            bit_width,
            n_eff,
            packed_codes,
            scales,
            tqplus_shift,
            tqplus_scale,
        );
        let Ok(index) = built else { continue };
        accepted += 1;

        // Property: an accepted index survives its own write → load.
        index
            .write(&path)
            .unwrap_or_else(|e| panic!("iter {iter}: write failed for accepted parts: {e}"));
        let loaded = TurboQuantIndex::load(&path).unwrap_or_else(|e| {
            panic!("iter {iter}: loader rejected a from_parts-accepted index: {e}")
        });
        assert_eq!(loaded.dim_opt(), index.dim_opt(), "iter {iter}: dim drift");
        assert_eq!(loaded.bit_width(), index.bit_width(), "iter {iter}: bit_width drift");
        assert_eq!(loaded.len(), index.len(), "iter {iter}: len drift");
        assert_eq!(loaded.packed_codes(), index.packed_codes(), "iter {iter}: codes drift");
        assert_eq!(loaded.scales(), index.scales(), "iter {iter}: scales drift");
        assert_eq!(
            loaded.tqplus_shift(),
            index.tqplus_shift(),
            "iter {iter}: tqplus_shift drift",
        );
        assert_eq!(
            loaded.tqplus_scale(),
            index.tqplus_scale(),
            "iter {iter}: tqplus_scale drift",
        );
    }
    let _ = std::fs::remove_file(&path);
    // Sanity: the generator must actually exercise the accept path.
    assert!(
        accepted >= 20,
        "fuzz generator produced only {accepted} accepted tuples out of 400",
    );
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

// ─── holes ──────────────────────────────────────────────────────────────────

/// The parts carry no block table, so they cannot say that a slot holds
/// no vector. A removal inside a sealed block leaves one, and
/// `packed_codes`/`scales` span the slot capacity and include it.
///
/// Two things have to hold. Following the documented recipe — passing
/// `len()` — must fail loudly rather than quietly producing an index
/// with the deleted vector back in it. And callers must be able to see
/// the condition coming, which is what `is_compact` is for.
#[test]
fn a_holed_index_cannot_be_rebuilt_from_its_parts() {
    const BS: usize = 1024;
    let n = 3 * BS;
    let rows: Vec<f32> = (0..n * DIM)
        .map(|i| ((i % 251) as f32 / 251.0) - 0.5)
        .collect();
    let mut src = TurboQuantIndex::with_block_size(DIM, BITS, BS).unwrap();
    src.add(&rows);
    assert!(src.is_compact(), "a freshly built index has no holes");

    src.swap_remove(0);
    assert!(!src.is_compact(), "a removal inside a sealed block left no hole");
    assert_eq!(src.len(), n - 1);
    assert_eq!(src.slot_capacity(), n, "the sealed block gave its extent back");
    assert!(!src.slot_is_live(BS - 1), "the vacated tail row is still live");

    // The documented recipe passes `len()`, and the codes are one row
    // longer than that, so it is refused rather than accepted.
    let err = TurboQuantIndex::from_parts(
        src.dim_opt(),
        src.bit_width(),
        src.len(),
        src.packed_codes().to_vec(),
        src.scales().to_vec(),
        src.tqplus_shift().to_vec(),
        src.tqplus_scale().to_vec(),
    )
    .expect_err("a holed index must not rebuild from its parts");
    assert!(
        matches!(err, FromPartsError::PackedCodesLengthMismatch { .. }),
        "expected a length mismatch naming the codes, got {err}",
    );

    // The supported route reproduces the holes exactly, because the file
    // carries the block table.
    let round_tripped = TurboQuantIndex::from_bytes(&src.to_bytes()).unwrap();
    assert_eq!(round_tripped.len(), n - 1);
    assert_eq!(round_tripped.slot_capacity(), n);
    assert!(!round_tripped.is_compact());
    assert!(!round_tripped.slot_is_live(BS - 1), "the hole did not survive the round trip");
}

// ─── to_parts ───────────────────────────────────────────────────────────────

/// `to_parts` exists so the unsafe route is unrepresentable rather than
/// merely undocumented. `is_compact` makes the hazard visible, but a
/// caller who never reads the doc and reaches for `slot_capacity()` to
/// make the lengths line up still gets the removed rows back. Handing
/// out the parts through a checked method removes that route.
#[test]
fn to_parts_round_trips_a_compact_index() {
    let src = built_index();
    let parts = src.to_parts().expect("a compact index hands out its parts");
    assert_eq!(parts.dim, src.dim_opt());
    assert_eq!(parts.bit_width, src.bit_width());
    assert_eq!(parts.n_vectors, src.len());

    let rebuilt = parts.into_index().expect("consistent parts");
    assert_eq!(rebuilt.len(), src.len());
    assert_eq!(rebuilt.packed_codes(), src.packed_codes());
    assert_eq!(rebuilt.scales(), src.scales());
    assert_eq!(rebuilt.tqplus_shift(), src.tqplus_shift());

    let q: Vec<f32> = (0..DIM).map(|i| (i as f32 / DIM as f32) - 0.5).collect();
    assert_eq!(src.search(&q, 5), rebuilt.search(&q, 5));
}

#[test]
fn to_parts_refuses_an_index_with_dead_slots() {
    const BS: usize = 1024;
    let n = 3 * BS;
    let rows: Vec<f32> = (0..n * DIM)
        .map(|i| ((i % 251) as f32 / 251.0) - 0.5)
        .collect();
    let mut src = TurboQuantIndex::with_block_size(DIM, BITS, BS).unwrap();
    src.add(&rows);
    assert!(src.to_parts().is_ok(), "no holes yet");

    src.swap_remove(0);
    let err = src.to_parts().expect_err("a holed index must not hand out parts");
    assert_eq!(
        err,
        ToPartsError::NotCompact { live: n - 1, slots: n },
        "the error must name both counts so the caller can see the gap",
    );
    assert!(
        err.to_string().contains("to_bytes"),
        "the refusal must point at the route that does work: {err}",
    );

    // Draining the hole's block back to nothing is not a fix — the
    // extent stays — but emptying the whole index is, since there is
    // then nothing to be inconsistent about.
    let bytes = src.to_bytes();
    let back = TurboQuantIndex::from_bytes(&bytes).unwrap();
    assert!(back.to_parts().is_err(), "the round trip preserved the holes, as it should");
}

#[test]
fn to_parts_handles_the_empty_and_lazy_shapes() {
    let lazy = TurboQuantIndex::new_lazy(BITS).unwrap();
    let parts = lazy.to_parts().expect("a lazy index is trivially compact");
    assert_eq!(parts.dim, None);
    assert_eq!(parts.n_vectors, 0);
    assert!(parts.packed_codes.is_empty() && parts.scales.is_empty());
    assert_eq!(parts.into_index().unwrap().dim_opt(), None);

    let empty = TurboQuantIndex::new(DIM, BITS).unwrap();
    let rebuilt = empty.to_parts().unwrap().into_index().unwrap();
    assert_eq!(rebuilt.dim_opt(), Some(DIM));
    assert_eq!(rebuilt.len(), 0);
}
