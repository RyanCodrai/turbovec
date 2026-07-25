//! Errors returned by the user-facing add and construct paths.
//!
//! [`AddError`] is returned by the add paths
//! ([`TurboQuantIndex::add_2d`](crate::TurboQuantIndex::add_2d),
//! [`IdMapIndex::add_with_ids_2d`](crate::IdMapIndex::add_with_ids_2d),
//! [`IdMapIndex::add_with_ids`](crate::IdMapIndex::add_with_ids)).
//!
//! [`ConstructError`] is returned by the constructors
//! ([`TurboQuantIndex::new`](crate::TurboQuantIndex::new),
//! [`TurboQuantIndex::new_lazy`](crate::TurboQuantIndex::new_lazy),
//! [`IdMapIndex::new`](crate::IdMapIndex::new),
//! [`IdMapIndex::new_lazy`](crate::IdMapIndex::new_lazy)).
//!
//! [`FromPartsError`] is returned by the low-level validated constructor
//! [`TurboQuantIndex::from_parts`](crate::TurboQuantIndex::from_parts),
//! which builds an index directly from already-decoded fields and checks
//! every structural invariant at that single chokepoint.
//!
//! Both are forms of user input error — wrong shape, wrong dim, wrong
//! bit_width, or duplicate id — that callers can recover from. Internal
//! preconditions (e.g. calling the low-level `add(&self, &[f32])` on a
//! lazy index that hasn't been committed) still panic, since that
//! signals a contract violation rather than bad input.

use std::error::Error;
use std::fmt;

// Eq dropped from the derive because `InvalidInputValue` carries an f32,
// which is not `Eq` (NaN != NaN). PartialEq still works for the
// finite-input cases tests assert against.
// `#[non_exhaustive]` so adding error variants in future releases is not a
// breaking change — downstream `match` on this enum must carry a wildcard arm.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AddError {
    /// Batch dim does not match the index's already-locked dim.
    DimMismatch { existing: usize, got: usize },

    /// First-add dim on a lazy index must be a multiple of 8.
    DimNotMultipleOf8(usize),

    /// First-add dim on a lazy index exceeds [`MAX_DIM`](crate::MAX_DIM).
    /// Bounds the lazily-built `dim`×`dim` rotation matrix allocation.
    DimTooLarge { dim: usize, max: usize },

    /// `vectors.len()` is not a whole multiple of `dim`.
    VectorBufferNotMultipleOfDim { vectors_len: usize, dim: usize },

    /// Number of ids does not equal number of vectors (`vectors.len() / dim`).
    IdsCountMismatch { expected: usize, got: usize },

    /// External id was already present in the index.
    IdAlreadyPresent(u64),

    /// A coordinate in the input vectors is not finite (NaN, +Inf, -Inf)
    /// or has magnitude `>= 1e16`. Either silently corrupts the index:
    ///   - NaN/Inf: poisons the per-vector scale via `0 * NaN = NaN`,
    ///     making the slot exist in `len()` but never reachable through
    ///     `search`.
    ///   - Huge magnitude: overflows the f32 sum-of-squares in the norm
    ///     computation to `+Inf`, so `scale[i] = Inf` and the slot
    ///     incorrectly wins top-k against every query.
    InvalidInputValue {
        vector_index: usize,
        coord_index: usize,
        value: f32,
    },
}

impl fmt::Display for AddError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimMismatch { existing, got } => {
                write!(f, "dim mismatch: index dim={existing}, batch dim={got}")
            }
            Self::DimNotMultipleOf8(dim) => {
                write!(f, "dim must be a multiple of 8, got {dim}")
            }
            Self::DimTooLarge { dim, max } => {
                write!(f, "dim {dim} exceeds maximum {max}")
            }
            Self::VectorBufferNotMultipleOfDim { vectors_len, dim } => write!(
                f,
                "vector buffer length {vectors_len} not a multiple of dim {dim}",
            ),
            Self::IdsCountMismatch { expected, got } => {
                write!(f, "expected {expected} ids, got {got}")
            }
            Self::IdAlreadyPresent(id) => {
                write!(f, "id {id} already present in index")
            }
            Self::InvalidInputValue {
                vector_index,
                coord_index,
                value,
            } => write!(
                f,
                "invalid input value at vector {vector_index}, coord {coord_index}: {value} \
                 (must be finite and |value| < 1e16 to avoid f32 norm overflow)",
            ),
        }
    }
}

impl Error for AddError {}

// `#[non_exhaustive]` so adding error variants in future releases is not a
// breaking change — downstream `match` on this enum must carry a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConstructError {
    /// `bit_width` must be 2, 3, or 4.
    BitWidthOutOfRange(usize),

    /// `dim` must be a positive multiple of 8.
    DimNotPositiveMultipleOf8(usize),

    /// `dim` exceeds [`MAX_DIM`](crate::MAX_DIM). Bounds the lazily-built
    /// `dim`×`dim` rotation matrix allocation.
    DimTooLarge { dim: usize, max: usize },
}

impl fmt::Display for ConstructError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitWidthOutOfRange(bw) => {
                write!(f, "bit_width must be 2, 3, or 4, got {bw}")
            }
            Self::DimNotPositiveMultipleOf8(dim) => {
                write!(f, "dim must be a positive multiple of 8, got {dim}")
            }
            Self::DimTooLarge { dim, max } => {
                write!(f, "dim {dim} exceeds maximum {max}")
            }
        }
    }
}

impl Error for ConstructError {}

/// Error returned by
/// [`TurboQuantIndex::from_parts`](crate::TurboQuantIndex::from_parts) when
/// the supplied fields violate one of the index's structural invariants.
///
/// `from_parts` is the single validated entry point for constructing an
/// index directly from already-decoded bytes (the low-level API a
/// database-storage embedder builds against — see the crate docs). Every
/// invariant it checks maps to one variant here, so a caller passing a
/// mismatched buffer, an out-of-range `bit_width`, or an inconsistent lazy
/// state gets a named error instead of a panic, an out-of-bounds read, or a
/// silently-wrong index.
///
/// `#[non_exhaustive]` so adding variants in future releases is not a
/// breaking change — downstream `match` must carry a wildcard arm.
// Eq is not derived because the value-validation variants carry an f32,
// which is not `Eq` (NaN != NaN). PartialEq still works for the finite
// values tests assert against.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FromPartsError {
    /// `bit_width` must be 2, 3, or 4.
    BitWidthOutOfRange(usize),

    /// `dim` (when committed, i.e. `Some`) must be a positive multiple of 8.
    /// The packed layout allocates `dim / 8` bytes per bit-plane, so no
    /// other dim has a valid layout.
    DimNotPositiveMultipleOf8(usize),

    /// `dim` exceeds [`MAX_DIM`](crate::MAX_DIM). Bounds the lazily-built
    /// `dim`×`dim` rotation matrix and the `bit_width`/`dim` codebook
    /// allocation (guards the unbounded-allocation DoS class).
    DimTooLarge { dim: usize, max: usize },

    /// `n_vectors * dim * bit_width / 8` overflows `usize`, so no
    /// `packed_codes` buffer of the implied length can exist. Mirrors the
    /// loader's checked size arithmetic.
    PackedCodesSizeOverflow {
        n_vectors: usize,
        dim: usize,
        bit_width: usize,
    },

    /// `packed_codes.len()` does not equal the length implied by
    /// `n_vectors * dim * bit_width / 8`.
    PackedCodesLengthMismatch { expected: usize, got: usize },

    /// `scales.len()` does not equal `n_vectors`.
    ScalesLengthMismatch { expected: usize, got: usize },

    /// The two TQ+ calibration arrays disagree in length
    /// (`tqplus_shift.len() != tqplus_scale.len()`).
    TqplusLengthMismatch { shift_len: usize, scale_len: usize },

    /// A non-empty TQ+ calibration array has a length that is not `dim`.
    TqplusLengthNotDim { got: usize, dim: usize },

    /// A per-vector scale is not finite or is negative. The encoder only
    /// ever emits finite, non-negative scales; an Inf slot would win every
    /// top-1 and a NaN slot would vanish from all results. Mirrors the
    /// loader's value validation, so a `from_parts`-accepted index always
    /// survives its own `write` → `load` round-trip.
    InvalidScaleValue { slot: usize, value: f32 },

    /// A TQ+ shift coordinate is not finite. Mirrors the loader's value
    /// validation.
    InvalidTqplusShiftValue { coord: usize, value: f32 },

    /// A TQ+ scale coordinate is not finite or is `<= 0`. Search divides
    /// by `tqplus_scale`, so such a value silently turns every query's
    /// scores into NaN/Inf. Mirrors the loader's value validation.
    InvalidTqplusScaleValue { coord: usize, value: f32 },

    /// Lazy (uncommitted, `dim == None`) index must have `n_vectors == 0`.
    LazyMustHaveZeroVectors(usize),

    /// Lazy index must have empty `packed_codes`.
    LazyMustHaveEmptyPackedCodes(usize),

    /// Lazy index must have empty `scales`.
    LazyMustHaveEmptyScales(usize),

    /// Lazy index must have empty TQ+ calibration arrays.
    LazyMustHaveEmptyTqplus(usize),
}

impl fmt::Display for FromPartsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitWidthOutOfRange(bw) => {
                write!(f, "bit_width must be 2, 3, or 4, got {bw}")
            }
            Self::DimNotPositiveMultipleOf8(dim) => {
                write!(f, "dim must be a positive multiple of 8, got {dim}")
            }
            Self::DimTooLarge { dim, max } => {
                write!(f, "dim {dim} exceeds maximum {max}")
            }
            Self::PackedCodesSizeOverflow { n_vectors, dim, bit_width } => write!(
                f,
                "packed code size n_vectors({n_vectors}) * dim({dim}) * \
                 bit_width({bit_width}) / 8 overflows usize",
            ),
            Self::PackedCodesLengthMismatch { expected, got } => write!(
                f,
                "packed_codes length {got} != n_vectors * dim * bit_width / 8 = {expected}",
            ),
            Self::ScalesLengthMismatch { expected, got } => {
                write!(f, "scales length {got} != n_vectors {expected}")
            }
            Self::TqplusLengthMismatch { shift_len, scale_len } => write!(
                f,
                "tqplus_shift length {shift_len} != tqplus_scale length {scale_len}",
            ),
            Self::TqplusLengthNotDim { got, dim } => {
                write!(f, "non-empty TQ+ calibration length {got} must equal dim {dim}")
            }
            Self::InvalidScaleValue { slot, value } => write!(
                f,
                "invalid per-vector scale at slot {slot}: {value} (must be finite and non-negative)",
            ),
            Self::InvalidTqplusShiftValue { coord, value } => {
                write!(f, "invalid TQ+ shift at coord {coord}: {value} (must be finite)")
            }
            Self::InvalidTqplusScaleValue { coord, value } => write!(
                f,
                "invalid TQ+ scale at coord {coord}: {value} (must be finite and > 0)",
            ),
            Self::LazyMustHaveZeroVectors(n) => {
                write!(f, "lazy (uncommitted-dim) index must have n_vectors=0, got {n}")
            }
            Self::LazyMustHaveEmptyPackedCodes(len) => {
                write!(f, "lazy index must have empty packed_codes, got length {len}")
            }
            Self::LazyMustHaveEmptyScales(len) => {
                write!(f, "lazy index must have empty scales, got length {len}")
            }
            Self::LazyMustHaveEmptyTqplus(len) => {
                write!(f, "lazy index must have empty TQ+ calibration, got length {len}")
            }
        }
    }
}

impl Error for FromPartsError {}
