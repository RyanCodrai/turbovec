//! Errors returned by the user-facing construct, add and search paths.
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
//! [`SearchError`] is returned by the fallible search paths
//! ([`TurboQuantIndex::try_search`](crate::TurboQuantIndex::try_search),
//! [`TurboQuantIndex::try_search_with_mask`](crate::TurboQuantIndex::try_search_with_mask),
//! [`IdMapIndex::search_with_allowlist`](crate::IdMapIndex::search_with_allowlist)).
//!
//! [`FromPartsError`] is returned by the low-level validated constructor
//! [`TurboQuantIndex::from_parts`](crate::TurboQuantIndex::from_parts),
//! which builds an index directly from already-decoded fields and checks
//! every structural invariant at that single chokepoint.
//!
//! All four are forms of user input error — wrong shape, wrong dim, wrong
//! bit_width, a non-representable coordinate, or a duplicate id — that
//! callers can recover from. Internal
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
/// Why an `add` / `add_with_ids` batch was rejected.
///
/// Every variant is raised before any row is written, so a rejected batch
/// leaves the index exactly as it was.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AddError {
    /// Batch dim does not match the index's already-locked dim.
    DimMismatch {
        /// Dim the index is already committed to.
        existing: usize,
        /// Dim implied by this batch.
        got: usize,
    },

    /// First-add dim on a lazy index must be a multiple of 8.
    DimNotMultipleOf8(usize),

    /// First-add dim on a lazy index exceeds [`MAX_DIM`](crate::MAX_DIM).
    /// Bounds the lazily-built `dim`×`dim` rotation matrix allocation.
    DimTooLarge {
        /// Dim the batch asked for.
        dim: usize,
        /// The ceiling, [`MAX_DIM`](crate::MAX_DIM).
        max: usize,
    },

    /// `vectors.len()` is not a whole multiple of `dim`.
    VectorBufferNotMultipleOfDim {
        /// Length of the flat `vectors` slice.
        vectors_len: usize,
        /// Dim it was divided by.
        dim: usize,
    },

    /// `dim` is 0 — the batch has no columns at all. Kept distinct from
    /// [`Self::VectorBufferNotMultipleOfDim`] and
    /// [`Self::DimNotMultipleOf8`]: neither describes a zero dim
    /// truthfully (every length is a multiple of 0, and `% 0` is
    /// undefined), and the real cause is almost always an embedder that
    /// returned empty embeddings.
    ZeroDim,

    /// Number of ids does not equal number of vectors (`vectors.len() / dim`).
    IdsCountMismatch {
        /// Number of vector rows in the batch.
        expected: usize,
        /// Number of ids supplied.
        got: usize,
    },

    /// External id was already present in the index.
    IdAlreadyPresent(u64),

    /// External id appears more than once within the same batch. Kept
    /// distinct from [`Self::IdAlreadyPresent`], which would send the
    /// caller hunting for a prior insert that never happened.
    DuplicateIdInBatch(u64),

    /// A coordinate in the input vectors is not finite (NaN, +Inf, -Inf)
    /// or has magnitude `>= 1e16`. Either silently corrupts the index:
    ///   - NaN/Inf: poisons the per-vector scale via `0 * NaN = NaN`,
    ///     making the slot exist in `len()` but never reachable through
    ///     `search`.
    ///   - Huge magnitude: overflows the f32 sum-of-squares in the norm
    ///     computation to `+Inf`, so `scale[i] = Inf` and the slot
    ///     incorrectly wins top-k against every query.
    InvalidInputValue {
        /// Row within the batch (0-based), not a slot in the index.
        vector_index: usize,
        /// Coordinate within that row.
        coord_index: usize,
        /// The offending value.
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
            Self::ZeroDim => write!(
                f,
                "dim is 0: the vectors have no columns (an embedder that \
                 returned empty embeddings is the usual cause)",
            ),
            Self::IdsCountMismatch { expected, got } => {
                write!(f, "expected {expected} ids, got {got}")
            }
            Self::IdAlreadyPresent(id) => {
                write!(f, "id {id} already present in index")
            }
            Self::DuplicateIdInBatch(id) => {
                write!(f, "duplicate id {id} appears more than once in this batch")
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
/// Why a `new` / `with_bit_width` constructor rejected its arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConstructError {
    /// `bit_width` must be 2, 3, or 4.
    BitWidthOutOfRange(usize),

    /// `dim` must be a positive multiple of 8.
    DimNotPositiveMultipleOf8(usize),

    /// `dim` exceeds [`MAX_DIM`](crate::MAX_DIM). Bounds the lazily-built
    /// `dim`×`dim` rotation matrix allocation.
    DimTooLarge {
        /// Dim the caller asked for.
        dim: usize,
        /// The ceiling, [`MAX_DIM`](crate::MAX_DIM).
        max: usize,
    },

    /// `block_size` must be a positive multiple of
    /// [`MIN_BLOCK_SIZE`](crate::MIN_BLOCK_SIZE) — the granularity at
    /// which the 32-row SIMD code layout and the 64-slot packed search
    /// mask both start a fresh unit, and so the only granularity at
    /// which a block is searchable as a self-contained range.
    BlockSizeInvalid {
        /// Block size the caller asked for.
        block_size: usize,
        /// The required granularity, [`MIN_BLOCK_SIZE`](crate::MIN_BLOCK_SIZE).
        granularity: usize,
    },
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
            Self::BlockSizeInvalid { block_size, granularity } => {
                write!(
                    f,
                    "block_size must be a positive multiple of {granularity}, got {block_size}"
                )
            }
        }
    }
}

impl Error for ConstructError {}

/// Error returned by the crate's fallible search paths:
/// [`TurboQuantIndex::try_search`](crate::TurboQuantIndex::try_search),
/// [`TurboQuantIndex::try_search_with_mask`](crate::TurboQuantIndex::try_search_with_mask)
/// and
/// [`IdMapIndex::search_with_allowlist`](crate::IdMapIndex::search_with_allowlist).
///
/// Every variant describes *caller-supplied data* that the index cannot
/// score: a query buffer whose length disagrees with the index dim, a
/// coordinate the scoring kernel cannot represent, a mask sized for a
/// different index, or an allowlist that drifted out of step with the
/// index's contents. All four arrive from outside the process in a real
/// service — an embedding endpoint, a metadata store, an HTTP body — so
/// they are reported rather than panicked. The Python binding already
/// maps them to `ValueError` / `KeyError`.
///
/// Which variants a given method can produce:
///
/// | variant | `try_search` | `try_search_with_mask` | `search_with_allowlist` |
/// |---|---|---|---|
/// | [`QueryBufferNotMultipleOfDim`](Self::QueryBufferNotMultipleOfDim) | yes | yes | yes |
/// | [`InvalidQueryValue`](Self::InvalidQueryValue) | yes | yes | yes |
/// | [`MaskLengthMismatch`](Self::MaskLengthMismatch) | no | yes | no |
/// | [`AllowlistEmpty`](Self::AllowlistEmpty) | no | no | yes |
/// | [`UnknownId`](Self::UnknownId) | no | no | yes |
///
/// `#[non_exhaustive]` so adding variants in future releases is not a
/// breaking change — downstream `match` must carry a wildcard arm.
// Eq is not derived because `InvalidQueryValue` carries an f32, which is
// not `Eq` (NaN != NaN) — the same reason `AddError` and `FromPartsError`
// drop it. PartialEq still works for the finite values tests assert
// against, and every other variant compares as before.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SearchError {
    /// The allowlist was `Some` but empty. An empty allowlist selects no
    /// slots, which is almost always a caller-side filter bug rather than
    /// a request for zero results; pass `None` to search everything.
    AllowlistEmpty,

    /// An allowlist id is not present in the index.
    UnknownId(u64),

    /// `queries.len()` is not a whole multiple of the index dim, so the
    /// buffer does not describe a whole number of query rows.
    QueryBufferNotMultipleOfDim {
        /// Length of the flat `queries` slice.
        queries_len: usize,
        /// Index dim it was divided by.
        dim: usize,
    },

    /// A query coordinate is not finite (NaN, +Inf, -Inf) or has
    /// magnitude `>= 1e16`. Such a value poisons the SIMD scoring kernel:
    /// the accumulator goes to NaN/Inf and the query's top-`k` becomes
    /// arbitrary indices with meaningless scores, silently.
    InvalidQueryValue {
        /// Query row within the batch (0-based).
        query_index: usize,
        /// Coordinate within that row.
        coord_index: usize,
        /// The offending value.
        value: f32,
    },

    /// The search mask's length does not equal the index's slot count,
    /// so slot `i` of the mask does not name slot `i` of the index.
    MaskLengthMismatch {
        /// The index's
        /// [`slot_capacity()`](crate::TurboQuantIndex::slot_capacity),
        /// which the mask must match — one entry per storage slot. Not
        /// `len()`: a removal can leave a slot holding nothing, and the
        /// mask still has an entry for it.
        expected: usize,
        /// The mask length supplied.
        got: usize,
    },
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllowlistEmpty => write!(f, "allowlist is empty"),
            Self::UnknownId(id) => {
                write!(f, "id {id} in allowlist is not present in index")
            }
            Self::QueryBufferNotMultipleOfDim { queries_len, dim } => write!(
                f,
                "query buffer length {queries_len} not a multiple of dim {dim}",
            ),
            Self::InvalidQueryValue {
                query_index,
                coord_index,
                value,
            } => write!(
                f,
                "invalid query value at query {query_index}, coord {coord_index}: {value} \
                 (must be finite and |value| < 1e16 to avoid f32 overflow)",
            ),
            Self::MaskLengthMismatch { expected, got } => write!(
                f,
                "mask length {got} does not match index slot capacity {expected}",
            ),
        }
    }
}

impl Error for SearchError {}

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
    DimTooLarge {
        /// Dim supplied.
        dim: usize,
        /// The ceiling, [`MAX_DIM`](crate::MAX_DIM).
        max: usize,
    },

    /// `n_vectors * dim * bit_width / 8` overflows `usize`, so no
    /// `packed_codes` buffer of the implied length can exist. Mirrors the
    /// loader's checked size arithmetic.
    PackedCodesSizeOverflow {
        /// Row count supplied.
        n_vectors: usize,
        /// Dim supplied.
        dim: usize,
        /// Bit width supplied.
        bit_width: usize,
    },

    /// `packed_codes.len()` does not equal the length implied by
    /// `n_vectors * dim * bit_width / 8`.
    PackedCodesLengthMismatch {
        /// Byte length implied by `n_vectors`, `dim` and `bit_width`.
        expected: usize,
        /// Byte length of the `packed_codes` supplied.
        got: usize,
    },

    /// `scales.len()` does not equal `n_vectors`.
    ScalesLengthMismatch {
        /// `n_vectors`, which `scales` must match.
        expected: usize,
        /// Length of the `scales` supplied.
        got: usize,
    },

    /// The two TQ+ calibration arrays disagree in length
    /// (`tqplus_shift.len() != tqplus_scale.len()`).
    TqplusLengthMismatch {
        /// Length of `tqplus_shift`.
        shift_len: usize,
        /// Length of `tqplus_scale`.
        scale_len: usize,
    },

    /// A non-empty TQ+ calibration array has a length that is not `dim`.
    TqplusLengthNotDim {
        /// Length of the offending calibration array.
        got: usize,
        /// The dim it had to equal.
        dim: usize,
    },

    /// A per-vector scale is not finite or is negative. The encoder only
    /// ever emits finite, non-negative scales; an Inf slot would win every
    /// top-1 and a NaN slot would vanish from all results. Mirrors the
    /// loader's value validation, so a `from_parts`-accepted index always
    /// survives its own `write` → `load` round-trip.
    InvalidScaleValue {
        /// Index into `scales` (equivalently, the index slot).
        slot: usize,
        /// The offending value.
        value: f32,
    },

    /// A TQ+ shift coordinate is not finite. Mirrors the loader's value
    /// validation.
    InvalidTqplusShiftValue {
        /// Coordinate index into `tqplus_shift`.
        coord: usize,
        /// The offending value.
        value: f32,
    },

    /// A TQ+ scale coordinate is not finite or is `<= 0`. Search divides
    /// by `tqplus_scale`, so such a value silently turns every query's
    /// scores into NaN/Inf. Mirrors the loader's value validation.
    InvalidTqplusScaleValue {
        /// Coordinate index into `tqplus_scale`.
        coord: usize,
        /// The offending value.
        value: f32,
    },

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
