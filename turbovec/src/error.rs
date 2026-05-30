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
#[derive(Debug, Clone, PartialEq)]
pub enum AddError {
    /// Batch dim does not match the index's already-locked dim.
    DimMismatch { existing: usize, got: usize },

    /// First-add dim on a lazy index must be a multiple of 8.
    DimNotMultipleOf8(usize),

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructError {
    /// `bit_width` must be 2, 3, or 4.
    BitWidthOutOfRange(usize),

    /// `dim` must be a positive multiple of 8.
    DimNotPositiveMultipleOf8(usize),
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
        }
    }
}

impl Error for ConstructError {}
