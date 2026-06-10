//! TurboQuant implementation for vector search.
//!
//! Compresses high-dimensional vectors to 2-4 bits per coordinate with
//! near-optimal distortion. Data-oblivious — no training required.
//!
//! ```no_run
//! use turbovec::TurboQuantIndex;
//!
//! // 1536-dim vectors compressed to 4 bits per coordinate.
//! let mut index = TurboQuantIndex::new(1536, 4).unwrap();
//!
//! // `vectors` is a flat [f32] of length n * dim, `queries` likewise.
//! let vectors: Vec<f32> = vec![0.0; 1536 * 10];
//! let queries: Vec<f32> = vec![0.0; 1536 * 2];
//!
//! index.add(&vectors);
//! let results = index.search(&queries, 10);
//! index.write("index.tv").unwrap();
//! let loaded = TurboQuantIndex::load("index.tv").unwrap();
//! ```
//!
//! # Concurrent search
//!
//! `search` takes `&self` and is safe to call from multiple threads
//! concurrently. Internally the rotation matrix, the Lloyd-Max centroids
//! and the SIMD-blocked code layout are initialised lazily via
//! [`std::sync::OnceLock`], so the first caller pays the one-time
//! initialisation cost and every subsequent caller reads the caches
//! without locking. [`TurboQuantIndex::prepare`] can be called once
//! after `add`/`load` to pay that cost up front.
//!
//! Mutation still flows through `&mut self`: `add` extends the packed
//! codes and invalidates the blocked layout cache by replacing its
//! `OnceLock`. This keeps the invariant that once a cache is populated
//! from `&self`, it matches the current `packed_codes`.

pub mod codebook;
pub mod encode;
pub mod error;
pub mod id_map;
pub mod io;
pub mod pack;
pub mod rotation;
pub mod search;

pub use error::{AddError, ConstructError};
pub use id_map::IdMapIndex;

use std::path::Path;
use std::sync::OnceLock;

const ROTATION_SEED: u64 = 42;
const BLOCK: usize = 32;

/// Upper bound on vector dimensionality. `search`/`prepare` lazily build a
/// `dim`×`dim` f64 rotation matrix, an allocation that scales with `dim²`
/// and is NOT bounded by the size of any loaded file — so an untrusted
/// `.tv`/`.tvim` declaring a huge `dim` could otherwise drive a
/// multi-gigabyte allocation (resource-exhaustion DoS) from a tiny file.
/// 65536 is far above any real embedding dimension (largest in common use
/// is ~4096) and rejects the catastrophic cases. Enforced at construction,
/// first add, and load.
pub const MAX_DIM: usize = 65536;
const FLUSH_EVERY: usize = 256;

/// Maximum permitted coordinate magnitude. Beyond this, f32 sum-of-
/// squares in the norm computation can overflow to +Inf for any
/// reasonable dim (sqrt(f32::MAX / dim) for dim=2^16 is ~7e16; this
/// bound leaves a 7x safety margin and is still ~16 orders of
/// magnitude above any realistic embedding value).
const MAX_INPUT_MAGNITUDE: f32 = 1e16;

/// Reject non-finite (NaN, +Inf, -Inf) or extremely-large input values.
/// Returns the first offending vector/coord/value tuple, or `None` if
/// the input is clean.
///
/// Called from `add` / `add_2d` / `search` / `search_with_mask`. Without
/// this check the encode pipeline silently corrupts the index:
///   - NaN: `0 * NaN = NaN` poisons `vec_scales[slot]`, so the slot
///     exists in `len()` but is never reachable through search.
///   - Inf: same path via `1/Inf = 0`.
///   - Huge magnitude: `simd_norm`'s f32 sum-of-squares overflows to
///     +Inf, `scale[i] = Inf` gets stored, slot incorrectly wins
///     top-k against every query.
pub fn first_invalid_coord(values: &[f32], dim: usize) -> Option<(usize, usize, f32)> {
    for (i, x) in values.iter().enumerate() {
        if !x.is_finite() || x.abs() >= MAX_INPUT_MAGNITUDE {
            let vector_index = if dim == 0 { 0 } else { i / dim };
            let coord_index = if dim == 0 { i } else { i % dim };
            return Some((vector_index, coord_index, *x));
        }
    }
    None
}

/// SIMD-blocked cache derived from `packed_codes`.
///
/// Materialised lazily by [`TurboQuantIndex::search`] on first call
/// and re-materialised when [`TurboQuantIndex::add`] resets the
/// enclosing `OnceLock`.
struct BlockedCache {
    data: Vec<u8>,
    n_blocks: usize,
}

pub struct TurboQuantIndex {
    /// Vector dimensionality. `None` means the index was constructed
    /// without a known dim (lazy mode) and hasn't seen its first add yet.
    /// Once set — either eagerly in [`Self::new`] or implicitly on the
    /// first [`Self::add_2d`] call — it never changes.
    dim: Option<usize>,
    bit_width: usize,
    n_vectors: usize,
    packed_codes: Vec<u8>,
    scales: Vec<f32>,

    /// TQ+ per-coord calibration. Both have length `dim` once the first
    /// add has happened (and the batch had enough samples to fit them);
    /// empty otherwise. Frozen after the first add — subsequent adds
    /// reuse them so all vectors in the index live in the same
    /// calibrated coordinate system. Loaded indexes from pre-TQ+ files
    /// arrive empty and behave as identity calibration (no recall gain,
    /// no behaviour change vs the old encoding).
    tqplus_shift: Vec<f32>,
    tqplus_scale: Vec<f32>,

    // Thread-safe lazy caches. These are initialised from `&self` via
    // `OnceLock::get_or_init`, which allows `search` to take `&self`
    // and run concurrently from multiple threads without external
    // locking. `add` resets `blocked` by replacing its `OnceLock` (it
    // already has `&mut self` for the underlying extend on
    // `packed_codes` and `scales`).
    //
    // `rotation`, `boundaries`, and `centroids` are deterministic functions
    // of `(dim, ROTATION_SEED)` and `(bit_width, dim)`, so they never need
    // to be invalidated.
    rotation: OnceLock<Vec<f32>>,
    boundaries: OnceLock<Vec<f32>>,
    centroids: OnceLock<Vec<f32>>,
    blocked: OnceLock<BlockedCache>,
}

pub struct SearchResults {
    pub scores: Vec<f32>,
    pub indices: Vec<i64>,
    pub nq: usize,
    pub k: usize,
}

impl SearchResults {
    pub fn scores_for_query(&self, qi: usize) -> &[f32] {
        &self.scores[qi * self.k..(qi + 1) * self.k]
    }

    pub fn indices_for_query(&self, qi: usize) -> &[i64] {
        &self.indices[qi * self.k..(qi + 1) * self.k]
    }
}

impl TurboQuantIndex {
    /// Construct an index with a known dimensionality. The dim is locked
    /// at construction; subsequent [`Self::add`] / [`Self::add_2d`] calls
    /// must match.
    ///
    /// Returns [`ConstructError::BitWidthOutOfRange`] if `bit_width` is
    /// not in `{2, 3, 4}` and [`ConstructError::DimNotPositiveMultipleOf8`]
    /// if `dim == 0` or `dim % 8 != 0`.
    pub fn new(dim: usize, bit_width: usize) -> Result<Self, ConstructError> {
        if !(2..=4).contains(&bit_width) {
            return Err(ConstructError::BitWidthOutOfRange(bit_width));
        }
        if dim == 0 || dim % 8 != 0 {
            return Err(ConstructError::DimNotPositiveMultipleOf8(dim));
        }
        if dim > MAX_DIM {
            return Err(ConstructError::DimTooLarge { dim, max: MAX_DIM });
        }

        Ok(Self {
            dim: Some(dim),
            bit_width,
            n_vectors: 0,
            packed_codes: Vec::new(),
            scales: Vec::new(),
            tqplus_shift: Vec::new(),
            tqplus_scale: Vec::new(),
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
        })
    }

    /// Construct an empty index without committing to a dimensionality.
    /// The dim is inferred and locked on the first [`Self::add_2d`] call
    /// (or [`Self::add`] if the caller wires dim in separately).
    ///
    /// Returns [`ConstructError::BitWidthOutOfRange`] if `bit_width` is
    /// not in `{2, 3, 4}`.
    pub fn new_lazy(bit_width: usize) -> Result<Self, ConstructError> {
        if !(2..=4).contains(&bit_width) {
            return Err(ConstructError::BitWidthOutOfRange(bit_width));
        }
        Ok(Self {
            dim: None,
            bit_width,
            n_vectors: 0,
            packed_codes: Vec::new(),
            scales: Vec::new(),
            tqplus_shift: Vec::new(),
            tqplus_scale: Vec::new(),
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
        })
    }

    /// Add a flat batch of vectors. `dim` must be set (either eagerly at
    /// construction or by a prior [`Self::add_2d`] call).
    ///
    /// `vectors.len()` must be a multiple of `dim`; an empty input is a
    /// no-op.
    ///
    /// # Panics
    ///
    /// - If `dim` is not set (call [`Self::new_lazy`] then [`Self::add_2d`]
    ///   instead).
    /// - If `vectors.len()` is not a multiple of `dim`.
    /// - If any coordinate is non-finite (NaN, +Inf, -Inf) or has
    ///   magnitude `>= 1e16`. Callers handling untrusted input should
    ///   prefer [`Self::add_2d`], which returns a typed
    ///   [`AddError::InvalidInputValue`] instead.
    pub fn add(&mut self, vectors: &[f32]) {
        let dim = self.dim.expect(
            "TurboQuantIndex dim is not set; use add_2d(vectors, dim) on the \
             first add or construct via TurboQuantIndex::new(dim, bit_width)",
        );
        let n = vectors.len() / dim;
        assert_eq!(
            vectors.len(),
            n * dim,
            "vectors length must be a multiple of dim"
        );
        // Empty add is a true no-op — return before touching calibration
        // or caches. Previously, an empty first add hit the
        // `n < TQPLUS_MIN_SAMPLES` branch in `encode`, returned identity
        // calibration, and locked `tqplus_shift` to that identity for the
        // lifetime of the index. Every subsequent add — even a million
        // vectors — then saw `Some(identity)` and silently skipped
        // fitting fresh calibration. The user lost TQ+ entirely with no
        // warning.
        if n == 0 {
            return;
        }
        if let Some((vi, ci, v)) = first_invalid_coord(vectors, dim) {
            panic!(
                "invalid input value at vector {vi}, coord {ci}: {v} \
                 (must be finite and |value| < 1e16 to avoid f32 norm overflow)",
            );
        }

        let rotation = self
            .rotation
            .get_or_init(|| rotation::make_rotation_matrix(dim));
        if self.boundaries.get().is_none() || self.centroids.get().is_none() {
            let (boundaries, centroids) = codebook::codebook(self.bit_width, dim);
            let _ = self.boundaries.set(boundaries);
            let _ = self.centroids.set(centroids);
        }
        let boundaries = self
            .boundaries
            .get()
            .expect("boundaries cache is initialized");
        let centroids = self
            .centroids
            .get()
            .expect("centroids cache is initialized");
        // On subsequent adds, reuse the calibration fitted on the first
        // batch so all vectors live in the same calibrated coord system.
        // On the first add, encode() fits a fresh calibration.
        let existing = if self.tqplus_shift.is_empty() {
            None
        } else {
            Some((self.tqplus_shift.as_slice(), self.tqplus_scale.as_slice()))
        };
        let (packed, scales, shift, scale_tq) = encode::encode(
            vectors,
            n,
            dim,
            rotation,
            boundaries,
            centroids,
            self.bit_width,
            existing,
        );

        if self.n_vectors == 0 {
            self.packed_codes = packed;
            self.scales = scales;
            self.tqplus_shift = shift;
            self.tqplus_scale = scale_tq;
        } else {
            self.packed_codes.extend_from_slice(&packed);
            self.scales.extend_from_slice(&scales);
            // tqplus_shift/scale unchanged — locked by the first add.
        }
        self.n_vectors += n;

        // Invalidate the blocked cache — it was derived from the old
        // `packed_codes` and no longer matches the extended vector set.
        // Rotation, boundaries, and centroids remain valid (they only depend
        // on `(dim, ROTATION_SEED)` and `(bit_width, dim)`).
        self.blocked = OnceLock::new();
    }

    /// Add `vectors` of dimension `dim`. On a lazy index this locks the
    /// index dim; on an already-dim'd index `dim` must match the index's
    /// existing dim.
    ///
    /// This is the form that bindings with shape information (e.g. the
    /// Python binding receiving a 2D numpy array) should use, since a
    /// flat `&[f32]` alone is ambiguous about its shape.
    ///
    /// Returns:
    /// - [`AddError::DimMismatch`] if `dim` does not match the
    ///   already-locked dim.
    /// - [`AddError::DimNotMultipleOf8`] when committing a lazy index
    ///   to a dim that is not a multiple of 8.
    /// - [`AddError::InvalidInputValue`] if any coordinate is non-finite
    ///   or has magnitude `>= 1e16`.
    ///
    /// # Panics
    ///
    /// Panics if `vectors.len()` is not a multiple of `dim`. (This
    /// indicates a caller-side bug rather than recoverable bad data, so
    /// it isn't returned as a typed error.)
    pub fn add_2d(&mut self, vectors: &[f32], dim: usize) -> Result<(), AddError> {
        match self.dim {
            Some(existing) if existing != dim => {
                return Err(AddError::DimMismatch { existing, got: dim });
            }
            Some(_) => {}
            None => {
                // `dim == 0` slips past the `% 8` check (0 % 8 == 0) but is a
                // degenerate dim: committing it wedges the lazy index and the
                // first `add` divides by zero (`vectors.len() / dim`). Reject
                // it here, mirroring IdMapIndex::add_with_ids_2d.
                if dim == 0 || dim % 8 != 0 {
                    return Err(AddError::DimNotMultipleOf8(dim));
                }
                if dim > MAX_DIM {
                    return Err(AddError::DimTooLarge { dim, max: MAX_DIM });
                }
                // Don't commit dim until value validation passes — otherwise
                // a lazy index is left with a committed dim and no vectors,
                // which would let a follow-up wrong-dim add see a confusing
                // DimMismatch instead of a fresh start.
            }
        }
        if let Some((vi, ci, v)) = first_invalid_coord(vectors, dim) {
            return Err(AddError::InvalidInputValue {
                vector_index: vi,
                coord_index: ci,
                value: v,
            });
        }
        // Lazy commit happens via add() (which goes through `self.dim.expect`),
        // so re-do the dim assignment here for the lazy-first-add case.
        if self.dim.is_none() {
            self.dim = Some(dim);
        }
        self.add(vectors);
        Ok(())
    }

    /// Run a top-`k` search against the index.
    ///
    /// Takes `&self` and is safe to call concurrently from multiple
    /// threads. The first caller on a fresh index pays the one-time
    /// cache initialisation cost (rotation matrix, Lloyd-Max centroids
    /// and the SIMD-blocked code layout). Subsequent callers read the
    /// caches without locking.
    ///
    /// Call [`TurboQuantIndex::prepare`] once after `add`/`load` to
    /// pay that cost up front if you want deterministic first-query
    /// latency.
    ///
    /// # Panics
    ///
    /// Panics if `queries.len()` is not a multiple of `dim`, or if any
    /// query coordinate is non-finite (NaN, +Inf, -Inf) or has
    /// magnitude `>= 1e16`. Validate untrusted input at the caller
    /// (e.g. the Python binding raises `ValueError`).
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        self.search_with_mask(queries, k, None)
    }

    /// Run a top-`k` search restricted to slots whose `mask` entry is `true`.
    ///
    /// `mask`, when `Some`, must have length equal to [`Self::len`]. Only
    /// slots with `mask[i] == true` contribute to the returned top-`k`. The
    /// effective result count per query is `min(k, n_allowed)` where
    /// `n_allowed` is the number of `true` entries in `mask`.
    ///
    /// Passing `mask = None` is equivalent to [`Self::search`].
    ///
    /// # Panics
    ///
    /// - If `mask.len() != self.len()` (when `mask` is `Some`).
    /// - If `queries.len()` is not a multiple of `dim`.
    /// - If any query coordinate is non-finite or has magnitude `>= 1e16`.
    pub fn search_with_mask(
        &self,
        queries: &[f32],
        k: usize,
        mask: Option<&[bool]>,
    ) -> SearchResults {
        // A lazy index that's never seen an add returns an empty result
        // shaped according to the caller's query count (best effort: we
        // don't know dim, so nq is 0). Matches Python users' expectation
        // that `search` on an empty store is a no-op rather than an error.
        let Some(dim) = self.dim else {
            return SearchResults {
                scores: Vec::new(),
                indices: Vec::new(),
                nq: 0,
                k: 0,
            };
        };
        let nq = queries.len() / dim;
        assert_eq!(queries.len(), nq * dim);
        // Reject non-finite / huge-magnitude queries. Same rationale as
        // `add`: NaN / Inf / overflow-magnitude values poison the SIMD
        // scoring kernel and produce arbitrary indices with NaN scores,
        // silently rather than as a typed error.
        if let Some((vi, ci, v)) = first_invalid_coord(queries, dim) {
            panic!(
                "invalid query value at query {vi}, coord {ci}: {v} \
                 (must be finite and |value| < 1e16 to avoid f32 overflow)",
            );
        }

        let rotation = self
            .rotation
            .get_or_init(|| rotation::make_rotation_matrix(dim));
        let centroids = self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bit_width, dim);
            c
        });
        let blocked = self.blocked.get_or_init(|| {
            let (data, n_blocks) =
                pack::repack(&self.packed_codes, self.n_vectors, self.bit_width, dim);
            BlockedCache { data, n_blocks }
        });

        let packed_mask = mask.map(|m| {
            assert_eq!(
                m.len(),
                self.n_vectors,
                "mask length {} does not match index size {}",
                m.len(),
                self.n_vectors,
            );
            let n_words = (self.n_vectors + 63) / 64;
            let mut buf = vec![0u64; n_words];
            for (i, &b) in m.iter().enumerate() {
                if b {
                    buf[i >> 6] |= 1u64 << (i & 63);
                }
            }
            buf
        });

        let n_allowed = packed_mask.as_ref().map_or(self.n_vectors, |p| {
            p.iter().map(|w| w.count_ones() as usize).sum::<usize>()
        });
        let effective_k = k.min(self.n_vectors).min(n_allowed);

        let (scores, indices) = search::search(
            queries,
            nq,
            rotation,
            &blocked.data,
            centroids,
            &self.scales,
            &self.tqplus_shift,
            &self.tqplus_scale,
            self.bit_width,
            dim,
            self.n_vectors,
            blocked.n_blocks,
            k,
            packed_mask.as_deref(),
        );

        SearchResults {
            scores,
            indices,
            nq,
            k: effective_k,
        }
    }

    /// Eagerly populate the search caches (rotation matrix, centroids
    /// and SIMD-blocked code layout).
    ///
    /// Calling `prepare` is optional — `search` will materialise the
    /// caches on its first call if needed. Use it to move the one-time
    /// cost out of the first query path, for example right after
    /// [`TurboQuantIndex::load`] or after a batch of [`add`] calls.
    ///
    /// Safe to call multiple times and from multiple threads.
    pub fn prepare(&self) {
        // On a lazy index that's seen no add, there's nothing to prepare
        // — dim is unknown and the caches depend on it.
        let Some(dim) = self.dim else { return };
        self.rotation
            .get_or_init(|| rotation::make_rotation_matrix(dim));
        self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bit_width, dim);
            c
        });
        self.blocked.get_or_init(|| {
            let (data, n_blocks) =
                pack::repack(&self.packed_codes, self.n_vectors, self.bit_width, dim);
            BlockedCache { data, n_blocks }
        });
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        // Sentinel: dim=0 in the file header means "lazy index, dim never
        // committed". The loader interprets dim=0 + n_vectors=0 as a
        // freshly-constructed lazy state. dim=0 is otherwise meaningless
        // (the constructor asserts dim % 8 == 0 with dim >= 8), so this
        // doesn't collide with any valid eager index.
        io::write(
            path,
            self.bit_width,
            self.dim.unwrap_or(0),
            self.n_vectors,
            &self.packed_codes,
            &self.scales,
            &self.tqplus_shift,
            &self.tqplus_scale,
        )
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale) =
            io::load(path)?;
        let dim_opt = if dim == 0 { None } else { Some(dim) };
        Ok(Self::from_parts(
            dim_opt,
            bit_width,
            n_vectors,
            packed_codes,
            scales,
            tqplus_shift,
            tqplus_scale,
        ))
    }

    pub(crate) fn from_parts(
        dim: Option<usize>,
        bit_width: usize,
        n_vectors: usize,
        packed_codes: Vec<u8>,
        scales: Vec<f32>,
        tqplus_shift: Vec<f32>,
        tqplus_scale: Vec<f32>,
    ) -> Self {
        // Structural invariants every caller must uphold. `from_parts` is
        // pub(crate); today the only callers are `io::load` and
        // `id_map::load`, both of which validate at the read layer — but
        // pinning the invariants here makes future callers (and refactors
        // of the existing ones) safe by construction.
        assert_eq!(
            tqplus_shift.len(),
            tqplus_scale.len(),
            "from_parts: tqplus_shift.len()={} != tqplus_scale.len()={}",
            tqplus_shift.len(),
            tqplus_scale.len(),
        );
        match dim {
            Some(d) => {
                let expected_packed = n_vectors * d * bit_width / 8;
                assert_eq!(
                    packed_codes.len(),
                    expected_packed,
                    "from_parts: packed_codes.len()={} != n_vectors({}) * dim({}) * bit_width({}) / 8 = {}",
                    packed_codes.len(),
                    n_vectors,
                    d,
                    bit_width,
                    expected_packed,
                );
                assert_eq!(
                    scales.len(),
                    n_vectors,
                    "from_parts: scales.len()={} != n_vectors={}",
                    scales.len(),
                    n_vectors,
                );
                if !tqplus_shift.is_empty() {
                    assert_eq!(
                        tqplus_shift.len(),
                        d,
                        "from_parts: non-empty TQ+ length {} must equal dim {}",
                        tqplus_shift.len(),
                        d,
                    );
                }
            }
            None => {
                // Lazy uncommitted state — every storage field must be empty.
                assert_eq!(n_vectors, 0, "from_parts: lazy index must have n_vectors=0");
                assert!(
                    packed_codes.is_empty(),
                    "from_parts: lazy index must have empty packed_codes",
                );
                assert!(scales.is_empty(), "from_parts: lazy index must have empty scales");
                assert!(
                    tqplus_shift.is_empty(),
                    "from_parts: lazy index must have empty tqplus_shift",
                );
            }
        }

        // v2 files (pre-TQ+) load with empty TQ+ vectors and a positive
        // n_vectors. If we leave `tqplus_shift` empty, the next `add()`
        // would see `existing = None` (the lazy-first-add signal),
        // call `encode()` with `existing = None`, get a fresh fitted
        // calibration back — and then silently drop it because
        // `n_vectors != 0` takes the else branch that only extends
        // `packed_codes` / `scales`. The new vectors would be encoded
        // with that fitted calibration but searched against identity,
        // producing silently-wrong scores.
        //
        // Populate explicit identity here so the "is the calibration
        // committed?" check always agrees with the actual state of the
        // stored vectors.
        let (tqplus_shift, tqplus_scale) = if tqplus_shift.is_empty() && n_vectors > 0 {
            let d = dim.expect(
                "from_parts: n_vectors > 0 implies a committed dim — \
                 mismatch indicates a corrupted side-car or a misuse",
            );
            (vec![0.0; d], vec![1.0; d])
        } else {
            (tqplus_shift, tqplus_scale)
        };
        Self {
            dim,
            bit_width,
            n_vectors,
            packed_codes,
            scales,
            tqplus_shift,
            tqplus_scale,
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
        }
    }

    pub(crate) fn packed_codes(&self) -> &[u8] {
        &self.packed_codes
    }

    pub(crate) fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub(crate) fn tqplus_shift(&self) -> &[f32] {
        &self.tqplus_shift
    }

    pub(crate) fn tqplus_scale(&self) -> &[f32] {
        &self.tqplus_scale
    }

    /// Remove the vector at `idx` in O(1) by swapping with the last vector.
    ///
    /// Semantics match [`Vec::swap_remove`]: the last vector is moved into
    /// the deleted slot, so **order is not preserved** and the index of the
    /// previously-last vector changes. Any external references to the moved
    /// vector's old index must be updated. For stable external IDs, wrap in
    /// an ID-map layer.
    ///
    /// Returns the old index of the moved vector (`n_vectors - 1` before
    /// the call); equals `idx` when `idx` was already the last element.
    /// Panics if `idx >= n_vectors`.
    pub fn swap_remove(&mut self, idx: usize) -> usize {
        assert!(
            idx < self.n_vectors,
            "index {idx} out of bounds (n_vectors = {})",
            self.n_vectors
        );

        // n_vectors > 0 (asserted above) implies a successful add, which
        // implies self.dim was committed at that point. Unwrap is safe.
        let dim = self.dim.expect("n_vectors > 0 but dim is None");
        let bytes_per_vec = dim * self.bit_width / 8;
        let last = self.n_vectors - 1;

        if idx != last {
            // Move last vector's packed bytes into slot `idx`.
            let src = last * bytes_per_vec;
            let dst = idx * bytes_per_vec;
            self.packed_codes.copy_within(src..src + bytes_per_vec, dst);

            // Move last norm into slot `idx`.
            self.scales[idx] = self.scales[last];
        }

        // Truncate both arrays.
        self.packed_codes.truncate(last * bytes_per_vec);
        self.scales.truncate(last);
        self.n_vectors -= 1;

        // Invalidate the blocked cache since it was derived from the old layout.
        self.blocked = OnceLock::new();

        last
    }

    pub fn len(&self) -> usize {
        self.n_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Vector dimensionality, or `0` if this index was constructed lazily
    /// and hasn't seen an add yet. `0` is a safe sentinel because the
    /// eager constructor asserts `dim >= 8` (multiple of 8). Use
    /// [`Self::dim_opt`] when you need to distinguish "not set" from a
    /// (nonsensical) zero.
    pub fn dim(&self) -> usize {
        self.dim.unwrap_or(0)
    }

    /// Vector dimensionality as an [`Option`], where `None` means the
    /// index is lazy and hasn't been committed to a dim yet.
    pub fn dim_opt(&self) -> Option<usize> {
        self.dim
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }
}

#[cfg(test)]
mod from_parts_tests {
    //! Unit tests for `TurboQuantIndex::from_parts` length-invariant
    //! checks. `from_parts` is `pub(crate)`, so these live inside the
    //! crate; the assertions catch any future caller (or refactor of
    //! the existing `io::load` callers) that hands in a malformed
    //! tuple of fields.

    use super::TurboQuantIndex;

    #[test]
    #[should_panic(expected = "packed_codes.len()")]
    fn from_parts_panics_on_packed_codes_length_mismatch() {
        // Expected packed_codes length for dim=64, bit_width=4, n=2 is
        // 2 * 64 * 4 / 8 = 64 bytes. Pass 32 to trigger the assert.
        let _ = TurboQuantIndex::from_parts(
            Some(64),
            4,
            2,
            vec![0u8; 32],
            vec![1.0f32; 2],
            Vec::new(),
            Vec::new(),
        );
    }

    #[test]
    #[should_panic(expected = "scales.len()")]
    fn from_parts_panics_on_scales_length_mismatch() {
        let _ = TurboQuantIndex::from_parts(
            Some(64),
            4,
            2,
            vec![0u8; 64],
            vec![1.0f32; 5],  // n_vectors says 2; scales has 5
            Vec::new(),
            Vec::new(),
        );
    }

    #[test]
    #[should_panic(expected = "tqplus_shift.len()")]
    fn from_parts_panics_on_mismatched_tqplus_lengths() {
        let _ = TurboQuantIndex::from_parts(
            Some(64),
            4,
            2,
            vec![0u8; 64],
            vec![1.0f32; 2],
            vec![0.0f32; 64],   // length 64
            vec![1.0f32; 32],   // length 32 — mismatch
        );
    }

    #[test]
    #[should_panic(expected = "non-empty TQ+ length")]
    fn from_parts_panics_when_tqplus_length_does_not_equal_dim() {
        let _ = TurboQuantIndex::from_parts(
            Some(64),
            4,
            2,
            vec![0u8; 64],
            vec![1.0f32; 2],
            vec![0.0f32; 48],   // length 48 != dim 64
            vec![1.0f32; 48],
        );
    }

    #[test]
    #[should_panic(expected = "lazy index must have n_vectors=0")]
    fn from_parts_panics_on_lazy_with_nonzero_n_vectors() {
        let _ = TurboQuantIndex::from_parts(
            None,
            4,
            5,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
    }

    #[test]
    fn from_parts_accepts_lazy_uncommitted() {
        // Lazy + everything empty + n_vectors=0 is the canonical lazy
        // state the constructor must accept.
        let idx = TurboQuantIndex::from_parts(
            None,
            4,
            0,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(idx.dim_opt(), None);
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn from_parts_accepts_eager_with_consistent_lengths() {
        // dim=64, bit_width=4, n=2 → packed=64 bytes, scales=2.
        // Empty TQ+ vectors are valid input (v2-loaded shape); the
        // identity-population logic fills them in below.
        let idx = TurboQuantIndex::from_parts(
            Some(64),
            4,
            2,
            vec![0u8; 64],
            vec![1.0f32; 2],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(idx.dim(), 64);
        assert_eq!(idx.len(), 2);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod x86_scalar_fallback_tests {
    //! Verify the x86 scalar fallback (score_query_into_heap, taken on
    //! pre-AVX2 CPUs) returns the SAME top-k as the SIMD kernels on this
    //! host. score_query_into_heap is not compiled on aarch64, so this is
    //! the only place its full scoring path — including the issue-#106
    //! perm0 de-interleave — runs end to end.
    use super::TurboQuantIndex;
    use crate::search::FORCE_SCALAR_FALLBACK;
    use std::sync::atomic::Ordering;

    fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut out = vec![0.0f32; n * dim];
        for row in out.chunks_mut(dim) {
            let mut norm = 0.0f64;
            for x in row.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
                *x = v as f32;
                norm += v * v;
            }
            let inv = 1.0 / (norm.sqrt() + 1e-9);
            for x in row.iter_mut() {
                *x = (*x as f64 * inv) as f32;
            }
        }
        out
    }

    fn topk_sets(indices: &[i64], nq: usize, k: usize) -> Vec<std::collections::BTreeSet<i64>> {
        (0..nq)
            .map(|q| indices[q * k..(q + 1) * k].iter().copied().collect())
            .collect()
    }

    #[test]
    fn scalar_fallback_matches_simd_topk() {
        let dim = 64;
        let n = 600;
        let nq = 12;
        let k = 16;
        for &bits in &[2usize, 3, 4] {
            let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
            idx.add(&unit_vectors(n, dim, 11));
            let queries = unit_vectors(nq, dim, 22);

            FORCE_SCALAR_FALLBACK.store(false, Ordering::Relaxed);
            let simd = idx.search(&queries, k);
            FORCE_SCALAR_FALLBACK.store(true, Ordering::Relaxed);
            let scalar = idx.search(&queries, k);
            FORCE_SCALAR_FALLBACK.store(false, Ordering::Relaxed);

            assert_eq!(simd.k, scalar.k, "bits={bits}: differing result width");
            // Compare per-query top-k as sets (tie order between kernels may
            // differ; membership must not).
            assert_eq!(
                topk_sets(&simd.indices, nq, simd.k),
                topk_sets(&scalar.indices, nq, scalar.k),
                "bits={bits}: scalar fallback returned a different top-k than SIMD",
            );
        }
    }
}
