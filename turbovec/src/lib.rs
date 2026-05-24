//! TurboQuant implementation for vector search.
//!
//! Compresses high-dimensional vectors to 2-4 bits per coordinate with
//! near-optimal distortion. Data-oblivious — no training required.
//!
//! ```no_run
//! use turbovec::TurboQuantIndex;
//!
//! // 1536-dim vectors compressed to 4 bits per coordinate.
//! let mut index = TurboQuantIndex::new(1536, 4);
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

pub use error::AddError;
pub use id_map::IdMapIndex;

use std::path::Path;
use std::sync::OnceLock;

const ROTATION_SEED: u64 = 42;
const BLOCK: usize = 32;
const FLUSH_EVERY: usize = 256;

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
    pub fn new(dim: usize, bit_width: usize) -> Self {
        assert!((2..=4).contains(&bit_width), "bit_width must be 2, 3, or 4");
        assert!(dim % 8 == 0, "dim must be a multiple of 8");

        Self {
            dim: Some(dim),
            bit_width,
            n_vectors: 0,
            packed_codes: Vec::new(),
            scales: Vec::new(),
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
        }
    }

    /// Construct an empty index without committing to a dimensionality.
    /// The dim is inferred and locked on the first [`Self::add_2d`] call
    /// (or [`Self::add`] if the caller wires dim in separately).
    pub fn new_lazy(bit_width: usize) -> Self {
        assert!((2..=4).contains(&bit_width), "bit_width must be 2, 3, or 4");
        Self {
            dim: None,
            bit_width,
            n_vectors: 0,
            packed_codes: Vec::new(),
            scales: Vec::new(),
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
        }
    }

    /// Add a flat batch of vectors. `dim` must be set (either eagerly at
    /// construction or by a prior [`Self::add_2d`] call). Panics otherwise.
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
        let (packed, scales) = encode::encode(
            vectors,
            n,
            dim,
            rotation,
            boundaries,
            centroids,
            self.bit_width,
        );

        if self.n_vectors == 0 {
            self.packed_codes = packed;
            self.scales = scales;
        } else {
            self.packed_codes.extend_from_slice(&packed);
            self.scales.extend_from_slice(&scales);
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
    /// Returns [`AddError::DimMismatch`] if `dim` does not match the
    /// already-locked dim, and [`AddError::DimNotMultipleOf8`] when
    /// committing a lazy index to a dim that is not a multiple of 8.
    pub fn add_2d(&mut self, vectors: &[f32], dim: usize) -> Result<(), AddError> {
        match self.dim {
            Some(existing) if existing != dim => {
                return Err(AddError::DimMismatch { existing, got: dim });
            }
            Some(_) => {}
            None => {
                if dim % 8 != 0 {
                    return Err(AddError::DimNotMultipleOf8(dim));
                }
                self.dim = Some(dim);
            }
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
        )
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, packed_codes, scales) = io::load(path)?;
        let dim_opt = if dim == 0 { None } else { Some(dim) };
        Ok(Self::from_parts(dim_opt, bit_width, n_vectors, packed_codes, scales))
    }

    pub(crate) fn from_parts(
        dim: Option<usize>,
        bit_width: usize,
        n_vectors: usize,
        packed_codes: Vec<u8>,
        scales: Vec<f32>,
    ) -> Self {
        Self {
            dim,
            bit_width,
            n_vectors,
            packed_codes,
            scales,
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
