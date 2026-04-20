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
pub mod id_map;
pub mod io;
pub mod pack;
pub mod rotation;
pub mod search;

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
    dim: usize,
    bit_width: usize,
    n_vectors: usize,
    packed_codes: Vec<u8>,
    norms: Vec<f32>,

    // Thread-safe lazy caches. These are initialised from `&self` via
    // `OnceLock::get_or_init`, which allows `search` to take `&self`
    // and run concurrently from multiple threads without external
    // locking. `add` resets `blocked` by replacing its `OnceLock` (it
    // already has `&mut self` for the underlying extend on
    // `packed_codes` and `norms`).
    //
    // `rotation` and `centroids` are deterministic functions of `(dim,
    // ROTATION_SEED)` and `(bit_width, dim)` respectively, so they
    // never need to be invalidated.
    rotation: OnceLock<Vec<f32>>,
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
    pub fn new(dim: usize, bit_width: usize) -> Self {
        assert!((2..=4).contains(&bit_width), "bit_width must be 2, 3, or 4");
        assert!(dim % 8 == 0, "dim must be a multiple of 8");

        Self {
            dim,
            bit_width,
            n_vectors: 0,
            packed_codes: Vec::new(),
            norms: Vec::new(),
            rotation: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
        }
    }

    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(
            vectors.len(),
            n * self.dim,
            "vectors length must be a multiple of dim"
        );

        let rotation = self
            .rotation
            .get_or_init(|| rotation::make_rotation_matrix(self.dim));
        let (boundaries, _) = codebook::codebook(self.bit_width, self.dim);
        let (packed, norms) =
            encode::encode(vectors, n, self.dim, rotation, &boundaries, self.bit_width);

        if self.n_vectors == 0 {
            self.packed_codes = packed;
            self.norms = norms;
        } else {
            self.packed_codes.extend_from_slice(&packed);
            self.norms.extend_from_slice(&norms);
        }
        self.n_vectors += n;

        // Invalidate the blocked cache — it was derived from the old
        // `packed_codes` and no longer matches the extended vector set.
        // Rotation and centroids remain valid (they only depend on
        // `(dim, ROTATION_SEED)` and `(bit_width, dim)`).
        self.blocked = OnceLock::new();
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
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);

        let rotation = self
            .rotation
            .get_or_init(|| rotation::make_rotation_matrix(self.dim));
        let centroids = self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bit_width, self.dim);
            c
        });
        let blocked = self.blocked.get_or_init(|| {
            let (data, n_blocks) =
                pack::repack(&self.packed_codes, self.n_vectors, self.bit_width, self.dim);
            BlockedCache { data, n_blocks }
        });

        let k = k.min(self.n_vectors);

        let (scores, indices) = search::search(
            queries,
            nq,
            rotation,
            &blocked.data,
            centroids,
            &self.norms,
            self.bit_width,
            self.dim,
            self.n_vectors,
            blocked.n_blocks,
            k,
        );

        SearchResults {
            scores,
            indices,
            nq,
            k,
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
        self.rotation
            .get_or_init(|| rotation::make_rotation_matrix(self.dim));
        self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bit_width, self.dim);
            c
        });
        self.blocked.get_or_init(|| {
            let (data, n_blocks) =
                pack::repack(&self.packed_codes, self.n_vectors, self.bit_width, self.dim);
            BlockedCache { data, n_blocks }
        });
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        io::write(
            path,
            self.bit_width,
            self.dim,
            self.n_vectors,
            &self.packed_codes,
            &self.norms,
        )
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, packed_codes, norms) = io::load(path)?;
        Ok(Self {
            dim,
            bit_width,
            n_vectors,
            packed_codes,
            norms,
            rotation: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
        })
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

        let bytes_per_vec = self.dim * self.bit_width / 8;
        let last = self.n_vectors - 1;

        if idx != last {
            // Move last vector's packed bytes into slot `idx`.
            let src = last * bytes_per_vec;
            let dst = idx * bytes_per_vec;
            self.packed_codes.copy_within(src..src + bytes_per_vec, dst);

            // Move last norm into slot `idx`.
            self.norms[idx] = self.norms[last];
        }

        // Truncate both arrays.
        self.packed_codes.truncate(last * bytes_per_vec);
        self.norms.truncate(last);
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

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }
}
