//! Stable external IDs on top of [`TurboQuantIndex`].
//!
//! [`TurboQuantIndex`] stores vectors positionally: calling `swap_remove`
//! invalidates external references because the previously-last vector
//! moves into the deleted slot. `IdMapIndex` wraps the positional index
//! with a bidirectional `id ↔ slot` mapping so callers can identify
//! vectors by a stable `u64` ID that doesn't change when other vectors
//! are inserted or removed.
//!
//! Roughly analogous to FAISS's `IndexIDMap2` (hash-table backed). The
//! wrapper delegates all vector storage, rotation, scoring and
//! serialization questions to the inner [`TurboQuantIndex`] and only
//! owns the ID table.
//!
//! ```no_run
//! use turbovec::IdMapIndex;
//!
//! let mut index = IdMapIndex::new(1536, 4);
//! let vectors: Vec<f32> = vec![0.0; 1536 * 3];
//! index.add_with_ids(&vectors, &[1001, 1002, 1003]);
//!
//! let queries: Vec<f32> = vec![0.0; 1536];
//! let (scores, ids) = index.search(&queries, 3);
//!
//! index.remove(1002);
//! assert_eq!(index.len(), 2);
//! ```
//!
//! # Complexity
//!
//! - `add_with_ids(n vectors)` — O(n) encode + O(n) HashMap inserts.
//! - `remove(id)` — O(1): one HashMap lookup, one HashMap update for the
//!   vector that moved into the deleted slot, and the inner
//!   [`TurboQuantIndex::swap_remove`].
//! - `search` — same as the inner index, plus an O(nq·k) ID translation
//!   pass over the returned slot indices.

use std::collections::HashMap;

use crate::TurboQuantIndex;

/// ID-addressed wrapper around [`TurboQuantIndex`].
pub struct IdMapIndex {
    inner: TurboQuantIndex,
    /// slot → external id. `slot_to_id[i]` is the id of the vector
    /// currently stored in slot `i` of `inner`.
    slot_to_id: Vec<u64>,
    /// external id → slot. Kept in sync with `slot_to_id`.
    id_to_slot: HashMap<u64, usize>,
}

impl IdMapIndex {
    pub fn new(dim: usize, bit_width: usize) -> Self {
        Self {
            inner: TurboQuantIndex::new(dim, bit_width),
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
        }
    }

    /// Add `n = vectors.len() / dim` vectors with the given external ids.
    ///
    /// Panics if `ids.len() != n`, if any id is already present in the
    /// index, or if `ids` contains duplicates within this call.
    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) {
        let dim = self.inner.dim();
        let n = vectors.len() / dim;
        assert_eq!(
            vectors.len(),
            n * dim,
            "vector buffer length {} not a multiple of dim {}",
            vectors.len(),
            dim,
        );
        assert_eq!(
            ids.len(),
            n,
            "expected {n} ids, got {}",
            ids.len(),
        );

        // Reserve first so that a partial failure is impossible.
        self.id_to_slot.reserve(n);
        self.slot_to_id.reserve(n);

        let base_slot = self.inner.len();
        for (i, &id) in ids.iter().enumerate() {
            let slot = base_slot + i;
            if self.id_to_slot.insert(id, slot).is_some() {
                panic!("id {id} already present in index");
            }
        }
        self.slot_to_id.extend_from_slice(ids);

        self.inner.add(vectors);
    }

    /// Remove the vector with the given external id.
    ///
    /// Returns `true` if the id was present and removed, `false`
    /// otherwise. O(1) via the inner [`TurboQuantIndex::swap_remove`].
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(slot) = self.id_to_slot.remove(&id) else {
            return false;
        };
        let last = self.slot_to_id.len() - 1;

        let moved_from = self.inner.swap_remove(slot);
        debug_assert_eq!(moved_from, last);

        // Mirror the swap-and-pop in our tables.
        if slot != last {
            let moved_id = self.slot_to_id[last];
            self.slot_to_id[slot] = moved_id;
            // The previously-last id now lives at `slot`.
            self.id_to_slot.insert(moved_id, slot);
        }
        self.slot_to_id.pop();

        true
    }

    /// Search for the top-`k` nearest ids for each query.
    ///
    /// Returns `(scores, ids)` flattened row-major: row `qi` occupies
    /// indices `qi * k .. (qi + 1) * k` in both arrays. Number of rows
    /// is `queries.len() / dim`.
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        let res = self.inner.search(queries, k);
        // `res.k` may be smaller than the requested `k` if the index
        // has fewer than `k` vectors.
        let effective_k = res.k;
        let mut ids = Vec::with_capacity(res.indices.len());
        for &slot in &res.indices {
            // Inner returns i64 slot indices. Convert via slot_to_id.
            // Slot indices are always in-bounds (the kernel never
            // returns negative or out-of-range values for a valid
            // index), so this lookup cannot fail in practice; the
            // bounds check makes that invariant crash-loud if it ever
            // does.
            let id = self.slot_to_id[slot as usize];
            ids.push(id);
        }
        let _ = effective_k; // keep `k` in the returned vec length
        (res.scores, ids)
    }

    /// True if the index currently contains a vector with this id.
    pub fn contains(&self, id: u64) -> bool {
        self.id_to_slot.contains_key(&id)
    }

    pub fn len(&self) -> usize {
        self.slot_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slot_to_id.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    pub fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }

    /// Eagerly populate the inner search caches. See
    /// [`TurboQuantIndex::prepare`].
    pub fn prepare(&self) {
        self.inner.prepare();
    }
}
