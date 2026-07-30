//! Stable external IDs on top of [`TurboQuantIndex`].
//!
//! [`TurboQuantIndex`] stores vectors positionally: calling `swap_remove`
//! invalidates external references because the previously-last vector
//! moves into the deleted slot. `IdMapIndex` wraps the positional index
//! with a bidirectional `id ↔ slot` mapping so callers can identify
//! vectors by a stable `u64` ID that doesn't change when other vectors
//! are inserted or removed.
//!
//! A bidirectional hash-table-backed `u64 ↔ slot` mapping layered over
//! the inner [`TurboQuantIndex`]. The wrapper delegates all vector
//! storage, rotation, scoring and serialization questions to the inner
//! index and only owns the ID table.
//!
//! ```no_run
//! use turbovec::IdMapIndex;
//!
//! let mut index = IdMapIndex::new(1536, 4).unwrap();
//! let vectors: Vec<f32> = vec![0.0; 1536 * 3];
//! index.add_with_ids(&vectors, &[1001, 1002, 1003]).unwrap();
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
use std::hash::{BuildHasherDefault, Hasher};

/// Multiply-shift hasher for the external-id maps. Ids are caller-chosen
/// u64s, not attacker-controlled protocol input, so SipHash's HashDoS
/// resistance buys nothing here while costing a measurable slice of the
/// O(1) remove path.
///
/// The multiply alone is not enough. hashbrown derives the bucket index
/// from the **low** bits of the hash, and multiplication only propagates
/// entropy upward: the low `t` bits of `id * K` depend solely on the low
/// `t` bits of `id`. So any id scheme whose low bits are constant —
/// `shard << 32 | seq` composite ids with `seq` starting at zero being
/// the obvious benign one — lands every key in the same bucket region
/// and the map degrades to linear probing over the whole table.
///
/// The finalizing xor-shift folds the high half back down, so the bucket
/// index sees the entropy the multiply pushed up. Measured on 100k ids
/// of the form `i << 32`: lookup went from 476 ms to 0.2 ms, i.e. from
/// quadratic to flat, at no cost on sequential ids.
///
/// Note the "not attacker controlled" premise is still an application
/// assumption: the hash remains trivially invertible, so a service that
/// lets untrusted callers choose ids can still craft collisions.
#[derive(Default)]
pub(crate) struct IdHasher(u64);

/// Fibonacci multiply plus an xor-shift finalizer (splitmix-style):
/// mixes the input into both halves of the hash, then folds the high
/// half into the low one so hashbrown's bucket index is well-distributed
/// even for inputs whose low bits are constant.
#[inline]
fn mix(x: u64) -> u64 {
    let z = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^ (z >> 32)
}

impl Hasher for IdHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Only u64 keys are ever hashed by the id maps; this fallback
        // keeps the impl total for completeness.
        for &b in bytes {
            self.0 = mix(self.0 ^ b as u64);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = mix(i);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// `BuildHasher` for [`IdHasher`]-keyed maps.
pub(crate) type IdBuildHasher = BuildHasherDefault<IdHasher>;
use std::path::Path;

use crate::io;
use crate::{AddError, ConstructError, SearchError, TurboQuantIndex};

// Comparisons made against the load-time sorted table, counted per
// thread so a test can assert the search is logarithmic. Test-only: the
// hook below compiles to nothing otherwise.
#[cfg(test)]
thread_local! {
    static TABLE_PROBES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Record one comparison against the sorted table. A no-op outside the
/// crate's own tests.
#[inline(always)]
fn record_table_probe() {
    #[cfg(test)]
    {
        let _ = TABLE_PROBES.try_with(|p| p.set(p.get() + 1));
    }
}

/// Is `id` in the load-time sorted table?
///
/// `binary_search_by` rather than `binary_search` purely so the
/// comparator is ours and can be counted — the algorithm, and therefore
/// the cost, is std's binary search either way, and outside `cfg(test)`
/// the hook is empty so this is what `binary_search` already compiled to.
///
/// The counting is not decoration. The O(n)-per-add defect this module
/// guards (#383) can live on the read side just as easily as the write
/// side: swapping this for a linear `sorted.contains(&id)` leaves the
/// table unrewritten and the deferred set exact, so the structural
/// assertions in `deferred_adds_below_the_table_do_not_scale_with_n`
/// pass while every add has silently become O(n). The probe count is
/// what closes that hole, and unlike a wall-clock ratio it cannot be
/// moved by a change to the constant term (#409, #420).
fn table_contains(sorted: &[u64], id: u64) -> bool {
    sorted
        .binary_search_by(|probe| {
            record_table_probe();
            probe.cmp(&id)
        })
        .is_ok()
}

/// ID-addressed wrapper around [`TurboQuantIndex`].
#[derive(Debug)]
pub struct IdMapIndex {
    inner: TurboQuantIndex,
    /// slot → external id. `slot_to_id[i]` is the id of the vector
    /// currently stored in slot `i` of `inner`.
    slot_to_id: Vec<u64>,
    /// external id → slot. Kept in sync with `slot_to_id`. Built lazily
    /// after a load (cold start — load + search — never consults it;
    /// duplicate-id validation happens at load via a sort instead), and
    /// eagerly on every other path.
    id_to_slot: std::sync::OnceLock<HashMap<u64, usize, IdBuildHasher>>,
    /// Sorted copy of the load-time id table, kept ONLY while
    /// `id_to_slot` is unset: the load already sorts the ids to validate
    /// uniqueness, and keeping the result lets post-load adds validate
    /// new ids by binary search instead of forcing the O(n) map build
    /// into the add path. Freed the moment the map materializes (in
    /// [`Self::ids`]), and ignored thereafter.
    ///
    /// Never mutated after the load — ids added inside the deferred
    /// window go to `deferred_added` instead, so an add's cost is
    /// independent of `n` no matter where the new ids sort (#383).
    ///
    /// Note this is NOT freed by a load+search-only workload: plain
    /// `search` and `search_with_allowlist(None)` never consult the map,
    /// so such an index carries the table (8 bytes/vector) for its
    /// lifetime, where main dropped it at the end of the load. The
    /// trade is deliberate — it buys post-load adds an O(log n) presence
    /// check instead of forcing the O(n) map build — but a read-only
    /// serving deployment pays for something it never uses.
    ///
    /// `Mutex` purely for that free: materialization happens behind
    /// `&self`. Mutating callers hold `&mut self` and use `get_mut`, so
    /// they never lock; the one `lock` is the map build itself.
    sorted_ids: std::sync::Mutex<Vec<u64>>,
    /// Ids added while the deferred window is open, i.e. the ids present
    /// in the index but not in `sorted_ids`. Together the two cover
    /// exactly the live id set, so a presence check is one binary search
    /// plus one hash lookup and an add costs O(rows added), never O(n).
    /// Freed alongside `sorted_ids` when the map materializes; `Mutex`
    /// for the same reason.
    ///
    /// Carries the same not-freed caveat as `sorted_ids`, and more
    /// sharply: only a map-materializing call (`remove`, `contains`, an
    /// allowlist search) ends the window, so a load → adds →
    /// plain-`search` workload never ends it and this set grows by one
    /// entry (plus hashbrown's load-factor slack) per added id for the
    /// index's lifetime. It is bounded by the adds actually made, not by
    /// `n`, and the same ids are already retained in `slot_to_id`, so the
    /// overhead is a constant factor on the added rows rather than
    /// unbounded growth against a fixed workload — but a long-lived
    /// writer that never removes or checks membership does pay it.
    deferred_added: std::sync::Mutex<std::collections::HashSet<u64, IdBuildHasher>>,
}

impl IdMapIndex {
    /// Construct an id-map index with a known dim. The dim is locked at
    /// construction. Propagates the same errors as
    /// [`TurboQuantIndex::new`].
    pub fn new(dim: usize, bit_width: usize) -> Result<Self, ConstructError> {
        Ok(Self {
            inner: TurboQuantIndex::new(dim, bit_width)?,
            slot_to_id: Vec::new(),
            id_to_slot: std::sync::OnceLock::from(HashMap::default()),
            sorted_ids: std::sync::Mutex::new(Vec::new()),
            deferred_added: std::sync::Mutex::new(Default::default()),
        })
    }

    /// Construct an empty id-map index without committing to a dim. The
    /// dim is inferred and locked on the first [`Self::add_with_ids_2d`]
    /// call. Propagates the same errors as [`TurboQuantIndex::new_lazy`].
    pub fn new_lazy(bit_width: usize) -> Result<Self, ConstructError> {
        Ok(Self {
            inner: TurboQuantIndex::new_lazy(bit_width)?,
            slot_to_id: Vec::new(),
            id_to_slot: std::sync::OnceLock::from(HashMap::default()),
            sorted_ids: std::sync::Mutex::new(Vec::new()),
            deferred_added: std::sync::Mutex::new(Default::default()),
        })
    }

    /// The id → slot map, built from `slot_to_id` on first use after a
    /// load. Loads validated id uniqueness, so the sizes always agree.
    fn ids(&self) -> &HashMap<u64, usize, IdBuildHasher> {
        self.id_to_slot.get_or_init(|| {
            let map = self
                .slot_to_id
                .iter()
                .enumerate()
                .map(|(slot, &id)| (id, slot))
                .collect();
            // The map is authoritative from here on; release the
            // load-time sorted copy (8 bytes/vector) and the deferred-add
            // set rather than carry them for the index's lifetime.
            *self.sorted_ids.lock().expect("sorted_ids lock poisoned") = Vec::new();
            *self
                .deferred_added
                .lock()
                .expect("deferred_added lock poisoned") = Default::default();
            map
        })
    }

    fn ids_mut(&mut self) -> &mut HashMap<u64, usize, IdBuildHasher> {
        self.ids();
        self.id_to_slot.get_mut().expect("ids just materialized")
    }

    /// Add `n = vectors.len() / dim` vectors with the given external ids.
    /// Requires the inner index's dim to already be set (eager constructor
    /// or a previous lazy add).
    ///
    /// Returns the same errors as
    /// [`Self::add_with_ids_2d`]. Panics only if the inner index is still
    /// in lazy/uninitialized state — that signals API misuse (use
    /// `add_with_ids_2d` on a lazy index), not bad input.
    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), AddError> {
        let dim = self.inner.dim_opt().expect(
            "IdMapIndex dim is not set; use add_with_ids_2d(vectors, dim, ids) \
             on the first add or construct with IdMapIndex::new(dim, bit_width)",
        );
        self.add_with_ids_2d(vectors, dim, ids)
    }

    /// Add `vectors` of dimensionality `dim` with the given external ids.
    /// On a lazy index this locks the dim; on an already-dim'd index
    /// `dim` must match.
    ///
    /// This is the form bindings with shape information (e.g. the Python
    /// binding receiving a 2D ndarray) should use, since a flat
    /// `&[f32]` alone is ambiguous about shape.
    ///
    /// Returns
    /// [`AddError::VectorBufferNotMultipleOfDim`](crate::AddError::VectorBufferNotMultipleOfDim),
    /// [`AddError::IdsCountMismatch`](crate::AddError::IdsCountMismatch),
    /// [`AddError::ZeroDim`](crate::AddError::ZeroDim),
    /// [`AddError::IdAlreadyPresent`](crate::AddError::IdAlreadyPresent),
    /// [`AddError::DuplicateIdInBatch`](crate::AddError::DuplicateIdInBatch),
    /// or any error returned by
    /// [`TurboQuantIndex::add_2d`](crate::TurboQuantIndex::add_2d).
    pub fn add_with_ids_2d(
        &mut self,
        vectors: &[f32],
        dim: usize,
        ids: &[u64],
    ) -> Result<(), AddError> {
        if dim == 0 {
            return Err(AddError::ZeroDim);
        }
        if vectors.len() % dim != 0 {
            return Err(AddError::VectorBufferNotMultipleOfDim {
                vectors_len: vectors.len(),
                dim,
            });
        }
        let n = vectors.len() / dim;
        if ids.len() != n {
            return Err(AddError::IdsCountMismatch {
                expected: n,
                got: ids.len(),
            });
        }

        // Validate all ids up-front so a partial failure is impossible.
        // Reject both ids already in the index and duplicates within
        // this call. In the post-load window (map unset) presence checks
        // run against the retained load-time table, so an add never
        // forces the O(n) map build; the map stays lazy for
        // remove/contains to build if ever needed.
        let deferred = self.id_to_slot.get().is_none();
        let mut seen_this_call: std::collections::HashSet<u64, IdBuildHasher> =
            std::collections::HashSet::with_capacity_and_hasher(n, IdBuildHasher::default());
        // Split by window: in the deferred (post-load, map-unset) window
        // presence is a binary search over the retained load-time table
        // plus a hash lookup in the set of ids added since, so an add
        // never forces the O(n) map build. Both windows keep main's typed
        // distinction between an id already in the index and a duplicate
        // within this batch.
        if deferred {
            let sorted = self.sorted_ids.get_mut().expect("sorted_ids lock poisoned");
            let added = self
                .deferred_added
                .get_mut()
                .expect("deferred_added lock poisoned");
            for &id in ids {
                if table_contains(sorted, id) || added.contains(&id) {
                    return Err(AddError::IdAlreadyPresent(id));
                }
                if !seen_this_call.insert(id) {
                    return Err(AddError::DuplicateIdInBatch(id));
                }
            }
        } else {
            for &id in ids {
                if self.ids().contains_key(&id) {
                    return Err(AddError::IdAlreadyPresent(id));
                }
                if !seen_this_call.insert(id) {
                    return Err(AddError::DuplicateIdInBatch(id));
                }
            }
        }

        // Capture the slot the first new vector will occupy BEFORE we
        // touch the inner index, then run the inner add first. If `add_2d`
        // returns Err (e.g. DimMismatch on a committed-dim index) the ID
        // tables stay untouched — otherwise we'd leave `n` ghost entries
        // pointing at slots that don't exist in the inner index, and the
        // next search_with_allowlist / remove would corrupt further.
        let base_slot = self.inner.len();
        self.inner.add_2d(vectors, dim)?;

        if deferred {
            // Record the new ids for the next add's presence check; the
            // map itself stays unset (a later lazy build reads the
            // extended slot_to_id and includes these rows). They go in a
            // side set rather than into `sorted_ids`: merging them into
            // the sorted table costs O(n) per add whenever a new id sorts
            // below the table's tail, which is quadratic over a chatty
            // post-load pattern (#383). Cost here is O(rows added).
            let added = self
                .deferred_added
                .get_mut()
                .expect("deferred_added lock poisoned");
            added.reserve(n);
            added.extend(ids.iter().copied());
        } else {
            self.ids_mut().reserve(n);
            for (i, &id) in ids.iter().enumerate() {
                self.ids_mut().insert(id, base_slot + i);
            }
        }
        self.slot_to_id.reserve(n);
        self.slot_to_id.extend_from_slice(ids);

        Ok(())
    }

    /// Remove the vector with the given external id.
    ///
    /// Returns `true` if the id was present and removed, `false`
    /// otherwise. O(1) via the inner [`TurboQuantIndex::swap_remove`].
    pub fn remove(&mut self, id: u64) -> bool {
        // Look the slot up without mutating, then run the inner removal
        // first and update the tables only after it returns — "index
        // first, then the maps", the order the Python stores' delete
        // paths use.
        //
        // Ordering hardening, not a live bug fix: `inner.swap_remove` has
        // no unwind reachable *from here* today. Its one documented panic
        // is the `idx < n_vectors` assert, and `slot` comes from the id
        // table, so it is in bounds by construction. Past that assert it
        // calls `packed_mut()` only inside `if self.packed_codes.get()
        // .is_some()`, so the lazy O(n·dim) rebuild is never triggered
        // from a remove — that `get_or_init` is always a hit — and the
        // rest is in-bounds indexing and allocation-free lane ops (no
        // rayon, no allocation). What the order buys is
        // that a future fallible inner removal (an incrementally
        // materializing `packed_mut`, say) cannot corrupt the tables:
        // mutating them first would leave `id_to_slot` short while
        // `slot_to_id` stayed one longer than `inner.len()`, so the
        // vector would be searchable but unresolvable and every later
        // `remove` would compute `last` off the wrong length (#380).
        //
        // This orders the two halves; it does not make the operation
        // atomic. `inner.swap_remove` is itself multi-step, so an unwind
        // partway through it would leave the inner index short against
        // full tables — the same desync with the opposite polarity, which
        // no ordering here can prevent.
        let Some(&slot) = self.ids().get(&id) else {
            return false;
        };
        let last = self.slot_to_id.len() - 1;

        let moved_from = self.inner.swap_remove(slot);
        debug_assert_eq!(moved_from, last);

        self.ids_mut().remove(&id);

        // Mirror the swap-and-pop in our tables.
        if slot != last {
            let moved_id = self.slot_to_id[last];
            self.slot_to_id[slot] = moved_id;
            // The previously-last id now lives at `slot`.
            self.ids_mut().insert(moved_id, slot);
        }
        self.slot_to_id.pop();

        true
    }

    /// Search for the top-`k` nearest ids for each query.
    ///
    /// The effective result count per query is `min(k, self.len())` —
    /// `k` is clamped when the index holds fewer than `k` vectors.
    ///
    /// Returns `(scores, ids)` flattened row-major: row `qi` occupies
    /// indices `qi * effective_k .. (qi + 1) * effective_k` in both
    /// arrays, where `effective_k = min(k, self.len())`. Number of rows
    /// is `nq = queries.len() / dim`, so callers can recover the stride
    /// as `scores.len() / nq` when `nq > 0` (a lazy-uninitialized index
    /// has no committed `dim` and returns empty results).
    /// # Panics
    ///
    /// - If `queries.len()` is not a multiple of `dim`.
    /// - If any query coordinate is non-finite or has magnitude `>= 1e16`.
    ///
    /// This is the panicking form, matching
    /// [`TurboQuantIndex::search`](crate::TurboQuantIndex::search). Use
    /// [`Self::search_with_allowlist`] with `allowlist = None` for the
    /// same search as a `Result`. Neither condition can fire on an index
    /// with no committed `dim` — that case returns the empty result
    /// before any validation.
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        // Passing `None` rules out the two allowlist variants, but not
        // the query-shape ones, which `search_with_allowlist` now
        // returns rather than letting escape as a panic from inside
        // (#412). So this cannot be an `.expect` on "cannot fail": it
        // re-panics with the error's `Display`, exactly as
        // `TurboQuantIndex::search_with_mask` does, which keeps the
        // payload the descriptive message it has always been instead of
        // burying it behind a `Debug` rendering.
        self.search_with_allowlist(queries, k, None)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Search restricted to the given `allowlist` of external ids.
    ///
    /// `allowlist`, when `Some`, restricts the returned top-`k` to ids in the
    /// allowlist. The allowlist is deduplicated: the effective result count
    /// per query is `min(k, number of unique ids in allowlist)`, so repeated
    /// ids don't widen the result.
    ///
    /// Returns [`SearchError::AllowlistEmpty`] if `allowlist` is `Some`
    /// and empty, or [`SearchError::UnknownId`] if it contains an id not
    /// currently present in the index. Duplicate ids in the allowlist are
    /// accepted and deduplicated.
    ///
    /// The query-shape conditions are reported the same way: a `queries`
    /// length that is not a whole multiple of the index dim yields
    /// [`SearchError::QueryBufferNotMultipleOfDim`], and a non-finite or
    /// out-of-range coordinate yields [`SearchError::InvalidQueryValue`].
    /// Both used to escape as panics from the inner index even though
    /// this method already declared them in its error type (#412), so a
    /// service that matched on `SearchError` and mapped it to a 400 lost
    /// the request thread to a ragged body anyway. Every condition this
    /// method can detect now arrives as `Err`.
    ///
    /// Passing `allowlist = None` searches the whole index — the same
    /// search [`Self::search`] performs, differing only in that
    /// `search` re-panics on the query-shape errors.
    pub fn search_with_allowlist(
        &self,
        queries: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> Result<(Vec<f32>, Vec<u64>), SearchError> {
        let mask_buf: Option<Vec<bool>> = match allowlist {
            Some(ids) => {
                if ids.is_empty() {
                    return Err(SearchError::AllowlistEmpty);
                }
                let mut mask = vec![false; self.inner.len()];
                for &id in ids {
                    let slot = *self.ids().get(&id).ok_or(SearchError::UnknownId(id))?;
                    mask[slot] = true;
                }
                Some(mask)
            }
            None => None,
        };

        // The checked form: the panicking `search_with_mask` would raise
        // the query-shape conditions as panics out of a method whose
        // signature already promises them as `SearchError` (#412).
        // `MaskLengthMismatch` cannot fire here — the mask is built
        // above at exactly `self.inner.len()` — but it is propagated
        // rather than special-cased so this stays a single exit.
        let res = self
            .inner
            .try_search_with_mask(queries, k, mask_buf.as_deref())?;

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
        Ok((res.scores, ids))
    }

    /// True if the index currently contains a vector with this id.
    pub fn contains(&self, id: u64) -> bool {
        self.ids().contains_key(&id)
    }

    pub fn len(&self) -> usize {
        self.slot_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slot_to_id.is_empty()
    }

    /// Vector dimensionality, or `0` for a lazy index that hasn't seen an
    /// add yet.
    ///
    /// **Deprecated — prefer [`Self::dim_opt`].** See
    /// [`TurboQuantIndex::dim`] for why the `0` is a footgun (#318).
    #[deprecated(
        since = "0.10.0",
        note = "returns 0 for a lazy index, which is unsafe to do arithmetic with; use dim_opt()"
    )]
    pub fn dim(&self) -> usize {
        self.inner.dim_opt().unwrap_or(0)
    }

    /// Vector dimensionality as an [`Option`], where `None` means the
    /// index is lazy and uncommitted.
    pub fn dim_opt(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    pub fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }

    /// Eagerly populate the inner search caches **and** the lazy
    /// id → slot map. See [`TurboQuantIndex::prepare`].
    ///
    /// Forwarding to the inner index alone left `id_to_slot` unbuilt, so
    /// the first `search_with_allowlist`, `contains` or `remove` after a
    /// load still paid the O(n) map build that `prepare` promises to
    /// absorb (#348). Materializing here also frees the load-time
    /// `sorted_ids`/`deferred_added` side-tables (see [`Self::ids`]),
    /// which is the same steady state a first allowlist search would
    /// have reached.
    ///
    /// Idempotent and O(1) once warm: [`Self::slots_ready`] and
    /// [`Self::packed_ready`] both only go false → true.
    pub fn prepare(&self) {
        self.inner.prepare();
        self.ids();
    }

    /// TQ+ calibration state of the inner index. See
    /// [`TurboQuantIndex::calibration_state`] and
    /// [`CalibrationState`](crate::CalibrationState).
    pub fn calibration_state(&self) -> crate::CalibrationState {
        self.inner.calibration_state()
    }

    /// See [`TurboQuantIndex::packed_ready`].
    pub fn packed_ready(&self) -> bool {
        self.inner.packed_ready()
    }

    /// True when the lazy id → slot map is already materialized. A v6 load
    /// leaves it empty (see [`Self::ids`]), so the first `remove` after a
    /// load pays an O(n) map build; callers that must not stall on that
    /// (the Python binding, which would hold the GIL — issue #319) probe
    /// this first. Like [`Self::packed_ready`] it only goes false → true.
    pub fn slots_ready(&self) -> bool {
        self.id_to_slot.get().is_some()
    }

    /// Serialize to a `.tvim` file — the inner quantized index plus the
    /// id-map side-tables. Round-trips exactly through [`Self::load`].
    ///
    /// Saving while [`Self::calibration_state`] is still
    /// [`WarmingUp`](crate::CalibrationState::WarmingUp) commits the
    /// **reloaded** index to
    /// [`Identity`](crate::CalibrationState::Identity) calibration for
    /// good, so it never fits a TQ+ calibration however many vectors are
    /// added later — see [`TurboQuantIndex::write`] for the full
    /// statement. Applies equally to
    /// [`Self::write_with_durability`], [`Self::write_to_writer`] and
    /// [`Self::to_bytes`].
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.write_with_durability(path, io::Durability::Durable)
    }

    /// [`Self::write`] with an explicit [`io::Durability`] level (see
    /// [`TurboQuantIndex::write_with_durability`]).
    pub fn write_with_durability(
        &self,
        path: impl AsRef<Path>,
        durability: io::Durability,
    ) -> std::io::Result<()> {
        // Mirror TurboQuantIndex::write: dim=0 means lazy-uninitialized.
        let (boundaries, centroids) = self.inner.codebook_for_write();
        // Warm blocked cache: borrow it (fused per-chunk deinterleave in
        // the x86 writer threads; direct borrow elsewhere) — see
        // TurboQuantIndex::write_with_durability.
        if let Some(native) = self.inner.blocked_native_for_write() {
            #[cfg(target_arch = "x86_64")]
            return io::write_id_map_native_with_durability(
                path,
                self.inner.bit_width(),
                self.inner.dim_opt().unwrap_or(0),
                self.inner.len(),
                native,
                &boundaries,
                &centroids,
                self.inner.scales(),
                self.inner.tqplus_shift(),
                self.inner.tqplus_scale(),
                &self.slot_to_id,
                durability,
            );
            #[cfg(not(target_arch = "x86_64"))]
            return io::write_id_map_with_durability(
                path,
                self.inner.bit_width(),
                self.inner.dim_opt().unwrap_or(0),
                self.inner.len(),
                native,
                &boundaries,
                &centroids,
                self.inner.scales(),
                self.inner.tqplus_shift(),
                self.inner.tqplus_scale(),
                &self.slot_to_id,
                durability,
            );
        }
        io::write_id_map_with_durability(
            path,
            self.inner.bit_width(),
            self.inner.dim_opt().unwrap_or(0),
            self.inner.len(),
            &self.inner.codes_blocked_seq(),
            &boundaries,
            &centroids,
            self.inner.scales(),
            self.inner.tqplus_shift(),
            self.inner.tqplus_scale(),
            &self.slot_to_id,
            durability,
        )
    }

    /// Load a `.tvim` file previously written by [`Self::write`].
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Self::from_loaded(io::load_id_map(path)?)
    }

    /// Serialize the index in the `.tvim` byte format to any
    /// [`std::io::Write`] sink. Emits exactly the bytes [`Self::write`]
    /// would put in the file.
    ///
    /// Unlike [`Self::write`] there is no atomic-replace behaviour: the
    /// caller owns the sink.
    pub fn write_to_writer<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        let (boundaries, centroids) = self.inner.codebook_for_write();
        io::write_id_map_to(
            w,
            self.inner.bit_width(),
            self.inner.dim_opt().unwrap_or(0),
            self.inner.len(),
            &self.inner.codes_blocked_seq(),
            &boundaries,
            &centroids,
            self.inner.scales(),
            self.inner.tqplus_shift(),
            self.inner.tqplus_scale(),
            &self.slot_to_id,
        )
    }

    /// Serialize the index to `.tvim`-format bytes in memory —
    /// byte-identical to the file [`Self::write`] produces. Pairs with
    /// [`Self::from_bytes`] for callers that persist the index through
    /// their own storage (a database column, a cache, a pickle payload)
    /// instead of the filesystem.
    ///
    /// Serializing an index that is still
    /// [`WarmingUp`](crate::CalibrationState::WarmingUp) commits the
    /// deserialized copy to
    /// [`Identity`](crate::CalibrationState::Identity) calibration for
    /// good — see [`Self::write`]. This is the path a clone-by-round-trip
    /// takes, so a copy of a sub-1000-vector index is weaker than the
    /// original, which keeps its warm-up buffer.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to_writer(&mut buf)
            .expect("writing to a Vec<u8> cannot fail");
        buf
    }

    /// Deserialize an index from any [`std::io::Read`] source of
    /// `.tvim`-format bytes. Applies exactly the same validation as
    /// [`Self::load`] — version handling (v5 only), structural and
    /// value-level checks, and the duplicate-id table check — so a byte
    /// stream and the file it came from load, or fail, identically.
    pub fn load_from_reader<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Self::from_loaded(io::load_id_map_from(r)?)
    }

    /// Deserialize an index from in-memory `.tvim`-format bytes, as
    /// produced by [`Self::to_bytes`] (or read out of a `.tvim` file).
    /// Same validation as [`Self::load`]; see
    /// [`Self::load_from_reader`].
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        Self::load_from_reader(&mut &bytes[..])
    }

    /// Shared tail of [`Self::load`] / [`Self::load_from_reader`]:
    /// assemble the wrapper from an io-layer payload.
    #[allow(clippy::type_complexity)]
    fn from_loaded(
        parts: (usize, usize, usize, io::CodePayload, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u64>),
    ) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale, slot_to_id) =
            parts;
        let inner = TurboQuantIndex::from_loaded((
            bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale,
        ))?;
        // Reject corrupt payloads where the id table contains duplicates —
        // this would desync the two tables. Validated with a sort (cheap,
        // cache-friendly) so the id → slot map itself can build lazily:
        // the cold-start path (load + search) never consults it.
        let mut sorted = slot_to_id.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "duplicate ids in .tvim file",
            ));
        }
        Ok(Self {
            inner,
            slot_to_id,
            id_to_slot: std::sync::OnceLock::new(),
            sorted_ids: std::sync::Mutex::new(sorted),
            deferred_added: std::sync::Mutex::new(Default::default()),
        })
    }
}

/// The property [`IdHasher`]'s finalizer exists for (#311), asserted
/// directly instead of through the id maps' observable behaviour.
///
/// The pre-fix hasher was a bare Fibonacci multiply. It returns *correct*
/// results for every id layout — the bug was quadratic probing cost, not
/// wrong answers — so the id-layout round-trip tests in
/// `tests/id_map.rs` pass on it unchanged. What it gets wrong is the
/// distribution hashbrown actually consults: the bucket index comes from
/// the **low** bits of the hash, and multiplication only carries entropy
/// upward, so `id * K` for ids whose low bits are constant lands every
/// key in one bucket.
///
/// Asserting on bucket occupancy rather than wall-clock time keeps this
/// deterministic — a timing-ratio test for the same property would be at
/// the mercy of CI load — while still failing loudly on the exact defect:
/// the reverted hasher puts all 100k `i << 32` ids in bucket 0 at every
/// table size.
#[cfg(test)]
mod hasher_distribution {
    use super::{IdBuildHasher, IdHasher};
    use std::hash::{BuildHasher, Hasher};

    fn hash(id: u64) -> u64 {
        let mut h: IdHasher = IdBuildHasher::default().build_hasher();
        h.write_u64(id);
        h.finish()
    }

    /// Largest number of ids sharing one bucket, for a table of
    /// `1 << bits` buckets indexed the way hashbrown indexes: the low
    /// bits of the hash.
    fn max_bucket_load(ids: impl Iterator<Item = u64>, bits: u32) -> usize {
        let mut buckets = vec![0usize; 1 << bits];
        for id in ids {
            buckets[(hash(id) as usize) & ((1 << bits) - 1)] += 1;
        }
        buckets.into_iter().max().unwrap_or(0)
    }

    #[test]
    fn composite_ids_spread_across_buckets_at_every_table_size() {
        // `shard << 32 | seq` composite ids with seq starting at zero —
        // the benign real-world layout that degraded the map. Also the
        // pure low-bit-constant layouts either side of it.
        // Shifts up to 32 are what the one-round finalizer repairs:
        // `i << s` zeroes the product's low `s` bits, and `z ^ (z >> 32)`
        // folds bits 32.. back down over them.
        //
        // NOTE the fold only reaches so far. For **any** shift `s > 32`
        // the product's bits 32..s are zero as well, so the folded-in
        // half contributes nothing to the low `s - 32` hash bits — which
        // are exactly the low bucket-index bits — and `i << s` ids
        // cluster again, worse as `s` grows: s = 36 degenerates on tables
        // of up to 2^(s-32) buckets, s >= 40 on every table size tried
        // here. That is a residual limitation of the finalizer across the
        // whole `s > 32` range, not a quirk of one shift width, and it is
        // filed rather than silently pinned by widening this loop.
        for shift in [8u32, 16, 32] {
            let n = 8192usize;
            // A perfect hash would give n / 2^bits per bucket; allow 8x
            // that (plus slack for small tables) before calling it
            // clustered. The defect overshoots by orders of magnitude:
            // it puts all n in one bucket.
            for bits in [4u32, 8, 10, 13] {
                let ideal = n as f64 / f64::from(1u32 << bits);
                let limit = (ideal * 8.0).ceil() as usize + 8;
                let load = max_bucket_load((0..n as u64).map(|i| i << shift), bits);
                assert!(
                    load <= limit,
                    "ids of the form i << {shift} cluster: {load} of {n} share one \
                     bucket of {} (limit {limit}) — the hash's low bits, which are \
                     the bucket index, carry no entropy",
                    1u32 << bits,
                );
            }
        }
    }

    #[test]
    fn every_bucket_of_a_small_table_is_reachable_from_composite_ids() {
        // The sharpest form of the same property, and the one the
        // pre-fix hasher fails hardest: with 4096 ids of the form
        // `i << 32` over a 256-bucket table, every bucket should be hit.
        // The bare multiply hits exactly one.
        let mut seen = vec![false; 256];
        for i in 0..4096u64 {
            seen[(hash(i << 32) as usize) & 255] = true;
        }
        let reached = seen.iter().filter(|s| **s).count();
        assert_eq!(
            reached, 256,
            "only {reached}/256 buckets reachable from `i << 32` ids",
        );
    }

    #[test]
    fn sequential_ids_stay_well_distributed() {
        // The control: plain sequential ids were never the problem, and
        // the finalizer must not have made them worse.
        for bits in [4u32, 8, 10, 13] {
            let n = 8192usize;
            let ideal = n as f64 / f64::from(1u32 << bits);
            let limit = (ideal * 8.0).ceil() as usize + 8;
            let load = max_bucket_load(0..n as u64, bits);
            assert!(load <= limit, "sequential ids cluster: {load} in one bucket");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loaded_index() -> IdMapIndex {
        let dim = 64usize;
        let mut src = IdMapIndex::new(dim, 4).unwrap();
        let vectors: Vec<f32> = (0..100 * dim).map(|i| (i % 97) as f32 / 97.0).collect();
        let ids: Vec<u64> = (0..100u64).map(|i| 1000 + i * 7).collect();
        src.add_with_ids(&vectors, &ids).unwrap();
        let mut bytes = Vec::new();
        src.write_to_writer(&mut bytes).unwrap();
        IdMapIndex::from_bytes(&bytes).unwrap()
    }

    fn sorted_len(ix: &IdMapIndex) -> usize {
        ix.sorted_ids.lock().expect("sorted_ids lock").len()
    }

    /// The load-time sorted table is released as soon as the id → slot
    /// map materializes — including via a read-only path (`contains`),
    /// so a load+search-only index never carries both.
    #[test]
    fn sorted_ids_freed_when_map_materializes() {
        let dim = 64usize;
        let mut ix = loaded_index();
        assert_eq!(sorted_len(&ix), 100, "load should keep the sorted table");
        let more: Vec<f32> = (0..dim).map(|i| (i % 31) as f32 / 31.0).collect();
        ix.add_with_ids(&more, &[7]).unwrap();
        assert!(ix.contains(1000), "sanity: id present");
        assert_eq!(
            sorted_len(&ix),
            0,
            "materializing the map must release the sorted table"
        );
        assert!(
            ix.deferred_added.lock().expect("lock").is_empty(),
            "materializing the map must release the deferred-add set"
        );
        assert!(ix.contains(7), "sanity: deferred-window id present");
    }

    /// Deferred-window adds stay outside the load-time sorted table, and
    /// a duplicate is caught whichever of the two halves holds it.
    #[test]
    fn deferred_adds_reject_duplicates_without_touching_the_loaded_table() {
        let dim = 64usize;
        let mut ix = loaded_index();
        let more: Vec<f32> = (0..10 * dim).map(|i| (i % 31) as f32 / 31.0).collect();
        // Interleaves with the loaded ids (which step by 7 from 1000).
        let new_ids: Vec<u64> = vec![1, 1003, 1500, 999_999, 2, 1004, 1600, 3, 4, 5];
        ix.add_with_ids(&more, &new_ids).unwrap();
        assert_eq!(
            sorted_len(&ix),
            100,
            "the load-time table must not be rewritten by an add"
        );
        {
            let s = ix.sorted_ids.lock().expect("lock");
            assert!(s.windows(2).all(|w| w[0] <= w[1]), "table left unsorted");
        }
        // A duplicate from the loaded set and one from the deferred set
        // are both caught, without building the map.
        for dup in [1000u64, 1500] {
            let err = ix.add_with_ids(&more[..dim], &[dup]).unwrap_err();
            assert!(matches!(err, AddError::IdAlreadyPresent(d) if d == dup));
        }
        assert!(
            ix.id_to_slot.get().is_none(),
            "adds must not force the map build"
        );
        // The two halves together are exactly the live id set.
        let mut covered: Vec<u64> = ix.sorted_ids.lock().expect("lock").clone();
        covered.extend(ix.deferred_added.lock().expect("lock").iter().copied());
        covered.sort_unstable();
        let mut live = ix.slot_to_id.clone();
        live.sort_unstable();
        assert_eq!(covered, live);
    }

    /// #383: a deferred-window add whose id sorts BELOW the loaded table
    /// must not cost O(n). The pre-fix merge rewrote the whole table on
    /// every such add, so per-add time tracked `n` exactly.
    ///
    /// Pinned structurally — the load-time table is untouched and each add
    /// lands in the deferred set — rather than as a wall-clock ratio. The
    /// ratio form was vacuous by arithmetic rather than by property: it
    /// divided a per-add time whose n-dependent part is only ~2 ps per
    /// vector (~400 ns at n = 200k) by a constant term that used to be
    /// ~3000 ns, so it read as 1.1x and passed for reasons that had
    /// nothing to do with the table. Removing that constant elsewhere in
    /// the crate (#409) left the same ~400 ns slope sitting on a ~350 ns
    /// base, and the untouched gate started failing at 4.5x on CI while
    /// the code under test had got several times *faster* at every size.
    /// A ratio cannot survive its own denominator moving; these assertions
    /// are machine-independent and fail in microseconds.
    ///
    /// The defect this guards is exact: merging a below-the-table id into
    /// `sorted_ids` grows and rewrites it. Comparing the table against a
    /// snapshot catches that on the first add.
    #[test]
    fn deferred_adds_below_the_table_do_not_scale_with_n() {
        const DIM: usize = 8;
        const ADDS: usize = 100;
        const BASE: u64 = 10_000_000;

        // `ADDS` single-row adds with ids that sort strictly below `BASE`,
        // onto a loaded index of `n` ids. Returns the table before and
        // after, plus the deferred set's size.
        fn add_below_table(n: usize) -> (Vec<u64>, Vec<u64>, usize, bool, u64) {
            let mut src = IdMapIndex::new(DIM, 4).unwrap();
            let vectors: Vec<f32> = (0..n * DIM).map(|i| (i % 251) as f32 / 251.0).collect();
            let ids: Vec<u64> = (0..n as u64).map(|i| BASE + i).collect();
            src.add_with_ids(&vectors, &ids).unwrap();
            let mut ix = IdMapIndex::from_bytes(&src.to_bytes()).unwrap();
            let before = ix.sorted_ids.lock().expect("lock").clone();
            let row = vec![0.25f32; DIM];
            TABLE_PROBES.with(|p| p.set(0));
            for i in 0..ADDS as u64 {
                ix.add_with_ids(&row, &[BASE - 1 - i]).unwrap();
            }
            let probes = TABLE_PROBES.with(|p| p.get());
            let after = ix.sorted_ids.lock().expect("lock").clone();
            let deferred = ix.deferred_added.lock().expect("lock").len();
            (before, after, deferred, ix.id_to_slot.get().is_none(), probes)
        }

        // Two sizes an order of magnitude apart: the work done per add is
        // identical, which is the property the timing ratio was proxying.
        for n in [2_000usize, 20_000] {
            let (before, after, deferred, still_deferred, probes) = add_below_table(n);
            assert!(still_deferred, "adds must stay deferred (n={n})");

            // Read side: the presence check must be a binary search. A
            // linear scan leaves every structural assertion below intact
            // while making each add O(n), so the probe count is the only
            // thing standing between that regression and a green suite.
            // `ADDS` probes is the floor (one comparison per add); the
            // ceiling is a generous logarithmic bound.
            let max_probes = ADDS as u64 * (usize::BITS - n.leading_zeros() + 2) as u64;
            assert!(
                probes >= ADDS as u64 && probes <= max_probes,
                "presence check made {probes} comparisons for {ADDS} adds against a \
                 {n}-id table; a binary search makes at most {max_probes} — a linear \
                 scan here is O(n) per add even with the table left untouched"
            );
            assert_eq!(
                before.len(),
                n,
                "sanity: the load-time table holds every loaded id (n={n})"
            );
            assert_eq!(
                after, before,
                "the load-time sorted table was rewritten by a below-the-table add \
                 (n={n}): it went from {} to {} entries — the O(n) per-add merge is back",
                before.len(),
                after.len(),
            );
            assert_eq!(
                deferred, ADDS,
                "the deferred set must grow by exactly the rows added (n={n})"
            );
        }
    }
}
