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
        })
    }

    /// The id → slot map, built from `slot_to_id` on first use after a
    /// load. Loads validated id uniqueness, so the sizes always agree.
    fn ids(&self) -> &HashMap<u64, usize, IdBuildHasher> {
        self.id_to_slot.get_or_init(|| {
            self.slot_to_id
                .iter()
                .enumerate()
                .map(|(slot, &id)| (id, slot))
                .collect()
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
    /// [`AddError::IdAlreadyPresent`](crate::AddError::IdAlreadyPresent),
    /// or any error returned by
    /// [`TurboQuantIndex::add_2d`](crate::TurboQuantIndex::add_2d).
    pub fn add_with_ids_2d(
        &mut self,
        vectors: &[f32],
        dim: usize,
        ids: &[u64],
    ) -> Result<(), AddError> {
        if dim == 0 || vectors.len() % dim != 0 {
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
        // this call.
        let mut seen_this_call: std::collections::HashSet<u64, IdBuildHasher> =
            std::collections::HashSet::with_capacity_and_hasher(n, IdBuildHasher::default());
        for &id in ids {
            if self.ids().contains_key(&id) || !seen_this_call.insert(id) {
                return Err(AddError::IdAlreadyPresent(id));
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

        self.ids_mut().reserve(n);
        self.slot_to_id.reserve(n);
        for (i, &id) in ids.iter().enumerate() {
            self.ids_mut().insert(id, base_slot + i);
        }
        self.slot_to_id.extend_from_slice(ids);

        Ok(())
    }

    /// Remove the vector with the given external id.
    ///
    /// Returns `true` if the id was present and removed, `false`
    /// otherwise. O(1) via the inner [`TurboQuantIndex::swap_remove`].
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(slot) = self.ids_mut().remove(&id) else {
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
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        // Only the allowlist can produce a SearchError, and there is none.
        self.search_with_allowlist(queries, k, None)
            .expect("search_with_allowlist cannot fail without an allowlist")
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
    /// Passing `allowlist = None` is equivalent to [`Self::search`] and
    /// never returns an error.
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

        let res = self
            .inner
            .search_with_mask(queries, k, mask_buf.as_deref());

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

    /// Eagerly populate the inner search caches. See
    /// [`TurboQuantIndex::prepare`].
    pub fn prepare(&self) {
        self.inner.prepare();
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
        })
    }
}
