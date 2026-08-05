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
//! concurrently. Internally the rotation, the Lloyd-Max centroids
//! and the SIMD-blocked code layout are initialised lazily via
//! [`std::sync::OnceLock`], so the first caller pays the one-time
//! initialisation cost and every subsequent caller reads the caches
//! without locking. [`TurboQuantIndex::prepare`] can be called once
//! after `add`/`load` to pay that cost up front.
//!
//! Mutation still flows through `&mut self`, and the invariant it keeps
//! is stated in terms of what a reader can observe rather than in terms
//! of what any one mutator does: **whenever the index is reachable
//! through `&self`, every populated cache describes exactly the
//! `len()` rows the index currently holds.**
//!
//! That holds by construction. The rotation, boundaries and centroids
//! are pure functions of `dim` and `bit_width`, neither of which ever
//! changes after the first add, so they can never go stale. The blocked
//! layout and the packed bit-plane rows are two encodings of the same
//! rows, each derivable from the other; a mutation holds `&mut self` for
//! its whole duration, so no concurrent reader exists while one of them
//! is being brought up to date, and by the time that borrow ends both
//! the row count and every populated cache describe the same rows.
//!
//! Which of the two encodings a mutation updates is an implementation
//! detail that has changed more than once and is deliberately not
//! promised here. A [`TurboQuantIndex::load`]ed index may hold only the
//! blocked form until something needs the packed rows
//! ([`TurboQuantIndex::packed_ready`] reports which); elsewhere the
//! packed rows lead. Both give bit-identical search results.

// turbovec is 64-bit by design: the SIMD kernels, the `usize` size/offset
// arithmetic in `encode`/`pack`/`search`, and all benchmarks assume a 64-bit
// pointer width. On a 32-bit (or 16-bit) target those size computations could
// overflow `usize` and index out of bounds. Refuse to compile there rather
// than ship a silently-unsafe build — supporting 32-bit/wasm would require a
// dedicated checked-arithmetic pass first.
#[cfg(not(target_pointer_width = "64"))]
compile_error!("turbovec requires a 64-bit target (target_pointer_width = \"64\")");

pub mod codebook;
pub mod encode;
pub mod error;
pub mod id_map;
pub mod io;
mod io_v7;
pub mod pack;
pub mod rotation;
pub mod search;
pub mod warning;

// Kernel-level correctness tests that exercise the crate-internal leaves
// (`codebook`, `encode`, `pack`). These moved in-crate when those functions
// became `pub(crate)` (they trust caller invariants and are no longer part
// of the public surface); the coverage is unchanged.
#[cfg(test)]
mod kernel_tests;

pub use error::{AddError, CalibrateError, ConstructError, FromPartsError, SearchError};
pub use id_map::{IdMapIndex, IdSearchResults};
pub use warning::{set_warning_hook, WarningHook};

use std::path::Path;
use std::sync::OnceLock;

const BLOCK: usize = 32;

/// Upper bound on vector dimensionality. The block-Hadamard rotation and
/// the search-side query buffers scale linearly with `dim`, but a loaded
/// `.tv`/`.tvim` header declaring a huge `dim` still drives allocations
/// (codebook, blocked layout, per-query rotate scratch) that are NOT
/// bounded by the file's own size — so an untrusted tiny file could
/// otherwise request multi-gigabyte buffers (resource-exhaustion DoS).
/// 16384 leaves >4x headroom over the largest embedding dimensions in
/// common use (~4096; rare research models reach 8k-12k). Enforced
/// identically at construction, first add, and load, so any index this
/// build can create it can also load back.
pub const MAX_DIM: usize = 16384;
const FLUSH_EVERY: usize = 256;

/// Maximum permitted coordinate magnitude. Beyond this, f32 sum-of-
/// squares in the norm computation can overflow to +Inf for any
/// reasonable dim (sqrt(f32::MAX / dim) for dim=2^16 is ~7e16; this
/// bound leaves a 7x safety margin and is still ~16 orders of
/// magnitude above any realistic embedding value).
const MAX_INPUT_MAGNITUDE: f32 = 1e16;

// See [`TurboQuantIndex::force_repack_panic`]. Thread-local; see
// FORCE_ENCODE_PANIC for why these cannot be process-globals (#373).
#[cfg(test)]
thread_local! {
    static FORCE_REPACK_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// See [`TurboQuantIndex::force_encode_panic`].
//
// Thread-local, not a global: this switch *panics*, so a stray set would
// take down whichever test happened to reach `encode` next. `cargo test`
// runs the unit binary's tests in parallel threads, and the arming test
// does full input validation plus `packed()` before the check, leaving a
// wide window for another test to consume a global flag (#373). The
// check runs on the calling thread inside `catch_unwind`, before
// `encode` fans out to rayon, so thread-local scoping is sufficient.
// (`search::FORCE_SCALAR_FALLBACK` can be global because taking the
// scalar path still produces correct results; this one cannot.)
#[cfg(test)]
thread_local! {
    static FORCE_ENCODE_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// See [`TurboQuantIndex::force_fit_panic`]. Thread-local for exactly the
// reason [`FORCE_ENCODE_PANIC`] is — it is checked on the calling thread,
// before `fit_calibration` fans out to rayon (#373).
#[cfg(test)]
thread_local! {
    static FORCE_FIT_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// See `TurboQuantIndex::force_swap_remove_panic`. Thread-local for
// exactly the reason `FORCE_ENCODE_PANIC` is (#373). Plain comments, not
// doc comments: `///` does not attach to a `thread_local!` invocation —
// rustdoc generates nothing for macro invocations, so the text would
// render nowhere. Third occurrence of this trap in this file today; the
// clippy leg from #389 is what catches it.
#[cfg(test)]
thread_local! {
    static FORCE_SWAP_REMOVE_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Norm at or below which a vector has no representable direction.
///
/// The encoder stores every vector as (unit direction, norm). At or
/// below this threshold there is no meaningful direction to store, so
/// the vector is encoded with scale 0 and scores exactly 0 against
/// every query. This is documented behaviour, not an error: 0 is the
/// conventional cosine similarity of a zero vector, so the slot is
/// counted in `len()` and is returned by `search` only after every
/// vector that does have a direction. Callers for whom a zero-norm
/// embedding is a bug should reject it before `add`.
pub const MIN_INPUT_NORM: f32 = 1e-10;

/// Fewest rows [`TurboQuantIndex::calibrate`] will fit from.
///
/// A structural floor, not a quality one: the fit maps a low and a high
/// order statistic of each coordinate onto the codebook's edges, and two
/// distinct ranks need two rows. It is deliberately not a "this sample
/// is good enough" gate — see [`RECOMMENDED_CALIBRATION_ROWS`] for the
/// size to actually aim for, and [`TurboQuantIndex::calibrate`] for why
/// no row count can make an unrepresentative sample safe.
pub const MIN_CALIBRATION_ROWS: usize = encode::MIN_CALIBRATION_ROWS;

/// Calibration sample size to aim for: see
/// [`encode::RECOMMENDED_CALIBRATION_ROWS`].
pub const RECOMMENDED_CALIBRATION_ROWS: usize = encode::RECOMMENDED_CALIBRATION_ROWS;

/// The canonical Lloyd-Max codebook for `(bit_width, dim)` —
/// `(boundaries, centroids)`. The codebook is a pure function of these
/// two parameters; the v6 loader rejects a file whose embedded codebook
/// is not the one this function returns (#320) — it checks the defining
/// properties rather than re-deriving them, since the solve is far more
/// expensive than the load (#357) — so callers serializing through the
/// raw [`io`] writers must embed exactly these arrays (or use
/// [`TurboQuantIndex::codebook_for_write`]).
///
/// # Panics
///
/// If `bit_width` is not 2, 3 or 4, or `dim` is not a positive
/// multiple of 8 (the same bounds the index constructors enforce).
pub fn expected_codebook(bit_width: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(
        (2..=4).contains(&bit_width),
        "bit_width must be 2, 3 or 4, got {bit_width}"
    );
    assert!(
        dim >= 8 && dim % 8 == 0,
        "dim must be a positive multiple of 8, got {dim}"
    );
    codebook::codebook(bit_width, dim)
}

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
    // The parallel scan lives in encode.rs — one of the audited rayon
    // chokepoint files (fork safety, issue #147). Binding entry points
    // must reach it inside `with_pool` whenever
    // [`validation_parallelizes`] is true; below that threshold the scan
    // is a single chunk folded on the calling thread and touches no pool.
    encode::par_first_invalid_coord(values, dim, MAX_INPUT_MAGNITUDE)
}

/// True when [`first_invalid_coord`] on `len` values splits into more than
/// one rayon chunk, i.e. injects work into the current pool. Callers that
/// must control which pool that is (the Python binding, whose global pool
/// is a fork-unsafe sentinel — issue #288) gate on this.
pub fn validation_parallelizes(len: usize) -> bool {
    len > encode::VALIDATE_CHUNK
}

/// SIMD-blocked encoding of the index's rows — the layout the search
/// kernel scores directly.
///
/// Populated by a v6 load (the file already stores this layout), or by
/// repacking `packed_codes` — which [`TurboQuantIndex::search`] does on
/// first call and [`TurboQuantIndex::prepare`] does up front. Until one
/// of those happens the cache stays cold, and a mutation leaves it
/// cold. Once populated it is kept in step with the index under
/// `&mut self` rather than discarded: `data` always holds exactly
/// `n_blocks` blocks covering the index's current `n_vectors` rows,
/// including the zero padding of a partial tail block.
#[derive(Debug)]
struct BlockedCache {
    data: Vec<u8>,
    n_blocks: usize,
}

/// Whether an index has a TQ+ per-coordinate calibration.
///
/// TQ+ is a per-coordinate `(shift, scale)` that rescales the rotated
/// coordinates onto the distribution the codebook was solved against. It
/// is fitted from the empirical quantiles of a sample the caller
/// supplies to [`TurboQuantIndex::calibrate`], and from nothing else —
/// an index is calibrated exactly when someone calibrated it.
///
/// There are only these two states, and an index moves between them only
/// through `calibrate`. Adding, removing, saving and loading rows never
/// change it.
///
/// Query it with [`TurboQuantIndex::calibration_state`] /
/// [`IdMapIndex::calibration_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CalibrationState {
    /// No calibration: plain TurboQuant. The default for a new index and
    /// for one loaded from a file written without a calibration. Fully
    /// functional — just without the TQ+ recall gain, which is worth
    /// ~2.5 pp R@10 on average and up to ~8.7 pp on the most anisotropic
    /// data measured (SIFT-128 at 2 bits).
    Uncalibrated,
    /// A calibration is committed, and every stored row is encoded under
    /// it — including rows that were added *before* the
    /// [`TurboQuantIndex::calibrate`] call, which that call re-encoded.
    Calibrated,
}

/// Positional TurboQuant index.
///
/// Stores vectors compressed to `bit_width` bits per coordinate
/// (`{2, 3, 4}`) and identifies each vector by its insertion slot
/// (`0..len`). Slots are not stable across [`Self::swap_remove`] — the
/// last vector moves into the removed slot. For stable external `u64`
/// ids, use [`IdMapIndex`].
#[derive(Debug)]
pub struct TurboQuantIndex {
    /// Vector dimensionality. `None` means the index was constructed
    /// without a known dim (lazy mode) and hasn't seen its first add yet.
    /// Once set — either eagerly in [`Self::new`] or implicitly on the
    /// first [`Self::add_2d`] call — it never changes.
    dim: Option<usize>,
    bit_width: usize,
    n_vectors: usize,
    /// Per-vector bit-plane packed codes — the canonical in-memory form
    /// every mutation operates on. Materialized lazily: a v6 load seeds
    /// only the SIMD-blocked cache (the file's layout is one cheap
    /// transform from it), and the packed rows are reconstructed from
    /// that cache on first need (a mutation, or serialization without a
    /// warm cache) via `pack::native_to_seq` + `pack::seq_to_packed`.
    /// Every other construction path sets it eagerly, so the lazy path
    /// exists only between a v6 load and the first mutation.
    packed_codes: OnceLock<Vec<u8>>,
    scales: Vec<f32>,

    /// TQ+ per-coord calibration: both have length `dim` when the index
    /// is calibrated, and both are empty when it is not.
    ///
    /// Set by exactly one thing — an explicit [`Self::calibrate`] call —
    /// and never by an `add`. An index that has never been calibrated
    /// stays empty for its whole life and encodes as plain TurboQuant;
    /// so does a pre-TQ+ file loaded from disk. Nothing about the rows,
    /// their count, their order or their batching can change this pair,
    /// which is what makes the stored codes a pure function of (rows,
    /// calibration).
    ///
    /// A `calibrate` call on a populated index re-encodes every stored
    /// row into the new pair (see [`Self::calibrate`]), so the invariant
    /// "every stored row is encoded under the currently declared pair"
    /// holds at every observable point.
    tqplus_shift: Vec<f32>,
    tqplus_scale: Vec<f32>,

    // Thread-safe lazy caches. These are initialised from `&self` via
    // `OnceLock::get_or_init`, which allows `search` to take `&self`
    // and run concurrently from multiple threads without external
    // locking.
    //
    // `rotation`, `boundaries`, and `centroids` are deterministic functions
    // of `dim` (and `bit_width`), so they never need to be invalidated.
    //
    // `blocked` is row-dependent and so does need maintaining, but only
    // ever under `&mut self`, where no `&self` reader can be observing
    // it. Both mutators patch it in place through `get_mut` rather than
    // discarding it — `add` rewrites the tail block and appends any new
    // ones, `swap_remove` moves one lane and truncates — and a cold
    // cache stays cold, so neither pays for a layout nobody has asked
    // for yet. The only place the `OnceLock` is replaced outright is a
    // `calibrate` refit, which rewrites every row and so has no prior
    // state worth keeping. Whichever path runs, `blocked` covers exactly
    // `n_vectors` rows by the time the borrow ends.
    rotation: OnceLock<rotation::Rotation>,
    boundaries: OnceLock<Vec<f32>>,
    centroids: OnceLock<Vec<f32>>,
    blocked: OnceLock<BlockedCache>,

    /// Reusable encode scratch (the rotated-batch buffer). Purely
    /// derived state: never serialized, contents meaningless between
    /// calls — kept only so repeated adds reuse one allocation instead
    /// of paying a fresh multi-MB mmap + page-fault walk per call.
    encode_scratch: Vec<f32>,
    /// Element count the *previous* add asked of `encode_scratch`. Sizes
    /// the retention target in [`retain_scratch`], so a buffer is only
    /// kept while the adds around it are still using one that big.
    encode_scratch_prev: usize,
    /// Cursor into the last-synced v7 file, when this index has one:
    /// the commit the file holds, so `sync` writes only the delta.
    sync_cursor: Option<io_v7::SyncCursor>,
    /// The path the cursor belongs to. Syncing to a different path
    /// writes full and rebinds.
    sync_path: Option<std::path::PathBuf>,
    /// Slots whose redo ops ride the last-committed header: declared
    /// there, not yet materialized into their units. The next sync
    /// either materializes them (their live bytes ARE the committed
    /// bytes) or carries them forward if their unit got dirtied again.
    sync_pending: std::collections::HashSet<usize>,
    /// Sequential code bytes for slots `swap_remove` dirtied since the
    /// last sync, captured at the move that already had them in hand, laid
    /// end to end. Paired with `sync_capture_at`.
    ///
    /// A cache, nothing more: header assembly falls back to reading the row
    /// when a slot is absent, so correctness never depends on an entry
    /// existing — only on an entry that exists being right.
    ///
    /// One arena and one push per removal, rather than a map of owned rows:
    /// the bytes are free at the move, but a `Vec` allocation and a hash per
    /// removal are not, and they land on `remove()` where nothing asks for
    /// them.
    sync_capture_buf: Vec<u8>,
    /// `(slot, offset)` into `sync_capture_buf`, in capture order — so for a
    /// slot captured twice the later entry wins when the lookup is built.
    /// Length is capped at the header's op cap, past which a sync rewrites
    /// the file whole and needs no ops at all.
    sync_capture_at: Vec<(u32, u32)>,
    /// Disk-committed slots dirtied since the last sync. No value is
    /// captured — a redo op is an absolute write, so the live row at
    /// plan time is the op.
    sync_fresh: std::collections::HashSet<usize>,
    /// Bumped by every committed `calibrate`; a mismatch with the cursor
    /// forces the next sync to compact, since a refit rewrites every
    /// stored code.
    calib_gen: u64,
}

/// Release a reused scratch buffer that is far larger than the adds
/// around it need, and return the demand this call records for the next.
///
/// `prev` is the previous call's demand and `want` is this call's. The
/// target retained is the previous demand plus half again, and the
/// buffer is only touched when its capacity exceeds twice that. Both
/// margins are load-bearing, for different workloads:
///
/// * The **hysteresis** is what keeps ordinary shapes at zero extra
///   work: equal-sized, growing and jittering adds all sit at a capacity
///   below `2 * target`, so the branch never fires for them. Without it,
///   `shrink_to` sets capacity to *exactly* the target and discards the
///   headroom `Vec::reserve`'s amortized growth had built, so every
///   batch even slightly larger than the last pays a grow *and* a
///   shrink.
/// * The **slack** then covers the jumps the hysteresis alone does not.
///   Measured over twenty adds growing 5% each, driving a real `Vec`
///   through this exact sequence: 40 reallocations with neither margin,
///   7 with hysteresis alone, 9 with slack alone, 5 with both. For a
///   batch that triples and then holds, only the pair helps — 5, 5 and
///   3 respectively.
///
/// Neither margin changes what a steady same-size or one-shot bulk
/// workload does; all five variants measured identically on those.
///
/// A one-shot bulk add has `prev == 0`, so it releases the whole buffer
/// on the call that allocated it. There is no retention floor because
/// there is nothing for one to save: `Vec::reserve` from a zero capacity
/// allocates once, exactly as it would from any smaller capacity.
///
/// `truncate` before `shrink_to` is load-bearing on that release path.
/// `shrink_to` never goes below `len`, and the encode path leaves the
/// scratch at the full `n * dim` it just rotated — which is above the
/// target whenever there is anything to release, so `shrink_to` on its
/// own would do nothing there. (It is *not* inert in general: against a
/// short `len` it does shrink, which is why the old condition released
/// on a large-then-small pair.) `truncate` is itself a no-op when the
/// length is already at or below the target.
fn retain_scratch(scratch: &mut Vec<f32>, prev: usize, want: usize) -> usize {
    let target = prev.saturating_add(prev / 2);
    if scratch.capacity() > target.saturating_mul(2) {
        scratch.truncate(target);
        scratch.shrink_to(target);
    }
    want
}

/// Top-`k` results for a batch of queries, as returned by
/// [`TurboQuantIndex::search`] / [`TurboQuantIndex::search_with_mask`].
///
/// `scores` and `indices` are flattened row-major with one row per
/// query: row `qi` occupies indices `qi * k .. (qi + 1) * k` in both,
/// where `k` is the *effective* per-query result count stored in
/// [`Self::k`] — the requested `k` clamped to the number of searchable
/// vectors — not necessarily the `k` the caller asked for.
///
/// `Eq`/`Hash` are deliberately absent: `scores` holds `f32`, which has
/// no total equality. The derived `PartialEq` compares the four fields
/// in order, which means the score comparison is `f32`'s `==` and
/// inherits IEEE-754 semantics rather than bit equality — `NaN` is not
/// equal to itself (a result carrying one never equals its own clone,
/// however it was produced) and `+0.0 == -0.0` despite differing bit
/// patterns. Good enough for `assert_eq!` on results the index actually
/// returns; not a substitute for comparing scores within a tolerance,
/// and not enough to key a map.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResults {
    /// Scores, row-major `nq × k`, sorted descending within each row
    /// (best match first).
    pub scores: Vec<f32>,
    /// Slot indices into the index, row-major `nq × k`, aligned with
    /// [`Self::scores`].
    pub indices: Vec<i64>,
    /// Number of query rows; `0` when the index is lazy-uninitialized,
    /// since `dim` — and hence the row count — is unknown.
    pub nq: usize,
    /// Effective per-query result count: the requested `k` clamped to
    /// `min(k, len, n_allowed)`, where `n_allowed` is the number of
    /// mask-allowed vectors ([`len`](TurboQuantIndex::len) when no
    /// mask is given).
    pub k: usize,
}

impl SearchResults {
    /// The row of [`Self::scores`] for query `qi`:
    /// `&self.scores[qi * self.k..(qi + 1) * self.k]`.
    ///
    /// # Panics
    ///
    /// If the row is out of bounds (`qi >= nq` with `k > 0`).
    pub fn scores_for_query(&self, qi: usize) -> &[f32] {
        &self.scores[qi * self.k..(qi + 1) * self.k]
    }

    /// The row of [`Self::indices`] for query `qi`, aligned with
    /// [`Self::scores_for_query`].
    ///
    /// # Panics
    ///
    /// If the row is out of bounds (`qi >= nq` with `k > 0`).
    pub fn indices_for_query(&self, qi: usize) -> &[i64] {
        &self.indices[qi * self.k..(qi + 1) * self.k]
    }
}

impl TurboQuantIndex {
    /// The packed bit-plane codes, materializing them from the blocked
    /// cache if this index was v6-loaded and hasn't needed them yet.
    /// O(n·dim) on that first materialization, O(1) afterwards.
    fn packed(&self) -> &Vec<u8> {
        self.packed_codes.get_or_init(|| {
            let (Some(dim), Some(cache)) = (self.dim, self.blocked.get()) else {
                // Reaching here with vectors would mean a mutation
                // invalidated `blocked` before materializing packed —
                // an ordering bug that would silently wipe the codes.
                debug_assert!(
                    self.n_vectors == 0,
                    "packed_codes unset with no blocked cache but n_vectors > 0"
                );
                return Vec::new();
            };
            if self.n_vectors == 0 {
                return Vec::new();
            }
            let seq = pack::native_to_seq(&cache.data);
            pack::seq_to_packed(&seq, self.n_vectors, self.bit_width, dim)
        })
    }

    /// Whether the packed bit-plane rows are materialized.
    ///
    /// On a **non-empty** v6 [`Self::load`], `false` until something
    /// calls [`Self::packed_codes`] — and **nothing else does**. The
    /// blocked cache the load seeds is authoritative in that state, so
    /// [`Self::add`] takes the lazy-append branch, [`Self::swap_remove`]
    /// patches the cache with O(dim) lane ops, and serialization copies
    /// the cache verbatim; none of them triggers the O(n·dim)
    /// reconstruction, and none of them sets this flag. Such an index can
    /// therefore report `false` for its entire lifetime however much it
    /// is mutated.
    ///
    /// The one path that does set it without `packed_codes` is out of
    /// reach there: a v6 load of an **empty** index seeds the lock
    /// directly.
    ///
    /// So this is **not** a "first mutation after a load" probe, and
    /// gating a binding's fast path on it means gating it off forever on
    /// every loaded index — the defect behind #392. It answers exactly
    /// one question: which of the two code layouts is currently
    /// materialized.
    pub fn packed_ready(&self) -> bool {
        self.packed_codes.get().is_some()
    }

    /// Mutable access to the packed codes, materializing first (see
    /// [`Self::packed`]). Callers that mutate must also invalidate
    /// `blocked`, as before.
    fn packed_mut(&mut self) -> &mut Vec<u8> {
        self.packed();
        self.packed_codes
            .get_mut()
            .expect("packed_codes just materialized")
    }

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
            packed_codes: OnceLock::from(Vec::new()),
            scales: Vec::new(),
            tqplus_shift: Vec::new(),
            tqplus_scale: Vec::new(),
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
            encode_scratch: Vec::new(),
            encode_scratch_prev: 0,
            sync_cursor: None,
            sync_path: None,
            sync_pending: std::collections::HashSet::new(),
            sync_fresh: std::collections::HashSet::new(),
            sync_capture_buf: Vec::new(),
            sync_capture_at: Vec::new(),
            calib_gen: 0,
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
            packed_codes: OnceLock::from(Vec::new()),
            scales: Vec::new(),
            tqplus_shift: Vec::new(),
            tqplus_scale: Vec::new(),
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
            encode_scratch: Vec::new(),
            encode_scratch_prev: 0,
            sync_cursor: None,
            sync_path: None,
            sync_pending: std::collections::HashSet::new(),
            sync_fresh: std::collections::HashSet::new(),
            sync_capture_buf: Vec::new(),
            sync_capture_at: Vec::new(),
            calib_gen: 0,
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
    ///
    /// A vector whose L2 norm is `<= 1e-10` ([`MIN_INPUT_NORM`]) is not
    /// an error: it is stored with scale 0 and scores 0 against every
    /// query. See that constant for the rationale.
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
        // Empty add is a true no-op.
        if n == 0 {
            return;
        }
        if let Some((vi, ci, v)) = first_invalid_coord(vectors, dim) {
            panic!(
                "invalid input value at vector {vi}, coord {ci}: {v} \
                 (must be finite and |value| < 1e16 to avoid f32 norm overflow)",
            );
        }
        // One path, always. `add` reads the committed calibration and
        // never writes one — there is no warm-up buffer, no sample
        // threshold, and no batch that means more to the encoding than
        // any other. Whatever this index is calibrated to was set by an
        // explicit `calibrate` call, so a row's encoded bytes depend on
        // the row and the calibration and on nothing else: same rows,
        // same calibration, same bytes, however they were batched and in
        // whatever order they arrived.
        self.encode_and_append(vectors, n, dim);
    }

    /// Test-only switch that makes the next `encode` call panic, so tests
    /// can exercise the unwind guard below — and the ordering that guard
    /// depends on (#353). Panics inside `encode` are otherwise only
    /// reachable via a kernel invariant assert or a rayon worker fault,
    /// neither of which is inducible through the public API. Compiled only
    /// under `cfg(test)`, and thread-local — see the static's note on why
    /// this one cannot be a process-global the way
    /// `search::FORCE_SCALAR_FALLBACK` is (#373).
    #[cfg(test)]
    pub(crate) fn force_encode_panic(on: bool) {
        FORCE_ENCODE_PANIC.with(|f| f.set(on));
    }

    /// Test-only sibling of [`Self::force_encode_panic`] that unwinds
    /// from *inside* `encode`, after the batch has been appended to the
    /// output buffers — the only way to give `encode_and_append`'s
    // Test-only switch that makes the eager path's blocked-cache repack
    // panic, so the guard around it can be exercised. Thread-local for the
    // same reason the other switches are (#373): it panics, so a stray set
    // would take down whichever test reached the repack next.
    #[cfg(test)]
    pub(crate) fn force_repack_panic(on: bool) {
        FORCE_REPACK_PANIC.with(|f| f.set(on));
    }

    /// unwind guard real truncation work. See
    /// [`encode::force_panic_after_append`].
    #[cfg(test)]
    pub(crate) fn force_encode_panic_after_append(on: bool) {
        encode::force_panic_after_append(on);
    }

    /// Sibling of [`Self::force_encode_panic`] for the calibration fit
    /// inside [`Self::calibrate_2d`], thread-local for the same reason
    /// (#373). The fit and the refit's re-encode have to roll back
    /// different state, so they need separately targetable switches.
    #[cfg(test)]
    pub(crate) fn force_fit_panic(on: bool) {
        FORCE_FIT_PANIC.with(|f| f.set(on));
    }

    /// Sibling of [`Self::force_encode_panic`] for [`Self::swap_remove`],
    /// thread-local for the same reason (#373).
    ///
    /// `swap_remove` does unwind on a caller error — the `idx <
    /// n_vectors` assert below is documented and reachable from the
    /// public API. What it has no reachable unwind for is a *valid*
    /// `idx`: `packed_mut()` is called only under `if self.packed_codes
    /// .get().is_some()`, so its lazy rebuild never fires from here, and
    /// what remains is in-bounds indexing and allocation-free lane ops.
    /// That is the case [`crate::IdMapIndex::remove`] is in — its slot
    /// comes from the id table, so it is in bounds by construction.
    ///
    /// This switch exists to pin that caller's statement order anyway —
    /// it must not mutate its tables before calling this — so the
    /// ordering keeps holding if `swap_remove` ever becomes fallible for
    /// a valid `idx` (an incrementally materializing `packed_mut`, say).
    /// Same category as [`encode::force_panic_after_append`], which pins
    /// a guard whose `truncate` is likewise a no-op against today's
    /// `encode` and defense against a future incremental one (#384).
    ///
    /// Fires before anything in the index is touched, so it exercises
    /// exactly that ordering and nothing else: a panic *partway through*
    /// `swap_remove` would tear the inner index against its callers'
    /// tables, which no caller-side ordering can prevent.
    #[cfg(test)]
    pub(crate) fn force_swap_remove_panic(on: bool) {
        FORCE_SWAP_REMOVE_PANIC.with(|f| f.set(on));
    }

    /// Encode `n` rows and append them to the stored codes, using the
    /// committed calibration when there is one and fitting (and
    /// committing) a fresh one otherwise. Assumes the caller has already
    /// validated `vectors` and resolved `dim`.
    fn encode_and_append(&mut self, vectors: &[f32], n: usize, dim: usize) {
        // Rows land at slots [n_vectors, n_vectors + n), and none of them
        // can carry a capture. A capture is only taken for a slot below
        // n_vectors, and n_vectors only ever falls through `swap_remove`,
        // which drops the capture for the slot it retires. So by the time
        // a slot is free for an add to write, its entry is already gone —
        // asserted rather than re-swept, since a sweep here would be dead
        // code that reads as though it were load-bearing.
        debug_assert!(
            (self.n_vectors..self.n_vectors + n)
                .all(|slot| !self
                    .sync_capture_at
                    .iter()
                    .any(|&(s, _)| s as usize == slot)),
            "a slot about to be written by add still holds a removal capture",
        );
        let rotation = self
            .rotation
            .get_or_init(|| rotation::Rotation::new(dim));
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
        // The committed calibration, or `None` for an uncalibrated
        // index — encode applies it and never replaces it.
        let existing = if self.tqplus_shift.is_empty() {
            None
        } else {
            Some((self.tqplus_shift.as_slice(), self.tqplus_scale.as_slice()))
        };
        // In the v6-load window (blocked cache seeded from the file,
        // packed rows unmaterialized) the blocked cache stays
        // authoritative: encode the new rows into a temp buffer, append
        // them to the cache as direct lane writes, and leave packed
        // unset — the O(n·dim) materialization never runs for the
        // load→add→search/save flow. Everywhere else, materialize and
        // append in place as before.
        let lazy_append = self.n_vectors > 0
            && self.packed_codes.get().is_none()
            && self.blocked.get().is_some();
        if !lazy_append {
            // Materialize the packed rows (a v6-loaded index rebuilds
            // them from the still-valid blocked cache) so encode has the
            // existing rows to append after.
            self.packed();
        }
        // Take the scratch and output buffers out of self so they can be
        // borrowed mutably alongside the shared cache borrows above;
        // encode appends the new rows directly at their tails. In the
        // lazy window `take()` yields nothing and encode fills a fresh
        // temp holding only the new rows.
        let mut scratch = std::mem::take(&mut self.encode_scratch);
        let mut packed_codes = self.packed_codes.take().unwrap_or_default();
        debug_assert!(
            lazy_append || self.n_vectors == 0 || !packed_codes.is_empty(),
            "eager add must start from materialized packed rows"
        );
        let mut scales_buf = std::mem::take(&mut self.scales);
        // Unwind guard: encode appends to the taken buffers, so a panic
        // inside it (kernel invariant assert, rayon worker panic) must
        // not leave `self` with emptied buffers while n_vectors still
        // counts the old rows. On unwind, truncate back to the pre-call
        // lengths (encode never touches the existing prefix) and restore
        // the buffers before propagating.
        let packed_len_before = packed_codes.len();
        let scales_len_before = scales_buf.len();
        let bit_width = self.bit_width;
        let encode_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[cfg(test)]
            if FORCE_ENCODE_PANIC.with(|f| f.replace(false)) {
                panic!("forced encode panic (test)");
            }
            encode::encode(
                vectors,
                n,
                dim,
                rotation,
                boundaries,
                centroids,
                bit_width,
                existing,
                &mut scratch,
                &mut packed_codes,
                &mut scales_buf,
            )
        }));
        if let Err(panic) = encode_result {
            {
                scales_buf.truncate(scales_len_before);
                if !lazy_append {
                    packed_codes.truncate(packed_len_before);
                    self.packed_codes = OnceLock::from(packed_codes);
                }
                // lazy: the temp holds only new rows — drop it and leave
                // the lock unset; blocked was never touched.
                self.scales = scales_buf;
                self.encode_scratch = scratch;
                std::panic::resume_unwind(panic);
            }
        }
        // Keep the scratch warm for same-size adds, but don't let a
        // one-time huge bulk load pin its full rotated-batch capacity
        // for the index lifetime (#333).
        self.encode_scratch_prev = retain_scratch(&mut scratch, self.encode_scratch_prev, n * dim);
        self.encode_scratch = scratch;
        // `scales` is published per branch below, at the same commit point
        // as the codes and the count — publishing it here would leave it
        // holding `new_n` rows if the eager branch's cache patch panicked
        // (#388).

        // Nothing to commit calibration-side: `encode` applies the pair
        // it was handed and produces none, so an add can never change
        // what this index declares (the whole family of #284/#285/#303
        // bugs was about exactly that assignment).
        let old_n = self.n_vectors;
        // `n_vectors` is published only once the store it must agree with
        // is consistent (below, per branch). Incrementing first would
        // leave the count ahead of the codes if the cache update panicked
        // — and in the lazy window the blocked cache is the *only*
        // authoritative store, so anything reading `n_vectors` against it
        // afterwards (search, swap_remove, serialization) would index
        // past its real length.
        let new_n = old_n + n;

        if lazy_append {
            // packed stays unset (the lock was left empty by take());
            // append the temp's rows to the blocked cache as direct lane
            // writes (fresh blocks zero-padded, the partial tail block's
            // existing lanes untouched — the cache's exact-bytes
            // invariant carries them). The temp drops here.
            let bit_width = self.bit_width;
            let cache = self
                .blocked
                .get_mut()
                .expect("lazy_append requires a blocked cache");
            pack::append_lanes(&mut cache.data, &packed_codes, old_n, n, bit_width, dim);
            let (new_n_blocks, _, _) = pack::blocked_geometry(new_n, bit_width, dim);
            cache.n_blocks = new_n_blocks;
            self.scales = scales_buf;
            self.n_vectors = new_n;
            return;
        }
        // Eager path: the packed rows are authoritative and already carry
        // the new vectors. NOTHING is published until every fallible step
        // below has succeeded — the cache patch can panic (allocation, and
        // the repack itself), and publishing `packed_codes`/`scales` first
        // would leave them holding `new_n` rows while `n_vectors` still
        // reads `old_n`. A caller that catches the panic and keeps using
        // the index then addresses its next add past the orphans, which is
        // silent slot corruption rather than a detectable inconsistency
        // (#388). The patch is therefore built from the local buffer, and
        // codes, scales, cache and count are committed together at the end.

        // Maintain the blocked cache incrementally instead of discarding
        // it: appended rows only affect the (possibly partial) tail block
        // and the new blocks after it, so recompute exactly those from
        // the packed rows. A cold cache stays cold (first search builds
        // it). Rotation, boundaries, and centroids remain valid (they
        // only depend on `(dim, ROTATION_SEED)` and `(bit_width, dim)`).
        if self.blocked.get().is_some() {
            let (new_n_blocks, n_byte_groups, _) =
                pack::blocked_geometry(new_n, self.bit_width, dim);
            let block_bytes = n_byte_groups * BLOCK;
            let first_block = old_n / BLOCK;
            // Build the patch BEFORE touching the cache: `truncate` then
            // compute would leave a short cache behind if the repack
            // panicked, and the cache is serialized verbatim.
            //
            // The repack is the last fallible step, and `packed_codes` /
            // `scales_buf` are still owned locally here — taken out of
            // `self` before `encode` and not yet republished. So a panic
            // would drop them and leave the index with empty buffers
            // against a non-zero `n_vectors`. Restore the pre-call state
            // and resume, the same contract `encode`'s guard above keeps
            // (#388).
            let bit_width = self.bit_width;
            let patch = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                #[cfg(test)]
                if FORCE_REPACK_PANIC.with(|f| f.replace(false)) {
                    panic!("forced repack panic (test)");
                }
                pack::repack_block_range(
                    &packed_codes,
                    new_n,
                    bit_width,
                    dim,
                    first_block,
                    new_n_blocks,
                )
            })) {
                Ok(patch) => patch,
                Err(panic) => {
                    packed_codes.truncate(packed_len_before);
                    scales_buf.truncate(scales_len_before);
                    self.packed_codes = OnceLock::from(packed_codes);
                    self.scales = scales_buf;
                    std::panic::resume_unwind(panic);
                }
            };
            let cache = self.blocked.get_mut().expect("blocked present");
            cache.data.truncate(first_block * block_bytes);
            cache.data.extend_from_slice(&patch);
            cache.n_blocks = new_n_blocks;
        }
        // Commit point: every fallible step above has succeeded.
        self.packed_codes = OnceLock::from(packed_codes);
        self.scales = scales_buf;
        self.n_vectors = new_n;
    }

    /// Add `vectors` of dimension `dim`. On a lazy index this locks the
    /// index dim; on an already-dim'd index `dim` must match the index's
    /// existing dim.
    ///
    /// A zero-row batch is a no-op: `dim` is still validated (and must
    /// match an already-locked dim), but a lazy index stays lazy and its
    /// serialized bytes are unchanged.
    ///
    /// This is the form that bindings with shape information (e.g. the
    /// Python binding receiving a 2D numpy array) should use, since a
    /// flat `&[f32]` alone is ambiguous about its shape.
    ///
    /// Returns:
    /// - [`AddError::DimMismatch`] if `dim` does not match the
    ///   already-locked dim.
    /// - [`AddError::ZeroDim`] when committing a lazy index to `dim == 0`.
    /// - [`AddError::DimNotMultipleOf8`] when committing a lazy index
    ///   to a nonzero dim that is not a multiple of 8.
    /// - [`AddError::InvalidInputValue`] if any coordinate is non-finite
    ///   or has magnitude `>= 1e16`.
    ///
    /// A vector whose L2 norm is `<= 1e-10` ([`MIN_INPUT_NORM`]) is
    /// accepted and stored with scale 0 — see that constant.
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
                // it here, mirroring IdMapIndex::add_with_ids_2d — and as
                // its own variant, since "must be a multiple of 8" names
                // the wrong cause for an empty-embedding batch.
                if dim == 0 {
                    return Err(AddError::ZeroDim);
                }
                if dim % 8 != 0 {
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
        // Validate the length/dim relationship BEFORE committing dim on a
        // lazy index. add() re-checks this, but by then the dim would
        // already be locked — a panic there left the lazy index wedged
        // (committed dim, zero vectors), turning a follow-up add_2d with a
        // different dim into a confusing DimMismatch instead of a fresh
        // start (#129).
        assert_eq!(
            vectors.len() % dim,
            0,
            "vectors length must be a multiple of dim"
        );
        // A zero-row batch is a no-op (see the guard in `add`), so return
        // before the lazy dim commit below. Committing first made a no-op
        // permanently lock a lazy index's dim and change its serialized
        // bytes (the `dim=0` sentinel became the batch's dim), which then
        // survived save/load (#308). The dim validation above still runs,
        // so a zero-row batch with a mismatched or malformed dim reports
        // the same error it always did.
        if vectors.is_empty() {
            return Ok(());
        }
        // Lazy commit happens via add() (which goes through `self.dim.expect`),
        // so re-do the dim assignment here for the lazy-first-add case.
        if self.dim.is_none() {
            // `add` is fallible (an encode panic — kernel invariant
            // assert or rayon worker fault), and it needs the dim
            // committed to run at all. Committing it and leaving it
            // committed after an unwind wedges the lazy index at
            // "committed dim, zero vectors", so a follow-up `add_2d` with
            // a different dim gets `DimMismatch` instead of the fresh
            // start #129 established. Roll the commit back — along with
            // all three caches `add` derives from this dim, which the
            // next add at a different dim would otherwise reuse. Each
            // matters differently, so none of the three resets is
            // redundant: the rotation asserts its input row length, so
            // reusing it turns the retry into a panic inside `rotation`
            // rather than a fresh start (loud, but still wrong), while
            // `boundaries`/`centroids` are dim-dependent *and* length-
            // compatible — a stale codebook for the old dim would be
            // accepted and silently mis-quantize every row. The codebook
            // case is the silent one and only unreachable because the
            // rotation assert fires first; resetting the rotation alone
            // would leave it exposed the moment that ordering changed.
            // With all three rolled back a caught panic leaves the index
            // exactly as lazy as it was (#380). `encode_and_append`'s own
            // guard restores the code and scale buffers, and nothing else
            // is touched before the encode.
            self.dim = Some(dim);
            if let Err(panic) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.add(vectors)))
            {
                self.dim = None;
                self.rotation = OnceLock::new();
                self.boundaries = OnceLock::new();
                self.centroids = OnceLock::new();
                std::panic::resume_unwind(panic);
            }
            return Ok(());
        }
        self.add(vectors);
        Ok(())
    }

    /// Run a top-`k` search against the index.
    ///
    /// Takes `&self` and is safe to call concurrently from multiple
    /// threads. The first caller on a fresh index pays the one-time
    /// cache initialisation cost (rotation, Lloyd-Max centroids
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
    /// magnitude `>= 1e16`. Both indicate the caller handed the index a
    /// buffer it cannot score at all.
    ///
    /// Neither check can run on an index with no committed `dim` (a
    /// [`Self::new_lazy`] index that has never been added to): there is
    /// no dim to measure the buffer against, so any `queries` returns
    /// the empty result below.
    ///
    /// Use [`Self::try_search`] to get these conditions back as a
    /// [`SearchError`] instead — the right choice whenever the query
    /// vectors come from outside the process.
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        self.search_with_mask(queries, k, None)
    }

    /// [`Self::search`] as a `Result`: the non-panicking form.
    ///
    /// Identical to `search` on well-formed input — same results, same
    /// caches, same cost. The difference is only in how a malformed
    /// `queries` buffer is reported: [`SearchError`] instead of a panic
    /// that unwinds the calling thread. A service scoring vectors it did
    /// not produce (an HTTP body, an embedding provider that emitted a
    /// NaN) wants this one; `search` stays the right call when a ragged
    /// or non-finite query would be a bug in your own code.
    ///
    /// Returns [`SearchError::QueryBufferNotMultipleOfDim`] or
    /// [`SearchError::InvalidQueryValue`]. See
    /// [`Self::try_search_with_mask`] for the masked form.
    pub fn try_search(&self, queries: &[f32], k: usize) -> Result<SearchResults, SearchError> {
        self.try_search_with_mask(queries, k, None)
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
    /// A mask names slots, and [`Self::swap_remove`] renumbers them, so
    /// **any** mutation invalidates a mask — not only one that changes
    /// the length. The length check below is not what protects you: a
    /// `swap_remove(i)` + `add` pair restores the original length while
    /// leaving a different vector in slot `i`, so a mask built before
    /// that pair passes validation and then silently selects a
    /// different set of vectors than the caller intended. Rebuild the
    /// mask after every mutation.
    ///
    /// # Panics
    ///
    /// - If `mask.len() != self.len()` (when `mask` is `Some`).
    /// - If `queries.len()` is not a multiple of `dim`.
    /// - If any query coordinate is non-finite or has magnitude `>= 1e16`.
    ///
    /// As with [`Self::search`], none of the three can fire on an index
    /// with no committed `dim` — that case returns the empty result
    /// before any validation. Use [`Self::try_search_with_mask`] for the
    /// non-panicking form.
    pub fn search_with_mask(
        &self,
        queries: &[f32],
        k: usize,
        mask: Option<&[bool]>,
    ) -> SearchResults {
        // Single source of validation: the checked form below owns all
        // three conditions, and this one turns them back into panics.
        // Re-validating here instead would run `first_invalid_coord`'s
        // O(nq·dim) scan twice per query batch. The payload is now the
        // error's `Display` rather than an `assert_eq!` rendering, so
        // three of the four sites report differently than they did —
        // see `try_search_with_mask` for the before/after.
        self.try_search_with_mask(queries, k, mask)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::search_with_mask`] as a `Result`: the non-panicking form.
    ///
    /// Returns [`SearchError::QueryBufferNotMultipleOfDim`],
    /// [`SearchError::InvalidQueryValue`], or
    /// [`SearchError::MaskLengthMismatch`]. On success the result is
    /// exactly what `search_with_mask` would have returned.
    ///
    /// `search_with_mask` now calls this function and panics with the
    /// error's `Display` text, so the two forms cannot diverge in what
    /// they detect: the conditions, the order they are checked in and
    /// the results returned are all exactly as before.
    ///
    /// The panic *text* did change at three of the four sites, which
    /// were previously raised by `assert_eq!` and now carry the error's
    /// `Display` alone. (Four sites, three conditions: the mask-length
    /// check has one site for an empty index and one for a populated
    /// one.)
    ///
    /// ```text
    /// before: assertion `left == right` failed: mask length 99 does not match index size 16
    ///           left: 99
    ///          right: 16
    /// after:  mask length 99 does not match index size 16
    ///
    /// before: assertion `left == right` failed
    ///           left: 65
    ///          right: 64
    /// after:  query buffer length 65 not a multiple of dim 64
    /// ```
    ///
    /// The fourth site, the non-finite-coordinate panic, is
    /// byte-identical: it was always a `panic!` rather than an assert.
    ///
    /// What still matches and what does not: at the two mask sites the
    /// message text was already inside the old payload, so a
    /// `should_panic(expected = "mask length")` keeps matching. The
    /// ragged-buffer assert carried *no* message, so its old and new
    /// payloads have nothing in common but the two numbers (`65` and
    /// `64`); any `expected =` string that matched the old one will not
    /// match the new.
    pub fn try_search_with_mask(
        &self,
        queries: &[f32],
        k: usize,
        mask: Option<&[bool]>,
    ) -> Result<SearchResults, SearchError> {
        // A lazy index that's never seen an add returns an empty result
        // shaped according to the caller's query count (best effort: we
        // don't know dim, so nq is 0). Matches Python users' expectation
        // that `search` on an empty store is a no-op rather than an error.
        let Some(dim) = self.dim else {
            return Ok(SearchResults {
                scores: Vec::new(),
                indices: Vec::new(),
                nq: 0,
                k: 0,
            });
        };
        let nq = queries.len() / dim;
        if queries.len() != nq * dim {
            return Err(SearchError::QueryBufferNotMultipleOfDim {
                queries_len: queries.len(),
                dim,
            });
        }
        // Reject non-finite / huge-magnitude queries. Same rationale as
        // `add`: NaN / Inf / overflow-magnitude values poison the SIMD
        // scoring kernel and produce arbitrary indices with NaN scores,
        // silently rather than as a typed error.
        if let Some((vi, ci, v)) = first_invalid_coord(queries, dim) {
            return Err(SearchError::InvalidQueryValue {
                query_index: vi,
                coord_index: ci,
                value: v,
            });
        }

        // An empty index has nothing to score: return the empty result
        // shape without building the rotation/centroid/blocked caches.
        // Besides skipping wasted work for a legitimately-empty index,
        // this stops a tiny file declaring a large dim with n_vectors=0
        // from driving the codebook/blocked-layout build on first search.
        if self.n_vectors == 0 {
            if let Some(m) = mask {
                if !m.is_empty() {
                    return Err(SearchError::MaskLengthMismatch {
                        expected: 0,
                        got: m.len(),
                    });
                }
            }
            return Ok(SearchResults {
                scores: Vec::new(),
                indices: Vec::new(),
                nq,
                k: 0,
            });
        }

        let rotation = self
            .rotation
            .get_or_init(|| rotation::Rotation::new(dim));
        let centroids = self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bit_width, dim);
            c
        });
        let blocked = self.blocked.get_or_init(|| {
            let (data, n_blocks) =
                pack::repack(self.packed(), self.n_vectors, self.bit_width, dim);
            BlockedCache { data, n_blocks }
        });

        // A wrong-length mask is caller data, so it leaves through the
        // `Result` rather than aborting midway through the bitset build
        // below, which is where the `assert_eq!` this replaces used to
        // sit. Note this is still *after* the rotation/centroid/blocked
        // caches are warmed above: that ordering is inherited, not
        // chosen, and it means a bad mask on a cold index pays for the
        // layout build before it is rejected. Moving the check above
        // those `get_or_init` calls would be a strict improvement and is
        // deliberately left out of the change that introduced this
        // `Result`, so that "validation order is unchanged" stays true.
        if let Some(m) = mask {
            if m.len() != self.n_vectors {
                return Err(SearchError::MaskLengthMismatch {
                    expected: self.n_vectors,
                    got: m.len(),
                });
            }
        }
        let packed_mask = mask.map(|m| {
            // Build word-at-a-time out of 64-bool chunks and count the
            // allowed slots in the same pass. The byte-at-a-time form
            // this replaces did one bounds-checked read-modify-write of
            // `buf` per slot and then a second full pass to popcount,
            // which is measurable (sub-millisecond but a double-digit
            // share of masked-search time) at index sizes in the
            // millions.
            let n_words = self.n_vectors.div_ceil(64);
            let mut buf = Vec::with_capacity(n_words);
            let mut allowed = 0usize;
            for chunk in m.chunks(64) {
                let mut word = 0u64;
                for (bit, &b) in chunk.iter().enumerate() {
                    word |= (b as u64) << bit;
                }
                allowed += word.count_ones() as usize;
                buf.push(word);
            }
            debug_assert_eq!(buf.len(), n_words);
            (buf, allowed)
        });

        let n_allowed = packed_mask.as_ref().map_or(self.n_vectors, |p| p.1);
        let packed_mask = packed_mask.map(|p| p.0);
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

        Ok(SearchResults {
            scores,
            indices,
            nq,
            k: effective_k,
        })
    }

    /// Eagerly populate the search caches (rotation, centroids
    /// and SIMD-blocked code layout).
    ///
    /// Calling `prepare` is optional — `search` will materialise the
    /// caches on its first call if needed. Use it to move the one-time
    /// cost out of the first query path, for example right after
    /// [`TurboQuantIndex::load`] or after a batch of [`Self::add`] calls.
    ///
    /// Safe to call multiple times and from multiple threads.
    pub fn prepare(&self) {
        // On a lazy index that's seen no add, there's nothing to prepare
        // — dim is unknown and the caches depend on it.
        let Some(dim) = self.dim else { return };
        // Same for an empty index: search short-circuits before touching
        // the caches, and `add` builds the rotation itself if vectors
        // arrive later — so building here is pure wasted work (and a
        // DoS on a loaded empty file declaring a large dim).
        if self.n_vectors == 0 {
            return;
        }
        self.rotation
            .get_or_init(|| rotation::Rotation::new(dim));
        self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bit_width, dim);
            c
        });
        self.blocked.get_or_init(|| {
            let (data, n_blocks) =
                pack::repack(self.packed(), self.n_vectors, self.bit_width, dim);
            BlockedCache { data, n_blocks }
        });
    }

    /// First row NOT covered by the synced file's committed whole
    /// blocks — slots below this live in units on disk.
    pub(crate) fn sync_watermark(&self) -> usize {
        self.sync_cursor
            .map(|c| (c.n_synced as usize) / BLOCK * BLOCK)
            .unwrap_or(0)
    }

    /// Test-only: how many removal captures are live for the next sync.
    ///
    /// The capture is a pure optimization — a miss re-reads the row — so no
    /// output test can tell whether it is populated, only that results are
    /// right either way. This exposes activation so a test can pin that it
    /// still happens for the slots it is meant to cover.
    ///
    /// Gated to x86 alongside the capture itself: elsewhere nothing is ever
    /// captured, so the only caller is compiled out and this would be dead.
    #[cfg(all(test, target_arch = "x86_64"))]
    pub(crate) fn captured_len_for_test(&self) -> usize {
        self.sync_capture_at.len()
    }

    /// slot -> (offset, len) into `sync_capture_buf`, built once per sync.
    ///
    /// Empty when `packed_codes` is live: a capture is only ever taken with
    /// the blocked cache authoritative, and consulting one afterwards is
    /// what would let a re-calibration's re-encoding be read through a
    /// stale row. Built in capture order so a slot captured twice resolves
    /// to its later bytes, and entries are contiguous, so each length is
    /// the next entry's offset.
    fn capture_lookup(&self) -> std::collections::HashMap<usize, (usize, usize)> {
        if self.packed_codes.get().is_some() || self.sync_capture_at.is_empty() {
            return std::collections::HashMap::new();
        }
        let mut m = std::collections::HashMap::with_capacity(self.sync_capture_at.len());
        for (i, &(slot, off)) in self.sync_capture_at.iter().enumerate() {
            let end = match self.sync_capture_at.get(i + 1) {
                Some(&(_, next)) => next as usize,
                None => self.sync_capture_buf.len(),
            };
            m.insert(slot as usize, (off as usize, end - off as usize));
        }
        m
    }

    /// Mark a disk-committed slot as diverged. No bytes are captured —
    /// the redo op serialized at the next sync reads the live row.
    fn mark_dirty(&mut self, slot: usize) {
        if slot < self.sync_watermark() {
            self.sync_fresh.insert(slot);
        }
    }

    /// The plan the next incremental sync would run, without running
    /// it — the crash harness tears these batches at every byte.
    #[cfg(test)]
    pub(crate) fn plan_next_sync(&mut self, kind: u8, ids: Option<&[u64]>) -> io_v7::SyncPlan {
        let dim = self.dim.expect("plan_next_sync on a lazy index");
        if self.blocked.get().is_none() {
            self.packed();
        }
        if self.boundaries.get().is_none() || self.centroids.get().is_none() {
            let (b, c) = codebook::codebook(self.bit_width, dim);
            let _ = self.boundaries.set(b);
            let _ = self.centroids.set(c);
        }
        let seq_blocks = |from: usize, to: usize| self.seq_blocks_range(from, to);
        let capture_lookup = self.capture_lookup();
        let row_codes = |idx: usize, out: &mut Vec<u8>| {
            match capture_lookup.get(&idx) {
                Some(&(off, len)) => {
                    out.extend_from_slice(&self.sync_capture_buf[off..off + len])
                }
                None => out.extend_from_slice(&self.seq_row(idx)),
            }
        };
        let source = io_v7::SyncSource {
            kind,
            dim,
            bit_width: self.bit_width,
            n_vectors: self.n_vectors,
            seq_blocks: &seq_blocks,
            row_codes: &row_codes,
            scales: &self.scales,
            ids,
            tqplus_shift: &self.tqplus_shift,
            tqplus_scale: &self.tqplus_scale,
            boundaries: self.boundaries.get().expect("seeded above"),
            centroids: self.centroids.get().expect("seeded above"),
        };
        io_v7::plan_incremental(
            &source,
            self.sync_cursor.expect("plan_next_sync on an unbound index"),
            &self.sync_pending,
            &self.sync_fresh,
        )
        .expect("plan_next_sync: ops exceed the header capacity")
    }

    /// This row's codes in the arch-neutral sequential layout (one byte
    /// per byte-group), read from whichever in-memory layout is live —
    /// O(dim), no whole-index materialization.
    fn seq_row(&self, idx: usize) -> Vec<u8> {
        let dim = self.dim.expect("seq_row on a dim-less index");
        // Two different strides: packed rows are bit-packed
        // (`dim * bits / 8`), the sequential-blocked layout stores one
        // byte per group (`dim / (8 / bits)`). They agree for 2- and
        // 4-bit but NOT for 3-bit, whose codes occupy 4-bit fields.
        let packed_row = dim * self.bit_width / 8;
        let (_, row_bytes, _) = pack::blocked_geometry(1, self.bit_width, dim);
        if let Some(packed) = self.packed_codes.get() {
            return pack::extract_codes_flat(
                &packed[idx * packed_row..(idx + 1) * packed_row],
                1,
                self.bit_width,
                dim,
            );
        }
        // O(dim) straight off the blocked cache — removal capture must
        // not pay for the other 31 rows of the block. Off x86 the
        // native layout IS sequential-blocked, so this is a stride-32
        // gather; on x86 each byte de-interleaves from its nibble
        // planes (the primitive the scalar search fallback uses).
        let cache = self.blocked.get().expect("no code layout materialized");
        let block_bytes = row_bytes * BLOCK;
        let base = (idx / BLOCK) * block_bytes;
        let lane = idx % BLOCK;
        #[cfg(target_arch = "x86_64")]
        return (0..row_bytes)
            .map(|g| pack::deinterleave_x86_code_byte(&cache.data, base + g * BLOCK, lane))
            .collect();
        #[cfg(not(target_arch = "x86_64"))]
        (0..row_bytes)
            .map(|g| pack::seq_lane_byte(&cache.data, base, g, lane))
            .collect()
    }

    /// Sequential-blocked codes for rows `[from, to)` — whole 32-row
    /// blocks only. O(range), not O(index), from either layout.
    fn seq_blocks_range(&self, from: usize, to: usize) -> Vec<u8> {
        debug_assert!(from.is_multiple_of(BLOCK) && to.is_multiple_of(BLOCK) && from <= to);
        let dim = self.dim.expect("seq_blocks_range on a dim-less index");
        let packed_row = dim * self.bit_width / 8;
        let (_, row_bytes, _) = pack::blocked_geometry(1, self.bit_width, dim);
        if let Some(packed) = self.packed_codes.get() {
            let flat = pack::extract_codes_flat(
                &packed[from * packed_row..to * packed_row],
                to - from,
                self.bit_width,
                dim,
            );
            let n = to - from;
            return pack::pack_blocked_sequential(
                n,
                n / BLOCK,
                row_bytes,
                n / BLOCK * row_bytes * BLOCK,
                &flat,
            );
        }
        let cache = self.blocked.get().expect("no code layout materialized");
        let block_bytes = row_bytes * BLOCK;
        pack::native_to_seq(&cache.data[from / BLOCK * block_bytes..to / BLOCK * block_bytes])
    }

    /// Persist this index's changes to `path` incrementally.
    ///
    /// The first sync of a path — or any sync after a
    /// [`Self::calibrate`] call, or to a different path than last time —
    /// writes the whole index as a fresh sync container (temp file + atomic
    /// rename, so a previous file at `path` survives a crash). Every
    /// other sync appends only what changed since the last one: the rows
    /// added, one small patch record per removal, and a commit record —
    /// kilobytes, where [`Self::write`] rewrites the whole file.
    ///
    /// Crash safety: appended bytes are made durable before the commit
    /// header that adopts them flips, and a removal never touches
    /// committed bytes during the sync that commits it — it rides the
    /// header as a redo op, materialized idempotently by a later sync.
    /// A crash at any byte of a sync recovers the previous commit
    /// exactly: a torn commit header fails its checksum and load falls
    /// back to the previous one. Damage from outside the writer (bit
    /// rot, mangled copies) is out of scope, as it is for `write`.
    ///
    /// [`Self::write`] / [`Self::load`] keep their meaning; `load`
    /// recognises both formats, and the first `sync` to a v6 file's
    /// path replaces it with the sync container.
    ///
    /// Single writer: one process syncs a given path at a time. Each
    /// full write stamps the file with a fresh random nonce, so if
    /// another process does replace the file, the next sync here
    /// reports it as foreign rather than corrupting it — but two
    /// processes syncing the same path concurrently is unsupported.
    pub fn sync(&mut self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.sync_v7_impl(path.as_ref(), 0, None)
    }

    /// The shared v7 sync engine. `IdMapIndex` drives it with `kind` 1
    /// and the id table (redo ops and appended units read ids from it).
    pub(crate) fn sync_v7_impl(
        &mut self,
        path: &Path,
        kind: u8,
        ids_full: Option<&[u64]>,
    ) -> std::io::Result<()> {
        let Some(dim) = self.dim else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot sync a lazy index that has never seen an add or calibrate",
            ));
        };
        if self.blocked.get().is_none() {
            self.packed();
        }
        if self.boundaries.get().is_none() || self.centroids.get().is_none() {
            let (b, c) = codebook::codebook(self.bit_width, dim);
            let _ = self.boundaries.set(b);
            let _ = self.centroids.set(c);
        }
        let geo = io_v7::Geo {
            kind,
            dim,
            bit_width: self.bit_width,
            n_calib: self.tqplus_shift.len(),
        };
        // The identity check runs whenever this index is BOUND to the
        // path — including when a calibrate has queued a compaction.
        // Deciding "full rewrite" without opening the file would skip
        // the nonce comparison, and a compaction that renames over a
        // file another writer has taken over destroys their commits.
        let bound = matches!(
            (&self.sync_cursor, &self.sync_path),
            (Some(_), Some(p)) if p == path
        );
        let state = if bound {
            let c = self.sync_cursor.as_ref().expect("checked above");
            io_v7::cursor_state(path, c, &geo)?
        } else {
            io_v7::CursorState::Replaced
        };
        let incremental = bound && {
            let c = self.sync_cursor.as_ref().expect("checked above");
            c.calib_gen == self.calib_gen
        };
        if matches!(state, io_v7::CursorState::Foreign) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "the v7 file at {} no longer matches this index's last sync \
                     (another writer advanced or replaced it); load() the file to \
                     adopt its state, or choose a different path",
                    path.display()
                ),
            ));
        }
        let result = {
            let seq_blocks = |from: usize, to: usize| self.seq_blocks_range(from, to);
            let capture_lookup = self.capture_lookup();
            let row_codes = |idx: usize, out: &mut Vec<u8>| {
                match capture_lookup.get(&idx) {
                    Some(&(off, len)) => {
                        out.extend_from_slice(&self.sync_capture_buf[off..off + len])
                    }
                    None => out.extend_from_slice(&self.seq_row(idx)),
                }
            };
            let source = io_v7::SyncSource {
                kind,
                dim,
                bit_width: self.bit_width,
                n_vectors: self.n_vectors,
                seq_blocks: &seq_blocks,
                row_codes: &row_codes,
                scales: &self.scales,
                ids: ids_full,
                tqplus_shift: &self.tqplus_shift,
                tqplus_scale: &self.tqplus_scale,
                boundaries: self.boundaries.get().expect("seeded above"),
                centroids: self.centroids.get().expect("seeded above"),
            };
            match state {
                io_v7::CursorState::Intact if incremental => {
                    let c = self.sync_cursor.expect("checked above");
                    // `None` when the carried ops exceed the header's
                    // capacity — a mass removal — where a full rewrite
                    // is proportionate.
                    match io_v7::plan_incremental(
                        &source,
                        c,
                        &self.sync_pending,
                        &self.sync_fresh,
                    ) {
                        Some(plan) => {
                            io_v7::run_sync(path, &plan).map(|c| (c, plan.carried))
                        }
                        None => io_v7::write_full(path, &source, self.calib_gen)
                            .map(|c| (c, Vec::new())),
                    }
                }
                _ => io_v7::write_full(path, &source, self.calib_gen)
                    .map(|c| (c, Vec::new())),
            }
        };
        match result {
            Ok((cursor, carried)) => {
                self.sync_pending = carried.into_iter().collect();
                self.sync_fresh.clear();
                self.sync_capture_buf.clear();
                self.sync_capture_at.clear();
                self.sync_cursor = Some(io_v7::SyncCursor {
                    calib_gen: self.calib_gen,
                    ..cursor
                });
                self.sync_path = Some(path.to_path_buf());
                Ok(())
            }
            // A failed sync may still have landed bytes — including a
            // complete, self-verifying commit header (write and fsync
            // errors surface after bytes reach the OS, and after a
            // failed fsync the page cache can no longer be trusted).
            // If the cursor stayed bound, a landed header would make
            // every retry report "another writer advanced this file"
            // forever. Drop the binding: the next sync takes the full
            // write path (temp file + atomic rename), which is correct
            // from any on-disk state.
            Err(e) => {
                self.sync_cursor = None;
                self.sync_path = None;
                Err(e)
            }
        }
    }

    /// Load a v7 file written by [`Self::sync`]; the reloaded index is
    /// bound to `path` as its sync target, so the next `sync` writes
    /// only the delta.
    ///
    /// Lands in the same state a v6 load lands in: only the SIMD-blocked
    /// search layout is seeded — `packed_codes` stays unset, and adds
    /// take the lazy-append branch — so a synced-file load holds one copy of the
    /// codes, not two. This is the RAM property #471 exists for.
    fn load_v7(path: &Path) -> std::io::Result<Self> {
        let l = io_v7::load(path, 0, 0)?;
        Self::from_v7(l, path)
    }

    /// Assemble an index from a v7 payload — the shared tail of
    /// [`Self::load_v7`] and `IdMapIndex`'s v7 loader.
    pub(crate) fn from_v7(l: io_v7::V7Load, path: &Path) -> std::io::Result<Self> {
        let n_blocks = l.n_vectors.div_ceil(BLOCK);
        // The units already hold the seq-blocked layout; one platform
        // transform in place (identity off x86) and it IS the search
        // cache.
        let native = pack::seq_into_native(l.seq_blocked);
        let (tqplus_shift, tqplus_scale) =
            Self::normalize_calibration(l.tqplus_shift, l.tqplus_scale);
        let (boundaries, centroids) = codebook::codebook(l.bit_width, l.dim);
        let blocked = OnceLock::new();
        let boundaries_lock = OnceLock::new();
        let centroids_lock = OnceLock::new();
        let packed_codes = if l.n_vectors == 0 {
            OnceLock::from(Vec::new())
        } else {
            let _ = blocked.set(BlockedCache {
                data: native,
                n_blocks,
            });
            let _ = boundaries_lock.set(boundaries);
            let _ = centroids_lock.set(centroids);
            OnceLock::new()
        };
        Ok(Self {
            dim: Some(l.dim),
            bit_width: l.bit_width,
            n_vectors: l.n_vectors,
            packed_codes,
            scales: l.scales,
            tqplus_shift,
            tqplus_scale,
            rotation: OnceLock::new(),
            boundaries: boundaries_lock,
            centroids: centroids_lock,
            blocked,
            encode_scratch: Vec::new(),
            encode_scratch_prev: 0,
            sync_cursor: Some(l.cursor),
            sync_path: Some(path.to_path_buf()),
            sync_pending: l.pending_slots.iter().copied().collect(),
            sync_fresh: std::collections::HashSet::new(),
            sync_capture_buf: Vec::new(),
            sync_capture_at: Vec::new(),
            calib_gen: 0,
        })
    }

    /// Save the index to `path` in the `.tv` format.
    ///
    /// The write is atomic with respect to `path`: the bytes go to a
    /// sibling temp file which is fsynced and renamed over the
    /// destination, so `path` never holds a torn index and any previous
    /// file there survives a failed write. `Err` means the save did not
    /// commit.
    ///
    /// Reload with [`Self::load`]. See
    /// [`Self::write_with_durability`] to trade the fsync for speed, and
    /// [`Self::write_to_writer`] / [`Self::to_bytes`] for the in-memory
    /// forms.
    ///
    /// The calibration travels with the file. A
    /// [`Calibrated`](CalibrationState::Calibrated) index writes its
    /// `(shift, scale)` trailer and reloads calibrated; an
    /// [`Uncalibrated`](CalibrationState::Uncalibrated) one writes no
    /// trailer and reloads uncalibrated, ready to be calibrated later.
    /// Neither depends on how many vectors the index holds — there is no
    /// state a save can silently forfeit.
    ///
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.write_with_durability(path, io::Durability::Durable)
    }

    /// [`Self::write`] with an explicit [`io::Durability`] level:
    /// `Durable` (the default) fsyncs before the atomic rename; `Fast`
    /// keeps the temp-file + atomic-rename protocol (the destination can
    /// never hold a torn index and the previous file survives a process
    /// crash) but skips fsync, so a power loss shortly after a completed
    /// save may lose the new file.
    pub fn write_with_durability(
        &self,
        path: impl AsRef<Path>,
        durability: io::Durability,
    ) -> std::io::Result<()> {
        // Sentinel: dim=0 in the file header means "lazy index, dim never
        // committed". The loader interprets dim=0 + n_vectors=0 as a
        // freshly-constructed lazy state. dim=0 is otherwise meaningless
        // (the constructor asserts dim % 8 == 0 with dim >= 8), so this
        // doesn't collide with any valid eager index.
        let (boundaries, centroids) = self.codebook_for_write();
        // Warm blocked cache: borrow it instead of materializing the
        // sequential payload. On x86 the per-chunk deinterleave runs
        // inside the writer threads (overlapping device writes); on other
        // arches the cache IS the sequential layout, so this skips the
        // whole-payload copy. Bytes are identical either way.
        if self.n_vectors > 0 && self.dim.is_some() {
            if let Some(cache) = self.blocked.get() {
                #[cfg(target_arch = "x86_64")]
                return io::write_native_with_durability(
                    path,
                    self.bit_width,
                    self.dim.unwrap_or(0),
                    self.n_vectors,
                    &cache.data,
                    &boundaries,
                    &centroids,
                    &self.scales,
                    &self.tqplus_shift,
                    &self.tqplus_scale,
                    durability,
                );
                #[cfg(not(target_arch = "x86_64"))]
                return io::write_with_durability(
                    path,
                    self.bit_width,
                    self.dim.unwrap_or(0),
                    self.n_vectors,
                    &cache.data,
                    &boundaries,
                    &centroids,
                    &self.scales,
                    &self.tqplus_shift,
                    &self.tqplus_scale,
                    durability,
                );
            }
        }
        io::write_with_durability(
            path,
            self.bit_width,
            self.dim.unwrap_or(0),
            self.n_vectors,
            &self.codes_blocked_seq(),
            &boundaries,
            &centroids,
            &self.scales,
            &self.tqplus_shift,
            &self.tqplus_scale,
            durability,
        )
    }

    /// Borrow the warm native blocked cache for a fused write, if one
    /// exists. `None` for empty/lazy indexes or a cold cache (callers
    /// fall back to [`Self::codes_blocked_seq`]).
    pub(crate) fn blocked_native_for_write(&self) -> Option<&[u8]> {
        if self.n_vectors == 0 || self.dim.is_none() {
            return None;
        }
        self.blocked.get().map(|c| c.data.as_slice())
    }

    /// The v6 file payload: codes in the arch-neutral sequential blocked
    /// layout. Cheap when the SIMD-blocked cache is warm (a per-block
    /// nibble de-interleave on x86, a copy elsewhere); otherwise the full
    /// O(n·dim) bit-plane repack — the same cost the pre-v6 format paid
    /// on every load instead of once per write.
    pub fn codes_blocked_seq(&self) -> Vec<u8> {
        let Some(dim) = self.dim else {
            return Vec::new();
        };
        if self.n_vectors == 0 {
            return Vec::new();
        }
        if let Some(cache) = self.blocked.get() {
            return pack::native_to_seq(&cache.data);
        }
        pack::repack_seq(self.packed(), self.n_vectors, self.bit_width, dim)
    }

    /// The codebook arrays the v6 file embeds — `(boundaries,
    /// centroids)`: the real (cached or freshly computed) Lloyd-Max
    /// codebook when the index has vectors, all-zero placeholders for an
    /// empty/lazy index (ignored on load). Pairs with
    /// [`Self::codes_blocked_seq`] for callers serializing through the
    /// raw [`io`] writers.
    pub fn codebook_for_write(&self) -> (Vec<f32>, Vec<f32>) {
        let n_levels = 1usize << self.bit_width;
        let Some(dim) = self.dim else {
            return (vec![0.0; n_levels - 1], vec![0.0; n_levels]);
        };
        if self.n_vectors == 0 {
            return (vec![0.0; n_levels - 1], vec![0.0; n_levels]);
        }
        // Solve once and seed both locks (mirrors `add`) — the cold
        // from_parts → write path would otherwise run the ~60 ms
        // Lloyd-Max solve twice.
        if self.boundaries.get().is_none() || self.centroids.get().is_none() {
            let (b, c) = codebook::codebook(self.bit_width, dim);
            let _ = self.boundaries.set(b);
            let _ = self.centroids.set(c);
        }
        let boundaries = self.boundaries.get().expect("boundaries just seeded");
        let centroids = self.centroids.get().expect("centroids just seeded");
        (boundaries.clone(), centroids.clone())
    }

    /// Serialize the index in the `.tv` byte format to any
    /// [`std::io::Write`] sink. Emits exactly the bytes [`Self::write`]
    /// would put in the file.
    ///
    /// Unlike [`Self::write`] there is no atomic-replace behaviour: the
    /// caller owns the sink.
    pub fn write_to_writer<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        let (boundaries, centroids) = self.codebook_for_write();
        // Off x86 the warm cache already holds the sequential layout the
        // format persists, so it is written straight from the cache; the
        // `codes_blocked_seq()` fallback below would copy it first. On x86
        // the native cache is perm0-nibble-interleaved and has to be
        // de-interleaved into a materialized buffer — a deliberate,
        // documented asymmetry (#409): streaming that transform chunk-wise
        // is what the file writer does, and it needs a positioned sink,
        // which a bare `Write` is not.
        #[cfg(not(target_arch = "x86_64"))]
        if let Some(native) = self.blocked_native_for_write() {
            return io::write_to(
                w,
                self.bit_width,
                self.dim.unwrap_or(0),
                self.n_vectors,
                native,
                &boundaries,
                &centroids,
                &self.scales,
                &self.tqplus_shift,
                &self.tqplus_scale,
            );
        }
        io::write_to(
            w,
            self.bit_width,
            self.dim.unwrap_or(0),
            self.n_vectors,
            &self.codes_blocked_seq(),
            &boundaries,
            &centroids,
            &self.scales,
            &self.tqplus_shift,
            &self.tqplus_scale,
        )
    }

    /// The exact number of bytes [`Self::to_bytes`] returns and
    /// [`Self::write`] puts in the file, computed from the index's
    /// geometry without serializing anything.
    ///
    /// Use it to size a buffer, a database column or a quota check before
    /// paying for the bytes. It is exact, not an estimate: `to_bytes()`
    /// always returns a `Vec` of precisely this length.
    pub fn serialized_len(&self) -> usize {
        // A still-lazy index writes no codes section. An empty one needs
        // no special case: zero vectors is zero blocks is zero bytes, and
        // `codebook_for_write` emits placeholder codebook arrays of the
        // same length the real ones would have. (Guarding `n_vectors > 0`
        // here would be redundant with `blocked_geometry`, which is worse
        // than merely untidy — it is a branch no test can distinguish, so
        // it reads as an uncovered mutant forever.)
        let codes_len = match self.dim {
            Some(dim) => pack::blocked_geometry(self.n_vectors, self.bit_width, dim).2,
            None => 0,
        };
        io::serialized_len(
            self.bit_width,
            codes_len,
            self.scales.len(),
            self.tqplus_shift.len(),
        )
    }

    /// Serialize the index to `.tv`-format bytes in memory —
    /// byte-identical to the file [`Self::write`] produces. Pairs with
    /// [`Self::from_bytes`] for callers that persist the index through
    /// their own storage (a database column, a cache, a pickle payload)
    /// instead of the filesystem.
    ///
    /// The round trip preserves the calibration exactly, so a
    /// clone-by-round-trip is byte-for-byte the index it was copied
    /// from.
    ///
    pub fn to_bytes(&self) -> Vec<u8> {
        // Sized exactly up front: growing from empty reallocates and
        // copies the whole payload log-many times, so peak live bytes
        // reached about three times the final size (#409).
        let mut buf = Vec::with_capacity(self.serialized_len());
        self.write_to_writer(&mut buf)
            .expect("writing to a Vec<u8> cannot fail");
        buf
    }

    /// Deserialize an index from any [`std::io::Read`] source of
    /// `.tv`-format bytes. Applies exactly the same validation as
    /// [`Self::load`] — version handling (v5 only), structural and
    /// value-level checks — so a byte stream and the file it came from
    /// load, or fail, identically.
    pub fn load_from_reader<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Self::from_loaded(io::load_from(r)?)
    }

    /// Deserialize an index from in-memory `.tv`-format bytes, as
    /// produced by [`Self::to_bytes`] (or read out of a `.tv` file).
    /// Same validation as [`Self::load`]; see
    /// [`Self::load_from_reader`].
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        Self::load_from_reader(&mut &bytes[..])
    }

    /// Load an index from a `.tv` file written by [`Self::write`].
    ///
    /// This is the crate's validation chokepoint for untrusted bytes and
    /// the definition [`Self::load_from_reader`] and [`Self::from_bytes`]
    /// defer to. A file is accepted only if its version is supported,
    /// every declared length agrees with the bytes actually present, and
    /// every float it carries is one the encoder could have emitted; a
    /// file that fails any of those is refused with an `Err` rather than
    /// producing an index that mis-scores. Corrupt input therefore
    /// surfaces here, not as a wrong answer from a later `search`.
    ///
    /// How much of the returned index is already materialized depends on
    /// the file's format version. A v6 file — what [`Self::write`] emits
    /// — stores the codebook and the blocked search layout, so a
    /// non-empty v6 load seeds both straight from the file and leaves
    /// only the rotation cold; the packed rows stay unmaterialized until
    /// something needs them ([`Self::packed_ready`] reports which
    /// encoding is present). A v5 file carries packed rows instead, so it
    /// loads fully cold and the search layout is built on first use, as
    /// does a v6 file holding no vectors (there is nothing to seed).
    ///
    /// [`Self::prepare`] does whatever remains up front instead of on the
    /// first [`Self::search`]. After a v6 load that is the rotation
    /// alone — not the O(n·dim) repack the v5 path still pays.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        if io_v7::is_v7(path.as_ref()) {
            return Self::load_v7(path.as_ref());
        }
        Self::from_loaded(io::load(path)?)
    }

    /// Shared tail of [`Self::load`] / [`Self::load_from_reader`]:
    /// assemble an index from an io-layer core payload. What gets seeded
    /// differs per arm. The v5 arm seeds nothing — a v5 file carries only
    /// the packed rows, and the rotation is deterministic and cheap to
    /// (re)build — so the three caches a search needs (`rotation`,
    /// `centroids`, `blocked`) fill lazily on first search. `boundaries`
    /// is encode-side: no search ever fills it, so a v5-loaded index
    /// that is only ever searched leaves it cold. The two
    /// v6 arms seed the codebook and the blocked search layout from the
    /// file, for any file holding at least one vector. The rotation is
    /// left cold on every path.
    pub(crate) fn from_loaded(
        parts: (usize, usize, usize, io::CodePayload, Vec<f32>, Vec<f32>, Vec<f32>),
    ) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale) = parts;
        let dim_opt = if dim == 0 { None } else { Some(dim) };
        match codes {
            // v5 file: packed rows, exactly the pre-v6 load path.
            io::CodePayload::Packed(packed_codes) => Self::from_parts(
                dim_opt,
                bit_width,
                n_vectors,
                packed_codes,
                scales,
                tqplus_shift,
                tqplus_scale,
            )
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())),
            // v6 file: seed the search cache directly from the blocked
            // payload (the whole point of the format — no O(n·dim)
            // first-search repack) and leave `packed_codes` to lazy
            // reconstruction. Validation: the io layer checked the
            // payload length against the header geometry; scales length
            // is checked here as from_parts would.
            io::CodePayload::BlockedNative { codes, boundaries, centroids } => {
                if scales.len() != n_vectors {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "scales length {} does not match n_vectors {n_vectors}",
                            scales.len()
                        ),
                    ));
                }
                let blocked = OnceLock::new();
                let boundaries_lock = OnceLock::new();
                let centroids_lock = OnceLock::new();
                if let Some(d) = dim_opt {
                    if n_vectors > 0 {
                        let (n_blocks, _, _) = pack::blocked_geometry(n_vectors, bit_width, d);
                        // Already the native kernel layout — no transform.
                        let _ = blocked.set(BlockedCache { data: codes, n_blocks });
                        let _ = boundaries_lock.set(boundaries);
                        let _ = centroids_lock.set(centroids);
                    }
                }
                let packed_codes = if n_vectors == 0 {
                    OnceLock::from(Vec::new())
                } else {
                    OnceLock::new()
                };
                // Same normalization `from_parts` applies, so every
                // construction path lands in the same calibration state.
                let (tqplus_shift, tqplus_scale) =
                    Self::normalize_calibration(tqplus_shift, tqplus_scale);
                Ok(Self {
                    dim: dim_opt,
                    bit_width,
                    n_vectors,
                    packed_codes,
                    scales,
                    tqplus_shift,
                    tqplus_scale,
                    encode_scratch: Vec::new(),
                    encode_scratch_prev: 0,
            sync_cursor: None,
            sync_path: None,
            sync_pending: std::collections::HashSet::new(),
            sync_fresh: std::collections::HashSet::new(),
            sync_capture_buf: Vec::new(),
            sync_capture_at: Vec::new(),
            calib_gen: 0,
                    rotation: OnceLock::new(),
                    boundaries: boundaries_lock,
                    centroids: centroids_lock,
                    blocked,
                })
            }
            io::CodePayload::BlockedSeq { codes: seq, boundaries, centroids } => {
                if scales.len() != n_vectors {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "scales length {} does not match n_vectors {n_vectors}",
                            scales.len()
                        ),
                    ));
                }
                let blocked = OnceLock::new();
                let boundaries_lock = OnceLock::new();
                let centroids_lock = OnceLock::new();
                if let Some(d) = dim_opt {
                    if n_vectors > 0 {
                        let (n_blocks, _, _) = pack::blocked_geometry(n_vectors, bit_width, d);
                        let data = pack::seq_into_native(seq);
                        let _ = blocked.set(BlockedCache { data, n_blocks });
                        // Seed the codebook from the file — the second
                        // half of skipping the first-search rebuild (the
                        // Lloyd-Max solve is ~60 ms at dim 768).
                        let _ = boundaries_lock.set(boundaries);
                        let _ = centroids_lock.set(centroids);
                    }
                }
                let packed_codes = if n_vectors == 0 {
                    OnceLock::from(Vec::new())
                } else {
                    OnceLock::new()
                };
                // Same normalization `from_parts` applies, so every
                // construction path lands in the same calibration state.
                let (tqplus_shift, tqplus_scale) =
                    Self::normalize_calibration(tqplus_shift, tqplus_scale);
                Ok(Self {
                    dim: dim_opt,
                    bit_width,
                    n_vectors,
                    packed_codes,
                    scales,
                    tqplus_shift,
                    tqplus_scale,
                    encode_scratch: Vec::new(),
                    encode_scratch_prev: 0,
            sync_cursor: None,
            sync_path: None,
            sync_pending: std::collections::HashSet::new(),
            sync_fresh: std::collections::HashSet::new(),
            sync_capture_buf: Vec::new(),
            sync_capture_at: Vec::new(),
            calib_gen: 0,
                    rotation: OnceLock::new(),
                    boundaries: boundaries_lock,
                    centroids: centroids_lock,
                    blocked,
                })
            }
        }
    }

    /// Normalize the decoded `(tqplus_shift, tqplus_scale)` every
    /// construction path must agree on.
    ///
    /// One rule: **a pair that applies no transform is stored as no
    /// pair.** A payload can carry the calibration as an empty pair (the
    /// v2 wire shape, and pre-TQ+ files) or as an explicit length-`dim`
    /// identity; both describe an uncalibrated index, and collapsing
    /// them here means [`CalibrationState`] and every byte the index
    /// writes back are functions of the calibration itself rather than
    /// of which spelling the file happened to use (#418).
    ///
    /// Nothing else needs normalizing any more. The old second rule —
    /// "stored rows always come with a declared calibration", which
    /// filled an explicit identity in beside `n_vectors > 0` — existed
    /// because an empty pair was `encode`'s signal to *fit* one, so a
    /// v2-loaded index that took an add would fit, encode into the fit,
    /// and drop it (#303). `encode` no longer fits under any
    /// circumstances: an empty pair means identity and stays empty, and
    /// an add to a loaded uncalibrated index encodes exactly as its
    /// existing rows did.
    fn normalize_calibration(
        tqplus_shift: Vec<f32>,
        tqplus_scale: Vec<f32>,
    ) -> (Vec<f32>, Vec<f32>) {
        let declares_nothing = tqplus_shift.iter().all(|&x| x == 0.0)
            && tqplus_scale.iter().all(|&x| x == 1.0);
        if declares_nothing {
            return (Vec::new(), Vec::new());
        }
        (tqplus_shift, tqplus_scale)
    }

    /// Construct an index directly from already-decoded fields, validating
    /// every structural invariant at this single chokepoint.
    ///
    /// This is the low-level construction path for embedders that hold the
    /// index payload in memory (e.g. read out of a database page or a
    /// `bytea` column) and want to skip the `.tv`/`.tvim` file round-trip.
    /// It is the only validated way to build an index from raw parts: the
    /// per-module kernels (`encode`, `pack`, `search`, `codebook`) are
    /// crate-internal precisely because they trust their caller's
    /// invariants, whereas `from_parts` checks them and returns a named
    /// [`FromPartsError`] for any violation instead of panicking, reading
    /// out of bounds, or producing a silently-wrong index.
    ///
    /// Pair it with the [`bit_width`](Self::bit_width),
    /// [`dim_opt`](Self::dim_opt), [`len`](Self::len),
    /// [`packed_codes`](Self::packed_codes), [`scales`](Self::scales),
    /// [`tqplus_shift`](Self::tqplus_shift) and
    /// [`tqplus_scale`](Self::tqplus_scale) accessors on an existing index
    /// to round-trip an index through your own storage format.
    ///
    /// # Arguments
    ///
    /// - `dim`: `Some(d)` for a committed index (`d` must be a positive
    ///   multiple of 8, `<= `[`MAX_DIM`]); `None` for a lazy,
    ///   never-added index whose dim is not yet known.
    /// - `bit_width`: bits per coordinate, one of `{2, 3, 4}`.
    /// - `n_vectors`: number of stored vectors.
    /// - `packed_codes`: bit-plane packed codes.
    /// - `scales`: per-vector correction scale.
    /// - `tqplus_shift` / `tqplus_scale`: TQ+ per-coordinate calibration,
    ///   both length `dim` or both empty (empty = identity, the v2-file
    ///   shape).
    ///
    /// # Checked invariants
    ///
    /// Every one of these maps to a [`FromPartsError`] variant:
    ///
    /// - `bit_width` in `{2, 3, 4}`
    ///   ([`BitWidthOutOfRange`](FromPartsError::BitWidthOutOfRange)).
    /// - committed `dim` is a positive multiple of 8
    ///   ([`DimNotPositiveMultipleOf8`](FromPartsError::DimNotPositiveMultipleOf8))
    ///   and `<= `[`MAX_DIM`]
    ///   ([`DimTooLarge`](FromPartsError::DimTooLarge)).
    /// - `packed_codes.len() == n_vectors * dim * bit_width / 8`
    ///   ([`PackedCodesLengthMismatch`](FromPartsError::PackedCodesLengthMismatch)).
    /// - `scales.len() == n_vectors`
    ///   ([`ScalesLengthMismatch`](FromPartsError::ScalesLengthMismatch)).
    /// - `tqplus_shift.len() == tqplus_scale.len()`
    ///   ([`TqplusLengthMismatch`](FromPartsError::TqplusLengthMismatch)).
    /// - a non-empty TQ+ array has length `dim`
    ///   ([`TqplusLengthNotDim`](FromPartsError::TqplusLengthNotDim)).
    /// - a lazy (`dim == None`) index has `n_vectors == 0` and every
    ///   storage field empty
    ///   ([`LazyMustHaveZeroVectors`](FromPartsError::LazyMustHaveZeroVectors)
    ///   and siblings).
    /// - the implied packed size `n_vectors * dim * bit_width / 8` does not
    ///   overflow `usize` — computed with checked arithmetic
    ///   ([`PackedCodesSizeOverflow`](FromPartsError::PackedCodesSizeOverflow)).
    /// - every per-vector scale is finite and non-negative
    ///   ([`InvalidScaleValue`](FromPartsError::InvalidScaleValue)).
    /// - every TQ+ shift is finite
    ///   ([`InvalidTqplusShiftValue`](FromPartsError::InvalidTqplusShiftValue))
    ///   and every TQ+ scale is finite and `> 0`
    ///   ([`InvalidTqplusScaleValue`](FromPartsError::InvalidTqplusScaleValue)).
    ///
    /// The value checks exactly mirror the `.tv`/`.tvim` loader's, so an
    /// index accepted by `from_parts` always survives its own
    /// [`write`](Self::write) → [`load`](Self::load) round-trip.
    ///
    /// Validating `bit_width` and `dim` here also transitively bounds the
    /// lazily-built codebook (`codebook(bit_width, dim)`) and rotation
    /// matrix, so a constructed index can never drive the unbounded
    /// codebook allocation that a raw `bit_width`/`dim` could.
    ///
    /// # Example
    ///
    /// ```
    /// use turbovec::TurboQuantIndex;
    ///
    /// // Build an index normally, then reconstruct it from its raw parts
    /// // — the shape an embedder reads out of its own storage.
    /// let mut src = TurboQuantIndex::new(64, 4).unwrap();
    /// src.add(&vec![0.1f32; 64 * 8]);
    ///
    /// let rebuilt = TurboQuantIndex::from_parts(
    ///     src.dim_opt(),
    ///     src.bit_width(),
    ///     src.len(),
    ///     src.packed_codes().to_vec(),
    ///     src.scales().to_vec(),
    ///     src.tqplus_shift().to_vec(),
    ///     src.tqplus_scale().to_vec(),
    /// )
    /// .expect("consistent parts");
    /// assert_eq!(rebuilt.len(), src.len());
    /// ```
    pub fn from_parts(
        dim: Option<usize>,
        bit_width: usize,
        n_vectors: usize,
        packed_codes: Vec<u8>,
        scales: Vec<f32>,
        tqplus_shift: Vec<f32>,
        tqplus_scale: Vec<f32>,
    ) -> Result<Self, FromPartsError> {
        // bit_width gates the codebook level count (`1 << bit_width`); a
        // value outside {2,3,4} is both meaningless and — via the raw
        // codebook — an unbounded-allocation hazard. Check it first.
        if !(2..=4).contains(&bit_width) {
            return Err(FromPartsError::BitWidthOutOfRange(bit_width));
        }
        // The two TQ+ arrays are compared regardless of dim state.
        if tqplus_shift.len() != tqplus_scale.len() {
            return Err(FromPartsError::TqplusLengthMismatch {
                shift_len: tqplus_shift.len(),
                scale_len: tqplus_scale.len(),
            });
        }
        match dim {
            Some(d) => {
                // dim bounds the codebook and the rotation;
                // it must be a positive multiple of 8 (the packed layout
                // allocates dim/8 bytes per bit-plane) and within MAX_DIM.
                if d == 0 || d % 8 != 0 {
                    return Err(FromPartsError::DimNotPositiveMultipleOf8(d));
                }
                if d > MAX_DIM {
                    return Err(FromPartsError::DimTooLarge { dim: d, max: MAX_DIM });
                }
                // Checked arithmetic, mirroring io::read_header_codes_scales:
                // `n_vectors` is caller-controlled, so the product can
                // overflow `usize` — a debug-panic / release-wrap that would
                // break the returns-named-error contract and neuter the
                // length check. `d % 8 == 0` is already established, so
                // `(d / 8) * bit_width * n_vectors == n_vectors*d*bit_width/8`.
                let expected_packed = (d / 8)
                    .checked_mul(bit_width)
                    .and_then(|x| x.checked_mul(n_vectors))
                    .ok_or(FromPartsError::PackedCodesSizeOverflow {
                        n_vectors,
                        dim: d,
                        bit_width,
                    })?;
                if packed_codes.len() != expected_packed {
                    return Err(FromPartsError::PackedCodesLengthMismatch {
                        expected: expected_packed,
                        got: packed_codes.len(),
                    });
                }
                if scales.len() != n_vectors {
                    return Err(FromPartsError::ScalesLengthMismatch {
                        expected: n_vectors,
                        got: scales.len(),
                    });
                }
                if !tqplus_shift.is_empty() && tqplus_shift.len() != d {
                    return Err(FromPartsError::TqplusLengthNotDim {
                        got: tqplus_shift.len(),
                        dim: d,
                    });
                }
            }
            None => {
                // Lazy uncommitted state — every storage field must be empty.
                if n_vectors != 0 {
                    return Err(FromPartsError::LazyMustHaveZeroVectors(n_vectors));
                }
                if !packed_codes.is_empty() {
                    return Err(FromPartsError::LazyMustHaveEmptyPackedCodes(
                        packed_codes.len(),
                    ));
                }
                if !scales.is_empty() {
                    return Err(FromPartsError::LazyMustHaveEmptyScales(scales.len()));
                }
                if !tqplus_shift.is_empty() {
                    return Err(FromPartsError::LazyMustHaveEmptyTqplus(tqplus_shift.len()));
                }
            }
        }

        // Value-level validation, exactly mirroring io::load's checks: the
        // encoder only ever emits finite non-negative per-vector scales,
        // finite TQ+ shifts, and finite strictly-positive TQ+ scales.
        // Anything else silently corrupts search (an Inf scale wins every
        // top-1, a NaN slot vanishes; search divides by tqplus_scale) —
        // and, because the loader rejects such values, an index accepted
        // here would otherwise fail to load its own written file. Keeping
        // parity guarantees a from_parts-accepted index always survives
        // its write → load round-trip. (Lazy inputs have empty arrays, so
        // these loops are no-ops there.)
        if let Some((i, &s)) = scales
            .iter()
            .enumerate()
            .find(|(_, s)| !s.is_finite() || **s < 0.0)
        {
            return Err(FromPartsError::InvalidScaleValue { slot: i, value: s });
        }
        if let Some((i, &v)) = tqplus_shift
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            return Err(FromPartsError::InvalidTqplusShiftValue { coord: i, value: v });
        }
        if let Some((i, &v)) = tqplus_scale
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite() || **v <= 0.0)
        {
            return Err(FromPartsError::InvalidTqplusScaleValue { coord: i, value: v });
        }

        // See `normalize_calibration`. Shared with the v6 load arms so
        // every construction path lands in the same calibration state.
        let (tqplus_shift, tqplus_scale) =
            Self::normalize_calibration(tqplus_shift, tqplus_scale);
        Ok(Self {
            dim,
            bit_width,
            n_vectors,
            packed_codes: OnceLock::from(packed_codes),
            scales,
            tqplus_shift,
            tqplus_scale,
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
            encode_scratch: Vec::new(),
            encode_scratch_prev: 0,
            sync_cursor: None,
            sync_path: None,
            sync_pending: std::collections::HashSet::new(),
            sync_fresh: std::collections::HashSet::new(),
            sync_capture_buf: Vec::new(),
            sync_capture_at: Vec::new(),
            calib_gen: 0,
        })
    }

    /// Bit-plane packed codes backing this index. Pairs with
    /// [`Self::from_parts`] to round-trip an index through external storage.
    ///
    /// After a v6 [`Self::load`] the packed rows are reconstructed from
    /// the loaded blocked layout on the first call (O(n·dim)); every
    /// other path — and every subsequent call — is O(1).
    pub fn packed_codes(&self) -> &[u8] {
        self.packed()
    }

    /// Per-vector correction scales. Pairs with [`Self::from_parts`].
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// TQ+ per-coordinate shift calibration (length `dim`, or empty for a
    /// v2/identity index). Pairs with [`Self::from_parts`].
    pub fn tqplus_shift(&self) -> &[f32] {
        &self.tqplus_shift
    }

    /// TQ+ per-coordinate scale calibration (length `dim`, or empty for a
    /// v2/identity index). Pairs with [`Self::from_parts`].
    pub fn tqplus_scale(&self) -> &[f32] {
        &self.tqplus_scale
    }

    /// Fit the TQ+ calibration from a caller-supplied sample.
    ///
    /// `sample` is `rows * dim` coordinates, row-major. The fit uses
    /// every row given. Recall tracks how well the sample represents the
    /// population the index will hold: ~1024 uniformly random rows match
    /// a fit on the entire corpus to within measurement noise, while the
    /// same count drawn as a prefix of a sorted corpus destroys recall.
    /// **Choosing a representative sample is the caller's
    /// responsibility** — nothing here can tell a random draw from a
    /// biased one.
    ///
    /// May be called at any time and repeatedly. On an empty index it
    /// simply commits the fitted pair. On a populated index it also
    /// re-encodes every stored row into the new coordinate system,
    /// reconstructing each row from its stored codes — the float32
    /// originals are not required. That re-encode is lossy in principle
    /// (a second quantization), in practice: at 2 and 3 bits the
    /// reconstructed codes are bit-identical to the originals whenever
    /// the old and new calibrations are close; at 4 bits repeated
    /// refits cost roughly measurable fractions of a recall point each.
    /// Refitting with a pair equal to the committed one reproduces the
    /// stored codes exactly.
    ///
    /// Lazy indexes ([`Self::new_lazy`]) are supported: a successful
    /// call locks `dim`, exactly as a first add would.
    ///
    /// After a successful call the index reports
    /// [`CalibrationState::Calibrated`]; every later add encodes under
    /// the committed pair, and it round-trips through
    /// [`Self::to_bytes`]/[`Self::from_bytes`] and the file formats.
    ///
    /// # Errors
    ///
    /// See [`CalibrateError`]. Every error is raised before the index is
    /// touched, so a rejected call changes nothing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dim = 128;
    /// # let corpus: Vec<f32> = (0..dim * 100_000).map(|i| (i as f32).cos()).collect();
    /// # let sample: Vec<f32> = (0..dim * 1024).map(|i| (i as f32).sin()).collect();
    /// let mut index = turbovec::TurboQuantIndex::new(dim, 4)?;
    /// // `sample` is a uniform random draw of rows from `corpus`.
    /// index.calibrate_2d(&sample, dim)?;
    /// index.add_2d(&corpus, dim)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn calibrate_2d(&mut self, sample: &[f32], dim: usize) -> Result<(), CalibrateError> {
        // Dim checks first: every later check is expressed in terms of
        // `dim`.
        match self.dim {
            Some(existing) if existing != dim => {
                return Err(CalibrateError::DimMismatch { existing, got: dim });
            }
            Some(_) => {}
            None => {
                if dim == 0 {
                    return Err(CalibrateError::ZeroDim);
                }
                if dim % 8 != 0 {
                    return Err(CalibrateError::DimNotMultipleOf8(dim));
                }
                if dim > MAX_DIM {
                    return Err(CalibrateError::DimTooLarge { dim, max: MAX_DIM });
                }
            }
        }
        if sample.len() % dim != 0 {
            return Err(CalibrateError::SampleBufferNotMultipleOfDim {
                sample_len: sample.len(),
                dim,
            });
        }
        let n = sample.len() / dim;
        if n < MIN_CALIBRATION_ROWS {
            return Err(CalibrateError::SampleTooSmall {
                rows: n,
                min: MIN_CALIBRATION_ROWS,
            });
        }
        if let Some((vi, ci, v)) = first_invalid_coord(sample, dim) {
            return Err(CalibrateError::InvalidInputValue {
                vector_index: vi,
                coord_index: ci,
                value: v,
            });
        }

        // Everything below is fallible-then-commit: the fit and the
        // re-encode both write only into locals, so an unwind anywhere
        // before the commit block leaves the index exactly as it was.
        let rotation = self.rotation.get_or_init(|| rotation::Rotation::new(dim));
        if self.boundaries.get().is_none() || self.centroids.get().is_none() {
            let (b, c) = codebook::codebook(self.bit_width, dim);
            let _ = self.boundaries.set(b);
            let _ = self.centroids.set(c);
        }
        let centroids = self.centroids.get().expect("centroids seeded above");
        let mut scratch = std::mem::take(&mut self.encode_scratch);
        let fitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[cfg(test)]
            if FORCE_FIT_PANIC.with(|f| f.replace(false)) {
                panic!("forced fit panic (test)");
            }
            encode::fit_calibration(sample, n, dim, rotation, centroids, &mut scratch)
        }));
        self.encode_scratch_prev = retain_scratch(&mut scratch, self.encode_scratch_prev, n * dim);
        self.encode_scratch = scratch;
        let (shift, scale_tq) = match fitted {
            Ok(pair) => pair,
            Err(panic) => {
                // The caches seeded above are pure functions of
                // `(dim, bit_width)` and `dim` is not committed yet, so
                // reset them for the same reason `add_2d`'s lazy path
                // does: a retry at a different dim must not reuse them.
                if self.dim.is_none() {
                    self.rotation = OnceLock::new();
                    self.boundaries = OnceLock::new();
                    self.centroids = OnceLock::new();
                }
                std::panic::resume_unwind(panic);
            }
        };
        debug_assert_eq!(shift.len(), dim, "fit returns a full-length pair");
        // A degenerate sample — no per-coordinate spread anywhere — fits
        // exact identity. Committing it would report `Calibrated` while
        // behaving as `Uncalibrated`, and the pair would not even
        // round-trip (serialization canonicalizes identity to the empty
        // representation). A typed rejection, not a debug assert: the
        // condition is reachable through perfectly valid input (all-equal
        // rows pass every check above), and nothing has been committed
        // yet, so refusing here preserves the rejected-calls-are-no-ops
        // contract.
        if shift.iter().all(|&x| x == 0.0) && scale_tq.iter().all(|&x| x == 1.0) {
            if self.dim.is_none() {
                // Same cache-reset rule as the fit-panic arm above: dim
                // is not committed, so a retry at another dim must not
                // reuse these.
                self.rotation = OnceLock::new();
                self.boundaries = OnceLock::new();
                self.centroids = OnceLock::new();
            }
            return Err(CalibrateError::DegenerateSample);
        }

        // Refit: re-encode every stored row under the new pair. Reads
        // `self` and writes locals only, so it commits nothing on its
        // own; `n_vectors > 0` implies `self.dim` is already `Some(dim)`
        // (checked above), so the geometry below is the committed one.
        let reencoded = (self.n_vectors > 0).then(|| self.reencode_stored_rows(dim, (&shift, &scale_tq)));

        // Commit — nothing below can fail.
        self.dim = Some(dim);
        if let Some((packed, scales)) = reencoded {
            self.packed_codes = OnceLock::from(packed);
            self.scales = scales;
            // The SIMD-blocked cache mirrors the packed codes; every
            // byte of it is stale now.
            self.blocked = OnceLock::new();
        }
        self.tqplus_shift = shift;
        self.tqplus_scale = scale_tq;
        // Every commit re-encodes (or newly governs) the stored codes,
        // so a synced file's segments are stale: force the next sync to
        // compact.
        self.calib_gen += 1;
        // Re-encoding changes every row's bytes, so every captured row now
        // describes the old encoding.
        self.sync_capture_buf.clear();
        self.sync_capture_at.clear();
        Ok(())
    }

    /// [`Self::calibrate_2d`] for an index whose `dim` is already known
    /// (constructed via [`Self::new`], or already added to).
    ///
    /// # Panics
    ///
    /// Panics if the index has no committed dim (a [`Self::new_lazy`]
    /// index that has never been added to or calibrated). Use
    /// [`Self::calibrate_2d`], which carries the dim, in that case.
    pub fn calibrate(&mut self, sample: &[f32]) -> Result<(), CalibrateError> {
        let dim = self.dim.expect(
            "TurboQuantIndex dim is not set; use calibrate_2d(sample, dim) on a \
             lazy index or construct via TurboQuantIndex::new(dim, bit_width)",
        );
        self.calibrate_2d(sample, dim)
    }

    /// Re-encode every stored row under `new_pair`, reconstructing each
    /// row's rotated form from its stored codes and the committed
    /// calibration. Returns fresh `(packed_codes, scales)` buffers; the
    /// caller commits them. Reads `self` only.
    ///
    /// The per-vector scale is recomputed, not carried over: the stored
    /// scale is `||v|| / <u_rot, x_hat_old>`, and with the true rotated
    /// row unavailable the reconstruction stands in for it, giving
    /// `scale_new = scale_old * <x_hat_old, x_hat_old> / <x_hat_old,
    /// x_hat_new>`. Feeding `encode_prerotated` the reconstruction as
    /// the row and `scale_old * <x_hat_old, x_hat_old>` as its norm
    /// produces exactly that, in the same kernel a fresh add uses — so
    /// a refit under the committed pair reproduces the stored codes
    /// bit-identically (re-quantizing an exact centroid value lands in
    /// its own cell), which is what pins the decode below to the pack
    /// layout.
    ///
    /// Degenerate rows (stored scale `0.0`) stay degenerate: their
    /// effective norm is `0.0`, so the kernel stores `0.0` again.
    fn reencode_stored_rows(&self, dim: usize, new_pair: (&[f32], &[f32])) -> (Vec<u8>, Vec<f32>) {
        let n = self.n_vectors;
        let bits = self.bit_width;
        let bytes_per_row = bits * (dim / 8);
        let packed = self.packed();
        let boundaries = self.boundaries.get().expect("populated index has a codebook");
        let centroids = self.centroids.get().expect("populated index has a codebook");
        // The committed pair the stored codes decode under; an
        // uncalibrated index is arithmetically the identity pair.
        let identity;
        let (old_shift, old_inv): (&[f32], Vec<f32>) = if self.tqplus_shift.is_empty() {
            identity = vec![0.0f32; dim];
            (&identity, vec![1.0f32; dim])
        } else {
            (
                self.tqplus_shift.as_slice(),
                self.tqplus_scale.iter().map(|s| 1.0 / s).collect(),
            )
        };
        // Decode geometry, mirroring `build_extract_lut`: coordinate `d`
        // lives in group byte `d / cpb` at field offset
        // `(cpb - 1 - d % cpb) * field` (3-bit codes occupy 4-bit
        // fields).
        let cpb = 8 / bits;
        let field = if bits == 3 { 4 } else { bits };
        let mask = (1u8 << bits) - 1;
        let n_byte_groups = dim / cpb;

        let mut new_packed = Vec::with_capacity(n * bytes_per_row);
        let mut new_scales = Vec::with_capacity(n);
        // Chunked so the transient float reconstruction stays bounded
        // (~24 MB at dim 1536) however large the index is.
        const REENCODE_CHUNK_ROWS: usize = 4096;
        let chunk_rows = REENCODE_CHUNK_ROWS.min(n);
        let mut recon = vec![0.0f32; chunk_rows * dim];
        let mut norms = vec![0.0f32; chunk_rows];
        let mut start = 0usize;
        while start < n {
            let rows = chunk_rows.min(n - start);
            let codes_flat = pack::extract_codes_flat(
                &packed[start * bytes_per_row..(start + rows) * bytes_per_row],
                rows,
                bits,
                dim,
            );
            for i in 0..rows {
                let row = &codes_flat[i * n_byte_groups..(i + 1) * n_byte_groups];
                let out = &mut recon[i * dim..(i + 1) * dim];
                let mut sumsq = 0.0f64;
                for (d, slot) in out.iter_mut().enumerate() {
                    let code = ((row[d / cpb] >> ((cpb - 1 - d % cpb) * field)) & mask) as usize;
                    let x = centroids[code] * old_inv[d] - old_shift[d];
                    *slot = x;
                    sumsq += f64::from(x) * f64::from(x);
                }
                norms[i] = (f64::from(self.scales[start + i]) * sumsq) as f32;
            }
            encode::encode_prerotated(
                &recon[..rows * dim],
                &norms[..rows],
                rows,
                dim,
                boundaries,
                centroids,
                bits,
                Some(new_pair),
                &mut new_packed,
                &mut new_scales,
            );
            start += rows;
        }
        (new_packed, new_scales)
    }

    /// Whether this index has a TQ+ calibration committed. See
    /// [`CalibrationState`].
    pub fn calibration_state(&self) -> CalibrationState {
        // `normalize_calibration` collapses a no-op pair to an empty
        // one on every construction path, and `calibrate` refuses to
        // commit one, so emptiness is the whole test.
        if self.tqplus_shift.is_empty() {
            CalibrationState::Uncalibrated
        } else {
            CalibrationState::Calibrated
        }
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
    ///
    /// # Panics
    ///
    /// Panics if `idx >= len()`, including on an empty index where every
    /// `idx` is out of bounds. A slot index is caller-held state, not
    /// external input, so an out-of-range one is a contract violation
    /// rather than something to report.
    pub fn swap_remove(&mut self, idx: usize) -> usize {
        #[cfg(test)]
        if FORCE_SWAP_REMOVE_PANIC.with(|f| f.replace(false)) {
            panic!("forced swap_remove panic (test)");
        }
        assert!(
            idx < self.n_vectors,
            "index {idx} out of bounds (n_vectors = {})",
            self.n_vectors
        );
        // Divergence marking for the sync journal: a removal diverges up
        // to two slots — the hole (`idx`, which takes the filler's bytes
        // or dies as a pop) and the moved-from slot (`last`, which goes
        // dead but may be refilled by a later add). Slots stay marked
        // until a sync materializes their unit.
        if self.sync_cursor.is_some() {
            self.mark_dirty(idx);
            let last = self.n_vectors - 1;
            if last != idx {
                self.mark_dirty(last);
            }
        }
        // Capture slot `idx`'s incoming bytes only where they will actually
        // be serialized as a redo op: the slot must be one this file has
        // committed (the same test `mark_dirty` applies), the blocked cache
        // must be the authority (with `packed_codes` live, header assembly
        // reads the contiguous packed row instead and the capture would be
        // dead weight), and the map stays under the header's op cap — past
        // it the sync rewrites the file whole and needs no ops at all.
        // x86 only, and not as a portability hedge — as arithmetic. The
        // capture adds one store per byte group to the lane move. On x86 the
        // move is a de-interleave plus a nibble-merge write, so one more
        // store is a small fraction of it and the sync saves far more than
        // the removals pay. Off x86 the move IS a byte load and a byte
        // store, so capturing is a third memory op on a two-op loop — a ~50%
        // tax on `remove()` to save less than that in `sync()`. Measured
        // both ways (H5-H7): the ARM side never nets out.
        let capture_this = cfg!(target_arch = "x86_64")
            && self.sync_cursor.is_some()
            && idx < self.sync_watermark()
            && self.packed_codes.get().is_none()
            && self.sync_capture_at.len() < io_v7::MAX_OPS;
        let capture_off = self.sync_capture_buf.len();


        // n_vectors > 0 (asserted above) implies a successful add, which
        // implies self.dim was committed at that point. Unwrap is safe.
        let dim = self.dim.expect("n_vectors > 0 but dim is None");
        let bytes_per_vec = dim * self.bit_width / 8;
        let last = self.n_vectors - 1;
        // At least one code representation must exist, or the branches
        // below would silently update neither and corrupt the index.
        // Every current path guarantees this (constructors and adds set
        // packed; v6 loads seed blocked); this makes a future violation
        // loud instead of silent.
        debug_assert!(
            self.packed_codes.get().is_some() || self.blocked.get().is_some(),
            "swap_remove: neither packed_codes nor the blocked cache is present"
        );

        // Maintain packed rows only if they are materialized. In the
        // v6-load window (blocked seeded from the file, packed unset) the
        // blocked cache is authoritative: leave the OnceLock empty and the
        // lazy rebuild reconstructs post-removal packed on demand — a
        // remove no longer forces the O(n·dim) materialization.
        if self.packed_codes.get().is_some() {
            if idx != last {
                let src = last * bytes_per_vec;
                let dst = idx * bytes_per_vec;
                self.packed_mut().copy_within(src..src + bytes_per_vec, dst);
            }
            self.packed_mut().truncate(last * bytes_per_vec);
        }

        if idx != last {
            // Move last norm into slot `idx`.
            self.scales[idx] = self.scales[last];
        }
        self.scales.truncate(last);
        self.n_vectors -= 1;

        // Maintain the blocked cache with O(dim) lane ops: copy the last
        // vector's lane into the vacated slot, zero the vacated last lane
        // (serialization copies the cache verbatim — a stale lane would
        // break byte determinism), then truncate to the new geometry.
        // Borrowed alongside the blocked cache: both are fields of `self`,
        // so the two mutable borrows are disjoint.
        let capture_buf = &mut self.sync_capture_buf;
        if let Some(cache) = self.blocked.get_mut() {
            let (new_n_blocks, n_byte_groups, _) =
                pack::blocked_geometry(self.n_vectors, self.bit_width, dim);
            let block_bytes = n_byte_groups * BLOCK;
            if idx != last {
                // The move already computes slot `idx`'s new code bytes; keep
                // them when this removal will be serialized as a redo op, so
                // the sync does not walk the block again to recover them.
                // Written straight into the arena — `cache` and the capture
                // buffer are separate fields, so both borrows stand.
                let dst = if capture_this {
                    let off = capture_buf.len();
                    capture_buf.resize(off + n_byte_groups, 0);
                    Some(&mut capture_buf[off..])
                } else {
                    None
                };
                pack::move_lane(&mut cache.data, n_byte_groups, last, idx, dst);
            }
            pack::zero_lane(&mut cache.data, n_byte_groups, last);
            cache.data.truncate(new_n_blocks * block_bytes);
            cache.n_blocks = new_n_blocks;
        }
        // No entry is retired here: `last` is now beyond `n_vectors`, and the
        // only readers — the header's tail rows and the plan's live op slots
        // — are both below it, so an entry left for a retired slot is
        // unreachable rather than wrong. The lookup is built in capture
        // order, so a slot captured twice resolves to its later bytes.
        if capture_this && idx != last {
            self.sync_capture_at.push((idx as u32, capture_off as u32));
        }

        last
    }

    /// Number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Whether the index holds no vectors. Equivalent to `len() == 0`.
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Vector dimensionality, or `0` for a lazy index that hasn't seen an
    /// add yet.
    ///
    /// **Deprecated — prefer [`Self::dim_opt`].** The `0` is only safe for
    /// comparisons, and callers do arithmetic with a dim: `buf.len() /
    /// idx.dim()` divides by zero and `vec![0.0; idx.dim()]` silently
    /// yields a zero-length buffer (#318). `dim_opt` makes the
    /// uncommitted case impossible to ignore.
    #[deprecated(
        since = "0.10.0",
        note = "returns 0 for a lazy index, which is unsafe to do arithmetic with; use dim_opt()"
    )]
    pub fn dim(&self) -> usize {
        self.dim.unwrap_or(0)
    }

    /// Vector dimensionality as an [`Option`], where `None` means the
    /// index is lazy and hasn't been committed to a dim yet.
    pub fn dim_opt(&self) -> Option<usize> {
        self.dim
    }

    /// Bits per coordinate (2, 3 or 4). Fixed at construction; never
    /// changes over the life of the index.
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }
}

#[cfg(test)]
mod scratch_retention_tests {
    //! The encode scratch is private derived state, so its retention can
    //! only be pinned from inside the crate (#333). These drive the real
    //! `add_2d` path and read `encode_scratch.capacity()` afterwards.

    use super::TurboQuantIndex;

    const DIM: usize = 256;

    fn rows(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim)
            .map(|i| ((i % 97) as f32 / 97.0) - 0.5)
            .collect()
    }

    /// A one-shot bulk add must not pin its rotated-batch buffer for the
    /// index's lifetime.
    #[test]
    fn one_shot_bulk_add_releases_the_encode_scratch() {
        let n = 24_000;
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.add_2d(&rows(n, DIM), DIM).unwrap();
        assert_eq!(idx.len(), n);
        assert!(
            idx.encode_scratch.capacity() < n * DIM / 4,
            "one-shot bulk add retained {} scratch elements (batch was {})",
            idx.encode_scratch.capacity(),
            n * DIM,
        );
    }

    /// A workload that keeps asking for the same size must keep its warm
    /// buffer, or the release above becomes a realloc on every add.
    #[test]
    fn repeated_same_size_adds_keep_the_scratch_warm() {
        let n = 24_000;
        let batch = rows(n, DIM);
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        for _ in 0..3 {
            idx.add_2d(&batch, DIM).unwrap();
        }
        assert_eq!(idx.len(), 3 * n);
        assert!(
            idx.encode_scratch.capacity() >= n * DIM,
            "steady same-size adds dropped the warm scratch to {} elements (need {})",
            idx.encode_scratch.capacity(),
            n * DIM,
        );
    }

    /// The regression that shrinking to the bare previous demand causes:
    /// `shrink_to` sets capacity *exactly*, so a batch even slightly
    /// larger than the last one finds no headroom and has to grow — then
    /// gets shrunk right back, on every add, forever. The buffer must
    /// stay at or above what the most recent add needed.
    #[test]
    fn growing_batch_sizes_keep_their_growth_headroom() {
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        let mut n = 8_000;
        let mut last = 0;
        for _ in 0..6 {
            idx.add_2d(&rows(n, DIM), DIM).unwrap();
            last = n;
            n += n / 20; // +5% per batch
        }
        assert!(
            idx.encode_scratch.capacity() >= last * DIM,
            "a growing batch size left only {} scratch elements after a \
             {}-element add, so the next add must grow and be shrunk again",
            idx.encode_scratch.capacity(),
            last * DIM,
        );
    }

    /// The same headroom property for a batch size that jitters rather
    /// than grows monotonically — the shape a real ingest loop has.
    #[test]
    fn jittering_batch_sizes_keep_their_growth_headroom() {
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        let sizes = [9_000, 11_000, 9_500, 10_800, 9_200, 10_400];
        for n in sizes {
            idx.add_2d(&rows(n, DIM), DIM).unwrap();
        }
        let biggest_recent = 10_400;
        assert!(
            idx.encode_scratch.capacity() >= biggest_recent * DIM,
            "a jittering batch size left only {} scratch elements, below \
             the {} the last add needed",
            idx.encode_scratch.capacity(),
            biggest_recent * DIM,
        );
    }

    /// A batch that steps up sharply and then holds must not have the
    /// step shrunk away underneath it. The hysteresis alone does not
    /// cover this — at a 3x step `capacity == 3 * prev` clears
    /// `2 * prev`, so without the slack in the target the buffer is cut
    /// straight back to the smaller batch and the next add regrows it.
    #[test]
    fn a_step_up_in_batch_size_is_not_shrunk_back() {
        let small = 6_000;
        let big = 3 * small;
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.add_2d(&rows(small, DIM), DIM).unwrap();
        idx.add_2d(&rows(big, DIM), DIM).unwrap();
        assert!(
            idx.encode_scratch.capacity() >= big * DIM,
            "a {small}->{big} step left only {} scratch elements, below the \
             {} the larger batch needed",
            idx.encode_scratch.capacity(),
            big * DIM,
        );
    }

    /// The converse of the step-up case, and the issue's own complaint
    /// restated: retention must not simply equal the largest single add
    /// ever made. One spike batch in a run of smaller ones has to be
    /// given back once the smaller ones resume.
    #[test]
    fn a_one_off_spike_does_not_stay_pinned() {
        let n = 6_000;
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.add_2d(&rows(n, DIM), DIM).unwrap();
        idx.add_2d(&rows(4 * n, DIM), DIM).unwrap();
        idx.add_2d(&rows(n, DIM), DIM).unwrap();
        assert!(
            idx.encode_scratch.capacity() < 4 * n * DIM,
            "a 4x spike left {} scratch elements pinned after the batch \
             size dropped back to {}",
            idx.encode_scratch.capacity(),
            n * DIM,
        );
    }

    /// `shrink_to` never goes below `len`, and the encode path leaves the
    /// scratch at its full length — so on the release path a shrink
    /// without a preceding truncate does nothing. Pin the truncate.
    #[test]
    fn retain_scratch_truncates_before_shrinking() {
        let big = 8 << 20;
        let mut scratch: Vec<f32> = vec![0.0; big];
        let prev = super::retain_scratch(&mut scratch, 0, big);
        assert_eq!(prev, big, "returns this call's demand");
        assert_eq!(
            scratch.capacity(),
            0,
            "a buffer no recent add needed was not released",
        );
    }
}

#[cfg(test)]
mod from_parts_tests {
    //! Unit tests for `TurboQuantIndex::from_parts` invariant checks that
    //! reach for private state (`dim`, calibration internals). The full
    //! public-surface coverage of every [`FromPartsError`] variant lives in
    //! `tests/from_parts.rs`; these pin the internal identity-population and
    //! accept paths.

    use super::TurboQuantIndex;
    use crate::FromPartsError;

    #[test]
    fn from_parts_rejects_packed_codes_length_mismatch() {
        // Expected packed_codes length for dim=64, bit_width=4, n=2 is
        // 2 * 64 * 4 / 8 = 64 bytes. Pass 32 to trigger the error.
        let err = TurboQuantIndex::from_parts(
            Some(64),
            4,
            2,
            vec![0u8; 32],
            vec![1.0f32; 2],
            Vec::new(),
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FromPartsError::PackedCodesLengthMismatch { expected: 64, got: 32 }
        ));
    }

    #[test]
    fn from_parts_rejects_lazy_with_nonzero_n_vectors() {
        let err = TurboQuantIndex::from_parts(
            None,
            4,
            5,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(err, FromPartsError::LazyMustHaveZeroVectors(5)));
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
        )
        .unwrap();
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
        )
        .unwrap();
        assert_eq!(idx.dim_opt(), Some(64));
        assert_eq!(idx.len(), 2);
        // v2-shape input (empty TQ+) stays empty: "declares nothing"
        // has exactly one representation, and it reads as Uncalibrated.
        assert!(idx.tqplus_shift().is_empty());
        assert!(idx.tqplus_scale().is_empty());
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

/// The crash contract, exhaustively: every batch of a sync's write plan
/// torn at every byte (in order, reversed, and each op alone), plus a
/// bit flipped in every byte of a committed file. See `io_v7`'s module
/// doc for the protocol these tests pin.
#[cfg(test)]
mod v7_crash_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    const DIM: usize = 64;

    fn rows(n: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * DIM];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        }
        for row in v.chunks_mut(DIM) {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
        v
    }

    fn temp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("turbovec-v7crash-{nonce}-{name}"));
        std::fs::create_dir(&p).unwrap();
        p.push("index.tv");
        p
    }

    fn apply(file: &mut Vec<u8>, off: u64, bytes: &[u8]) {
        let end = off as usize + bytes.len();
        if file.len() < end {
            file.resize(end, 0);
        }
        file[off as usize..end].copy_from_slice(bytes);
    }

    fn state_of(scratch: &Path, bytes: &[u8]) -> Option<Vec<u8>> {
        std::fs::write(scratch, bytes).unwrap();
        TurboQuantIndex::load(scratch).ok().map(|i| i.to_bytes())
    }



    /// A sync with adds AND removals (3 barriers), torn at every byte
    /// of every op: the loaded state is the previous commit until the
    /// header op's final byte completes, then the new commit. Never an
    /// error, never a third state.
    #[test]
    fn a_sync_torn_at_any_byte_recovers_the_previous_commit() {
        let path = temp("torn");
        let scratch = path.with_file_name("torn-scratch.tv");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 15)).unwrap();
        idx.add(&rows(100, 16));
        idx.sync(&path).unwrap();
        let base = std::fs::read(&path).unwrap();
        let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();

        idx.add(&rows(37, 17));
        idx.swap_remove(5);
        idx.swap_remove(60);
        idx.swap_remove(idx.len() - 1);
        let plan = idx.plan_next_sync(0, None);
        // Every incremental sync is ONE batch, ONE fsync: the header's
        // delta descriptor makes a commit that persists before its data
        // detectable, so no ordering barrier separates them.
        assert_eq!(plan.batches.len(), 1, "single-fsync sync");

        // The fully-applied plan is the live index, byte for byte.
        let mut done = base.clone();
        for b in &plan.batches {
            for (off, bytes) in &b.ops {
                apply(&mut done, *off, bytes);
            }
        }
        let state_b = state_of(&scratch, &done).expect("full plan must load");
        assert_eq!(state_b, idx.to_bytes(), "the standard oracle");

        let header_batch = plan.batches.len() - 1;
        for bi in 0..plan.batches.len() {
            let ops = &plan.batches[bi].ops;
            // Three intra-batch schedules: in order, reversed, and each
            // op alone — a batch's ops may hit disk in any order.
            for schedule in 0..3 {
                for (oj, (off, bytes)) in ops.iter().enumerate() {
                    for cut in 0..=bytes.len() {
                        let mut torn = base.clone();
                        for prev in &plan.batches[..bi] {
                            for (o, b) in &prev.ops {
                                apply(&mut torn, *o, b);
                            }
                        }
                        match schedule {
                            0 => {
                                for (o, b) in &ops[..oj] {
                                    apply(&mut torn, *o, b);
                                }
                            }
                            1 => {
                                for (o, b) in ops[oj + 1..].iter().rev() {
                                    apply(&mut torn, *o, b);
                                }
                            }
                            _ => {}
                        }
                        apply(&mut torn, *off, &bytes[..cut]);
                        let got = state_of(&scratch, &torn).unwrap_or_else(|| {
                            panic!("batch {bi} op {oj} sched {schedule} cut {cut}: unloadable")
                        });
                        let complete = bi == header_batch
                            && cut == bytes.len()
                            && (schedule == 0 && oj == ops.len() - 1
                                || schedule == 1 && oj == 0
                                || ops.len() == 1);
                        let want = if complete { &state_b } else { &state_a };
                        assert_eq!(
                            &got, want,
                            "batch {bi} op {oj} sched {schedule} cut {cut}: wrong state"
                        );
                    }
                }
            }
        }
    }

    /// Commit removals as pending ops, then tear the NEXT sync — the
    /// one that materializes them — at its batch boundaries, load the
    /// recovered state, keep working, and sync forward. The redo ops in
    /// the committed header must repair every partial materialization.
    #[test]
    fn a_recovery_load_syncs_forward_and_survives_a_second_tear() {
        let path = temp("double");
        let scratch = path.with_file_name("double-scratch.tv");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 25)).unwrap();
        idx.add(&rows(96, 26));
        idx.sync(&path).unwrap();

        // Removals commit as pending ops in the header (one barrier).
        idx.swap_remove(3);
        idx.swap_remove(40);
        assert_eq!(idx.plan_next_sync(0, None).batches.len(), 1);
        idx.sync(&path).unwrap();
        let base = std::fs::read(&path).unwrap();
        let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();
        assert_eq!(state_a, idx.to_bytes());

        // The next sync materializes those ops and appends new units.
        idx.add(&rows(40, 27));
        let plan = idx.plan_next_sync(0, None);
        assert_eq!(plan.batches.len(), 1, "single-fsync sync");
        assert!(plan.carried.is_empty());

        // Tear it after the data writes (everything except the header
        // op, which the planner pushes last): the old header still
        // carries the ops, and re-applying them over the materialized
        // units is a converging no-op.
        let mut crashed = base.clone();
        let ops = &plan.batches[0].ops;
        for (off, bytes) in &ops[..ops.len() - 1] {
            apply(&mut crashed, *off, bytes);
        }
        std::fs::write(&path, &crashed).unwrap();
        let mut rec = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(rec.to_bytes(), state_a, "recovery must be exact");

        // Work on the recovered index and sync it for real; its pending
        // set came from the loaded header.
        rec.add(&rows(5, 28));
        rec.swap_remove(10);
        rec.sync(&path).unwrap();
        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(loaded.to_bytes(), rec.to_bytes());

        // And tearing that recovery-continuation sync anywhere still
        // yields exactly one of the two adjacent commits.
        let plan2 = {
            let mut again = TurboQuantIndex::load(scratch_write(&scratch, &crashed)).unwrap();
            again.add(&rows(5, 28));
            again.swap_remove(10);
            again.plan_next_sync(0, None)
        };
        let state_b = {
            let mut done = crashed.clone();
            for b in &plan2.batches {
                for (off, bytes) in &b.ops {
                    apply(&mut done, *off, bytes);
                }
            }
            state_of(&scratch, &done).expect("full plan2 must load")
        };
        for bi in 0..plan2.batches.len() {
            let ops = &plan2.batches[bi].ops;
            for (oj, (off, bytes)) in ops.iter().enumerate() {
                for cut in [0, bytes.len() / 2, bytes.len()] {
                    let mut torn = crashed.clone();
                    for prev in &plan2.batches[..bi] {
                        for (o, b) in &prev.ops {
                            apply(&mut torn, *o, b);
                        }
                    }
                    for (o, b) in &ops[..oj] {
                        apply(&mut torn, *o, b);
                    }
                    apply(&mut torn, *off, &bytes[..cut]);
                    let got = state_of(&scratch, &torn)
                        .unwrap_or_else(|| panic!("batch {bi} op {oj} cut {cut}: unloadable"));
                    assert!(
                        got == state_a || got == state_b,
                        "batch {bi} op {oj} cut {cut}: neither adjacent commit"
                    );
                }
            }
        }
    }

    fn scratch_write<'a>(p: &'a Path, bytes: &[u8]) -> &'a Path {
        std::fs::write(p, bytes).unwrap();
        p
    }

    /// Barrier economics, pinned: pure appends need 2 barriers, a
    /// removal inside the tail needs only the header, and removals in
    /// committed units need 3.
    #[test]
    fn barrier_counts_match_the_change() {
        let path = temp("barriers");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 35)).unwrap();
        idx.add(&rows(70, 36));
        idx.sync(&path).unwrap();

        idx.add(&rows(40, 37));
        assert_eq!(
            idx.plan_next_sync(0, None).batches.len(),
            1,
            "pure append: one fsync"
        );
        idx.sync(&path).unwrap();

        idx.add(&rows(3, 38)); // tail now 3 rows past a block boundary
        idx.swap_remove(idx.len() - 1); // pop inside the tail
        assert_eq!(
            idx.plan_next_sync(0, None).batches.len(),
            1,
            "tail-only change: header alone"
        );
        idx.sync(&path).unwrap();

        idx.swap_remove(0);
        assert_eq!(
            idx.plan_next_sync(0, None).batches.len(),
            1,
            "committed-unit removal: one fsync — the op rides the header"
        );
        idx.sync(&path).unwrap();
        // The op is pending now; the next sync (any content)
        // materializes it inside its single batch, then clears it.
        idx.add(&rows(1, 39));
        let plan = idx.plan_next_sync(0, None);
        assert_eq!(plan.batches.len(), 1, "materialization folds into the batch");
        assert!(
            plan.batches[0].ops.len() >= 2,
            "the materialized unit write precedes the header op"
        );
        assert!(plan.carried.is_empty(), "no fresh dirt: nothing carried");
        idx.sync(&path).unwrap();
        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(loaded.to_bytes(), idx.to_bytes());
    }

    /// A header whose claimed n does not fit the file must be rejected
    /// as a candidate — refusing or falling back, never sizing an
    /// allocation from it (a hostile file could otherwise abort the
    /// process instead of returning Err).
    #[test]
    fn an_absurd_header_n_is_refused_not_allocated() {
        let path = temp("hostilen");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 65)).unwrap();
        idx.add(&rows(64, 66)); // whole blocks, empty tail: fixed prefix
        idx.sync(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();

        let geo = io_v7::Geo {
            kind: 0,
            dim: DIM,
            bit_width: 4,
            n_calib: DIM,
        };
        // Gen 0 lives in slot 0. Rewrite n to an absurd value and
        // re-seal the used prefix's CRC so only the bound can refuse.
        let at = geo.hdr_at_for_test(0);
        bytes[at + 8..at + 16].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
        let used = 16 + 4; // no tail rows (n % 32 == 0), no op groups
        let c = io_v7::crc32(&bytes[at..at + used]);
        bytes[at + used..at + used + 4].copy_from_slice(&c.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            TurboQuantIndex::load(&path).is_err(),
            "an absurd n must refuse, not allocate"
        );
    }

    /// Bit-rot over every byte of the STRUCTURAL region (superblock and
    /// both header slots) — the region that still carries guarantees
    /// now that blocks have no checksums (block damage from outside the
    /// writer is out of scope, as it was for v6): each flip either
    /// refuses, loads the identical current state, or — only inside the
    /// newest header slot — falls back to exactly the previous commit.
    /// Flips inside the newest commit's delta-covered blocks must also
    /// refuse or fall back — the digest owns those bytes.
    #[test]
    fn bit_rot_in_any_byte_is_never_served_silently() {
        let path = temp("rot");
        let scratch = path.with_file_name("rot-scratch.tv");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 45)).unwrap();
        idx.add(&rows(70, 46));
        idx.sync(&path).unwrap();
        let prev = TurboQuantIndex::load(&path).unwrap().to_bytes();

        idx.add(&rows(20, 47));
        idx.swap_remove(5);
        idx.sync(&path).unwrap();
        let cur = TurboQuantIndex::load(&path).unwrap().to_bytes();
        let file = std::fs::read(&path).unwrap();

        let geo = io_v7::Geo {
            kind: 0,
            dim: DIM,
            bit_width: 4,
            n_calib: DIM,
        };
        // gen 1 lives in slot 1.
        let newest_hdr = geo.hdr_at_for_test(1)..geo.hdr_at_for_test(1) + geo.hdr_len();

        let structural_end = geo.unit_at_for_test(0).min(file.len());
        for at in 0..structural_end {
            let mut bytes = file.clone();
            bytes[at] ^= 1 << (at % 8);
            match state_of(&scratch, &bytes) {
                None => {}
                Some(got) if got == cur => {}
                Some(got) if got == prev && newest_hdr.contains(&at) => {}
                Some(_) => panic!("flip at byte {at} served a state it must not"),
            }
        }
    }
}

/// Crash coverage for the two capture paths the main harness misses:
/// removal capture on a RELOADED (blocked-only) index, where `seq_row`
/// takes the lane-gather arm rather than the packed arm.
#[cfg(test)]
mod v7_crash_blocked_tests {
    use super::*;
    use std::path::PathBuf;

    const DIM: usize = 64;

    fn rows(n: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * DIM];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        }
        for row in v.chunks_mut(DIM) {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
        v
    }

    fn temp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("turbovec-v7blk-{nonce}-{name}"));
        std::fs::create_dir(&p).unwrap();
        p.push("index.tv");
        p
    }

    fn apply(file: &mut Vec<u8>, off: u64, bytes: &[u8]) {
        let end = off as usize + bytes.len();
        if file.len() < end {
            file.resize(end, 0);
        }
        file[off as usize..end].copy_from_slice(bytes);
    }

    /// Removals on a blocked-only index (fresh from load) serialize
    /// their redo ops through the lane-gather arm of `seq_row`. The
    /// committed header's op bytes are consumed on every load, and the
    /// materializing sync is torn after its data batch so recovery
    /// must verify those bytes against the expected unit CRCs — a
    /// wrong gather cannot survive this.
    #[test]
    fn blocked_only_capture_survives_a_torn_sync() {
        let path = temp("blkcap");
        let scratch = path.with_file_name("scratch.tv");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 55)).unwrap();
        idx.add(&rows(100, 56));
        idx.sync(&path).unwrap();

        // Reload: blocked cache only, packed unset — the gather arm.
        let mut idx = TurboQuantIndex::load(&path).unwrap();
        assert!(!idx.packed_ready(), "reload must be blocked-only");
        idx.swap_remove(1); // lane 1 of block 0: offset arithmetic visible
        idx.swap_remove(37); // lane 5 of block 1: base + group stride visible
        idx.sync(&path).unwrap();
        // The header's op bytes came from the gather arm; loading
        // consumes and verifies them.
        let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();
        assert_eq!(state_a, idx.to_bytes(), "gathered op bytes must be exact");

        // Tear the materializing sync after its data batch.
        let base = std::fs::read(&path).unwrap();
        idx.add(&rows(1, 57));
        let plan = idx.plan_next_sync(0, None);
        assert_eq!(plan.batches.len(), 1, "single-fsync sync");
        let mut torn = base.clone();
        let ops = &plan.batches[0].ops;
        for (off, bytes) in &ops[..ops.len() - 1] {
            apply(&mut torn, *off, bytes);
        }
        std::fs::write(&scratch, &torn).unwrap();
        let recovered = TurboQuantIndex::load(&scratch).unwrap();
        assert_eq!(recovered.to_bytes(), state_a, "recovery must be exact");

        // And the completed sync round-trips.
        idx.sync(&path).unwrap();
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes());
    }
}

/// The two crash states the review's executed reproductions exposed:
/// a spliced commit header whose data never landed (the delta digest
/// must reject it — a concatenation of self-consistent unit codewords
/// used to hash to a content-free constant), and syncing forward after
/// load fell back past such a commit (cursor_state must agree with the
/// loader about which commit is newest, or every sync wedges Foreign).
#[cfg(test)]
mod v7_delta_tests {
    use super::*;
    use std::path::PathBuf;

    const DIM: usize = 64; // small dim: single-chain CRC, the vacuous case

    fn rows(n: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * DIM];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        }
        for row in v.chunks_mut(DIM) {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
        v
    }

    fn temp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("turbovec-v7delta-{nonce}-{name}"));
        std::fs::create_dir(&p).unwrap();
        p.push("index.tv");
        p
    }

    /// The removal capture must actually engage for slots below the
    /// committed block floor — not merely at it.
    ///
    /// It is a pure optimization: a slot without an entry is re-read from
    /// its block, so every output test passes whether it engages or not.
    /// That makes the activation condition invisible to the rest of the
    /// suite, and a narrowing of it — `<` becoming `==`, say — would
    /// silently give back the whole win with nothing going red. This pins
    /// the condition itself.
    #[test]
    #[cfg(target_arch = "x86_64")] // capture is x86-only by measurement
    fn the_removal_capture_engages_below_the_block_floor() {
        let path = temp("capture-engages");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.add(&rows(200, 5));
        idx.sync(&path).unwrap();
        // Reload: the capture is only taken while the blocked cache is
        // authoritative, which is the state a synced-file load lands in.
        let mut idx = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(idx.captured_len_for_test(), 0, "nothing captured yet");

        // Committed floor is 192 (200 / 32 * 32). Remove well below it.
        idx.swap_remove(5);
        assert_eq!(
            idx.captured_len_for_test(),
            1,
            "a removal at slot 5, far below the committed floor of 192, must \
             be captured — if only the boundary slot captures, the sync is \
             back to re-reading every row it serializes",
        );
        idx.swap_remove(100);
        assert_eq!(idx.captured_len_for_test(), 2, "and again mid-range");

        // Above the floor there is nothing on disk to patch, so no capture.
        let before = idx.captured_len_for_test();
        idx.swap_remove(idx.len() - 1);
        assert_eq!(
            idx.captured_len_for_test(),
            before,
            "a pop above the committed floor has no redo op to feed",
        );

        // And the entries are spent by the sync that serializes them.
        idx.sync(&path).unwrap();
        assert_eq!(idx.captured_len_for_test(), 0, "cleared by the commit");
    }

    /// Splice ONLY the new commit header over the pre-sync file — the
    /// exact "commit reached disk before its data" state — with a
    /// materialized one-unit delta at a small dim. The digest must be
    /// content-sensitive: the loader lands on the previous commit, and
    /// a removed vector is never resurrected.
    #[test]
    fn a_spliced_header_without_its_data_is_not_adopted() {
        let path = temp("splice");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 11)).unwrap();
        idx.add(&rows(96, 12));
        idx.sync(&path).unwrap();
        // Ops ride the header...
        idx.swap_remove(3);
        idx.add(&rows(1, 13));
        idx.sync(&path).unwrap();
        let pre = std::fs::read(&path).unwrap();
        let state_pre = TurboQuantIndex::load(&path).unwrap().to_bytes();
        // ...and this sync materializes block 0: a one-unit delta.
        // (The fresh dirt goes to block 1 — dirtying block 0 again
        // would carry its ops instead of materializing them.)
        idx.swap_remove(40);
        idx.add(&rows(1, 14));
        let plan = idx.plan_next_sync(0, None);
        let (hdr_off, hdr_bytes) = plan.batches[0]
            .ops
            .last()
            .expect("plan has a header op")
            .clone();
        assert!(
            plan.batches[0].ops.len() > 1,
            "the sync must materialize at least one unit for this test"
        );
        let mut spliced = pre.clone();
        let end = hdr_off as usize + hdr_bytes.len();
        if spliced.len() < end {
            spliced.resize(end, 0);
        }
        spliced[hdr_off as usize..end].copy_from_slice(&hdr_bytes);
        std::fs::write(&path, &spliced).unwrap();
        let got = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(
            got.to_bytes(),
            state_pre,
            "a header without its data must fall back, not resurrect stale blocks"
        );
    }

    /// A crafted header whose op group names an out-of-range block must
    /// refuse the load — never reach the op-application indexing.
    #[test]
    fn an_out_of_range_op_block_is_refused_not_indexed() {
        let path = temp("hostileb");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 31)).unwrap();
        idx.add(&rows(65, 32)); // 65 so the post-removal n is a whole block
        idx.sync(&path).unwrap();
        // Commit a real op so the header layout has a group to mutate.
        idx.swap_remove(3);
        idx.sync(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();

        let geo = io_v7::Geo {
            kind: 0,
            dim: DIM,
            bit_width: 4,
            n_calib: DIM,
        };
        // Gen 1 lives in slot 1. Its used prefix: gen8 | n8 | tail(0) |
        // n_units4 | group { block4, crc4, n_ops1, op... }. Overwrite
        // the group's block index with an absurd value and re-seal the
        // header CRC so only the bound can refuse.
        let at = geo.hdr_at_for_test(1);
        let gb = at + 16 + 4;
        bytes[gb..gb + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let row = DIM / 2;
        let op_size = 1 + row + 4;
        let used = 16 + 4 + 5 + op_size + 4 + 12;
        let c = io_v7::crc32(&bytes[at..at + used]);
        bytes[at + used..at + used + 4].copy_from_slice(&c.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        // The corrupt gen-1 header must be rejected as a candidate; the
        // file falls back to gen 0 or errs — it must never panic.
        let _ = TurboQuantIndex::load(&path);
    }

    /// A negative scale smuggled into the commit tail must refuse the
    /// load — the sign check is load's only guard for tail scales.
    #[test]
    fn a_negative_tail_scale_is_refused() {
        let path = temp("negscale");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 41)).unwrap();
        idx.add(&rows(65, 42)); // one tail row rides the header
        idx.sync(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();

        let geo = io_v7::Geo {
            kind: 0,
            dim: DIM,
            bit_width: 4,
            n_calib: DIM,
        };
        let row_bytes = DIM / 2;
        // Gen 0, slot 0: gen8 | n8 | tail row { codes, scale } | ops(0)
        // | delta(empty) | crc. Negate the tail scale and re-seal.
        let at = geo.hdr_at_for_test(0);
        let sc = at + 16 + row_bytes;
        let v = f32::from_le_bytes(bytes[sc..sc + 4].try_into().unwrap());
        bytes[sc..sc + 4].copy_from_slice(&(-v.max(0.5)).to_le_bytes());
        let used = 16 + (row_bytes + 4) + 4 + 16;
        let c = io_v7::crc32(&bytes[at..at + used]);
        bytes[at + used..at + used + 4].copy_from_slice(&c.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            TurboQuantIndex::load(&path).is_err(),
            "a negative tail scale must refuse the load"
        );
    }

    /// A tampered calibration value (zero scale, resealed superblock
    /// CRC) must refuse the load — search divides by these, and v6
    /// refuses the identical payload.
    #[test]
    fn a_zero_tqplus_scale_is_refused() {
        let path = temp("zeroscale");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 51)).unwrap();
        idx.add(&rows(40, 52));
        idx.sync(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();

        // Superblock: magic4 ver1 bit1 kind1 dim4 nonce8 | boundaries
        // (nl-1)*4 | centroids nl*4 | n_calib4 | shift dim*4 | scale
        // dim*4 | crc4. Zero scale[5] and reseal.
        let nl = 16;
        let scale5 = 23 + (nl - 1) * 4 + nl * 4 + 4 + DIM * 4 + 5 * 4;
        bytes[scale5..scale5 + 4].copy_from_slice(&0.0f32.to_le_bytes());
        let sb_end = 23 + (nl - 1) * 4 + nl * 4 + 4 + DIM * 8;
        let c = io_v7::crc32(&bytes[..sb_end]);
        bytes[sb_end..sb_end + 4].copy_from_slice(&c.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = TurboQuantIndex::load(&path).unwrap_err();
        assert!(err.to_string().contains("TQ+ scale"), "{err}");
    }

    /// calibrate() queues a compaction; if another writer has taken
    /// over the path since, that compaction must refuse like any other
    /// sync — never rename over the foreign file unchecked.
    #[test]
    fn a_queued_compaction_still_respects_the_foreign_file_guard() {
        let path = temp("calibforeign");
        let mut a = TurboQuantIndex::new(DIM, 4).unwrap();
        a.calibrate(&rows(1024, 53)).unwrap();
        a.add(&rows(40, 54));
        a.sync(&path).unwrap();

        let mut b = TurboQuantIndex::new(DIM, 4).unwrap();
        b.calibrate(&rows(1024, 55)).unwrap();
        b.add(&rows(20, 56));
        b.sync(&path).unwrap(); // B takes over the path (new nonce)

        a.calibrate(&rows(1024, 57)).unwrap(); // queues A's compaction
        let err = a.sync(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(loaded.to_bytes(), b.to_bytes(), "B's file must be untouched");
    }

    /// The corruption matrix: overwrite every early field and hundreds
    /// of seeded random bytes across a committed file, RESEAL the
    /// checksums so only semantic validation can object, and demand the
    /// loader always ends politely — Ok or Err, never a panic. This is
    /// the systematic form of the crafted-file attacks the reviews kept
    /// finding one at a time.
    #[test]
    fn every_field_tamper_loads_politely() {
        let path = temp("matrix");
        let scratch = path.with_file_name("matrix-scratch.tv");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 61)).unwrap();
        idx.add(&rows(70, 62));
        idx.sync(&path).unwrap();
        idx.swap_remove(3); // a pending op rides the header
        idx.add(&rows(2, 63));
        idx.sync(&path).unwrap();
        let base = std::fs::read(&path).unwrap();
        let geo = io_v7::Geo {
            kind: 0,
            dim: DIM,
            bit_width: 4,
            n_calib: DIM,
        };

        let try_load = |bytes: &[u8], what: &str| {
            std::fs::write(&scratch, bytes).unwrap();
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = TurboQuantIndex::load(&scratch);
            }));
            assert!(r.is_ok(), "loader panicked on tamper: {what}");
        };

        // Hostile values in every byte of the structural regions: the
        // whole superblock and both header slots.
        let hostile: [u8; 4] = [0xFF, 0x00, 0x80, 0x01];
        // The parser reads the superblock and each header slot's used
        // prefix; the rest of the slots is reserved slack it never
        // touches. Tamper every read byte with every hostile value, and
        // stride-sample the slack (a prime stride so successive runs
        // land on different offsets per slot).
        let mut targets: Vec<usize> = Vec::new();
        let sb_end = geo.hdr_at_for_test(0);
        targets.extend(0..sb_end);
        for slot in [0usize, 1] {
            let at = geo.hdr_at_for_test(slot);
            let used = io_v7::hdr_used_for_test(&base, &geo, slot) + 8;
            let end = at + geo.hdr_len();
            targets.extend(at..(at + used).min(end));
            targets.extend(((at + used)..end).step_by(251));
        }
        let structural_end = geo.unit_at_for_test(0);
        let _ = structural_end;
        for &at in targets.iter().filter(|&&a| a < base.len()) {
            for v in hostile {
                if base[at] == v {
                    continue;
                }
                let mut bytes = base.clone();
                bytes[at] = v;
                io_v7::reseal_for_test(&mut bytes, &geo);
                try_load(&bytes, &format!("byte {at} <- {v:#04x}"));
            }
        }
        // Seeded random multi-byte tampers across the entire file.
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        for i in 0..400 {
            let mut bytes = base.clone();
            for _ in 0..1 + (i % 4) {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let at = (s as usize) % bytes.len();
                bytes[at] = (s >> 32) as u8;
            }
            io_v7::reseal_for_test(&mut bytes, &geo);
            try_load(&bytes, &format!("random tamper {i}"));
        }
        // Truncations at every structural boundary and random lengths.
        for cut in [0usize, 4, 11, 19, structural_end / 2, structural_end] {
            try_load(&base[..cut.min(base.len())], &format!("truncate {cut}"));
        }
    }

    /// A negative scale inside a BLOCK must refuse the load too — the
    /// blocks carry no checksums, so the sign check is the only guard.
    #[test]
    fn a_negative_block_scale_is_refused() {
        let path = temp("negblockscale");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 71)).unwrap();
        idx.add(&rows(64, 72));
        idx.sync(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let geo = io_v7::Geo {
            kind: 0,
            dim: DIM,
            bit_width: 4,
            n_calib: DIM,
        };
        // Unit 0: codes (32 * row) then 32 scales. Negate scale of lane 7.
        let row_bytes = DIM / 2;
        let sc = geo.unit_at_for_test(0) + 32 * row_bytes + 7 * 4;
        let v = f32::from_le_bytes(bytes[sc..sc + 4].try_into().unwrap());
        bytes[sc..sc + 4].copy_from_slice(&(-v.max(0.5)).to_le_bytes());
        // No reseal needed: blocks carry no checksum. Only the sign
        // check can object.
        std::fs::write(&path, &bytes).unwrap();
        let err = TurboQuantIndex::load(&path).unwrap_err();
        assert!(err.to_string().contains("scale"), "{err}");
    }

    /// More dirtied slots than one header can carry must stay
    /// incremental — the dirty blocks are committed directly in the
    /// same batch, never a full rewrite (which would swap the nonce).
    #[test]
    fn a_mass_removal_stays_incremental() {
        let path = temp("massremove");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 73)).unwrap();
        idx.add(&rows(200, 74));
        idx.sync(&path).unwrap();
        let nonce_before = std::fs::read(&path).unwrap()[11..19].to_vec();
        // Dirty far more than MAX_OPS distinct committed slots.
        for i in 0..80 {
            idx.swap_remove(i);
        }
        idx.sync(&path).unwrap();
        let nonce_after = std::fs::read(&path).unwrap()[11..19].to_vec();
        assert_eq!(nonce_before, nonce_after, "sync degraded to a full rewrite");
        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(loaded.to_bytes(), idx.to_bytes());
        // And the file keeps syncing incrementally afterwards.
        idx.add(&rows(1, 75));
        idx.sync(&path).unwrap();
        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(loaded.to_bytes(), idx.to_bytes());
    }

    /// Materializing a unit that the FALLBACK header's delta also names
    /// (append + removal committed together, cleaned up one sync later),
    /// torn before the new header lands, must still load the previous
    /// commit exactly — safe because ops are absolute writes of the
    /// unit's current bytes, so materialization rewrites byte-identical
    /// content.
    #[test]
    fn a_torn_materialize_of_a_delta_named_unit_recovers() {
        let path = temp("mat-delta");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 41)).unwrap();
        idx.add(&rows(32, 42));
        idx.sync(&path).unwrap();
        // Sync 2: append unit 1 AND remove a slot inside it -> unit 1 is
        // fresh (op carried) and named by header 2's delta as an append.
        idx.add(&rows(64, 43));
        idx.swap_remove(40);
        idx.sync(&path).unwrap();
        let committed = TurboQuantIndex::load(&path).unwrap().to_bytes();
        // Sync 3: nothing new -> materializes unit 1 (op from header 2).
        let plan = idx.plan_next_sync(0, None);
        // Tear: apply every op EXCEPT the header write (last op).
        let batch = &plan.batches[0];
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        use std::io::{Seek, SeekFrom, Write};
        for (off, bytes) in &batch.ops[..batch.ops.len() - 1] {
            f.seek(SeekFrom::Start(*off)).unwrap();
            f.write_all(bytes).unwrap();
        }
        drop(f);
        match TurboQuantIndex::load(&path) {
            Ok(loaded) => assert_eq!(
                loaded.to_bytes(),
                committed,
                "loaded state differs from the previous commit"
            ),
            Err(e) => panic!("file refused to load: {e}"),
        }
    }

    /// Past [`io_v7::MAX_OPS`] scattered removals between syncs, the
    /// sync must fall back to a full rewrite (fresh blocks may never be
    /// overwritten in the sync that first commits their changes — no
    /// fallback header would describe them) and round-trip exactly.
    #[test]
    fn an_op_overflow_falls_back_to_a_full_rewrite() {
        let path = temp("ovf-full");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 51)).unwrap();
        idx.add(&rows(2_112, 52));
        idx.sync(&path).unwrap();
        let nonce_before = std::fs::read(&path).unwrap()[11..19].to_vec();
        // One more dirtied slot than one header holds (the cap counts
        // slots; contiguous ones overflow it as well as scattered).
        for v in (5..5 + io_v7::MAX_OPS + 1).rev() {
            idx.swap_remove(v);
        }
        idx.sync(&path).unwrap();
        let nonce_after = std::fs::read(&path).unwrap()[11..19].to_vec();
        assert_ne!(nonce_before, nonce_after, "overflow must full-rewrite");
        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(loaded.to_bytes(), idx.to_bytes());
        // And the rewritten file keeps syncing incrementally.
        idx.add(&rows(1, 53));
        idx.sync(&path).unwrap();
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes());
    }

    /// A failed sync must not wedge the binding: if the commit landed
    /// but sync() reported an error, the cursor is dropped and the next
    /// sync recovers by writing full — never "another writer advanced".
    #[test]
    // set_readonly(false) on the throwaway test file is deliberate: the
    // cross-platform way to restore writability after the induced error.
    #[allow(clippy::permissions_set_readonly_false)]
    fn a_failed_sync_recovers_via_full_write() {
        let path = temp("failedsync");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 91)).unwrap();
        idx.add(&rows(64, 92));
        idx.sync(&path).unwrap();
        idx.add(&rows(32, 93));
        // Make the incremental write itself fail: after any failed
        // sync, nothing on disk can be trusted (a commit may or may
        // not have landed), so the binding must drop.
        let mut perm = std::fs::metadata(&path).unwrap().permissions();
        perm.set_readonly(true);
        std::fs::set_permissions(&path, perm.clone()).unwrap();
        assert!(idx.sync(&path).is_err(), "read-only sync must fail");
        perm.set_readonly(false);
        std::fs::set_permissions(&path, perm).unwrap();
        // The retry must succeed (full write path) and round-trip.
        idx.sync(&path).unwrap();
        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(loaded.to_bytes(), idx.to_bytes());
    }

    /// After load falls back past a data-less commit, sync must keep
    /// working: cursor_state has to reject that commit exactly as the
    /// loader did, or the file wedges Foreign forever.
    #[test]
    fn sync_keeps_working_after_a_fallback_load() {
        let path = temp("wedge");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 21)).unwrap();
        idx.add(&rows(64, 22));
        idx.sync(&path).unwrap();
        let state_g0 = TurboQuantIndex::load(&path).unwrap().to_bytes();
        idx.add(&rows(32, 23));
        idx.sync(&path).unwrap();

        // Zero the appended unit's bytes in place: header gen 1 landed,
        // its data did not (lengths and both header slots untouched).
        let geo = io_v7::Geo {
            kind: 0,
            dim: DIM,
            bit_width: 4,
            n_calib: DIM,
        };
        let mut bytes = std::fs::read(&path).unwrap();
        let at = geo.unit_at(2); // blocks 0,1 belong to gen 0; block 2 was gen 1's append
        for b in bytes[at..at + geo.unit_len()].iter_mut() {
            *b = 0;
        }
        std::fs::write(&path, &bytes).unwrap();

        // Load falls back to gen 0...
        let mut rec = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(rec.to_bytes(), state_g0, "fallback must be exact");
        // ...and the very next sync must succeed, not wedge Foreign.
        rec.add(&rows(5, 24));
        rec.sync(&path).unwrap();
        let after = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(after.to_bytes(), rec.to_bytes());

        // The truncation flavour of the same state.
        let path2 = temp("wedge-trunc");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 25)).unwrap();
        idx.add(&rows(64, 26));
        idx.sync(&path2).unwrap();
        let pre_len = std::fs::metadata(&path2).unwrap().len();
        let state_g0 = TurboQuantIndex::load(&path2).unwrap().to_bytes();
        idx.add(&rows(32, 27));
        idx.sync(&path2).unwrap();
        // Keep the header region (it precedes the units), drop the data.
        let full = std::fs::read(&path2).unwrap();
        std::fs::write(&path2, &full[..pre_len as usize]).unwrap();
        let mut rec = TurboQuantIndex::load(&path2).unwrap();
        assert_eq!(rec.to_bytes(), state_g0);
        rec.swap_remove(0);
        rec.sync(&path2).unwrap();
        assert_eq!(
            TurboQuantIndex::load(&path2).unwrap().to_bytes(),
            rec.to_bytes()
        );
    }
}
