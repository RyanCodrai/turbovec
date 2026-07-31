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

pub use error::{AddError, ConstructError, FromPartsError, SearchError};
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

// See [`TurboQuantIndex::force_encode_panic_at`]. Thread-local for the
// same reason the one-shot switch beside it is (#373). A countdown
// rather than a flag because `add` splits a batch into one encode per
// calibration block, and the case worth reaching is a panic on a
// *later* chunk — after an earlier one has already committed rows.
#[cfg(test)]
thread_local! {
    static FORCE_ENCODE_PANIC_AT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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

/// Rows in a calibration block, when per-block calibration is enabled
/// with [`TurboQuantIndex::with_block_size`].
///
/// A block seals when it fills: it fits `(shift, scale)` from its own
/// rows, re-encodes them under that fit, and is never refitted and never
/// merged with another. Insertion order therefore cannot decide any
/// block's calibration beyond the rows that block actually holds.
///
/// 8192 is chosen as never-worst rather than best. Measured across
/// GloVe-200, SIFT-128, fashion-MNIST-784, gte-small-384 and OpenAI-1536
/// at 2 and 4 bits against a single global fit: on i.i.d. insertion every
/// size from 2048 to 32768 is a wash (within 0.9 pp R@10); on worst-case
/// PC1-sorted insertion no size wins everywhere — SIFT prefers 2048
/// (+6.2 pp), fashion-MNIST at 2 bits prefers 32768 (2048 costs it
/// 3.3 pp). 8192 is never the worst option on any measured row, worst
/// case -1.2 pp.
///
/// Per-block overhead is `2 * dim` floats, i.e. `64 / (block_size *
/// bits)` of the code size — 0.39% at 8192 rows and 2 bits, whatever the
/// dim.
pub const DEFAULT_BLOCK_SIZE: usize = 8192;

/// The smallest permitted block size, and the granularity every other
/// permitted size is a multiple of.
///
/// Two independent layouts have to line up on a block boundary for a
/// block to be searchable as a self-contained range: the SIMD-blocked
/// code layout, which groups 32 rows per block, and the packed search
/// mask, which packs 64 slots per `u64` word. 64 is their least common
/// multiple.
pub const MIN_BLOCK_SIZE: usize = 64;

/// A sealed calibration block: its frozen `(shift, scale)` pair and the
/// number of rows in it that are still live.
///
/// Block `b` owns storage rows `b * block_size .. (b + 1) * block_size`
/// and that extent never changes, so a removal inside one block cannot
/// renumber any other block's slots — which is what makes the O(1)
/// block-local `swap_remove` expressible at all. The live rows are the
/// dense prefix `b * block_size .. b * block_size + len`; the rest of
/// the extent is dead storage that no search ever scores.
#[derive(Debug, Clone)]
struct SealedBlock {
    shift: Vec<f32>,
    scale: Vec<f32>,
    len: usize,
}

/// State of an index's TQ+ per-coordinate calibration.
///
/// TQ+ fits a `(shift, scale)` pair per coordinate from the empirical
/// quantiles of the vectors added to the index. The fit needs at least
/// 1000 vectors to be stable, so an index that has seen fewer is still
/// warming up: its rows are stored under an identity calibration and are
/// re-encoded once the 1000th vector arrives.
///
/// Query it with [`TurboQuantIndex::calibration_state`] /
/// [`IdMapIndex::calibration_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationState {
    /// Fewer than 1000 vectors have been added and the raw rows are
    /// still buffered. Search works (under identity calibration), and a
    /// later add that brings the total to 1000 re-encodes every stored
    /// row with a fitted calibration.
    WarmingUp,
    /// A calibration fitted from at least 1000 vectors is committed;
    /// every stored row is encoded in it, and every later add reuses it.
    Fitted,
    /// The index is committed to identity calibration for good: no TQ+
    /// recall gain, now or later. Reached by loading (or reconstructing
    /// through [`TurboQuantIndex::from_parts`]) an index whose stored
    /// rows were encoded under identity — including one saved while it
    /// was still warming up, since a file carries no warm-up buffer.
    /// Recovering the TQ+ gain requires rebuilding from the original
    /// float32 vectors. A **payload** with no stored rows never loads
    /// into this state: it has nothing encoded under identity, so it
    /// reloads as [`WarmingUp`](CalibrationState::WarmingUp) (#418). An
    /// index already committed to identity keeps that commitment when
    /// `swap_remove` drains it, exactly as a fitted one does (#284).
    Identity,
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
    /// Storage extent: the number of rows `packed_codes`, `scales` and
    /// the blocked cache describe. Equal to [`Self::len`] unless a
    /// block-local `swap_remove` has left dead rows in the tail of a
    /// sealed block — see [`SealedBlock`].
    n_vectors: usize,
    /// Storage rows that hold no live vector: the tails of sealed
    /// blocks that a block-local `swap_remove` has shortened. Always 0
    /// while `sealed` is empty, and [`Self::len`] is
    /// `n_vectors - dead_slots`.
    dead_slots: usize,
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

    /// TQ+ per-coord calibration. Both have length `dim` once the first
    /// add has happened (and the batch had enough samples to fit them);
    /// empty otherwise. Frozen after the first add — subsequent adds
    /// reuse them so all vectors in the index live in the same
    /// calibrated coordinate system. Loaded indexes from pre-TQ+ files
    /// arrive empty and behave as identity calibration (no recall gain,
    /// no behaviour change vs the old encoding).
    tqplus_shift: Vec<f32>,
    tqplus_scale: Vec<f32>,

    /// Warm-up buffer: the raw (un-encoded) float rows added so far while
    /// the index has yet to see `TQPLUS_MIN_SAMPLES` vectors in total.
    /// `Some` exactly while the index is in the warm-up phase — i.e. it
    /// has never committed a fitted calibration and its stored codes (if
    /// any) were all produced from rows still held here. `None` once a
    /// calibration is committed for good.
    ///
    /// Rows are kept in slot order and mirror `swap_remove`, so the
    /// buffer's row `i` is always the index's slot `i`. Bounded by
    /// `TQPLUS_MIN_SAMPLES` rows at rest (the batch that reaches the
    /// threshold is encoded immediately and the buffer dropped), i.e.
    /// `< 1000 * dim * 4` bytes.
    ///
    /// See [`CalibrationState`] for what this buys: an index seeded with
    /// a sub-threshold first add can still fit a real calibration once
    /// enough vectors have arrived, by re-encoding those early rows.
    warmup: Option<Vec<f32>>,

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
    // for yet. The only place the `OnceLock` is replaced outright is the
    // TQ+ threshold crossing in `add`, which re-encodes every row from
    // the warm-up buffer and so has no prior state worth keeping.
    // Whichever path runs, `blocked` covers exactly `n_vectors` rows by
    // the time the borrow ends.
    rotation: OnceLock<rotation::Rotation>,
    boundaries: OnceLock<Vec<f32>>,
    centroids: OnceLock<Vec<f32>>,
    blocked: OnceLock<BlockedCache>,

    /// Whether TQ+ calibration is fitted at all.
    ///
    /// `false` puts the index in [`CalibrationState::Identity`] for good
    /// the moment its dim is known: no warm-up buffer, no fit, no
    /// re-encode, and every row stored under the identity coordinate
    /// system. Purely a construction-time choice — it is expressed
    /// entirely through the committed `(shift, scale)` pair, which the
    /// file format already carries, so an uncalibrated index round-trips
    /// as one without a format change.
    calibration_enabled: bool,

    /// Rows per calibration block, or `None` for one calibration over
    /// the whole index (the default, and what every index built before
    /// [`Self::with_block_size`] existed does).
    ///
    /// Frozen at construction: changing it changes which rows share a
    /// calibration, i.e. the encoded bytes.
    block_size: Option<usize>,

    /// Sealed calibration blocks, in slot order. Empty unless
    /// `block_size` is `Some`. The index's remaining rows live in the
    /// *open* block, whose calibration is `tqplus_shift`/`tqplus_scale`
    /// and which starts at slot `sealed.len() * block_size`.
    sealed: Vec<SealedBlock>,

    /// Raw rows of the open block, kept so the block can be refitted and
    /// re-encoded from its own rows when it seals.
    ///
    /// `None` when there is nothing to keep them for: a single-block
    /// index (no sealing), an index that opted out of calibration, or one
    /// loaded from a file — a file carries no float rows, so a loaded
    /// index's open block seals on the calibration it already has rather
    /// than refitting. Bounded by `block_size` rows, since the block
    /// seals the moment it is full.
    open_rows: Option<Vec<f32>>,

    /// Reusable encode scratch (the rotated-batch buffer). Purely
    /// derived state: never serialized, contents meaningless between
    /// calls — kept only so repeated adds reuse one allocation instead
    /// of paying a fresh multi-MB mmap + page-fault walk per call.
    encode_scratch: Vec<f32>,
    /// Element count the *previous* add asked of `encode_scratch`. Sizes
    /// the retention target in [`retain_scratch`], so a buffer is only
    /// kept while the adds around it are still using one that big.
    encode_scratch_prev: usize,
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
    /// The two paths that do set it without `packed_codes` are both out
    /// of reach there: a v6 load of an **empty** index seeds the lock
    /// directly, and the TQ+ warm-up crossing replaces it — but a
    /// non-empty load never enters warm-up, since
    /// `normalize_calibration` only buffers when the dim is uncommitted
    /// or the index is empty.
    ///
    /// So this is **not** a "first mutation after a load" probe, and
    /// gating a binding's fast path on it means gating it off forever on
    /// every loaded index — the defect behind #392. It answers exactly
    /// one question: which of the two code layouts is currently
    /// materialized.
    pub fn packed_ready(&self) -> bool {
        self.packed_codes.get().is_some()
    }

    /// Whether an [`Self::add`] of `n_rows` rows will run parallel work
    /// that is not proportional to `n_rows`.
    ///
    /// True exactly when this batch crosses the TQ+ warm-up threshold:
    /// the crossing fits a calibration and re-encodes the whole buffered
    /// prefix — up to ~1000 rows of splitting work for an add of a
    /// single row. Callers that decide whether to enter a fork-safe pool
    /// from the row count alone would otherwise miss it (#364).
    pub fn add_parallelizes(&self, n_rows: usize) -> bool {
        let (Some(dim), Some(buffer)) = (self.dim, self.warmup.as_ref()) else {
            return false;
        };
        buffer.len() / dim + n_rows >= encode::TQPLUS_MIN_SAMPLES
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
            dead_slots: 0,
            packed_codes: OnceLock::from(Vec::new()),
            scales: Vec::new(),
            tqplus_shift: Vec::new(),
            tqplus_scale: Vec::new(),
            warmup: Some(Vec::new()),
            calibration_enabled: true,
            block_size: Some(DEFAULT_BLOCK_SIZE),
            sealed: Vec::new(),
            open_rows: Some(Vec::new()),
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
            encode_scratch: Vec::new(),
            encode_scratch_prev: 0,
        })
    }

    /// Construct an index with TQ+ calibration turned off.
    ///
    /// The index goes to [`CalibrationState::Identity`] immediately and
    /// stays there: no warm-up buffer, no fit, no re-encode of earlier
    /// rows, and every vector stored in the identity coordinate system.
    /// Otherwise identical to [`Self::new`].
    ///
    /// # When you want this
    ///
    /// Calibration buys recall on data whose rotated coordinates sit off
    /// the canonical marginal — embeddings with a strong mean direction,
    /// image descriptors, raw pixels. On well-centred modern text
    /// embeddings it is worth well under a point, and it is not free:
    /// each block carries `2 * dim` floats, the open block keeps its raw
    /// rows in memory so it can refit when it seals, and sealing
    /// re-encodes the block once more.
    ///
    /// Turning it off trades a small expected recall loss for an index
    /// with no fitted state at all: no warm-up, nothing to go stale, no
    /// per-block pairs, no raw rows kept, and identical behaviour
    /// whatever order the rows arrive in. Rows still live in blocks —
    /// that is where slots come from — but every block is identity.
    ///
    /// Returns the same errors as [`Self::new`].
    pub fn new_uncalibrated(dim: usize, bit_width: usize) -> Result<Self, ConstructError> {
        let mut ix = Self::new(dim, bit_width)?;
        ix.calibration_enabled = false;
        ix.commit_identity_calibration(dim);
        Ok(ix)
    }

    /// [`Self::new_lazy`] with TQ+ calibration turned off — see
    /// [`Self::new_uncalibrated`] for what that means and when to want it.
    ///
    /// The identity calibration is committed on the first add, since it
    /// is `dim`-shaped and the dim is not known before then. An index
    /// serialized before its first add therefore carries no calibration
    /// pair — but it does carry the opt-out itself, in bit 31 of the
    /// TQ+ trailer's length word, so it reloads with calibration still
    /// disabled and the first add after the reload commits identity
    /// exactly as it would have. The flag is the reason v7 exists: the
    /// pair alone cannot tell an index that opted out from one that is
    /// warming up with nothing stored yet.
    pub fn new_lazy_uncalibrated(bit_width: usize) -> Result<Self, ConstructError> {
        let mut ix = Self::new_lazy(bit_width)?;
        ix.calibration_enabled = false;
        Ok(ix)
    }

    /// Construct an index that calibrates in blocks of `block_size` rows
    /// instead of once for the whole index.
    ///
    /// # What this changes
    ///
    /// [`Self::new`] fits one TQ+ `(shift, scale)` pair from the first
    /// batch large enough to make it and keeps that pair for the life of
    /// the index. That is the right trade when the rows arrive i.i.d.,
    /// and the wrong one when they do not: an index fed a *sorted*
    /// stream commits a calibration describing only the head of the
    /// stream, and every later row is quantized in a coordinate system
    /// fitted to data that does not look like it.
    ///
    /// With a block size, rows accumulate in an **open** block. When the
    /// open block fills it *seals*: it fits a calibration from its own
    /// `block_size` rows, re-encodes those rows under that fit, and
    /// freezes. Sealed blocks are never refitted and never merged —
    /// re-encoding a row under another block's calibration would need
    /// the float32 original, which the index does not keep. A search
    /// scores each block against its own calibration and merges the
    /// per-block results.
    ///
    /// Slot numbers are unaffected: block `b` owns storage rows
    /// `b * block_size ..`, and a [`Self::swap_remove`] fills the hole
    /// from the last live row of the *same* block, so no other block's
    /// slots move. That leaves dead rows in the tail of a shortened
    /// block, which is why [`Self::len`] (live rows) and
    /// [`Self::slot_capacity`] (storage rows, and the length a search
    /// mask must have) can differ once a block-local removal has
    /// happened.
    ///
    /// # Serialization
    ///
    /// The file carries the block table — every sealed block's frozen
    /// pair and live row count, the block size, and (while they cost
    /// less than the codes they improve) the open block's raw rows — so
    /// a blocked index round-trips with its calibration and its holes
    /// intact and scores identically. This is format v7; v5 and v6 files
    /// cannot express it and no longer load.
    ///
    /// # A block below 1000 rows fits nothing
    ///
    /// A calibration needs about 1000 samples to be stable, and a block
    /// is its own fit sample, so a block smaller than that seals on
    /// identity and the index gets no TQ+ gain at all — it behaves
    /// exactly like [`Self::new_uncalibrated`], and
    /// [`Self::calibration_state`] reports
    /// [`Identity`](CalibrationState::Identity), which is how to check.
    /// Measured on a drifting 64-dim stream at 4 bits: top-1
    /// self-recall 0.128 at a 64- or 128-row block size — the same
    /// figure an uncalibrated index scores on it — against 0.998 at
    /// 1024.
    ///
    /// Sizes below 1000 are still accepted, because the block is also
    /// the unit of deletion locality and a caller may want a small one
    /// for that. But if calibration is what you are here for, stay well
    /// above 1000.
    ///
    /// # Errors
    ///
    /// [`ConstructError::BlockSizeInvalid`] if `block_size` is zero or
    /// not a multiple of [`MIN_BLOCK_SIZE`]; otherwise the same errors
    /// as [`Self::new`].
    ///
    /// [`DEFAULT_BLOCK_SIZE`] documents the measurements behind 8192.
    pub fn with_block_size(
        dim: usize,
        bit_width: usize,
        block_size: usize,
    ) -> Result<Self, ConstructError> {
        if block_size == 0 || block_size % MIN_BLOCK_SIZE != 0 {
            return Err(ConstructError::BlockSizeInvalid {
                block_size,
                granularity: MIN_BLOCK_SIZE,
            });
        }
        let mut ix = Self::new(dim, bit_width)?;
        ix.block_size = Some(block_size);
        Ok(ix)
    }

    /// Rows per calibration block, or `None` when one calibration covers
    /// the whole index. See [`Self::with_block_size`].
    pub fn block_size(&self) -> Option<usize> {
        self.block_size
    }

    /// Number of calibration blocks that have sealed. Always 0 for an
    /// index without a block size.
    pub fn sealed_blocks(&self) -> usize {
        self.sealed.len()
    }

    /// Number of storage slots, i.e. one past the largest slot index any
    /// live vector can occupy, and the length a `mask` passed to
    /// [`Self::search_with_mask`] must have.
    ///
    /// Equal to [`Self::len`] for every index without a block size. With
    /// one, a block-local [`Self::swap_remove`] leaves dead rows in the
    /// tail of the shortened block rather than renumbering every later
    /// slot, so this can exceed `len()`.
    pub fn slot_capacity(&self) -> usize {
        self.n_vectors
    }

    /// Whether `slot` currently holds a vector.
    ///
    /// Every slot below [`Self::slot_capacity`] is live unless a
    /// block-local [`Self::swap_remove`] has left it in the dead tail of
    /// a shortened block — see [`Self::with_block_size`]. Callers that
    /// keep their own slot-indexed tables need this to tell the two
    /// apart, since the storage extent does not shrink when an earlier
    /// block does.
    pub fn slot_is_live(&self, slot: usize) -> bool {
        if slot >= self.n_vectors {
            return false;
        }
        let Some(bs) = self.block_size else {
            return true;
        };
        match self.sealed.get(slot / bs) {
            Some(blk) => slot % bs < blk.len,
            None => true,
        }
    }

    /// First storage slot of the open block.
    fn open_base(&self) -> usize {
        match self.block_size {
            Some(bs) => self.sealed.len() * bs,
            None => 0,
        }
    }

    /// The blocks a search has to score, as
    /// `(base_slot, live_rows, shift, scale)` in slot order. Exactly one
    /// entry for an index with no block size, so the single-block path
    /// stays the one this crate has always taken.
    fn block_layout(&self) -> Vec<(usize, usize, &[f32], &[f32])> {
        let mut out = Vec::with_capacity(self.sealed.len() + 1);
        let bs = self.block_size.unwrap_or(0);
        for (b, blk) in self.sealed.iter().enumerate() {
            out.push((b * bs, blk.len, blk.shift.as_slice(), blk.scale.as_slice()));
        }
        let base = self.open_base();
        let open_len = self.n_vectors - base;
        if open_len > 0 || out.is_empty() {
            out.push((
                base,
                open_len,
                self.tqplus_shift.as_slice(),
                self.tqplus_scale.as_slice(),
            ));
        }
        out
    }

    /// Whether this index will fit a TQ+ calibration.
    ///
    /// `false` only for indexes built by [`Self::new_uncalibrated`] /
    /// [`Self::new_lazy_uncalibrated`]. Note that a `true` here does not
    /// mean a calibration *is* fitted — see
    /// [`Self::calibration_state`] for that.
    pub fn calibration_enabled(&self) -> bool {
        self.calibration_enabled
    }

    /// Commit an explicit identity `(shift, scale)` and drop the warm-up
    /// buffer, which is exactly [`CalibrationState::Identity`].
    ///
    /// Explicit rather than empty: an empty pair means "nothing fitted
    /// yet" and makes the next `encode` fit one from its batch, which is
    /// the opposite of what an uncalibrated index wants.
    fn commit_identity_calibration(&mut self, dim: usize) {
        self.tqplus_shift = vec![0.0; dim];
        self.tqplus_scale = vec![1.0; dim];
        self.warmup = None;
        // Nothing will ever be refitted, so the open block has no use
        // for its rows — and keeping them would cost `block_size * dim`
        // floats for a buffer no seal reads.
        self.open_rows = None;
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
            dead_slots: 0,
            packed_codes: OnceLock::from(Vec::new()),
            scales: Vec::new(),
            tqplus_shift: Vec::new(),
            tqplus_scale: Vec::new(),
            warmup: Some(Vec::new()),
            calibration_enabled: true,
            block_size: Some(DEFAULT_BLOCK_SIZE),
            sealed: Vec::new(),
            open_rows: Some(Vec::new()),
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
            encode_scratch: Vec::new(),
            encode_scratch_prev: 0,
        })
    }

    /// Restore the index to the state its constructor left it in: lazy,
    /// empty, and carrying nothing shaped for any dim.
    ///
    /// Rebuilt from [`Self::new_lazy`] rather than by putting a saved
    /// value back in each field. `add` commits *per block*: a chunk
    /// publishes `packed_codes`, `scales` and `n_vectors`, `open_rows`
    /// grows, and a full block pushes a `SealedBlock` carrying a
    /// dim-shaped pair. A field-by-field rollback has to name all of
    /// them, and the next field added to this type falls off the list
    /// silently — which is how the row and block state came to be
    /// missing from it. Taking the constructor's own output means a new
    /// field gets its lazy default here for free.
    ///
    /// Construction-time choices are the exception, since they are not
    /// state the add produced: the bit width, the calibration opt-out,
    /// the block size, and the scratch allocation, which is pure derived
    /// state and worth keeping rather than reallocating on the retry.
    fn reset_to_lazy(&mut self) {
        let mut fresh =
            Self::new_lazy(self.bit_width).expect("bit_width was validated at construction");
        fresh.calibration_enabled = self.calibration_enabled;
        fresh.block_size = self.block_size;
        if !self.calibration_enabled {
            // `new_lazy` arms the open-block buffer; an uncalibrated
            // index has no refit to feed it to.
            fresh.open_rows = None;
        }
        fresh.encode_scratch = std::mem::take(&mut self.encode_scratch);
        fresh.encode_scratch_prev = self.encode_scratch_prev;
        *self = fresh;
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

        // Calibration turned off at construction: commit the explicit
        // identity pair the moment the dim is known, which drops the
        // warm-up buffer and puts the index in `Identity` for good. Done
        // here rather than in the constructor because a lazy index has
        // no dim until now, and the pair is dim-shaped. Idempotent: once
        // committed, `warmup` is `None` and the branch below is skipped
        // anyway.
        if !self.calibration_enabled && self.warmup.is_some() {
            self.commit_identity_calibration(dim);
        }

        // Per-block calibration: never let one call span a block
        // boundary. Split the batch at the open block's remaining
        // capacity, seal when the block fills, and keep going. Each
        // slice takes exactly the path a whole batch takes on a
        // single-block index, so nothing below this point has to know
        // blocks exist.
        if let Some(bs) = self.block_size {
            let mut off = 0usize;
            while off < n {
                let capacity = bs - (self.n_vectors - self.open_base());
                let take = (n - off).min(capacity);
                self.add_to_open_block(&vectors[off * dim..(off + take) * dim], take, dim);
                if self.n_vectors - self.open_base() == bs {
                    self.seal_open_block(dim);
                }
                off += take;
            }
        } else {
            self.add_to_open_block(vectors, n, dim);
        }

        // Keep the scratch warm for same-size adds, but don't let a
        // one-time huge bulk load pin its full rotated-batch capacity
        // for the index lifetime (#333). Decided here, against the
        // *caller's* batch, rather than inside the encode: the block
        // split above means the encode never sees more than one block at
        // a time, so deciding per chunk would read a 100k-row bulk load
        // as a steady stream of 8192-row ones and retain a block's worth
        // of scratch for good.
        self.encode_scratch_prev = retain_scratch(&mut self.encode_scratch, self.encode_scratch_prev, n * dim);
    }

    /// Add `n` rows that are known to fit inside the open block, then
    /// record them in the open block's raw-row buffer.
    ///
    /// The buffer is extended *after* the encode, not before: the encode
    /// paths below all restore the index to its pre-call state on an
    /// unwind, so extending first would leave the buffer holding rows
    /// the index does not — the same ordering `warmup` keeps (#353).
    fn add_to_open_block(&mut self, vectors: &[f32], n: usize, dim: usize) {
        self.add_within_open_block(vectors, n, dim);
        if let Some(buf) = self.open_rows.as_mut() {
            buf.extend_from_slice(vectors);
        }
    }

    /// Holds the whole of the pre-block-model `add` body: warm-up
    /// buffering, the TQ+ threshold crossing, and the plain append.
    fn add_within_open_block(&mut self, vectors: &[f32], n: usize, dim: usize) {

        // Warm-up phase: the index has not yet seen enough vectors to fit
        // a real TQ+ calibration. Keep the raw rows so that, once the
        // total crosses the threshold, everything can be re-encoded in a
        // properly fitted coordinate system rather than being frozen to
        // identity by whatever the first add happened to contain.
        if let Some(buffered) = self.warmup.as_ref().map(|b| b.len() / dim) {
            if buffered + n < encode::TQPLUS_MIN_SAMPLES {
                // Still below the threshold: buffer the rows and encode
                // them under identity so the index stays fully
                // searchable and serializable in the meantime. Declared
                // calibration stays identity, which is exactly how these
                // codes were produced.
                // Encode FIRST. `encode_and_append` has an unwind guard
                // that restores the index to its pre-call state without
                // incrementing `n_vectors`, so extending the buffer
                // before it would leave `warmup.len()/dim` ahead of
                // `n_vectors` after a caught panic — permanently breaking
                // "buffer row i is slot i" and resurrecting the failed
                // batch's rows into the re-encode (#353).
                self.encode_and_append(vectors, n, dim);
                self.warmup
                    .as_mut()
                    .expect("warmup is Some in this branch")
                    .extend_from_slice(vectors);
                return;
            }
            // Threshold crossed. Leave warm-up for good.
            if self
                .warmup
                .as_ref()
                .expect("warmup is Some in this branch")
                .is_empty()
            {
                // Nothing to re-encode — this batch alone fits the
                // calibration, which is the plain bulk-add path.
                //
                // An empty buffer means no stored rows (the buffer holds
                // one row per slot), so whatever calibration is committed
                // here describes nothing: it is the non-empty *identity*
                // an earlier sub-threshold add committed, whose rows have
                // since all been `swap_remove`d. Discard it, or `encode`
                // sees `existing = Some(identity)`, takes the reuse path,
                // declines to fit — and the index is frozen to identity
                // for the rest of its life while `calibration_state()`
                // still reports a recoverable `WarmingUp` (#360, #366).
                // Both halves are cleared in one statement: an empty
                // `shift` beside a length-`dim` `scale` is a state no
                // other path in this type can produce.
                (self.tqplus_shift, self.tqplus_scale) = (Vec::new(), Vec::new());
                // Encode AFTER that, and set `warmup` only once it
                // returns, for the same reason the sub-threshold branch
                // does: a caught panic must leave the index still warming
                // up rather than silently forfeiting TQ+ (#361). Clearing
                // the calibration first is safe under the same rule —
                // with no stored rows there is nothing it could
                // mis-declare, so the unwound index is an empty
                // warming-up one either way.
                self.encode_and_append(vectors, n, dim);
                self.warmup = None;
                return;
            }
            let buffer = self.warmup.take().expect("warmup is Some in this branch");
            // Fit the calibration up front so the buffered rows and this
            // batch land in the same coordinate system, then re-encode
            // the buffered rows (their identity-encoded codes are
            // discarded) followed by this batch. Slot order is preserved,
            // so external id maps stay valid.
            //
            // The fit sample is this batch when it alone clears the
            // threshold (a copy of a potentially huge batch would be the
            // only way to include the <1000 buffered rows, and they could
            // not move the quantiles anyway); otherwise the concatenation
            // of buffer + batch, which is at most ~2000 rows.
            let rotation = self.rotation.get_or_init(|| rotation::Rotation::new(dim));
            // The fit anchors on the codebook's outermost centroid (#454),
            // so the same cached codebook the encode path uses has to be
            // in hand before fitting. Seed both locks from the one solve,
            // as `encode_and_append` does: this closure is dead today
            // (reaching here needs a non-empty warm-up buffer, and
            // buffering went through `encode_and_append`, which already
            // seeded both), but if a refactor ever makes it live, filling
            // only `centroids` would leave the `encode_and_append` below
            // to re-run the Lloyd-Max solve for `boundaries`.
            if self.centroids.get().is_none() || self.boundaries.get().is_none() {
                let (b, c) = codebook::codebook(self.bit_width, dim);
                let _ = self.boundaries.set(b);
                let _ = self.centroids.set(c);
            }
            let centroids = self.centroids.get().expect("seeded above");
            let mut scratch = std::mem::take(&mut self.encode_scratch);
            let concat;
            let (fit_src, fit_n): (&[f32], usize) = if n >= encode::TQPLUS_MIN_SAMPLES {
                (vectors, n)
            } else {
                concat = [buffer.as_slice(), vectors].concat();
                (concat.as_slice(), buffered + n)
            };
            // The crossing rewrites every row, so it necessarily commits
            // state that must stay in step with `n_vectors` (the codes,
            // the calibration, `warmup` itself) before all of the
            // fallible work is done. Both fallible steps therefore run
            // under an unwind guard that restores the whole pre-crossing
            // state, so a caught panic leaves a still-warming-up index
            // with its rows intact instead of an empty one (#361) or one
            // that has silently forfeited TQ+ for good.
            let fitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                #[cfg(test)]
                if FORCE_FIT_PANIC.with(|f| f.replace(false)) {
                    panic!("forced calibration fit panic (test)");
                }
                encode::fit_calibration(fit_src, fit_n, dim, rotation, centroids, &mut scratch)
            }));
            self.encode_scratch = scratch;
            let (shift, scale_tq) = match fitted {
                Ok(pair) => pair,
                Err(panic) => {
                    self.warmup = Some(buffer);
                    std::panic::resume_unwind(panic);
                }
            };
            // Drop the identity-encoded rows and commit the fitted
            // calibration before re-encoding, so both encode calls below
            // take the reuse path with the new calibration — holding on
            // to the old values so the guard below can put them back.
            let old_packed = std::mem::replace(&mut self.packed_codes, OnceLock::from(Vec::new()));
            let old_scales = std::mem::take(&mut self.scales);
            let old_blocked = std::mem::replace(&mut self.blocked, OnceLock::new());
            let old_n = self.n_vectors;
            let old_shift = std::mem::replace(&mut self.tqplus_shift, shift);
            let old_scale = std::mem::replace(&mut self.tqplus_scale, scale_tq);
            self.n_vectors = 0;
            let reencoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.encode_and_append(&buffer, buffered, dim);
                self.encode_and_append(vectors, n, dim);
            }));
            if let Err(panic) = reencoded {
                self.packed_codes = old_packed;
                self.scales = old_scales;
                self.blocked = old_blocked;
                self.n_vectors = old_n;
                self.tqplus_shift = old_shift;
                self.tqplus_scale = old_scale;
                self.warmup = Some(buffer);
                std::panic::resume_unwind(panic);
            }
            return;
        }

        self.encode_and_append(vectors, n, dim);
    }

    /// Seal the open block: fit a calibration from its own rows,
    /// re-encode them under that fit, freeze it, and open the next
    /// block.
    ///
    /// Called only with a full open block. When the block's raw rows are
    /// not available — a loaded index, or one that opted out of
    /// calibration — the block seals on the calibration it already
    /// carries and nothing is re-encoded; the pair is what its rows were
    /// encoded with either way, which is the only property sealing has
    /// to preserve.
    ///
    /// The next block starts on the sealed block's pair as a provisional
    /// calibration rather than warming up again from identity. Warming
    /// up would mean re-encoding the new block's first 1000 rows in
    /// place, and that re-encode rebuilds the code buffers from slot 0 —
    /// an O(n) copy per block, i.e. quadratic over an index's life. The
    /// provisional pair is only ever what the block's rows are *stored*
    /// in until it seals and refits from itself.
    fn seal_open_block(&mut self, dim: usize) {
        let bs = self.block_size.expect("seal requires a block size");
        let base = self.open_base();
        debug_assert_eq!(self.n_vectors, base + bs, "sealing a block that is not full");

        if let Some(rows) = self.open_rows.take() {
            debug_assert_eq!(rows.len(), bs * dim);
            self.refit_and_reencode_open_block(&rows, base, bs, dim);
        }
        self.sealed.push(SealedBlock {
            shift: self.tqplus_shift.clone(),
            scale: self.tqplus_scale.clone(),
            len: bs,
        });
        // Leave the global warm-up for good, whether or not it ever
        // crossed its threshold. Blocks refit from their own rows, so it
        // has nothing left to do — and leaving it armed is actively
        // wrong, because the crossing rewrites *every* stored row from
        // slot 0 under one freshly fitted pair. That path predates
        // blocks and is only sound while none have sealed: run it after
        // a seal and the sealed rows are re-encoded under the new global
        // pair while `sealed[b]` still declares the pair they were
        // sealed with, so search decodes them in a coordinate system
        // they are no longer stored in.
        //
        // Reachable whenever `block_size` is below TQPLUS_MIN_SAMPLES:
        // the block fills before the buffer does, so the seal happens
        // first and the buffer keeps accumulating across it. Measured on
        // a drifting stream at dim 64, 4 bits, 4096 rows — top-1
        // self-recall 0.025 at a 64-row block size against 0.998 at
        // 1024. It compounds with removes: `swap_remove` only mirrors
        // the buffer for the open block, so a removal from a sealed one
        // left the deleted row in the buffer and the crossing wrote it
        // back into a live slot.
        //
        // The consequence is that a block smaller than
        // TQPLUS_MIN_SAMPLES seals on identity — `fit_calibration`
        // declines to fit below that many samples — so such an index
        // gets no TQ+ gain. That is the honest reading of the sample
        // floor, and it is what [`Self::with_block_size`] documents.
        self.warmup = None;
        // The freshly opened block buffers its own rows only if there is
        // a refit for them to feed. An uncalibrated index has none by
        // construction; a loaded one has no rows for the block it was
        // loaded mid-way through, but every block it opens from here on
        // is built entirely in memory, so it does.
        self.open_rows = if self.calibration_enabled {
            Some(Vec::new())
        } else {
            None
        };
    }

    /// Fit `(shift, scale)` from the open block's own `bs` rows and
    /// rewrite the block's stored codes under it.
    ///
    /// Ordered so that nothing is committed until every fallible step
    /// has succeeded: the fit and the re-encode both run before any
    /// stored byte is touched, and the re-encode targets scratch buffers
    /// rather than the index's. A panic in either therefore leaves the
    /// block exactly as it was, still encoded under its provisional
    /// calibration and still searchable.
    fn refit_and_reencode_open_block(&mut self, rows: &[f32], base: usize, bs: usize, dim: usize) {
        let rotation = self.rotation.get_or_init(|| rotation::Rotation::new(dim));
        if self.boundaries.get().is_none() || self.centroids.get().is_none() {
            let (b, c) = codebook::codebook(self.bit_width, dim);
            let _ = self.boundaries.set(b);
            let _ = self.centroids.set(c);
        }
        let boundaries = self.boundaries.get().expect("seeded above");
        let centroids = self.centroids.get().expect("seeded above");
        let mut scratch = std::mem::take(&mut self.encode_scratch);
        #[cfg(test)]
        if FORCE_FIT_PANIC.with(|f| f.replace(false)) {
            self.encode_scratch = scratch;
            panic!("forced calibration fit panic (test)");
        }
        let (shift, scale) =
            encode::fit_calibration(rows, bs, dim, rotation, centroids, &mut scratch);
        let bytes_per_vec = dim * self.bit_width / 8;
        let mut new_packed = Vec::with_capacity(bs * bytes_per_vec);
        let mut new_scales = Vec::with_capacity(bs);
        encode::encode(
            rows,
            bs,
            dim,
            rotation,
            boundaries,
            centroids,
            self.bit_width,
            Some((shift.as_slice(), scale.as_slice())),
            &mut scratch,
            &mut new_packed,
            &mut new_scales,
        );
        self.encode_scratch = scratch;

        // Commit. `packed()` materializes the rows a v6 load left
        // implicit; there is no lazy-append shortcut here because the
        // rewrite is not an append.
        self.packed();
        let packed = self.packed_mut();
        packed[base * bytes_per_vec..].copy_from_slice(&new_packed);
        self.scales[base..].copy_from_slice(&new_scales);
        self.tqplus_shift = shift;
        self.tqplus_scale = scale;

        // Bring the blocked cache back in step with the rows it
        // describes. Only this block's SIMD blocks changed, so recompute
        // exactly those. If the recompute fails, drop the cache rather
        // than leave a stale one behind: the packed rows are already
        // committed and authoritative, so a cold cache is merely slow
        // whereas a stale one mis-scores.
        if self.blocked.get().is_some() {
            let (n_blocks, n_byte_groups, _) =
                pack::blocked_geometry(self.n_vectors, self.bit_width, dim);
            let block_bytes = n_byte_groups * BLOCK;
            let first_block = base / BLOCK;
            let bit_width = self.bit_width;
            let n_vectors = self.n_vectors;
            let patch = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pack::repack_block_range(
                    self.packed(),
                    n_vectors,
                    bit_width,
                    dim,
                    first_block,
                    n_blocks,
                )
            })) {
                Ok(patch) => patch,
                Err(panic) => {
                    self.blocked = OnceLock::new();
                    std::panic::resume_unwind(panic);
                }
            };
            let cache = self.blocked.get_mut().expect("blocked present");
            cache.data.truncate(first_block * block_bytes);
            cache.data.extend_from_slice(&patch);
            cache.n_blocks = n_blocks;
        }
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

    /// Test-only sibling of [`Self::force_encode_panic`] that fires on
    /// the `n`th encode after arming instead of the first, so a test can
    /// reach a panic on a later chunk of a batch `add` split across
    /// calibration blocks. `0` disarms.
    #[cfg(test)]
    pub(crate) fn force_encode_panic_at(n: usize) {
        FORCE_ENCODE_PANIC_AT.with(|c| c.set(n));
    }

    /// Sibling of [`Self::force_encode_panic`] for the calibration fit,
    /// thread-local for the same reason (#373). The threshold crossing
    /// fits before it re-encodes, and the two failure points have to roll
    /// back different state, so they need separately targetable switches.
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
        // On subsequent adds, reuse the calibration fitted on the first
        // batch so all vectors live in the same calibrated coord system.
        // On the first add, encode() fits a fresh calibration.
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
            #[cfg(test)]
            if FORCE_ENCODE_PANIC_AT.with(|c| {
                let left = c.get();
                if left > 0 {
                    c.set(left - 1);
                }
                left == 1
            }) {
                panic!("forced encode panic on the nth call (test)");
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
        let (shift, scale_tq) = match encode_result {
            Ok(pair) => pair,
            Err(panic) => {
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
        };
        self.encode_scratch = scratch;
        // `scales` is published per branch below, at the same commit point
        // as the codes and the count — publishing it here would leave it
        // holding `new_n` rows if the eager branch's cache patch panicked
        // (#388).

        // Commit only what `encode` actually produced. It returns a
        // non-empty pair exactly when it fitted one for this batch (i.e.
        // `existing` was `None`); on the reuse path it returns empty
        // vectors, and assigning those would declare identity
        // calibration over codes encoded with the committed one —
        // silently wrong scores, and an `n_calib = 0` trailer on the
        // next write. That is what the old `n_vectors == 0` guard did
        // after an index was drained to empty and re-added to (#284).
        if !shift.is_empty() {
            self.tqplus_shift = shift;
            self.tqplus_scale = scale_tq;
        }
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
            // guard restores the code and scale buffers.
            //
            // The calibration triple goes back too. An uncalibrated index
            // commits its dim-shaped identity `(shift, scale)` and drops
            // `warmup` *before* the fallible encode — the encode needs the
            // pair already in place, or a large first batch fits a real
            // calibration instead of staying identity — so those three
            // are also touched before the encode and would otherwise
            // survive the unwind. That leaves an index reporting
            // `dim == None` and `len() == 0` while holding a calibration
            // shaped for the abandoned dim: a retry at a different dim
            // then trips `existing shift length must equal dim` inside
            // `encode`, and `to_bytes` panics on a dim-0 sentinel beside
            // a length-`dim` pair. Both are raw asserts rather than an
            // `AddError`, and the retry's own panic re-enters this same
            // rollback, so the wedge is permanent.
            self.dim = Some(dim);
            if let Err(panic) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.add(vectors)))
            {
                self.reset_to_lazy();
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
        let layout = self.block_layout();
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
            for chunk in m.chunks(64) {
                let mut word = 0u64;
                for (bit, &b) in chunk.iter().enumerate() {
                    word |= (b as u64) << bit;
                }
                buf.push(word);
            }
            debug_assert_eq!(buf.len(), n_words);
            // Clear the bits of slots no vector lives in — the dead tail
            // of a block a `swap_remove` shortened. The caller's mask is
            // one bool per *slot*, so it has an entry for those, and
            // whatever it says about them must not be counted into the
            // result width or the kernel would be asked for more matches
            // than exist. Per-block scoring already refuses to read them.
            let mut allowed = 0usize;
            for &(base, live, _, _) in &layout {
                for slot in base..base + live {
                    allowed += ((buf[slot / 64] >> (slot % 64)) & 1) as usize;
                }
            }
            for &(base, live, _, _) in &layout {
                for slot in base + live..(base + live).next_multiple_of(64) {
                    if slot < self.n_vectors {
                        buf[slot / 64] &= !(1u64 << (slot % 64));
                    }
                }
            }
            (buf, allowed)
        });

        let n_live = self.len();
        let n_allowed = packed_mask.as_ref().map_or(n_live, |p| p.1);
        let packed_mask = packed_mask.map(|p| p.0);
        let effective_k = k.min(n_live).min(n_allowed);

        // One search per calibration block, each against its own slice
        // of the codes, scales and mask. Block `b` starts at slot
        // `b * block_size`, which is a multiple of both the 32-row SIMD
        // block and the 64-slot mask word, so every slice is exact and
        // the kernel's range-relative indexing lands where it should;
        // `n_byte_groups` is a function of `dim` and `bit_width` alone,
        // so the byte offset is just the block's first SIMD block.
        let (_, n_byte_groups, _) = pack::blocked_geometry(self.n_vectors, self.bit_width, dim);
        let block_bytes = n_byte_groups * BLOCK;
        let mut per_block = Vec::with_capacity(layout.len());
        for &(base, live, shift, scale) in &layout {
            if live == 0 {
                continue;
            }
            let (n_blocks, _, _) = pack::blocked_geometry(live, self.bit_width, dim);
            let byte_start = (base / BLOCK) * block_bytes;
            let mask_slice = packed_mask.as_deref().map(|m| {
                let w0 = base / 64;
                &m[w0..w0 + live.div_ceil(64)]
            });
            let (scores, indices) = search::search(
                queries,
                nq,
                rotation,
                &blocked.data[byte_start..byte_start + n_blocks * block_bytes],
                centroids,
                &self.scales[base..base + live],
                shift,
                scale,
                self.bit_width,
                dim,
                live,
                n_blocks,
                k,
                mask_slice,
            );
            per_block.push((base, scores, indices));
        }

        // One block *starting at slot 0* is the whole index: its
        // results are already the answer and rebasing is a no-op, so
        // return them untouched rather than round-tripping them through
        // the merge below. That keeps every index without a block size
        // on exactly the code path it has always taken.
        //
        // The base check is not redundant. Blocks with no live rows are
        // skipped above, and a sealed block keeps its extent when it is
        // drained, so an index whose earlier blocks have all been
        // emptied leaves exactly one entry here with `base > 0` — and
        // every slot it returned came back off by that base.
        if per_block.len() == 1 && per_block[0].0 == 0 {
            let (_, scores, indices) = per_block.pop().expect("len 1");
            return Ok(SearchResults { scores, indices, nq, k: effective_k });
        }

        let mut scores = Vec::with_capacity(nq * effective_k);
        let mut indices = Vec::with_capacity(nq * effective_k);
        let mut merged: Vec<(f32, i64)> = Vec::new();
        for qi in 0..nq {
            merged.clear();
            for (base, block_scores, block_indices) in &per_block {
                let kb = block_scores.len() / nq;
                for j in 0..kb {
                    merged.push((
                        block_scores[qi * kb + j],
                        block_indices[qi * kb + j] + *base as i64,
                    ));
                }
            }
            // Descending by score, ascending by slot on a tie — the same
            // order the per-block kernels already merge their own thread
            // ranges in, so a result set does not depend on how the rows
            // happen to be divided into blocks.
            merged.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for &(s, i) in merged.iter().take(effective_k) {
                scores.push(s);
                indices.push(i);
            }
        }

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

    /// The per-block calibration a v8 file carries: the sealed blocks'
    /// frozen pairs and live lengths, plus the open block's raw rows.
    ///
    /// Empty for an index with no block size, which is what makes such
    /// an index's file identical in content to what it was before v8
    /// (sixteen bytes of zeroed table aside).
    ///
    /// The open rows go in because a block seals by refitting from them:
    /// a file that dropped them would leave the reloaded index to seal
    /// on whatever provisional calibration it was carrying, silently
    /// giving up the refit for the block that happened to be open when
    /// the index was saved.
    ///
    /// They ride along only while they cost no more than the codes they
    /// exist to improve. Raw `f32` is 8 to 16 times the size of the
    /// codes for the same rows, so writing them unconditionally makes
    /// the file 9x larger at 4 bits and 17x at 2 bits for any index
    /// below one full block — where they buy nothing anyway, since such
    /// an index has no sealed block whose calibration the open one would
    /// otherwise be stuck with. The rule crosses over at around eight
    /// blocks and its cost falls away from there: 1.66x of a 100k-row
    /// index at dim 1536 and 4 bits, 1.07x at a million rows.
    pub(crate) fn block_table_for_write(&self) -> io::BlockTable {
        let Some(block_size) = self.block_size else {
            return io::BlockTable::default();
        };
        let dim = self.dim.unwrap_or(0);
        let mut table = io::BlockTable {
            block_size,
            lens: Vec::with_capacity(self.sealed.len()),
            shift: Vec::with_capacity(self.sealed.len() * dim),
            scale: Vec::with_capacity(self.sealed.len() * dim),
            open_rows: Vec::new(),
        };
        if let Some(rows) = self.open_rows.as_ref() {
            let codes_bytes = self.n_vectors * dim * self.bit_width / 8;
            if rows.len() * 4 <= codes_bytes {
                table.open_rows = rows.clone();
            }
        }
        for blk in &self.sealed {
            table.lens.push(blk.len);
            table.shift.extend_from_slice(&blk.shift);
            table.scale.extend_from_slice(&blk.scale);
        }
        table
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
    /// # Saving while still warming up
    ///
    /// The format carries no warm-up buffer, so an index that holds at
    /// least one vector and whose
    /// [`calibration_state`](Self::calibration_state) is
    /// [`WarmingUp`](CalibrationState::WarmingUp) writes an identity TQ+
    /// trailer and the **reloaded** copy is committed to
    /// [`Identity`](CalibrationState::Identity) for its whole life: it
    /// never fits a real calibration however many vectors are added
    /// afterwards, and gives up the TQ+ recall gain (most of it at 2
    /// bits). This index is unaffected — it keeps its buffer. So "save
    /// at 500 vectors, reload, add 10,000" produces a permanently weaker
    /// index than adding all 10,500 to one index does. Add at least 1000
    /// vectors before saving, or rebuild the reloaded index from the
    /// original float32 vectors. The same applies to
    /// [`Self::write_with_durability`], [`Self::write_to_writer`] and
    /// [`Self::to_bytes`].
    ///
    /// An index holding **zero** vectors is the exception: it has
    /// nothing encoded under identity, so it round-trips back into
    /// [`WarmingUp`](CalibrationState::WarmingUp) and the next add can
    /// still fit a real calibration (#418). That covers a drained
    /// warming-up index and a drained identity one. A drained
    /// [`Fitted`](CalibrationState::Fitted) index also holds zero
    /// vectors but writes its real calibration, so it reloads
    /// `Fitted` and keeps it (#284).
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
                    self.calibration_enabled,
                    &self.block_table_for_write(),
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
                    self.calibration_enabled,
                    &self.block_table_for_write(),
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
            self.calibration_enabled,
            &self.block_table_for_write(),
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
                self.calibration_enabled,
                &self.block_table_for_write(),
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
            self.calibration_enabled,
            &self.block_table_for_write(),
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
            &self.block_table_for_write(),
        )
    }

    /// Serialize the index to `.tv`-format bytes in memory —
    /// byte-identical to the file [`Self::write`] produces. Pairs with
    /// [`Self::from_bytes`] for callers that persist the index through
    /// their own storage (a database column, a cache, a pickle payload)
    /// instead of the filesystem.
    ///
    /// Serializing an index that is still
    /// [`WarmingUp`](CalibrationState::WarmingUp) commits the
    /// deserialized copy to [`Identity`](CalibrationState::Identity)
    /// calibration for good — see [`Self::write`] for the full statement.
    /// This is the path a clone-by-round-trip takes, so a copy of a
    /// sub-1000-vector index is weaker than the original it was copied
    /// from, which keeps its warm-up buffer.
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
        parts: (usize, usize, usize, io::CodePayload, Vec<f32>, Vec<f32>, Vec<f32>, bool, io::BlockTable),
    ) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale, calibration_enabled, blocks) = parts;
        let dim_opt = if dim == 0 { None } else { Some(dim) };
        let (block_size, sealed, dead_slots, open_rows) = Self::blocks_from_table(blocks, dim)?;
        match codes {
            // Seed the search cache directly from the blocked
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
                // Same identity-population / warm-up decision
                // `from_parts` applies on the v5 arm — skipping it left
                // a v6 file with an empty TQ+ trailer able to swallow a
                // later add whole (#303).
                let (tqplus_shift, tqplus_scale, warmup) =
                    Self::normalize_calibration(dim_opt, n_vectors, tqplus_shift, tqplus_scale, calibration_enabled);
                Ok(Self {
                    dim: dim_opt,
                    bit_width,
                    n_vectors,
                    dead_slots,
                    packed_codes,
                    scales,
                    tqplus_shift,
                    tqplus_scale,
                    warmup,
                    calibration_enabled,
                    block_size,
                    sealed,
                    open_rows,
                    encode_scratch: Vec::new(),
                    encode_scratch_prev: 0,
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
                // Same identity-population / warm-up decision
                // `from_parts` applies on the v5 arm — skipping it left
                // a v6 file with an empty TQ+ trailer able to swallow a
                // later add whole (#303).
                let (tqplus_shift, tqplus_scale, warmup) =
                    Self::normalize_calibration(dim_opt, n_vectors, tqplus_shift, tqplus_scale, calibration_enabled);
                Ok(Self {
                    dim: dim_opt,
                    bit_width,
                    n_vectors,
                    dead_slots,
                    packed_codes,
                    scales,
                    tqplus_shift,
                    tqplus_scale,
                    warmup,
                    calibration_enabled,
                    block_size,
                    sealed,
                    open_rows,
                    encode_scratch: Vec::new(),
                    encode_scratch_prev: 0,
                    rotation: OnceLock::new(),
                    boundaries: boundaries_lock,
                    centroids: centroids_lock,
                    blocked,
                })
            }
        }
    }

    /// Rebuild the in-memory block state from a loaded [`io::BlockTable`]:
    /// `(block_size, sealed, dead_slots, open_rows)`.
    ///
    /// The io layer has already checked the table's shape against the
    /// file's own geometry. What is checked here is the one thing it
    /// cannot see: that the pairs are `dim`-shaped, since the io layer
    /// validates lengths in aggregate and a `dim` of zero would let a
    /// table through that describes blocks with no coordinates.
    fn blocks_from_table(
        table: io::BlockTable,
        dim: usize,
    ) -> std::io::Result<(Option<usize>, Vec<SealedBlock>, usize, Option<Vec<f32>>)> {
        if table.block_size == 0 {
            return Ok((None, Vec::new(), 0, None));
        }
        // A lazy index has a block size and nothing else — no dim to
        // shape a pair with, and no rows to have sealed one. Keeping the
        // size across the round trip is what stops a reloaded lazy index
        // silently becoming a single-calibration one.
        if dim == 0 {
            if !table.lens.is_empty() || !table.open_rows.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "per-block calibration describes blocks on an index with no committed dim",
                ));
            }
            return Ok((Some(table.block_size), Vec::new(), 0, None));
        }
        let mut sealed = Vec::with_capacity(table.lens.len());
        let mut dead_slots = 0usize;
        for (b, &len) in table.lens.iter().enumerate() {
            sealed.push(SealedBlock {
                shift: table.shift[b * dim..(b + 1) * dim].to_vec(),
                scale: table.scale[b * dim..(b + 1) * dim].to_vec(),
                len,
            });
            dead_slots += table.block_size - len;
        }
        // No rows in the file means no refit for the block that was open
        // when it was written — it seals on the calibration it carries.
        let open_rows = if table.open_rows.is_empty() {
            None
        } else {
            Some(table.open_rows)
        };
        Ok((Some(table.block_size), sealed, dead_slots, open_rows))
    }

    /// Normalize the calibration state every construction path must
    /// agree on, given the decoded `(tqplus_shift, tqplus_scale)` and the
    /// number of stored vectors. Returns the calibration to store plus
    /// the initial warm-up buffer.
    ///
    /// Two rules, both mandatory:
    ///
    /// - **Stored rows always come with a declared calibration.** A
    ///   payload with `n_vectors > 0` and an empty TQ+ pair (the v2 wire
    ///   shape, and what the public [`io`] writers permit) gets explicit
    ///   identity. Left empty, the next `add` would see the lazy
    ///   `existing = None` signal, fit a fresh calibration, encode the
    ///   new rows in it and — before #285's commit-site fix — drop it,
    ///   producing vectors that `len` counts but search can never
    ///   return (#303).
    /// - **Nothing stored and nothing declared means warm-up.** Such an
    ///   index is indistinguishable from a fresh one, so it may still
    ///   fit a real calibration from what arrives next. An *exactly
    ///   identity* pair declares no transform, so a payload carrying one
    ///   beside `n_vectors == 0` states nothing this rule does not
    ///   already cover, and is normalized to the same empty pair (#418).
    ///
    /// That second rule is what keeps the round trip in step with the
    /// in-memory behaviour. A sub-threshold `add` commits a *non-empty
    /// identity* pair for the rows it stores; `swap_remove`-ing them all
    /// away leaves that pair committed beside an empty warm-up buffer,
    /// and the live index stays recoverable — the threshold crossing
    /// discards a calibration that describes no stored rows. Without the
    /// exact-identity arm below, serializing that same index wrote a
    /// full-length identity trailer, `!tqplus_shift.is_empty()` took the
    /// early return, and the reloaded copy was committed to
    /// [`CalibrationState::Identity`] for the rest of its life while
    /// holding zero vectors (#418).
    ///
    /// The arm is deliberately narrow. A drained *fitted* index keeps its
    /// calibration on reload exactly as it does in memory (#284): its
    /// trailer is a real fit, not identity, so it does not match.
    fn normalize_calibration(
        dim: Option<usize>,
        n_vectors: usize,
        tqplus_shift: Vec<f32>,
        tqplus_scale: Vec<f32>,
        calibration_enabled: bool,
    ) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
        if !tqplus_shift.is_empty() {
            // Zero stored rows plus a pair that applies no transform is
            // the state a fresh index is already in, so restore the
            // warm-up buffer rather than freezing an empty index to a
            // calibration it never used (#418). Nothing is encoded under
            // the discarded pair — there is nothing stored at all.
            // An index that opted out of calibration keeps its committed
            // identity pair even with no rows: the pair is the whole
            // point, not an artefact of warming up. Without this an
            // uncalibrated index that was drained — or saved before its
            // first add — reloads as `WarmingUp` and fits a real
            // calibration on the next add, silently undoing the opt-out
            // (#457). The two states are byte-identical apart from this
            // flag, which is why v7 carries it.
            let declares_nothing = calibration_enabled
                && n_vectors == 0
                && tqplus_shift.iter().all(|&x| x == 0.0)
                && tqplus_scale.iter().all(|&x| x == 1.0);
            if declares_nothing {
                return (Vec::new(), Vec::new(), Some(Vec::new()));
            }
            return (tqplus_shift, tqplus_scale, None);
        }
        match dim {
            Some(d) if n_vectors > 0 => (vec![0.0; d], vec![1.0; d], None),
            _ => (tqplus_shift, tqplus_scale, Some(Vec::new())),
        }
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
    /// # Per-block calibration
    ///
    /// The parts carry a single `(shift, scale)` pair, so what they can
    /// describe is one calibration block. Up to
    /// [`DEFAULT_BLOCK_SIZE`] rows that is exactly what a
    /// [`Self::new`] index is, and the reconstruction is
    /// indistinguishable from the original — including byte-for-byte on
    /// [`Self::to_bytes`]. Above it the reconstruction is a
    /// single-calibration index: correct, and scoring exactly what its
    /// parts say, but not the same thing as an index of that size built
    /// by `new`, which would have sealed blocks with pairs of their own.
    /// Round-trip such an index through [`Self::to_bytes`] /
    /// [`Self::from_bytes`] instead — the file carries the block table.
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

        // Identity-population / warm-up decision — see
        // `normalize_calibration`. Shared with the v6 load arms so every
        // construction path lands in the same calibration state.
        // `from_parts` takes no calibration flag: its caller supplies the
        // pair directly, so "was calibration ever enabled" is not a
        // question the parts answer. `true` reproduces the pre-v7
        // behaviour exactly — the only difference the flag makes is
        // keeping a committed identity pair on a *rowless* index, which
        // a caller wanting that can express by passing rows.
        let (tqplus_shift, tqplus_scale, warmup) =
            Self::normalize_calibration(dim, n_vectors, tqplus_shift, tqplus_scale, true);
        Ok(Self {
            dim,
            bit_width,
            n_vectors,
            dead_slots: 0,
            packed_codes: OnceLock::from(packed_codes),
            scales,
            tqplus_shift,
            tqplus_scale,
            warmup,
            calibration_enabled: true,
            // The parts carry one calibration pair and no block table,
            // so what they describe is a single open block. That is the
            // default index's state right up until its first seal, so
            // adopt the default block size while the rows still fit in
            // one block — which keeps `from_parts` round-tripping to
            // byte-identical output — and fall back to a single
            // calibration for the whole index when they do not.
            block_size: (n_vectors <= DEFAULT_BLOCK_SIZE).then_some(DEFAULT_BLOCK_SIZE),
            sealed: Vec::new(),
            // No float rows to refit from: the parts are codes.
            open_rows: None,
            rotation: OnceLock::new(),
            boundaries: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
            encode_scratch: Vec::new(),
            encode_scratch_prev: 0,
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

    /// Whether this index has a TQ+ calibration fitted, is still warming
    /// up towards one, or is committed to identity for good. See
    /// [`CalibrationState`].
    pub fn calibration_state(&self) -> CalibrationState {
        if self.warmup.is_some() {
            return CalibrationState::WarmingUp;
        }
        let identity = self.tqplus_shift.iter().all(|&s| s == 0.0)
            && self.tqplus_scale.iter().all(|&s| s == 1.0);
        if identity {
            CalibrationState::Identity
        } else {
            CalibrationState::Fitted
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
        // The vector that fills the hole comes from `idx`'s own
        // calibration block. Taking the index's last row instead would
        // move codes into a block whose `(shift, scale)` is not the one
        // they were quantized under, so the row would decode to a
        // different vector than the one that was stored — the whole
        // reason a block-local move is the only correct one here.
        //
        // Only the *open* block can give its storage back: shortening
        // any earlier block would renumber every slot after it, which is
        // both an O(block_size * dim) memmove (~6 MB at dim 1536) and a
        // silent renumbering of slots every caller is holding. So a
        // sealed block keeps its extent and the vacated tail row simply
        // stops being live.
        let (base, live) = self.live_block_of(idx);
        let last = base + live - 1;
        let in_open_block = base == self.open_base();

        // n_vectors > 0 (implied above) means a successful add, which
        // implies self.dim was committed at that point. Unwrap is safe.
        let dim = self.dim.expect("n_vectors > 0 but dim is None");
        let bytes_per_vec = dim * self.bit_width / 8;
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
            if in_open_block {
                self.packed_mut().truncate(last * bytes_per_vec);
            } else {
                // The row stays in the storage extent, so leave it
                // determinate rather than a stale copy of the vector
                // that just moved.
                self.packed_mut()[last * bytes_per_vec..(last + 1) * bytes_per_vec].fill(0);
            }
        }

        if idx != last {
            // Move last norm into slot `idx`.
            self.scales[idx] = self.scales[last];
        }
        if in_open_block {
            self.scales.truncate(last);
            self.n_vectors -= 1;
        } else {
            self.scales[last] = 0.0;
            self.sealed[base / self.block_size.expect("sealed block implies a block size")].len -=
                1;
            self.dead_slots += 1;
        }

        // The warm-up buffer and the open block's raw-row buffer both
        // hold one row per open-block slot in slot order, so they take
        // the same swap-remove. Keeping them aligned is what lets a
        // later threshold-crossing add — or the block's own seal —
        // re-encode the survivors into their existing slots. Neither
        // describes a sealed block, so a removal from one leaves them
        // alone.
        if in_open_block {
            let open_base = base;
            for buf in [self.warmup.as_mut(), self.open_rows.as_mut()]
                .into_iter()
                .flatten()
            {
                let local_idx = idx - open_base;
                let local_last = last - open_base;
                if local_idx != local_last {
                    let (head, tail) = buf.split_at_mut(local_last * dim);
                    head[local_idx * dim..(local_idx + 1) * dim].copy_from_slice(&tail[..dim]);
                }
                buf.truncate(local_last * dim);
            }
        }

        // Maintain the blocked cache with O(dim) lane ops: copy the last
        // vector's lane into the vacated slot, zero the vacated last lane
        // (serialization copies the cache verbatim — a stale lane would
        // break byte determinism), then truncate to the new geometry. A
        // sealed block keeps its extent, so there is nothing to truncate
        // there and the zeroed lane is what makes the dead row inert.
        if let Some(cache) = self.blocked.get_mut() {
            let (new_n_blocks, n_byte_groups, _) =
                pack::blocked_geometry(self.n_vectors, self.bit_width, dim);
            let block_bytes = n_byte_groups * BLOCK;
            if idx != last {
                pack::move_lane(&mut cache.data, n_byte_groups, last, idx);
            }
            pack::zero_lane(&mut cache.data, n_byte_groups, last);
            if in_open_block {
                cache.data.truncate(new_n_blocks * block_bytes);
                cache.n_blocks = new_n_blocks;
            }
        }

        last
    }

    /// The `(base_slot, live_rows)` of the calibration block holding
    /// live slot `idx`.
    ///
    /// # Panics
    ///
    /// If `idx` is not a live slot — past the end of the storage extent,
    /// or in the dead tail a block-local removal left behind. Both are
    /// contract violations: a slot index is caller-held state, and the
    /// caller was told which slot moved on every removal.
    fn live_block_of(&self, idx: usize) -> (usize, usize) {
        let bs = match self.block_size {
            Some(bs) => bs,
            None => {
                assert!(
                    idx < self.n_vectors,
                    "index {idx} out of bounds (n_vectors = {})",
                    self.n_vectors
                );
                return (0, self.n_vectors);
            }
        };
        assert!(
            idx < self.n_vectors,
            "index {idx} out of bounds (slot capacity = {})",
            self.n_vectors
        );
        let b = idx / bs;
        let (base, live) = match self.sealed.get(b) {
            Some(blk) => (b * bs, blk.len),
            None => (self.open_base(), self.n_vectors - self.open_base()),
        };
        assert!(
            idx < base + live,
            "slot {idx} holds no vector: block {b} has {live} live rows of {bs}"
        );
        (base, live)
    }

    /// How much of what this index allocates is live, searchable
    /// payload — `1.0` for a freshly built index, falling as overhead
    /// and dead weight accumulate.
    ///
    /// The numerator is the code bytes of rows a search can actually
    /// return. The denominator is every byte the index holds for its
    /// rows: those same codes, the codes of rows that are stored but
    /// unreachable, and the per-block calibration.
    ///
    /// Three things pull it down, and it is deliberately one number for
    /// all three, because they trade against each other and a caller
    /// deciding whether to rebuild wants the total rather than three
    /// figures to weigh:
    ///
    /// * **Dead rows.** A block-local [`Self::swap_remove`] leaves the
    ///   tail of a shortened sealed block allocated. Nothing compacts
    ///   across blocks — that would renumber slots — so a workload that
    ///   deletes from old blocks and appends to new ones only ever
    ///   accrues these. This is also the only signal that an *interior*
    ///   block has emptied out entirely, since such a block keeps its
    ///   full extent and no other count changes.
    /// * **Unsearchable rows.** A row stored with a degenerate scale
    ///   (a vector at or below [`MIN_INPUT_NORM`]) has no direction, so
    ///   it scores exactly 0 against every query and is returned only
    ///   after every row that does. It occupies its full code budget
    ///   regardless.
    /// * **Calibration.** Each block carries `2 * dim` floats. Fixed per
    ///   block, so its share grows as blocks empty out.
    ///
    /// `1.0` for an index with no rows: nothing is allocated, so nothing
    /// is wasted. Rebuilding from the source vectors is what restores
    /// it; there is no in-place compaction, by design.
    pub fn health(&self) -> f32 {
        let Some(dim) = self.dim else { return 1.0 };
        if self.n_vectors == 0 {
            return 1.0;
        }
        let bytes_per_row = dim * self.bit_width / 8;
        let searchable = self
            .block_layout()
            .iter()
            .map(|&(base, live, _, _)| {
                self.scales[base..base + live]
                    .iter()
                    .filter(|s| s.is_finite() && **s > 0.0)
                    .count()
            })
            .sum::<usize>();
        let calibration_bytes = self.calibration_pairs() * 2 * dim * 4;
        let allocated = self.n_vectors * bytes_per_row + calibration_bytes;
        (searchable * bytes_per_row) as f32 / allocated as f32
    }

    /// Number of `(shift, scale)` pairs this index holds — one per
    /// block, or one for the whole index when it has no block size.
    /// Zero when nothing is calibrated at all.
    fn calibration_pairs(&self) -> usize {
        if self.tqplus_shift.is_empty() && self.sealed.is_empty() {
            return 0;
        }
        self.sealed.len() + usize::from(!self.tqplus_shift.is_empty())
    }

    /// Number of vectors currently stored.
    ///
    /// Not necessarily one past the largest valid slot index: with
    /// per-block calibration a removal can leave dead rows behind, and
    /// [`Self::slot_capacity`] is what bounds slot numbers then.
    pub fn len(&self) -> usize {
        self.n_vectors - self.dead_slots
    }

    /// Whether the index holds no vectors. Equivalent to `len() == 0`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    use super::{TurboQuantIndex, DEFAULT_BLOCK_SIZE};

    const DIM: usize = 256;

    /// The rotated-batch scratch an add of `n` rows needs.
    ///
    /// `add` splits a batch at the open calibration block's remaining
    /// capacity, so the encode never rotates more than one block at a
    /// time however large the batch is. One block is therefore the hard
    /// ceiling on this buffer — which is why the retention *decision*
    /// is made against the caller's batch rather than the chunk: judged
    /// per chunk, a 100k-row bulk load looks like a steady stream of
    /// block-sized ones and its scratch is kept for good.
    fn want(n: usize) -> usize {
        n.min(DEFAULT_BLOCK_SIZE) * DIM
    }

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
            idx.encode_scratch.capacity() >= want(n),
            "steady same-size adds dropped the warm scratch to {} elements (need {})",
            idx.encode_scratch.capacity(),
            want(n),
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
            idx.encode_scratch.capacity() >= want(last),
            "a growing batch size left only {} scratch elements after a \
             {}-element add, so the next add must grow and be shrunk again",
            idx.encode_scratch.capacity(),
            want(last),
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
            idx.encode_scratch.capacity() >= want(biggest_recent),
            "a jittering batch size left only {} scratch elements, below \
             the {} the last add needed",
            idx.encode_scratch.capacity(),
            want(biggest_recent),
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
            idx.encode_scratch.capacity() >= want(big),
            "a {small}->{big} step left only {} scratch elements, below the \
             {} the larger batch needed",
            idx.encode_scratch.capacity(),
            want(big),
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
        // v2-shape input (empty TQ+) is populated with identity so the
        // committed-calibration check agrees with the stored vectors.
        assert_eq!(idx.tqplus_shift(), &vec![0.0f32; 64][..]);
        assert_eq!(idx.tqplus_scale(), &vec![1.0f32; 64][..]);
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
