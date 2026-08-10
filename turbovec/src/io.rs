//! Read/write TurboVec index files.
//!
//! Every format has two symmetric entry-point pairs: a path-based pair
//! ([`write()`]/[`load`], [`write_id_map`]/[`load_id_map`]) that adds
//! atomic-replace semantics on the write side, and a generic pair
//! ([`write_to`]/[`load_from`], [`write_id_map_to`]/[`load_id_map_from`])
//! over any [`std::io::Write`]/[`std::io::Read`] for callers that hold
//! the payload in memory (e.g. a database page or a network buffer).
//! Both pairs produce and accept exactly the same bytes and apply
//! exactly the same validation.
//!
//! ## Atomic-write protocol
//!
//! Path-based writes go to a sibling temp file named
//! `<dest>.tmp.{pid}.{seq}.{rand}` — pid plus a process-wide counter
//! plus a random component — opened with `O_CREAT|O_EXCL`, so a
//! pre-existing file or planted symlink at the temp name is refused
//! rather than followed, then atomically renamed over the destination.
//! A write killed between temp creation and rename (SIGKILL, power
//! loss) can leave the temp behind; the next save to the same
//! destination sweeps siblings matching this exact pattern whose mtime
//! is over an hour old. Note that if the destination itself is a
//! symlink, the rename replaces the *link* with a regular file (the
//! link's target is left untouched) — standard atomic-replace
//! semantics.
//!
//! Two formats live here:
//! * `.tv` — [`TurboQuantIndex`](crate::TurboQuantIndex) — 4-byte magic
//!   "TVPI" + version + bit_width/dim/n_vectors header + packed codes +
//!   per-vector scales + (v3+) TQ+ per-coord calibration.
//! * `.tvim` — [`IdMapIndex`](crate::IdMapIndex) — 4-byte magic "TVIM"
//!   + version + the same core-index payload + a trailing `slot_to_id`
//!   table of `u64` values.
//!
//! ## Format versioning
//!
//! Both formats are at version 6. The writer emits version 6 only; the
//! loader accepts versions 5 and 6.
//!
//! Version 6 changed the code payload's *layout*, not its content: the
//! file stores the codes in the arch-neutral **sequential blocked**
//! layout (32-vector blocks, one code byte per lane, vectors in order)
//! instead of per-vector bit-plane rows. That layout is exactly what the
//! non-x86 search kernel consumes, and one cheap in-block nibble
//! interleave away from what the x86 kernel consumes — so a load seeds
//! the search cache directly instead of paying the O(n·dim) bit-plane
//! repack on first search. The transformation is invertible and
//! deterministic, so v6 files are byte-identical across platforms and a
//! v5 file (same rotation, same code content) is accepted and converted
//! on load. There is no v5 writer: re-saving a v5 index produces v6.
//!
//! Version 5 replaced the rotation. Versions ≤ 4 encoded their quantized
//! codes through a dense QR-of-a-Gaussian rotation; v5 uses the
//! deterministic block-Hadamard rotation (see [`crate::rotation`]). That
//! changes every encoded byte, so v5 is a **hard format break**: a
//! v4-or-earlier index decoded against the v5 rotation would silently
//! return near-zero recall. The loader therefore refuses any version < 5
//! outright, with an actionable "rebuild the index" error — never a
//! silent mis-decode and never a panic.
//!
//! Because the v5 rotation is deterministic by construction (identical
//! bytes across platforms, CPU architectures, and thread counts), the
//! rotation-drift fingerprint that v4 carried is gone: there is no drift
//! to detect. The v5 core header is exactly v4's minus that fingerprint —
//! `bit_width` (u8) + `dim` (u32) + `n_vectors` (u64) — followed by the
//! packed codes, per-vector scales, and the TQ+ calibration trailer.
//! (`n_vectors` stays a `u64`, so indexes with ≥ 2^32 vectors serialize.)
//!
//! Version 1 `.tv` files had no magic — the file started with a bare
//! bit_width byte (2/3/4). Version 2+ prepends magic + version, which
//! lets us detect either a current file or "looks like a v1 turbovec
//! file" cleanly.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const TV_MAGIC: &[u8; 4] = b"TVPI";
const TV_VERSION: u8 = 6;
const TVIM_MAGIC: &[u8; 4] = b"TVIM";
const TVIM_VERSION: u8 = 6;

/// Recovery hint for any index written before the v5 rotation break
/// (format versions 1 through 4).
const REBUILD_HINT: &str =
    "Rebuild this index from the source vectors using the turbovec build that \
     produced this error. The v5 format replaced the index rotation with a \
     deterministic block-Hadamard transform, which changes every encoded byte; \
     there is no in-place migration.";

/// Durability level for path-based writes.
///
/// * [`Durability::Durable`] (the default everywhere): write to a
///   sibling temp file, `fsync` it (`F_FULLFSYNC` on macOS), then
///   atomically rename over the destination. Survives process crashes
///   AND power loss: the destination always holds a complete old or
///   complete new index, and a completed save is on stable storage.
/// * [`Durability::Fast`]: identical temp-file + atomic-rename protocol
///   but no fsync. Against a process crash the guarantee is the same as
///   `Durable`'s — the destination holds a complete old or complete new
///   index, and the previous file cannot be lost. Against power loss or
///   a kernel panic it holds nothing: the rename may be visible while
///   the new file's contents are not, so a "completed" save can be found
///   truncated or empty afterwards. Choose this only when the index is
///   reproducible or durability is handled elsewhere (e.g. the file is
///   about to be uploaded or the filesystem is transient).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Durability {
    /// Temp file + `fsync` + atomic rename. The default.
    #[default]
    Durable,
    /// Temp file + atomic rename, no `fsync`.
    Fast,
}

/// The code bytes a load produced, tagged with the layout the file
/// stored them in. v6 files carry the arch-neutral sequential blocked
/// layout; v5 files carry per-vector bit-plane rows. The caller
/// ([`crate::TurboQuantIndex::load`]) picks the cheap path for each.
#[derive(Debug, Clone, PartialEq)]
pub enum CodePayload {
    /// Per-vector bit-plane rows (v5 files).
    Packed(Vec<u8>),
    /// Sequential blocked layout (v6 files) — includes zero padding for
    /// the final partial block — plus the embedded Lloyd-Max codebook
    /// (`n_levels - 1` boundaries, `n_levels` centroids; all-zero for an
    /// empty index, where the loader ignores it). Embedding the codebook
    /// spares every load a ~60 ms Lloyd-Max solve and pins search to the
    /// writer's codebook rather than a recomputed one.
    BlockedSeq {
        /// The blocked code bytes as stored in the file.
        codes: Vec<u8>,
        /// The codebook's `n_levels - 1` decision boundaries.
        boundaries: Vec<f32>,
        /// The codebook's `n_levels` reconstruction centroids.
        centroids: Vec<f32>,
    },
    /// Codes already in the *native* kernel layout for this platform —
    /// produced by the fast path loader, whose extraction pass fuses the
    /// platform transform into the copy. Which layout is native depends on
    /// the host: vector-major where a dot-product kernel will read it, the
    /// perm0 nibble interleave on classic x86, and byte-identical to
    /// `BlockedSeq` elsewhere (the stored layout is native there).
    BlockedNative {
        /// The code bytes already in this platform's kernel layout.
        codes: Vec<u8>,
        /// The codebook's `n_levels - 1` decision boundaries.
        boundaries: Vec<f32>,
        /// The codebook's `n_levels` reconstruction centroids.
        centroids: Vec<f32>,
    },
}

/// The native-layout bytes this host's fast-path loader produces for the
/// given stored sequential-blocked codes — the same transform selection the
/// loader itself uses, exposed so tests can state loader expectations
/// without mirroring the per-host layout math and feature detection.
#[doc(hidden)]
pub fn native_layout_of_stored(bit_width: usize, dim: usize, seq: &[u8]) -> Vec<u8> {
    let mut out = seq.to_vec();
    crate::pack::apply_native_transform(&mut out, bit_width, dim / (8 / bit_width));
    out
}

/// Core payload — what a fully-deserialized index needs.
type CoreLoad = (usize, usize, usize, CodePayload, Vec<f32>, Vec<f32>, Vec<f32>);

/// `.tv` write — positional index.
///
/// The write is atomic with respect to the destination: the payload goes
/// to a sibling temp file which is fsynced and then renamed over `path`,
/// so a failed or interrupted write leaves any previous file at `path`
/// intact. The rename is the commit point, and `Err` means the save did
/// not commit — a parent-directory fsync that fails after the rename
/// leaves the new file in place, so it warns on stderr and still returns
/// `Ok` rather than claiming the previous file survived (#365).
///
/// # Panics
///
/// All six slice arguments carry a length invariant and `bit_width` a
/// range bound, and a violation
/// aborts rather than joining the `io::Result` — the `Result` reports
/// what happened to the sink, while these describe a caller-assembled
/// shape the writer cannot negotiate:
///
/// - `tqplus_shift.len() != tqplus_scale.len()`, or the pair is
///   non-empty and its length is not `dim`. Empty means identity
///   calibration; any other length has no valid trailer encoding.
/// - `codebook_boundaries.len() != (1 << bit_width) - 1` or
///   `codebook_centroids.len() != 1 << bit_width`.
/// - `bit_width >= 64`, for which `1 << bit_width` has no `usize` value
///   to compare those lengths against. Widths under that bound but
///   outside the format's 2..=4 range are *not* rejected here: they
///   still write, and the readers still refuse the header (#411).
/// - `scales.len() != n_vectors`.
/// - `codes_blocked_seq.len()` is not the blocked-layout size the header
///   implies — `n_vectors` rounded up to whole 32-vector blocks, times
///   `dim / (8 / bit_width)` bytes per vector. This is the length the
///   loader sizes the codes section to; any other length shifts every
///   later section, and the outcome is a file that may load clean and
///   silently mis-score rather than one that is rejected (#407). Widths
///   outside 2..=4 skip this bullet only — see #411.
///
/// These are the same conditions [`crate::TurboQuantIndex::from_parts`]
/// rejects, so both entry points to the format agree on what a valid
/// index is. Every check runs before anything is written, so a violation
/// cannot truncate or replace an existing index. The simplest way to
/// satisfy them is to take the two buffers from `codes_blocked_seq()` /
/// `scales()` on a real index, or to use
/// [`crate::TurboQuantIndex::write`].
#[allow(clippy::too_many_arguments)]
pub fn write(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_seq: &[u8],
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    // Validate before any file is created so a violation cannot destroy
    // a previous good index at `path`. (`write_to` re-asserts —
    // harmlessly — for its direct callers.)
    write_with_durability(
        path, bit_width, dim, n_vectors, codes_blocked_seq,
        codebook_boundaries, codebook_centroids, scales,
        tqplus_shift, tqplus_scale, Durability::Durable,
    )
}

/// [`write()`] with an explicit [`Durability`] level.
///
/// # Panics
///
/// All six slice arguments carry a length invariant and `bit_width` a
/// range bound, and a violation
/// aborts rather than joining the `io::Result` — the `Result` reports
/// what happened to the sink, while these describe a caller-assembled
/// shape the writer cannot negotiate:
///
/// - `tqplus_shift.len() != tqplus_scale.len()`, or the pair is
///   non-empty and its length is not `dim`. Empty means identity
///   calibration; any other length has no valid trailer encoding.
/// - `codebook_boundaries.len() != (1 << bit_width) - 1` or
///   `codebook_centroids.len() != 1 << bit_width`.
/// - `bit_width >= 64`, for which `1 << bit_width` has no `usize` value
///   to compare those lengths against. Widths under that bound but
///   outside the format's 2..=4 range are *not* rejected here: they
///   still write, and the readers still refuse the header (#411).
/// - `scales.len() != n_vectors`.
/// - `codes_blocked_seq.len()` is not the blocked-layout size the header
///   implies — `n_vectors` rounded up to whole 32-vector blocks, times
///   `dim / (8 / bit_width)` bytes per vector. This is the length the
///   loader sizes the codes section to; any other length shifts every
///   later section, and the outcome is a file that may load clean and
///   silently mis-score rather than one that is rejected (#407). Widths
///   outside 2..=4 skip this bullet only — see #411.
///
/// These are the same conditions [`crate::TurboQuantIndex::from_parts`]
/// rejects, so both entry points to the format agree on what a valid
/// index is. Every check runs before anything is written, so a violation
/// cannot truncate or replace an existing index. The simplest way to
/// satisfy them is to take the two buffers from `codes_blocked_seq()` /
/// `scales()` on a real index, or to use
/// [`crate::TurboQuantIndex::write`].
#[allow(clippy::too_many_arguments)]
pub fn write_with_durability(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_seq: &[u8],
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    durability: Durability,
) -> io::Result<()> {
    assert_tqplus_calibration(dim, tqplus_shift, tqplus_scale);
    assert_codes_and_scales_lengths(bit_width, dim, n_vectors, codes_blocked_seq, scales);
    write_atomic_parallel(path.as_ref(), durability, TV_MAGIC, TV_VERSION, |head| {
        head_core(head, bit_width, dim, n_vectors, codebook_boundaries, codebook_centroids)
    }, codes_blocked_seq, None, |tail| {
        tail_core(tail, scales, tqplus_shift, tqplus_scale);
        Ok(())
    })
}

/// x86 fused-write variant of [`write_with_durability`]: takes the codes
/// in the *native* (perm0-interleaved) layout and deinterleaves each
/// chunk inside the writer threads — no whole-payload sequential
/// intermediate, and the transform overlaps device writes. Emits bytes
/// identical to the seq-taking form (the transform is block-local).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_native_with_durability(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_native: &[u8],
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    durability: Durability,
) -> io::Result<()> {
    assert_tqplus_calibration(dim, tqplus_shift, tqplus_scale);
    assert_codes_and_scales_lengths(bit_width, dim, n_vectors, codes_blocked_native, scales);
    write_atomic_parallel(path.as_ref(), durability, TV_MAGIC, TV_VERSION, |head| {
        head_core(head, bit_width, dim, n_vectors, codebook_boundaries, codebook_centroids)
    }, codes_blocked_native, Some(crate::pack::deinterleave_chunk_into), |tail| {
        tail_core(tail, scales, tqplus_shift, tqplus_scale);
        Ok(())
    })
}

/// `.tv` write to any [`Write`] sink — the in-memory counterpart of
/// [`write()`]. Emits exactly the bytes [`write()`] would put in the file
/// (magic + version + v5 core payload), so a `Vec<u8>` filled by this
/// function is byte-identical to the corresponding `.tv` file.
///
/// Unlike [`write()`] there is no atomicity story: the caller owns the
/// sink.
///
/// # Panics
///
/// All six slice arguments carry a length invariant and `bit_width` a
/// range bound, and a violation
/// aborts rather than joining the `io::Result` — the `Result` reports
/// what happened to the sink, while these describe a caller-assembled
/// shape the writer cannot negotiate:
///
/// - `tqplus_shift.len() != tqplus_scale.len()`, or the pair is
///   non-empty and its length is not `dim`. Empty means identity
///   calibration; any other length has no valid trailer encoding.
/// - `codebook_boundaries.len() != (1 << bit_width) - 1` or
///   `codebook_centroids.len() != 1 << bit_width`.
/// - `bit_width >= 64`, for which `1 << bit_width` has no `usize` value
///   to compare those lengths against. Widths under that bound but
///   outside the format's 2..=4 range are *not* rejected here: they
///   still write, and the readers still refuse the header (#411).
/// - `scales.len() != n_vectors`.
/// - `codes_blocked_seq.len()` is not the blocked-layout size the header
///   implies — `n_vectors` rounded up to whole 32-vector blocks, times
///   `dim / (8 / bit_width)` bytes per vector. This is the length the
///   loader sizes the codes section to; any other length shifts every
///   later section, and the outcome is a file that may load clean and
///   silently mis-score rather than one that is rejected (#407). Widths
///   outside 2..=4 skip this bullet only — see #411.
///
/// These are the same conditions [`crate::TurboQuantIndex::from_parts`]
/// rejects, so both entry points to the format agree on what a valid
/// index is. Every check runs before anything is written, so a violation
/// cannot truncate or replace an existing index. The simplest way to
/// satisfy them is to take the two buffers from `codes_blocked_seq()` /
/// `scales()` on a real index, or to use
/// [`crate::TurboQuantIndex::write`].
#[allow(clippy::too_many_arguments)]
pub fn write_to<W: Write>(
    w: &mut W,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_seq: &[u8],
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    assert_tqplus_calibration(dim, tqplus_shift, tqplus_scale);
    assert_codes_and_scales_lengths(bit_width, dim, n_vectors, codes_blocked_seq, scales);
    w.write_all(TV_MAGIC)?;
    w.write_all(&[TV_VERSION])?;
    write_core(
        w, bit_width, dim, n_vectors, codes_blocked_seq,
        codebook_boundaries, codebook_centroids, scales,
        tqplus_shift, tqplus_scale,
    )
}

/// The exact byte count [`write_to`] emits, from section lengths alone —
/// the sizing hint an in-memory serializer needs to allocate its buffer
/// once instead of growing it (#409).
///
/// Kept immediately next to [`write_to`] because it is a second statement
/// of the same layout: magic, version byte, `bit_width` byte, `dim` as
/// `u32`, `n_vectors` as `u64`, the two codebook arrays, the codes, the
/// scales, then the TQ+ trailer (a `u32` length prefix followed by the
/// shift and scale arrays, which [`assert_tqplus_calibration`] has already
/// forced to equal lengths). `turbovec/tests/bytes_io.rs` checks it
/// against what the writer actually produced, so a format change that
/// misses this function fails there rather than silently degrading the
/// hint.
pub(crate) fn serialized_len(
    bit_width: usize,
    codes_len: usize,
    n_scales: usize,
    n_tqplus: usize,
) -> usize {
    let n_levels = 1usize << bit_width;
    TV_MAGIC.len()
        + 1 // version byte
        + 1 // bit_width byte
        + 4 // dim: u32
        + 8 // n_vectors: u64
        + (n_levels - 1) * 4 // codebook boundaries
        + n_levels * 4 // codebook centroids
        + codes_len
        + n_scales * 4
        + 4 // TQ+ length prefix
        + n_tqplus * 4 // TQ+ shift
        + n_tqplus * 4 // TQ+ scale
}

/// `.tv` load — positional index. Accepts versions 5 and 6; any earlier
/// version (1 through 4) is rejected with an actionable rebuild error,
/// because the v5 rotation break changed every encoded byte. Files
/// with empty TQ+ are treated as identity calibration.
pub fn load(path: impl AsRef<Path>) -> io::Result<CoreLoad> {
    let f = File::open(path)?;
    // The file's real length caps section preallocation: a section can
    // pre-reserve its declared size only when the bytes provably exist,
    // so a tiny file declaring a huge payload still cannot drive a large
    // allocation (same posture as the capped incremental read).
    let cap = f.metadata()?.len();
    // Fast v6 path: sections read straight from the file — the fixed
    // header offset lets the codes section parallel-pread directly into
    // its final buffer (no intermediate copy), then transform in place
    // on warm pages. v5/malformed files fall back to the generic
    // streamed reader for canonical errors.
    if let Some((core, _tail, _rest_off, _ids, _sorted)) = try_load_v6_fast(&f, cap, TV_MAGIC)? {
        return Ok(core);
    }
    let buf = read_file_parallel(&f, cap)?;
    load_from_capped(&mut &buf[..], cap)
}

/// `.tv` load from any [`Read`] source — the in-memory counterpart of
/// [`load`]. Applies exactly the same version handling and validation
/// (structural checks, value-level float validation), so a byte slice
/// and the file it came from load — or fail — identically.
pub fn load_from<R: Read>(f: &mut R) -> io::Result<CoreLoad> {
    load_from_capped(f, 0)
}

/// [`load_from`] with a preallocation cap from a trusted source (the
/// real file length for path loads; `0` = never preallocate).
fn load_from_capped<R: Read>(f: &mut R, alloc_cap: u64) -> io::Result<CoreLoad> {
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TV_MAGIC {
        // Version 1 .tv files had no magic — first byte was the bit_width
        // (always 2, 3, or 4). If we see one of those as the first byte,
        // emit a targeted error rather than the generic "wrong magic"
        // message; otherwise treat it as a non-turbovec file.
        // A v7 sync container reaching a byte/reader entry point is a
        // specific, likely mistake — `load(path)` opens these fine — so
        // say that rather than "wrong magic". v7 is deliberately not
        // supported here: its two-slot header, block units and redo ops
        // need random access, which a `Read` stream cannot serve, and
        // `to_bytes` only ever emits v6 (#486).
        if &magic == crate::io_v7::V7_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "this is a v7 sync container (written by sync()); open it with \
                 load(<path>) — from_bytes / load_from_reader read the v6 \
                 format that write() and to_bytes() produce",
            ));
        }
        if (2..=4).contains(&magic[0]) {
            return Err(incompatible_version_error(1, ".tv"));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a turbovec .tv file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    read_core_versioned(f, version[0], TV_VERSION, ".tv", alloc_cap)
}

/// Error for an index written in a pre-v5 format (versions 1 through 4).
/// The v5 rotation break makes those bytes undecodable, so the loader
/// refuses them loudly and points at the only recovery path — a rebuild.
fn incompatible_version_error(version: u8, label: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "this {label} index is format version {version}, which is \
             incompatible with the v5 rotation. Loading it against the v5 \
             rotation would silently return near-zero recall, so it is \
             refused. {REBUILD_HINT}"
        ),
    )
}


/// `.tvim` write — positional index plus the id-map side-tables.
///
/// Atomic with respect to the destination, like [`write()`].
///
/// # Panics
///
/// All seven slice arguments carry a length invariant and `bit_width` a
/// range bound, and a violation
/// aborts rather than joining the `io::Result` — the `Result` reports
/// what happened to the sink, while these describe a caller-assembled
/// shape the writer cannot negotiate:
///
/// - `tqplus_shift.len() != tqplus_scale.len()`, or the pair is
///   non-empty and its length is not `dim`. Empty means identity
///   calibration; any other length has no valid trailer encoding.
/// - `codebook_boundaries.len() != (1 << bit_width) - 1` or
///   `codebook_centroids.len() != 1 << bit_width`.
/// - `bit_width >= 64`, for which `1 << bit_width` has no `usize` value
///   to compare those lengths against. Widths under that bound but
///   outside the format's 2..=4 range are *not* rejected here: they
///   still write, and the readers still refuse the header (#411).
/// - `slot_to_id.len() != n_vectors`.
/// - `scales.len() != n_vectors`.
/// - `codes_blocked_seq.len()` is not the blocked-layout size the header
///   implies — `n_vectors` rounded up to whole 32-vector blocks, times
///   `dim / (8 / bit_width)` bytes per vector. This is the length the
///   loader sizes the codes section to; any other length shifts every
///   later section, and the outcome is a file that may load clean and
///   silently mis-score rather than one that is rejected (#407). Widths
///   outside 2..=4 skip this bullet only — see #411.
///
/// These are the same conditions [`crate::TurboQuantIndex::from_parts`]
/// rejects, so both entry points to the format agree on what a valid
/// index is. Every check runs before anything is written, so a violation
/// cannot truncate or replace an existing index. The simplest way to
/// satisfy them is to take the two buffers from `codes_blocked_seq()` /
/// `scales()` on a real index, or to use
/// [`crate::TurboQuantIndex::write`].
#[allow(clippy::too_many_arguments)]
pub fn write_id_map(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_seq: &[u8],
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    slot_to_id: &[u64],
) -> io::Result<()> {
    // Validate before any file is created so a violation cannot destroy
    // a previous good index at `path`. (`write_id_map_to` re-asserts —
    // harmlessly — for its direct callers.)
    assert_eq!(
        slot_to_id.len(),
        n_vectors,
        "slot_to_id length {} does not match n_vectors {}",
        slot_to_id.len(),
        n_vectors,
    );
    assert_tqplus_calibration(dim, tqplus_shift, tqplus_scale);
    assert_codes_and_scales_lengths(bit_width, dim, n_vectors, codes_blocked_seq, scales);

    write_id_map_with_durability(
        path, bit_width, dim, n_vectors, codes_blocked_seq,
        codebook_boundaries, codebook_centroids, scales,
        tqplus_shift, tqplus_scale, slot_to_id, Durability::Durable,
    )
}

/// [`write_id_map`] with an explicit [`Durability`] level.
///
/// # Panics
///
/// All seven slice arguments carry a length invariant and `bit_width` a
/// range bound, and a violation
/// aborts rather than joining the `io::Result` — the `Result` reports
/// what happened to the sink, while these describe a caller-assembled
/// shape the writer cannot negotiate:
///
/// - `tqplus_shift.len() != tqplus_scale.len()`, or the pair is
///   non-empty and its length is not `dim`. Empty means identity
///   calibration; any other length has no valid trailer encoding.
/// - `codebook_boundaries.len() != (1 << bit_width) - 1` or
///   `codebook_centroids.len() != 1 << bit_width`.
/// - `bit_width >= 64`, for which `1 << bit_width` has no `usize` value
///   to compare those lengths against. Widths under that bound but
///   outside the format's 2..=4 range are *not* rejected here: they
///   still write, and the readers still refuse the header (#411).
/// - `slot_to_id.len() != n_vectors`.
/// - `scales.len() != n_vectors`.
/// - `codes_blocked_seq.len()` is not the blocked-layout size the header
///   implies — `n_vectors` rounded up to whole 32-vector blocks, times
///   `dim / (8 / bit_width)` bytes per vector. This is the length the
///   loader sizes the codes section to; any other length shifts every
///   later section, and the outcome is a file that may load clean and
///   silently mis-score rather than one that is rejected (#407). Widths
///   outside 2..=4 skip this bullet only — see #411.
///
/// These are the same conditions [`crate::TurboQuantIndex::from_parts`]
/// rejects, so both entry points to the format agree on what a valid
/// index is. Every check runs before anything is written, so a violation
/// cannot truncate or replace an existing index. The simplest way to
/// satisfy them is to take the two buffers from `codes_blocked_seq()` /
/// `scales()` on a real index, or to use
/// [`crate::TurboQuantIndex::write`].
#[allow(clippy::too_many_arguments)]
pub fn write_id_map_with_durability(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_seq: &[u8],
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    slot_to_id: &[u64],
    durability: Durability,
) -> io::Result<()> {
    assert_eq!(
        slot_to_id.len(),
        n_vectors,
        "slot_to_id length {} does not match n_vectors {}",
        slot_to_id.len(),
        n_vectors,
    );
    assert_tqplus_calibration(dim, tqplus_shift, tqplus_scale);
    assert_codes_and_scales_lengths(bit_width, dim, n_vectors, codes_blocked_seq, scales);
    write_atomic_parallel(path.as_ref(), durability, TVIM_MAGIC, TVIM_VERSION, |head| {
        head_core(head, bit_width, dim, n_vectors, codebook_boundaries, codebook_centroids)
    }, codes_blocked_seq, None, |tail| {
        tail_core(tail, scales, tqplus_shift, tqplus_scale);
        for &id in slot_to_id {
            tail.extend_from_slice(&id.to_le_bytes());
        }
        Ok(())
    })
}

/// x86 fused-write variant of [`write_id_map_with_durability`]; see
/// [`write_native_with_durability`].
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_id_map_native_with_durability(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_native: &[u8],
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    slot_to_id: &[u64],
    durability: Durability,
) -> io::Result<()> {
    assert_eq!(
        slot_to_id.len(),
        n_vectors,
        "slot_to_id length {} does not match n_vectors {}",
        slot_to_id.len(),
        n_vectors,
    );
    assert_tqplus_calibration(dim, tqplus_shift, tqplus_scale);
    assert_codes_and_scales_lengths(bit_width, dim, n_vectors, codes_blocked_native, scales);
    write_atomic_parallel(path.as_ref(), durability, TVIM_MAGIC, TVIM_VERSION, |head| {
        head_core(head, bit_width, dim, n_vectors, codebook_boundaries, codebook_centroids)
    }, codes_blocked_native, Some(crate::pack::deinterleave_chunk_into), |tail| {
        tail_core(tail, scales, tqplus_shift, tqplus_scale);
        for &id in slot_to_id {
            tail.extend_from_slice(&id.to_le_bytes());
        }
        Ok(())
    })
}

/// `.tvim` write to any [`Write`] sink — the in-memory counterpart of
/// [`write_id_map`]. Emits exactly the bytes [`write_id_map`] would put
/// in the file.
///
/// # Panics
///
/// All seven slice arguments carry a length invariant and `bit_width` a
/// range bound, and a violation
/// aborts rather than joining the `io::Result` — the `Result` reports
/// what happened to the sink, while these describe a caller-assembled
/// shape the writer cannot negotiate:
///
/// - `tqplus_shift.len() != tqplus_scale.len()`, or the pair is
///   non-empty and its length is not `dim`. Empty means identity
///   calibration; any other length has no valid trailer encoding.
/// - `codebook_boundaries.len() != (1 << bit_width) - 1` or
///   `codebook_centroids.len() != 1 << bit_width`.
/// - `bit_width >= 64`, for which `1 << bit_width` has no `usize` value
///   to compare those lengths against. Widths under that bound but
///   outside the format's 2..=4 range are *not* rejected here: they
///   still write, and the readers still refuse the header (#411).
/// - `slot_to_id.len() != n_vectors`.
/// - `scales.len() != n_vectors`.
/// - `codes_blocked_seq.len()` is not the blocked-layout size the header
///   implies — `n_vectors` rounded up to whole 32-vector blocks, times
///   `dim / (8 / bit_width)` bytes per vector. This is the length the
///   loader sizes the codes section to; any other length shifts every
///   later section, and the outcome is a file that may load clean and
///   silently mis-score rather than one that is rejected (#407). Widths
///   outside 2..=4 skip this bullet only — see #411.
///
/// These are the same conditions [`crate::TurboQuantIndex::from_parts`]
/// rejects, so both entry points to the format agree on what a valid
/// index is. Every check runs before anything is written, so a violation
/// cannot truncate or replace an existing index. The simplest way to
/// satisfy them is to take the two buffers from `codes_blocked_seq()` /
/// `scales()` on a real index, or to use
/// [`crate::TurboQuantIndex::write`].
#[allow(clippy::too_many_arguments)]
pub fn write_id_map_to<W: Write>(
    w: &mut W,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_seq: &[u8],
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    slot_to_id: &[u64],
) -> io::Result<()> {
    assert_eq!(
        slot_to_id.len(),
        n_vectors,
        "slot_to_id length {} does not match n_vectors {}",
        slot_to_id.len(),
        n_vectors,
    );
    assert_tqplus_calibration(dim, tqplus_shift, tqplus_scale);
    assert_codes_and_scales_lengths(bit_width, dim, n_vectors, codes_blocked_seq, scales);

    w.write_all(TVIM_MAGIC)?;
    w.write_all(&[TVIM_VERSION])?;
    write_core(
        w, bit_width, dim, n_vectors, codes_blocked_seq,
        codebook_boundaries, codebook_centroids, scales,
        tqplus_shift, tqplus_scale,
    )?;
    for &id in slot_to_id {
        w.write_all(&id.to_le_bytes())?;
    }
    Ok(())
}

/// `.tvim` load — positional index plus the id-map side-tables. Accepts
/// versions 5 and 6, with the same loud pre-v5 rejection as [`load`].
#[allow(clippy::type_complexity)]
pub fn load_id_map(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, CodePayload, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u64>)> {
    let (core, ids, _) = load_id_map_prepared(path)?;
    let (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale) = core;
    Ok((bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale, ids))
}

/// [`load_id_map`] that also returns the sorted duplicate-check table
/// when the fast loader built it on its tail thread, so `IdMapIndex`
/// can adopt it instead of sorting a second copy on the critical path.
#[allow(clippy::type_complexity)]
pub(crate) fn load_id_map_prepared(
    path: impl AsRef<Path>,
) -> io::Result<(CoreLoad, Vec<u64>, Option<Vec<u64>>)> {
    let f = File::open(path)?;
    // See `load` for the allocation-cap rationale.
    let cap = f.metadata()?.len();
    // See `load` — direct-to-destination fast v6 path.
    if let Some((core, tail, rest_off, pre_ids, pre_sorted)) = try_load_v6_fast(&f, cap, TVIM_MAGIC)? {
        let (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale) = core;
        let id_bytes = n_vectors
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "id table size overflows usize"))?;
        // Decode straight out of the tail buffer. The bytes cannot be
        // reinterpreted in place (no alignment guarantee, and the file is
        // little-endian by definition), so one pass is irreducible — but
        // it is now the only one. `read_exact_vec_capped` reports a short
        // table the same way, and this reproduces its error verbatim so
        // truncated files keep failing with the message the tests pin.
        let raw = &tail[rest_off.min(tail.len())..];
        if raw.len() < id_bytes {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("truncated file: expected {id_bytes} bytes, got {}", raw.len()),
            ));
        }
        // Already decoded on the tail thread, inside the codes read's
        // overlap window, whenever the table was intact there.
        let slot_to_id: Vec<u64> = if pre_ids.len() == n_vectors {
            pre_ids
        } else {
            raw[..id_bytes]
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .collect()
        };
        let sorted = (pre_sorted.len() == n_vectors).then_some(pre_sorted);
        return Ok((
            (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale),
            slot_to_id,
            sorted,
        ));
    }
    let buf = read_file_parallel(&f, cap)?;
    let (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale, ids) =
        load_id_map_from_capped(&mut &buf[..], cap)?;
    Ok((
        (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale),
        ids,
        None,
    ))
}

/// `.tvim` load from any [`Read`] source — the in-memory counterpart of
/// [`load_id_map`], with exactly the same version handling and
/// validation, so a byte slice and the file it came from load — or
/// fail — identically.
#[allow(clippy::type_complexity)]
pub fn load_id_map_from<R: Read>(
    f: &mut R,
) -> io::Result<(usize, usize, usize, CodePayload, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u64>)> {
    load_id_map_from_capped(f, 0)
}

/// [`load_id_map_from`] with a trusted preallocation cap (see
/// [`load_from_capped`]).
#[allow(clippy::type_complexity)]
fn load_id_map_from_capped<R: Read>(
    f: &mut R,
    alloc_cap: u64,
) -> io::Result<(usize, usize, usize, CodePayload, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u64>)> {
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVIM_MAGIC {
        // See `load_from_capped`: a v7 sync container here is a specific
        // mistake worth naming (#486).
        if &magic == crate::io_v7::V7_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "this is a v7 sync container (written by sync()); open it with \
                 load(<path>) — from_bytes / load_from_reader read the v6 \
                 format that write() and to_bytes() produce",
            ));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a turbovec .tvim file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    let (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale) =
        read_core_versioned(f, version[0], TVIM_VERSION, ".tvim", alloc_cap)?;

    // Read the slot_to_id table via the capped reader rather than
    // `Vec::with_capacity(n_vectors)` — `n_vectors` is attacker-controlled and
    // pre-reserving it allows a tiny file to drive a huge allocation.
    let id_bytes = n_vectors
        .checked_mul(8)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "id table size overflows usize"))?;
    let raw = read_exact_vec_capped(f, id_bytes, alloc_cap)?;
    let slot_to_id: Vec<u64> = raw
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();

    Ok((
        bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale,
        slot_to_id,
    ))
}

/// v5 core header: bit_width u8 + dim u32 + n_vectors u64.
const V5_HEADER_SIZE: usize = 13;

/// TQ+ calibration length invariant shared by [`write()`] and
/// [`write_id_map`]. Must run before any file is created — see the
/// callers.
fn assert_tqplus_calibration(dim: usize, tqplus_shift: &[f32], tqplus_scale: &[f32]) {
    // n_calib == 0 means identity calibration (lazy index with no add
    // yet, or a loaded pre-TQ+ index that's been resaved); otherwise
    // must equal dim.
    assert!(
        tqplus_shift.len() == tqplus_scale.len()
            && (tqplus_shift.is_empty() || tqplus_shift.len() == dim),
        "TQ+ shift/scale must have equal length and either be empty or equal dim"
    );
}

/// Codebook length invariant shared by [`write()`] and [`write_id_map`].
/// Like [`assert_tqplus_calibration`], must run before any file is
/// created — a panic after temp creation would leak the temp (#313).
fn assert_codebook_lengths(bit_width: usize, boundaries: &[f32], centroids: &[f32]) {
    // `1usize << bit_width` is out of range for `bit_width >= 64`, and
    // the two build profiles disagree about what that means: debug
    // panics `attempt to shift left with overflow` (naming neither the
    // argument nor this function), release masks the shift to `<< 0` and
    // proceeds with `n_levels == 1` — which is satisfiable, so a caller
    // supplying one centroid and no boundaries writes a header no reader
    // will accept. Reject the shift itself so one actionable message
    // covers every profile (#411).
    //
    // The bound is the shift, not the format's 2..=4. Widths below 64
    // but outside that range keep writing a file the load-side header
    // check refuses, which is the behaviour
    // `assert_codes_and_scales_lengths` also preserves and
    // `out_of_range_bit_width_is_left_to_the_load_side_header_check`
    // pins; widening this assert to `2..=4` would contradict both.
    assert!(
        bit_width < usize::BITS as usize,
        "bit_width {bit_width} is out of range: 2^{bit_width} codebook \
         levels do not fit in a usize (the format encodes 2, 3, or 4)",
    );
    let n_levels = 1usize << bit_width;
    assert_eq!(boundaries.len(), n_levels - 1, "codebook boundaries length");
    assert_eq!(centroids.len(), n_levels, "codebook centroids length");
}

/// Codes/scales length invariant shared by every `write*` entry point:
/// the two payload buffers must describe exactly the `n_vectors` rows
/// the header declares. Like [`assert_tqplus_calibration`], must run
/// before any file is created — a panic after temp creation would leak
/// the temp (#313).
///
/// Both sections are sized from the header on load, never from a length
/// prefix, so a buffer of any other length shifts every later section.
/// The result is not a rejected file but an undefined one: sometimes a
/// load error, sometimes a clean load that silently mis-scores — a
/// short codes buffer has produced a top score above the cosine ceiling
/// (#407), and a compensating pair that preserves the total byte count
/// shifts nothing downstream, so no header-derived check can fire at
/// all. The loader cannot separate "short section" from "the header
/// lied", so the writer is the last point at which the caller can still
/// be told. These are the same two conditions
/// [`crate::TurboQuantIndex::from_parts`] rejects
/// ([`PackedCodesLengthMismatch`](crate::FromPartsError::PackedCodesLengthMismatch)
/// and [`ScalesLengthMismatch`](crate::FromPartsError::ScalesLengthMismatch)),
/// so the two entry points to the same on-disk format now agree.
fn assert_codes_and_scales_lengths(
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_seq: &[u8],
    scales: &[f32],
) {
    // One scale per vector. Independent of bit_width, so it is checked
    // unconditionally and first.
    assert_eq!(
        scales.len(),
        n_vectors,
        "scales length {} does not match n_vectors {}",
        scales.len(),
        n_vectors,
    );
    // Compare the codes buffer against `v6_blocked_len` — the loader's
    // own section-sizing function, equal to
    // `pack::blocked_geometry(n_vectors, bit_width, dim).2` — so the
    // writer rejects exactly the lengths the loader would mis-size.
    //
    // `v6_blocked_len` divides by `8 / bit_width`, presuming the 2..=4
    // range `validate_header_fields` enforces on load. A width outside
    // that range is left to the load-side header check on purpose, and
    // permanently: it still writes a file both readers reject, and this
    // check steps aside rather than absorbing it.
    // `assert_codebook_lengths` holds the same line — it bounds the
    // shift (#411), not the format's 2..=4, precisely so the two agree.
    // Widening either to 2..=4 would break
    // `out_of_range_bit_width_is_left_to_the_load_side_header_check`.
    if !(2..=4).contains(&bit_width) {
        return;
    }
    let expected = v6_blocked_len(bit_width, dim, n_vectors)
        .expect("blocked code size overflows usize");
    assert_eq!(
        codes_blocked_seq.len(),
        expected,
        "codes_blocked_seq length {} does not match the blocked layout for \
         n_vectors {}, bit_width {}, dim {} ({expected} bytes)",
        codes_blocked_seq.len(),
        n_vectors,
        bit_width,
        dim,
    );
}

/// Process-wide counter distinguishing concurrent saves to the same
/// path from one process: `.tmp.{pid}` alone would interleave two
/// threads' writes into one temp file and rename the corruption into
/// place, defeating the torn-index guarantee.
static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Filename byte budget for the temp sibling: NAME_MAX is 255 bytes on
/// every filesystem we target (ext4, APFS, NTFS component limit).
const TMP_NAME_MAX: usize = 255;

/// A fresh unpredictable value for the temp-name suffix. `RandomState`
/// is seeded from OS entropy; folding in the current time varies the
/// value per call. (Unpredictability is defense in depth — `O_EXCL` in
/// [`create_tmp`] is what actually defeats a planted symlink; the
/// randomness keeps collisions with a crash-leaked temp from a reused
/// pid from turning into save failures.)
pub(crate) fn file_nonce() -> u64 {
    ((tmp_rand() as u64) << 32) | tmp_rand() as u64
}

fn tmp_rand() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    h.write_u64(now);
    h.finish() as u32
}

/// Sibling temp name `<dest>.tmp.{pid}.{seq}.{rand:08x}` in the same
/// directory as `path`. If the destination's own filename would push
/// the sibling past NAME_MAX, the base portion is truncated to fit —
/// the temp name only has to be unique and recognizable, not complete
/// (#299).
fn tmp_sibling(path: &Path, rand: u32) -> PathBuf {
    let suffix = format!(
        ".tmp.{}.{}.{:08x}",
        std::process::id(),
        TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        rand,
    );
    let base = path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    if base.len() + suffix.len() <= TMP_NAME_MAX {
        let mut name = base;
        name.push(suffix);
        return path.with_file_name(name);
    }
    // Truncation goes through a lossy UTF-8 view so the cut lands on a
    // char boundary; a mangled non-UTF-8 byte in a *temp* name is
    // harmless (the destination name is untouched).
    let s = base.to_string_lossy();
    let mut cut = (TMP_NAME_MAX - suffix.len()).min(s.len());
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    path.with_file_name(format!("{}{}", &s[..cut], suffix))
}

/// Open a fresh sibling temp file with `create_new` (`O_CREAT|O_EXCL`):
/// refuses any existing file *and never follows a symlink*, so a
/// pre-planted `<dest>.tmp.*` symlink cannot redirect the write outside
/// the destination directory (#293). A collision — possible only via a
/// crash-leaked temp from a reused pid, since the name embeds pid, a
/// process-wide counter, and a random component — retries with a fresh
/// name a few times rather than failing the save.
pub(crate) fn create_tmp(path: &Path) -> io::Result<(File, PathBuf)> {
    let mut last_err = None;
    for _ in 0..8 {
        let tmp = tmp_sibling(path, tmp_rand());
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(f) => return Ok((f, tmp)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last_err = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last_err.expect("retry loop ran"))
}

/// True for the Win32 status codes a rename can return *transiently*
/// while another party still holds the destination.
///
/// - 32 ERROR_SHARING_VIOLATION: another handle lacks FILE_SHARE_DELETE
///   — CPython's `open()` and antivirus/indexer scans both qualify
///   (#313).
/// - 5 ERROR_ACCESS_DENIED: the destination is *delete-pending*. Windows
///   leaves a file in that state from the moment its last handle is
///   marked delete-on-close until that handle actually closes, and every
///   open, rename or unlink against it fails with ACCESS_DENIED rather
///   than SHARING_VIOLATION. Replacing a destination puts it there, so
///   two threads saving to one path race through that window (#415).
///
/// Both clear on their own within microseconds. Everything else — a
/// read-only destination, a directory in the way, a missing privilege —
/// is permanent and must surface immediately rather than after a third
/// of a second of pointless sleeping.
///
/// Not `cfg(windows)`-gated so the table itself stays testable on every
/// platform; only its caller is Windows-only.
#[cfg_attr(not(windows), allow(dead_code))]
fn is_transient_rename_error(raw_os_error: Option<i32>) -> bool {
    matches!(raw_os_error, Some(5) | Some(32))
}

/// Atomically rename `tmp` over `path`, retrying briefly with backoff on
/// the transient Windows failures above — the same posture cargo,
/// rustup, and git take (#313), and the same set the Python writer uses
/// (`turbovec-python/python/turbovec/_persist.py`). The two must stay in
/// step: they implement one protocol against one on-disk format.
pub(crate) fn rename_atomic(tmp: &Path, path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        let mut delay_ms = 1u64;
        for _ in 0..10 {
            match std::fs::rename(tmp, path) {
                Err(e) if is_transient_rename_error(e.raw_os_error()) => {
                    std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                    delay_ms = (delay_ms * 2).min(64);
                }
                r => return r,
            }
        }
    }
    std::fs::rename(tmp, path)
}

/// True when `s` is the part after `<dest>.tmp.` of a name this module
/// generates: `{pid}.{seq}` (pre-0.10.1 writers) or `{pid}.{seq}.{hex8}`.
fn is_our_tmp_suffix(s: &str) -> bool {
    let mut parts = s.split('.');
    let (Some(pid), Some(seq)) = (parts.next(), parts.next()) else {
        return false;
    };
    if pid.is_empty() || seq.is_empty() {
        return false;
    }
    if !pid.bytes().all(|b| b.is_ascii_digit()) || !seq.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    match (parts.next(), parts.next()) {
        (None, _) => true,
        (Some(rand), None) => rand.len() == 8 && rand.bytes().all(|b| b.is_ascii_hexdigit()),
        _ => false,
    }
}

/// Destinations already swept by this process, so a repeated save to
/// the same path pays the directory scan once rather than every time.
/// The leak this reclaims is per-*process* (a writer killed before its
/// rename), so a crash-looping writer still sweeps on each restart.
static SWEPT: std::sync::Mutex<Option<std::collections::HashSet<PathBuf>>> =
    std::sync::Mutex::new(None);

/// Serializes the tests that assert on `SWEPT`'s claim/skip behaviour.
/// The set is process-global and `claim_first_sweep` declines rather than
/// waits (see below), so a test holding it would make a concurrent test's
/// claim fail for the wrong reason.
#[cfg(test)]
static SWEPT_TEST_SERIAL: Mutex<()> = Mutex::new(());

/// True the first time this process is asked about `path`.
///
/// `try_lock`, not `lock`, for the same reason as the two codebook memos
/// above: a fork that lands while another thread holds this would leave
/// the child with a permanently-locked mutex, and the child's first
/// `write` would hang on it. Declining to claim (the poisoned-lock
/// behaviour this already had) just skips an opportunistic sweep.
fn claim_first_sweep(path: &Path) -> bool {
    let Ok(mut guard) = SWEPT.try_lock() else {
        return false;
    };
    let seen = guard.get_or_insert_with(std::collections::HashSet::new);
    // A process writing an unbounded number of distinct destinations
    // must not accumulate them forever; past the cap, fall back to
    // sweeping every time (correct, just not deduplicated).
    if seen.len() >= 4096 {
        return true;
    }
    seen.insert(path.to_path_buf())
}

/// Best-effort reclaim of temp files leaked by a killed writer (#299):
/// SIGKILL between temp creation and rename leaves a full-size
/// `<dest>.tmp.*` sibling that nothing else ever deletes, so a
/// crash-looping writer fills the volume. Swept on this process's first
/// save to a given destination — the scan is `O(entries in the parent
/// directory)`, which measurably slows saves into a crowded directory
/// if repeated (+38% at 20k siblings), and nothing new can leak at that
/// destination while this process is the one writing it. Only names
/// matching this module's exact pattern for this destination are
/// candidates, and only when their mtime is over an hour old — a save
/// takes seconds, so a live writer's in-flight temp is never touched.
/// Every error is ignored: sweeping is opportunistic and must never
/// fail a save.
pub(crate) fn sweep_stale_tmps(path: &Path) {
    // A destination that is itself one of our temp names means someone
    // is staging through us — `_persist.atomic_save` writes the index to
    // a fresh `<dest>.tmp.…` name on every save, so all four Python
    // integrations land here. Such a destination is unique per save: it
    // can have no leaked siblings of its own, the memo below would never
    // hit, and the scan would run on every save. Skip it; the outer
    // destination gets swept when something writes to it directly.
    if path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .and_then(|n| n.rsplit_once(".tmp."))
        .is_some_and(|(_, suffix)| is_our_tmp_suffix(suffix))
    {
        return;
    }
    if !claim_first_sweep(path) {
        return;
    }
    const STALE_AGE: std::time::Duration = std::time::Duration::from_secs(60 * 60);
    let Some(base) = path.file_name().and_then(std::ffi::OsStr::to_str) else {
        return;
    };
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // Split at the LAST `.tmp.` so a destination whose own name
        // contains one still parses; `is_our_tmp_suffix` rejects a wrong
        // split anyway.
        let Some((stem, suffix)) = name.rsplit_once(".tmp.") else {
            continue;
        };
        if !is_our_tmp_suffix(suffix) {
            continue;
        }
        // `tmp_sibling` truncates the destination's basename when
        // `base + suffix` would exceed NAME_MAX, so for a long
        // destination the leaked temp does NOT start with the full
        // basename and matching on it swept nothing — the reclaim was a
        // permanent no-op past 234 bytes (#488).
        //
        // A truncated temp is recognisable exactly: its stem is a prefix
        // of the destination's basename, and the cut was chosen to make
        // the whole name land on NAME_MAX. Requiring that length keeps
        // this from matching an unrelated destination that merely shares
        // a prefix.
        // Reproduce `tmp_sibling`'s cut exactly rather than inferring it
        // from the length: it walks *backwards* to a char boundary, so a
        // budget landing inside a multi-byte character emits a name 1-3
        // bytes short of NAME_MAX. An equality-on-length test misses
        // every such name, which would leave #488 alive for non-ASCII
        // destinations. The suffix is known here, so the budget — and
        // therefore the cut — is computable.
        let budget = TMP_NAME_MAX.saturating_sub(".tmp.".len() + suffix.len());
        let ours = stem == base || {
            // Searching down for the boundary rather than walking a
            // mutable index: the walk's guard was equivalent under
            // mutation (`is_char_boundary(0)` is always true, so `cut > 0`
            // never decides anything) and its `cut -= 1` inverted into an
            // unbounded loop. Neither is testable; both are gone here.
            let want = budget.min(base.len());
            let cut = (0..=want).rev().find(|&c| base.is_char_boundary(c));
            cut.is_some_and(|cut| {
                // A truncated stem is itself a legal filename, so the name
                // we would reclaim is byte-identical to the temp a *shorter*
                // destination whose whole basename is that prefix would
                // create — length cannot separate them, since both land on
                // NAME_MAX by construction. If such a destination exists,
                // the file is more plausibly its write than our leak, so
                // leave it: that destination's own writer sweeps it, and
                // failing to reclaim a leak is survivable where deleting a
                // live staged index is not.
                // No `cut < base.len()` clause: at equality `stem` is
                // the whole basename, which branch one already matched,
                // so the test can never decide anything (the gate caught
                // it as an untestable mutant).
                !stem.is_empty()
                    && stem == &base[..cut]
                    && !entry.path().with_file_name(stem).exists()
            })
        };
        if !ours {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let stale = meta.modified().is_ok_and(|m| {
            std::time::SystemTime::now()
                .duration_since(m)
                .is_ok_and(|age| age > STALE_AGE)
        });
        if stale {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// In `Durable` mode, fsync the parent directory after the rename so the
/// rename itself — not just the new file's contents — is on stable
/// storage; without it, power loss can roll the rename back to the
/// previous file. (That older state is still a complete index either
/// way; this closes the gap between the documented guarantee and the
/// implementation.) Windows has no directory-fsync equivalent; rename
/// durability there follows NTFS metadata journaling.
fn sync_parent_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            let dir = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
            File::open(dir)?.sync_all()?;
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Run the post-rename parent-directory fsync, which cannot fail the save.
///
/// The rename is the commit point: once it returns, the new file is the
/// one readers see and the temp name is gone. A failure of the directory
/// fsync *after* that point is a durability shortfall on an
/// already-committed file, not a failed save — reporting it as `Err`
/// would tell a caller its previous file is still in place when it is
/// not, sending retry/rollback policies down a destructive path, and the
/// error cleanup would then try to unlink a temp name that no longer
/// exists (#365). So the save succeeds and the shortfall is reported as
/// a non-fatal diagnostic through [`crate::warning`], which an embedder
/// can route into its own logging (or silence) instead of being handed
/// an unconditional line on stderr.
pub(crate) fn sync_parent_dir_after_commit(path: &Path) {
    if let Err(e) = sync_parent_dir(path) {
        crate::warning::warn(&format!(
            "{} was written and committed, but syncing its parent directory \
             failed ({e}); the file is visible now but the rename may not \
             survive power loss",
            path.display(),
        ));
    }
}

/// Serialize the fixed head (post-magic/version core header + codebook)
/// into a buffer.
fn head_core(
    head: &mut Vec<u8>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
) -> io::Result<()> {
    assert_codebook_lengths(bit_width, codebook_boundaries, codebook_centroids);
    head.push(bit_width as u8);
    head.extend_from_slice(&(dim as u32).to_le_bytes());
    head.extend_from_slice(&(n_vectors as u64).to_le_bytes());
    for &b in codebook_boundaries {
        head.extend_from_slice(&b.to_le_bytes());
    }
    for &c in codebook_centroids {
        head.extend_from_slice(&c.to_le_bytes());
    }
    Ok(())
}

/// Serialize the post-codes tail sections (scales + TQ+ trailer).
fn tail_core(tail: &mut Vec<u8>, scales: &[f32], tqplus_shift: &[f32], tqplus_scale: &[f32]) {
    for &s in scales {
        tail.extend_from_slice(&s.to_le_bytes());
    }
    tail.extend_from_slice(&(tqplus_shift.len() as u32).to_le_bytes());
    for &s in tqplus_shift {
        tail.extend_from_slice(&s.to_le_bytes());
    }
    for &s in tqplus_scale {
        tail.extend_from_slice(&s.to_le_bytes());
    }
}

/// Atomic path write with the large codes section written by parallel
/// positioned writes (mirror of the load-side fast path): head and tail
/// serialize into small buffers and pwrite at their computed offsets;
/// the codes span is split across scoped threads. Byte-identical output
/// to the streamed writer (same sections, same order on disk), and the
/// durability protocol is unchanged: everything lands in the temp file,
/// fsync, then atomic rename — the commit point, after which nothing
/// can return `Err` (#365). Small payloads take one serial write.
fn write_atomic_parallel(
    path: &Path,
    durability: Durability,
    magic: &[u8; 4],
    version: u8,
    head_fn: impl FnOnce(&mut Vec<u8>) -> io::Result<()>,
    codes: &[u8],
    codes_transform: Option<fn(&[u8], &mut Vec<u8>)>,
    tail_fn: impl FnOnce(&mut Vec<u8>) -> io::Result<()>,
) -> io::Result<()> {
    let mut head = Vec::with_capacity(4096);
    head.extend_from_slice(magic);
    head.push(version);
    head_fn(&mut head)?;
    let mut tail = Vec::new();
    tail_fn(&mut tail)?;

    sweep_stale_tmps(path);
    let (f, tmp) = create_tmp(path)?;
    let result = (|| {
        const PAR_MIN: usize = 8 * 1024 * 1024;
        let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(4);
        if codes.len() < PAR_MIN || n_threads < 2 {
            let mut w = BufWriter::new(&f);
            w.write_all(&head)?;
            match codes_transform {
                Some(t) => {
                    let mut buf = Vec::new();
                    t(codes, &mut buf);
                    w.write_all(&buf)?;
                }
                None => w.write_all(codes)?,
            }
            w.write_all(&tail)?;
            w.flush()?;
            drop(w);
        } else {
            f.set_len((head.len() + codes.len() + tail.len()) as u64)?;
            write_all_at(&f, &head, 0)?;
            write_all_at(&f, &tail, (head.len() + codes.len()) as u64)?;
            let base = head.len() as u64;
            let chunk = codes.len().div_ceil(n_threads).max(PAR_MIN).next_multiple_of(4096);
            let n_chunks = codes.len().div_ceil(chunk);
            let next = std::sync::atomic::AtomicUsize::new(0);
            let failed = std::sync::atomic::AtomicBool::new(false);
            let err: std::sync::Mutex<Option<io::Error>> = std::sync::Mutex::new(None);
            std::thread::scope(|s| {
                for _ in 0..n_threads.min(n_chunks) {
                    s.spawn(|| {
                        // Per-thread scratch for the fused transform: the
                        // chunk deinterleaves into it, then writes — no
                        // whole-payload intermediate, and the transform
                        // overlaps the other threads' device writes.
                        // (Chunks are 4096-multiples, so block-aligned.)
                        let mut scratch = Vec::new();
                        loop {
                            if failed.load(std::sync::atomic::Ordering::Relaxed) {
                                break;
                            }
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= n_chunks {
                                break;
                            }
                            let off = i * chunk;
                            let this = chunk.min(codes.len() - off);
                            let src: &[u8] = match codes_transform {
                                Some(t) => {
                                    t(&codes[off..off + this], &mut scratch);
                                    &scratch
                                }
                                None => &codes[off..off + this],
                            };
                            if let Err(e) = write_all_at(&f, src, base + off as u64) {
                                failed.store(true, std::sync::atomic::Ordering::Relaxed);
                                *err.lock().expect("err lock") = Some(e);
                                break;
                            }
                        }
                    });
                }
            });
            if let Some(e) = err.into_inner().expect("err lock") {
                return Err(e);
            }
        }
        if durability == Durability::Durable {
            f.sync_all()?;
        }
        drop(f);
        rename_atomic(&tmp, path)?;
        if durability == Durability::Durable {
            sync_parent_dir_after_commit(path);
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(unix)]
fn write_all_at(f: &File, buf: &[u8], off: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.write_all_at(buf, off)
}

#[cfg(windows)]
fn write_all_at(f: &File, mut buf: &[u8], mut off: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match f.seek_write(buf, off)? {
            // std's `write_all` turns a 0-byte write into WriteZero
            // rather than retrying forever; mirror it.
            0 => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ))
            }
            n => {
                buf = &buf[n..];
                off += n as u64;
            }
        }
    }
    Ok(())
}

/// Core header + packed codes + per-vector scales + TQ+ calibration —
/// shared by `.tv` and `.tvim`. Writes the v5 core layout: `bit_width`
/// u8 + `dim` u32 + `n_vectors` u64 (no u32 count ceiling), then the
/// payload. No rotation fingerprint — the v5 rotation is deterministic,
/// so there is no drift to guard against.
#[allow(clippy::too_many_arguments)]
fn write_core<W: Write>(
    w: &mut W,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    codes_blocked_seq: &[u8],
    codebook_boundaries: &[f32],
    codebook_centroids: &[f32],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    assert_codebook_lengths(bit_width, codebook_boundaries, codebook_centroids);
    w.write_all(&[bit_width as u8])?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&(n_vectors as u64).to_le_bytes())?;
    for &b in codebook_boundaries {
        w.write_all(&b.to_le_bytes())?;
    }
    for &c in codebook_centroids {
        w.write_all(&c.to_le_bytes())?;
    }
    w.write_all(codes_blocked_seq)?;
    for &s in scales {
        w.write_all(&s.to_le_bytes())?;
    }
    // TQ+ trailer. Lengths are asserted by the callers before any file
    // is created (`assert_tqplus_calibration`).
    let n_calib = tqplus_shift.len() as u32;
    w.write_all(&n_calib.to_le_bytes())?;
    for &s in tqplus_shift {
        w.write_all(&s.to_le_bytes())?;
    }
    for &s in tqplus_scale {
        w.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

/// Read the core payload, dispatching on the version byte. v5 is the
/// only supported version; versions 1 through 4 are refused with the
/// actionable rebuild error (the v5 rotation break made those bytes
/// undecodable), and any other value is an unknown format.
fn read_core_versioned<R: Read>(
    r: &mut R,
    version: u8,
    expected: u8,
    label: &str,
    alloc_cap: u64,
) -> io::Result<CoreLoad> {
    match version {
        6 => read_core_v6(r, alloc_cap),
        5 => read_core_v5(r),
        1..=4 => Err(incompatible_version_error(version, label)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported {label} format version: {version} (this build \
                 writes version {expected} and reads versions 5 and {expected})",
            ),
        )),
    }
}

/// v6: identical header and trailer to v5, but the code payload is the
/// arch-neutral sequential blocked layout (see [`CodePayload`]); its
/// length is the padded blocked size, not the packed row size.
fn read_core_v6<R: Read>(r: &mut R, alloc_cap: u64) -> io::Result<CoreLoad> {
    let (bit_width, dim, n_vectors) = read_v5_header(r)?;
    let n_levels = 1usize << bit_width;
    let boundaries = read_f32_array(r, n_levels - 1)?;
    let centroids = read_f32_array(r, n_levels)?;
    validate_codebook(bit_width, dim, n_vectors, &boundaries, &centroids)?;
    let blocked_bytes = v6_blocked_len(bit_width, dim, n_vectors)?;
    let blocked = read_exact_vec_capped(r, blocked_bytes, alloc_cap)?;
    let scales = read_scales_validated(r, n_vectors)?;
    let (tqplus_shift, tqplus_scale) = read_tqplus_trailer(r, dim)?;
    Ok((
        bit_width,
        dim,
        n_vectors,
        CodePayload::BlockedSeq { codes: blocked, boundaries, centroids },
        scales,
        tqplus_shift,
        tqplus_scale,
    ))
}

/// Codebooks already proven to satisfy the Lloyd-Max fixed-point
/// condition in this process, keyed by `(bit_width, dim)`.
///
/// `try_lock` only, never `lock` — same requirement, same reason as
/// [`crate::codebook`]'s memo: this sits on the **load** path, the first
/// thing a forked worker touches, and a mutex inherited in the locked
/// state from a thread the child does not have would never be released.
/// A miss re-runs the closed-form residual check (tens of microseconds),
/// so falling through is only ever slower, never weaker: nothing is
/// accepted that the check has not accepted.
static ACCEPTED_CODEBOOKS: Mutex<Option<HashMap<(usize, usize), Vec<f32>>>> = Mutex::new(None);

/// Codebook value validation shared by the streamed and fast v6 loaders.
fn validate_codebook(
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    boundaries: &[f32],
    centroids: &[f32],
) -> io::Result<()> {
    // Codebook value validation (skipped for an empty index, whose
    // codebook is an ignored all-zero placeholder): search uses these to
    // decode every score, so a non-finite or out-of-support value would
    // silently poison results. |v| <= 1 is the quantizer's support.
    if n_vectors > 0 {
        for (name, vals) in [("boundaries", boundaries), ("centroids", centroids)] {
            if let Some((i, &v)) = vals
                .iter()
                .enumerate()
                .find(|(_, v)| !v.is_finite() || v.abs() > 1.0)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid codebook {name} at index {i}: {v} (must be finite, |v| <= 1)"),
                ));
            }
        }
        // Boundaries must be strictly increasing — search bisects them to
        // decode every score, so a shuffled or degenerate codebook loads
        // structurally clean but silently mis-scores everything.
        if let Some(i) = boundaries.windows(2).position(|w| w[0] >= w[1]) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid codebook boundaries at index {}: {} >= {} (must be strictly increasing)",
                    i, boundaries[i], boundaries[i + 1]
                ),
            ));
        }
        // Centroids must be strictly increasing too: the boundary scan
        // decodes a code as the count of boundaries below x, so a
        // collapsed or reversed centroid array loads structurally clean
        // and silently mis-scores every query (#320).
        if let Some(i) = centroids.windows(2).position(|w| w[0] >= w[1]) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid codebook centroids at index {}: {} >= {} (must be strictly increasing)",
                    i, centroids[i], centroids[i + 1]
                ),
            ));
        }
        // The centroids are then pinned by the Lloyd-Max fixed-point
        // condition they were solved for: each must equal the Beta
        // conditional mean of its own cell. Evaluating that residual is
        // a closed-form CDF expression costing tens of microseconds,
        // whereas re-deriving the codebook to compare against costs
        // 25–100 ms and put a full analytic solve on the load path
        // (#357). Tolerance 1e-4 as before: the true codebook's residual
        // is at most 1.1e-6 over every supported (bit_width, dim), while
        // displacing a centroid by d moves the residual by ~d/2, so the
        // rejection strength for the #320 attacks is unchanged (a
        // reversed array scores 7e-2, a collapsed one 2e-2).
        const CODEBOOK_TOLERANCE: f64 = 1e-4;
        // Loads repeat: an accepted (bit_width, dim, centroids) triple is
        // remembered so a second load of the same shape is a slice
        // compare. Bit-exact, so this can only skip work for a codebook
        // already proven to satisfy the condition above.
        let already = ACCEPTED_CODEBOOKS.try_lock().ok().is_some_and(|m| {
            m.as_ref()
                .and_then(|m| m.get(&(bit_width, dim)))
                .is_some_and(|c| c == centroids)
        });
        if !already {
            let residual = crate::codebook::fixed_point_residual(dim, centroids);
            if residual > CODEBOOK_TOLERANCE {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid codebook centroids for bit_width {bit_width}, dim {dim}: \
                         they are not the Lloyd-Max quantizer of the coordinate \
                         distribution (fixed-point residual {residual:.3e} exceeds \
                         {CODEBOOK_TOLERANCE:e}); the codebook is a pure function of the \
                         header and this file's disagrees — corrupt or tampered file",
                    ),
                ));
            }
            if let Ok(mut m) = ACCEPTED_CODEBOOKS.try_lock() {
                m.get_or_insert_with(HashMap::new)
                    .insert((bit_width, dim), centroids.to_vec());
            }
        }
        // With the centroids pinned, the boundaries follow exactly: each
        // is by construction the f32 midpoint of its two neighbouring
        // f32 centroids, and that identity is bit-exact on every
        // platform (f32 add is correctly rounded, `* 0.5` is exact) —
        // it is what makes the codebook cross-platform reproducible at
        // all (#259). So it needs no tolerance to hide in.
        for i in 0..boundaries.len() {
            let expected = (centroids[i] + centroids[i + 1]) * 0.5;
            if boundaries[i].to_bits() != expected.to_bits() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid codebook boundaries at index {i}: {} (expected {expected}, \
                         the exact f32 midpoint of centroids {i} and {}) — corrupt or \
                         tampered file",
                        boundaries[i],
                        i + 1,
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// The v6 blocked-payload length for a validated header, with checked
/// arithmetic (dim/n_vectors are attacker-controlled).
fn v6_blocked_len(bit_width: usize, dim: usize, n_vectors: usize) -> io::Result<usize> {
    if dim == 0 {
        return Ok(0);
    }
    let codes_per_byte = 8 / bit_width;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = n_vectors
        .checked_add(crate::BLOCK - 1)
        .map(|x| x / crate::BLOCK)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "block count overflows usize"))?;
    n_blocks
        .checked_mul(n_byte_groups)
        .and_then(|x| x.checked_mul(crate::BLOCK))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "blocked code size overflows usize"))
}

/// v5: header (`bit_width` u8 + `dim` u32 + `n_vectors` u64) + codes +
/// scales + TQ+ trailer. No rotation fingerprint — the v5 rotation is
/// deterministic, so there is no drift to verify.
fn read_core_v5<R: Read>(r: &mut R) -> io::Result<CoreLoad> {
    let (bit_width, dim, n_vectors) = read_v5_header(r)?;
    let (packed_codes, scales) = read_codes_scales(r, bit_width, dim, n_vectors)?;
    let (tqplus_shift, tqplus_scale) = read_tqplus_trailer(r, dim)?;

    Ok((
        bit_width,
        dim,
        n_vectors,
        CodePayload::Packed(packed_codes),
        scales,
        tqplus_shift,
        tqplus_scale,
    ))
}

/// The header shared by v5 and v6: `bit_width` u8 + `dim` u32 +
/// `n_vectors` u64, with field validation.
fn read_v5_header<R: Read>(r: &mut R) -> io::Result<(usize, usize, usize)> {
    let mut header = [0u8; V5_HEADER_SIZE];
    r.read_exact(&mut header)?;
    let bit_width = header[0] as usize;
    let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let n_vectors_u64 = u64::from_le_bytes([
        header[5], header[6], header[7], header[8],
        header[9], header[10], header[11], header[12],
    ]);
    let n_vectors = usize::try_from(n_vectors_u64).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "n_vectors {n_vectors_u64} does not fit this platform's usize \
                 (32-bit build); the index cannot be loaded here",
            ),
        )
    })?;
    validate_header_fields(bit_width, dim, n_vectors)?;
    Ok((bit_width, dim, n_vectors))
}

/// TQ+ trailer: `n_calib` (0 or `dim`) + shift + scale arrays, with
/// value-level validation.
fn read_tqplus_trailer<R: Read>(r: &mut R, dim: usize) -> io::Result<(Vec<f32>, Vec<f32>)> {
    let mut n_calib_bytes = [0u8; 4];
    r.read_exact(&mut n_calib_bytes)?;
    let n_calib = u32::from_le_bytes(n_calib_bytes) as usize;
    if n_calib != 0 && n_calib != dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid TQ+ n_calib {n_calib}: must be 0 or equal to dim {dim}"),
        ));
    }
    let tqplus_shift = read_f32_array(r, n_calib)?;
    let tqplus_scale = read_f32_array(r, n_calib)?;

    validate_calibration(&tqplus_shift, &tqplus_scale)?;

    Ok((tqplus_shift, tqplus_scale))
}

/// Calibration bounds that keep the query transform finite.
///
/// Both are derived from the real transform in `search::calibrate_queries`
/// and from the input cap the add and search paths enforce
/// (`MAX_INPUT_MAGNITUDE = 1e16`), and both scale with `dim` because the
/// transform reduces across every coordinate:
///
/// ```text
/// calib_row[d] = q_row[d] / tqplus_scale[d];      // per coordinate
/// bias        -= q_row[d] as f64 * shift[d] as f64;  // summed over dim
/// ```
///
/// A first cut budgeted the whole f32 range to the single division and
/// picked flat constants. That was wrong in both directions: the divided
/// query is reduced across `dim` coordinates before it becomes a score,
/// and the bias is a `dim`-long dot product narrowed back to f32, so a
/// scale of `1e-22` still produced all-NaN scores at `dim = 128` and a
/// shift at the flat bound did the same.
///
/// The factor of 10 on each is margin, and costs nothing: a real fit is
/// O(1) — magnitude-invariant, with a minimum near 1.62 even on a corpus
/// scaled by 1e-10 — so these sit fifteen or more orders away from
/// anything an honest index contains.
///
/// What they do *not* cover: the score is finally multiplied by the
/// per-vector scale, which is data rather than calibration. That factor
/// is bounded separately by [`MAX_VECTOR_SCALE`], and an adversarial
/// combination of a legal-but-extreme calibration with a
/// legal-but-extreme per-vector scale can still overflow. These bounds
/// exclude the poisonous calibration range; they are not a proof of
/// finiteness over every admissible input.
pub(crate) fn min_tqplus_scale(dim: usize) -> f32 {
    let need = (dim.max(1) as f32) * crate::MAX_INPUT_MAGNITUDE / f32::MAX;
    need * 10.0
}

pub(crate) fn max_tqplus_shift(dim: usize) -> f32 {
    let cap = f32::MAX / ((dim.max(1) as f32) * crate::MAX_INPUT_MAGNITUDE);
    cap / 10.0
}

/// Largest per-vector scale that cannot by itself drive a score to
/// infinity. A scale is a vector magnitude and coordinates are capped at
/// `1e16`, so this leaves six orders over the largest reachable value.
pub(crate) const MAX_VECTOR_SCALE: f32 = 1e22;

/// Value-level calibration validation — THE rule, shared by every
/// loader (v6 here, v7 in `io_v7`): the encoder only ever emits finite
/// shifts and strictly-positive scales, so anything else is corruption
/// or an attacker payload. Search divides by `tqplus_scale`, so a
/// zero/negative/non-finite value — which a bare is_finite() check
/// would not fully catch — silently turns every query's scores into
/// NaN/Inf.
pub(crate) fn validate_calibration(
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    if let Some((i, &v)) = tqplus_shift
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || v.abs() > max_tqplus_shift(tqplus_shift.len()))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid TQ+ shift at coord {i}: {v} (must be finite and \
                 |shift| <= {:e} at dim {})",
                max_tqplus_shift(tqplus_shift.len()),
                tqplus_shift.len()
            ),
        ));
    }
    if let Some((i, &v)) = tqplus_scale
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || **v < min_tqplus_scale(tqplus_scale.len()))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid TQ+ scale at coord {i}: {v} (must be finite and \
                 >= {:e} at dim {}; search divides by it and sums across \
                 every coordinate, so a smaller value turns every score \
                 into Inf/NaN)",
                min_tqplus_scale(tqplus_scale.len()),
                tqplus_scale.len()
            ),
        ));
    }
    Ok(())
}

/// Header-field validation shared by every format version.
fn validate_header_fields(bit_width: usize, dim: usize, n_vectors: usize) -> io::Result<()> {
    // Validate header fields before allocating anything. The constructors
    // (`new`/`add_2d`) enforce these invariants, but the load path bypasses
    // them — so an untrusted file could otherwise smuggle a `bit_width` that
    // divides-by-zero in `pack::repack` (0 or >8), a `bit_width` of 5..8 that
    // silently passes `from_parts`'s length check and returns wrong scores,
    // or a `dim` that isn't a multiple of 8 (the bit-plane layout is
    // undefined for it and the size formulas diverge → panic). `dim == 0` is
    // the lazy-index sentinel and is only valid alongside `n_vectors == 0`.
    if !(2..=4).contains(&bit_width) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid bit_width {bit_width}: must be 2, 3, or 4"),
        ));
    }
    if dim == 0 {
        if n_vectors != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dim 0 (lazy sentinel) requires n_vectors 0, got {n_vectors}"),
            ));
        }
    } else if dim % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid dim {dim}: must be a multiple of 8"),
        ));
    } else if dim > crate::MAX_DIM {
        // Bound the dim-dependent load-time allocations (codebook, blocked
        // layout, per-query rotate scratch): a tiny file can declare a
        // huge dim and drive a multi-GB allocation otherwise.
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid dim {dim}: exceeds maximum {}", crate::MAX_DIM),
        ));
    }
    Ok(())
}

/// Packed codes + per-vector scales (with value validation) for an
/// already-validated header. Shared by every format version.
fn read_codes_scales<R: Read>(
    r: &mut R,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
) -> io::Result<(Vec<u8>, Vec<f32>)> {
    // Checked arithmetic: `dim`/`n_vectors` are attacker-controlled, so
    // the product can overflow `usize` (on 32-bit targets this wrap would
    // yield an undersized buffer and later out-of-bounds reads).
    let packed_bytes = (dim / 8)
        .checked_mul(bit_width)
        .and_then(|x| x.checked_mul(n_vectors))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "packed code size overflows usize")
        })?;
    let packed_codes = read_exact_vec(r, packed_bytes)?;
    let scales = read_scales_validated(r, n_vectors)?;
    Ok((packed_codes, scales))
}

/// Per-vector scales with value validation, shared by v5 and v6.
fn read_scales_validated<R: Read>(r: &mut R, n_vectors: usize) -> io::Result<Vec<f32>> {
    let scales = read_f32_array(r, n_vectors)?;
    // Value-level validation: the encoder only ever emits finite,
    // non-negative per-vector scales. A NaN/Inf/negative scale loads
    // without structural error but silently corrupts search — an Inf
    // slot wins every top-1, a NaN slot vanishes from all results.
    if let Some((i, &s)) = scales
        .iter()
        .enumerate()
        .find(|(_, s)| !s.is_finite() || **s < 0.0 || **s > MAX_VECTOR_SCALE)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid per-vector scale at slot {i}: {s} (must be finite, \
                 non-negative and <= {MAX_VECTOR_SCALE:e})"
            ),
        ));
    }
    Ok(scales)
}

/// Read exactly `n` bytes without pre-allocating `n` up front. A malicious
/// header can declare a multi-gigabyte length from a tiny file; `read_to_end`
/// on a `take`-limited reader grows the buffer only to the bytes actually
/// present, so we never reserve the attacker's claimed size before confirming
/// the data exists. The length check then rejects a truncated file cleanly.
/// Read a whole file of known length with positioned reads across scoped
/// threads — the generic-fallback whole-file form of
/// [`read_range_parallel`], used when the fast v6 path declines. The
/// buffer is allocated uninitialized and every
/// byte is written by exactly one positioned read before `set_len`
/// exposes it (on any error the Vec drops with len 0, so uninitialized
/// bytes are never readable). Scoped `std::thread` (not rayon) keeps
/// this outside the fork-safety pool machinery. Small files read
/// serially.
fn read_file_parallel(f: &File, len: u64) -> io::Result<Vec<u8>> {
    read_range_parallel(f, 0, len)
}

/// Positioned parallel read of `[off, off+len)` into a fresh exact-size
/// buffer — the offset form lets the fast v6 loader read the codes
/// section straight into its final home (no intermediate whole-file
/// buffer, no extraction copy).
fn read_range_parallel(f: &File, range_off: u64, len: u64) -> io::Result<Vec<u8>> {
    read_range_parallel_transform(f, range_off, len, None)
}

/// [`read_range_parallel`] with an optional in-place transform fused
/// into the read: each thread reads its span in L2-sized sub-chunks
/// (256 KB, a 32-byte-block multiple) and transforms each sub-chunk
/// immediately, while its lines are still resident — the transform's
/// separate cold memory pass disappears.
fn read_range_parallel_transform(
    f: &File,
    range_off: u64,
    len: u64,
    transform: Option<fn(&mut [u8])>,
) -> io::Result<Vec<u8>> {
    let len_usize = usize::try_from(len)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "file too large for this platform"))?;
    const CHUNK_MIN: usize = 8 * 1024 * 1024;
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut buf: Vec<u8> = Vec::with_capacity(len_usize);
    if len_usize < 2 * CHUNK_MIN || n_threads < 2 {
        // Positioned serial read — must honor `range_off` (a plain
        // `take` would read from the descriptor's current position).
        buf.resize(len_usize, 0);
        read_exact_at(f, &mut buf, range_off)?;
        if let Some(t) = transform {
            t(&mut buf);
        }
        return Ok(buf);
    }
    // How the span is divided depends on whether a transform is fused
    // into the read, because that decides whether the chunks cost the
    // same as each other.
    //
    // With a transform (x86, where the stored layout is interleaved on
    // the way in), each chunk carries a fixed amount of compute per
    // byte, so chunk times are uniform and the split can be static —
    // but three chunks per thread rather than one, so a thread that
    // draws a slow chunk costs the join a third as much (H21 took it
    // from one to two at x1.077, H47 from two to three at x1.060, on a
    // c3-standard-8).
    //
    // Without one (every other target reads its native layout straight
    // through), what is left is page-fault and page-cache variance,
    // which is *not* uniform across chunks — so an even split leaves the
    // join waiting on whichever thread drew the slow one. Fixed 4 MB
    // chunks (half of CHUNK_MIN) give the queue below more chunks than
    // threads and let a thread that finishes early steal the difference
    // (H18: x1.153 on a c4a-standard-8 over an even split; the
    // transform side loses from work-stealing instead — H17, 0 of 5 —
    // so it keeps the static split).
    //
    // See LOG_persist.md H15-H18, H21, H47.
    let chunk = match transform {
        Some(_) => len_usize.div_ceil(n_threads * 3).max(1 << 20).next_multiple_of(4096),
        None => (CHUNK_MIN / 2).next_multiple_of(4096),
    };
    let n_chunks = len_usize.div_ceil(chunk);
    // Pointer wrapper carrying real provenance across the thread
    // boundary (an integer round-trip would fail strict-provenance
    // tooling like Miri).
    #[derive(Clone, Copy)]
    struct BasePtr(*mut u8);
    unsafe impl Send for BasePtr {}
    unsafe impl Sync for BasePtr {}
    let base = BasePtr(buf.spare_capacity_mut().as_mut_ptr() as *mut u8);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let err: std::sync::Mutex<Option<io::Error>> = std::sync::Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..n_threads.min(n_chunks) {
            s.spawn(|| {
                // Capture the wrapper whole (2021 field-precise capture
                // would otherwise grab the raw pointer field directly).
                let base = base;
                loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n_chunks {
                    break;
                }
                let off = i * chunk;
                let this = chunk.min(len_usize - off);
                // SAFETY: chunks are disjoint [off, off+this) views of the
                // allocation; each is written (never read) by exactly one
                // thread via read_exact_at.
                let chunk =
                    unsafe { std::slice::from_raw_parts_mut(base.0.add(off), this) };
                let res = match transform {
                    None => read_exact_at(f, chunk, range_off + off as u64),
                    Some(t) => {
                        const SUB: usize = 256 * 1024;
                        let mut r = Ok(());
                        let mut sub_off = 0usize;
                        while sub_off < chunk.len() {
                            let sub_len = SUB.min(chunk.len() - sub_off);
                            let sub = &mut chunk[sub_off..sub_off + sub_len];
                            r = read_exact_at(f, sub, range_off + (off + sub_off) as u64);
                            if r.is_err() {
                                break;
                            }
                            t(sub);
                            sub_off += sub_len;
                        }
                        r
                    }
                };
                if let Err(e) = res {
                    *err.lock().expect("err lock") = Some(e);
                    break;
                }
                }
            });
        }
    });
    if let Some(e) = err.into_inner().expect("err lock") {
        return Err(e);
    }
    // SAFETY: every byte in 0..len was filled by a successful
    // read_exact_at (any failure returned above).
    unsafe { buf.set_len(len_usize) };
    Ok(buf)
}

/// Fast v6 loader: reads sections straight from the file. The prefix
/// (magic + version + header + codebook — 142 bytes at 4-bit, bounded by
/// PREFIX_MAX) is read serially and parsed with the same validators as
/// the streamed reader; the codes section is then parallel-pread into
/// its exact final buffer and transformed to the native layout in place
/// (pages just faulted by the read, so the transform runs warm); the
/// small tail (scales + TQ+ [+ id table]) is read serially and returned
/// for the caller to finish. Returns `Ok(None)` for anything that is
/// not a well-formed v6 magic+version prefix — callers fall back to the
/// generic reader, which produces the canonical error messages.
#[allow(clippy::type_complexity)]
fn try_load_v6_fast(
    f: &File,
    cap: u64,
    magic: &[u8; 4],
) -> io::Result<Option<(CoreLoad, Vec<u8>, usize, Vec<u64>, Vec<u64>)>> {
    const PREFIX_MAX: usize = 4096;
    let cap_usize = usize::try_from(cap)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "file too large for this platform"))?;
    let prefix_len = cap_usize.min(PREFIX_MAX);
    if prefix_len < 5 {
        return Ok(None);
    }
    let mut prefix = vec![0u8; prefix_len];
    read_exact_at(f, &mut prefix, 0)?;
    if &prefix[0..4] != magic || prefix[4] != 6 {
        return Ok(None);
    }
    let mut r: &[u8] = &prefix[5..];
    let (bit_width, dim, n_vectors) = read_v5_header(&mut r)?;
    let n_levels = 1usize << bit_width;
    let boundaries = read_f32_array(&mut r, n_levels - 1)?;
    let centroids = read_f32_array(&mut r, n_levels)?;
    validate_codebook(bit_width, dim, n_vectors, &boundaries, &centroids)?;
    let blocked_bytes = v6_blocked_len(bit_width, dim, n_vectors)?;
    let codes_start = (prefix_len - r.len()) as u64;
    let codes_end = codes_start
        .checked_add(blocked_bytes as u64)
        .filter(|&e| e <= cap)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "truncated file: expected {blocked_bytes} bytes, got {}",
                    cap.saturating_sub(codes_start)
                ),
            )
        })?;
    // Native transform fused into the read at L2 granularity; `None` when
    // the stored layout is already this target's native one.
    let transform: Option<fn(&mut [u8])> =
        crate::pack::native_transform(bit_width, dim / (8 / bit_width));
    // Tail (scales + TQ+ (+ id table for .tvim), ~a few MB) reads and
    // validates on a scoped thread while the main thread runs the big
    // parallel codes read — the tail pread + scales scan otherwise
    // serializes after it. Same scoped-thread pattern as the parallel
    // read itself.
    //
    // Gated on the codes payload being large enough to hide the spawn
    // (~20-30 µs), like the sibling size gates on the parallel read and
    // write paths: a small index's whole load is shorter than that, so
    // an unconditional spawn made small loads ~1.2x slower. Swept on a
    // c4a-standard-8: the crossover sits between 160 KB (neutral) and
    // 640 KB (already a 0.80x win), so gate at 256 KB — 1 MB was too
    // coarse and gave back the win at ~20k vectors.
    const TAIL_OVERLAP_MIN: usize = 256 * 1024;
    let tail_len = cap_usize - codes_end as usize;
    // The remainder past the TQ+ trailer (the `.tvim` id table, empty for
    // `.tv`) is handed back as the tail buffer plus the offset it starts
    // at, not as a fresh `Vec`. At 200k ids that slice is 1.6 MB, and
    // copying it out here only to have the caller copy it again into its
    // own buffer before decoding cost two allocations and two passes for
    // a borrow that outlives neither.
    type TailParts = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u8>, usize, Vec<u64>, Vec<u64>);
    let read_tail = || -> io::Result<TailParts> {
        // Zero-filled: forming a `&mut [u8]` over uninitialized capacity
        // and handing it to the safe Read family is documented UB, no
        // matter that today's impls only write (review finding; twice).
        // The memset this reintroduces is bounded by the tail (scales +
        // calibration + id table), not the codes payload — small next to
        // the read it fronts.
        let mut tail: Vec<u8> = vec![0u8; tail_len];
        read_exact_at(f, &mut tail, codes_end)?;
        let mut tr: &[u8] = &tail[..];
        let scales = read_scales_validated(&mut tr, n_vectors)?;
        let (tqplus_shift, tqplus_scale) = read_tqplus_trailer(&mut tr, dim)?;
        let rest_off = tail_len - tr.len();
        // Decode the id table here, on the tail thread, inside the codes
        // read's overlap window. H6 moved this *and* the sorted-table
        // build and regressed; this moves only the decode, halving the
        // allocation that lands in the window.
        let ids: Vec<u64> = if n_vectors > 0 && tail_len - rest_off >= n_vectors * 8 {
            tail[rest_off..rest_off + n_vectors * 8]
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .collect()
        } else {
            Vec::new()
        };
        // The sorted duplicate-check copy, built here alongside the
        // decode. H6 moved both and lost to allocation contention with
        // the reader threads; H38 moved the decode alone and won, so the
        // question is whether the second 1.6 MB now fits too.
        let sorted: Vec<u64> = if ids.is_empty() {
            Vec::new()
        } else {
            let mut v = ids.clone();
            v.sort_unstable();
            v
        };
        Ok((scales, tqplus_shift, tqplus_scale, tail, rest_off, ids, sorted))
    };
    let (codes_res, tail_res) = if blocked_bytes >= TAIL_OVERLAP_MIN {
        std::thread::scope(|s| {
            let tail_handle = s.spawn(read_tail);
            let codes =
                read_range_parallel_transform(f, codes_start, blocked_bytes as u64, transform);
            (
                codes,
                tail_handle
                    .join()
                    .unwrap_or_else(|p| std::panic::resume_unwind(p)),
            )
        })
    } else {
        let codes = read_range_parallel_transform(f, codes_start, blocked_bytes as u64, transform);
        (codes, read_tail())
    };
    let codes = codes_res?;
    let (scales, tqplus_shift, tqplus_scale, tail, rest_off, ids, sorted) = tail_res?;
    Ok(Some((
        (
            bit_width,
            dim,
            n_vectors,
            CodePayload::BlockedNative { codes, boundaries, centroids },
            scales,
            tqplus_shift,
            tqplus_scale,
        ),
        tail,
        rest_off,
        ids,
        sorted,
    )))
}

#[cfg(unix)]
fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}

#[cfg(windows)]
fn read_exact_at(f: &File, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, off)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated file"));
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

fn read_exact_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<u8>> {
    read_exact_vec_capped(r, n, 0)
}

/// [`read_exact_vec`] with a trusted allocation cap: when the declared
/// section size provably fits the source (`n <= alloc_cap`, where the
/// cap comes from real file metadata), pre-reserve it exactly — the
/// capped `read_to_end` then fills spare capacity with no zero-fill and
/// no growth-doubling copies. Otherwise fall back to the incremental
/// read, which never trusts `n` for allocation.
fn read_exact_vec_capped<R: Read>(r: &mut R, n: usize, alloc_cap: u64) -> io::Result<Vec<u8>> {
    let mut buf = if (n as u64) <= alloc_cap {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };
    let read = r.take(n as u64).read_to_end(&mut buf)?;
    if read != n {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("truncated file: expected {n} bytes, got {read}"),
        ));
    }
    Ok(buf)
}

#[cfg(test)]
mod rename_retry_tests {
    use super::*;

    // #415. The retry existed but whitelisted only ERROR_SHARING_VIOLATION
    // (32), so the ERROR_ACCESS_DENIED (5) that a delete-pending
    // destination returns went straight to the caller as a failed save.
    // The retry loop itself is `cfg(windows)` and cannot run here; the
    // decision table it consults is not, so the part that was actually
    // wrong is checked on every platform.
    #[test]
    fn transient_windows_rename_errors_are_retried() {
        assert!(is_transient_rename_error(Some(5)), "ERROR_ACCESS_DENIED");
        assert!(
            is_transient_rename_error(Some(32)),
            "ERROR_SHARING_VIOLATION"
        );
    }

    // The guard against widening this into retry-everything: a permanent
    // condition must surface on the first attempt.
    #[test]
    fn permanent_windows_rename_errors_are_not_retried() {
        for code in [2, 3, 19, 267, 1314] {
            assert!(
                !is_transient_rename_error(Some(code)),
                "os error {code} is permanent and must not be retried"
            );
        }
        assert!(!is_transient_rename_error(None));
    }
}

#[cfg(test)]
mod tmp_protocol_tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("turbovec_io_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn tmp_sibling_names_are_unique_and_recognizable() {
        let p = Path::new("/some/dir/index.tv");
        let a = tmp_sibling(p, 0xdeadbeef);
        let b = tmp_sibling(p, 0xdeadbeef);
        assert_ne!(a, b, "process-wide counter must distinguish same-rand names");
        let name = a.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("index.tv.tmp."));
        assert!(is_our_tmp_suffix(name.strip_prefix("index.tv.tmp.").unwrap()));
    }

    #[test]
    fn tmp_sibling_truncates_to_name_max() {
        // 250-char base + ~30-char suffix would exceed NAME_MAX (255).
        let long = "x".repeat(250);
        let p = std::env::temp_dir().join(&long);
        let tmp = tmp_sibling(&p, 1);
        let name = tmp.file_name().unwrap().to_str().unwrap();
        assert!(name.len() <= TMP_NAME_MAX, "temp name {} bytes", name.len());
        assert!(name.starts_with("xxx"));
        assert!(name.contains(".tmp."));
        // Short names are passed through untruncated.
        let short = tmp_sibling(Path::new("a.tv"), 1);
        assert!(short.file_name().unwrap().to_str().unwrap().starts_with("a.tv.tmp."));
    }

    #[test]
    fn is_our_tmp_suffix_matches_only_our_pattern() {
        assert!(is_our_tmp_suffix("1234.0"));
        assert!(is_our_tmp_suffix("1234.7.deadbeef"));
        assert!(!is_our_tmp_suffix("1234"));
        assert!(!is_our_tmp_suffix("1234.x"));
        assert!(!is_our_tmp_suffix("1234.7.deadbee")); // 7 hex chars
        assert!(!is_our_tmp_suffix("1234.7.deadbeef.9"));
        assert!(!is_our_tmp_suffix("abc.0"));
        assert!(!is_our_tmp_suffix(""));
    }

    #[cfg(unix)]
    #[test]
    fn create_new_refuses_planted_symlink() {
        // The O_EXCL property #293 relies on: an open through
        // `create_new` must refuse a symlink at the temp name instead
        // of following it and overwriting the victim.
        let dir = test_dir("symlink_excl");
        let victim = dir.join("victim.txt");
        std::fs::write(&victim, b"precious").unwrap();
        let planted = dir.join("index.tv.tmp.999.0.00000000");
        std::os::unix::fs::symlink(&victim, &planted).unwrap();
        let err = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&planted)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_tmp_skips_colliding_names() {
        // create_tmp must survive an existing file at a candidate name
        // by retrying with a fresh one, not fail the save.
        let dir = test_dir("create_tmp");
        let dest = dir.join("index.tv");
        let (f, tmp) = create_tmp(&dest).unwrap();
        drop(f);
        // A second call never reuses the live temp's name.
        let (f2, tmp2) = create_tmp(&dest).unwrap();
        drop(f2);
        assert_ne!(tmp, tmp2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sweep_skips_destinations_that_are_themselves_temps() {
        let _serial = super::SWEPT_TEST_SERIAL
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // `_persist.atomic_save` writes the index to a fresh
        // `<dest>.tmp.{pid}.{seq}.{hex}` name on every save, so the
        // destination is unique per save and can have no leaked
        // siblings: sweeping it would scan the directory every time and
        // never dedup.
        let dir = test_dir("sweep_nested");
        let staged = dir.join(format!("index.tvim.tmp.{}.0.deadbeef", std::process::id()));
        sweep_stale_tmps(&staged);
        assert!(
            claim_first_sweep(&staged),
            "a staged temp destination must not be memoized — it was never swept"
        );
        // A real destination is still swept.
        let real = dir.join("index.tvim");
        sweep_stale_tmps(&real);
        assert!(!claim_first_sweep(&real), "a real destination is swept and memoized");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sweep_runs_once_per_destination_per_process() {
        let _serial = super::SWEPT_TEST_SERIAL
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // The scan is O(entries in the parent dir); repeated saves to
        // one destination must not pay it more than once.
        let dir = test_dir("sweep_once");
        let dest = dir.join("index.tv");
        assert!(claim_first_sweep(&dest), "first save sweeps");
        assert!(!claim_first_sweep(&dest), "later saves skip the scan");
        assert!(claim_first_sweep(&dir.join("other.tv")), "a new destination sweeps");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn sweep_removes_only_stale_matching_temps() {
        let dir = test_dir("sweep");
        let dest = dir.join("index.tv");
        std::fs::write(&dest, b"dest").unwrap();
        let stale = dir.join("index.tv.tmp.4242.0.deadbeef");
        let stale_legacy = dir.join("index.tv.tmp.4242.1");
        let fresh = dir.join("index.tv.tmp.4242.2.deadbeef");
        let other = dir.join("other.tv.tmp.4242.0.deadbeef");
        let non_pattern = dir.join("index.tv.tmp.notes");
        for p in [&stale, &stale_legacy, &fresh, &other, &non_pattern] {
            std::fs::write(p, b"x").unwrap();
        }
        // Age the stale candidates well past the one-hour threshold.
        for p in [&stale, &stale_legacy, &other] {
            let out = std::process::Command::new("touch")
                .args(["-t", "202001010000", p.to_str().unwrap()])
                .status()
                .unwrap();
            assert!(out.success());
        }
        sweep_stale_tmps(&dest);
        assert!(!stale.exists(), "stale matching temp must be swept");
        assert!(!stale_legacy.exists(), "stale legacy-pattern temp must be swept");
        assert!(fresh.exists(), "fresh temp (possible live writer) must survive");
        assert!(other.exists(), "another destination's temp must survive");
        assert!(non_pattern.exists(), "non-matching name must survive");
        assert!(dest.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn read_f32_array<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let n_bytes = n
        .checked_mul(4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "f32 array size overflows usize"))?;
    let bytes = read_exact_vec(r, n_bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[cfg(test)]
mod fork_safety_tests {
    use super::{claim_first_sweep, validate_codebook, ACCEPTED_CODEBOOKS, SWEPT};
    use std::path::Path;
    use std::sync::mpsc;
    use std::time::Duration;

    /// Run `body` on another thread while this one holds a process-global
    /// lock that it will not release until `body` has answered — the
    /// state a forked child inherits for every mutex whose owner did not
    /// come across the `fork`. Returns false if `body` never finished.
    fn completes_while_lock_held<T: Send + 'static>(
        hold: impl FnOnce() -> Box<dyn std::any::Any>,
        body: impl FnOnce() -> T + Send + 'static,
    ) -> bool {
        let held = hold();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = tx.send(body());
        });
        let finished = rx.recv_timeout(Duration::from_secs(30)).is_ok();
        drop(held);
        let _ = worker.join();
        finished
    }

    /// The v6 load path consults a memo of already-accepted codebooks.
    /// Taking it with a blocking `lock()` puts a deadlock on the load
    /// path of any forked worker — the #147/#288/#321/#364 failure mode.
    #[test]
    fn codebook_validation_does_not_block_on_a_held_memo_lock() {
        let (boundaries, centroids) = crate::codebook::codebook(4, 64);
        let ok = completes_while_lock_held(
            || Box::new(ACCEPTED_CODEBOOKS.lock().unwrap_or_else(|e| e.into_inner())),
            move || validate_codebook(4, 64, 1, &boundaries, &centroids),
        );
        assert!(
            ok,
            "validate_codebook blocked on the accepted-codebook memo — a forked \
             worker would hang here on its first load",
        );
    }

    /// Same hazard on the save path: the stale-temp sweep dedupes
    /// destinations through a process-global set.
    #[test]
    fn sweep_claim_does_not_block_on_a_held_lock() {
        let _serial = super::SWEPT_TEST_SERIAL
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ok = completes_while_lock_held(
            || Box::new(SWEPT.lock().unwrap_or_else(|e| e.into_inner())),
            || claim_first_sweep(Path::new("/nonexistent/fork-safety-probe.tv")),
        );
        assert!(
            ok,
            "claim_first_sweep blocked on the swept-destination lock — a forked \
             worker would hang on its first write",
        );
    }
}

#[cfg(test)]
mod accepted_codebook_memo_tests {
    use super::{validate_codebook, ACCEPTED_CODEBOOKS};

    /// The accepted-codebook memo is keyed by `(bit_width, dim)`, but a hit
    /// requires the stored centroids to match the file's *bit-exactly*. That
    /// comparison is what keeps the memo a pure optimisation: a second file
    /// claiming a shape already loaded, with different centroids, is a miss
    /// and must still face the fixed-point residual check. Match the wrong
    /// way round and the memo becomes a validation *bypass* — one honest
    /// load of a shape would excuse every tampered codebook of that shape
    /// for the life of the process, reopening #320 through the memo.
    #[test]
    fn a_memoised_shape_does_not_excuse_different_centroids() {
        const BITS: usize = 2;
        const DIM: usize = 32;
        let (boundaries, centroids) = crate::codebook::codebook(BITS, DIM);

        // Seed the memo with the genuine codebook for this shape. The insert
        // is a `try_lock` and may decline while another test holds the memo,
        // so insist until the entry is actually there.
        let mut seeded = false;
        for _ in 0..1000 {
            validate_codebook(BITS, DIM, 1, &boundaries, &centroids)
                .expect("the real codebook for this shape must validate");
            seeded = ACCEPTED_CODEBOOKS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_some_and(|m| m.get(&(BITS, DIM)).is_some_and(|c| c == &centroids));
            if seeded {
                break;
            }
        }
        assert!(seeded, "could not seed the accepted-codebook memo");

        // A tampered codebook for the same shape: every centroid scaled by a
        // positive constant, which keeps the array finite, inside |v| <= 1
        // and strictly increasing, with boundaries recomputed as the exact
        // f32 midpoints. Every structural check therefore passes and the
        // fixed-point residual is the only thing left that can reject it.
        let tampered: Vec<f32> = centroids.iter().map(|c| c * 0.9).collect();
        let tampered_boundaries: Vec<f32> =
            tampered.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
        let err = validate_codebook(BITS, DIM, 1, &tampered_boundaries, &tampered).expect_err(
            "a shape present in the memo with different centroids must still be \
             checked against the Lloyd-Max fixed point, not waved through",
        );
        assert!(
            err.to_string().contains("fixed-point residual"),
            "rejected for the wrong reason: {err}",
        );
    }
}

/// Coverage for [`assert_codes_and_scales_lengths`] on the paths the
/// public integration tests cannot reach: the private core writer (which
/// is deliberately unvalidated, so it can still serialize a header no
/// caller could supply buffers for) and the x86-only fused native
/// writers, which are `pub(crate)`.
#[cfg(test)]
mod codes_scales_validation_tests {
    use super::*;

    /// The v4 header widened `n_vectors` to u64. A ≥2^32-vector index
    /// cannot be built — nor, now, written through the public writers,
    /// whose buffers would have to be 2^32 entries long — so the field
    /// width is pinned at the byte level on the core writer, which is
    /// where the bytes are actually emitted. Must be the exact u64, not
    /// `n mod 2^32` (the pre-v4 silent wrap) and not an error (the
    /// v3-era u32 ceiling).
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn core_header_stores_n_vectors_over_u32_max_exactly() {
        let n = (1usize << 32) + 2;
        let (boundaries, centroids) = crate::codebook::codebook(2, 8);
        let mut buf = Vec::new();
        write_core(&mut buf, 2, 8, n, &[], &boundaries, &centroids, &[], &[], &[]).unwrap();
        // Core layout: bit_width(1) + dim(4) + n_vectors(8).
        let stored = u64::from_le_bytes(buf[5..13].try_into().unwrap());
        assert_eq!(stored, n as u64, "header must store the exact 64-bit count");
    }

    #[cfg(target_arch = "x86_64")]
    fn native_fixture() -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // bw 4, dim 32, n 2 -> 1 block * 32 * (32/2) = 512 bytes.
        let (boundaries, centroids) = crate::codebook::codebook(4, 32);
        (vec![0u8; 512], vec![1.0f32; 2], boundaries, centroids)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[should_panic(expected = "codes_blocked_seq length 504 does not match")]
    fn native_write_rejects_short_codes() {
        let (codes, scales, b, c) = native_fixture();
        let dir = std::env::temp_dir().join(format!("tv_native_bad_{}", std::process::id()));
        let _ = write_native_with_durability(
            dir.join("x.tv"), 4, 32, 2, &codes[..504], &b, &c, &scales, &[], &[],
            Durability::Fast,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[should_panic(expected = "scales length 1 does not match n_vectors 2")]
    fn native_id_map_write_rejects_short_scales() {
        let (codes, scales, b, c) = native_fixture();
        let dir = std::env::temp_dir().join(format!("tvim_native_bad_{}", std::process::id()));
        let _ = write_id_map_native_with_durability(
            dir.join("x.tvim"), 4, 32, 2, &codes, &b, &c, &scales[..1], &[], &[], &[1u64, 2],
            Durability::Fast,
        );
    }
}
