//! Read/write TurboVec index files.
//!
//! Every format has two symmetric entry-point pairs: a path-based pair
//! ([`write`]/[`load`], [`write_id_map`]/[`load_id_map`]) that adds
//! atomic-replace semantics on the write side, and a generic pair
//! ([`write_to`]/[`load_from`], [`write_id_map_to`]/[`load_id_map_from`])
//! over any [`std::io::Write`]/[`std::io::Read`] for callers that hold
//! the payload in memory (e.g. a database page or a network buffer).
//! Both pairs produce and accept exactly the same bytes and apply
//! exactly the same validation.
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

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const TV_MAGIC: &[u8; 4] = b"TVPI";
const TV_VERSION: u8 = 6;
const TVIM_MAGIC: &[u8; 4] = b"TVIM";
const TVIM_VERSION: u8 = 6;

/// Recovery hint for any index written before the v5 rotation break
/// (format versions 1 through 4).
const REBUILD_HINT: &str =
    "Rebuild this index from the source vectors using turbovec 0.10.0 or later. \
     turbovec 0.10.0 replaced the index rotation with a deterministic \
     block-Hadamard transform, which changes every encoded byte; there is no \
     in-place migration.";

/// Durability level for path-based writes.
///
/// * [`Durability::Durable`] (the default everywhere): write to a
///   sibling temp file, `fsync` it (`F_FULLFSYNC` on macOS), then
///   atomically rename over the destination. Survives process crashes
///   AND power loss: the destination always holds a complete old or
///   complete new index, and a completed save is on stable storage.
/// * [`Durability::Fast`]: identical temp-file + atomic-rename protocol
///   but no fsync. The destination still can never hold a torn index
///   and a process crash cannot lose the previous file — but a power
///   loss or kernel panic shortly after a "completed" save may lose or
///   truncate the new file. Choose this only when the index is
///   reproducible or durability is handled elsewhere (e.g. the file is
///   about to be uploaded or the filesystem is transient).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Durability {
    #[default]
    Durable,
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
        codes: Vec<u8>,
        boundaries: Vec<f32>,
        centroids: Vec<f32>,
    },
    /// Codes already in the *native* kernel layout for this platform —
    /// produced by the fast path loader, whose extraction pass fuses the
    /// platform transform into the copy. Byte-identical to `BlockedSeq`
    /// on non-x86 (the stored layout is native there).
    BlockedNative {
        codes: Vec<u8>,
        boundaries: Vec<f32>,
        centroids: Vec<f32>,
    },
}

/// Core payload — what a fully-deserialized index needs.
type CoreLoad = (usize, usize, usize, CodePayload, Vec<f32>, Vec<f32>, Vec<f32>);

/// `.tv` write — positional index.
///
/// The write is atomic with respect to the destination: the payload goes
/// to a sibling temp file which is fsynced and then renamed over `path`,
/// so a failed or interrupted write leaves any previous file at `path`
/// intact.
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

/// [`write`] with an explicit [`Durability`] level.
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
    #[cfg(target_arch = "x86_64")]
    {
        return write_atomic_parallel(path.as_ref(), durability, TV_MAGIC, TV_VERSION, |head| {
            head_core(head, bit_width, dim, n_vectors, codebook_boundaries, codebook_centroids)
        }, codes_blocked_seq, |tail| {
            tail_core(tail, scales, tqplus_shift, tqplus_scale);
            Ok(())
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    write_atomic(path.as_ref(), durability, |f| {
        write_to(
            f, bit_width, dim, n_vectors, codes_blocked_seq,
            codebook_boundaries, codebook_centroids, scales,
            tqplus_shift, tqplus_scale,
        )
    })
}

/// `.tv` write to any [`Write`] sink — the in-memory counterpart of
/// [`write`]. Emits exactly the bytes [`write`] would put in the file
/// (magic + version + v5 core payload), so a `Vec<u8>` filled by this
/// function is byte-identical to the corresponding `.tv` file.
///
/// Unlike [`write`] there is no atomicity story: the caller owns the
/// sink.
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
    w.write_all(TV_MAGIC)?;
    w.write_all(&[TV_VERSION])?;
    write_core(
        w, bit_width, dim, n_vectors, codes_blocked_seq,
        codebook_boundaries, codebook_centroids, scales,
        tqplus_shift, tqplus_scale,
    )
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
    if let Some((core, _tail)) = try_load_v6_fast(&f, cap, TV_MAGIC)? {
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
             incompatible with the turbovec 0.10.0 (v5) rotation. Loading it \
             against the v5 rotation would silently return near-zero recall, \
             so it is refused. {REBUILD_HINT}"
        ),
    )
}


/// `.tvim` write — positional index plus the id-map side-tables.
///
/// Atomic with respect to the destination, like [`write`].
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

    write_id_map_with_durability(
        path, bit_width, dim, n_vectors, codes_blocked_seq,
        codebook_boundaries, codebook_centroids, scales,
        tqplus_shift, tqplus_scale, slot_to_id, Durability::Durable,
    )
}

/// [`write_id_map`] with an explicit [`Durability`] level.
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
    #[cfg(target_arch = "x86_64")]
    {
        return write_atomic_parallel(path.as_ref(), durability, TVIM_MAGIC, TVIM_VERSION, |head| {
            head_core(head, bit_width, dim, n_vectors, codebook_boundaries, codebook_centroids)
        }, codes_blocked_seq, |tail| {
            tail_core(tail, scales, tqplus_shift, tqplus_scale);
            for &id in slot_to_id {
                tail.extend_from_slice(&id.to_le_bytes());
            }
            Ok(())
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    write_atomic(path.as_ref(), durability, |f| {
        write_id_map_to(
            f, bit_width, dim, n_vectors, codes_blocked_seq,
            codebook_boundaries, codebook_centroids, scales,
            tqplus_shift, tqplus_scale, slot_to_id,
        )
    })
}

/// `.tvim` write to any [`Write`] sink — the in-memory counterpart of
/// [`write_id_map`]. Emits exactly the bytes [`write_id_map`] would put
/// in the file.
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
    let f = File::open(path)?;
    // See `load` for the allocation-cap rationale.
    let cap = f.metadata()?.len();
    // See `load` — direct-to-destination fast v6 path.
    if let Some((core, tail)) = try_load_v6_fast(&f, cap, TVIM_MAGIC)? {
        let (bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale) = core;
        let mut r = &tail[..];
        let id_bytes = n_vectors
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "id table size overflows usize"))?;
        let raw = read_exact_vec_capped(&mut r, id_bytes, cap)?;
        let slot_to_id: Vec<u64> = raw
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
        return Ok((
            bit_width, dim, n_vectors, codes, scales, tqplus_shift, tqplus_scale, slot_to_id,
        ));
    }
    let buf = read_file_parallel(&f, cap)?;
    load_id_map_from_capped(&mut &buf[..], cap)
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
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a TVIM file: wrong magic",
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

/// TQ+ calibration length invariant shared by [`write`] and
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

/// Process-wide counter distinguishing concurrent saves to the same
/// path from one process: `.tmp.{pid}` alone would interleave two
/// threads' writes into one temp file and rename the corruption into
/// place, defeating the torn-index guarantee.
static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn tmp_sibling(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    name.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    path.with_file_name(name)
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

#[cfg(target_arch = "x86_64")]
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
    let n_levels = 1usize << bit_width;
    assert_eq!(codebook_boundaries.len(), n_levels - 1, "codebook boundaries length");
    assert_eq!(codebook_centroids.len(), n_levels, "codebook centroids length");
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

#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "x86_64")]
/// Atomic path write with the large codes section written by parallel
/// positioned writes (mirror of the load-side fast path): head and tail
/// serialize into small buffers and pwrite at their computed offsets;
/// the codes span is split across scoped threads. Byte-identical output
/// to the streamed writer (same sections, same order on disk), and the
/// durability protocol is unchanged: everything lands in the temp file,
/// fsync, then atomic rename. Small payloads take one serial write.
fn write_atomic_parallel(
    path: &Path,
    durability: Durability,
    magic: &[u8; 4],
    version: u8,
    head_fn: impl FnOnce(&mut Vec<u8>) -> io::Result<()>,
    codes: &[u8],
    tail_fn: impl FnOnce(&mut Vec<u8>) -> io::Result<()>,
) -> io::Result<()> {
    let mut head = Vec::with_capacity(4096);
    head.extend_from_slice(magic);
    head.push(version);
    head_fn(&mut head)?;
    let mut tail = Vec::new();
    tail_fn(&mut tail)?;

    let tmp: PathBuf = tmp_sibling(path);
    let result = (|| {
        let f = File::create(&tmp)?;
        const PAR_MIN: usize = 8 * 1024 * 1024;
        let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(4);
        if codes.len() < PAR_MIN || n_threads < 2 {
            let mut w = BufWriter::new(&f);
            w.write_all(&head)?;
            w.write_all(codes)?;
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
                    s.spawn(|| loop {
                        if failed.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n_chunks {
                            break;
                        }
                        let off = i * chunk;
                        let this = chunk.min(codes.len() - off);
                        if let Err(e) =
                            write_all_at(&f, &codes[off..off + this], base + off as u64)
                        {
                            failed.store(true, std::sync::atomic::Ordering::Relaxed);
                            *err.lock().expect("err lock") = Some(e);
                            break;
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
        std::fs::rename(&tmp, path)?;
        if durability == Durability::Durable {
            sync_parent_dir(path)?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(all(unix, target_arch = "x86_64"))]
fn write_all_at(f: &File, buf: &[u8], off: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.write_all_at(buf, off)
}

#[cfg(all(windows, target_arch = "x86_64"))]
fn write_all_at(f: &File, mut buf: &[u8], mut off: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_write(buf, off)?;
        buf = &buf[n..];
        off += n as u64;
    }
    Ok(())
}

/// Atomically replace `path` with a freshly-written payload: write to a
/// sibling temp file in the same directory, flush + fsync (in `Durable`
/// mode), then rename over the destination (atomic on POSIX). On any
/// failure the previous file at `path` is left untouched and the temp
/// file is removed (best effort), so a reader never observes a partial
/// index. Non-x86 streamed-path counterpart of
/// [`write_atomic_parallel`].
#[cfg(not(target_arch = "x86_64"))]
fn write_atomic(
    path: &Path,
    durability: Durability,
    write_payload: impl FnOnce(&mut BufWriter<&File>) -> io::Result<()>,
) -> io::Result<()> {
    let tmp: PathBuf = tmp_sibling(path);
    let result = (|| {
        let f = File::create(&tmp)?;
        let mut w = BufWriter::new(&f);
        write_payload(&mut w)?;
        w.flush()?;
        drop(w);
        if durability == Durability::Durable {
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        if durability == Durability::Durable {
            sync_parent_dir(path)?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
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
    let n_levels = 1usize << bit_width;
    assert_eq!(codebook_boundaries.len(), n_levels - 1, "codebook boundaries length");
    assert_eq!(codebook_centroids.len(), n_levels, "codebook centroids length");
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
    validate_codebook(n_vectors, &boundaries, &centroids)?;
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

/// Codebook value validation shared by the streamed and fast v6 loaders.
fn validate_codebook(n_vectors: usize, boundaries: &[f32], centroids: &[f32]) -> io::Result<()> {
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

    // Value-level validation, mirroring the header checks: the encoder
    // only ever emits finite shifts and strictly-positive scales
    // (encode.rs initialises scale to 1.0 and overwrites it only with a
    // positive span), so anything else is corruption or an attacker
    // payload. Search divides by `tqplus_scale`, so a zero/negative/
    // non-finite value — which a bare is_finite() check would not fully
    // catch — silently turns every query's scores into NaN/Inf.
    if let Some((i, &v)) = tqplus_shift
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid TQ+ shift at coord {i}: {v} (must be finite)"),
        ));
    }
    if let Some((i, &v)) = tqplus_scale
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || **v <= 0.0)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid TQ+ scale at coord {i}: {v} (must be finite and > 0)"),
        ));
    }

    Ok((tqplus_shift, tqplus_scale))
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
        .find(|(_, s)| !s.is_finite() || **s < 0.0)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid per-vector scale at slot {i}: {s} (must be finite and non-negative)"),
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
    // One even chunk per thread, one positioned read each — fewer,
    // larger syscalls measure faster than a fine-grained work queue on
    // both target platforms.
    let chunk = len_usize.div_ceil(n_threads).max(CHUNK_MIN).next_multiple_of(4096);
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
) -> io::Result<Option<(CoreLoad, Vec<u8>)>> {
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
    validate_codebook(n_vectors, &boundaries, &centroids)?;
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
    // Native transform fused into the read at L2 granularity; identity
    // on non-x86, where the stored layout is already native.
    #[cfg(target_arch = "x86_64")]
    let transform: Option<fn(&mut [u8])> = Some(crate::pack::interleave_chunk_x86);
    #[cfg(not(target_arch = "x86_64"))]
    let transform: Option<fn(&mut [u8])> = None;
    let codes = read_range_parallel_transform(f, codes_start, blocked_bytes as u64, transform)?;
    // Tail: scales + TQ+ (+ id table for .tvim) — small.
    let tail_len = cap_usize - codes_end as usize;
    let mut tail = vec![0u8; tail_len];
    read_exact_at(f, &mut tail, codes_end)?;
    let mut tr: &[u8] = &tail[..];
    let scales = read_scales_validated(&mut tr, n_vectors)?;
    let (tqplus_shift, tqplus_scale) = read_tqplus_trailer(&mut tr, dim)?;
    let rest = tr.to_vec();
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
        rest,
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
