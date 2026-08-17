//! The shared write protocol behind TurboVec index files.
//!
//! Index bytes themselves are v7 and live in `io_v7`; every entry point
//! is a method on the index types ([`crate::TurboQuantIndex::write`],
//! `to_bytes`, `write_to_writer` and their `IdMapIndex` counterparts).
//! What remains here is what those share: the atomic-replace protocol,
//! the stale-temp sweep, [`Durability`], and the value-level bounds a
//! load enforces on a calibration.
//!
//! Older files (v5, v6) are refused, and `turbovec::convert` is the one
//! place that still reads or writes them.
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

use std::fs::File;
use std::io::{self};
#[cfg(test)]
use std::sync::Mutex;
use std::path::{Path, PathBuf};

const TV_MAGIC: &[u8; 4] = b"TVPI";
const TVIM_MAGIC: &[u8; 4] = b"TVIM";

/// How durable a save must be before it reports success.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Durability {
    /// Temp file + `fsync` + atomic rename. The default.
    #[default]
    Durable,
    /// Temp file + atomic rename, no `fsync`.
    Fast,
}











/// The error a pre-v7 file gets on every load entry point.
///
/// v7 is the only format turbovec reads or writes. It names what the
/// file actually is, so "this is an old index" and "this is not an
/// index" are not the same message.
pub(crate) fn legacy_format_error(path: &Path) -> io::Error {
    // A file we cannot open is not a format problem: propagate the real
    // error so a missing path still surfaces as NotFound (#156) rather
    // than as "this is not a turbovec index".
    let mut head = [0u8; 5];
    let opened = match File::open(path) {
        Ok(f) => f,
        Err(e) => return e,
    };
    let version = read_exact_at(&opened, &mut head, 0)
        .ok()
        .and_then(|()| (&head[0..4] == TV_MAGIC || &head[0..4] == TVIM_MAGIC).then_some(head[4]));
    let what = match version {
        Some(v) => format!("a version {v} turbovec index"),
        None => "not a turbovec index".to_string(),
    };
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "{} is {what}; this build reads only the v7 format. Convert it with \
             the turbovec converter, or re-write it from the source vectors.",
            path.display(),
        ),
    )
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
    // Never zero: zero is the "unclaimed" marker a snapshot carries, and
    // a claimed file that happened to draw it would be adopted as a sync
    // destination by a loader that must not adopt it.
    loop {
        let n = ((tmp_rand() as u64) << 32) | tmp_rand() as u64;
        if n != crate::io_v7::UNCLAIMED_NONCE {
            return n;
        }
    }
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








#[cfg(unix)]
pub(crate) fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
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


#[cfg(test)]
mod fork_safety_tests {
    use super::{claim_first_sweep, SWEPT};
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


