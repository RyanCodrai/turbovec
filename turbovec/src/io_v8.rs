//! The v8 incremental container: a headerized slot array.
//!
//! The file is the index's memory laid flat — no log, no replay:
//!
//! ```text
//! [superblock]   geometry, codebook, calibration; immutable between
//!                compactions; trailing CRC
//! [undo dirs]    two alternating directory slots naming the in-flight
//!                sync's undo blob, its suspect units, their old CRCs
//! [header A]     alternating commit slots: generation, n, the partial
//! [header B]     tail block's rows, CRC
//! [block units]  one unit per completed 32-row block: codes in the
//!                sequential-blocked search layout, 32 scales, 32 ids
//!                (id-mapped files), unit CRC
//! [undo blob]    transient; only meaningful while a sync is in flight
//! ```
//!
//! Invariant: committed whole blocks = `n / 32`; the `n % 32` tail rows
//! live in the current header slot. Completed blocks are bytes the
//! search cache holds, verbatim.
//!
//! ## Commit protocol
//!
//! A sync flips between the two header slots: generation `g` lives in
//! slot `g % 2`, so writing generation `g+1` only ever overwrites the
//! slot of generation `g-1` — the previous commit is never touched. A
//! torn header write fails its CRC and load falls back to the other
//! slot.
//!
//! Removals fill holes *inside* committed units. That in-place write is
//! made recoverable by an undo step: the sync first writes the affected
//! rows' old bytes as an undo blob past the committed region and a
//! directory naming the blob, the suspect units, and their old CRCs,
//! then fsyncs; only then does it touch the units; only then does it
//! flip the header. Crash between the last two barriers → load sees the
//! old header plus an undo targeting `old_gen + 1`, restores the old
//! rows in memory (read-only repair), and re-verifies each restored
//! unit against its saved old CRC. The undo is dead the moment the new
//! header is durable. Pure-append syncs skip the undo barrier; a change
//! confined to the tail is the header flip alone.
//!
//! Undo attempts are versioned: retries alternate directory slots by
//! attempt parity and place their blob past the last durable one, so a
//! torn retry never destroys the undo that a still-divergent file
//! needs. Load walks candidates newest-first, falling back to any
//! candidate whose blob is intact — or, when a blob is torn, proving
//! from the directory's own CRCs that the in-place phase never ran.
//!
//! ## After a recovery load
//!
//! Disk units covered by the undo may still hold the aborted sync's
//! bytes while the loaded state is the recovered one. The load returns
//! that block list, and the next sync rewrites those units regardless
//! of whether they are dirty again — otherwise a later load would find
//! self-consistent "future" units under an old header and refuse.
//!
//! ## Rot in the newest header
//!
//! Every byte the current state depends on is CRC-covered (superblock,
//! each unit, the header). The newest header slot is the one place rot
//! is indistinguishable from a torn sync, and the stated degradation is
//! the same as ever: fall back to the other slot — exactly the previous
//! commit — or refuse if it is invalid too. Bytes the current state
//! does *not* depend on (the stale header's tail area, dead units past
//! `n`, spent undo bytes) can rot freely: a flip there either changes
//! nothing or is caught by the fallback path's own CRCs.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) const V8_MAGIC: &[u8; 4] = b"TV8\0";
pub(crate) const V8_VERSION: u8 = 1;
const BLOCK: usize = 32;

/// An incremental sync may rewrite at most this many committed units in
/// place (= up to 64 * 32 dirtied rows); beyond it, the sync falls back
/// to a full rewrite. Bounds the undo directory to a fixed slot so the
/// suspect-unit list survives independently of the undo blob.
pub(crate) const MAX_UNDO_UNITS: usize = 64;

// ---------------------------------------------------------------------
// CRC-32C, hardware where available (see io: chosen over CRC-32/ISO
// because aarch64 and x86_64 both carry it in silicon; the integrity
// pass runs at memcpy speed). Large inputs checksum as three
// interleaved thirds — the crc instruction is latency-bound, so three
// independent chains run ~3x faster — and the stored value is the CRC
// of the three digests. The split is length-derived, so writer and
// reader always agree.
// ---------------------------------------------------------------------

static CRC_TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();

pub(crate) fn crc32(data: &[u8]) -> u32 {
    if data.len() < 4096 {
        return crc32_one(data);
    }
    let third = data.len() / 3;
    let (a, rest) = data.split_at(third);
    let (b, c) = rest.split_at(third);
    let (ca, cb, cc) = crc32_three(a, b, c);
    let mut digest = [0u8; 12];
    digest[..4].copy_from_slice(&ca.to_le_bytes());
    digest[4..8].copy_from_slice(&cb.to_le_bytes());
    digest[8..].copy_from_slice(&cc.to_le_bytes());
    crc32_one(&digest)
}

fn crc32_one(data: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        return unsafe { crc32c_hw_aarch64(data) };
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.2") {
        return unsafe { crc32c_hw_x86(data) };
    }
    crc32c_soft(data)
}

fn crc32_three(a: &[u8], b: &[u8], c: &[u8]) -> (u32, u32, u32) {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        return unsafe { crc32c_three_hw_aarch64(a, b, c) };
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.2") {
        return unsafe { crc32c_three_hw_x86(a, b, c) };
    }
    (crc32c_soft(a), crc32c_soft(b), crc32c_soft(c))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_hw_aarch64(data: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};
    let mut crc = 0xFFFF_FFFFu32;
    let (chunks, tail) = data.as_chunks::<8>();
    for c in chunks {
        crc = __crc32cd(crc, u64::from_le_bytes(*c));
    }
    for &b in tail {
        crc = __crc32cb(crc, b);
    }
    !crc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_three_hw_aarch64(a: &[u8], b: &[u8], c: &[u8]) -> (u32, u32, u32) {
    use std::arch::aarch64::{__crc32cb, __crc32cd};
    let n = a.len().min(b.len()).min(c.len()) / 8;
    let (mut x, mut y, mut z) = (0xFFFF_FFFFu32, 0xFFFF_FFFFu32, 0xFFFF_FFFFu32);
    for i in 0..n {
        x = __crc32cd(x, u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap()));
        y = __crc32cd(y, u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()));
        z = __crc32cd(z, u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap()));
    }
    let fin = |mut crc: u32, tail: &[u8]| {
        for &v in tail {
            crc = __crc32cb(crc, v);
        }
        !crc
    };
    (fin(x, &a[n * 8..]), fin(y, &b[n * 8..]), fin(z, &c[n * 8..]))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_hw_x86(data: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};
    let mut crc = 0xFFFF_FFFFu64;
    let (chunks, tail) = data.as_chunks::<8>();
    for c in chunks {
        crc = _mm_crc32_u64(crc, u64::from_le_bytes(*c));
    }
    let mut crc = crc as u32;
    for &b in tail {
        crc = _mm_crc32_u8(crc, b);
    }
    !crc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_three_hw_x86(a: &[u8], b: &[u8], c: &[u8]) -> (u32, u32, u32) {
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};
    let n = a.len().min(b.len()).min(c.len()) / 8;
    let (mut x, mut y, mut z) = (0xFFFF_FFFFu64, 0xFFFF_FFFFu64, 0xFFFF_FFFFu64);
    for i in 0..n {
        x = _mm_crc32_u64(x, u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap()));
        y = _mm_crc32_u64(y, u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()));
        z = _mm_crc32_u64(z, u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap()));
    }
    let fin = |mut crc: u32, tail: &[u8]| {
        for &v in tail {
            crc = _mm_crc32_u8(crc, v);
        }
        !crc
    };
    (
        fin(x as u32, &a[n * 8..]),
        fin(y as u32, &b[n * 8..]),
        fin(z as u32, &c[n * 8..]),
    )
}

fn crc32c_soft(data: &[u8]) -> u32 {
    let table = CRC_TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                let mask = (c & 1).wrapping_neg();
                c = (c >> 1) ^ (0x82F6_3B78 & mask);
            }
            *e = c;
        }
        t
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = (crc >> 8) ^ table[usize::from((crc as u8) ^ b)];
    }
    !crc
}

// ---------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------

/// All offsets in a v8 file derive from these five numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geo {
    pub kind: u8,
    pub dim: usize,
    pub bit_width: usize,
    pub n_calib: usize,
}

impl Geo {
    pub fn row_bytes(&self) -> usize {
        self.dim * self.bit_width / 8
    }
    fn n_levels(&self) -> usize {
        1 << self.bit_width
    }
    fn id_bytes(&self, rows: usize) -> usize {
        if self.kind == 1 {
            rows * 8
        } else {
            0
        }
    }
    pub fn sb_len(&self) -> usize {
        11 + (self.n_levels() - 1) * 4 + self.n_levels() * 4 + 4 + self.n_calib * 8 + 4
    }
    /// Two undo-directory slots, alternating by attempt parity: a torn
    /// directory write can then never destroy the latest durable one.
    fn undo_dir_at(&self, slot: usize) -> usize {
        self.sb_len() + slot * Self::undo_dir_len()
    }
    /// offset, len, target gen, attempt, unit count, MAX_UNDO_UNITS
    /// (block, old-crc) pairs, crc.
    fn undo_dir_len() -> usize {
        8 + 8 + 8 + 8 + 4 + MAX_UNDO_UNITS * 8 + 4
    }
    /// One header slot: gen, n, tail codes (31 rows), tail scales, tail
    /// ids, CRC.
    pub fn hdr_len(&self) -> usize {
        16 + 31 * self.row_bytes() + 31 * 4 + self.id_bytes(31) + 4
    }
    fn hdr_at(&self, slot: usize) -> usize {
        self.undo_dir_at(2) + slot * self.hdr_len()
    }
    /// Test-only view of a header slot's offset (the rot harness needs
    /// the newest header's byte range).
    #[cfg(test)]
    pub fn hdr_at_for_test(&self, slot: usize) -> usize {
        self.hdr_at(slot)
    }
    pub fn unit_len(&self) -> usize {
        BLOCK * self.row_bytes() + BLOCK * 4 + self.id_bytes(BLOCK) + 4
    }
    pub fn unit_at(&self, block: usize) -> usize {
        self.hdr_at(2) + block * self.unit_len()
    }

}

// ---------------------------------------------------------------------
// Cursor, source, dirty capture
// ---------------------------------------------------------------------

/// The in-memory binding between an index and the file it syncs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SyncCursor {
    /// Generation of the commit this index last wrote or loaded.
    pub gen: u64,
    /// Rows in that commit (whole blocks = n_synced / 32).
    pub n_synced: u64,
    /// The index's calibration generation at that commit; a mismatch
    /// forces a compaction, since a refit rewrites every stored code.
    pub calib_gen: u64,
    /// Highest undo attempt already on disk for the in-flight target
    /// generation (0 = none). A retry after a failed or torn sync must
    /// not overwrite the last durable attempt's directory or blob.
    pub undo_attempt: u64,
    /// End of that attempt's blob — where the next attempt's blob may
    /// start.
    pub undo_end: u64,
}

/// Everything a sync needs from the index, borrowed.
pub(crate) struct SyncSource<'a> {
    /// 0 = TurboQuantIndex, 1 = IdMapIndex — stamped in the superblock
    /// so each index type refuses the other's files.
    pub kind: u8,
    pub dim: usize,
    pub bit_width: usize,
    pub n_vectors: usize,
    /// Sequential-blocked codes for whole blocks `[from, to)` (row
    /// indexes, multiples of 32).
    pub seq_blocks: &'a dyn Fn(usize, usize) -> Vec<u8>,
    /// One row's sequential codes (tail rows).
    pub row_codes: &'a dyn Fn(usize) -> Vec<u8>,
    pub scales: &'a [f32],
    /// slot → external id; `Some` iff kind 1.
    pub ids: Option<&'a [u64]>,
    pub tqplus_shift: &'a [f32],
    pub tqplus_scale: &'a [f32],
    pub boundaries: &'a [f32],
    pub centroids: &'a [f32],
}

impl SyncSource<'_> {
    fn geo(&self) -> Geo {
        Geo {
            kind: self.kind,
            dim: self.dim,
            bit_width: self.bit_width,
            n_calib: self.tqplus_shift.len(),
        }
    }
}

/// A committed slot's on-disk content, captured the moment the live
/// index made it diverge (removal hole-fill, pop, moved-from slot). The
/// undo blob is built from these.
#[derive(Clone, Debug)]
pub(crate) struct DirtyOld {
    pub codes: Vec<u8>,
    pub scale: f32,
    /// Old external id at the slot; 0 and unused for kind 0.
    pub id: u64,
}

// ---------------------------------------------------------------------
// Serialization pieces
// ---------------------------------------------------------------------

fn superblock(src: &SyncSource<'_>) -> Vec<u8> {
    let mut sb = Vec::new();
    sb.extend_from_slice(V8_MAGIC);
    sb.push(V8_VERSION);
    sb.push(src.bit_width as u8);
    sb.push(src.kind);
    sb.extend_from_slice(&(src.dim as u32).to_le_bytes());
    for v in src.boundaries {
        sb.extend_from_slice(&v.to_le_bytes());
    }
    for v in src.centroids {
        sb.extend_from_slice(&v.to_le_bytes());
    }
    sb.extend_from_slice(&(src.tqplus_shift.len() as u32).to_le_bytes());
    for v in src.tqplus_shift {
        sb.extend_from_slice(&v.to_le_bytes());
    }
    for v in src.tqplus_scale {
        sb.extend_from_slice(&v.to_le_bytes());
    }
    let c = crc32(&sb);
    sb.extend_from_slice(&c.to_le_bytes());
    debug_assert_eq!(sb.len(), src.geo().sb_len());
    sb
}

/// One header slot's bytes for commit (`gen`, `n`): the tail rows ride
/// here until their block completes.
fn header_slot(src: &SyncSource<'_>, gen: u64, n: usize) -> Vec<u8> {
    let geo = src.geo();
    let row_bytes = geo.row_bytes();
    let n_tail = n % BLOCK;
    let first_tail = n - n_tail;
    let mut h = Vec::with_capacity(geo.hdr_len());
    h.extend_from_slice(&gen.to_le_bytes());
    h.extend_from_slice(&(n as u64).to_le_bytes());
    let mut tail_codes = vec![0u8; 31 * row_bytes];
    let mut tail_scales = vec![0u8; 31 * 4];
    let mut tail_ids = vec![0u8; geo.id_bytes(31)];
    for k in 0..n_tail {
        let r = first_tail + k;
        tail_codes[k * row_bytes..(k + 1) * row_bytes].copy_from_slice(&(src.row_codes)(r));
        tail_scales[k * 4..k * 4 + 4].copy_from_slice(&src.scales[r].to_le_bytes());
        if let Some(ids) = src.ids {
            tail_ids[k * 8..k * 8 + 8].copy_from_slice(&ids[r].to_le_bytes());
        }
    }
    h.extend_from_slice(&tail_codes);
    h.extend_from_slice(&tail_scales);
    h.extend_from_slice(&tail_ids);
    let c = crc32(&h);
    h.extend_from_slice(&c.to_le_bytes());
    debug_assert_eq!(h.len(), geo.hdr_len());
    h
}

/// One block unit's bytes: seq-blocked codes, scales, ids, CRC.
fn unit_bytes(src: &SyncSource<'_>, block: usize) -> Vec<u8> {
    let geo = src.geo();
    let from = block * BLOCK;
    let mut u = (src.seq_blocks)(from, from + BLOCK);
    debug_assert_eq!(u.len(), BLOCK * geo.row_bytes());
    for lane in 0..BLOCK {
        let r = from + lane;
        let v = if r < src.n_vectors { src.scales[r] } else { 0.0 };
        u.extend_from_slice(&v.to_le_bytes());
    }
    if let Some(ids) = src.ids {
        for lane in 0..BLOCK {
            let r = from + lane;
            let v = if r < src.n_vectors { ids[r] } else { 0 };
            u.extend_from_slice(&v.to_le_bytes());
        }
    }
    let c = crc32(&u);
    u.extend_from_slice(&c.to_le_bytes());
    debug_assert_eq!(u.len(), geo.unit_len());
    u
}

// ---------------------------------------------------------------------
// Write batches: the sync plan is data, so the torn-write harness can
// tear it at every byte of every op in every barrier.
// ---------------------------------------------------------------------

/// Positioned writes that must all be durable before the next batch
/// starts. Production applies a batch with pwrite + fsync; the harness
/// applies arbitrary prefixes of it.
#[derive(Debug, Default)]
pub(crate) struct Batch {
    pub ops: Vec<(u64, Vec<u8>)>,
}

/// A planned sync: barrier-separated batches plus the cursor that
/// becomes current once the last batch lands.
pub(crate) struct SyncPlan {
    pub batches: Vec<Batch>,
    pub new_cursor: SyncCursor,
    /// The undo attempt this plan writes (cursor bookkeeping for a
    /// failed run: the attempt and blob extent are on disk even if the
    /// commit never landed).
    pub undo_attempt: u64,
    pub undo_end: u64,
}

/// Plan one incremental sync from the committed state `cursor` to the
/// live state in `src`. `dirty` maps committed slots to their on-disk
/// content; `force_units` are blocks that must be rewritten even if
/// clean (the undo-covered set after a recovery load). `None` when the
/// sync would touch more than [`MAX_UNDO_UNITS`] committed units — the
/// caller writes full instead.
pub(crate) fn plan_incremental(
    src: &SyncSource<'_>,
    cursor: SyncCursor,
    dirty: &std::collections::HashMap<usize, DirtyOld>,
    force_units: &[usize],
) -> Option<SyncPlan> {
    let geo = src.geo();
    let row_bytes = geo.row_bytes();
    let old_blocks = (cursor.n_synced as usize) / BLOCK;
    let new_blocks = src.n_vectors / BLOCK;
    let live_blocks = old_blocks.min(new_blocks);
    let gen = cursor.gen + 1;

    // Units to rewrite in place: any committed-and-still-live block
    // holding a dirtied slot, plus the forced set.
    let mut touched: Vec<usize> = dirty
        .keys()
        .filter(|&&s| s < live_blocks * BLOCK)
        .map(|&s| s / BLOCK)
        .chain(force_units.iter().copied().filter(|&b| b < live_blocks))
        .collect();
    touched.sort_unstable();
    touched.dedup();

    if touched.len() > MAX_UNDO_UNITS {
        return None;
    }
    let attempt = cursor.undo_attempt + 1;
    let mut undo_end = cursor.undo_end;
    let mut batches = Vec::new();

    // Barrier 1: the undo blob (old row payloads) and the undo
    // directory (target generation, suspect units, their old CRCs) —
    // only when committed bytes are about to change. The directory is
    // the authority on WHICH units an interrupted sync may have
    // touched; the blob is only the bytes to put back. Recovery can
    // therefore distinguish "in-place phase never ran" (every suspect
    // unit still hashes to its old CRC) from "rows lost" (refuse)
    // without trusting the blob.
    if !touched.is_empty() {
        let rows: Vec<(&usize, &DirtyOld)> = {
            let mut v: Vec<_> = dirty
                .iter()
                .filter(|(&s, _)| s < live_blocks * BLOCK)
                .collect();
            v.sort_by_key(|(&s, _)| s);
            v
        };
        // The committed body of every touched unit: the live unit with
        // the dirty rows' old bytes put back — identical to what the
        // disk held at `cursor`'s commit (for forced units the live
        // state IS the committed one; their disk bytes are the aborted
        // sync's).
        let committed_body = |b: usize| -> Vec<u8> {
            let mut u = unit_bytes(src, b);
            u.truncate(geo.unit_len() - 4);
            for (&s, old) in &rows {
                if s / BLOCK == b {
                    let lane = s % BLOCK;
                    for g in 0..row_bytes {
                        u[g * BLOCK + lane] = old.codes[g];
                    }
                    let so = BLOCK * row_bytes + lane * 4;
                    u[so..so + 4].copy_from_slice(&old.scale.to_le_bytes());
                    if geo.kind == 1 {
                        let io_ = BLOCK * row_bytes + BLOCK * 4 + lane * 8;
                        u[io_..io_ + 8].copy_from_slice(&old.id.to_le_bytes());
                    }
                }
            }
            u
        };
        // Forced units (carried out of a recovery load) may differ from
        // the commit in ANY lane — their whole committed body rides the
        // blob, where a dirty unit only needs its old rows.
        let mut forced_sorted: Vec<usize> = force_units
            .iter()
            .copied()
            .filter(|&b| b < live_blocks)
            .collect();
        forced_sorted.sort_unstable();
        forced_sorted.dedup();
        let mut blob = Vec::new();
        blob.extend_from_slice(&gen.to_le_bytes());
        blob.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        for (&s, old) in &rows {
            blob.extend_from_slice(&(s as u64).to_le_bytes());
            blob.extend_from_slice(&old.codes);
            blob.extend_from_slice(&old.scale.to_le_bytes());
            if geo.kind == 1 {
                blob.extend_from_slice(&old.id.to_le_bytes());
            }
        }
        blob.extend_from_slice(&(forced_sorted.len() as u32).to_le_bytes());
        for &b in &forced_sorted {
            blob.extend_from_slice(&(b as u32).to_le_bytes());
            blob.extend_from_slice(&committed_body(b));
        }
        let c = crc32(&blob);
        blob.extend_from_slice(&c.to_le_bytes());

        // Past the committed regions AND past the previous attempt's
        // blob: a retry after a failed or interrupted sync must leave
        // the last durable undo intact until this one is durable.
        let undo_off = (geo.unit_at(old_blocks.max(new_blocks)) as u64).max(cursor.undo_end);
        let mut dir = Vec::with_capacity(Geo::undo_dir_len());
        dir.extend_from_slice(&undo_off.to_le_bytes());
        dir.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        dir.extend_from_slice(&gen.to_le_bytes());
        dir.extend_from_slice(&attempt.to_le_bytes());
        dir.extend_from_slice(&(touched.len() as u32).to_le_bytes());
        for &b in &touched {
            dir.extend_from_slice(&(b as u32).to_le_bytes());
            // The unit's *old* CRC: what the restored unit must hash to.
            dir.extend_from_slice(&crc32(&committed_body(b)).to_le_bytes());
        }
        dir.resize(Geo::undo_dir_len() - 4, 0);
        let c = crc32(&dir);
        dir.extend_from_slice(&c.to_le_bytes());
        undo_end = undo_off + blob.len() as u64;
        batches.push(Batch {
            ops: vec![
                (undo_off, blob),
                (geo.undo_dir_at((attempt % 2) as usize) as u64, dir),
            ],
        });
    }

    // Barrier 2: in-place unit rewrites and appended units.
    let mut data = Batch::default();
    for &b in &touched {
        data.ops.push((geo.unit_at(b) as u64, unit_bytes(src, b)));
    }
    for b in old_blocks..new_blocks {
        data.ops.push((geo.unit_at(b) as u64, unit_bytes(src, b)));
    }
    if !data.ops.is_empty() {
        batches.push(data);
    }

    // Barrier 3: the header flip. Generation g lives in slot g % 2, so
    // this only ever overwrites the slot of generation g - 1.
    let slot = (gen % 2) as usize;
    batches.push(Batch {
        ops: vec![(
            geo.hdr_at(slot) as u64,
            header_slot(src, gen, src.n_vectors),
        )],
    });

    Some(SyncPlan {
        batches,
        new_cursor: SyncCursor {
            gen,
            n_synced: src.n_vectors as u64,
            calib_gen: cursor.calib_gen,
            // A durable commit retires every undo targeting it.
            undo_attempt: 0,
            undo_end: 0,
        },
        undo_attempt: attempt,
        undo_end,
    })
}

fn fsync_if(f: &File, durable: bool) -> io::Result<()> {
    if durable {
        // `sync_all`, not `sync_data`: durability parity with
        // `write(durable=True)` on every platform — macOS F_FULLFSYNC
        // included — and syncs change the file length.
        f.sync_all()?;
    }
    Ok(())
}

/// Apply a planned sync to the file: each batch's ops, then a barrier.
pub(crate) fn run_sync(path: &Path, plan: &SyncPlan, durable: bool) -> io::Result<SyncCursor> {
    let mut f = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
    for batch in &plan.batches {
        for (off, bytes) in &batch.ops {
            f.seek(SeekFrom::Start(*off))?;
            f.write_all(bytes)?;
        }
        f.flush()?;
        fsync_if(&f, durable)?;
    }
    Ok(plan.new_cursor)
}

// ---------------------------------------------------------------------
// Full write (creation and compaction)
// ---------------------------------------------------------------------

/// The whole file as one byte image for generation `gen` — creation and
/// compaction both go through this, via a temp sibling + atomic rename
/// (the v6 writer's own machinery: naming, Windows retry, stale-temp
/// sweep, parent-dir fsync posture).
pub(crate) fn write_full(
    path: &Path,
    src: &SyncSource<'_>,
    calib_gen: u64,
    durable: bool,
) -> io::Result<SyncCursor> {
    let geo = src.geo();
    let gen = 0u64;
    let n_blocks = src.n_vectors / BLOCK;
    let mut image = superblock(src);
    // Undo directory: no undo in flight; the all-zero slot is
    // CRC-invalid on purpose, so recovery treats it as absent.
    image.extend_from_slice(&vec![0u8; 2 * Geo::undo_dir_len()]);
    // Slot 0 carries generation 0; slot 1 starts invalid (zeroed).
    image.extend_from_slice(&header_slot(src, gen, src.n_vectors));
    image.extend_from_slice(&vec![0u8; geo.hdr_len()]);
    for b in 0..n_blocks {
        image.extend_from_slice(&unit_bytes(src, b));
    }

    crate::io::sweep_stale_tmps(path);
    let (mut f, tmp) = crate::io::create_tmp(path)?;
    let result = (|| {
        f.write_all(&image)?;
        fsync_if(&f, durable)
    })();
    let result = result.and_then(|()| {
        drop(f);
        crate::io::rename_atomic(&tmp, path)
    });
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if durable {
        crate::io::sync_parent_dir_after_commit(path);
    }
    Ok(SyncCursor {
        gen,
        n_synced: src.n_vectors as u64,
        calib_gen,
        undo_attempt: 0,
        undo_end: 0,
    })
}

// ---------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------

/// Everything a v8 load yields. `seq_blocked` covers `n.div_ceil(32)`
/// blocks with tail lanes written and dead lanes zeroed — the blocked
/// cache's exact bytes (one platform transform away on x86).
type UndoPayload = (Vec<(usize, DirtyOld)>, Vec<(usize, Vec<u8>)>);

pub(crate) struct V8Load {
    pub dim: usize,
    pub bit_width: usize,
    pub n_vectors: usize,
    pub seq_blocked: Vec<u8>,
    pub scales: Vec<f32>,
    /// External ids (kind 1); empty for kind 0.
    pub ids: Vec<u64>,
    pub tqplus_shift: Vec<f32>,
    pub tqplus_scale: Vec<f32>,
    pub cursor: SyncCursor,
    /// Blocks whose on-disk bytes may belong to an aborted sync (the
    /// undo-covered set applied during a recovery load). The next sync
    /// must rewrite these units even if nothing dirties them again.
    pub recovered_units: Vec<usize>,
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn read_u32(raw: &[u8], at: usize) -> io::Result<u32> {
    raw.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| bad("unexpected end of file"))
}

fn read_u64(raw: &[u8], at: usize) -> io::Result<u64> {
    raw.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| bad("unexpected end of file"))
}

fn read_f32(raw: &[u8], at: usize) -> io::Result<f32> {
    raw.get(at..at + 4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| bad("unexpected end of file"))
}

pub(crate) fn load(path: &Path, expect_calib_gen: u64, expect_kind: u8) -> io::Result<V8Load> {
    let raw = std::fs::read(path)?;
    if raw.len() < 11 || &raw[..4] != V8_MAGIC {
        return Err(bad("not a v8 file"));
    }
    if raw[4] != V8_VERSION {
        return Err(bad(format!("unsupported v8 revision {}", raw[4])));
    }
    let bit_width = raw[5] as usize;
    if !(2..=4).contains(&bit_width) {
        return Err(bad(format!("bit_width {bit_width} out of range")));
    }
    let kind = raw[6];
    if kind != expect_kind {
        return Err(bad(match kind {
            1 => "this v8 file holds an IdMapIndex; load it with IdMapIndex::load".to_string(),
            0 => {
                "this v8 file holds a TurboQuantIndex; load it with TurboQuantIndex::load"
                    .to_string()
            }
            k => format!("unknown v8 index kind {k}"),
        }));
    }
    let dim = read_u32(&raw, 7)? as usize;
    if dim == 0 || !dim.is_multiple_of(8) || dim > crate::MAX_DIM {
        return Err(bad(format!("dim {dim} invalid")));
    }

    // Codebook must match the canonical one, same as the v6 loader
    // (#320): a drifted codebook silently mis-scores.
    let (canon_b, canon_c) = crate::codebook::codebook(bit_width, dim);
    let mut off = 11;
    for want in canon_b.iter().chain(canon_c.iter()) {
        if read_f32(&raw, off)? != *want {
            return Err(bad("embedded codebook drifted from the canonical one"));
        }
        off += 4;
    }
    let n_calib = read_u32(&raw, off)? as usize;
    off += 4;
    if n_calib != 0 && n_calib != dim {
        return Err(bad(format!("calibration length {n_calib} != dim {dim}")));
    }
    let mut tqplus_shift = Vec::with_capacity(n_calib);
    let mut tqplus_scale = Vec::with_capacity(n_calib);
    for k in 0..n_calib {
        tqplus_shift.push(read_f32(&raw, off + k * 4)?);
    }
    off += n_calib * 4;
    for k in 0..n_calib {
        tqplus_scale.push(read_f32(&raw, off + k * 4)?);
    }
    off += n_calib * 4;
    let stored = read_u32(&raw, off)?;
    if crc32(&raw[..off]) != stored {
        return Err(bad("corrupt superblock (crc mismatch)"));
    }

    let geo = Geo {
        kind,
        dim,
        bit_width,
        n_calib,
    };
    let row_bytes = geo.row_bytes();
    let hdr_len = geo.hdr_len();

    // --- pick the newest valid header --------------------------------
    let hdr = |slot: usize| -> Option<(u64, usize)> {
        let at = geo.hdr_at(slot);
        let bytes = raw.get(at..at + hdr_len)?;
        let stored = u32::from_le_bytes(bytes[hdr_len - 4..].try_into().unwrap());
        if crc32(&bytes[..hdr_len - 4]) != stored {
            return None;
        }
        let gen = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        // Generation g must live in slot g % 2 — a valid-looking header
        // in the wrong slot is not ours.
        if (gen % 2) as usize != slot {
            return None;
        }
        let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        Some((gen, n))
    };
    let (gen, n_vectors) = match (hdr(0), hdr(1)) {
        (Some(a), Some(b)) => {
            if a.0 > b.0 {
                a
            } else {
                b
            }
        }
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return Err(bad("no valid commit header — unrecoverable v8 file")),
    };

    // --- undo directories: meaningful only if they target gen + 1 -----
    // (a sync toward gen + 1 was interrupted after an undo barrier).
    // Attempts alternate between the two slots, so the latest durable
    // attempt survives a torn write of the next one. Walk candidates
    // newest-first: use the first whose blob is intact; a candidate
    // whose suspect units all still hash to their old CRCs proves the
    // in-place phase never ran and the load is clean. If every
    // candidate fails both tests, rows are lost — refuse. A mixed
    // state is never served.
    struct UndoCand {
        attempt: u64,
        units: Vec<(usize, u32)>,
        rows: Vec<(usize, DirtyOld)>,
        whole: Vec<(usize, Vec<u8>)>,
        blob_ok: bool,
        blob_end: u64,
    }
    let mut cands: Vec<UndoCand> = Vec::new();
    for slot in 0..2 {
        let dir_at = geo.undo_dir_at(slot);
        let Some(dir) = raw.get(dir_at..dir_at + Geo::undo_dir_len()) else {
            continue;
        };
        let dl = Geo::undo_dir_len();
        let stored = u32::from_le_bytes(dir[dl - 4..].try_into().unwrap());
        if crc32(&dir[..dl - 4]) != stored {
            continue;
        }
        let target = u64::from_le_bytes(dir[16..24].try_into().unwrap());
        if target != gen + 1 {
            continue;
        }
        let attempt = u64::from_le_bytes(dir[24..32].try_into().unwrap());
        if (attempt % 2) as usize != slot {
            continue;
        }
        let n_units = u32::from_le_bytes(dir[32..36].try_into().unwrap()) as usize;
        if n_units > MAX_UNDO_UNITS {
            continue;
        }
        let mut units = Vec::with_capacity(n_units);
        for k in 0..n_units {
            let at = 36 + k * 8;
            units.push((
                u32::from_le_bytes(dir[at..at + 4].try_into().unwrap()) as usize,
                u32::from_le_bytes(dir[at + 4..at + 8].try_into().unwrap()),
            ));
        }
        let u_off = u64::from_le_bytes(dir[..8].try_into().unwrap()) as usize;
        let u_len = u64::from_le_bytes(dir[8..16].try_into().unwrap()) as usize;
        let mut cand = UndoCand {
            attempt,
            units,
            rows: Vec::new(),
            whole: Vec::new(),
            blob_ok: false,
            blob_end: (u_off + u_len) as u64,
        };
        let blob = u_off
            .checked_add(u_len)
            .and_then(|e| raw.get(u_off..e))
            .filter(|b| {
                b.len() >= 16
                    && crc32(&b[..b.len() - 4])
                        == u32::from_le_bytes(b[b.len() - 4..].try_into().unwrap())
                    && u64::from_le_bytes(b[..8].try_into().unwrap()) == gen + 1
            });
        if let Some(blob) = blob {
            let parse = || -> io::Result<UndoPayload> {
                let mut rows = Vec::new();
                let mut whole = Vec::new();
                let n_rows = read_u32(blob, 8)? as usize;
                let mut p = 12;
                for _ in 0..n_rows {
                    let slot = read_u64(blob, p)? as usize;
                    p += 8;
                    let codes = blob
                        .get(p..p + row_bytes)
                        .ok_or_else(|| bad("truncated undo row"))?
                        .to_vec();
                    p += row_bytes;
                    let scale = read_f32(blob, p)?;
                    p += 4;
                    let id = if kind == 1 {
                        let v = read_u64(blob, p)?;
                        p += 8;
                        v
                    } else {
                        0
                    };
                    rows.push((slot, DirtyOld { codes, scale, id }));
                }
                let n_whole = read_u32(blob, p)? as usize;
                p += 4;
                let body_len = geo.unit_len() - 4;
                for _ in 0..n_whole {
                    let b = read_u32(blob, p)? as usize;
                    p += 4;
                    let body = blob
                        .get(p..p + body_len)
                        .ok_or_else(|| bad("truncated whole-unit undo"))?
                        .to_vec();
                    p += body_len;
                    whole.push((b, body));
                }
                Ok((rows, whole))
            };
            if let Ok((rows, whole)) = parse() {
                cand.rows = rows;
                cand.whole = whole;
                cand.blob_ok = true;
            }
        }
        cands.push(cand);
    }
    cands.sort_by_key(|c| std::cmp::Reverse(c.attempt));
    // Pessimistic bookkeeping for the next sync, independent of which
    // candidate wins: never place a new blob below any surviving one.
    let undo_attempt = cands.first().map_or(0, |c| c.attempt);
    let undo_blob_end = cands.iter().map(|c| c.blob_end).max().unwrap_or(0);
    // Pick the recovery source.
    let unit_body_at = |b: usize| {
        let at = geo.unit_at(b);
        raw.get(at..at + geo.unit_len() - 4)
    };
    let mut undo_units: Vec<(usize, u32)> = Vec::new();
    let mut undo_rows: Vec<(usize, DirtyOld)> = Vec::new();
    let mut undo_whole: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut undo_rows_lost = false;
    let mut resolved = cands.is_empty();
    for cand in cands {
        if cand.blob_ok {
            undo_units = cand.units;
            undo_rows = cand.rows;
            undo_whole = cand.whole;
            resolved = true;
            break;
        }
        // Blob torn: clean iff every suspect unit still hashes to its
        // committed CRC (the in-place phase never started).
        let n_blocks_now = n_vectors / BLOCK;
        let untouched = cand.units.iter().all(|&(b, old_crc)| {
            b >= n_blocks_now
                || unit_body_at(b).is_some_and(|body| crc32(body) == old_crc)
        });
        if untouched {
            resolved = true;
            break;
        }
    }
    if !resolved {
        // Every candidate's blob is gone and some suspect unit has
        // diverged: the committed bytes are unrecoverable.
        undo_rows_lost = true;
    }

    // --- read the units ----------------------------------------------
    let n_blocks = n_vectors / BLOCK;
    let total_blocks = n_vectors.div_ceil(BLOCK);
    let block_bytes = BLOCK * row_bytes;
    let mut seq_blocked = vec![0u8; total_blocks * block_bytes];
    let mut scales = vec![0f32; n_vectors];
    let mut ids = vec![0u64; if kind == 1 { n_vectors } else { 0 }];
    let undo_for: std::collections::HashMap<usize, u32> = undo_units.iter().copied().collect();
    if undo_rows_lost && n_blocks > 0 {
        return Err(bad(
            "interrupted sync cannot be recovered: committed blocks diverged \
             and every undo attempt's rows are lost",
        ));
    }
    let mut restored: Vec<u8> = Vec::new();
    for b in 0..n_blocks {
        let at = geo.unit_at(b);
        let unit = raw
            .get(at..at + geo.unit_len())
            .ok_or_else(|| bad("truncated block unit"))?;
        let body_len = geo.unit_len() - 4;
        // Common path: verify and adopt the unit's bytes in place — no
        // copy. Only an undo restore materializes a scratch body.
        let body: &[u8] = if let Some(&old_crc) = undo_for.get(&b) {
            restored.clear();
            restored.extend_from_slice(&unit[..body_len]);
            let body = &mut restored;
            // Whole-unit restores first (units divergent since an
            // earlier interrupted sync), then row restores.
            if let Some((_, w)) = undo_whole.iter().find(|(wb, _)| *wb == b) {
                body.copy_from_slice(w);
            }
            // This unit may hold an aborted sync's bytes (its own CRC
            // may even be valid). Restore the undo rows and verify
            // against the *old* CRC — the state the fallen-back header
            // describes.
            for (slot, old) in undo_rows.iter().filter(|(s, _)| s / BLOCK == b) {
                let lane = slot % BLOCK;
                for g in 0..row_bytes {
                    body[g * BLOCK + lane] = old.codes[g];
                }
                let so = block_bytes + lane * 4;
                body[so..so + 4].copy_from_slice(&old.scale.to_le_bytes());
                if kind == 1 {
                    let io_ = block_bytes + BLOCK * 4 + lane * 8;
                    body[io_..io_ + 8].copy_from_slice(&old.id.to_le_bytes());
                }
            }
            if crc32(body) != old_crc {
                return Err(bad(format!(
                    "block {b} does not restore to its committed state (crc mismatch)"
                )));
            }
            &restored
        } else {
            let stored = u32::from_le_bytes(unit[body_len..].try_into().unwrap());
            let body = &unit[..body_len];
            if crc32(body) != stored {
                return Err(bad(format!("corrupt committed block {b} (crc mismatch)")));
            }
            body
        };
        seq_blocked[b * block_bytes..(b + 1) * block_bytes].copy_from_slice(&body[..block_bytes]);
        for lane in 0..BLOCK {
            let v = f32::from_le_bytes(
                body[block_bytes + lane * 4..block_bytes + lane * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            if !v.is_finite() || v < 0.0 {
                return Err(bad(format!("invalid per-vector scale in block {b}")));
            }
            scales[b * BLOCK + lane] = v;
        }
        if kind == 1 {
            for lane in 0..BLOCK {
                let at = block_bytes + BLOCK * 4 + lane * 8;
                ids[b * BLOCK + lane] =
                    u64::from_le_bytes(body[at..at + 8].try_into().unwrap());
            }
        }
    }

    // --- tail rows from the chosen header -----------------------------
    let n_tail = n_vectors % BLOCK;
    if n_tail > 0 {
        let at = geo.hdr_at((gen % 2) as usize);
        let h = &raw[at..at + hdr_len];
        for k in 0..n_tail {
            let r = n_blocks * BLOCK + k;
            let lane = r % BLOCK;
            let cb = &h[16 + k * row_bytes..16 + (k + 1) * row_bytes];
            for g in 0..row_bytes {
                seq_blocked[n_blocks * block_bytes + g * BLOCK + lane] = cb[g];
            }
            let so = 16 + 31 * row_bytes + k * 4;
            let v = f32::from_le_bytes(h[so..so + 4].try_into().unwrap());
            if !v.is_finite() || v < 0.0 {
                return Err(bad("invalid per-vector scale in the commit tail"));
            }
            scales[r] = v;
            if kind == 1 {
                let io_ = 16 + 31 * row_bytes + 31 * 4 + k * 8;
                ids[r] = u64::from_le_bytes(h[io_..io_ + 8].try_into().unwrap());
            }
        }
    }

    Ok(V8Load {
        dim,
        bit_width,
        n_vectors,
        seq_blocked,
        scales,
        ids,
        tqplus_shift,
        tqplus_scale,
        cursor: SyncCursor {
            gen,
            n_synced: n_vectors as u64,
            calib_gen: expect_calib_gen,
            undo_attempt,
            undo_end: undo_blob_end,
        },
        recovered_units: undo_units.into_iter().map(|(b, _)| b).collect(),
    })
}

// ---------------------------------------------------------------------
// Cursor-vs-file identity
// ---------------------------------------------------------------------

/// What a cursor finds when checked against the file it points at.
pub(crate) enum CursorState {
    /// The file's newest commit is the cursor's: sync in place safely.
    Intact,
    /// A valid v8 file whose newest commit is not the cursor's —
    /// another writer advanced or replaced it. Touching it would
    /// silently destroy their commits: refuse.
    Foreign,
    /// Gone or not v8 (a v6 write(), an empty file): nothing another
    /// v8 writer committed is at stake — write full.
    Replaced,
}

/// Verify `cursor` against the file before a sync touches a byte.
pub(crate) fn cursor_state(
    path: &Path,
    cursor: &SyncCursor,
    geo: &Geo,
) -> io::Result<CursorState> {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(CursorState::Replaced),
        Err(e) => return Err(e),
    };
    let mut magic = [0u8; 5];
    match f.read_exact(&mut magic) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Ok(CursorState::Replaced)
        }
        Err(e) => return Err(e),
    }
    if &magic[..4] != V8_MAGIC || magic[4] != V8_VERSION {
        return Ok(CursorState::Replaced);
    }
    let hdr_len = geo.hdr_len();
    let mut newest: Option<u64> = None;
    for slot in 0..2 {
        let mut bytes = vec![0u8; hdr_len];
        if f.seek(SeekFrom::Start(geo.hdr_at(slot) as u64)).is_err()
            || f.read_exact(&mut bytes).is_err()
        {
            continue;
        }
        let stored = u32::from_le_bytes(bytes[hdr_len - 4..].try_into().unwrap());
        if crc32(&bytes[..hdr_len - 4]) != stored {
            continue;
        }
        let gen = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        if (gen % 2) as usize != slot {
            continue;
        }
        newest = Some(newest.map_or(gen, |g: u64| g.max(gen)));
    }
    match newest {
        Some(g) if g == cursor.gen => Ok(CursorState::Intact),
        Some(_) => Ok(CursorState::Foreign),
        // A v8 magic with no valid header is a corrupt file; a full
        // rewrite is the only safe way to sync onto it.
        None => Ok(CursorState::Replaced),
    }
}

/// Sniff: is this a v8 file?
pub(crate) fn is_v8(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|_| &magic == V8_MAGIC)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three-way interleaved CRC must agree with an independent
    /// composition over the soft single-chain implementation — pinned
    /// on a buffer large enough to take the split path, so a broken
    /// `crc32_three` cannot hide behind small test files.
    #[test]
    fn interleaved_crc_matches_the_reference_composition() {
        let mut data = vec![0u8; 100_000];
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for b in data.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (s >> 56) as u8;
        }
        let third = data.len() / 3;
        let (a, rest) = data.split_at(third);
        let (b, c) = rest.split_at(third);
        let mut digest = [0u8; 12];
        digest[..4].copy_from_slice(&crc32c_soft(a).to_le_bytes());
        digest[4..8].copy_from_slice(&crc32c_soft(b).to_le_bytes());
        digest[8..].copy_from_slice(&crc32c_soft(c).to_le_bytes());
        assert_eq!(crc32(&data), crc32c_soft(&digest));
        // And it discriminates: any flip changes the value.
        let reference = crc32(&data);
        data[50_000] ^= 1;
        assert_ne!(crc32(&data), reference);
    }
}
