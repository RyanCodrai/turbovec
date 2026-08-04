//! The v8 incremental container: a headerized slot array.
//!
//! The file is the index's memory laid flat — no log, no replay:
//!
//! ```text
//! [superblock]   geometry, codebook, calibration; immutable between
//!                compactions; trailing CRC
//! [header A]     alternating commit slots: generation, n, the partial
//! [header B]     tail block's rows, and the pending redo ops; CRC
//! [block units]  one unit per completed 32-row block: codes in the
//!                sequential-blocked search layout, 32 scales, 32 ids
//!                (id-mapped files), unit CRC
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
//! slot. Appends land past the committed region first (one barrier),
//! then the header commits them (the second); a change confined to the
//! tail block is the header barrier alone.
//!
//! ## Removals: redo ops riding the commit
//!
//! A removal fills its hole *inside* a committed unit — but never
//! during the sync that commits it. The commit header instead carries
//! the change as a redo op: an absolute write ("this slot holds these
//! bytes") plus the unit's expected CRC once every op is applied over
//! its disk bytes. Committing a removal is therefore just the header
//! barrier. A later sync materializes pending ops into their units in
//! its data barrier — safe under the old header because an op-bearing
//! slot's live bytes ARE its committed bytes, and idempotent under any
//! tear because re-applying an absolute write converges. If the unit is
//! dirtied again before materialization, its ops (old and new,
//! coalesced per slot) are carried into the next header instead and the
//! unit is left untouched on disk.
//!
//! Load applies pending ops in memory over each named unit and demands
//! the expected CRC — one pass that repairs any partially-materialized
//! state and detects rot. No state outside the committed header is ever
//! needed for recovery, which is why crash handling needs no undo area,
//! no attempt versioning, and no carried repair sets: a torn sync
//! leaves either the old header (whose ops still describe the old
//! state) or the new one (whose ops describe the new), never a third
//! thing.
//!
//! ## Rot in the newest header
//!
//! Every byte the current state depends on is CRC-covered (superblock,
//! each unit, the header — pending ops included). The newest header
//! slot is the one place rot is indistinguishable from a torn sync, and
//! the stated degradation is: fall back to the other slot — exactly the
//! previous commit — or refuse if it is invalid too. Bytes the current
//! state does *not* depend on (the stale header slot, dead units past
//! `n`) can rot freely: a flip there either changes nothing or is
//! caught by the fallback path's own CRCs.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) const V8_MAGIC: &[u8; 4] = b"TV8\0";
pub(crate) const V8_VERSION: u8 = 1;
const BLOCK: usize = 32;

/// A commit header carries at most this many pending redo ops (and at
/// most this many op-bearing units); beyond it, the sync falls back to
/// a full rewrite. Ops coalesce per slot, so this is 64 distinct
/// dirtied slots in flight, not 64 removals.
pub(crate) const MAX_OPS: usize = 64;

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
    /// One redo op's bytes in a header group: lane, codes, scale, id.
    fn op_size(&self) -> usize {
        1 + self.row_bytes() + 4 + self.id_bytes(1)
    }
    /// One header slot's fixed capacity: gen, n, tail (31 rows), the
    /// pending-op groups, CRC. Writes and parses cover only the used
    /// prefix — every length inside is derivable from `n` and the group
    /// counts, so a small sync writes a small header.
    pub fn hdr_len(&self) -> usize {
        16 + 31 * (self.row_bytes() + 4 + self.id_bytes(1))
            + 4
            + MAX_OPS * (9 + self.op_size())
            + 4
    }
    fn hdr_at(&self, slot: usize) -> usize {
        self.sb_len() + slot * self.hdr_len()
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
/// here until their block completes, and `ops` — pending redo writes,
/// grouped per unit with the unit's expected post-apply CRC — until a
/// later sync materializes them. Only the used prefix is returned
/// (variable length, self-describing, CRC last).
fn header_slot(
    src: &SyncSource<'_>,
    gen: u64,
    n: usize,
    ops: &[(usize, Vec<usize>)],
) -> Vec<u8> {
    let geo = src.geo();
    let n_tail = n % BLOCK;
    let first_tail = n - n_tail;
    let mut h = Vec::with_capacity(geo.hdr_len());
    h.extend_from_slice(&gen.to_le_bytes());
    h.extend_from_slice(&(n as u64).to_le_bytes());
    for k in 0..n_tail {
        let r = first_tail + k;
        h.extend_from_slice(&(src.row_codes)(r));
        h.extend_from_slice(&src.scales[r].to_le_bytes());
        if let Some(ids) = src.ids {
            h.extend_from_slice(&ids[r].to_le_bytes());
        }
    }
    h.extend_from_slice(&(ops.len() as u32).to_le_bytes());
    for (block, slots) in ops {
        h.extend_from_slice(&(*block as u32).to_le_bytes());
        // The unit's expected CRC once every op is applied over its
        // disk bytes: exactly the live unit, because a lane that ever
        // diverged from disk has an op here and every other lane still
        // equals disk.
        let mut u = unit_bytes(src, *block);
        u.truncate(geo.unit_len() - 4);
        h.extend_from_slice(&crc32(&u).to_le_bytes());
        h.push(slots.len() as u8);
        for &s in slots {
            h.push((s % BLOCK) as u8);
            h.extend_from_slice(&(src.row_codes)(s));
            h.extend_from_slice(&src.scales[s].to_le_bytes());
            if let Some(ids) = src.ids {
                h.extend_from_slice(&ids[s].to_le_bytes());
            }
        }
    }
    let c = crc32(&h);
    h.extend_from_slice(&c.to_le_bytes());
    debug_assert!(h.len() <= geo.hdr_len());
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
    /// Slots whose ops ride the new header (still un-materialized after
    /// this sync) — the index's pending set once the plan lands.
    pub carried: Vec<usize>,
}

/// Plan one incremental sync from the committed state `cursor` to the
/// live state in `src`.
///
/// `pending` are slots whose redo ops ride the CURRENT header (declared
/// but not yet materialized); `fresh` are slots dirtied since that
/// commit. Units with only pending ops are materialized in the data
/// batch (their live bytes ARE the committed state, so the write is
/// safe under the old header and idempotent under any tear). Units with
/// fresh dirt are never touched on disk — all their ops, old and new,
/// are carried in the new header instead, and a fallback to the old
/// header still finds those units exactly as its own ops expect.
///
/// `None` when the carried ops exceed [`MAX_OPS`] — the caller writes
/// full instead.
pub(crate) fn plan_incremental(
    src: &SyncSource<'_>,
    cursor: SyncCursor,
    pending: &std::collections::HashSet<usize>,
    fresh: &std::collections::HashSet<usize>,
) -> Option<SyncPlan> {
    let geo = src.geo();
    let old_blocks = (cursor.n_synced as usize) / BLOCK;
    let new_blocks = src.n_vectors / BLOCK;
    let live_blocks = old_blocks.min(new_blocks);
    let gen = cursor.gen + 1;

    // Ops only matter for units committed under BOTH states: below the
    // old floor they exist on disk, below the new floor they stay
    // validated. Popped units go stale harmlessly; a regrow rewrites
    // them whole as appends.
    let live = |s: &&usize| **s < live_blocks * BLOCK;
    let fresh_units: std::collections::HashSet<usize> =
        fresh.iter().filter(live).map(|&s| s / BLOCK).collect();
    let mut materialize: Vec<usize> = pending
        .iter()
        .filter(live)
        .map(|&s| s / BLOCK)
        .filter(|b| !fresh_units.contains(b))
        .collect();
    materialize.sort_unstable();
    materialize.dedup();

    // Carried ops: every dirtied slot (old or new) in a fresh unit,
    // grouped per unit, coalesced per slot.
    let mut carried: Vec<usize> = pending
        .iter()
        .chain(fresh.iter())
        .filter(live)
        .filter(|&&s| fresh_units.contains(&(s / BLOCK)))
        .copied()
        .collect();
    carried.sort_unstable();
    carried.dedup();
    if carried.len() > MAX_OPS {
        return None;
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for &s in &carried {
        match groups.last_mut() {
            Some((b, slots)) if *b == s / BLOCK => slots.push(s),
            _ => groups.push((s / BLOCK, vec![s])),
        }
    }

    let mut batches = Vec::new();
    let mut data = Batch::default();
    for &b in &materialize {
        data.ops.push((geo.unit_at(b) as u64, unit_bytes(src, b)));
    }
    for b in old_blocks..new_blocks {
        data.ops.push((geo.unit_at(b) as u64, unit_bytes(src, b)));
    }
    if !data.ops.is_empty() {
        batches.push(data);
    }

    // The commit: generation g lives in slot g % 2, so this only ever
    // overwrites the slot of generation g - 1.
    let slot = (gen % 2) as usize;
    batches.push(Batch {
        ops: vec![(
            geo.hdr_at(slot) as u64,
            header_slot(src, gen, src.n_vectors, &groups),
        )],
    });

    Some(SyncPlan {
        batches,
        new_cursor: SyncCursor {
            gen,
            n_synced: src.n_vectors as u64,
            calib_gen: cursor.calib_gen,
        },
        carried,
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
    // Slot 0 carries generation 0 (no pending ops); slot 1 starts
    // invalid (zeroed, CRC cannot match).
    let h = header_slot(src, gen, src.n_vectors, &[]);
    image.extend_from_slice(&h);
    image.extend_from_slice(&vec![0u8; geo.hdr_len() - h.len()]);
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
    })
}

// ---------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------

/// Everything a v8 load yields. `seq_blocked` covers `n.div_ceil(32)`
/// blocks with tail lanes written and dead lanes zeroed — the blocked
/// cache's exact bytes (one platform transform away on x86).
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
    /// Slots whose redo ops ride the loaded header (declared but not
    /// yet materialized). They seed the index's pending set so the next
    /// sync materializes or carries them.
    pub pending_slots: Vec<usize>,
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn read_u32(raw: &[u8], at: usize) -> io::Result<u32> {
    raw.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
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
    // A header is variable-length inside its fixed slot: gen, n, the
    // n%32 tail rows, then the pending-op groups, then the CRC. Every
    // interior length is derivable, so the parse walks the used prefix
    // and checks the CRC exactly where the writer put it.
    let tail_row = row_bytes + 4 + geo.id_bytes(1);
    let op_size = geo.op_size();
    /// One pending-op group: block, expected post-apply CRC, and each
    /// op's (global slot, payload offset in the raw file).
    type OpGroup = (usize, u32, Vec<(usize, usize)>);
    struct Hdr {
        gen: u64,
        n: usize,
        tail_at: usize,
        groups: Vec<OpGroup>,
    }
    let parse_hdr = |slot: usize| -> Option<Hdr> {
        let at = geo.hdr_at(slot);
        let bytes = raw.get(at..at + hdr_len)?;
        let gen = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        if (gen % 2) as usize != slot {
            return None;
        }
        let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut p = 16 + (n % BLOCK) * tail_row;
        let n_units = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().unwrap()) as usize;
        if n_units > MAX_OPS {
            return None;
        }
        p += 4;
        let mut groups = Vec::with_capacity(n_units);
        for _ in 0..n_units {
            let b = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().unwrap()) as usize;
            let crc = u32::from_le_bytes(bytes.get(p + 4..p + 8)?.try_into().unwrap());
            let n_ops = *bytes.get(p + 8)? as usize;
            p += 9;
            let mut ops = Vec::with_capacity(n_ops);
            for _ in 0..n_ops {
                let lane = *bytes.get(p)? as usize;
                if lane >= BLOCK {
                    return None;
                }
                ops.push((b * BLOCK + lane, at + p + 1));
                p += op_size;
            }
            groups.push((b, crc, ops));
        }
        let stored = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().unwrap());
        if crc32(&bytes[..p]) != stored {
            return None;
        }
        Some(Hdr {
            gen,
            n,
            tail_at: at + 16,
            groups,
        })
    };
    let chosen = match (parse_hdr(0), parse_hdr(1)) {
        (Some(a), Some(b)) => {
            if a.gen > b.gen {
                a
            } else {
                b
            }
        }
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return Err(bad("no valid commit header — unrecoverable v8 file")),
    };
    let gen = chosen.gen;
    let n_vectors = chosen.n;

    // --- read the units ----------------------------------------------
    // Units named by a pending-op group adopt their disk bytes with the
    // ops applied over them and must hash to the group's expected CRC --
    // repairing any partially-materialized state and detecting rot in
    // the same pass. Every other unit must hash to its own trailing CRC.
    let n_blocks = n_vectors / BLOCK;
    let total_blocks = n_vectors.div_ceil(BLOCK);
    let block_bytes = BLOCK * row_bytes;
    let mut seq_blocked = vec![0u8; total_blocks * block_bytes];
    let mut scales = vec![0f32; n_vectors];
    let mut ids = vec![0u64; if kind == 1 { n_vectors } else { 0 }];
    let op_group: std::collections::HashMap<usize, _> =
        chosen.groups.iter().map(|g| (g.0, g)).collect();
    let mut restored: Vec<u8> = Vec::new();
    for b in 0..n_blocks {
        let at = geo.unit_at(b);
        let unit = raw
            .get(at..at + geo.unit_len())
            .ok_or_else(|| bad("truncated block unit"))?;
        let body_len = geo.unit_len() - 4;
        let body: &[u8] = if let Some((_, expect, ops)) = op_group.get(&b) {
            restored.clear();
            restored.extend_from_slice(&unit[..body_len]);
            for &(slot, payload_at) in ops {
                let lane = slot % BLOCK;
                let op = raw
                    .get(payload_at..payload_at + row_bytes + 4 + geo.id_bytes(1))
                    .ok_or_else(|| bad("truncated pending op"))?;
                for g in 0..row_bytes {
                    restored[g * BLOCK + lane] = op[g];
                }
                let so = block_bytes + lane * 4;
                restored[so..so + 4].copy_from_slice(&op[row_bytes..row_bytes + 4]);
                if kind == 1 {
                    let io_ = block_bytes + BLOCK * 4 + lane * 8;
                    restored[io_..io_ + 8]
                        .copy_from_slice(&op[row_bytes + 4..row_bytes + 12]);
                }
            }
            if crc32(&restored) != *expect {
                return Err(bad(format!(
                    "block {b} does not reach its committed state under the \\
                     header's pending ops (crc mismatch)"
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
        for k in 0..n_tail {
            let r = n_blocks * BLOCK + k;
            let lane = r % BLOCK;
            let row = raw
                .get(chosen.tail_at + k * tail_row..chosen.tail_at + (k + 1) * tail_row)
                .ok_or_else(|| bad("truncated commit tail"))?;
            for g in 0..row_bytes {
                seq_blocked[n_blocks * block_bytes + g * BLOCK + lane] = row[g];
            }
            let v = f32::from_le_bytes(row[row_bytes..row_bytes + 4].try_into().unwrap());
            if !v.is_finite() || v < 0.0 {
                return Err(bad("invalid per-vector scale in the commit tail"));
            }
            scales[r] = v;
            if kind == 1 {
                ids[r] =
                    u64::from_le_bytes(row[row_bytes + 4..row_bytes + 12].try_into().unwrap());
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
        },
        pending_slots: chosen
            .groups
            .iter()
            .flat_map(|(_, _, ops)| ops.iter().map(|&(s, _)| s))
            .collect(),
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
    let tail_row = geo.row_bytes() + 4 + geo.id_bytes(1);
    let op_size = geo.op_size();
    let mut newest: Option<u64> = None;
    for slot in 0..2 {
        let mut bytes = vec![0u8; hdr_len];
        if f.seek(SeekFrom::Start(geo.hdr_at(slot) as u64)).is_err()
            || f.read_exact(&mut bytes).is_err()
        {
            continue;
        }
        // The same variable-length walk the loader does: gen, n, tail,
        // op groups, CRC over the used prefix.
        let gen = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        if (gen % 2) as usize != slot {
            continue;
        }
        let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut p = 16 + (n % 32) * tail_row;
        let Some(nu) = bytes.get(p..p + 4) else {
            continue;
        };
        let n_units = u32::from_le_bytes(nu.try_into().unwrap()) as usize;
        if n_units > MAX_OPS {
            continue;
        }
        p += 4;
        let mut ok = true;
        for _ in 0..n_units {
            let Some(&n_ops) = bytes.get(p + 8) else {
                ok = false;
                break;
            };
            p += 9 + n_ops as usize * op_size;
        }
        if !ok {
            continue;
        }
        let Some(stored) = bytes.get(p..p + 4) else {
            continue;
        };
        if crc32(&bytes[..p]) != u32::from_le_bytes(stored.try_into().unwrap()) {
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
