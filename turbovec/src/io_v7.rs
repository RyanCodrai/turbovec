//! The v7 incremental container: a headerized slot array.
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
//!                (id-mapped files) — no per-unit checksum
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
//! A generation is not unique over the life of a file. When load falls
//! back, the rejected header stays in its slot and the recovered index's
//! next sync writes the same generation into that same slot — where a
//! torn header write would leave the rejected commit standing, its delta
//! verifying against data the new sync wrote identically. That one sync
//! therefore opens with a repair barrier: the rejected header's bytes
//! are destroyed, and fsynced, before any data moves. It is the only
//! sync that ever runs two barriers.
//!
//! Every other sync is ONE write batch and ONE fsync. No ordering barrier
//! separates data from commit: the header carries a delta descriptor —
//! the units this sync wrote (materialized list + appended range) and
//! the CRC of their bytes — so a commit that reaches disk before its
//! data is *detectable* rather than prevented. Load adopts the newest
//! header whose delta verifies; a reordered or torn persistence fails
//! the check and the other slot wins. (This is the journal-checksum
//! trick that lets ext4 drop its commit barriers, applied to a
//! two-slot header.) Every sync is durable: the single fsync makes
//! everything stable before sync returns — there is no fast mode. The
//! delta check additionally refuses any commit whose data never
//! landed, so even OS-reordered persistence during a power cut
//! recovers to the newest complete commit.
//!
//! ## Removals: redo ops riding the commit
//!
//! A removal fills its hole *inside* a committed unit — but never
//! during the sync that commits it. The commit header instead carries
//! the change as a redo op: an absolute write ("this slot holds these
//! bytes"). Committing a removal is therefore just the header
//! barrier. A later sync materializes pending ops into their units in
//! its data barrier — safe under the old header because an op-bearing
//! slot's live bytes ARE its committed bytes, and idempotent under any
//! tear because re-applying an absolute write converges. If the unit is
//! dirtied again before materialization, its ops (old and new,
//! coalesced per slot) are carried into the next header instead and the
//! unit is left untouched on disk.
//!
//! Load applies pending ops in memory over each named unit — recovery
//! converges because the ops are absolute writes, so no state outside
//! the committed header is ever needed: no undo area, no attempt
//! versioning, no carried repair sets. A torn sync leaves either the
//! old header (whose ops still describe the old state) or the new one
//! (whose ops describe the new), never a third thing.
//!
//! ## What is checksummed, and why
//!
//! Only the commit headers and the superblock carry CRCs, because
//! validity IS the commit mechanism: a torn header write must be
//! recognizably invalid for the A/B fallback to work, and a torn
//! compaction is caught by the superblock the same way. Blocks carry
//! no checksums at all — the delta digest covers every block at the
//! sync that writes it, which is all the crash protocol needs, and
//! detecting later external damage (bit rot, bad copies) is out of
//! scope exactly as it is for v6. One stated consequence: a damaged
//! block named by the NEWEST commit's delta doesn't error — the digest
//! cannot distinguish rot from a torn sync, so the load silently falls
//! back to the previous commit.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) const V7_MAGIC: &[u8; 4] = b"TV7\0";
/// v7 revision.
///
/// Bumped to 2 when snapshots gained [`UNCLAIMED_NONCE`]: revision 1
/// gave the nonce field one meaning (this file's identity) and revision
/// 2 gives zero a second (nobody is syncing to this file). A revision-1
/// reader would bind a cursor to a zero-nonce snapshot and could then
/// patch an unrelated snapshot that also reads zero — exactly the
/// foreign-writer case the nonce exists to catch. The version byte is
/// what stops it from trying.
pub(crate) const V7_VERSION: u8 = 2;
const BLOCK: usize = 32;

/// A commit header carries at most this many pending redo ops (and at
/// most this many op-bearing units); past it, the sync falls back to a
/// full rewrite. Ops coalesce per slot. Sized so the fallback is
/// effectively unreachable (1024 scattered removals between two syncs)
/// at the cost of reserved header space only — each sync writes just
/// its used prefix, so an idle cap is free per sync.
///
/// FORMAT CONSTANT: this value sizes the header slots and therefore
/// every offset in the file. It is stamped into the superblock and
/// validated at load, so a binary with a different value refuses old
/// files cleanly instead of parsing garbage — but changing it still
/// means new files are unreadable by old binaries: bump V7_VERSION.
pub(crate) const MAX_OPS: usize = 1024;

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
    !crc32c_update(0xFFFF_FFFF, data)
}

/// Fold `data` into a running CRC-32C register, returning the raw state
/// (no final complement) so a value can be built from bytes that never
/// exist contiguously — see [`DeltaDigest`].
///
/// CRC-32C is byte-serial, so folding eight bytes with one instruction is
/// identical to folding them one at a time; a chunk boundary anywhere in
/// the stream therefore costs a short scalar tail, never a different
/// answer.
fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        return unsafe { crc32c_upd_hw_aarch64(crc, data) };
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.2") {
        return unsafe { crc32c_upd_hw_x86(crc, data) };
    }
    crc32c_upd_soft(crc, data)
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
unsafe fn crc32c_upd_hw_aarch64(mut crc: u32, data: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};
    let (chunks, tail) = data.as_chunks::<8>();
    for c in chunks {
        crc = __crc32cd(crc, u64::from_le_bytes(*c));
    }
    for &b in tail {
        crc = __crc32cb(crc, b);
    }
    crc
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
unsafe fn crc32c_upd_hw_x86(crc: u32, data: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};
    let mut wide = crc as u64;
    let (chunks, tail) = data.as_chunks::<8>();
    for c in chunks {
        wide = _mm_crc32_u64(wide, u64::from_le_bytes(*c));
    }
    let mut crc = wide as u32;
    for &b in tail {
        crc = _mm_crc32_u8(crc, b);
    }
    crc
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
    !crc32c_upd_soft(0xFFFF_FFFF, data)
}

fn crc32c_upd_soft(mut crc: u32, data: &[u8]) -> u32 {
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
    for &b in data {
        crc = (crc >> 8) ^ table[usize::from((crc as u8) ^ b)];
    }
    crc
}

/// The delta digest, computed over bytes that are never contiguous.
///
/// The digest is [`crc32`] of `gen || (block index || unit body)*`. That
/// buffer used to be materialized — a second copy of everything a sync
/// wrote, on the way into every sync and out of every load, on top of
/// the file image `load` already holds. This accumulates the identical
/// value from the pieces where they already live.
///
/// Identical means bit-for-bit: [`crc32`] splits an input of 4096 bytes
/// or more into three length-derived thirds, checksums each, and hashes
/// the three results. The total length is known before the first byte
/// arrives (every unit is `unit_len`), so the same three boundaries are
/// reproduced here and each pushed chunk is routed to the third — or
/// thirds — it falls in.
struct DeltaDigest {
    /// Bytes pushed so far: the cursor into the virtual buffer.
    at: usize,
    /// The two split points, or `None` when the input is short enough
    /// for `crc32`'s single-chain path.
    split: Option<(usize, usize)>,
    /// Running raw CRC state per third (only `[0]` is used when
    /// `split` is `None`).
    parts: [u32; 3],
}

impl DeltaDigest {
    fn new(total: usize) -> Self {
        let split = (total >= 4096).then(|| {
            let third = total / 3;
            (third, third * 2)
        });
        Self {
            at: 0,
            split,
            parts: [0xFFFF_FFFF; 3],
        }
    }

    fn push(&mut self, mut data: &[u8]) {
        let Some((s1, s2)) = self.split else {
            self.parts[0] = crc32c_update(self.parts[0], data);
            self.at += data.len();
            return;
        };
        // Route each run of bytes to the third it lands in, splitting at
        // a boundary the run straddles.
        while !data.is_empty() {
            let (part, end) = match self.at {
                a if a < s1 => (0, s1),
                a if a < s2 => (1, s2),
                _ => (2, usize::MAX),
            };
            let take = data.len().min(end - self.at);
            self.parts[part] = crc32c_update(self.parts[part], &data[..take]);
            self.at += take;
            data = &data[take..];
        }
    }

    fn finish(self) -> u32 {
        if self.split.is_none() {
            return !self.parts[0];
        }
        let mut digest = [0u8; 12];
        for (i, p) in self.parts.iter().enumerate() {
            digest[i * 4..i * 4 + 4].copy_from_slice(&(!p).to_le_bytes());
        }
        crc32_one(&digest)
    }
}

// ---------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------

/// All offsets in a v7 file derive from these five numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geo {
    pub kind: u8,
    pub dim: usize,
    pub bit_width: usize,
    pub n_calib: usize,
}

impl Geo {
    /// Bytes per row in the sequential-blocked layout, delegated to
    /// [`crate::pack::blocked_geometry`] — the ONE authority on this
    /// stride. (Restating the formula here is how the 3-bit bug
    /// happened: `dim * bits / 8` disagrees with the real stride for
    /// 3-bit codes, which occupy 4-bit fields.)
    pub fn row_bytes(&self) -> usize {
        let (_, n_byte_groups, _) = crate::pack::blocked_geometry(1, self.bit_width, self.dim);
        n_byte_groups
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
        23 + (self.n_levels() - 1) * 4 + self.n_levels() * 4 + 4 + self.n_calib * 8 + 4
    }
    /// One redo op's bytes in a header group: lane, codes, scale, id.
    fn op_size(&self) -> usize {
        1 + self.row_bytes() + 4 + self.id_bytes(1)
    }
    /// Bytes of a header slot that a commit carrying no pending redo ops
    /// can possibly use: the fixed fields, a full tail block, the delta
    /// descriptor, and the CRC — everything except the op-group region.
    ///
    /// A slot is *sized* for the 1024-op cap (~428 KB at dim 768), but the
    /// steady state — an append, or any sync following one that
    /// materialized its ops — carries none, and then this is the whole
    /// used prefix. Reading this much and widening only when the parse
    /// needs more turns a ~856 KB read on the way into every sync into a
    /// ~16 KB one.
    pub fn hdr_probe_len(&self) -> usize {
        16 + 31 * (self.row_bytes() + 4 + self.id_bytes(1))
            + 4
            + 4
            + MAX_OPS * 4
            + 12
            + 4
    }
    /// One header slot's fixed capacity: gen, n, tail (31 rows), the
    /// pending-op groups, CRC. Writes and parses cover only the used
    /// prefix — every length inside is derivable from `n` and the group
    /// counts, so a small sync writes a small header.
    pub fn hdr_len(&self) -> usize {
        16 + 31 * (self.row_bytes() + 4 + self.id_bytes(1))
            + 4
            + MAX_OPS * (5 + self.op_size())
            + 4
            + MAX_OPS * 4
            + 12
            + 4
    }
    fn hdr_at(&self, slot: usize) -> usize {
        self.sb_len() + slot * self.hdr_len()
    }
    /// [`Self::hdr_at`] for the length probe on the load path.
    fn hdr_at_for_len(&self, slot: usize) -> usize {
        self.hdr_at(slot)
    }
    /// Test-only view of a header slot's offset (the rot harness needs
    /// the newest header's byte range).
    #[cfg(test)]
    pub fn hdr_at_for_test(&self, slot: usize) -> usize {
        self.hdr_at(slot)
    }
    /// Test-only view of a unit's offset (the corruption matrix bounds
    /// its structural sweep at the first unit).
    #[cfg(test)]
    pub fn unit_at_for_test(&self, block: usize) -> usize {
        self.unit_at(block)
    }
    /// One block unit: codes, scales, ids. No trailing checksum — the
    /// commit's delta digest covers every unit at the sync that writes
    /// it, and detecting later external damage is out of scope (as it
    /// is for v6).
    pub fn unit_len(&self) -> usize {
        BLOCK * self.row_bytes() + BLOCK * 4 + self.id_bytes(BLOCK)
    }
    pub fn unit_at(&self, block: usize) -> usize {
        self.hdr_at(2) + block * self.unit_len()
    }
    /// Bytes one full image of `n_vectors` rows occupies: superblock,
    /// both header slots at their fixed reserve, then whole blocks.
    pub fn full_len(&self, n_vectors: usize) -> usize {
        self.unit_at(n_vectors / BLOCK)
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
    /// The file's identity, minted at every full write and stamped in
    /// the superblock. Generations alone cannot identify a file — two
    /// independent writers both start at generation 0 — so the cursor
    /// is only Intact against the file it actually wrote.
    pub nonce: u64,
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
    /// Append one row's sequential codes to the buffer (tail rows
    /// and redo ops).
    pub row_codes: &'a dyn Fn(usize, &mut Vec<u8>),
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

fn superblock(src: &SyncSource<'_>, nonce: u64) -> Vec<u8> {
    let mut sb = Vec::new();
    sb.extend_from_slice(V7_MAGIC);
    sb.push(V7_VERSION);
    sb.push(src.bit_width as u8);
    sb.push(src.kind);
    sb.extend_from_slice(&(src.dim as u32).to_le_bytes());
    sb.extend_from_slice(&nonce.to_le_bytes());
    sb.extend_from_slice(&(MAX_OPS as u32).to_le_bytes());
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
    delta: (&[usize], std::ops::Range<usize>, u32),
) -> Vec<u8> {
    let geo = src.geo();
    let n_tail = n % BLOCK;
    let first_tail = n - n_tail;
    let mut h = Vec::with_capacity(geo.hdr_len());
    h.extend_from_slice(&gen.to_le_bytes());
    h.extend_from_slice(&(n as u64).to_le_bytes());
    for k in 0..n_tail {
        let r = first_tail + k;
        (src.row_codes)(r, &mut h);
        h.extend_from_slice(&src.scales[r].to_le_bytes());
        if let Some(ids) = src.ids {
            h.extend_from_slice(&ids[r].to_le_bytes());
        }
    }
    h.extend_from_slice(&(ops.len() as u32).to_le_bytes());
    for (block, slots) in ops {
        h.extend_from_slice(&(*block as u32).to_le_bytes());
        h.push(slots.len() as u8);
        for &s in slots {
            h.push((s % BLOCK) as u8);
            (src.row_codes)(s, &mut h);
            h.extend_from_slice(&src.scales[s].to_le_bytes());
            if let Some(ids) = src.ids {
                h.extend_from_slice(&ids[s].to_le_bytes());
            }
        }
    }
    // The delta descriptor: which units this sync wrote (materialized
    // list + appended range) and the CRC of those units' bytes. A
    // commit that persisted before its data is thereby DETECTABLE, so
    // the sync needs no ordering barrier — one fsync commits it.
    let (mat, app, delta_crc) = delta;
    h.extend_from_slice(&(mat.len() as u32).to_le_bytes());
    for &b in mat {
        h.extend_from_slice(&(b as u32).to_le_bytes());
    }
    h.extend_from_slice(&(app.start as u32).to_le_bytes());
    h.extend_from_slice(&(app.end as u32).to_le_bytes());
    h.extend_from_slice(&delta_crc.to_le_bytes());
    let c = crc32(&h);
    h.extend_from_slice(&c.to_le_bytes());
    debug_assert!(h.len() <= geo.hdr_len());
    h
}

/// One block unit's bytes: seq-blocked codes, scales, ids, CRC.
fn unit_bytes(src: &SyncSource<'_>, block: usize) -> Vec<u8> {
    let geo = src.geo();
    let from = block * BLOCK;
    // Sized for the whole unit up front. `seq_blocks` hands back a
    // Vec allocated to exactly its codes, so appending the scales and
    // ids below used to grow it — and amortized growth doubles, leaving
    // every unit in `batch.ops` holding ~2x its bytes for the life of
    // the sync (#483).
    let codes = (src.seq_blocks)(from, from + BLOCK);
    debug_assert_eq!(codes.len(), BLOCK * geo.row_bytes());
    let mut u = Vec::with_capacity(geo.unit_len());
    u.extend_from_slice(&codes);
    drop(codes);
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
/// `clear_target` is `Some(used)` when the slot this sync commits into
/// already holds a parseable header at this generation or newer — the
/// post-rollback state described at the barrier below — and says how
/// many of its bytes the plan must destroy first.
///
/// `None` when the carried ops exceed [`MAX_OPS`] — the caller falls
/// back to a full rewrite, the only crash-safe way to land more
/// first-time changes than one header can describe.
pub(crate) fn plan_incremental(
    src: &SyncSource<'_>,
    cursor: SyncCursor,
    pending: &std::collections::HashSet<usize>,
    fresh: &std::collections::HashSet<usize>,
    clear_target: Option<usize>,
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
        // The header cannot commit this many first-time changes, and
        // overwriting their blocks directly in the same sync is NOT an
        // option: no committed header would describe the new bytes, so
        // a torn batch would silently corrupt the fallback commit
        // (proven by a torn-write probe). The full rewrite (temp +
        // atomic rename) is correct from any state.
        return None;
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for &s in &carried {
        match groups.last_mut() {
            Some((b, slots)) if *b == s / BLOCK => slots.push(s),
            _ => groups.push((s / BLOCK, vec![s])),
        }
    }

    // One batch, one fsync: the header's delta descriptor names every
    // unit this sync writes and their bytes' CRC, so a commit that
    // reaches disk before its data is detectable at load — no ordering
    // barrier needed. The digest folds each unit's bytes as they are
    // built, so the payload exists once (in the write op) rather than
    // twice.
    let n_written = materialize.len() + new_blocks.saturating_sub(old_blocks);
    let mut digest = DeltaDigest::new(8 + n_written * (4 + geo.unit_len()));
    digest.push(&gen.to_le_bytes());
    let mut batch = Batch::default();
    for b in materialize.iter().copied().chain(old_blocks..new_blocks) {
        let bytes = unit_bytes(src, b);
        digest.push(&(b as u32).to_le_bytes());
        digest.push(&bytes);
        batch.ops.push((geo.unit_at(b) as u64, bytes));
    }
    let delta_crc = digest.finish();

    // The commit: generation g lives in slot g % 2, so this only ever
    // overwrites the slot of generation g - 1.
    let slot = (gen % 2) as usize;
    batch.ops.push((
        geo.hdr_at(slot) as u64,
        header_slot(
            src,
            gen,
            src.n_vectors,
            &groups,
            (&materialize, old_blocks..new_blocks, delta_crc),
        ),
    ));

    // A generation number is not unique over the life of a file. When a
    // crash makes `load` fall back, the rejected header stays in its
    // slot and the recovered index syncs again at the SAME generation,
    // into that same slot. If this sync's data lands and its header
    // write does not, the rejected header is still there — and its
    // delta, which names units this sync has just rewritten with the
    // same bytes (materialization is an idempotent absolute write),
    // verifies. Load would adopt a commit that was already abandoned.
    //
    // So when the caller saw such a header, break it first, behind its
    // own barrier. Writing a generation whose parity disagrees with the
    // slot is the first thing `parse_header_at` rejects, so one 8-byte
    // write is enough and nothing rests on a CRC failing by luck. A
    // crash during this batch is harmless: no data has moved yet, so the
    // stale header's delta still fails exactly as it did at the load
    // that rejected it.
    //
    // Nothing happens in the steady state — a sync writing generation g
    // finds g - 2 in its slot.
    let batches = if let Some(used) = clear_target {
        // Breaking only the generation field is not enough: the header
        // write that follows restores those very bytes first, so a tear
        // one byte in would leave the new generation sitting on the old
        // header's intact body — the rejected commit, reassembled. Wipe
        // everything the old header occupies, so no prefix of the new
        // write can complete it.
        let mut bytes = vec![0u8; used.clamp(8, geo.hdr_len())];
        // And make the wiped slot's generation disagree with the slot's
        // parity, which `parse_header_at` rejects before reading a
        // further byte — so a clear that completes is refused outright
        // rather than on a CRC that merely ought not to match.
        bytes[..8].copy_from_slice(&((gen % 2) ^ 1).to_le_bytes());
        vec![Batch { ops: vec![(geo.hdr_at(slot) as u64, bytes)] }, batch]
    } else {
        vec![batch]
    };

    Some(SyncPlan {
        batches,
        new_cursor: SyncCursor {
            gen,
            n_synced: src.n_vectors as u64,
            calib_gen: cursor.calib_gen,
            nonce: cursor.nonce,
        },
        carried,
    })
}

/// The delta digest: generation, then each written unit's block index
/// and its body WITHOUT the trailing per-unit CRC. Hashing whole unit
/// codewords would be vacuous — `crc32c(m || crc32c(m))` is a fixed
/// residue, so a concatenation of self-consistent units hashes to a
/// content-independent constant. Excluding the codeword CRCs and mixing
/// in the indices and the generation makes the digest depend on every
/// byte and every position it commits.
fn delta_digest(gen: u64, units: &[(usize, &[u8])]) -> u32 {
    let total = units
        .iter()
        .fold(8usize, |n, (_, body)| n.saturating_add(4 + body.len()));
    let mut d = DeltaDigest::new(total);
    d.push(&gen.to_le_bytes());
    for (b, body) in units {
        d.push(&(*b as u32).to_le_bytes());
        d.push(body);
    }
    d.finish()
}

/// Every sync is durable: `sync_all`, not `sync_data` — on every
/// platform (macOS F_FULLFSYNC included), and syncs change the file
/// length, which data-only variants may not persist.
/// THE delta check, shared by the loader and `cursor_state` so
/// "adoptable commit" means exactly one thing: fetch each unit the
/// commit's sync wrote through `read_unit` (false = unavailable) and
/// compare the reconstructed digest.
/// `feed_unit` pushes one unit's bytes into the digest, or answers
/// `false` if that unit is not there — nothing is accumulated, so a
/// commit that appended gigabytes costs one unit of working memory to
/// check, not a second copy of the delta.
fn delta_verified(
    h: &ParsedHdr,
    unit_len: usize,
    mut feed_unit: impl FnMut(usize, &mut DeltaDigest) -> bool,
) -> bool {
    let count = h.delta_mat.len().saturating_add(h.delta_app.len());
    let mut d = DeltaDigest::new(8usize.saturating_add(count.saturating_mul(4 + unit_len)));
    d.push(&h.gen.to_le_bytes());
    for b in h.delta_mat.iter().copied().chain(h.delta_app.clone()) {
        d.push(&(b as u32).to_le_bytes());
        if !feed_unit(b, &mut d) {
            return false;
        }
    }
    d.finish() == h.delta_crc
}

fn fsync_commit(f: &File) -> io::Result<()> {
    f.sync_all()
}

/// Apply a planned sync to the file: each batch's ops, then a barrier.
pub(crate) fn run_sync(path: &Path, plan: &SyncPlan) -> io::Result<SyncCursor> {
    let mut f = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
    for batch in &plan.batches {
        for (off, bytes) in &batch.ops {
            f.seek(SeekFrom::Start(*off))?;
            f.write_all(bytes)?;
        }
        f.flush()?;
        fsync_commit(&f)?;
    }
    Ok(plan.new_cursor)
}

// ---------------------------------------------------------------------
// Full write (creation and compaction)
// ---------------------------------------------------------------------

/// The whole file for generation `gen` — creation and compaction both
/// go through this, via a temp sibling + atomic rename (the v6 writer's
/// own machinery: naming, Windows retry, stale-temp sweep, parent-dir
/// fsync posture). Streamed unit by unit through a buffered writer, so
/// peak extra memory is one write buffer plus one unit — never a second
/// image of the index.
/// Stream one full v7 image into `w`: superblock, both header slots at
/// their fixed reserve, then whole block units.
///
/// Shared by the file writer and by the in-memory form, so a `.tv` file
/// and the bytes `to_bytes` hands back are the same image byte for byte.
fn write_image<W: Write>(w: &mut W, src: &SyncSource<'_>, gen: u64, nonce: u64) -> io::Result<()> {
    let geo = src.geo();
    let n_blocks = src.n_vectors / BLOCK;
    // Slot 0 carries generation 0 (no pending ops); slot 1 starts
    // invalid (zeroed, CRC cannot match).
    let h = header_slot(
        src,
        gen,
        src.n_vectors,
        &[],
        (&[], 0..0, delta_digest(gen, &[])),
    );
    w.write_all(&superblock(src, nonce))?;
    w.write_all(&h)?;
    w.write_all(&vec![0u8; geo.hdr_len() - h.len()])?;
    w.write_all(&vec![0u8; geo.hdr_len()])?;
    for b in 0..n_blocks {
        w.write_all(&unit_bytes(src, b))?;
    }
    Ok(())
}

/// The nonce of an **unclaimed** image: one no index is syncing to.
///
/// A nonce answers one question for `sync`: "is the file at this path
/// still the one I last committed to, or did somebody replace it?"
/// Generations cannot answer it — every full write starts at 0 — so a
/// random value identifies the file's lineage.
///
/// That question only arises for a file some index is *syncing*. A
/// snapshot — `write`, `write_to_writer`, `to_bytes` — is not a sync
/// destination, so it is stamped `UNCLAIMED` and its bytes are a pure
/// function of index state: two images of equal indexes are identical,
/// and `to_bytes` matches the file `write` produces byte for byte.
///
/// `sync` claims a file by full-writing it with a random nonce, and
/// [`load`] refuses to bind a cursor to an unclaimed file — so the first
/// sync to a snapshot full-writes and claims it, exactly as it already
/// does for any path it is not bound to. That keeps every foreign-writer
/// check intact: no file that is being synced ever carries this value.
pub(crate) const UNCLAIMED_NONCE: u64 = 0;

/// One full v7 image as bytes — the same image [`write_full`] lands on
/// disk, built in memory instead.
/// Stream one unclaimed image into `w` — the snapshot form, for a sink
/// the caller owns. Unit-by-unit, so a large index does not need a
/// second copy of itself in memory just to reach a writer.
pub(crate) fn stream_image<W: Write>(w: &mut W, src: &SyncSource<'_>) -> io::Result<()> {
    write_image(w, src, 0, UNCLAIMED_NONCE)
}

pub(crate) fn image_bytes(src: &SyncSource<'_>) -> Vec<u8> {
    let geo = src.geo();
    let mut buf = Vec::with_capacity(geo.full_len(src.n_vectors));
    write_image(&mut buf, src, 0, UNCLAIMED_NONCE).expect("writing to a Vec<u8> cannot fail");
    buf
}

/// Bytes one full v7 image occupies, without building it.
pub(crate) fn image_len(src: &SyncSource<'_>) -> usize {
    src.geo().full_len(src.n_vectors)
}

/// The whole file for generation 0, staged through a temp and renamed:
/// creation and compaction both go through this, and it is the fallback
/// whenever an incremental plan is not the right shape. Claims the file
/// with a fresh nonce — see [`UNCLAIMED_NONCE`] for what that means and
/// for the snapshot form that does not.
pub(crate) fn write_full(
    path: &Path,
    src: &SyncSource<'_>,
    calib_gen: u64,
) -> io::Result<SyncCursor> {
    write_full_with_durability(path, src, calib_gen, crate::io::Durability::Durable)
}

/// [`write_full`] with the fsync made optional.
///
/// `Fast` keeps the temp-file + atomic-rename protocol — the destination
/// can never hold a torn image and a previous file survives a crash —
/// and skips only the fsync, so a power loss shortly after a completed
/// write may lose the new file. Same trade `write_with_durability` has
/// always offered.
pub(crate) fn write_full_with_durability(
    path: &Path,
    src: &SyncSource<'_>,
    calib_gen: u64,
    durability: crate::io::Durability,
) -> io::Result<SyncCursor> {
    write_full_inner(path, src, calib_gen, durability, crate::io::file_nonce())
}

/// [`write_full_with_durability`] leaving the file unclaimed — the
/// snapshot `write` produces, byte-identical to [`image_bytes`].
pub(crate) fn write_snapshot(
    path: &Path,
    src: &SyncSource<'_>,
    durability: crate::io::Durability,
) -> io::Result<()> {
    write_full_inner(path, src, 0, durability, UNCLAIMED_NONCE).map(|_| ())
}

fn write_full_inner(
    path: &Path,
    src: &SyncSource<'_>,
    calib_gen: u64,
    durability: crate::io::Durability,
    nonce: u64,
) -> io::Result<SyncCursor> {
    let gen = 0u64;

    crate::io::sweep_stale_tmps(path);
    let (f, tmp) = crate::io::create_tmp(path)?;
    let result = (|| {
        let mut w = std::io::BufWriter::with_capacity(1 << 20, &f);
        write_image(&mut w, src, gen, nonce)?;
        w.flush()?;
        drop(w);
        match durability {
            crate::io::Durability::Durable => fsync_commit(&f),
            crate::io::Durability::Fast => Ok(()),
        }
    })();
    let result = result.and_then(|()| {
        drop(f);
        crate::io::rename_atomic(&tmp, path)
    });
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    // Only when the caller asked for durability: the parent-dir fsync is
    // part of what `Fast` exists to skip, and on a network or slow
    // filesystem it is the expensive half. The v6 writer guarded it the
    // same way.
    if durability == crate::io::Durability::Durable {
        crate::io::sync_parent_dir_after_commit(path);
    }
    Ok(SyncCursor {
        gen,
        n_synced: src.n_vectors as u64,
        calib_gen,
        nonce,
    })
}

// ---------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------

/// Everything a v7 load yields. `seq_blocked` covers `n.div_ceil(32)`
/// blocks with tail lanes written and dead lanes zeroed — the blocked
/// cache's exact bytes (one platform transform away on x86).
pub(crate) struct V7Load {
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


fn read_u64_at(raw: &[u8], at: usize) -> io::Result<u64> {
    raw.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| bad("unexpected end of file"))
}

fn read_f32(raw: &[u8], at: usize) -> io::Result<f32> {
    raw.get(at..at + 4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| bad("unexpected end of file"))
}

/// One pending-op group: block, and each op's (global slot, payload
/// offset in the file).
type OpGroup = (usize, Vec<(usize, usize)>);

struct ParsedHdr {
    gen: u64,
    n: usize,
    tail_at: usize,
    groups: Vec<OpGroup>,
    /// Units this commit's sync wrote (materialized + appended range)
    /// and the digest of their bytes — checked before the commit is
    /// adopted, which is what lets a sync run on a single fsync.
    delta_mat: Vec<usize>,
    delta_app: std::ops::Range<usize>,
    delta_crc: u32,
    /// Bytes of the slot this header actually occupies, CRC included —
    /// what has to be destroyed to un-commit it.
    used: usize,
}

/// Parse one header slot out of `raw` (which must cover the header
/// region; offsets are absolute file offsets). `file_len` is the real
/// file size — a header whose claimed `n` does not fit the file is not
/// a valid commit and is rejected before its `n` can size anything.
/// Shared by the loader and `cursor_state`, so "newest valid header"
/// means the same thing everywhere.
fn parse_header_slot(raw: &[u8], geo: &Geo, slot: usize, file_len: usize) -> Option<ParsedHdr> {
    let at = geo.hdr_at(slot);
    let bytes = raw.get(at..at + geo.hdr_len())?;
    parse_header_at(bytes, at, geo, slot, file_len)
}

/// Parse one header slot from `bytes`, the slot's own bytes beginning at
/// absolute file offset `at`.
///
/// `bytes` may be a prefix of the slot rather than all of it. Every field
/// is read through a bounds-checked `get`, so a header whose used prefix
/// fits parses normally and one that needs more returns `None` — which is
/// what lets `cursor_state` read kilobytes instead of the slot's full
/// op-capacity and widen only when it must.
fn parse_header_at(
    bytes: &[u8],
    at: usize,
    geo: &Geo,
    slot: usize,
    file_len: usize,
) -> Option<ParsedHdr> {
    let tail_row = geo.row_bytes() + 4 + geo.id_bytes(1);
    let op_size = geo.op_size();
    let gen = u64::from_le_bytes(bytes.get(..8)?.try_into().unwrap());
    if (gen % 2) as usize != slot {
        return None;
    }
    let n64 = u64::from_le_bytes(bytes.get(8..16)?.try_into().unwrap());
    let n = usize::try_from(n64).ok()?;
    let units_end = (n / BLOCK)
        .checked_mul(geo.unit_len())
        .and_then(|u| u.checked_add(geo.unit_at(0)))?;
    if units_end > file_len {
        return None;
    }
    let mut p = 16 + (n % BLOCK) * tail_row;
    let n_units = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().unwrap()) as usize;
    if n_units > MAX_OPS {
        return None;
    }
    p += 4;
    let mut groups = Vec::with_capacity(n_units);
    for _ in 0..n_units {
        let b = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().unwrap()) as usize;
        // Ops only ever target committed blocks; an out-of-range block
        // index is a corrupt header, rejected here so it can never
        // reach the op-application loop's indexing.
        if b >= n / BLOCK {
            return None;
        }
        let n_ops = *bytes.get(p + 4)? as usize;
        p += 5;
        let mut ops = Vec::with_capacity(n_ops);
        for _ in 0..n_ops {
            let lane = *bytes.get(p)? as usize;
            if lane >= BLOCK {
                return None;
            }
            ops.push((b * BLOCK + lane, at + p + 1));
            p += op_size;
        }
        groups.push((b, ops));
    }
    let n_mat = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().unwrap()) as usize;
    if n_mat > MAX_OPS {
        return None;
    }
    p += 4;
    let mut delta_mat = Vec::with_capacity(n_mat);
    for _ in 0..n_mat {
        delta_mat.push(u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().unwrap()) as usize);
        p += 4;
    }
    let app_from = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().unwrap()) as usize;
    let app_to = u32::from_le_bytes(bytes.get(p + 4..p + 8)?.try_into().unwrap()) as usize;
    let delta_crc = u32::from_le_bytes(bytes.get(p + 8..p + 12)?.try_into().unwrap());
    p += 12;
    let stored = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().unwrap());
    if crc32(bytes.get(..p)?) != stored {
        return None;
    }
    Some(ParsedHdr {
        gen,
        n,
        tail_at: at + 16,
        groups,
        delta_mat,
        delta_app: app_from..app_to,
        delta_crc,
        used: p + 4,
    })
}

/// Load a v7 file. Blocks carry no checksums — the commit's delta
/// digest covers every unit at the sync that wrote it, which is all
/// the crash protocol needs; detecting later external damage is out of
/// scope, as it is for v6.
pub(crate) fn load(path: &Path, expect_calib_gen: u64, expect_kind: u8) -> io::Result<V7Load> {
    // Allocate for what the file DECLARES, capped by what it actually
    // holds — never for its apparent length. An image's size is fixed by
    // its geometry and row count, so a padded or sparse file is bytes
    // nobody asked for, and reading it wholesale made memory track the
    // file rather than the index. That is the v6 defect #487 fixed, and
    // v7 inherited it when it became the only format. Trailing bytes are
    // ignored, as they were before.
    let f = File::open(path)?;
    let on_disk = f.metadata()?.len();
    let want = declared_len(&f).unwrap_or(on_disk).min(on_disk);
    let mut raw = vec![
        0u8;
        usize::try_from(want).map_err(|_| io::Error::new(
            io::ErrorKind::InvalidData,
            "file too large for this platform"
        ))?
    ];
    crate::io::read_exact_at(&f, &mut raw, 0)?;
    load_image(raw, expect_calib_gen, expect_kind, &path.display().to_string())
}

/// Bytes a complete image of this file's geometry and row count occupies,
/// read from the superblock and the two header slots.
///
/// `None` whenever the prefix is not self-consistent: the parser then
/// sees the whole file and produces its own, better message rather than
/// a truncation artefact of a guess made here.
fn declared_len(f: &File) -> Option<u64> {
    let mut sb = [0u8; 64];
    crate::io::read_exact_at(f, &mut sb, 0).ok()?;
    if &sb[0..4] != V7_MAGIC || sb[4] != V7_VERSION {
        return None;
    }
    let bit_width = sb[5] as usize;
    let kind = sb[6];
    if !(2..=4).contains(&bit_width) || kind > 1 {
        return None;
    }
    let dim = u32::from_le_bytes(sb[7..11].try_into().ok()?) as usize;
    if dim != 0 && (!dim.is_multiple_of(8) || dim > crate::MAX_DIM) {
        return None;
    }
    // `n_calib` sits after the codebook, whose length follows bit_width.
    let n_levels = 1usize << bit_width;
    let n_calib_at = 23 + (2 * n_levels - 1) * 4;
    let mut n_calib_bytes = [0u8; 4];
    crate::io::read_exact_at(f, &mut n_calib_bytes, n_calib_at as u64).ok()?;
    let n_calib = u32::from_le_bytes(n_calib_bytes) as usize;
    if n_calib != 0 && n_calib != dim {
        return None;
    }
    let geo = Geo { kind, dim, bit_width, n_calib };

    // The row count lives in a header slot (gen, then n). Take the larger
    // of the two slots: whichever the loader ends up adopting, the image
    // it needs is no bigger than this.
    let mut n = 0u64;
    for slot in 0..2 {
        let mut buf = [0u8; 16];
        if crate::io::read_exact_at(f, &mut buf, geo.hdr_at_for_len(slot) as u64).is_err() {
            return None;
        }
        n = n.max(u64::from_le_bytes(buf[8..16].try_into().ok()?));
    }
    // Checked throughout: a tampered header can claim any n at all, and
    // this probe must not overflow computing a size for it — the parser
    // is what refuses an absurd count, with a message. Falling back to
    // `None` here just means reading the file as-is and letting it.
    let blocks = n / BLOCK as u64;
    let units = (geo.unit_len() as u64).checked_mul(blocks)?;
    (geo.unit_at(0) as u64).checked_add(units)
}

/// [`load`] over an image already in memory.
///
/// The path loader has always read the whole file up front and then
/// indexed around inside that buffer — v7's two header slots and block
/// units need random access to a *slice*, not to a seekable file. So a
/// byte image loads by exactly the same code, which is what lets
/// `from_bytes` accept v7.
///
/// `src` names the image in diagnostics (a path, or something like
/// "the byte image" for `from_bytes`).
pub(crate) fn load_image(
    mut raw: Vec<u8>,
    expect_calib_gen: u64,
    expect_kind: u8,
    src: &str,
) -> io::Result<V7Load> {
    if raw.len() < 11 || &raw[..4] != V7_MAGIC {
        return Err(bad("not a v7 file"));
    }
    if raw[4] != V7_VERSION {
        return Err(bad(format!("unsupported v7 revision {}", raw[4])));
    }
    let bit_width = raw[5] as usize;
    if !(2..=4).contains(&bit_width) {
        return Err(bad(format!("bit_width {bit_width} out of range")));
    }
    let kind = raw[6];
    if kind != expect_kind {
        return Err(bad(match kind {
            1 => "this v7 file holds an IdMapIndex; load it with IdMapIndex::load".to_string(),
            0 => {
                "this v7 file holds a TurboQuantIndex; load it with TurboQuantIndex::load"
                    .to_string()
            }
            k => format!("unknown v7 index kind {k}"),
        }));
    }
    let dim = read_u32(&raw, 7)? as usize;
    // dim 0 is the lazy sentinel: an index constructed without a
    // dimension that has never seen an add or a calibrate, so no
    // dimension is committed yet. It is only legal with no rows — the
    // geometry of a row is undefined without a dim — and it reloads as
    // the same lazy index. v6 carried the same sentinel; keeping it means
    // saving a store before its first write still works.
    if dim != 0 && (!dim.is_multiple_of(8) || dim > crate::MAX_DIM) {
        return Err(bad(format!("dim {dim} invalid")));
    }
    // Before anything driven by `dim` runs — the codebook solve below is
    // O(dim) and was where an absurd declared dim turned a load into a
    // hang. A complete image reserves MAX_OPS redo slots, each at least
    // one row wide, so a file smaller than that cannot be one whatever
    // else its header says. Cheap, and it does not depend on the size
    // guard above being the only thing that noticed.
    let min_header = MAX_OPS.saturating_mul(dim.saturating_mul(bit_width) / 8);
    if raw.len() < min_header {
        return Err(bad(format!(
            "truncated file: a {dim}-dim {bit_width}-bit image reserves at least \
             {min_header} bytes of header, but the file is {} bytes",
            raw.len(),
        )));
    }
    let nonce = read_u64_at(&raw, 11)?;
    let file_max_ops = read_u32(&raw, 19)? as usize;
    if file_max_ops != MAX_OPS {
        return Err(bad(format!(
            "unsupported header ops capacity {file_max_ops} (this build supports {MAX_OPS})"
        )));
    }

    // Codebook must match the canonical one, same as the v6 loader
    // (#320): a drifted codebook silently mis-scores.
    let n_levels = 1usize << bit_width;
    let mut off = 23;
    if dim == 0 {
        // The lazy sentinel embeds no codebook — one is solved per
        // dimension, and this image has no dimension and no rows. Skip
        // the bytes; there is nothing a drifted codebook could mis-score.
        off += (2 * n_levels - 1) * 4;
    } else {
        let (canon_b, canon_c) = crate::codebook::codebook(bit_width, dim);
        for want in canon_b.iter().chain(canon_c.iter()) {
            if read_f32(&raw, off)? != *want {
                return Err(bad("embedded codebook drifted from the canonical one"));
            }
            off += 4;
        }
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
    // THE calibration rule, shared with the v6 loader — one function,
    // so the two paths can never diverge again. (The superblock CRC is
    // no defence against an edited payload; it recomputes.)
    crate::io::validate_calibration(&tqplus_shift, &tqplus_scale)?;
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

    // --- pick the newest valid header --------------------------------
    // A header is variable-length inside its fixed slot: gen, n, the
    // n%32 tail rows, then the pending-op groups, then the CRC. Every
    // interior length is derivable, so the parse walks the used prefix
    // and checks the CRC exactly where the writer put it.
    let tail_row = row_bytes + 4 + geo.id_bytes(1);
    let parse_hdr = |slot: usize| parse_header_slot(&raw, &geo, slot, raw.len());
    // A commit is adopted only if the units its sync wrote are all
    // present with the bytes it recorded — the single-fsync protocol's
    // replacement for a write-ordering barrier. A commit that reached
    // disk before its data fails this and the other slot wins.
    let delta_ok = |h: &ParsedHdr| -> bool {
        delta_verified(h, geo.unit_len(), |b, d| {
            let at = geo.unit_at(b);
            raw.get(at..at + geo.unit_len()).map(|u| d.push(u)).is_some()
        })
    };
    let mut cands: Vec<ParsedHdr> =
        [parse_hdr(0), parse_hdr(1)].into_iter().flatten().collect();
    cands.sort_by_key(|h| std::cmp::Reverse(h.gen));
    let newest = cands.first().map(|h| h.gen);
    let Some(chosen) = cands.into_iter().find(delta_ok) else {
        return Err(bad("no valid commit header — unrecoverable v7 file"));
    };
    // Falling back is the protocol working, but it is also data loss:
    // the newest commit's own sync did not finish, and everything it
    // held is gone. Silence here reads as a clean load.
    if newest.is_some_and(|g| g != chosen.gen) {
        crate::warning::warn(&format!(
            "{}: the newest commit (generation {}) is incomplete — its sync did \
             not finish — so generation {} was loaded instead; changes made after \
             that commit are lost",
            src,
            newest.expect("checked"),
            chosen.gen,
        ));
    }
    let gen = chosen.gen;
    let n_vectors = chosen.n;
    // The lazy sentinel is only legal with no rows: without a committed
    // dimension a row has no geometry.
    if dim == 0 && n_vectors != 0 {
        return Err(bad(format!(
            "dim 0 with {n_vectors} rows: no dimension committed"
        )));
    }

    // --- gather the units, in place -----------------------------------
    // The read buffer becomes the cache: each block's codes roll
    // forward with copy_within instead of copying into a second
    // allocation (whose page faults were the bulk of load's cost).
    // Every destination lies strictly before its source — headers
    // precede the units and a unit is wider than its code payload — so
    // one forward pass parses block b's scales and ids from their
    // original position, then compacts its codes to `b * block_bytes`.
    // The header's tail rows and op payloads live in the region being
    // overwritten, so owned copies are taken first; pending ops are
    // applied to the compacted buffer at the end (their slots' scales
    // and ids override by index).
    let n_blocks = n_vectors / BLOCK;
    let total_blocks = n_vectors.div_ceil(BLOCK);
    let block_bytes = BLOCK * row_bytes;
    debug_assert!(geo.unit_at(0) >= block_bytes, "compaction dest must trail the sources");

    let n_tail = n_vectors % BLOCK;
    let tail_copy: Vec<u8> = raw
        .get(chosen.tail_at..chosen.tail_at + n_tail * tail_row)
        .ok_or_else(|| bad("truncated commit tail"))?
        .to_vec();
    let op_size = row_bytes + 4 + geo.id_bytes(1);
    // (block, ops as (slot, payload bytes)).
    type OwnedGroup = (usize, Vec<(usize, Vec<u8>)>);
    let mut ops_owned: Vec<OwnedGroup> = Vec::with_capacity(chosen.groups.len());
    for (b, ops) in &chosen.groups {
        let mut owned = Vec::with_capacity(ops.len());
        for &(slot, payload_at) in ops {
            let payload = raw
                .get(payload_at..payload_at + op_size)
                .ok_or_else(|| bad("truncated pending op"))?
                .to_vec();
            owned.push((slot, payload));
        }
        ops_owned.push((*b, owned));
    }

    let mut scales: Vec<f32> = Vec::with_capacity(n_vectors);
    let mut ids: Vec<u64> = Vec::with_capacity(if kind == 1 { n_vectors } else { 0 });
    for b in 0..n_blocks {
        let at = geo.unit_at(b);
        if raw.len() < at + geo.unit_len() {
            return Err(bad("truncated block unit"));
        }
        for lane in 0..BLOCK {
            let so = at + block_bytes + lane * 4;
            let v = f32::from_le_bytes(raw[so..so + 4].try_into().unwrap());
            if !v.is_finite() || !(0.0..=crate::io::MAX_VECTOR_SCALE).contains(&v) {
                return Err(bad(format!("invalid per-vector scale in block {b}")));
            }
            scales.push(v);
        }
        if kind == 1 {
            for lane in 0..BLOCK {
                let io_ = at + block_bytes + BLOCK * 4 + lane * 8;
                ids.push(u64::from_le_bytes(raw[io_..io_ + 8].try_into().unwrap()));
            }
        }
        raw.copy_within(at..at + block_bytes, b * block_bytes);
    }
    raw.truncate(n_blocks * block_bytes);
    raw.resize(total_blocks * block_bytes, 0);
    // `truncate` never releases capacity and the `resize` stays inside
    // it, so without this the whole-file `fs::read` allocation — header
    // reserve, per-block scale and id sections, post-`n` padding, all
    // bytes the search path never reads — is carried in the blocked
    // cache for the index's lifetime (#480). The shrinking realloc
    // releases the tail rather than copying.
    raw.shrink_to_fit();
    let mut seq_blocked = raw;

    // Pending ops override their slots in the compacted buffer.
    for (b, ops) in &ops_owned {
        for (slot, payload) in ops {
            let lane = slot % BLOCK;
            for g in 0..row_bytes {
                seq_blocked[b * block_bytes + g * BLOCK + lane] = payload[g];
            }
            let v = f32::from_le_bytes(payload[row_bytes..row_bytes + 4].try_into().unwrap());
            if !v.is_finite() || !(0.0..=crate::io::MAX_VECTOR_SCALE).contains(&v) {
                return Err(bad("invalid per-vector scale in a pending op"));
            }
            scales[*slot] = v;
            if kind == 1 {
                ids[*slot] =
                    u64::from_le_bytes(payload[row_bytes + 4..row_bytes + 12].try_into().unwrap());
            }
        }
    }

    // --- tail rows from the owned header copy --------------------------
    for k in 0..n_tail {
        let r = n_blocks * BLOCK + k;
        let lane = r % BLOCK;
        let row = &tail_copy[k * tail_row..(k + 1) * tail_row];
        for g in 0..row_bytes {
            seq_blocked[n_blocks * block_bytes + g * BLOCK + lane] = row[g];
        }
        let v = f32::from_le_bytes(row[row_bytes..row_bytes + 4].try_into().unwrap());
        if !v.is_finite() || !(0.0..=crate::io::MAX_VECTOR_SCALE).contains(&v) {
            return Err(bad("invalid per-vector scale in the commit tail"));
        }
        scales.push(v);
        if kind == 1 {
            ids.push(u64::from_le_bytes(row[row_bytes + 4..row_bytes + 12].try_into().unwrap()));
        }
    }

    Ok(V7Load {
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
            nonce,
        },
        pending_slots: chosen
            .groups
            .iter()
            .flat_map(|(_, ops)| ops.iter().map(|&(s, _)| s))
            .collect(),
    })
}

// ---------------------------------------------------------------------
// Cursor-vs-file identity
// ---------------------------------------------------------------------

/// What a cursor finds when checked against the file it points at.
pub(crate) enum CursorState {
    /// The file's newest commit is the cursor's: sync in place safely.
    Intact {
        /// `Some(used)` when a parseable header NEWER than the cursor's
        /// commit is sitting in the slot the next sync writes — a commit
        /// `load` rejected and fell back past — where `used` is how many
        /// of that slot's bytes it occupies. The next sync must destroy
        /// them before moving any data; see [`plan_incremental`].
        stale_ahead: Option<usize>,
    },
    /// A valid v7 file whose newest commit is not the cursor's —
    /// another writer advanced or replaced it. Touching it would
    /// silently destroy their commits: refuse.
    Foreign,
    /// Gone or not v7 (a v6 write(), an empty file): nothing another
    /// v7 writer committed is at stake — write full.
    Replaced,
}

/// Verify `cursor` against the file before a sync touches a byte.
///
/// "Newest header" here means the same thing it means to the loader —
/// the newest slot that parses AND whose delta verifies — via the same
/// [`parse_header_slot`] and [`delta_digest`]. Anything weaker
/// disagrees with `load` exactly in the crash states the delta
/// descriptor exists for: a commit whose data never landed would look
/// newest here while `load` correctly falls back, and every subsequent
/// sync would refuse as Foreign with no way out.
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
    let file_len =
        usize::try_from(f.metadata()?.len()).map_err(|_| bad("file too large"))?;
    let mut head = [0u8; 19];
    match f.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Ok(CursorState::Replaced)
        }
        Err(e) => return Err(e),
    }
    if &head[..4] != V7_MAGIC || head[4] != V7_VERSION {
        return Ok(CursorState::Replaced);
    }
    // Generations cannot identify a file — every full write starts at
    // 0 — so the superblock nonce is checked first: a different nonce
    // is a different file (another writer's, or another index type's),
    // whatever its generation says.
    let file_nonce = u64::from_le_bytes(head[11..19].try_into().unwrap());
    if file_nonce != cursor.nonce {
        // An *unclaimed* file is not another writer's sync destination —
        // nobody is syncing to it, by definition. It is a snapshot that
        // `write` dropped over this path, including possibly one this
        // very index wrote. Rebuilding is right there; erroring would
        // leave a `sync -> write -> sync` sequence permanently stuck.
        // A different *claimed* nonce is still foreign and still refused.
        if file_nonce == UNCLAIMED_NONCE {
            return Ok(CursorState::Replaced);
        }
        return Ok(CursorState::Foreign);
    }
    // The nonce matched, so this is the file this cursor wrote and the
    // caller's geometry is its geometry. Read the header region and
    // select exactly as the loader does.
    // Read each slot's used prefix, not the whole slot. Slots are sized
    // for the 1024-op cap, so reading both in full costs ~856 KB on the
    // way into every sync — including a 32-row append whose entire commit
    // is ~12 KB. A commit with no pending ops fits in `hdr_probe_len`;
    // only one carrying ops needs the rest, and then it pays for the
    // second read once.
    let mut read_slot = |slot: usize| -> Option<ParsedHdr> {
        let at = geo.hdr_at(slot);
        let avail = file_len.checked_sub(at)?;
        let mut want = geo.hdr_probe_len().min(avail);
        loop {
            let mut buf = vec![0u8; want];
            // An I/O error reads as "no candidate in this slot": if the
            // OTHER slot still parses, the verdict can be Foreign — a
            // misdiagnosis for a transient read fault — but the cursor
            // stays bound, so a retry re-reads and self-heals; if both
            // fail, Replaced forces the protective full rewrite.
            f.seek(SeekFrom::Start(at as u64)).ok()?;
            f.read_exact(&mut buf).ok()?;
            if let Some(h) = parse_header_at(&buf, at, geo, slot, file_len) {
                return Some(h);
            }
            // Either the header genuinely does not parse, or it carries
            // ops and outgrew the probe. Widen once to the full slot; a
            // second failure is a real one.
            let full = geo.hdr_len().min(avail);
            if want >= full {
                return None;
            }
            want = full;
        }
    };
    let mut cands: Vec<ParsedHdr> = [read_slot(0), read_slot(1)]
        .into_iter()
        .flatten()
        .collect();
    cands.sort_by_key(|h| std::cmp::Reverse(h.gen));
    // The cursor's own commit, still the newest header on the file, needs no
    // delta re-read: it was verified when the cursor was established, and
    // nothing since could have unverified it.
    //
    // A cursor is established exactly two ways. Either this process wrote
    // that commit — and `sync` returns Ok only after `sync_all` reports the
    // batch durable, so its units are on stable storage — or `load` adopted
    // it, and `load` selects a commit by running this very delta check. The
    // nonce matched above, so this is that same file; a newest header at the
    // cursor's generation is therefore the newest adoptable commit, which is
    // what Intact means. (One stated narrowing: damage from OUTSIDE the
    // writer landing on the cursor's own units after establishment was
    // previously caught here by the re-read and answered with a full
    // rewrite; it now syncs forward unnoticed. Out-of-writer damage is
    // explicitly out of scope for blocks — same stance as v6 and load —
    // so the skip trades a defense the format never promised.)
    // Re-reading every unit the last sync wrote to prove
    // that again costs the whole delta on the way into the next sync — 12.6
    // MB of read and CRC after a 1000-removal settle, three quarters of that
    // sync's cost.
    //
    // Any other generation still takes the full verifying walk below: a
    // newer header means someone else advanced the file, and deciding
    // Foreign vs Intact there does require knowing whether their data
    // landed. This shortcut only ever skips work for a commit already
    // proven, and in the unsupported concurrent-writer case it errs toward
    // refusing rather than adopting.
    // Nothing parses ahead of the cursor's own commit here — it is the
    // newest — so the slot the next sync targets holds generation
    // `gen - 1` and needs no repair.
    if cands.first().is_some_and(|h| h.gen == cursor.gen) {
        return Ok(CursorState::Intact { stale_ahead: None });
    }
    let stale_ahead = cands
        .iter()
        .filter(|h| h.gen > cursor.gen)
        .map(|h| h.used)
        .max();
    let mut adoptable: Option<u64> = None;
    // One reusable unit buffer, refilled per unit and folded straight
    // into the digest.
    let mut unit_buf = vec![0u8; geo.unit_len()];
    for h in &cands {
        let verified = delta_verified(h, geo.unit_len(), |b, d| {
            let at = geo.unit_at(b);
            if at + geo.unit_len() > file_len {
                return false;
            }
            if f.seek(SeekFrom::Start(at as u64)).is_err() || f.read_exact(&mut unit_buf).is_err() {
                return false;
            }
            d.push(&unit_buf);
            true
        });
        if verified {
            adoptable = Some(h.gen);
            break;
        }
    }
    match adoptable {
        Some(g) if g == cursor.gen => Ok(CursorState::Intact { stale_ahead }),
        Some(_) => Ok(CursorState::Foreign),
        // Our own file with no adoptable commit left is corrupt beyond
        // incremental repair; a full rewrite is the only safe sync.
        None => Ok(CursorState::Replaced),
    }
}

/// Test-only: recompute the superblock and header CRCs over whatever
/// bytes are present — the corruption matrix uses this so a tamper is
/// caught by semantic validation, never by a stale checksum. The walk
/// mirrors the parser without validating; when a tampered length runs
/// past its slot, that seal is skipped (the harness still demands a
/// polite refusal).
#[cfg(test)]
/// Test-only: the used-prefix length of a header slot — everything the
/// parser actually reads, CRC included — or a small floor when the slot
/// doesn't parse. The corruption matrix uses it to spend its budget on
/// bytes that are read, striding the never-read reserved slack.
pub(crate) fn hdr_used_for_test(raw: &[u8], geo: &Geo, slot: usize) -> usize {
    let tail_row = geo.row_bytes() + 4 + geo.id_bytes(1);
    let op_size = geo.op_size();
    let at = geo.hdr_at(slot);
    let Some(bytes) = raw.get(at..at + geo.hdr_len()) else {
        return 16;
    };
    let walk = || -> Option<usize> {
        let n = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().ok()?)).ok()?;
        let mut p = 16 + (n % BLOCK) * tail_row;
        let n_units = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().ok()?) as usize;
        p += 4;
        if n_units > MAX_OPS {
            return None;
        }
        for _ in 0..n_units {
            let n_ops = *bytes.get(p + 4)? as usize;
            p += 5 + n_ops * op_size;
        }
        let n_mat = u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().ok()?) as usize;
        if n_mat > MAX_OPS {
            return None;
        }
        p += 4 + n_mat * 4 + 12;
        bytes.get(p..p + 4)?;
        Some(p + 4)
    };
    walk().unwrap_or(16)
}

#[cfg(test)]
pub(crate) fn reseal_for_test(bytes: &mut [u8], geo: &Geo) {
    let sb = geo.sb_len();
    if bytes.len() >= sb {
        let c = crc32(&bytes[..sb - 4]);
        bytes[sb - 4..sb].copy_from_slice(&c.to_le_bytes());
    }
    let hdr_len = geo.hdr_len();
    let tail_row = geo.row_bytes() + 4 + geo.id_bytes(1);
    let op_size = geo.op_size();
    for slot in 0..2 {
        let at = geo.hdr_at(slot);
        if bytes.len() < at + hdr_len {
            continue;
        }
        let h = &bytes[at..at + hdr_len];
        let n = u64::from_le_bytes(h[8..16].try_into().unwrap()) as usize;
        let Some(mut p) = (n % BLOCK).checked_mul(tail_row).and_then(|t| t.checked_add(16))
        else {
            continue;
        };
        let read_u32_at = |h: &[u8], q: usize| -> Option<u32> {
            h.get(q..q + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        };
        let Some(n_units) = read_u32_at(h, p) else { continue };
        p += 4;
        let mut ok = true;
        for _ in 0..n_units {
            let Some(&n_ops) = h.get(p + 4) else {
                ok = false;
                break;
            };
            let Some(np) = (n_ops as usize)
                .checked_mul(op_size)
                .and_then(|v| v.checked_add(p + 5))
            else {
                ok = false;
                break;
            };
            p = np;
        }
        if !ok {
            continue;
        }
        let Some(n_mat) = read_u32_at(h, p) else { continue };
        let Some(np) = (n_mat as usize)
            .checked_mul(4)
            .and_then(|v| v.checked_add(p + 4 + 12))
        else {
            continue;
        };
        p = np;
        if p + 4 <= hdr_len {
            let c = crc32(&bytes[at..at + p]);
            bytes[at + p..at + p + 4].copy_from_slice(&c.to_le_bytes());
        }
    }
}

/// Sniff: is this a v7 file?
pub(crate) fn is_v7(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|_| &magic == V7_MAGIC)
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

    /// The split threshold itself, pinned at the boundary: 4095 bytes
    /// take the single-chain path, 4096 the interleaved one. Writer and
    /// reader share the rule, so the only way to catch an off-by-one is
    /// against independently-composed references.
    #[test]
    fn crc_split_threshold_is_exact() {
        for len in [4095usize, 4096, 4097] {
            let data: Vec<u8> = (0..len).map(|i| (i % 249) as u8).collect();
            let expect = if len < 4096 {
                crc32c_soft(&data)
            } else {
                let third = len / 3;
                let (a, rest) = data.split_at(third);
                let (b, c) = rest.split_at(third);
                let mut d = [0u8; 12];
                d[..4].copy_from_slice(&crc32c_soft(a).to_le_bytes());
                d[4..8].copy_from_slice(&crc32c_soft(b).to_le_bytes());
                d[8..].copy_from_slice(&crc32c_soft(c).to_le_bytes());
                crc32c_soft(&d)
            };
            assert_eq!(crc32(&data), expect, "len {len}");
        }
    }

    /// The digest must depend on every byte it covers — the property
    /// that was silently false when it hashed whole unit codewords
    /// (data followed by its own CRC drives CRC-32C to a fixed
    /// residue, so a list of self-consistent units hashed to a
    /// content-free constant). Never feed a checksum things that carry
    /// their own checksum.
    #[test]
    fn the_delta_digest_depends_on_every_byte() {
        let mut body_a = vec![7u8; 1000];
        let body_b = vec![9u8; 1000];
        let base = delta_digest(3, &[(0usize, body_a.as_slice()), (1, body_b.as_slice())]);
        // Content sensitivity, at every byte position.
        for i in [0usize, 1, 499, 998, 999] {
            body_a[i] ^= 1;
            let changed = delta_digest(3, &[(0usize, body_a.as_slice()), (1, body_b.as_slice())]);
            assert_ne!(base, changed, "flip at byte {i} must change the digest");
            body_a[i] ^= 1;
        }
        // Position and generation sensitivity.
        assert_ne!(
            base,
            delta_digest(3, &[(1usize, body_a.as_slice()), (0, body_b.as_slice())]),
            "block indices must bind"
        );
        assert_ne!(
            base,
            delta_digest(4, &[(0usize, body_a.as_slice()), (1, body_b.as_slice())]),
            "the generation must bind"
        );
        // The old bug's shape, demonstrated so it stays understood: a
        // payload that ends with its own CRC contributes only its
        // LENGTH to any CRC stream (combine-algebra) — mixing in
        // indices or a generation does not defeat it. The digest is
        // sound STRUCTURALLY: unit bodies carry no embedded checksums,
        // so a self-consistent codeword can never be what we hash.
        let mut u1 = vec![1u8; 512];
        let c = crc32(&u1);
        u1.extend_from_slice(&c.to_le_bytes());
        let mut u2 = vec![2u8; 512];
        let c = crc32(&u2);
        u2.extend_from_slice(&c.to_le_bytes());
        assert_eq!(
            delta_digest(1, &[(0usize, u1.as_slice())]),
            delta_digest(1, &[(0usize, u2.as_slice())]),
            "codeword payloads DO collide — which is why unit bodies must never embed their own CRC"
        );
    }

    /// The streaming digest must equal `crc32` of the buffer it stands
    /// in for, whatever the chunk boundaries. The value is a file
    /// format: a digest that disagreed with the materialized one would
    /// make every commit written by one build unverifiable by the other.
    ///
    /// Swept across the 4096-byte threshold where `crc32` switches to
    /// three interleaved thirds, and pushed in chunk sizes chosen to
    /// straddle both split points.
    #[test]
    fn the_streaming_digest_equals_the_materialized_one() {
        let mut data = vec![0u8; 20_000];
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        for b in data.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s >> 32) as u8;
        }
        for total in [0usize, 1, 7, 8, 4094, 4095, 4096, 4097, 6143, 6144, 12_289, 20_000] {
            let buf = &data[..total];
            let want = crc32(buf);
            for chunk in [1usize, 3, 8, 64, 1000, 2048, 4096, 65_536] {
                let mut d = DeltaDigest::new(total);
                for piece in buf.chunks(chunk.max(1)) {
                    d.push(piece);
                }
                assert_eq!(d.finish(), want, "total {total}, chunk {chunk}");
            }
            // Ragged pushes, so a boundary lands mid-chunk at every
            // offset the sweep above cannot reach.
            let mut d = DeltaDigest::new(total);
            let (mut at, mut step) = (0usize, 1usize);
            while at < total {
                let take = step.min(total - at);
                d.push(&buf[at..at + take]);
                at += take;
                step = step * 2 + 1;
            }
            assert_eq!(d.finish(), want, "total {total}, ragged");
        }
    }

    /// Every derived offset in the file, restated independently — an
    /// arithmetic slip anywhere in `Geo` breaks one of these before it
    /// can corrupt a file.
    #[test]
    fn geometry_is_pinned() {
        for (kind, dim, bit_width, n_calib) in
            [(0u8, 64usize, 4usize, 64usize), (1, 128, 2, 0), (0, 64, 3, 64)]
        {
            let geo = Geo {
                kind,
                dim,
                bit_width,
                n_calib,
            };
            let row = dim / (8 / bit_width);
            let id1 = if kind == 1 { 8 } else { 0 };
            let nl = 1usize << bit_width;
            let tail_row = row + 4 + id1;
            let op = 1 + row + 4 + id1;
            assert_eq!(geo.row_bytes(), row, "row stride");
            assert_eq!(geo.op_size(), op, "op size");
            assert_eq!(
                geo.sb_len(),
                23 + (nl - 1) * 4 + nl * 4 + 4 + n_calib * 8 + 4,
                "superblock"
            );
            assert_eq!(
                geo.hdr_len(),
                16 + 31 * tail_row + 4 + MAX_OPS * (5 + op) + 4 + MAX_OPS * 4 + 12 + 4,
                "header slot"
            );
            assert_eq!(geo.unit_len(), 32 * row + 128 + 32 * id1, "unit");
            assert_eq!(
                geo.unit_at(3) - geo.unit_at(2),
                geo.unit_len(),
                "unit stride"
            );
            // The probe read `cursor_state` opens a sync with. Pinned by
            // formula AND by what it has to cover, because it is a hint:
            // too short is still *correct* (the parse fails and the read
            // widens to the full slot), it just silently gives up the
            // optimization. Only the second assert notices that.
            assert_eq!(
                geo.hdr_probe_len(),
                16 + 31 * tail_row + 4 + 4 + MAX_OPS * 4 + 12 + 4,
                "header probe"
            );
            // Independently derived: the probe minus the full slot's op
            // region must be exactly the op region's size — i.e. the
            // probe concedes ONLY the op groups, never tail rows or the
            // delta descriptor an op-free commit still carries.
            assert_eq!(
                geo.hdr_len() - geo.hdr_probe_len(),
                MAX_OPS * (5 + geo.op_size()),
                "the probe must give up exactly the op-group region",
            );
            assert!(
                geo.hdr_probe_len() < geo.hdr_len(),
                "a probe that is not smaller than the slot saves nothing",
            );
        }
    }
}
