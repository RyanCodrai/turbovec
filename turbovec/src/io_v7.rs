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
//! slot.
//!
//! Every sync is ONE write batch and ONE fsync. No ordering barrier
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
//! Load applies pending ops in memory over each named unit — recovery
//! converges because the ops are absolute writes, so no state outside
//! the committed header is ever needed: no undo area, no attempt
//! versioning, no carried repair sets. A torn sync leaves either the
//! old header (whose ops still describe the old state) or the new one
//! (whose ops describe the new), never a third thing.
//!
//! ## What is checksummed, and why
//!
//! The commit headers and the superblock carry CRCs because validity
//! IS the commit mechanism: a torn header write must be recognizably
//! invalid for the A/B fallback to work, and a torn compaction is
//! caught by the superblock the same way. Block units also carry a
//! trailing CRC, but the default load does NOT verify them — the crash
//! protocol never needs it, so the only thing block checks could catch
//! is damage from outside the writer (bit rot, bad copies), which is
//! out of scope exactly as it was for v6. The bytes are written anyway
//! (4 per block, free) so a verifying load remains possible; the test
//! harness uses [`load_verified`] as development scaffolding.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) const V7_MAGIC: &[u8; 4] = b"TV7\0";
pub(crate) const V7_VERSION: u8 = 1;
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
        19 + (self.n_levels() - 1) * 4 + self.n_levels() * 4 + 4 + self.n_calib * 8 + 4
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
            + MAX_OPS * (5 + self.op_size())
            + 4
            + MAX_OPS * 4
            + 12
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

fn superblock(src: &SyncSource<'_>, nonce: u64) -> Vec<u8> {
    let mut sb = Vec::new();
    sb.extend_from_slice(V7_MAGIC);
    sb.push(V7_VERSION);
    sb.push(src.bit_width as u8);
    sb.push(src.kind);
    sb.extend_from_slice(&(src.dim as u32).to_le_bytes());
    sb.extend_from_slice(&nonce.to_le_bytes());
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
        h.extend_from_slice(&(src.row_codes)(r));
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
            h.extend_from_slice(&(src.row_codes)(s));
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

    // One batch, one fsync: the header's delta descriptor names every
    // unit this sync writes and their bytes' CRC, so a commit that
    // reaches disk before its data is detectable at load — no ordering
    // barrier needed.
    let mut batch = Batch::default();
    let mut delta_bytes: Vec<u8> = Vec::new();
    delta_bytes.extend_from_slice(&gen.to_le_bytes());
    for b in materialize.iter().copied().chain(old_blocks..new_blocks) {
        let bytes = unit_bytes(src, b);
        delta_bytes.extend_from_slice(&(b as u32).to_le_bytes());
        delta_bytes.extend_from_slice(&bytes);
        batch.ops.push((geo.unit_at(b) as u64, bytes));
    }
    let delta_crc = crc32(&delta_bytes);

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
    let batches = vec![batch];

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
fn delta_digest<'a>(gen: u64, units: impl Iterator<Item = (usize, &'a [u8])>) -> u32 {
    let mut buf = Vec::new();
    buf.extend_from_slice(&gen.to_le_bytes());
    for (b, body) in units {
        buf.extend_from_slice(&(b as u32).to_le_bytes());
        buf.extend_from_slice(body);
    }
    crc32(&buf)
}

/// Every sync is durable: `sync_all`, not `sync_data` — on every
/// platform (macOS F_FULLFSYNC included), and syncs change the file
/// length, which data-only variants may not persist.
/// THE delta check, shared by the loader and `cursor_state` so
/// "adoptable commit" means exactly one thing: fetch each unit the
/// commit's sync wrote through `read_unit` (false = unavailable) and
/// compare the reconstructed digest.
fn delta_verified(
    h: &ParsedHdr,
    mut read_unit: impl FnMut(usize, &mut Vec<u8>) -> bool,
) -> bool {
    let mut units: Vec<(usize, Vec<u8>)> = Vec::new();
    for b in h.delta_mat.iter().copied().chain(h.delta_app.clone()) {
        let mut u = Vec::new();
        if !read_unit(b, &mut u) {
            return false;
        }
        units.push((b, u));
    }
    delta_digest(h.gen, units.iter().map(|(b, u)| (*b, u.as_slice()))) == h.delta_crc
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

/// The whole file as one byte image for generation `gen` — creation and
/// compaction both go through this, via a temp sibling + atomic rename
/// (the v6 writer's own machinery: naming, Windows retry, stale-temp
/// sweep, parent-dir fsync posture).
pub(crate) fn write_full(
    path: &Path,
    src: &SyncSource<'_>,
    calib_gen: u64,
) -> io::Result<SyncCursor> {
    let geo = src.geo();
    let gen = 0u64;
    let nonce = crate::io::file_nonce();
    let n_blocks = src.n_vectors / BLOCK;
    let mut image = superblock(src, nonce);
    // Slot 0 carries generation 0 (no pending ops); slot 1 starts
    // invalid (zeroed, CRC cannot match).
    let h = header_slot(
        src,
        gen,
        src.n_vectors,
        &[],
        (&[], 0..0, delta_digest(gen, std::iter::empty())),
    );
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
        fsync_commit(&f)
    })();
    let result = result.and_then(|()| {
        drop(f);
        crate::io::rename_atomic(&tmp, path)
    });
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    crate::io::sync_parent_dir_after_commit(path);
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
}

/// Parse one header slot out of `raw` (which must cover the header
/// region; offsets are absolute file offsets). `file_len` is the real
/// file size — a header whose claimed `n` does not fit the file is not
/// a valid commit and is rejected before its `n` can size anything.
/// Shared by the loader and `cursor_state`, so "newest valid header"
/// means the same thing everywhere.
fn parse_header_slot(raw: &[u8], geo: &Geo, slot: usize, file_len: usize) -> Option<ParsedHdr> {
    let hdr_len = geo.hdr_len();
    let tail_row = geo.row_bytes() + 4 + geo.id_bytes(1);
    let op_size = geo.op_size();
    let at = geo.hdr_at(slot);
    let bytes = raw.get(at..at + hdr_len)?;
    let gen = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    if (gen % 2) as usize != slot {
        return None;
    }
    let n64 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
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
    if crc32(&bytes[..p]) != stored {
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
    })
}

/// Load a v7 file. Blocks carry no checksums — the commit's delta
/// digest covers every unit at the sync that wrote it, which is all
/// the crash protocol needs; detecting later external damage is out of
/// scope, as it is for v6.
pub(crate) fn load(path: &Path, expect_calib_gen: u64, expect_kind: u8) -> io::Result<V7Load> {
    let mut raw = std::fs::read(path)?;
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
    if dim == 0 || !dim.is_multiple_of(8) || dim > crate::MAX_DIM {
        return Err(bad(format!("dim {dim} invalid")));
    }
    let nonce = read_u64_at(&raw, 11)?;

    // Codebook must match the canonical one, same as the v6 loader
    // (#320): a drifted codebook silently mis-scores.
    let (canon_b, canon_c) = crate::codebook::codebook(bit_width, dim);
    let mut off = 19;
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
        delta_verified(h, |b, out| {
            let at = geo.unit_at(b);
            raw.get(at..at + geo.unit_len())
                .map(|u| out.extend_from_slice(u))
                .is_some()
        })
    };
    let mut cands: Vec<ParsedHdr> =
        [parse_hdr(0), parse_hdr(1)].into_iter().flatten().collect();
    cands.sort_by_key(|h| std::cmp::Reverse(h.gen));
    let Some(chosen) = cands.into_iter().find(delta_ok) else {
        return Err(bad("no valid commit header — unrecoverable v7 file"));
    };
    let gen = chosen.gen;
    let n_vectors = chosen.n;

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
            if !v.is_finite() || v < 0.0 {
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
    let mut seq_blocked = raw;

    // Pending ops override their slots in the compacted buffer.
    for (b, ops) in &ops_owned {
        for (slot, payload) in ops {
            let lane = slot % BLOCK;
            for g in 0..row_bytes {
                seq_blocked[b * block_bytes + g * BLOCK + lane] = payload[g];
            }
            let v = f32::from_le_bytes(payload[row_bytes..row_bytes + 4].try_into().unwrap());
            if !v.is_finite() || v < 0.0 {
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
        if !v.is_finite() || v < 0.0 {
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
    Intact,
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
    if u64::from_le_bytes(head[11..19].try_into().unwrap()) != cursor.nonce {
        return Ok(CursorState::Foreign);
    }
    // The nonce matched, so this is the file this cursor wrote and the
    // caller's geometry is its geometry. Read the header region and
    // select exactly as the loader does.
    let prefix_len = geo.unit_at(0).min(file_len);
    let mut prefix = vec![0u8; prefix_len];
    f.seek(SeekFrom::Start(0))?;
    if f.read_exact(&mut prefix).is_err() {
        return Ok(CursorState::Replaced);
    }
    let mut cands: Vec<ParsedHdr> = [
        parse_header_slot(&prefix, geo, 0, file_len),
        parse_header_slot(&prefix, geo, 1, file_len),
    ]
    .into_iter()
    .flatten()
    .collect();
    cands.sort_by_key(|h| std::cmp::Reverse(h.gen));
    let mut adoptable: Option<u64> = None;
    for h in &cands {
        let verified = delta_verified(h, |b, out| {
            let at = geo.unit_at(b);
            if at + geo.unit_len() > file_len {
                return false;
            }
            out.resize(geo.unit_len(), 0);
            f.seek(SeekFrom::Start(at as u64)).is_ok() && f.read_exact(out).is_ok()
        });
        if verified {
            adoptable = Some(h.gen);
            break;
        }
    }
    match adoptable {
        Some(g) if g == cursor.gen => Ok(CursorState::Intact),
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
        let base = delta_digest(3, [(0usize, body_a.as_slice()), (1, body_b.as_slice())].into_iter());
        // Content sensitivity, at every byte position.
        for i in [0usize, 1, 499, 998, 999] {
            body_a[i] ^= 1;
            let changed =
                delta_digest(3, [(0usize, body_a.as_slice()), (1, body_b.as_slice())].into_iter());
            assert_ne!(base, changed, "flip at byte {i} must change the digest");
            body_a[i] ^= 1;
        }
        // Position and generation sensitivity.
        assert_ne!(
            base,
            delta_digest(3, [(1usize, body_a.as_slice()), (0, body_b.as_slice())].into_iter()),
            "block indices must bind"
        );
        assert_ne!(
            base,
            delta_digest(4, [(0usize, body_a.as_slice()), (1, body_b.as_slice())].into_iter()),
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
            delta_digest(1, [(0usize, u1.as_slice())].into_iter()),
            delta_digest(1, [(0usize, u2.as_slice())].into_iter()),
            "codeword payloads DO collide — which is why unit bodies must never embed their own CRC"
        );
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
                19 + (nl - 1) * 4 + nl * 4 + 4 + n_calib * 8 + 4,
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
        }
    }
}
