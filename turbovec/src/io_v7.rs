//! The v7 incremental container: an append-only record log.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! SUPERBLOCK   "TV7\0" | ver u8 | bit_width u8 | dim u32
//!              | boundaries f32[(1<<bw)-1] | centroids f32[1<<bw]
//!              | n_calib u32 | shift f32[n_calib] | scale f32[n_calib]
//!              (immutable between compactions: the codebook is a pure
//!               function of (bit_width, dim), and the calibration can
//!               only change via `calibrate`, which forces the next
//!               sync to compact — so no commit ever re-serializes it)
//! RECORDS      any number of, in log order:
//!   "SEG1" | start_slot u64 | n_rows u32
//!          | codes (seq-blocked, n_rows/32 whole blocks)
//!          | scales f32[n_rows] | crc32 u32
//!   "PAT1" | slot u64 | codes row (dim*bits/8 bytes) | scale f32 | crc32
//!   "CMT1" | body_len u32 | body | crc32, where body =
//!          gen u64 | n_vectors u64 | n_tail u32
//!          | tail codes (n_tail rows, row-major) | tail scales f32[n_tail]
//!          | data_end u64
//! ```
//!
//! The one rule everything else falls out of: **nothing committed is
//! ever overwritten — commit records included.** A sync appends its
//! records after the previous commit, fsyncs, then appends a fresh
//! commit and fsyncs again. A crash between the two barriers leaves the
//! previous commit intact as the recovery anchor; recovery scans back
//! from EOF for the last commit whose CRC validates and whose
//! `data_end` is consistent, and truncates the torn suffix on the next
//! sync. Superseded commits stay in the log as validated, skippable
//! records until a compaction (any full rewrite — `calibrate` forces
//! one) reclaims them.
//!
//! Segments hold only whole 32-row SIMD blocks, so committed segment
//! bytes are never rewritten by a later append; the partial tail block
//! — the only mutable rows — lives inside the commit record itself,
//! re-serialized on every sync (bounded: at most 31 rows).
//!
//! A removal is one `PAT1` record: `swap_remove` moves the last live
//! row into the hole, so `(slot, filler codes, filler scale)` plus the
//! commit's decremented `n_vectors` expresses the whole operation.
//!
//! Replay is positional, not sequential: every segment declares the
//! first slot it covers, a patch overwrites exactly its slot, and the
//! commit's `n_vectors` truncates whatever the log wrote beyond the
//! live count. Later records win over earlier ones in log order, which
//! is what lets a shrink below the segment watermark be repaired by the
//! next sync's segment rewriting the now-dead range.
//!
//! Codes travel in the arch-neutral *sequential blocked* layout the v6
//! payload uses, so files stay byte-identical across platforms; the
//! caller converts from whichever in-memory layout it holds.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) const V7_MAGIC: &[u8; 4] = b"TV7\0";
pub(crate) const V7_VERSION: u8 = 1;
const SEG: &[u8; 4] = b"SEG1";
const PAT: &[u8; 4] = b"PAT1";
const CMT: &[u8; 4] = b"CMT1";

/// Plain CRC-32 (IEEE, reflected), table-free bitwise form. The log's
/// records are small relative to the payload they guard, and the load
/// path touches each committed byte once either way; a table variant is
/// a later optimization, not a format change.
static CRC_TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();

/// CRC-32C (Castagnoli). Chosen over CRC-32/ISO because both aarch64 and
/// x86_64 carry it in hardware, which keeps the load-time integrity pass
/// at memcpy speed instead of dominating replay.
///
/// Large inputs are checksummed as three near-equal thirds in one
/// interleaved pass — the hardware CRC instruction is latency-bound on
/// its dependency chain, so three independent chains run ~3x faster —
/// and the record's stored CRC is the CRC of the three digests. The
/// split is purely length-derived, so writer and reader always agree.
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
unsafe fn crc32c_three_hw_aarch64(a: &[u8], b: &[u8], c: &[u8]) -> (u32, u32, u32) {
    use std::arch::aarch64::__crc32cd;
    let n = a.len().min(b.len()).min(c.len()) / 8;
    let (mut x, mut y, mut z) = (0xFFFF_FFFFu32, 0xFFFF_FFFFu32, 0xFFFF_FFFFu32);
    for i in 0..n {
        x = __crc32cd(x, u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap()));
        y = __crc32cd(y, u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()));
        z = __crc32cd(z, u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap()));
    }
    (
        crc32c_hw_aarch64_cont(!x, &a[n * 8..]),
        crc32c_hw_aarch64_cont(!y, &b[n * 8..]),
        crc32c_hw_aarch64_cont(!z, &c[n * 8..]),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_hw_aarch64_cont(seed: u32, data: &[u8]) -> u32 {
    use std::arch::aarch64::__crc32cb;
    let mut crc = !seed;
    for &v in data {
        crc = __crc32cb(crc, v);
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

/// One removal, captured at `swap_remove` time: the filler row's codes
/// (seq row bytes) and scale as they now sit in `slot`.
#[derive(Clone, Debug)]
pub(crate) struct PatchOp {
    pub slot: u64,
    pub codes: Vec<u8>,
    pub scale: f32,
}

/// Where a synced file left off — the index-side cursor that lets the
/// next `sync` append rather than rewrite.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SyncCursor {
    /// Generation of the file's last commit.
    pub gen: u64,
    /// Rows durably stored in whole-block segments (not the tail).
    pub segment_rows: u64,
    /// End of the last commit record — where the next sync appends.
    pub log_end: u64,
    /// The index's calibration generation at last sync; a mismatch
    /// forces a compaction, since a refit rewrites every stored code.
    pub calib_gen: u64,
}

/// Everything a sync needs from the index, layout-agnostic.
pub(crate) struct SyncSource<'a> {
    pub dim: usize,
    pub bit_width: usize,
    pub n_vectors: usize,
    /// Sequential-blocked codes for rows `[from, to)`, whole blocks
    /// only (`from % 32 == 0`, `to % 32 == 0`).
    pub seq_blocks: &'a dyn Fn(usize, usize) -> Vec<u8>,
    /// Row-major seq codes for a single row (the tail serializer).
    pub row_codes: &'a dyn Fn(usize) -> Vec<u8>,
    pub scales: &'a [f32],
    pub tqplus_shift: &'a [f32],
    pub tqplus_scale: &'a [f32],
    pub boundaries: &'a [f32],
    pub centroids: &'a [f32],
}

fn commit_body(src: &SyncSource<'_>, gen: u64, segment_rows: u64, data_end: u64) -> Vec<u8> {
    let n = src.n_vectors as u64;
    let n_tail = (n.saturating_sub(segment_rows)) as usize;
    let first_tail = src.n_vectors - n_tail;
    let mut body = Vec::with_capacity(64 + n_tail * (src.dim * src.bit_width / 8 + 4));
    body.extend_from_slice(&gen.to_le_bytes());
    body.extend_from_slice(&n.to_le_bytes());
    body.extend_from_slice(&(n_tail as u32).to_le_bytes());
    for r in first_tail..src.n_vectors {
        body.extend_from_slice(&(src.row_codes)(r));
    }
    for r in first_tail..src.n_vectors {
        body.extend_from_slice(&src.scales[r].to_le_bytes());
    }
    body.extend_from_slice(&data_end.to_le_bytes());
    body
}

fn commit_record(body: &[u8]) -> Vec<u8> {
    let mut rec = Vec::with_capacity(12 + body.len());
    rec.extend_from_slice(CMT);
    rec.extend_from_slice(&(body.len() as u32).to_le_bytes());
    rec.extend_from_slice(body);
    rec.extend_from_slice(&crc32(body).to_le_bytes());
    rec
}

fn superblock(src: &SyncSource<'_>) -> Vec<u8> {
    let mut sb = Vec::new();
    sb.extend_from_slice(V7_MAGIC);
    sb.push(V7_VERSION);
    sb.push(src.bit_width as u8);
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
    // The superblock is only ever written whole (create / compaction),
    // so one trailing CRC covers it: a flipped calibration or codebook
    // byte must refuse to load, not silently mis-score every query.
    let c = crc32(&sb);
    sb.extend_from_slice(&c.to_le_bytes());
    sb
}

fn fsync_if(f: &File, durable: bool) -> io::Result<()> {
    if durable {
        f.sync_data()?;
    }
    Ok(())
}

/// Create `path` fresh as a v7 file holding the whole index: superblock,
/// one segment for every whole block, one commit. This is also the
/// compaction: it goes through a sibling temp file + atomic rename, so
/// the previous file survives a crash (same protocol as the v6 writer).
pub(crate) fn write_full(
    path: &Path,
    src: &SyncSource<'_>,
    gen: u64,
    calib_gen: u64,
    durable: bool,
) -> io::Result<SyncCursor> {
    let tmp = path.with_extension(format!("v7tmp.{}", std::process::id()));
    let segment_rows = (src.n_vectors / 32) * 32;
    let mut image = superblock(src);
    if segment_rows > 0 {
        let codes = (src.seq_blocks)(0, segment_rows);
        let mut body = Vec::with_capacity(12 + codes.len() + segment_rows * 4);
        body.extend_from_slice(&0u64.to_le_bytes());
        body.extend_from_slice(&(segment_rows as u32).to_le_bytes());
        body.extend_from_slice(&codes);
        for r in 0..segment_rows {
            body.extend_from_slice(&src.scales[r].to_le_bytes());
        }
        image.extend_from_slice(SEG);
        let c = crc32(&body);
        image.extend_from_slice(&body);
        image.extend_from_slice(&c.to_le_bytes());
    }
    let data_end = image.len() as u64;
    let body = commit_body(src, gen, segment_rows as u64, data_end);
    image.extend_from_slice(&commit_record(&body));
    let log_end = image.len() as u64;
    {
        let mut f = File::create(&tmp)?;
        f.write_all(&image)?;
        f.flush()?;
        fsync_if(&f, durable)?;
    }
    std::fs::rename(&tmp, path)?;
    if durable {
        if let Some(parent) = path.parent() {
            if let Ok(d) = File::open(parent) {
                let _ = d.sync_all();
            }
        }
    }
    Ok(SyncCursor {
        gen,
        segment_rows: segment_rows as u64,
        log_end,
        calib_gen,
    })
}

/// Append one sync to an existing v7 file: the whole blocks completed
/// since `cursor.segment_rows`, every pending removal patch, then a
/// fresh commit. Two barriers: records are durable before the commit is
/// written, so a crash at any byte recovers the previous commit.
pub(crate) fn append_sync(
    path: &Path,
    src: &SyncSource<'_>,
    cursor: SyncCursor,
    patches: &[PatchOp],
    durable: bool,
) -> io::Result<SyncCursor> {
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    // Shed any torn garbage a previous crash left past the last commit.
    f.set_len(cursor.log_end)?;
    f.seek(SeekFrom::Start(cursor.log_end))?;

    let mut records = Vec::new();
    for p in patches {
        let mut body = Vec::with_capacity(12 + p.codes.len() + 4);
        body.extend_from_slice(&p.slot.to_le_bytes());
        body.extend_from_slice(&p.codes);
        body.extend_from_slice(&p.scale.to_le_bytes());
        records.extend_from_slice(PAT);
        let c = crc32(&body);
        records.extend_from_slice(&body);
        records.extend_from_slice(&c.to_le_bytes());
    }
    let new_segment_rows = (src.n_vectors / 32) * 32;
    if new_segment_rows > cursor.segment_rows as usize {
        let from = cursor.segment_rows as usize;
        let codes = (src.seq_blocks)(from, new_segment_rows);
        let mut body = Vec::with_capacity(12 + codes.len());
        body.extend_from_slice(&(from as u64).to_le_bytes());
        body.extend_from_slice(&((new_segment_rows - from) as u32).to_le_bytes());
        body.extend_from_slice(&codes);
        for r in from..new_segment_rows {
            body.extend_from_slice(&src.scales[r].to_le_bytes());
        }
        records.extend_from_slice(SEG);
        let c = crc32(&body);
        records.extend_from_slice(&body);
        records.extend_from_slice(&c.to_le_bytes());
    }
    f.write_all(&records)?;
    f.flush()?;
    fsync_if(&f, durable)?; // barrier 1: data durable

    let gen = cursor.gen + 1;
    let data_end = cursor.log_end + records.len() as u64;
    let segment_rows = new_segment_rows.max(cursor.segment_rows as usize) as u64;
    let body = commit_body(src, gen, segment_rows, data_end);
    let rec = commit_record(&body);
    f.seek(SeekFrom::Start(data_end))?;
    f.write_all(&rec)?;
    f.flush()?;
    fsync_if(&f, durable)?; // barrier 2: commit durable

    Ok(SyncCursor {
        gen,
        segment_rows,
        log_end: data_end + rec.len() as u64,
        calib_gen: cursor.calib_gen,
    })
}

/// Everything a v7 load yields, in the packed row layout plus the state
/// the cursor needs. The caller builds the index from it.
pub(crate) struct V7Load {
    pub dim: usize,
    pub bit_width: usize,
    pub n_vectors: usize,
    /// The sequential-blocked payload (whole 32-row blocks, zero-padded
    /// dead lanes in the final block) — the v6 code layout, one platform
    /// transform away from the search kernel's.
    pub seq_blocked: Vec<u8>,
    pub scales: Vec<f32>,
    pub tqplus_shift: Vec<f32>,
    pub tqplus_scale: Vec<f32>,
    pub cursor: SyncCursor,
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn read_u32(b: &[u8], at: usize) -> io::Result<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| bad("truncated field"))
}
fn read_u64(b: &[u8], at: usize) -> io::Result<u64> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| bad("truncated field"))
}

/// Load a v7 file: find the last valid commit, then replay the
/// committed record log in order. Corruption inside the committed area
/// is a loud error, never a silently wrong index.
pub(crate) fn load(path: &Path, expect_calib_gen: u64) -> io::Result<V7Load> {
    let mut raw = std::fs::read(path)?;
    if raw.len() < 10 || &raw[..4] != V7_MAGIC {
        return Err(bad("not a v7 file"));
    }
    if raw[4] != V7_VERSION {
        return Err(bad(format!("unsupported v7 revision {}", raw[4])));
    }
    let bit_width = raw[5] as usize;
    if !(2..=4).contains(&bit_width) {
        return Err(bad(format!("bit_width {bit_width} out of range")));
    }
    let dim = read_u32(&raw, 6)? as usize;
    if dim == 0 || dim % 8 != 0 || dim > crate::MAX_DIM {
        return Err(bad(format!("dim {dim} invalid")));
    }
    let n_levels = 1usize << bit_width;
    let sb_min = 10 + (n_levels - 1) * 4 + n_levels * 4 + 4;
    if raw.len() < sb_min {
        return Err(bad("truncated superblock"));
    }
    // Validate the embedded codebook against the canonical one, same as
    // the v6 loader (#320): a drifted codebook silently mis-scores.
    let (canon_b, canon_c) = crate::codebook::codebook(bit_width, dim);
    let mut off = 10;
    for want in canon_b.iter() {
        let got = f32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
        if got != *want {
            return Err(bad("embedded codebook boundaries drifted"));
        }
        off += 4;
    }
    for want in canon_c.iter() {
        let got = f32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
        if got != *want {
            return Err(bad("embedded codebook centroids drifted"));
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
        tqplus_shift.push(f32::from_le_bytes(
            raw[off + k * 4..off + k * 4 + 4].try_into().unwrap(),
        ));
    }
    off += n_calib * 4;
    for k in 0..n_calib {
        let v = f32::from_le_bytes(raw[off + k * 4..off + k * 4 + 4].try_into().unwrap());
        if !v.is_finite() || v <= 0.0 {
            return Err(bad("non-positive calibration scale"));
        }
        tqplus_scale.push(v);
    }
    off += n_calib * 4;
    let stored = read_u32(&raw, off)?;
    if crc32(&raw[..off]) != stored {
        return Err(bad("corrupt superblock (crc mismatch)"));
    }
    let sb_len = off + 4;
    let row_bytes = dim * bit_width / 8;

    // --- find the last valid commit ------------------------------------
    let mut pos = raw.len();
    let mut found: Option<(usize, usize)> = None; // (record start, body len)
    while pos > sb_len {
        let Some(i) = rfind(&raw[sb_len..pos], CMT).map(|k| k + sb_len) else {
            break;
        };
        if let Ok(blen) = read_u32(&raw, i + 4) {
            let blen = blen as usize;
            let end = i + 8 + blen + 4;
            if end <= raw.len() {
                let body = &raw[i + 8..i + 8 + blen];
                let stored = read_u32(&raw, i + 8 + blen)?;
                if crc32(body) == stored {
                    if let Ok(data_end) = read_u64(body, blen - 8) {
                        if data_end as usize <= i {
                            found = Some((i, blen));
                            break;
                        }
                    }
                }
            }
        }
        pos = i;
    }
    let Some((cmt_at, blen)) = found else {
        return Err(bad("no valid commit record — unrecoverable v7 file"));
    };
    let body = &raw[cmt_at + 8..cmt_at + 8 + blen];
    let gen = read_u64(body, 0)?;
    let n_vectors = read_u64(body, 8)? as usize;
    let n_tail = read_u32(body, 16)? as usize;
    if n_tail >= 32 {
        return Err(bad("commit tail holds a whole block"));
    }
    let mut off = 20;
    let tail_codes = body
        .get(off..off + n_tail * row_bytes)
        .ok_or_else(|| bad("truncated tail codes"))?
        .to_vec();
    off += n_tail * row_bytes;
    let mut tail_scales = Vec::with_capacity(n_tail);
    for k in 0..n_tail {
        tail_scales.push(f32::from_le_bytes(
            body[off + k * 4..off + k * 4 + 4].try_into().unwrap(),
        ));
    }
    off += n_tail * 4;
    let data_end = read_u64(body, off)? as usize;

    // --- positional replay of the committed area -----------------------
    // Segments already carry the seq-blocked layout, so replay copies
    // them verbatim at their block offset; a patch is one lane write.
    // No row-major intermediate, no repack afterwards.
    let block_bytes = row_bytes * 32;
    let mut seq_blocked: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    // A file's first record is almost always one segment holding every
    // whole block (fresh sync, compaction). Deferring its copy lets the
    // common case reuse `raw` itself as the code buffer — a memmove
    // over already-faulted pages instead of faulting in a second
    // buffer. (codes_at, codes_len) of the deferred leading segment:
    let mut pending: Option<(usize, usize)> = None;
    fn settle(seq: &mut Vec<u8>, raw: &[u8], pending: &mut Option<(usize, usize)>) {
        if let Some((at, len)) = pending.take() {
            seq.extend_from_slice(&raw[at..at + len]);
        }
    }
    let mut p = sb_len;
    while p < data_end {
        let tag = raw.get(p..p + 4).ok_or_else(|| bad("truncated record"))?;
        if tag == SEG {
            let start = read_u64(&raw, p + 4)? as usize;
            let nr = read_u32(&raw, p + 12)? as usize;
            if nr % 32 != 0 || start % 32 != 0 {
                return Err(bad("segment not whole blocks"));
            }
            let codes_len = nr * row_bytes;
            let codes_at = p + 16;
            let body_end = codes_at + codes_len + nr * 4;
            let stored = read_u32(&raw, body_end)?;
            let bodyb = raw
                .get(p + 4..body_end)
                .ok_or_else(|| bad("truncated segment"))?;
            if crc32(bodyb) != stored {
                return Err(bad("corrupt committed segment (crc mismatch)"));
            }
            if start > scales.len() {
                return Err(bad("segment leaves a slot gap"));
            }
            if start == 0 && scales.is_empty() {
                pending = Some((codes_at, codes_len));
            } else if start == scales.len() {
                // Append path: grow without a zero-fill pass.
                settle(&mut seq_blocked, &raw, &mut pending);
                seq_blocked.extend_from_slice(&raw[codes_at..codes_at + codes_len]);
            } else {
                settle(&mut seq_blocked, &raw, &mut pending);
                if scales.len() < start + nr {
                    seq_blocked.resize((start + nr) / 32 * block_bytes, 0);
                }
                seq_blocked[start / 32 * block_bytes..(start + nr) / 32 * block_bytes]
                    .copy_from_slice(&raw[codes_at..codes_at + codes_len]);
            }
            if scales.len() < start + nr {
                scales.resize(start + nr, f32::NAN);
            }
            let sc = &raw[codes_at + codes_len..body_end];
            for k in 0..nr {
                let v = f32::from_le_bytes(sc[k * 4..k * 4 + 4].try_into().unwrap());
                if !v.is_finite() || v < 0.0 {
                    return Err(bad("invalid per-vector scale in segment"));
                }
                scales[start + k] = v;
            }
            p = body_end + 4;
        } else if tag == PAT {
            let body_end = p + 4 + 8 + row_bytes + 4;
            let stored = read_u32(&raw, body_end)?;
            let bodyb = raw
                .get(p + 4..body_end)
                .ok_or_else(|| bad("truncated patch"))?;
            if crc32(bodyb) != stored {
                return Err(bad("corrupt committed patch (crc mismatch)"));
            }
            let slot = read_u64(&raw, p + 4)? as usize;
            if slot >= scales.len() {
                return Err(bad("patch slot out of range"));
            }
            settle(&mut seq_blocked, &raw, &mut pending);
            let (b, lane) = (slot / 32, slot % 32);
            for g in 0..row_bytes {
                seq_blocked[b * block_bytes + g * 32 + lane] = raw[p + 12 + g];
            }
            scales[slot] = f32::from_le_bytes(
                raw[p + 12 + row_bytes..p + 16 + row_bytes].try_into().unwrap(),
            );
            p = body_end + 4;
        } else if tag == CMT {
            let blen2 = read_u32(&raw, p + 4)? as usize;
            let body_end = p + 8 + blen2;
            let stored = read_u32(&raw, body_end)?;
            let bodyb = raw
                .get(p + 8..body_end)
                .ok_or_else(|| bad("truncated superseded commit"))?;
            if crc32(bodyb) != stored {
                return Err(bad("corrupt superseded commit (crc mismatch)"));
            }
            p = body_end + 4;
        } else {
            return Err(bad(format!("unknown record tag {tag:?}")));
        }
    }
    // The commit's n governs. Live segment rows keep, dead tail rows
    // drop; then the commit's tail rows land as lane writes, and every
    // dead lane in the final block is zeroed (the blocked layout's
    // determinism invariant).
    let live_segment_rows = n_vectors.saturating_sub(n_tail);
    if scales.len() < live_segment_rows {
        return Err(bad(format!(
            "commit declares {live_segment_rows} segment rows but the log wrote {}",
            scales.len()
        )));
    }
    scales.truncate(live_segment_rows);
    if scales.iter().any(|v| v.is_nan()) {
        return Err(bad("a live slot was never written by any record"));
    }
    let segment_rows = scales.len() as u64;
    let total_blocks = n_vectors.div_ceil(32);
    if let Some((at, len)) = pending.take() {
        // The whole committed code area is one leading segment: shift it
        // to the front of the read buffer and adopt the buffer.
        raw.copy_within(at..at + len, 0);
        raw.truncate(len);
        seq_blocked = raw;
    }
    seq_blocked.resize(total_blocks * block_bytes, 0);
    for (k, sc) in tail_scales.iter().enumerate() {
        let slot = live_segment_rows + k;
        let (b, lane) = (slot / 32, slot % 32);
        for g in 0..row_bytes {
            seq_blocked[b * block_bytes + g * 32 + lane] =
                tail_codes[k * row_bytes + g];
        }
        scales.push(*sc);
    }
    for slot in n_vectors..total_blocks * 32 {
        let (b, lane) = (slot / 32, slot % 32);
        for g in 0..row_bytes {
            seq_blocked[b * block_bytes + g * 32 + lane] = 0;
        }
    }
    Ok(V7Load {
        dim,
        bit_width,
        n_vectors,
        seq_blocked,
        scales,
        tqplus_shift,
        tqplus_scale,
        cursor: SyncCursor {
            gen,
            segment_rows,
            log_end: (cmt_at + 12 + blen) as u64,
            calib_gen: expect_calib_gen,
        },
    })
}

fn rfind(haystack: &[u8], needle: &[u8; 4]) -> Option<usize> {
    if haystack.len() < 4 {
        return None;
    }
    (0..=haystack.len() - 4).rev().find(|&i| &haystack[i..i + 4] == needle)
}

/// Sniff: is this a v7 file?
pub(crate) fn is_v7(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|_| &magic == V7_MAGIC)
        .unwrap_or(false)
}
