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
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
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
    /// Row-major seq codes, one `dim*bits/8` row per vector.
    pub seq_rows: Vec<u8>,
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
    let mut raw = Vec::new();
    File::open(path)?.read_to_end(&mut raw)?;
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
        .ok_or_else(|| bad("truncated tail codes"))?;
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
    let mut seq_rows: Vec<u8> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
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
            if scales.len() < start + nr {
                seq_rows.resize((start + nr) * row_bytes, 0);
                scales.resize(start + nr, f32::NAN);
            }
            // seq-blocked -> row-major within each 32-row block.
            let block_bytes = row_bytes * 32;
            for b in 0..nr / 32 {
                let blk = &raw[codes_at + b * block_bytes..codes_at + (b + 1) * block_bytes];
                let dst = &mut seq_rows
                    [(start + b * 32) * row_bytes..(start + (b + 1) * 32) * row_bytes];
                crate::pack::seq_block_to_rows(blk, row_bytes, dst);
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
            seq_rows[slot * row_bytes..(slot + 1) * row_bytes]
                .copy_from_slice(&raw[p + 12..p + 12 + row_bytes]);
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
    // The commit's n governs: rows the log wrote past it are dead
    // (shrink below the segment watermark), and every live slot must
    // have been written by some record.
    let live_segment_rows = n_vectors.saturating_sub(n_tail);
    if scales.len() < live_segment_rows {
        return Err(bad(format!(
            "commit declares {live_segment_rows} segment rows but the log wrote {}",
            scales.len()
        )));
    }
    seq_rows.truncate(live_segment_rows * row_bytes);
    scales.truncate(live_segment_rows);
    if scales.iter().any(|v| v.is_nan()) {
        return Err(bad("a live slot was never written by any record"));
    }
    let segment_rows = scales.len() as u64;
    seq_rows.extend_from_slice(tail_codes);
    scales.extend_from_slice(&tail_scales);
    Ok(V7Load {
        dim,
        bit_width,
        n_vectors,
        seq_rows,
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
