//! Convert an index file between every format turbovec has written.
//!
//! The rest of the crate reads and writes v7 only. This module is the
//! one place that still understands v5 and v6, and it exists so an index
//! written by an older build can be brought forward — or, for a rollback
//! or a bug report against an older reader, taken back.
//!
//! Both directions between all three versions are supported, for both
//! `.tv` (positional) and `.tvim` (id-mapped) files, so an input in any
//! version can be written out in any version.
//!
//! # The formats
//!
//! All three start `TVPI` or `TVIM` (four bytes) then a version byte.
//!
//! * **v5** — `bit_width` u8, `dim` u32, `n_vectors` u64, then the
//!   bit-plane *packed* rows, then one f32 scale per row, then the TQ+
//!   trailer (`n_calib` u32, then `n_calib` shifts and `n_calib`
//!   scales). No codebook: it is derived from `(bit_width, dim)`.
//! * **v6** — the same header, then the codebook (`2^bits - 1`
//!   boundaries and `2^bits` centroids) so a load can verify it has not
//!   drifted, then the codes in the arch-neutral *sequential blocked*
//!   layout (padded to whole 32-row blocks), then scales, then the same
//!   trailer.
//! * **v7** — the container `sync` maintains: a superblock, two
//!   alternating header slots, then whole block units. Written here
//!   through the shipping writer rather than by hand, so a converted
//!   file is byte-identical to one this build would have produced.
//!
//! `.tvim` appends the id table — one u64 per row — after the core in
//! v5 and v6; in v7 the ids ride inside the block units.
//!
//! # What is preserved
//!
//! The codes, the per-row scales, the TQ+ calibration and the ids. The
//! quantized bytes are never re-encoded — converting is a re-container,
//! not a re-quantize, so a round-trip through any version returns the
//! same search results.
//!
//! What cannot survive a conversion *down*: v7's incremental state. A
//! v5 or v6 file is a flat snapshot with no commit history, so the
//! generation, the pending redo ops and the file's claim are dropped.
//! Converting back up produces an unclaimed snapshot, exactly as
//! [`crate::TurboQuantIndex::write`] does.
//!
//! The lazy sentinel *does* survive in both directions: all three
//! versions spell "no dimension committed yet" as `dim == 0` with no
//! rows, and the release before v7 wrote exactly that for a store saved
//! before its first add.

use std::io;
use std::path::Path;

use crate::{codebook, pack, BLOCK};

/// A format version of a turbovec index file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    /// Packed rows, no embedded codebook. Read-only in every shipped
    /// build since v6; this module can also write it.
    V5,
    /// Sequential-blocked codes with an embedded codebook.
    V6,
    /// The sync container. What this build reads and writes natively.
    V7,
}

impl Version {
    fn byte(self) -> u8 {
        match self {
            Version::V5 => 5,
            Version::V6 => 6,
            Version::V7 => 1, // v7's own revision byte; the magic distinguishes it
        }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Version::V5 => "v5",
            Version::V6 => "v6",
            Version::V7 => "v7",
        })
    }
}

/// Which index type a file holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `.tv` — positional, results are slot indices.
    Plain,
    /// `.tvim` — carries an external id per row.
    IdMapped,
}

/// The version-neutral contents of an index file.
///
/// Codes are always the canonical bit-plane *packed* rows here, whatever
/// the file stored, so every writer starts from one representation.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub bit_width: usize,
    /// `0` for a lazy index that never committed a dimension (only legal
    /// with `n_vectors == 0`, and only expressible in v6 and v7).
    pub dim: usize,
    pub n_vectors: usize,
    pub packed_codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub tqplus_shift: Vec<f32>,
    pub tqplus_scale: Vec<f32>,
    /// Present exactly when the file is `.tvim`.
    pub ids: Option<Vec<u64>>,
}

impl Image {
    fn kind(&self) -> Kind {
        if self.ids.is_some() {
            Kind::IdMapped
        } else {
            Kind::Plain
        }
    }
}

const TV_MAGIC: &[u8; 4] = b"TVPI";
const TVIM_MAGIC: &[u8; 4] = b"TVIM";
const V5_HEADER: usize = 13;

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// The version and kind of `bytes`, without decoding it.
pub fn detect(bytes: &[u8]) -> io::Result<(Version, Kind)> {
    if bytes.len() < 5 {
        return Err(bad("too short to be an index file"));
    }
    if &bytes[0..4] == crate::io_v7::V7_MAGIC {
        // v7 carries the kind in the superblock rather than the magic.
        let kind = match bytes.get(6) {
            Some(0) => Kind::Plain,
            Some(1) => Kind::IdMapped,
            other => return Err(bad(format!("unknown v7 index kind {other:?}"))),
        };
        return Ok((Version::V7, kind));
    }
    let kind = if &bytes[0..4] == TV_MAGIC {
        Kind::Plain
    } else if &bytes[0..4] == TVIM_MAGIC {
        Kind::IdMapped
    } else {
        return Err(bad("not a turbovec index (unrecognised magic)"));
    };
    match bytes[4] {
        5 => Ok((Version::V5, kind)),
        6 => Ok((Version::V6, kind)),
        1 => Err(bad(
            "version 1 (turbovec <= 0.4.3) was already refused by the build \
             that introduced version 2; it cannot be decoded and must be \
             rebuilt from the source vectors",
        )),
        v @ 2..=4 => Err(bad(format!(
            "version {v} stores codes encoded under the pre-v5 rotation — a QR \
             of a seeded Gaussian, built with a BLAS this crate no longer \
             depends on and which differed by ~1 ulp across CPU architectures \
             and thread counts (the reason v5 replaced it, and the reason v4 \
             carries a rotation fingerprint at all). Those codes cannot be \
             re-containered into v5+: they would have to be dequantized, \
             inverse-rotated under a rotation this build cannot reliably \
             reproduce, re-rotated and re-quantized, which loses accuracy and \
             is not guaranteed to be the rotation that wrote them. Rebuild \
             from the source vectors instead"
        ))),
        v => Err(bad(format!("unknown index format version {v}"))),
    }
}

// ---------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------

fn rd_u32(b: &[u8], at: usize) -> io::Result<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().expect("4 bytes")))
        .ok_or_else(|| bad("truncated file"))
}

fn rd_u64(b: &[u8], at: usize) -> io::Result<u64> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes")))
        .ok_or_else(|| bad("truncated file"))
}

fn rd_f32s(b: &[u8], at: usize, n: usize) -> io::Result<Vec<f32>> {
    let end = at.checked_add(n * 4).ok_or_else(|| bad("length overflow"))?;
    let s = b.get(at..end).ok_or_else(|| bad("truncated file"))?;
    Ok(s.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect())
}

/// Rows per packed row, and the sequential-blocked payload length.
fn geometry(bit_width: usize, dim: usize, n: usize) -> (usize, usize) {
    let packed_row = dim * bit_width / 8;
    let blocked = n.div_ceil(BLOCK) * BLOCK * (dim / (8 / bit_width));
    (packed_row, blocked)
}

/// Decode any supported file into the neutral [`Image`].
pub fn read(bytes: &[u8]) -> io::Result<Image> {
    let (version, kind) = detect(bytes)?;
    match version {
        Version::V7 => read_v7(bytes, kind),
        Version::V5 | Version::V6 => read_legacy(bytes, version, kind),
    }
}

fn read_v7(bytes: &[u8], kind: Kind) -> io::Result<Image> {
    let expect_kind = if kind == Kind::IdMapped { 1 } else { 0 };
    let mut l = crate::io_v7::load_image(bytes.to_vec(), 0, expect_kind, "the image")?;
    let ids = (kind == Kind::IdMapped).then(|| std::mem::take(&mut l.ids));
    // The units hold the sequential-blocked layout; the packed rows are
    // the neutral form every writer here starts from.
    let packed_codes = if l.n_vectors == 0 {
        Vec::new()
    } else {
        let (_, nbg, _) = pack::blocked_geometry(l.n_vectors, l.bit_width, l.dim);
        let _ = nbg;
        pack::seq_to_packed(&l.seq_blocked, l.n_vectors, l.bit_width, l.dim)
    };
    Ok(Image {
        bit_width: l.bit_width,
        dim: l.dim,
        n_vectors: l.n_vectors,
        packed_codes,
        scales: l.scales,
        tqplus_shift: l.tqplus_shift,
        tqplus_scale: l.tqplus_scale,
        ids,
    })
}

fn read_legacy(bytes: &[u8], version: Version, kind: Kind) -> io::Result<Image> {
    let mut at = 5;
    let hdr = bytes
        .get(at..at + V5_HEADER)
        .ok_or_else(|| bad("truncated header"))?;
    let bit_width = hdr[0] as usize;
    let dim = u32::from_le_bytes(hdr[1..5].try_into().expect("4 bytes")) as usize;
    let n_vectors = usize::try_from(u64::from_le_bytes(
        hdr[5..13].try_into().expect("8 bytes"),
    ))
    .map_err(|_| bad("n_vectors does not fit this platform's usize"))?;
    at += V5_HEADER;

    if !(2..=4).contains(&bit_width) {
        return Err(bad(format!("invalid bit_width {bit_width}")));
    }
    // `dim == 0` is the lazy sentinel — an index that never committed a
    // dimension — and v5/v6 carry it exactly as v7 does, valid only with
    // no rows. Files like that exist: the previous release's `write()`
    // emitted one for a store saved before its first add.
    if dim == 0 {
        if n_vectors != 0 {
            return Err(bad(format!(
                "dim 0 with {n_vectors} rows: no dimension committed"
            )));
        }
    } else if !dim.is_multiple_of(8) || dim > crate::MAX_DIM {
        return Err(bad(format!("invalid dim {dim}")));
    }
    // `n_vectors` is an untrusted u64 straight off the header, and every
    // size below multiplies it. Bound it by what the file could hold
    // before any of that arithmetic runs: each row costs at least one
    // byte of codes and four of scale, so a file this small cannot
    // describe that many rows however its header reads.
    if n_vectors.saturating_mul(5) > bytes.len() {
        return Err(bad(format!(
            "header claims {n_vectors} rows, which a {}-byte file cannot hold",
            bytes.len()
        )));
    }

    let (packed_row, blocked_len) = geometry(bit_width, dim, n_vectors);

    let packed_codes = match version {
        Version::V5 => {
            let n = packed_row * n_vectors;
            let s = bytes.get(at..at + n).ok_or_else(|| bad("truncated codes"))?;
            at += n;
            s.to_vec()
        }
        Version::V6 => {
            // The embedded codebook is skipped: it is a function of
            // (bit_width, dim) and every writer here re-derives it, so
            // carrying the file's copy forward would only let a drifted
            // one propagate.
            let n_levels = 1usize << bit_width;
            at += (2 * n_levels - 1) * 4;
            let s = bytes
                .get(at..at + blocked_len)
                .ok_or_else(|| bad("truncated codes"))?;
            at += blocked_len;
            if n_vectors == 0 {
                Vec::new()
            } else {
                pack::seq_to_packed(s, n_vectors, bit_width, dim)
            }
        }
        Version::V7 => unreachable!("handled by read_v7"),
    };

    let scales = rd_f32s(bytes, at, n_vectors)?;
    at += n_vectors * 4;
    let n_calib = rd_u32(bytes, at)? as usize;
    at += 4;
    if n_calib != 0 && n_calib != dim {
        return Err(bad(format!("calibration length {n_calib} != dim {dim}")));
    }
    let tqplus_shift = rd_f32s(bytes, at, n_calib)?;
    at += n_calib * 4;
    let tqplus_scale = rd_f32s(bytes, at, n_calib)?;
    at += n_calib * 4;

    let ids = match kind {
        Kind::Plain => None,
        Kind::IdMapped => {
            let mut v = Vec::with_capacity(n_vectors);
            for i in 0..n_vectors {
                v.push(rd_u64(bytes, at + i * 8)?);
            }
            Some(v)
        }
    };

    Ok(Image {
        bit_width,
        dim,
        n_vectors,
        packed_codes,
        scales,
        tqplus_shift,
        tqplus_scale,
        ids,
    })
}

// ---------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------

/// Encode an [`Image`] in `version`. The kind follows the image's ids.
pub fn write(image: &Image, version: Version) -> io::Result<Vec<u8>> {
    if image.scales.len() != image.n_vectors {
        return Err(bad(format!(
            "{} scales for {} rows",
            image.scales.len(),
            image.n_vectors
        )));
    }
    if let Some(ids) = &image.ids {
        if ids.len() != image.n_vectors {
            return Err(bad(format!(
                "{} ids for {} rows",
                ids.len(),
                image.n_vectors
            )));
        }
    }
    match version {
        Version::V7 => write_v7(image),
        Version::V5 | Version::V6 => write_legacy(image, version),
    }
}

fn write_v7(image: &Image) -> io::Result<Vec<u8>> {
    // Through the shipping writer, so a converted file is byte-identical
    // to one this build would have produced from the same index.
    let dim = (image.dim != 0).then_some(image.dim);
    let inner = crate::TurboQuantIndex::from_parts(
        dim,
        image.bit_width,
        image.n_vectors,
        image.packed_codes.clone(),
        image.scales.clone(),
        image.tqplus_shift.clone(),
        image.tqplus_scale.clone(),
    )
    .map_err(|e| bad(e.to_string()))?;
    match &image.ids {
        None => Ok(inner.to_bytes()),
        Some(ids) => {
            let m = crate::IdMapIndex::from_index_and_ids(inner, ids.clone())
                .map_err(|e| bad(e.to_string()))?;
            Ok(m.to_bytes())
        }
    }
}

fn write_legacy(image: &Image, version: Version) -> io::Result<Vec<u8>> {
    if image.dim == 0 && image.n_vectors != 0 {
        return Err(bad(format!(
            "{version}: dim 0 is the lazy sentinel and cannot carry \
             {} rows",
            image.n_vectors
        )));
    }
    let (_, blocked_len) = geometry(image.bit_width, image.dim, image.n_vectors);
    let mut out = Vec::new();
    out.extend_from_slice(match image.kind() {
        Kind::Plain => TV_MAGIC,
        Kind::IdMapped => TVIM_MAGIC,
    });
    out.push(version.byte());
    out.push(image.bit_width as u8);
    out.extend_from_slice(&(image.dim as u32).to_le_bytes());
    out.extend_from_slice(&(image.n_vectors as u64).to_le_bytes());

    if version == Version::V6 {
        // Re-derived, never copied from the source file: the codebook is
        // a pure function of (bit_width, dim), and a v6 load verifies it
        // against exactly this. The lazy sentinel has no dimension to
        // solve for, so it embeds zeros — which is what the previous
        // release wrote, and what its loader skips validating.
        let n_levels = 1usize << image.bit_width;
        let (boundaries, centroids) = if image.dim == 0 {
            (vec![0.0f32; n_levels - 1], vec![0.0f32; n_levels])
        } else {
            codebook::codebook(image.bit_width, image.dim)
        };
        for v in boundaries.iter().chain(centroids.iter()) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        if image.n_vectors == 0 {
            out.extend_from_slice(&vec![0u8; blocked_len]);
        } else {
            let (blocked, _) = pack::repack(
                &image.packed_codes,
                image.n_vectors,
                image.bit_width,
                image.dim,
            );
            let (_, nbg, _) =
                pack::blocked_geometry(image.n_vectors, image.bit_width, image.dim);
            let seq = pack::native_to_seq(&blocked, image.bit_width, nbg);
            debug_assert_eq!(seq.len(), blocked_len);
            out.extend_from_slice(&seq);
        }
    } else {
        out.extend_from_slice(&image.packed_codes);
    }

    for &s in &image.scales {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out.extend_from_slice(&(image.tqplus_shift.len() as u32).to_le_bytes());
    for v in image.tqplus_shift.iter().chain(image.tqplus_scale.iter()) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    if let Some(ids) = &image.ids {
        for &id in ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------

/// Read `src`, and write it to `dst` in `to`.
///
/// `dst` is written through a temp file and renamed, so a failure leaves
/// any previous file at that path intact.
pub fn convert_file(src: &Path, dst: &Path, to: Version) -> io::Result<()> {
    let bytes = std::fs::read(src)?;
    let image = read(&bytes)?;
    let out = write(&image, to)?;
    let (file, tmp) = crate::io::create_tmp(dst)?;
    let result = (|| {
        use std::io::Write as _;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, &file);
        w.write_all(&out)?;
        w.flush()?;
        drop(w);
        file.sync_all()
    })();
    if let Err(e) = result.and_then(|()| {
        drop(file);
        crate::io::rename_atomic(&tmp, dst)
    }) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    crate::io::sync_parent_dir_after_commit(dst);
    Ok(())
}

/// The version of the file at `path`.
pub fn version_of(path: &Path) -> io::Result<(Version, Kind)> {
    let mut head = [0u8; 8];
    let f = std::fs::File::open(path)?;
    crate::io::read_exact_at(&f, &mut head, 0)?;
    detect(&head)
}
