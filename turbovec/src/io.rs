//! Read/write TurboVec index files.
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
//! Both formats are at version 3 as of turbovec 0.6.x (TQ+ per-coord
//! calibration). Version 2 (turbovec 0.4.4 .. 0.6.0) is loaded transparently
//! with empty calibration — the index behaves like the old encoding, with
//! no recall change and no TQ+ gain. Re-encoding from source vectors picks
//! up the new calibration. Version 1 (turbovec ≤ 0.4.3) is incompatible
//! and refused with a rebuild hint.
//!
//! Version 1 `.tv` files had no magic — the file started with a bare
//! bit_width byte (2/3/4). Version 2+ prepends magic + version, which
//! lets us detect either a current file or "looks like a v1 turbovec
//! file" cleanly.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const TV_MAGIC: &[u8; 4] = b"TVPI";
const TV_VERSION: u8 = 3;
const TVIM_MAGIC: &[u8; 4] = b"TVIM";
const TVIM_VERSION: u8 = 3;

const REBUILD_HINT: &str =
    "Rebuild this index from the source vectors using turbovec 0.4.4 or later \
     (no in-place migration is provided; the format version 2 changes the meaning \
     of the per-vector scalar from ||v|| to a length-renormalization correction).";

/// Core payload — what a fully-deserialized index needs.
type CoreLoad = (usize, usize, usize, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>);

/// `.tv` write — positional index.
pub fn write(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TV_MAGIC)?;
    f.write_all(&[TV_VERSION])?;
    write_core(
        &mut f, bit_width, dim, n_vectors, packed_codes, scales,
        tqplus_shift, tqplus_scale,
    )?;
    f.flush()?;
    Ok(())
}

/// `.tv` load — positional index. Transparently handles v2 (no TQ+) and
/// v3 (with TQ+) files; v2 returns empty TQ+ vectors which the engine
/// treats as identity calibration.
pub fn load(path: impl AsRef<Path>) -> io::Result<CoreLoad> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TV_MAGIC {
        // Version 1 .tv files had no magic — first byte was the bit_width
        // (always 2, 3, or 4). If we see one of those as the first byte,
        // emit a targeted error rather than the generic "wrong magic"
        // message; otherwise treat it as a non-turbovec file.
        if (2..=4).contains(&magic[0]) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "this .tv file was written by turbovec ≤ 0.4.3 (format \
                     version 1). It is incompatible with turbovec 0.4.4+ \
                     because the per-vector scalar's meaning changed. {}",
                    REBUILD_HINT,
                ),
            ));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a turbovec .tv file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    read_core_versioned(&mut f, version[0], TV_VERSION, ".tv")
}

/// `.tvim` write — positional index plus the id-map side-tables.
pub fn write_id_map(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
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

    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVIM_MAGIC)?;
    f.write_all(&[TVIM_VERSION])?;
    write_core(
        &mut f, bit_width, dim, n_vectors, packed_codes, scales,
        tqplus_shift, tqplus_scale,
    )?;

    for &id in slot_to_id {
        f.write_all(&id.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

/// `.tvim` load — positional index plus the id-map side-tables.
pub fn load_id_map(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u64>)> {
    let mut f = BufReader::new(File::open(path)?);

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
    if version[0] == 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "this .tvim file was written by turbovec ≤ 0.4.3 (format \
                 version 1). It is incompatible with turbovec 0.4.4+ \
                 because the per-vector scalar's meaning changed. {}",
                REBUILD_HINT,
            ),
        ));
    }
    let (bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale) =
        read_core_versioned(&mut f, version[0], TVIM_VERSION, ".tvim")?;

    // Read the slot_to_id table via the capped reader rather than
    // `Vec::with_capacity(n_vectors)` — `n_vectors` is attacker-controlled and
    // pre-reserving it allows a tiny file to drive a huge allocation.
    let id_bytes = n_vectors
        .checked_mul(8)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "id table size overflows usize"))?;
    let raw = read_exact_vec(&mut f, id_bytes)?;
    let slot_to_id: Vec<u64> = raw
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();

    Ok((
        bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale,
        slot_to_id,
    ))
}

const CORE_HEADER_SIZE: usize = 9;

/// Core header + packed codes + per-vector scales + TQ+ calibration —
/// shared by `.tv` and `.tvim`.
fn write_core<W: Write>(
    w: &mut W,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    w.write_all(&[bit_width as u8])?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&(n_vectors as u32).to_le_bytes())?;
    w.write_all(packed_codes)?;
    for &s in scales {
        w.write_all(&s.to_le_bytes())?;
    }
    // TQ+ trailer. n_calib == 0 means identity calibration (lazy index
    // with no add yet, or a loaded pre-TQ+ index that's been resaved);
    // otherwise must equal dim.
    assert!(
        tqplus_shift.len() == tqplus_scale.len()
            && (tqplus_shift.is_empty() || tqplus_shift.len() == dim),
        "TQ+ shift/scale must have equal length and either be empty or equal dim"
    );
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

/// Read the core payload, dispatching on the version byte. Knows about
/// v2 (no TQ+) and v3 (with TQ+); anything else errors.
fn read_core_versioned<R: Read>(
    r: &mut R,
    version: u8,
    expected: u8,
    label: &str,
) -> io::Result<CoreLoad> {
    match version {
        2 => read_core_v2(r),
        3 => read_core_v3(r),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported {label} format version: {version} (this build \
                 supports versions 2 and {expected})",
            ),
        )),
    }
}

/// v2: header + codes + scales. Returns empty TQ+ vectors (identity calibration).
fn read_core_v2<R: Read>(r: &mut R) -> io::Result<CoreLoad> {
    let (bit_width, dim, n_vectors, packed_codes, scales) = read_header_codes_scales(r)?;
    Ok((bit_width, dim, n_vectors, packed_codes, scales, Vec::new(), Vec::new()))
}

/// v3: header + codes + scales + TQ+ trailer.
fn read_core_v3<R: Read>(r: &mut R) -> io::Result<CoreLoad> {
    let (bit_width, dim, n_vectors, packed_codes, scales) = read_header_codes_scales(r)?;

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

    Ok((bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale))
}

fn read_header_codes_scales<R: Read>(
    r: &mut R,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>)> {
    let mut header = [0u8; CORE_HEADER_SIZE];
    r.read_exact(&mut header)?;
    let bit_width = header[0] as usize;
    let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let n_vectors = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;

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
        // Bound the lazily-built dim×dim rotation matrix: a tiny file can
        // declare a huge dim and drive a multi-GB allocation on first search.
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid dim {dim}: exceeds maximum {}", crate::MAX_DIM),
        ));
    }

    // Checked arithmetic: `dim`/`n_vectors` are attacker-controlled u32s, so
    // the product can overflow `usize` (on 32-bit targets this wrap would
    // yield an undersized buffer and later out-of-bounds reads).
    let packed_bytes = (dim / 8)
        .checked_mul(bit_width)
        .and_then(|x| x.checked_mul(n_vectors))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "packed code size overflows usize")
        })?;
    let packed_codes = read_exact_vec(r, packed_bytes)?;

    let scales = read_f32_array(r, n_vectors)?;
    Ok((bit_width, dim, n_vectors, packed_codes, scales))
}

/// Read exactly `n` bytes without pre-allocating `n` up front. A malicious
/// header can declare a multi-gigabyte length from a tiny file; `read_to_end`
/// on a `take`-limited reader grows the buffer only to the bytes actually
/// present, so we never reserve the attacker's claimed size before confirming
/// the data exists. The length check then rejects a truncated file cleanly.
fn read_exact_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
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
