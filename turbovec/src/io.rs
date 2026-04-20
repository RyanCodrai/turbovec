//! Read/write TurboVec index files.
//!
//! Two formats live here:
//! * `.tv` — [`TurboQuantIndex`](crate::TurboQuantIndex) — 9-byte header
//!   + packed codes + norms.
//! * `.tvim` — [`IdMapIndex`](crate::IdMapIndex) — 4-byte magic "TVIM"
//!   + version + the same core-index payload + a trailing `slot_to_id`
//!   table of `u64` values.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const TV_HEADER_SIZE: usize = 9;
const TVIM_MAGIC: &[u8; 4] = b"TVIM";
const TVIM_VERSION: u8 = 1;

/// `.tv` write — positional index.
pub fn write(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    norms: &[f32],
) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    write_core(&mut f, bit_width, dim, n_vectors, packed_codes, norms)?;
    f.flush()?;
    Ok(())
}

/// `.tv` load — positional index.
pub fn load(path: impl AsRef<Path>) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>)> {
    let mut f = BufReader::new(File::open(path)?);
    read_core(&mut f)
}

/// `.tvim` write — positional index plus the id-map side-tables.
pub fn write_id_map(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    norms: &[f32],
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
    write_core(&mut f, bit_width, dim, n_vectors, packed_codes, norms)?;

    for &id in slot_to_id {
        f.write_all(&id.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

/// `.tvim` load — positional index plus the id-map side-tables.
pub fn load_id_map(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>, Vec<u64>)> {
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
    if version[0] != TVIM_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported TVIM version: {}", version[0]),
        ));
    }

    let (bit_width, dim, n_vectors, packed_codes, norms) = read_core(&mut f)?;

    let mut slot_to_id = Vec::with_capacity(n_vectors);
    let mut buf = [0u8; 8];
    for _ in 0..n_vectors {
        f.read_exact(&mut buf)?;
        slot_to_id.push(u64::from_le_bytes(buf));
    }

    Ok((bit_width, dim, n_vectors, packed_codes, norms, slot_to_id))
}

/// Core header + packed codes + norms — shared by `.tv` and `.tvim`.
fn write_core<W: Write>(
    w: &mut W,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    norms: &[f32],
) -> io::Result<()> {
    w.write_all(&[bit_width as u8])?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&(n_vectors as u32).to_le_bytes())?;
    w.write_all(packed_codes)?;
    for &n in norms {
        w.write_all(&n.to_le_bytes())?;
    }
    Ok(())
}

fn read_core<R: Read>(r: &mut R) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>)> {
    let mut header = [0u8; TV_HEADER_SIZE];
    r.read_exact(&mut header)?;

    let bit_width = header[0] as usize;
    let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let n_vectors = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;

    let packed_bytes = (dim / 8) * bit_width * n_vectors;
    let mut packed_codes = vec![0u8; packed_bytes];
    r.read_exact(&mut packed_codes)?;

    let mut norms_bytes = vec![0u8; n_vectors * 4];
    r.read_exact(&mut norms_bytes)?;
    let norms: Vec<f32> = norms_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok((bit_width, dim, n_vectors, packed_codes, norms))
}
