//! Dump the rotation matrix and Lloyd-Max codebook for a given (dim, bits) to
//! a raw little-endian f32 file an external tool can load with np.fromfile.
//!
//! Useful for cross-checking the Rust encode pipeline against a reference
//! implementation byte-for-byte at the parameter level — feed the dumped
//! rotation and codebook into the other pipeline and compare outputs.
//!
//! Layout per (dim, bits) tuple, written to `<out>/state_d{dim}_b{bits}.bin`:
//!   - rotation matrix:   dim * dim   f32, row-major
//!   - boundaries:        2^bits - 1  f32
//!   - centroids:         2^bits      f32

use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};

use turbovec::codebook::codebook;
use turbovec::rotation::make_rotation_matrix;

fn write_f32s<W: Write>(w: &mut W, vals: &[f32]) -> std::io::Result<()> {
    for &v in vals {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: dump_state <out_dir>");
        std::process::exit(1);
    }
    let out_dir = &args[1];
    create_dir_all(out_dir).unwrap();

    let configs = [(200usize, 2usize), (200, 4), (1536, 2), (1536, 4), (3072, 2), (3072, 4)];

    for &(dim, bits) in &configs {
        eprintln!("Dumping d={} bits={}", dim, bits);
        let rotation = make_rotation_matrix(dim);
        let (boundaries, centroids) = codebook(bits, dim);

        assert_eq!(rotation.len(), dim * dim);
        assert_eq!(boundaries.len(), (1 << bits) - 1);
        assert_eq!(centroids.len(), 1 << bits);

        let path = format!("{}/state_d{}_b{}.bin", out_dir, dim, bits);
        let mut w = BufWriter::new(File::create(&path).unwrap());
        write_f32s(&mut w, &rotation).unwrap();
        write_f32s(&mut w, &boundaries).unwrap();
        write_f32s(&mut w, &centroids).unwrap();
        w.flush().unwrap();
        eprintln!("  -> {}", path);
    }
}
