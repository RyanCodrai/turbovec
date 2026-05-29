//! Format-versioning tests for `.tv` and `.tvim`.
//!
//! Verifies:
//! 1. Round-trip via the public write/load functions works on the current
//!    format (version 2) for both file types.
//! 2. A hand-constructed version-1 `.tv` file (bare bit_width-first
//!    header, no magic) is rejected with the upgrade-hint error.
//! 3. A hand-constructed version-1 `.tvim` file (TVIM magic with
//!    version byte 1) is rejected with the upgrade-hint error.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use turbovec::io::{load, load_id_map, write, write_id_map};

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-{}-{}", nonce, name));
    p
}

#[test]
fn tv_round_trip_current_format() {
    let path = temp_path("v2.tv");
    let bit_width = 4;
    let dim = 32;
    let n_vectors = 3;
    let packed = vec![0xABu8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.5f32, 2.5, 3.5];

    // Round-trip with empty TQ+ calibration (identity); behaviour identical
    // to a v2 file otherwise. Separate test below covers populated calibration.
    write(&path, bit_width, dim, n_vectors, &packed, &scales, &[], &[]).unwrap();
    let (bw, d, n, p, s, shift, scale_tq) = load(&path).unwrap();

    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert!(shift.is_empty());
    assert!(scale_tq.is_empty());
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_round_trip_with_tqplus_calibration() {
    let path = temp_path("v3-tqplus.tv");
    let bit_width = 4;
    let dim = 32;
    let n_vectors = 3;
    let packed = vec![0xABu8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.5f32, 2.5, 3.5];
    let shift: Vec<f32> = (0..dim).map(|d| d as f32 * 0.01).collect();
    let scale_tq: Vec<f32> = (0..dim).map(|d| 1.0 + d as f32 * 0.02).collect();

    write(&path, bit_width, dim, n_vectors, &packed, &scales, &shift, &scale_tq).unwrap();
    let (bw, d, n, p, s, loaded_shift, loaded_scale) = load(&path).unwrap();

    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert_eq!(loaded_shift, shift);
    assert_eq!(loaded_scale, scale_tq);
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_v1_file_is_rejected_with_upgrade_hint() {
    // Hand-construct a turbovec ≤ 0.4.3 `.tv` file: bare header
    // (bit_width=4, dim=32, n_vectors=2), packed codes, two f32 norms.
    let path = temp_path("v1.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(32u32).to_le_bytes()).unwrap(); // dim
        f.write_all(&(2u32).to_le_bytes()).unwrap(); // n_vectors
        f.write_all(&vec![0u8; (32 / 8) * 4 * 2]).unwrap(); // packed codes
        f.write_all(&(1.0f32).to_le_bytes()).unwrap(); // norm 0
        f.write_all(&(2.0f32).to_le_bytes()).unwrap(); // norm 1
    }

    let err = load(&path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("turbovec ≤ 0.4.3") && msg.contains("Rebuild"),
        "expected upgrade hint, got: {}",
        msg
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tvim_round_trip_current_format() {
    let path = temp_path("v2.tvim");
    let bit_width = 2;
    let dim = 16;
    let n_vectors = 4;
    let packed = vec![0x55u8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![0.5f32, 1.0, 1.5, 2.0];
    let ids = vec![100u64, 200, 300, 400];

    write_id_map(&path, bit_width, dim, n_vectors, &packed, &scales, &[], &[], &ids).unwrap();
    let (bw, d, n, p, s, shift, scale_tq, slot_to_id) = load_id_map(&path).unwrap();

    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert!(shift.is_empty());
    assert!(scale_tq.is_empty());
    assert_eq!(slot_to_id, ids);
    std::fs::remove_file(&path).ok();
}

#[test]
fn tvim_v1_file_is_rejected_with_upgrade_hint() {
    // Hand-construct a turbovec ≤ 0.4.3 `.tvim` file: TVIM magic, version
    // byte = 1, then the same v1 core layout.
    let path = temp_path("v1.tvim");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVIM").unwrap();
        f.write_all(&[1u8]).unwrap(); // version
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(32u32).to_le_bytes()).unwrap(); // dim
        f.write_all(&(1u32).to_le_bytes()).unwrap(); // n_vectors
        f.write_all(&vec![0u8; (32 / 8) * 4]).unwrap(); // packed codes
        f.write_all(&(1.0f32).to_le_bytes()).unwrap(); // norm
        f.write_all(&(42u64).to_le_bytes()).unwrap(); // id
    }

    let err = load_id_map(&path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("turbovec ≤ 0.4.3") && msg.contains("Rebuild"),
        "expected upgrade hint, got: {}",
        msg
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_garbage_file_rejected_without_upgrade_hint() {
    let path = temp_path("garbage.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"NOPE").unwrap();
        f.write_all(&[0u8; 32]).unwrap();
    }
    let err = load(&path).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("wrong magic"), "expected wrong-magic error, got: {}", msg);
    assert!(
        !msg.contains("turbovec ≤ 0.4.3"),
        "should not suggest upgrade for garbage: {}",
        msg
    );
    std::fs::remove_file(&path).ok();
}
