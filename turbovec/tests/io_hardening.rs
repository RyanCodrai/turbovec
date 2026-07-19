//! Core I/O hardening tests.
//!
//! Covers:
//! 1. Atomic save — a failed or panicking `write`/`write_id_map` must leave
//!    a pre-existing index file at the destination intact, and successful
//!    writes must not leave stray temp files behind (#118).
//! 2. Write-side bounds — `n_vectors` larger than the format's u32 count
//!    field is rejected instead of silently wrapping (#119).
//! 3. Load-side value validation — non-finite or out-of-range floats in
//!    the per-vector scales or TQ+ calibration arrays are rejected at
//!    load instead of poisoning search results (#122).
//! 4. Empty-index search/prepare short-circuit — no dim×dim rotation
//!    build when there is nothing to score (#123).

use std::fs::File;
use std::io::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use turbovec::io::{load, load_id_map, write, write_id_map};
use turbovec::TurboQuantIndex;

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-{}-{}", nonce, name));
    std::fs::create_dir(&p).unwrap();
    p
}

/// Write a small valid index to `path` and return its parts for later
/// comparison.
fn write_good_tv(path: &PathBuf) -> (Vec<u8>, Vec<f32>) {
    let packed = vec![0xABu8; (32 / 8) * 4 * 2];
    let scales = vec![1.5f32, 2.5];
    write(path, 4, 32, 2, &packed, &scales, &[], &[]).unwrap();
    (packed, scales)
}

fn dir_entries(dir: &PathBuf) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// #118 — atomic save
// ---------------------------------------------------------------------------

#[test]
fn tv_panicking_write_leaves_previous_file_intact() {
    let dir = temp_dir("tv-panic-write");
    let path = dir.join("index.tv");
    let (packed, scales) = write_good_tv(&path);

    // TQ+ calibration length invariant violated (len 3 != dim 32): the
    // write must panic BEFORE creating or truncating anything at `path`.
    let result = catch_unwind(AssertUnwindSafe(|| {
        write(&path, 4, 32, 2, &packed, &scales, &[1.0; 3], &[1.0; 3])
    }));
    assert!(result.is_err(), "mismatched TQ+ lengths should panic");

    let (bw, d, n, p, s, _, _) = load(&path).expect("previous good index must survive");
    assert_eq!((bw, d, n), (4, 32, 2));
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert_eq!(dir_entries(&dir), vec!["index.tv"], "no partial/temp files may remain");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tvim_panicking_write_leaves_previous_file_intact() {
    let dir = temp_dir("tvim-panic-write");
    let path = dir.join("index.tvim");
    let packed = vec![0x55u8; (16 / 8) * 2 * 2];
    let scales = vec![0.5f32, 1.0];
    let ids = vec![7u64, 9];
    write_id_map(&path, 2, 16, 2, &packed, &scales, &[], &[], &ids).unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        write_id_map(&path, 2, 16, 2, &packed, &scales, &[1.0; 3], &[1.0; 3], &ids)
    }));
    assert!(result.is_err(), "mismatched TQ+ lengths should panic");

    let (bw, d, n, p, s, _, _, slot_to_id) =
        load_id_map(&path).expect("previous good index must survive");
    assert_eq!((bw, d, n), (2, 16, 2));
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert_eq!(slot_to_id, ids);
    assert_eq!(dir_entries(&dir), vec!["index.tvim"], "no partial/temp files may remain");
    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(target_pointer_width = "64")]
#[test]
fn tv_erroring_write_leaves_previous_file_intact_and_no_temp() {
    let dir = temp_dir("tv-error-write");
    let path = dir.join("index.tv");
    let (packed, scales) = write_good_tv(&path);

    // n_vectors over the u32 field errors mid-write (#119); the
    // destination must be untouched and the temp file cleaned up.
    let err = write(&path, 2, 8, (u32::MAX as usize) + 2, &[], &[], &[], &[])
        .expect_err("oversized n_vectors must error, not silently wrap");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

    let (bw, d, n, p, s, _, _) = load(&path).expect("previous good index must survive");
    assert_eq!((bw, d, n), (4, 32, 2));
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert_eq!(dir_entries(&dir), vec!["index.tv"], "no partial/temp files may remain");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tv_successful_overwrite_leaves_no_temp_files() {
    let dir = temp_dir("tv-overwrite");
    let path = dir.join("index.tv");
    write_good_tv(&path);

    let packed = vec![0xCDu8; (32 / 8) * 4 * 3];
    let scales = vec![1.0f32, 2.0, 3.0];
    write(&path, 4, 32, 3, &packed, &scales, &[], &[]).unwrap();

    let (_, _, n, p, s, _, _) = load(&path).unwrap();
    assert_eq!(n, 3);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert_eq!(dir_entries(&dir), vec!["index.tv"], "no temp files may remain");
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #119 — write-side n_vectors bounds
// ---------------------------------------------------------------------------

#[cfg(target_pointer_width = "64")]
#[test]
fn tv_write_rejects_n_vectors_over_u32_max() {
    let dir = temp_dir("tv-n-overflow");
    let path = dir.join("index.tv");

    let err = write(&path, 2, 8, (1usize << 32) + 2, &[], &[], &[], &[])
        .expect_err("n_vectors >= 2^32 must not silently truncate to u32");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let msg = err.to_string();
    assert!(
        msg.contains("n_vectors") && msg.contains("4294967298"),
        "error should name the offending count, got: {msg}",
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #122 — load-side float value validation
// ---------------------------------------------------------------------------

/// Craft a dim=8, n=1 v3 file through the raw writer (which does not
/// value-validate) and expect `load` to reject it.
fn expect_load_rejects(
    name: &str,
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    expect_in_msg: &str,
) {
    let dir = temp_dir(name);
    let path = dir.join("bad.tv");
    write(&path, 4, 8, 1, &[0x12, 0x34, 0x56, 0x78], scales, tqplus_shift, tqplus_scale)
        .unwrap();
    let err = load(&path).expect_err("bad float payload must be rejected at load");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains(expect_in_msg),
        "expected {expect_in_msg:?} in error, got: {err}",
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_rejects_bad_per_vector_scales() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        expect_load_rejects("scales-bad", &[bad], &[0.0; 8], &[1.0; 8], "scale");
    }
}

#[test]
fn load_rejects_nonfinite_tqplus_shift() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut shift = [0.0f32; 8];
        shift[3] = bad;
        expect_load_rejects("shift-bad", &[1.0], &shift, &[1.0; 8], "shift");
    }
}

#[test]
fn load_rejects_nonpositive_or_nonfinite_tqplus_scale() {
    // Zero and negative are finite — a bare is_finite() check would pass
    // them; they must still be rejected (division by tqplus_scale).
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        let mut scale = [1.0f32; 8];
        scale[5] = bad;
        expect_load_rejects("tqscale-bad", &[1.0], &[0.0; 8], &scale, "scale");
    }
}

#[test]
fn load_id_map_rejects_zero_tqplus_scale() {
    // Same core reader as .tv — one case to pin the shared path.
    let dir = temp_dir("tvim-tqscale-zero");
    let path = dir.join("bad.tvim");
    write_id_map(&path, 4, 8, 1, &[0x12, 0x34, 0x56, 0x78], &[1.0], &[0.0; 8], &[0.0; 8], &[42])
        .unwrap();
    let err = load_id_map(&path).expect_err("zero tqplus_scale must be rejected at load");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_v2_rejects_nonfinite_scales() {
    // Hand-construct a v2 file (no TQ+ trailer) with a NaN per-vector
    // scale — the v2 read path must apply the same value validation.
    let dir = temp_dir("v2-nan-scale");
    let path = dir.join("bad-v2.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVPI").unwrap();
        f.write_all(&[2u8]).unwrap(); // version 2
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&8u32.to_le_bytes()).unwrap(); // dim
        f.write_all(&1u32.to_le_bytes()).unwrap(); // n_vectors
        f.write_all(&[0xAA; 4]).unwrap(); // packed codes
        f.write_all(&f32::NAN.to_le_bytes()).unwrap(); // scale = NaN
    }
    let err = load(&path).expect_err("NaN per-vector scale must be rejected at load");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #123 — empty-index search/prepare short-circuit
// ---------------------------------------------------------------------------

const EMPTY_DIM: usize = 8192;
// Building the 8192x8192 rotation takes multiple seconds and ~0.75 GiB;
// the short-circuit path is sub-millisecond, so 2s gives wide CI margin
// while still failing decisively if the rotation build comes back.
const EMPTY_BUDGET: Duration = Duration::from_secs(2);

fn load_empty_large_dim(dir: &PathBuf) -> TurboQuantIndex {
    let path = dir.join("empty.tv");
    write(&path, 4, EMPTY_DIM, 0, &[], &[], &[], &[]).unwrap();
    TurboQuantIndex::load(&path).unwrap()
}

#[test]
fn empty_index_search_returns_empty_without_rotation_build() {
    let dir = temp_dir("empty-search");
    let idx = load_empty_large_dim(&dir);
    assert_eq!(idx.len(), 0);

    let query = vec![0.5f32; EMPTY_DIM];
    let start = Instant::now();
    let res = idx.search(&query, 5);
    let elapsed = start.elapsed();

    assert_eq!(res.nq, 1);
    assert_eq!(res.k, 0);
    assert!(res.scores.is_empty());
    assert!(res.indices.is_empty());
    assert!(
        elapsed < EMPTY_BUDGET,
        "search on an empty index took {elapsed:?} — it must not build the dim×dim rotation",
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn empty_index_prepare_skips_rotation_build() {
    let dir = temp_dir("empty-prepare");
    let idx = load_empty_large_dim(&dir);

    let start = Instant::now();
    idx.prepare();
    let elapsed = start.elapsed();
    assert!(
        elapsed < EMPTY_BUDGET,
        "prepare on an empty index took {elapsed:?} — it must not build the dim×dim rotation",
    );
    std::fs::remove_dir_all(&dir).ok();
}
