//! Core I/O hardening tests.
//!
//! Covers:
//! 1. Atomic save — a failed or panicking `write`/`write_id_map` must leave
//!    a pre-existing index file at the destination intact, and successful
//!    writes must not leave stray temp files behind (#118).
//! 2. Count-field width — the v4 format's `n_vectors` field is u64, so
//!    counts above u32::MAX serialize exactly instead of wrapping or
//!    erroring (#119).
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

use turbovec::io::{load, load_id_map, write, write_id_map, CodePayload};

/// v6 payload length: sequential blocked layout, padded to 32-vector blocks.

/// The native-layout bytes the fast-path loader returns for stored
/// sequential-blocked codes: the x86 perm0 nibble interleave (mirrored
/// from pack.rs — stable, format-documented math), identity elsewhere.
fn expected_native(seq: &[u8]) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        const PERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
        let mut out = vec![0u8; seq.len()];
        for (s, o) in seq.chunks_exact(32).zip(out.chunks_exact_mut(32)) {
            for j in 0..16 {
                let ba = s[PERM0[j]];
                let bb = s[PERM0[j] + 16];
                o[j] = (ba >> 4) | (bb & 0xF0);
                o[16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    seq.to_vec()
}

fn test_codebook(bit_width: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    // The v6 loader verifies the embedded codebook against the canonical
    // codebook(bit_width, dim) (#320), so fixture files must embed the
    // real one, not a synthetic monotone stand-in.
    turbovec::expected_codebook(bit_width, dim)
}

fn blocked_len(bit_width: usize, dim: usize, n_vectors: usize) -> usize {
    let codes_per_byte = 8 / bit_width;
    n_vectors.div_ceil(32) * 32 * (dim / codes_per_byte)
}
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
    let packed = vec![0xABu8; blocked_len(4, 32, 2)];
    let scales = vec![1.5f32, 2.5];
    let cb = test_codebook(4, 32);
    write(path, 4, 32, 2, &packed, &cb.0, &cb.1, &scales, &[], &[]).unwrap();
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
        write(&path, 4, 32, 2, &packed, &test_codebook(4, 32).0, &test_codebook(4, 32).1, &scales, &[1.0; 3], &[1.0; 3])
    }));
    assert!(result.is_err(), "mismatched TQ+ lengths should panic");

    let (bw, d, n, p, s, _, _) = load(&path).expect("previous good index must survive");
    assert_eq!((bw, d, n), (4, 32, 2));
    assert_eq!(
        p,
        CodePayload::BlockedNative {
            codes: expected_native(&packed),
            boundaries: test_codebook(4, 32).0,
            centroids: test_codebook(4, 32).1,
        }
    );
    assert_eq!(s, scales);
    assert_eq!(dir_entries(&dir), vec!["index.tv"], "no partial/temp files may remain");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tvim_panicking_write_leaves_previous_file_intact() {
    let dir = temp_dir("tvim-panic-write");
    let path = dir.join("index.tvim");
    let packed = vec![0x55u8; blocked_len(2, 16, 2)];
    let scales = vec![0.5f32, 1.0];
    let ids = vec![7u64, 9];
    let cb = test_codebook(2, 16);
    write_id_map(&path, 2, 16, 2, &packed, &cb.0, &cb.1, &scales, &[], &[], &ids).unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        write_id_map(&path, 2, 16, 2, &packed, &test_codebook(2, 16).0, &test_codebook(2, 16).1, &scales, &[1.0; 3], &[1.0; 3], &ids)
    }));
    assert!(result.is_err(), "mismatched TQ+ lengths should panic");

    let (bw, d, n, p, s, _, _, slot_to_id) =
        load_id_map(&path).expect("previous good index must survive");
    assert_eq!((bw, d, n), (2, 16, 2));
    assert_eq!(
        p,
        CodePayload::BlockedNative {
            codes: expected_native(&packed),
            boundaries: test_codebook(2, 16).0,
            centroids: test_codebook(2, 16).1,
        }
    );
    assert_eq!(s, scales);
    assert_eq!(slot_to_id, ids);
    assert_eq!(dir_entries(&dir), vec!["index.tvim"], "no partial/temp files may remain");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tv_successful_overwrite_leaves_no_temp_files() {
    let dir = temp_dir("tv-overwrite");
    let path = dir.join("index.tv");
    write_good_tv(&path);

    let packed = vec![0xCDu8; blocked_len(4, 32, 3)];
    let scales = vec![1.0f32, 2.0, 3.0];
    let cb = test_codebook(4, 32);
    write(&path, 4, 32, 3, &packed, &cb.0, &cb.1, &scales, &[], &[]).unwrap();

    let (_, _, n, p, s, _, _) = load(&path).unwrap();
    assert_eq!(n, 3);
    assert_eq!(
        p,
        CodePayload::BlockedNative {
            codes: expected_native(&packed),
            boundaries: test_codebook(4, 32).0,
            centroids: test_codebook(4, 32).1,
        }
    );
    assert_eq!(s, scales);
    assert_eq!(dir_entries(&dir), vec!["index.tv"], "no temp files may remain");
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #119 — n_vectors count-field width
// ---------------------------------------------------------------------------

#[cfg(target_pointer_width = "64")]
#[test]
fn tv_write_stores_n_vectors_over_u32_max_exactly() {
    // A ≥2^32-vector index can't be built in a test, but the field width
    // can be verified at the byte level: the raw writer accepts the
    // count and the header must contain the exact u64 — not `n mod
    // 2^32` (the pre-v4 silent wrap) and not an error (the v3-era u32
    // ceiling, lifted by the v4 u64 field).
    let dir = temp_dir("tv-n-u64");
    let path = dir.join("index.tv");

    let n = (1usize << 32) + 2;
    let cb = test_codebook(2, 8);
    write(&path, 2, 8, n, &[], &cb.0, &cb.1, &[], &[], &[])
        .expect("v4 write must accept n_vectors over u32::MAX");
    let bytes = std::fs::read(&path).unwrap();
    // Layout: magic(4) + version(1) + bit_width(1) + dim(4) + n_vectors(8).
    let stored = u64::from_le_bytes(bytes[10..18].try_into().unwrap());
    assert_eq!(stored, n as u64, "header must store the exact 64-bit count");
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
    let cb = test_codebook(4, 8);
    write(&path, 4, 8, 1, &[0x12u8; 128], &cb.0, &cb.1, scales, tqplus_shift, tqplus_scale)
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
    let cb = test_codebook(4, 8);
    write_id_map(&path, 4, 8, 1, &[0x12u8; 128], &cb.0, &cb.1, &[1.0], &[0.0; 8], &[0.0; 8], &[42])
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
    let cb = test_codebook(4, EMPTY_DIM);
    write(&path, 4, EMPTY_DIM, 0, &[], &cb.0, &cb.1, &[], &[], &[]).unwrap();
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

/// Fast-durability writes keep the atomic protocol: byte-identical
/// output to durable writes and no temp strays — only the fsync is
/// skipped. (Failure-path preservation is covered by the panicking-
/// write tests above.)
#[test]
fn fast_durability_write_is_atomic_and_byte_identical() {
    use turbovec::io::Durability;
    let dir = temp_dir("fast-durability");
    let p_durable = dir.join("d.tv");
    let p_fast = dir.join("f.tv");
    let mut idx = TurboQuantIndex::new(32, 4).unwrap();
    let v: Vec<f32> = (0..64 * 32).map(|i| (i % 97) as f32 / 97.0 - 0.5).collect();
    idx.add(&v);
    idx.write(&p_durable).unwrap();
    idx.write_with_durability(&p_fast, Durability::Fast).unwrap();
    assert_eq!(
        std::fs::read(&p_durable).unwrap(),
        std::fs::read(&p_fast).unwrap(),
        "fast writes must be byte-identical to durable writes",
    );
    // Overwrite with fast mode; loads fine, no temp strays.
    idx.write_with_durability(&p_fast, Durability::Fast).unwrap();
    TurboQuantIndex::load(&p_fast).unwrap();
    assert!(
        dir_entries(&dir).iter().all(|n| !n.contains(".tmp.")),
        "no temp strays after fast writes",
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The parallel-pwrite branch of the x86 path writer engages only for
/// codes >= 8 MiB; every other test uses tiny indexes and exercises the
/// serial fallback. This covers the branch that runs on real saves:
/// large synthetic sections, both durability modes, byte parity with
/// the streamed writer, and a successful load.
#[test]
fn large_payload_write_matches_streamed_writer_in_both_durability_modes() {
    use turbovec::io::{self, Durability};
    let dir = temp_dir("large-parallel-write");
    let (bit_width, dim, n_vectors) = (4usize, 768usize, 90_000usize);
    let blocked = {
        let len = n_vectors.div_ceil(32) * 32 * (dim / 2);
        let mut s = 0x00D1_CEu64;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect::<Vec<u8>>()
    };
    let cb = test_codebook(bit_width, dim);
    let scales = vec![1.0f32; n_vectors];

    let mut streamed = Vec::new();
    io::write_to(&mut streamed, bit_width, dim, n_vectors, &blocked, &cb.0, &cb.1, &scales, &[], &[])
        .unwrap();

    for (name, durability) in [("durable", Durability::Durable), ("fast", Durability::Fast)] {
        let p = dir.join(format!("{name}.tv"));
        io::write_with_durability(
            &p, bit_width, dim, n_vectors, &blocked, &cb.0, &cb.1, &scales, &[], &[], durability,
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&p).unwrap(),
            streamed,
            "{name}: parallel path writer must be byte-identical to the streamed writer",
        );
        io::load(&p).unwrap();
    }
    assert!(
        dir_entries(&dir).iter().all(|n| !n.contains(".tmp.")),
        "no temp strays",
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #293 — symlink attack on the temp name
// ---------------------------------------------------------------------------

/// The issue's repro: an attacker with write access to the destination
/// directory pre-plants symlinks at predictable `<dest>.tmp.{pid}.{seq}`
/// names pointing at a victim file. The save must neither write through
/// a planted link nor leave the destination as a symlink.
#[cfg(unix)]
#[test]
fn planted_tmp_symlinks_cannot_redirect_the_write() {
    let dir = temp_dir("symlink-attack");
    let victim_dir = temp_dir("symlink-victim");
    let victim = victim_dir.join("precious.txt");
    std::fs::write(&victim, b"precious").unwrap();

    let path = dir.join("index.tv");
    // Cover the old fully-predictable pattern for a generous seq range,
    // plus the destination itself.
    let pid = std::process::id();
    for seq in 0..512 {
        std::os::unix::fs::symlink(&victim, dir.join(format!("index.tv.tmp.{pid}.{seq}"))).unwrap();
    }
    write_good_tv(&path);

    assert_eq!(std::fs::read(&victim).unwrap(), b"precious", "victim must be untouched");
    let meta = std::fs::symlink_metadata(&path).unwrap();
    assert!(meta.file_type().is_file(), "destination must be a regular file, not a symlink");
    load(&path).expect("saved index must load");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&victim_dir).ok();
}

// ---------------------------------------------------------------------------
// #299 — NAME_MAX: long destination filenames must still save
// ---------------------------------------------------------------------------

/// A 250-byte destination filename is legal on ext4/APFS, but the temp
/// suffix used to push the sibling past NAME_MAX and fail the save. The
/// temp name's base is truncated to fit instead.
#[test]
fn long_destination_filename_saves_and_loads() {
    let dir = temp_dir("name-max");
    let name = format!("{}.tv", "v".repeat(247)); // 250-byte filename
    let path = dir.join(&name);
    let (packed, scales) = write_good_tv(&path);

    let (bw, d, n, p, s, _, _) = load(&path).expect("long-named index must load");
    assert_eq!((bw, d, n), (4, 32, 2));
    assert_eq!(
        p,
        CodePayload::BlockedNative {
            codes: expected_native(&packed),
            boundaries: test_codebook(4, 32).0,
            centroids: test_codebook(4, 32).1,
        }
    );
    assert_eq!(s, scales);
    assert_eq!(dir_entries(&dir), vec![name], "no temp strays");
    std::fs::remove_dir_all(&dir).ok();
}
