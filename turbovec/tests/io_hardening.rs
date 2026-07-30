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
//! 5. Write-time codes/scales length validation — the raw `io::write*`
//!    entry points reject a code or scale buffer inconsistent with the
//!    header instead of writing a file that loads clean and silently
//!    mis-scores (#407).

use std::fs::File;
use std::io::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use turbovec::io::{load, load_id_map, write, write_id_map, write_to, CodePayload};

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

// The byte-level check that the v4 header stores `n_vectors` as an exact
// u64 lives in `io::codes_scales_validation_tests` — a ≥2^32-vector
// header can only be emitted by the private core writer now that the
// public writers require code/scale buffers matching the count (#407),
// and 2^32 buffer entries cannot be allocated in a test.

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

// ---------------------------------------------------------------------------
// #299 — the stale-temp sweep is actually wired into the save paths
// ---------------------------------------------------------------------------

/// `sweep_stale_tmps` has good unit coverage in `io.rs`, but every one of
/// those tests calls it directly. Deleting the `sweep_stale_tmps(path)`
/// call from `write_atomic` / `write_atomic_parallel` — the whole fix —
/// leaves all of them green: the sweep would still be correct, just never
/// invoked. This drives it the way a user does, through `write`.
///
/// The planted leftovers mimic what a SIGKILL between temp creation and
/// rename leaves behind: a full-size `<dest>.tmp.{pid}.{seq}.{rand}`
/// sibling nothing else ever removes.
#[test]
fn a_real_save_sweeps_a_leaked_temp_sibling() {
    use std::time::{Duration, SystemTime};
    let dir = temp_dir("sweep-on-save");
    let dest = dir.join("index.tv");

    // Two hours old — past the sweep's one-hour staleness bar, which
    // exists so a live writer's in-flight temp is never touched.
    let old = SystemTime::now() - Duration::from_secs(2 * 60 * 60);
    let plant = |name: &str, age: Option<SystemTime>| {
        let p = dir.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(&[0u8; 4096]).unwrap();
        if let Some(t) = age {
            f.set_times(std::fs::FileTimes::new().set_modified(t)).unwrap();
        }
        p
    };
    let leaked = plant("index.tv.tmp.4242.0.deadbeef", Some(old));
    let leaked_legacy = plant("index.tv.tmp.4242.1", Some(old));
    let in_flight = plant("index.tv.tmp.4242.2.deadbeef", None);
    let other_dest = plant("other.tv.tmp.4242.0.deadbeef", Some(old));
    let unrelated = plant("index.tv.tmp.notes", Some(old));

    let mut idx = turbovec::TurboQuantIndex::new(32, 4).unwrap();
    let v: Vec<f32> = (0..64 * 32).map(|i| (i % 97) as f32 / 97.0 - 0.5).collect();
    idx.add(&v);
    idx.write(&dest).unwrap();

    assert!(!leaked.exists(), "save did not sweep the leaked temp sibling");
    assert!(!leaked_legacy.exists(), "save did not sweep the legacy-pattern leftover");
    assert!(in_flight.exists(), "save swept a fresh temp — a live writer's could be next");
    assert!(other_dest.exists(), "save swept another destination's temp");
    assert!(unrelated.exists(), "save swept a name it did not create");
    // And the save itself worked.
    turbovec::TurboQuantIndex::load(&dest).unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #365 — a committed save is never reported as a failure
// ---------------------------------------------------------------------------
//
// The rename is the commit point. If the parent-directory fsync that
// follows it fails, the new file is already the one readers see, so
// returning `Err` tells the caller its previous file survived when it did
// not — and integrations with a rollback policy then act on that. The
// durability shortfall is real and is warned about on stderr; the save
// itself must report success.

/// Make `dir` un-openable for reading (write+execute only), which is what
/// `sync_parent_dir`'s `File::open(dir)` needs, while still allowing the
/// temp-file create and the rename. Returns false when the sabotage does
/// not take effect (running as root), so the test can skip rather than
/// assert something it did not actually provoke.
#[cfg(unix)]
fn block_dir_fsync(dir: &PathBuf) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o300)).unwrap();
    File::open(dir).is_err()
}

#[cfg(unix)]
#[test]
fn durable_write_reports_success_when_only_the_parent_dir_fsync_fails() {
    use std::os::unix::fs::PermissionsExt;

    let dir = temp_dir("post-rename-fsync");
    let path = dir.join("index.tv");
    write_good_tv(&path);

    // A visibly different payload, so the assertion below distinguishes
    // "the new file committed" from "the old file survived".
    let new_packed = vec![0x11u8; blocked_len(4, 32, 2)];
    let new_scales = vec![3.5f32, 4.5];
    let cb = test_codebook(4, 32);

    if !block_dir_fsync(&dir) {
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        return;
    }

    let result = write(
        &path, 4, 32, 2, &new_packed, &cb.0, &cb.1, &new_scales, &[], &[],
    );
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();

    result.expect("a save whose rename committed must not be reported as a failure");

    let (_, _, _, p, s, _, _) = load(&path).expect("the committed file must load");
    assert_eq!(s, new_scales, "the destination must hold the new index");
    assert_eq!(
        p,
        CodePayload::BlockedNative {
            codes: expected_native(&new_packed),
            boundaries: cb.0,
            centroids: cb.1,
        }
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #407 — write-time codes/scales length validation
// ---------------------------------------------------------------------------
//
// The raw `io::write*` entry points are the documented low-level API, and
// they used to write whatever `codes_blocked_seq` / `scales` buffers they
// were handed. Both sections are sized from the header on load, so an
// inconsistent buffer does not produce a rejected file — it produces an
// undefined one. Measured on the fixture below (16 vectors, dim 64,
// bit_width 4: 1024 code bytes, 16 scales), before this validation
// existed:
//
//   control                 -> load Ok, top score 0.997532
//   codes -4 B              -> load Ok, top score 1.012978
//   codes -8 B              -> load Ok, top score 1.001488
//   codes -16 B             -> load Ok, top score 1.007280   <- above 1.0
//   codes -24 B             -> load Ok, top score 1.020110
//   codes -64 B             -> load Ok, top score 0.000000
//   scales -2               -> load Ok, top score 0.997532   (2 rows unscaled)
//   scales empty            -> load Ok, top score 0.000000
//   codes -8 B, scales +2   -> load Ok, top score 1.001488   (byte count preserved)
//   codes -1 B / -2 B / ... -> load Err (data-dependent luck)
//
// Nine of sixteen perturbations loaded clean and mis-scored, and the
// `-16 B` case returned a cosine of 1.0073 — above the ceiling, so the
// file was not merely inaccurate but self-evidently impossible, and
// nothing rejected it. The writer is the last point at which the caller
// can still be told, and it is what `from_parts` already does.

/// 16 vectors, dim 64, bit_width 4 — the geometry the #407 sweep used.
/// Rows are unit-normalized so search scores are true cosines and the
/// 1.0 ceiling is meaningful.
fn v407_index() -> (TurboQuantIndex, Vec<f32>) {
    let (dim, bit_width, n) = (64usize, 4usize, 16usize);
    let mut s = 0x2024_0407u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let mut data: Vec<f32> = (0..n * dim).map(|_| rnd()).collect();
    for row in data.chunks_exact_mut(dim) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    let mut idx = TurboQuantIndex::new(dim, bit_width).unwrap();
    idx.add(&data);
    let query = data[..dim].to_vec();
    (idx, query)
}

/// `(codes, scales, boundaries, centroids, tqplus_shift, tqplus_scale)`
/// — a self-consistent parts tuple for [`v407_index`].
type V407Parts = (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn v407_parts(idx: &TurboQuantIndex) -> V407Parts {
    let (boundaries, centroids) = idx.codebook_for_write();
    (
        idx.codes_blocked_seq(),
        idx.scales().to_vec(),
        boundaries,
        centroids,
        idx.tqplus_shift().to_vec(),
        idx.tqplus_scale().to_vec(),
    )
}

/// The fixture really is the geometry the messages below name — if this
/// drifts, every `should_panic` expectation in this section is testing
/// the wrong number.
#[test]
fn v407_fixture_geometry_is_1024_codes_and_16_scales() {
    let (idx, _) = v407_index();
    let (codes, scales, ..) = v407_parts(&idx);
    assert_eq!((codes.len(), scales.len(), idx.len()), (1024, 16, 16));
}

// --- codes-length rejection, one test per public entry point ---------------

#[test]
#[should_panic(expected = "codes_blocked_seq length 1008 does not match the blocked layout \
                           for n_vectors 16, bit_width 4, dim 64 (1024 bytes)")]
fn write_rejects_short_codes_buffer() {
    let dir = temp_dir("v407-write-short-codes");
    let (idx, _) = v407_index();
    let (codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    // -16 B: the perturbation that returned a cosine of 1.0073.
    write(dir.join("i.tv"), 4, 64, 16, &codes[..1008], &b, &c, &scales, &ts, &tsc).unwrap();
}

#[test]
#[should_panic(expected = "codes_blocked_seq length 1040 does not match the blocked layout \
                           for n_vectors 16, bit_width 4, dim 64 (1024 bytes)")]
fn write_rejects_long_codes_buffer() {
    let dir = temp_dir("v407-write-long-codes");
    let (idx, _) = v407_index();
    let (mut codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    codes.extend_from_slice(&[0u8; 16]);
    write(dir.join("i.tv"), 4, 64, 16, &codes, &b, &c, &scales, &ts, &tsc).unwrap();
}

#[test]
#[should_panic(expected = "codes_blocked_seq length 1008 does not match")]
fn write_to_rejects_short_codes_buffer() {
    let (idx, _) = v407_index();
    let (codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    let mut sink = Vec::new();
    turbovec::io::write_to(&mut sink, 4, 64, 16, &codes[..1008], &b, &c, &scales, &ts, &tsc)
        .unwrap();
}

#[test]
#[should_panic(expected = "codes_blocked_seq length 1008 does not match")]
fn write_with_durability_rejects_short_codes_buffer() {
    let dir = temp_dir("v407-durability-short-codes");
    let (idx, _) = v407_index();
    let (codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    turbovec::io::write_with_durability(
        dir.join("i.tv"), 4, 64, 16, &codes[..1008], &b, &c, &scales, &ts, &tsc,
        turbovec::io::Durability::Fast,
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "codes_blocked_seq length 1008 does not match")]
fn write_id_map_rejects_short_codes_buffer() {
    let dir = temp_dir("v407-idmap-short-codes");
    let (idx, _) = v407_index();
    let (codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    let ids: Vec<u64> = (0..16).collect();
    write_id_map(dir.join("i.tvim"), 4, 64, 16, &codes[..1008], &b, &c, &scales, &ts, &tsc, &ids)
        .unwrap();
}

#[test]
#[should_panic(expected = "codes_blocked_seq length 1008 does not match")]
fn write_id_map_with_durability_rejects_short_codes_buffer() {
    let dir = temp_dir("v407-idmap-dur-short-codes");
    let (idx, _) = v407_index();
    let (codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    let ids: Vec<u64> = (0..16).collect();
    turbovec::io::write_id_map_with_durability(
        dir.join("i.tvim"), 4, 64, 16, &codes[..1008], &b, &c, &scales, &ts, &tsc, &ids,
        turbovec::io::Durability::Durable,
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "codes_blocked_seq length 1008 does not match")]
fn write_id_map_to_rejects_short_codes_buffer() {
    let (idx, _) = v407_index();
    let (codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    let ids: Vec<u64> = (0..16).collect();
    let mut sink = Vec::new();
    turbovec::io::write_id_map_to(
        &mut sink, 4, 64, 16, &codes[..1008], &b, &c, &scales, &ts, &tsc, &ids,
    )
    .unwrap();
}

// --- scales-length rejection ----------------------------------------------

#[test]
#[should_panic(expected = "scales length 14 does not match n_vectors 16")]
fn write_rejects_short_scales() {
    let dir = temp_dir("v407-write-short-scales");
    let (idx, _) = v407_index();
    let (codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    write(dir.join("i.tv"), 4, 64, 16, &codes, &b, &c, &scales[..14], &ts, &tsc).unwrap();
}

#[test]
#[should_panic(expected = "scales length 0 does not match n_vectors 16")]
fn write_rejects_empty_scales() {
    let dir = temp_dir("v407-write-empty-scales");
    let (idx, _) = v407_index();
    let (codes, _, b, c, ts, tsc) = v407_parts(&idx);
    write(dir.join("i.tv"), 4, 64, 16, &codes, &b, &c, &[], &ts, &tsc).unwrap();
}

#[test]
#[should_panic(expected = "scales length 17 does not match n_vectors 16")]
fn write_to_rejects_long_scales() {
    let (idx, _) = v407_index();
    let (codes, mut scales, b, c, ts, tsc) = v407_parts(&idx);
    scales.push(1.0);
    let mut sink = Vec::new();
    turbovec::io::write_to(&mut sink, 4, 64, 16, &codes, &b, &c, &scales, &ts, &tsc).unwrap();
}

#[test]
#[should_panic(expected = "scales length 14 does not match n_vectors 16")]
fn write_id_map_to_rejects_short_scales() {
    let (idx, _) = v407_index();
    let (codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    let ids: Vec<u64> = (0..16).collect();
    let mut sink = Vec::new();
    turbovec::io::write_id_map_to(
        &mut sink, 4, 64, 16, &codes, &b, &c, &scales[..14], &ts, &tsc, &ids,
    )
    .unwrap();
}

/// The case no header-derived load check can ever catch: codes 8 B short
/// and scales 2 entries long keeps the total file size identical, so
/// nothing downstream shifts. Before this validation it loaded clean
/// every time, in every configuration tested.
#[test]
#[should_panic(expected = "scales length 18 does not match n_vectors 16")]
fn write_rejects_byte_count_preserving_compensating_pair() {
    let dir = temp_dir("v407-compensating");
    let (idx, _) = v407_index();
    let (codes, mut scales, b, c, ts, tsc) = v407_parts(&idx);
    scales.push(1.0);
    scales.push(1.0);
    write(dir.join("i.tv"), 4, 64, 16, &codes[..1016], &b, &c, &scales, &ts, &tsc).unwrap();
}

// --- the rejection must not cost the caller their existing index ----------

#[test]
fn rejected_write_leaves_the_previous_index_and_no_temp_files() {
    let dir = temp_dir("v407-atomic");
    let path = dir.join("index.tv");
    let (packed, scales) = write_good_tv(&path);

    let (idx, _) = v407_index();
    let (codes, bad_scales, b, c, ts, tsc) = v407_parts(&idx);
    for label in ["short codes", "short scales"] {
        let (cs, ss): (&[u8], &[f32]) = match label {
            "short codes" => (&codes[..1008], &bad_scales),
            _ => (&codes, &bad_scales[..14]),
        };
        let r = catch_unwind(AssertUnwindSafe(|| {
            write(&path, 4, 64, 16, cs, &b, &c, ss, &ts, &tsc)
        }));
        assert!(r.is_err(), "{label}: an inconsistent buffer must be rejected");
    }

    let (bw, d, n, p, s, _, _) = load(&path).expect("previous good index must survive");
    assert_eq!((bw, d, n), (4, 32, 2));
    assert_eq!(s, scales);
    assert_eq!(
        p,
        CodePayload::BlockedNative {
            codes: expected_native(&packed),
            boundaries: test_codebook(4, 32).0,
            centroids: test_codebook(4, 32).1,
        }
    );
    assert_eq!(dir_entries(&dir), vec!["index.tv"], "no partial/temp files may remain");
    std::fs::remove_dir_all(&dir).ok();
}

// --- the corruption this rejects is real ----------------------------------

/// Independent evidence that the rejected input is not merely
/// unconventional. The bytes are spliced directly — never through the
/// writer — into exactly what a 16-byte-short `codes_blocked_seq` would
/// have produced, and the result loads clean and returns a top score
/// above the cosine ceiling. Unit-norm rows mean 1.0 is the hard
/// maximum a correct cosine can reach, so a score above it is not
/// inaccuracy, it is an impossible answer that nothing rejected.
#[test]
fn a_short_codes_section_loads_clean_and_scores_above_the_cosine_ceiling() {
    let (idx, query) = v407_index();
    let good = idx.to_bytes();
    // magic(4) + version(1) + bit_width(1) + dim(4) + n_vectors(8)
    // + boundaries(15 f32) + centroids(16 f32) = 142, then the codes.
    const CODES_OFF: usize = 142;
    const CODES_LEN: usize = 1024;
    assert_eq!(
        good.len(),
        CODES_OFF + CODES_LEN + 16 * 4 + 4 + 64 * 4 + 64 * 4,
        "layout drifted; the splice offsets below are no longer the codes section",
    );

    let control = TurboQuantIndex::from_bytes(&good).expect("control must load");
    let control_top = control.search(&query, 1).scores[0];
    assert!(
        control_top <= 1.0,
        "a correct cosine cannot exceed 1.0; control was {control_top}",
    );

    let mut spliced = good[..CODES_OFF + CODES_LEN - 16].to_vec();
    spliced.extend_from_slice(&good[CODES_OFF + CODES_LEN..]);
    assert_eq!(spliced.len(), good.len() - 16);

    let corrupt = TurboQuantIndex::from_bytes(&spliced)
        .expect("the whole point of #407: this file loads clean");
    let top = corrupt.search(&query, 1).scores[0];
    assert!(
        top > 1.0,
        "a 16-byte-short codes section must still be observably impossible, \
         got {top} (control {control_top})",
    );
}

// --- the checks must accept everything that is actually valid -------------

/// The rejection has to be discriminating, not blanket: exact-length
/// buffers still write, load and score identically to the index they
/// came from, through both formats. A comparator flipped the wrong way
/// fails here rather than passing every `should_panic` above.
#[test]
fn exact_length_buffers_still_write_load_and_score() {
    let dir = temp_dir("v407-control");
    let (idx, query) = v407_index();
    let (codes, scales, b, c, ts, tsc) = v407_parts(&idx);
    let expect = idx.search(&query, 3);

    let tv = dir.join("i.tv");
    write(&tv, 4, 64, 16, &codes, &b, &c, &scales, &ts, &tsc).unwrap();
    let (bw, d, n, _, s, _, _) = load(&tv).unwrap();
    assert_eq!((bw, d, n), (4, 64, 16));
    assert_eq!(s, scales);

    let tvim = dir.join("i.tvim");
    let ids: Vec<u64> = (100..116).collect();
    write_id_map(&tvim, 4, 64, 16, &codes, &b, &c, &scales, &ts, &tsc, &ids).unwrap();
    let (.., slot_to_id) = load_id_map(&tvim).unwrap();
    assert_eq!(slot_to_id, ids);

    let mut sink = Vec::new();
    turbovec::io::write_to(&mut sink, 4, 64, 16, &codes, &b, &c, &scales, &ts, &tsc).unwrap();
    assert_eq!(sink, std::fs::read(&tv).unwrap(), "write_to must match the file bytes");
    let round_tripped = TurboQuantIndex::from_bytes(&sink).unwrap();
    assert_eq!(round_tripped.search(&query, 3).scores, expect.scores);

    std::fs::remove_dir_all(&dir).ok();
}

/// A zero-vector index writes empty code and scale buffers, and a lazy
/// one writes them alongside the `dim = 0` sentinel. Both are consistent
/// and must stay writable.
#[test]
fn empty_and_lazy_indexes_still_write() {
    let dir = temp_dir("v407-empty");
    let (b, c) = turbovec::expected_codebook(4, 64);
    write(dir.join("empty.tv"), 4, 64, 0, &[], &b, &c, &[], &[], &[]).unwrap();
    load(dir.join("empty.tv")).unwrap();
    // dim = 0 is the lazy sentinel; the blocked layout is empty for it.
    write(dir.join("lazy.tv"), 4, 0, 0, &[], &b, &c, &[], &[], &[]).unwrap();
    load(dir.join("lazy.tv")).unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

/// The codes check is defined only for the widths the format encodes,
/// so it steps aside outside 2..=4 rather than dividing by
/// `8 / bit_width`. That hole is #411's, not this change's: such a file
/// still writes and is still refused by the header validation on load,
/// exactly as before. (The scales check is width-independent and does
/// still apply.)
/// A `bit_width` at or above the usize width has no `1 << bit_width`,
/// and the two build profiles used to disagree about that: debug
/// panicked `attempt to shift left with overflow` from inside
/// `assert_codebook_lengths`, naming neither the argument nor the
/// function, while release masked the shift to `<< 0` and carried on
/// with `n_levels == 1`. This call satisfies that masked reading — no
/// boundaries, one centroid — so in release it used to return `Ok`,
/// emitting a 26-byte file with `bit_width = 64` in the header that
/// neither reader accepts. One message, both profiles, naming the
/// argument (#411).
#[test]
#[should_panic(expected = "bit_width 64 is out of range")]
fn bit_width_that_overflows_the_level_shift_is_rejected_in_every_profile() {
    let mut buf: Vec<u8> = Vec::new();
    let _ = write_to(&mut buf, 64, 64, 0, &[], &[], &[0.5], &[], &[], &[]);
}

/// The bound is the shift, not the format's 2..=4 — a width one below it
/// must still reach the ordinary length assertion rather than the new
/// range message, or the guard has quietly absorbed the carve-out that
/// `out_of_range_bit_width_is_left_to_the_load_side_header_check` pins.
#[test]
#[should_panic(expected = "codebook boundaries length")]
fn bit_width_just_below_the_shift_bound_still_reports_a_length_mismatch() {
    let mut buf: Vec<u8> = Vec::new();
    let _ = write_to(&mut buf, 63, 64, 0, &[], &[], &[0.5], &[], &[], &[]);
}

#[test]
fn out_of_range_bit_width_is_left_to_the_load_side_header_check() {
    let dir = temp_dir("v407-bw-oob");
    let path = dir.join("bw0.tv");
    // bit_width 0 -> 1 level: 0 boundaries, 1 centroid.
    write(&path, 0, 32, 2, &[0u8; 7], &[], &[0.0], &[1.0, 1.0], &[], &[]).unwrap();
    let err = load(&path).expect_err("bit_width 0 must be rejected on load");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("invalid bit_width 0"), "got: {err}");
    std::fs::remove_dir_all(&dir).ok();
}
