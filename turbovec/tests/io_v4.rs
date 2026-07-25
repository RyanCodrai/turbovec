//! Format v4 tests: u64 `n_vectors`, rotation-drift fingerprint, and
//! the load-time dim cap — plus v3 back-compat against fixture files
//! written by the actual v3 writer (turbovec @ main 2bd8a3e, generated
//! by a scratch build; see `tests/fixtures/`).
//!
//! Covers:
//! 1. v4 round-trip: write + load + search parity with the in-memory
//!    index, plus byte-level header-layout assertions (version byte 4,
//!    u64 count field, fingerprint hash).
//! 2. v3 back-compat: the checked-in v3 fixtures load and search
//!    identically to a freshly-built index over the same vectors, and
//!    match the exact results main produced when the fixtures were
//!    generated.
//! 3. Rotation-drift detection: corrupted fingerprint probes produce a
//!    clean, distinguishable "rotation drift" error; a hash-only
//!    mismatch with in-tolerance probes (the benign cross-environment
//!    build-noise case) still loads.
//! 4. Dim cap: a file declaring dim over `MAX_DIM` errors fast (no
//!    rotation build) on both the v3 and v4 paths; construction over
//!    the cap errors in the same family.
//! 5. Count-field width: a crafted v4 header with n = 2^33 fails on the
//!    length checks instead of wrapping to a small count.
//! 6. Fuzz-lite: single-byte mutations over valid v4 files load as
//!    `Ok` or a clean `Err` — never a panic.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use turbovec::error::{AddError, ConstructError};
use turbovec::io::load;
use turbovec::{IdMapIndex, TurboQuantIndex, MAX_DIM};

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

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// Deterministic vector source — must match the copy in the fixture
/// generator that produced `tests/fixtures/v3_index.tv{,im}` exactly.
fn lcg_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n * dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map the high 32 bits to [-1.0, 1.0).
            ((state >> 32) as u32 as f64 / 2_147_483_648.0 - 1.0) as f32
        })
        .collect()
}

const DIM: usize = 32;
const N: usize = 64;
const VEC_SEED: u64 = 0xDECAF;
const QUERY_SEED: u64 = 0xC0FFEE;

fn build_index() -> TurboQuantIndex {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add_2d(&lcg_vectors(N, DIM, VEC_SEED), DIM).unwrap();
    idx
}

/// FNV-1a over f32 LE bytes — a deliberate reimplementation of the
/// fingerprint hash, pinning the on-disk hash algorithm independently
/// of the crate's internal implementation.
fn fnv1a(data: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in data {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

// v4 .tv layout offsets (after 4-byte magic + 1-byte version):
// bit_width u8 @5, dim u32 @6, n_vectors u64 @10, fp hash u64 @18,
// fp probes 64×f32 @26, payload @282.
const OFF_VERSION: usize = 4;
const OFF_N: usize = 10;
const OFF_HASH: usize = 18;
const OFF_PROBES: usize = 26;
const N_PROBES: usize = 64;
const OFF_PAYLOAD: usize = OFF_PROBES + 4 * N_PROBES;

// ---------------------------------------------------------------------------
// 1. v4 round-trip + header layout
// ---------------------------------------------------------------------------

#[test]
fn tv_v4_round_trip_search_parity_and_header_layout() {
    let dir = temp_dir("v4-roundtrip");
    let path = dir.join("index.tv");
    let idx = build_index();
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let before = idx.search(&queries, 5);

    idx.write(&path).unwrap();
    let loaded = TurboQuantIndex::load(&path).unwrap();
    let after = loaded.search(&queries, 5);
    assert_eq!(before.scores, after.scores, "scores must survive a round-trip");
    assert_eq!(before.indices, after.indices, "indices must survive a round-trip");

    // Byte-level header assertions.
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"TVPI");
    assert_eq!(bytes[OFF_VERSION], 4, "writer must emit format version 4");
    assert_eq!(bytes[5], 4, "bit_width");
    assert_eq!(u32::from_le_bytes(bytes[6..10].try_into().unwrap()), DIM as u32);
    assert_eq!(
        u64::from_le_bytes(bytes[OFF_N..OFF_N + 8].try_into().unwrap()),
        N as u64,
        "n_vectors must be a u64 field",
    );
    let stored_hash = u64::from_le_bytes(bytes[OFF_HASH..OFF_HASH + 8].try_into().unwrap());
    let rotation = turbovec::rotation::make_rotation_matrix(DIM);
    assert_eq!(
        stored_hash,
        fnv1a(&rotation),
        "stored fingerprint hash must be FNV-1a over the rotation's f32 LE bytes",
    );
    // Every stored probe must be a value that occurs in the rotation.
    for i in 0..N_PROBES {
        let o = OFF_PROBES + 4 * i;
        let p = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        assert!(
            rotation.contains(&p),
            "probe {i} = {p} is not an element of the rotation matrix",
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tvim_v4_round_trip_search_parity() {
    let dir = temp_dir("v4-roundtrip-tvim");
    let path = dir.join("index.tvim");
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    let ids: Vec<u64> = (0..N as u64).map(|i| 1000 + i).collect();
    idx.add_with_ids(&lcg_vectors(N, DIM, VEC_SEED), &ids).unwrap();
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let (scores_before, ids_before) = idx.search(&queries, 5);

    idx.write(&path).unwrap();
    let loaded = IdMapIndex::load(&path).unwrap();
    let (scores_after, ids_after) = loaded.search(&queries, 5);
    assert_eq!(scores_before, scores_after);
    assert_eq!(ids_before, ids_after);

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"TVIM");
    assert_eq!(bytes[OFF_VERSION], 4);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tv_v4_empty_index_stores_zero_fingerprint() {
    // No vectors → no rotation is associated with the file: the
    // fingerprint region must be all zero and loading must skip
    // verification (no rotation build).
    let dir = temp_dir("v4-empty-fp");
    let path = dir.join("empty.tv");
    let idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert!(
        bytes[OFF_HASH..OFF_PAYLOAD].iter().all(|&b| b == 0),
        "empty index must store an all-zero fingerprint",
    );
    let loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.len(), 0);
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// 2. v3 back-compat via fixtures written by main's actual v3 writer
// ---------------------------------------------------------------------------

// Search results main @ 2bd8a3e produced on these fixtures at
// generation time (4 LCG queries, k=5).
const V3_EXPECTED_INDICES: [i64; 20] = [
    40, 29, 39, 24, 10, 1, 25, 43, 9, 60, 33, 59, 4, 46, 40, 38, 47, 14, 62, 20,
];
const V3_EXPECTED_IDS: [u64; 20] = [
    1040, 1029, 1039, 1024, 1010, 1001, 1025, 1043, 1009, 1060, 1033, 1059, 1004,
    1046, 1040, 1038, 1047, 1014, 1062, 1020,
];

#[test]
fn v3_fixture_tv_loads_with_identical_search_results() {
    let bytes = std::fs::read(fixture("v3_index.tv")).unwrap();
    assert_eq!(bytes[OFF_VERSION], 3, "fixture must be a genuine v3 file");

    let loaded = TurboQuantIndex::load(fixture("v3_index.tv")).unwrap();
    assert_eq!(loaded.len(), N);
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let from_file = loaded.search(&queries, 5);

    // Same ranking as a freshly-built index over the same vectors...
    let fresh = build_index().search(&queries, 5);
    assert_eq!(from_file.indices, fresh.indices);
    // ...and as what main itself returned when the fixture was written.
    assert_eq!(from_file.indices, V3_EXPECTED_INDICES);
    // Scores: the fixture's stored scales were computed by the generator
    // machine's BLAS at encode time; a fresh encode on the running
    // machine can differ by ~1 f32 ulp (BLAS accumulation order varies
    // across platforms — observed ~5e-7 relative on Linux/OpenBLAS CI),
    // so scores are compared with a tight relative tolerance rather
    // than bit equality. Result *order* above is asserted exactly.
    for (i, (a, b)) in from_file.scores.iter().zip(fresh.scores.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * a.abs().max(1.0),
            "score {i} diverged beyond encode-side BLAS noise: fixture={a}, fresh={b}",
        );
    }
}

#[test]
fn v3_fixture_tvim_loads_with_identical_search_results() {
    let bytes = std::fs::read(fixture("v3_index.tvim")).unwrap();
    assert_eq!(bytes[OFF_VERSION], 3, "fixture must be a genuine v3 file");

    let loaded = IdMapIndex::load(fixture("v3_index.tvim")).unwrap();
    assert_eq!(loaded.len(), N);
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let (_, ids) = loaded.search(&queries, 5);
    assert_eq!(ids, V3_EXPECTED_IDS);
}

#[test]
fn v3_resave_upgrades_to_v4_with_fingerprint() {
    // Loading a v3 file and re-saving writes the current (v4) format,
    // including a freshly-computed fingerprint.
    let dir = temp_dir("v3-resave");
    let path = dir.join("resaved.tv");
    let loaded = TurboQuantIndex::load(fixture("v3_index.tv")).unwrap();
    loaded.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes[OFF_VERSION], 4);
    let hash = u64::from_le_bytes(bytes[OFF_HASH..OFF_HASH + 8].try_into().unwrap());
    assert_ne!(hash, 0, "re-saved non-empty index must carry a fingerprint");

    let reloaded = TurboQuantIndex::load(&path).unwrap();
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    assert_eq!(reloaded.search(&queries, 5).indices, V3_EXPECTED_INDICES);
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// 3. Rotation-drift detection
// ---------------------------------------------------------------------------

#[test]
fn v4_drifted_fingerprint_is_a_clean_distinguishable_error() {
    // Simulate genuine rotation drift: every stored probe shifted far
    // beyond tolerance (a real algorithm change perturbs essentially
    // every element at the ~1e-2 scale; +1.0 is unambiguous), hash
    // perturbed to match no rebuild.
    let dir = temp_dir("v4-drift");
    let path = dir.join("index.tv");
    build_index().write(&path).unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    bytes[OFF_HASH] ^= 0xFF;
    for i in 0..N_PROBES {
        let o = OFF_PROBES + 4 * i;
        let p = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        bytes[o..o + 4].copy_from_slice(&(p + 1.0).to_le_bytes());
    }
    std::fs::write(&path, &bytes).unwrap();

    let err = match TurboQuantIndex::load(&path) {
        Err(e) => e,
        Ok(_) => panic!("drifted fingerprint must not load"),
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let msg = err.to_string();
    assert!(
        msg.contains("rotation drift"),
        "drift must be distinguishable from generic corruption, got: {msg}",
    );
    assert!(
        msg.contains("rebuild the index"),
        "error must tell the user how to recover, got: {msg}",
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn v4_hash_mismatch_with_in_tolerance_probes_still_loads() {
    // The benign cross-environment case: faer's QR differs by ~1 f32
    // ulp in a handful of elements across thread counts and CPU
    // architectures, which changes the exact hash but leaves every
    // probe within tolerance. Such files must load.
    let dir = temp_dir("v4-ulp-noise");
    let path = dir.join("index.tv");
    let idx = build_index();
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let expected = idx.search(&queries, 5);
    idx.write(&path).unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    bytes[OFF_HASH] ^= 0xFF; // hash no longer matches any rebuild
    std::fs::write(&path, &bytes).unwrap();

    let loaded = TurboQuantIndex::load(&path)
        .expect("in-tolerance probes must accept a hash-only mismatch");
    assert_eq!(loaded.search(&queries, 5).indices, expected.indices);
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// 4. Load-time dim cap
// ---------------------------------------------------------------------------

const OVER_CAP_DIM: usize = 32_768; // multiple of 8, over MAX_DIM, under the old 65536
// The guard must reject before any rotation work; building a
// 32768x32768 rotation would take minutes and ~8 GiB, so a small
// budget decisively proves no build was attempted.
const CAP_BUDGET: Duration = Duration::from_secs(2);

/// A structurally-complete v3 core payload for dim=OVER_CAP_DIM, n=1,
/// bit_width=2 — valid in every respect except the dim cap.
fn over_cap_core_payload() -> Vec<u8> {
    let mut b = Vec::new();
    b.push(2u8); // bit_width
    b.extend_from_slice(&(OVER_CAP_DIM as u32).to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes()); // n_vectors (v3: u32)
    b.extend_from_slice(&vec![0u8; OVER_CAP_DIM / 8 * 2]); // packed codes
    b.extend_from_slice(&1.0f32.to_le_bytes()); // scale
    b.extend_from_slice(&0u32.to_le_bytes()); // n_calib = 0
    b
}

#[test]
fn v3_file_with_dim_over_cap_errors_fast() {
    let dir = temp_dir("v3-dim-cap");
    let path = dir.join("bigdim.tv");
    let mut bytes = vec![];
    bytes.extend_from_slice(b"TVPI");
    bytes.push(3u8);
    bytes.extend_from_slice(&over_cap_core_payload());
    std::fs::write(&path, &bytes).unwrap();

    let start = Instant::now();
    let err = load(&path).expect_err("dim over MAX_DIM must be rejected");
    let elapsed = start.elapsed();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("exceeds maximum"),
        "expected the dim-cap error, got: {err}",
    );
    assert!(
        elapsed < CAP_BUDGET,
        "rejection took {elapsed:?} — it must not touch the rotation build",
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn v4_file_with_dim_over_cap_errors_fast() {
    let dir = temp_dir("v4-dim-cap");
    let path = dir.join("bigdim.tv");
    let mut bytes = vec![];
    bytes.extend_from_slice(b"TVPI");
    bytes.push(4u8);
    bytes.push(2u8); // bit_width
    bytes.extend_from_slice(&(OVER_CAP_DIM as u32).to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes()); // n_vectors (v4: u64)
    bytes.extend_from_slice(&[0u8; 8 + 4 * N_PROBES]); // fingerprint
    bytes.extend_from_slice(&vec![0u8; OVER_CAP_DIM / 8 * 2]);
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let start = Instant::now();
    let err = load(&path).expect_err("dim over MAX_DIM must be rejected");
    let elapsed = start.elapsed();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("exceeds maximum"), "got: {err}");
    assert!(elapsed < CAP_BUDGET, "rejection took {elapsed:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn construction_over_cap_errors_in_the_same_family() {
    // Round-trip consistency: the cap applies identically at
    // construction, first add, and load — an index this build can
    // create it can also load back.
    match TurboQuantIndex::new(OVER_CAP_DIM, 4) {
        Err(ConstructError::DimTooLarge { dim, max }) => {
            assert_eq!(dim, OVER_CAP_DIM);
            assert_eq!(max, MAX_DIM);
        }
        Err(other) => panic!("expected DimTooLarge, got {other:?}"),
        Ok(_) => panic!("construction over the cap must fail"),
    }
    let mut lazy = TurboQuantIndex::new_lazy(4).unwrap();
    match lazy.add_2d(&vec![0.5f32; OVER_CAP_DIM], OVER_CAP_DIM) {
        Err(AddError::DimTooLarge { dim, max }) => {
            assert_eq!(dim, OVER_CAP_DIM);
            assert_eq!(max, MAX_DIM);
        }
        other => panic!("expected DimTooLarge, got {other:?}"),
    }
    // The boundary itself is constructible (cap is inclusive).
    assert!(TurboQuantIndex::new(MAX_DIM, 4).is_ok());
}

// ---------------------------------------------------------------------------
// 5. u64 count field is really 64-bit on the read side
// ---------------------------------------------------------------------------

#[cfg(target_pointer_width = "64")]
#[test]
fn v4_huge_n_with_truncated_payload_fails_on_length_check_not_wrap() {
    // n = 2^33 with a payload that would be valid for n = 0. If the
    // reader wrapped the count to u32 (2^33 mod 2^32 == 0) the file
    // would load "successfully" as empty — the pre-v4 silent-wrap
    // failure mode. The 64-bit reader must instead demand 2^33 codes
    // and fail cleanly on the existing length checks.
    let dir = temp_dir("v4-huge-n");
    let path = dir.join("huge.tv");
    let mut bytes = vec![];
    bytes.extend_from_slice(b"TVPI");
    bytes.push(4u8);
    bytes.push(2u8); // bit_width
    bytes.extend_from_slice(&8u32.to_le_bytes()); // dim = 8
    bytes.extend_from_slice(&(1u64 << 33).to_le_bytes()); // n_vectors
    bytes.extend_from_slice(&[0u8; 8 + 4 * N_PROBES]); // fingerprint
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_calib (valid iff n wrapped to 0)
    std::fs::write(&path, &bytes).unwrap();

    let err = load(&path).expect_err("2^33 vectors with no payload must not load");
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::UnexpectedEof,
        "expected a truncation error from the length-capped reader, got: {err}",
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// 6. Fuzz-lite: single-byte mutations never panic
// ---------------------------------------------------------------------------

#[test]
fn v4_single_byte_mutations_load_or_error_cleanly() {
    let dir = temp_dir("v4-mutations");
    let tv = dir.join("index.tv");
    let tvim = dir.join("index.tvim");
    let mut small = TurboQuantIndex::new(8, 4).unwrap();
    small.add_2d(&lcg_vectors(3, 8, VEC_SEED), 8).unwrap();
    small.write(&tv).unwrap();
    let mut small_im = IdMapIndex::new(8, 4).unwrap();
    small_im
        .add_with_ids(&lcg_vectors(3, 8, VEC_SEED), &[7, 8, 9])
        .unwrap();
    small_im.write(&tvim).unwrap();

    let mutated = dir.join("mutated");
    for (orig, is_tvim) in [(&tv, false), (&tvim, true)] {
        let bytes = std::fs::read(orig).unwrap();
        for pos in 0..bytes.len() {
            for pattern in [0xFFu8, 0x01] {
                let mut m = bytes.clone();
                m[pos] ^= pattern;
                std::fs::write(&mutated, &m).unwrap();
                // Any outcome but a panic is acceptable: a mutation may
                // produce a still-valid file (Ok) or corruption (Err).
                if is_tvim {
                    let _ = IdMapIndex::load(&mutated);
                } else {
                    let _ = TurboQuantIndex::load(&mutated);
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
