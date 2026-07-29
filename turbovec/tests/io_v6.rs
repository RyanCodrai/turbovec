//! Format v5 tests: the block-Hadamard rotation break.
//!
//! Covers:
//! 1. v5 round-trip: write + load + search parity, plus byte-level header
//!    assertions (version byte 5, u64 count field, no fingerprint — the
//!    payload begins right after `n_vectors`).
//! 2. Hard break: v1/v2/v3/v4 files are refused on load with a clean,
//!    actionable "incompatible … rebuild the index" error — never a
//!    silent mis-decode and never a panic.
//! 3. Dim cap: a v5 file declaring dim over `MAX_DIM` errors fast.
//! 4. Count-field width: a crafted v5 header with n = 2^33 fails on the
//!    length checks instead of wrapping to a small count.
//! 5. Fuzz-lite: single-byte mutations over valid v5 files load as `Ok`
//!    or a clean `Err` — never a panic.

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

fn lcg_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n * dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 32) as u32 as f64 / 2_147_483_648.0 - 1.0) as f32
        })
        .collect()
}

const DIM: usize = 32;
const N: usize = 64;
const VEC_SEED: u64 = 0xDECAF;
const QUERY_SEED: u64 = 0xC0FFEE;

// v5 .tv layout offsets (after 4-byte magic + 1-byte version):
// bit_width u8 @5, dim u32 @6, n_vectors u64 @10, payload @18. No
// rotation fingerprint (the v5 rotation is deterministic).
const OFF_VERSION: usize = 4;
const OFF_N: usize = 10;
const OFF_PAYLOAD: usize = 18;

fn build_index() -> TurboQuantIndex {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add_2d(&lcg_vectors(N, DIM, VEC_SEED), DIM).unwrap();
    idx
}

// ---------------------------------------------------------------------------
// 1. v5 round-trip + header layout
// ---------------------------------------------------------------------------

#[test]
fn tv_v5_round_trip_search_parity_and_header_layout() {
    let dir = temp_dir("v5-roundtrip");
    let path = dir.join("index.tv");
    let idx = build_index();
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let before = idx.search(&queries, 5);

    idx.write(&path).unwrap();
    let loaded = TurboQuantIndex::load(&path).unwrap();
    let after = loaded.search(&queries, 5);
    assert_eq!(before.scores, after.scores, "scores must survive a round-trip");
    assert_eq!(before.indices, after.indices, "indices must survive a round-trip");

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"TVPI");
    assert_eq!(bytes[OFF_VERSION], 6, "writer must emit format version 6");
    assert_eq!(bytes[5], 4, "bit_width");
    assert_eq!(u32::from_le_bytes(bytes[6..10].try_into().unwrap()), DIM as u32);
    assert_eq!(
        u64::from_le_bytes(bytes[OFF_N..OFF_N + 8].try_into().unwrap()),
        N as u64,
        "n_vectors must be a u64 field",
    );
    // Payload (sequential blocked codes) begins immediately after
    // n_vectors: no fingerprint. Header is exactly 18 bytes; the blocked
    // payload is padded to whole 32-vector blocks.
    let blocked_len = N.div_ceil(32) * 32 * (DIM / 2); // bit_width=4
    assert!(bytes.len() >= OFF_PAYLOAD + blocked_len);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tvim_v5_round_trip_search_parity() {
    let dir = temp_dir("v5-roundtrip-tvim");
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
    assert_eq!(bytes[OFF_VERSION], 6);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tv_v5_empty_index_round_trips() {
    let dir = temp_dir("v5-empty");
    let path = dir.join("empty.tv");
    let idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.write(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes[OFF_VERSION], 6);
    // n_vectors == 0, header is 18 bytes, then TQ+ trailer n_calib=0.
    assert_eq!(u64::from_le_bytes(bytes[OFF_N..OFF_N + 8].try_into().unwrap()), 0);
    let loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.len(), 0);
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// 2. Hard break: every pre-v5 version is refused with an actionable error
// ---------------------------------------------------------------------------

/// A structurally-plausible pre-v5 `.tv` file at `version`: TVPI magic,
/// the given version byte, then a v3-shaped core (enough bytes that the
/// loader would have something to parse if it tried).
fn pre_v5_tv(version: u8) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"TVPI");
    b.push(version);
    b.push(4u8); // bit_width
    b.extend_from_slice(&(DIM as u32).to_le_bytes());
    b.extend_from_slice(&(1u32).to_le_bytes()); // v2/v3 u32 count
    b.extend_from_slice(&vec![0u8; DIM / 8 * 4]); // one row of codes
    b.extend_from_slice(&1.0f32.to_le_bytes()); // one scale
    b.extend_from_slice(&0u32.to_le_bytes()); // n_calib = 0
    b
}

#[test]
fn v4_and_earlier_tv_files_are_refused_with_rebuild_hint() {
    let dir = temp_dir("v5-reject");
    for version in [2u8, 3, 4] {
        let path = dir.join(format!("v{version}.tv"));
        std::fs::write(&path, pre_v5_tv(version)).unwrap();
        let err = match TurboQuantIndex::load(&path) {
            Err(e) => e,
            Ok(_) => panic!("format v{version} must not load under the v5 rotation break"),
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("version {version}")),
            "error must name the offending version, got: {msg}",
        );
        assert!(
            msg.contains("incompatible"),
            "error must flag incompatibility, got: {msg}",
        );
        assert!(
            msg.to_lowercase().contains("rebuild"),
            "error must tell the user to rebuild, got: {msg}",
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn v1_tv_file_without_magic_is_refused_with_rebuild_hint() {
    // v1 files had no magic; the first byte was the bit_width (2/3/4).
    let dir = temp_dir("v5-reject-v1");
    let path = dir.join("v1.tv");
    let mut b = Vec::new();
    b.push(4u8); // bit_width, mistaken-for-magic sentinel
    b.extend_from_slice(&(DIM as u32).to_le_bytes());
    b.extend_from_slice(&(1u32).to_le_bytes());
    b.extend_from_slice(&vec![0u8; DIM / 8 * 4]);
    b.extend_from_slice(&1.0f32.to_le_bytes());
    std::fs::write(&path, &b).unwrap();

    let err = TurboQuantIndex::load(&path).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("version 1"), "got: {msg}");
    assert!(msg.to_lowercase().contains("rebuild"), "got: {msg}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn v4_tvim_file_is_refused_with_rebuild_hint() {
    let dir = temp_dir("v5-reject-tvim");
    let path = dir.join("v4.tvim");
    let mut b = Vec::new();
    b.extend_from_slice(b"TVIM");
    b.push(4u8);
    b.extend_from_slice(&[0u8; 64]); // arbitrary payload; rejected on version
    std::fs::write(&path, &b).unwrap();

    let err = IdMapIndex::load(&path).unwrap_err();
    let msg = err.to_string();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(msg.contains("version 4"), "got: {msg}");
    assert!(msg.to_lowercase().contains("rebuild"), "got: {msg}");
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// 3. Load-time dim cap
// ---------------------------------------------------------------------------

const OVER_CAP_DIM: usize = 32_768; // multiple of 8, over MAX_DIM
const CAP_BUDGET: Duration = Duration::from_secs(2);

#[test]
fn v5_file_with_dim_over_cap_errors_fast() {
    let dir = temp_dir("v5-dim-cap");
    let path = dir.join("bigdim.tv");
    let mut bytes = vec![];
    bytes.extend_from_slice(b"TVPI");
    bytes.push(5u8);
    bytes.push(2u8); // bit_width
    bytes.extend_from_slice(&(OVER_CAP_DIM as u32).to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes()); // n_vectors (v5: u64)
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
    match TurboQuantIndex::new(OVER_CAP_DIM, 4) {
        Err(ConstructError::DimTooLarge { dim, max }) => {
            assert_eq!(dim, OVER_CAP_DIM);
            assert_eq!(max, MAX_DIM);
        }
        other => panic!("expected DimTooLarge, got {other:?}"),
    }
    let mut lazy = TurboQuantIndex::new_lazy(4).unwrap();
    match lazy.add_2d(&vec![0.5f32; OVER_CAP_DIM], OVER_CAP_DIM) {
        Err(AddError::DimTooLarge { dim, max }) => {
            assert_eq!(dim, OVER_CAP_DIM);
            assert_eq!(max, MAX_DIM);
        }
        other => panic!("expected DimTooLarge, got {other:?}"),
    }
    assert!(TurboQuantIndex::new(MAX_DIM, 4).is_ok());
}

// ---------------------------------------------------------------------------
// 4. u64 count field is really 64-bit on the read side
// ---------------------------------------------------------------------------

#[cfg(target_pointer_width = "64")]
#[test]
fn v5_huge_n_with_truncated_payload_fails_on_length_check_not_wrap() {
    let dir = temp_dir("v5-huge-n");
    let path = dir.join("huge.tv");
    let mut bytes = vec![];
    bytes.extend_from_slice(b"TVPI");
    bytes.push(5u8);
    bytes.push(2u8); // bit_width
    bytes.extend_from_slice(&8u32.to_le_bytes()); // dim = 8
    bytes.extend_from_slice(&(1u64 << 33).to_le_bytes()); // n_vectors
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
// 5. Fuzz-lite: single-byte mutations never panic
// ---------------------------------------------------------------------------

#[test]
fn v5_single_byte_mutations_load_or_error_cleanly() {
    let dir = temp_dir("v5-mutations");
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

// ---------------------------------------------------------------------------
// v6-specific behaviour: v5 acceptance, mutation-after-load, write parity
// ---------------------------------------------------------------------------

/// Assemble the v5 byte stream for an index — the pre-v6 on-disk format
/// (packed bit-plane payload, version byte 5). v6 removed the v5 writer,
/// so compat coverage builds the bytes by hand.
fn v5_bytes(idx: &TurboQuantIndex, magic: &[u8; 4], version: u8) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(magic);
    b.push(version);
    b.push(idx.bit_width() as u8);
    b.extend_from_slice(&(idx.dim_opt().unwrap() as u32).to_le_bytes());
    b.extend_from_slice(&(idx.len() as u64).to_le_bytes());
    b.extend_from_slice(idx.packed_codes());
    for &s in idx.scales() {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b.extend_from_slice(&(idx.tqplus_shift().len() as u32).to_le_bytes());
    for &s in idx.tqplus_shift() {
        b.extend_from_slice(&s.to_le_bytes());
    }
    for &s in idx.tqplus_scale() {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b
}

#[test]
fn v5_file_loads_and_searches_identically() {
    // A v5 (packed-payload) byte stream is accepted by the v6 loader and
    // produces identical search results — the v5→v6 change is a pure
    // re-layout of the same code content.
    let idx = build_index();
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let before = idx.search(&queries, 5);

    let v5 = v5_bytes(&idx, b"TVPI", 5);
    let loaded = TurboQuantIndex::from_bytes(&v5).expect("v5 bytes must load");
    let after = loaded.search(&queries, 5);
    assert_eq!(before.scores, after.scores);
    assert_eq!(before.indices, after.indices);

    // Re-saving the v5-loaded index emits v6.
    let resaved = loaded.to_bytes();
    assert_eq!(resaved[4], 6, "re-save of a v5 index must emit v6");
    // ... which byte-equals the v6 serialization of the original.
    assert_eq!(resaved, idx.to_bytes());
}

#[test]
fn v6_loaded_index_survives_mutation() {
    // After a v6 load the packed codes are reconstructed lazily; add and
    // swap_remove must operate on exactly the pre-load content.
    let dir = temp_dir("v6-mutate");
    let path = dir.join("m.tv");
    let mut idx = build_index();
    idx.write(&path).unwrap();

    let mut loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.packed_codes(), idx.packed_codes(), "lazy packed reconstruction must be exact");

    // Mutations behave identically on the original and the loaded copy.
    let extra = lcg_vectors(3, DIM, 0xFEED);
    idx.add(&extra);
    loaded.add(&extra);
    idx.swap_remove(1);
    loaded.swap_remove(1);

    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let a = idx.search(&queries, 5);
    let b = loaded.search(&queries, 5);
    assert_eq!(a.scores, b.scores);
    assert_eq!(a.indices, b.indices);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn v6_write_is_identical_before_and_after_search() {
    // The write path serializes from the warm blocked cache when one
    // exists and from the packed rows otherwise; both must produce the
    // same bytes.
    let idx = build_index();
    let cold = idx.to_bytes(); // no search yet: repack_seq path
    idx.prepare(); // populates the blocked cache
    let warm = idx.to_bytes(); // native_to_seq path
    assert_eq!(cold, warm, "cold and warm write paths must be byte-identical");
}

#[test]
fn v6_round_trip_is_byte_stable() {
    // write → load → write must reproduce the file exactly (the loaded
    // index serializes from its seeded cache).
    let idx = build_index();
    let bytes = idx.to_bytes();
    let loaded = TurboQuantIndex::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.to_bytes(), bytes);
}

#[test]
fn tvim_v5_file_loads_and_searches_identically() {
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    let ids: Vec<u64> = (0..N as u64).map(|i| 1000 + i).collect();
    idx.add_with_ids(&lcg_vectors(N, DIM, VEC_SEED), &ids).unwrap();
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let (scores_before, ids_before) = idx.search(&queries, 5);

    // v5 .tvim = TVIM magic + version 5 + v5 core + id table. The core
    // parts come from a positional index built from the same vectors in
    // the same order — the id-map wrapper encodes identically.
    let core_idx = build_index();
    let mut v5 = Vec::new();
    v5.extend_from_slice(b"TVIM");
    v5.push(5);
    let core = v5_bytes(&core_idx, b"TVPI", 5)[5..].to_vec(); // strip magic+version
    v5.extend_from_slice(&core);
    for id in &ids {
        v5.extend_from_slice(&id.to_le_bytes());
    }
    let loaded = IdMapIndex::from_bytes(&v5).expect("v5 tvim bytes must load");
    let (scores_after, ids_after) = loaded.search(&queries, 5);
    assert_eq!(scores_before, scores_after);
    assert_eq!(ids_before, ids_after);
}

/// The incrementally-maintained blocked cache must be indistinguishable
/// from a cold rebuild: after a mixed sequence of adds and removes on a
/// warm-cached index, serialization must be byte-identical to the same
/// logical index serialized with a cold cache (full repack path), and
/// search results must match. Guards the #274 incremental-maintenance
/// change (stale padding lanes or unpatched blocks would diverge here).
#[test]
fn incremental_blocked_cache_serializes_identically_to_cold_rebuild() {
    let dim = 64usize;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&lcg_vectors(100, dim, 0xA11CE));
    idx.prepare(); // warm the cache so mutations take the patch path

    let mut step = 0u64;
    for round in 0..6 {
        // Adds of varied sizes, crossing block boundaries both ways.
        let n_add = [1usize, 31, 32, 33, 7, 64][round];
        idx.add(&lcg_vectors(n_add, dim, 0xBEEF + step));
        step += 1;
        // Removes from the front, middle, tail.
        let len = idx.len();
        idx.swap_remove(0);
        idx.swap_remove(len / 2 - 1);
        idx.swap_remove(idx.len() - 1);
        idx.prepare(); // keep the cache warm through every round
    }

    let warm_bytes = idx.to_bytes();

    // Cold rebuild of the same logical index: round-trip through parts,
    // whose cache starts empty, forcing the full repack on write.
    let cold = TurboQuantIndex::from_parts(
        Some(dim),
        idx.bit_width(),
        idx.len(),
        idx.packed_codes().to_vec(),
        idx.scales().to_vec(),
        idx.tqplus_shift().to_vec(),
        idx.tqplus_scale().to_vec(),
    )
    .unwrap();
    let cold_bytes = cold.to_bytes();
    assert_eq!(warm_bytes, cold_bytes, "incremental cache diverged from cold rebuild");

    // Search parity between the warm-cached index and its cold twin.
    let q = lcg_vectors(1, dim, 0x5EED);
    let a = idx.search(&q, 10);
    let b = cold.search(&q, 10);
    assert_eq!(a.indices, b.indices);
    assert_eq!(a.scores, b.scores);
}
