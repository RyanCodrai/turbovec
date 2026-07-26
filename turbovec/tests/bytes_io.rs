//! In-memory serialization API: `to_bytes` / `from_bytes` /
//! `write_to_writer` / `load_from_reader` on both index types, and the
//! generic `io::write_to` / `io::load_from` /
//! `io::write_id_map_to` / `io::load_id_map_from` entry points.
//!
//! Covers:
//! 1. Byte identity: `to_bytes` (and the generic io writers) produce
//!    exactly the bytes `write` puts in the file, for populated, empty,
//!    and lazy-uncommitted indexes.
//! 2. Round-trip: `from_bytes(to_bytes(x))` reproduces `x` — search
//!    parity, ids, dim/bit_width/len, and a re-serialization that is
//!    byte-identical again.
//! 3. Error parity with `load`: corrupt or drifted bytes fail
//!    `from_bytes` with the same error kind and message as the same
//!    bytes written to a file and passed to `load` (io_v4-style cases:
//!    wrong magic, v1 version byte, unsupported version, truncation,
//!    rotation drift, duplicate `.tvim` ids).

use std::path::PathBuf;

use turbovec::{io, IdMapIndex, TurboQuantIndex};

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

/// Deterministic vector source (same LCG family as tests/io_v4.rs).
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

// v4 header offsets shared with tests/io_v4.rs.
const OFF_VERSION: usize = 4;
const OFF_HASH: usize = 18;
const OFF_PROBES: usize = 26;
const N_PROBES: usize = 64;

fn build_index() -> TurboQuantIndex {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add_2d(&lcg_vectors(N, DIM, VEC_SEED), DIM).unwrap();
    idx
}

fn build_id_map() -> IdMapIndex {
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    let ids: Vec<u64> = (0..N as u64).map(|i| 1000 + i).collect();
    idx.add_with_ids(&lcg_vectors(N, DIM, VEC_SEED), &ids).unwrap();
    idx
}

// ---------------------------------------------------------------------------
// 1. Byte identity with the file writers
// ---------------------------------------------------------------------------

#[test]
fn tv_to_bytes_is_byte_identical_to_write_file() {
    let dir = temp_dir("tv-byte-identity");
    let path = dir.join("index.tv");
    let idx = build_index();
    idx.write(&path).unwrap();
    let file_bytes = std::fs::read(&path).unwrap();

    assert_eq!(idx.to_bytes(), file_bytes, "to_bytes must equal the .tv file bytes");

    // The generic io writer (which rebuilds the rotation for the
    // fingerprint rather than using the index's cache) must produce the
    // same bytes too — the fingerprint is a deterministic function of dim.
    let mut via_io = Vec::new();
    io::write_to(
        &mut via_io,
        idx.bit_width(),
        idx.dim(),
        idx.len(),
        idx.packed_codes(),
        idx.scales(),
        idx.tqplus_shift(),
        idx.tqplus_scale(),
    )
    .unwrap();
    assert_eq!(via_io, file_bytes, "io::write_to must equal the .tv file bytes");

    // write_to_writer is the same serialization behind a Write sink.
    let mut via_writer = Vec::new();
    idx.write_to_writer(&mut via_writer).unwrap();
    assert_eq!(via_writer, file_bytes);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tvim_to_bytes_is_byte_identical_to_write_file() {
    let dir = temp_dir("tvim-byte-identity");
    let path = dir.join("index.tvim");
    let idx = build_id_map();
    idx.write(&path).unwrap();
    let file_bytes = std::fs::read(&path).unwrap();

    assert_eq!(idx.to_bytes(), file_bytes, "to_bytes must equal the .tvim file bytes");

    let mut via_writer = Vec::new();
    idx.write_to_writer(&mut via_writer).unwrap();
    assert_eq!(via_writer, file_bytes);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn empty_and_lazy_indexes_round_trip_byte_identically() {
    let dir = temp_dir("empty-lazy-bytes");

    // Eager but empty: dim committed, no vectors, zero fingerprint.
    let eager = TurboQuantIndex::new(DIM, 4).unwrap();
    let path = dir.join("eager.tv");
    eager.write(&path).unwrap();
    assert_eq!(eager.to_bytes(), std::fs::read(&path).unwrap());
    let back = TurboQuantIndex::from_bytes(&eager.to_bytes()).unwrap();
    assert_eq!(back.dim_opt(), Some(DIM));
    assert_eq!(back.len(), 0);

    // Lazy uncommitted: dim=0 sentinel.
    let lazy = IdMapIndex::new_lazy(3).unwrap();
    let path = dir.join("lazy.tvim");
    lazy.write(&path).unwrap();
    assert_eq!(lazy.to_bytes(), std::fs::read(&path).unwrap());
    let back = IdMapIndex::from_bytes(&lazy.to_bytes()).unwrap();
    assert_eq!(back.dim_opt(), None);
    assert_eq!(back.bit_width(), 3);
    assert_eq!(back.len(), 0);
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// 2. Round-trip equality
// ---------------------------------------------------------------------------

#[test]
fn tv_from_bytes_round_trip_search_parity() {
    let idx = build_index();
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let before = idx.search(&queries, 5);

    let back = TurboQuantIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(back.len(), idx.len());
    assert_eq!(back.dim(), idx.dim());
    assert_eq!(back.bit_width(), idx.bit_width());
    let after = back.search(&queries, 5);
    assert_eq!(before.scores, after.scores, "scores must survive the bytes round-trip");
    assert_eq!(before.indices, after.indices, "indices must survive the bytes round-trip");

    // Idempotence: re-serializing the deserialized index is
    // byte-identical (full state equality at the wire level).
    assert_eq!(back.to_bytes(), idx.to_bytes());
}

#[test]
fn tvim_from_bytes_round_trip_search_and_id_parity() {
    let idx = build_id_map();
    let queries = lcg_vectors(4, DIM, QUERY_SEED);
    let (scores_before, ids_before) = idx.search(&queries, 5);

    let back = IdMapIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(back.len(), idx.len());
    let (scores_after, ids_after) = back.search(&queries, 5);
    assert_eq!(scores_before, scores_after);
    assert_eq!(ids_before, ids_after);
    for id in 1000..1000 + N as u64 {
        assert!(back.contains(id), "id {id} must survive the bytes round-trip");
    }
    assert_eq!(back.to_bytes(), idx.to_bytes());

    // The generic reader sees the same payload.
    let bytes = idx.to_bytes();
    let via_reader = IdMapIndex::load_from_reader(&mut &bytes[..]).unwrap();
    assert_eq!(via_reader.len(), idx.len());
}

#[test]
fn io_generic_id_map_round_trip_matches_file_load() {
    // The raw io-module pair round-trips the same parts the file pair
    // does (the #70 ask: no tmpfile needed to (de)serialize a .tvim
    // payload held in memory).
    let idx = build_id_map();
    let mut buf = Vec::new();
    io::write_id_map_to(
        &mut buf,
        idx.bit_width(),
        idx.dim(),
        idx.len(),
        // Round-trip through the accessors like an external embedder would.
        TurboQuantIndex::from_bytes(&build_index().to_bytes()).unwrap().packed_codes(),
        &vec![1.0; N],
        &[],
        &[],
        &(0..N as u64).collect::<Vec<_>>(),
    )
    .unwrap();
    let (bit_width, dim, n_vectors, _codes, scales, _shift, _scale, slot_to_id) =
        io::load_id_map_from(&mut &buf[..]).unwrap();
    assert_eq!((bit_width, dim, n_vectors), (4, DIM, N));
    assert_eq!(scales, vec![1.0; N]);
    assert_eq!(slot_to_id, (0..N as u64).collect::<Vec<_>>());

    // And io::load_from parses a .tv byte payload.
    let tv_bytes = build_index().to_bytes();
    let (bit_width, dim, n_vectors, ..) = io::load_from(&mut &tv_bytes[..]).unwrap();
    assert_eq!((bit_width, dim, n_vectors), (4, DIM, N));
}

// ---------------------------------------------------------------------------
// 3. Error parity with `load` on corrupt / drifted bytes
// ---------------------------------------------------------------------------

/// Assert `from_bytes` and `load` fail identically on the same payload.
fn assert_same_error_tv(dir: &PathBuf, name: &str, bytes: &[u8]) -> std::io::Error {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    let from_bytes_err =
        TurboQuantIndex::from_bytes(bytes).expect_err("corrupt bytes must not deserialize");
    let load_err = TurboQuantIndex::load(&path).expect_err("corrupt file must not load");
    assert_eq!(from_bytes_err.kind(), load_err.kind(), "{name}: error kind must match load");
    assert_eq!(
        from_bytes_err.to_string(),
        load_err.to_string(),
        "{name}: error message must match load",
    );
    from_bytes_err
}

#[test]
fn tv_from_bytes_rejects_corrupt_bytes_with_same_errors_as_load() {
    let dir = temp_dir("tv-corrupt-parity");
    let good = build_index().to_bytes();

    // Wrong magic.
    let mut wrong_magic = good.clone();
    wrong_magic[0] = b'X';
    let e = assert_same_error_tv(&dir, "wrong-magic.tv", &wrong_magic);
    assert!(e.to_string().contains("wrong magic"), "got: {e}");

    // v1 sentinel (bare bit_width first byte).
    let mut v1 = good.clone();
    v1[0] = 4; // in 2..=4 → the targeted v1 error
    let e = assert_same_error_tv(&dir, "v1.tv", &v1);
    assert!(e.to_string().contains("turbovec ≤ 0.4.3"), "got: {e}");

    // Unsupported version byte.
    let mut future = good.clone();
    future[OFF_VERSION] = 9;
    let e = assert_same_error_tv(&dir, "future.tv", &future);
    assert!(e.to_string().contains("unsupported .tv format version"), "got: {e}");

    // Truncation.
    let truncated = &good[..good.len() - 7];
    let e = assert_same_error_tv(&dir, "truncated.tv", truncated);
    assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);

    // Rotation drift: shift every probe far out of tolerance and break
    // the hash (the io_v4 drift construction).
    let mut drifted = good.clone();
    drifted[OFF_HASH] ^= 0xFF;
    for i in 0..N_PROBES {
        let o = OFF_PROBES + 4 * i;
        let p = f32::from_le_bytes(drifted[o..o + 4].try_into().unwrap());
        drifted[o..o + 4].copy_from_slice(&(p + 1.0).to_le_bytes());
    }
    let e = assert_same_error_tv(&dir, "drifted.tv", &drifted);
    assert!(e.to_string().contains("rotation drift"), "got: {e}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tvim_from_bytes_rejects_duplicate_ids_like_load() {
    let dir = temp_dir("tvim-dup-parity");
    let mut idx = IdMapIndex::new(8, 4).unwrap();
    idx.add_with_ids(&lcg_vectors(3, 8, VEC_SEED), &[7, 8, 9]).unwrap();
    let mut bytes = idx.to_bytes();

    // The id table is the trailing 3 × u64; overwrite id 9 with 7.
    let n = bytes.len();
    bytes[n - 8..].copy_from_slice(&7u64.to_le_bytes());

    let path = dir.join("dup.tvim");
    std::fs::write(&path, &bytes).unwrap();
    let from_bytes_err = IdMapIndex::from_bytes(&bytes).expect_err("duplicate ids must not load");
    let load_err = IdMapIndex::load(&path).expect_err("duplicate ids must not load");
    assert_eq!(from_bytes_err.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(from_bytes_err.to_string(), load_err.to_string());
    assert!(from_bytes_err.to_string().contains("duplicate ids"), "got: {from_bytes_err}");

    // A .tv payload fed to the .tvim reader is the wrong-magic error.
    let tv_bytes = build_index().to_bytes();
    let e = IdMapIndex::from_bytes(&tv_bytes).expect_err(".tv bytes are not a TVIM payload");
    assert!(e.to_string().contains("not a TVIM file"), "got: {e}");
    std::fs::remove_dir_all(&dir).ok();
}
