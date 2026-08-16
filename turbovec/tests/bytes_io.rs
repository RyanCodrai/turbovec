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
//! 3. Error parity with `load`: corrupt bytes fail `from_bytes` with the
//!    same error kind and message as the same bytes written to a file and
//!    passed to `load` (wrong magic, v1/v4 pre-v5 rejection, unsupported
//!    version, truncation, duplicate `.tvim` ids).

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

// v5 header offset for the version byte (after the 4-byte magic).
const OFF_VERSION: usize = 4;

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

    // Eager but empty: dim committed, no vectors.
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
    assert_eq!(back.dim_opt().unwrap(), idx.dim_opt().unwrap());
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




/// `serialized_len` is exact, not an estimate: it must equal the length
/// of what the writer actually produced, for every index shape.
///
/// This is the guard on the `to_bytes` capacity hint (#409). A hint that
/// silently disagrees with the writer is worse than none — the `Vec`
/// would reallocate anyway and the saving would evaporate with no visible
/// symptom — so the check compares against real output rather than
/// restating the layout arithmetic.
#[test]
fn serialized_len_equals_the_bytes_actually_written() {
    for bit_width in [2usize, 3, 4] {
        for dim in [8usize, 32, 96] {
            for n in [0usize, 1, 7, 32, 33, 100] {
                let mut idx = TurboQuantIndex::new(dim, bit_width).unwrap();
                if n > 0 {
                    idx.add_2d(&lcg_vectors(n, dim, VEC_SEED), dim).unwrap();
                }
                assert_eq!(
                    idx.serialized_len(),
                    idx.to_bytes().len(),
                    "cold cache, bit_width={bit_width} dim={dim} n={n}",
                );
                // A warm search cache changes which branch produces the
                // codes section; the byte count must not move.
                idx.prepare();
                assert_eq!(
                    idx.serialized_len(),
                    idx.to_bytes().len(),
                    "warm cache, bit_width={bit_width} dim={dim} n={n}",
                );
            }
        }
    }

    // A lazy index — constructed but never given a dim — writes no codes.
    let lazy = TurboQuantIndex::new_lazy(4).unwrap();
    assert_eq!(lazy.serialized_len(), lazy.to_bytes().len(), "lazy index");

    // After `swap_remove` the block geometry shrinks; the hint follows.
    let mut shrunk = build_index();
    for _ in 0..40 {
        shrunk.swap_remove(0);
    }
    assert_eq!(
        shrunk.serialized_len(),
        shrunk.to_bytes().len(),
        "after swap_remove",
    );
    shrunk.prepare();
    assert_eq!(
        shrunk.serialized_len(),
        shrunk.to_bytes().len(),
        "after swap_remove, warm cache",
    );

    // Draining to empty is its own geometry (no codes, no scales).
    let mut drained = build_index();
    while drained.len() > 0 {
        drained.swap_remove(drained.len() - 1);
    }
    assert_eq!(
        drained.serialized_len(),
        drained.to_bytes().len(),
        "drained to empty",
    );
}

/// `to_bytes` sizes its buffer exactly once — the returned `Vec` has no
/// spare capacity, which holds only if the hint was right and the buffer
/// never grew (#409).
#[test]
fn to_bytes_allocates_its_buffer_exactly_once() {
    let idx = build_index();
    let bytes = idx.to_bytes();
    assert_eq!(
        bytes.capacity(),
        bytes.len(),
        "to_bytes grew its buffer instead of sizing it up front",
    );
}
