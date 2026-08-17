//! Write-path guarantees that survived the v7-only switch.
//!
//! `io_hardening.rs` was retired with the v5/v6 formats it mostly tested,
//! but four of its properties are about the *writer*, which v7 keeps:
//! the temp-file + atomic-rename protocol, both durability levels, and
//! not leaving temps behind. `Durability::Fast` in particular is public
//! API (`durable=False` in Python) and was left with no Rust coverage.

use std::path::PathBuf;

use turbovec::io::Durability;
use turbovec::TurboQuantIndex;

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-wp-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p
}

fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * dim];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
    }
    v
}

fn build(dim: usize, n: usize) -> TurboQuantIndex {
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&rows(n, dim, 1));
    idx
}

fn strays(dir: &PathBuf) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp."))
        .collect()
}

/// `Fast` trades only the fsync: same bytes, same atomic replace.
#[test]
fn fast_durability_writes_the_same_bytes_and_leaves_no_temp() {
    let dir = temp_dir("fast");
    let idx = build(64, 128);
    let durable = dir.join("durable.tv");
    let fast = dir.join("fast.tv");

    idx.write_with_durability(&durable, Durability::Durable).unwrap();
    idx.write_with_durability(&fast, Durability::Fast).unwrap();

    assert_eq!(
        std::fs::read(&durable).unwrap(),
        std::fs::read(&fast).unwrap(),
        "Fast and Durable must write identical bytes"
    );
    assert!(strays(&dir).is_empty(), "temp files left: {:?}", strays(&dir));

    let q = rows(2, 64, 7);
    assert_eq!(
        TurboQuantIndex::load(&fast).unwrap().search(&q, 5).indices,
        idx.search(&q, 5).indices
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The destination is replaced atomically, so an overwrite never leaves
/// the previous index half-written — and never leaves a temp behind.
#[test]
fn overwriting_replaces_atomically_and_cleans_up() {
    let dir = temp_dir("overwrite");
    let path = dir.join("index.tv");

    let first = build(64, 64);
    first.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    let second = build(64, 200);
    second.write(&path).unwrap();
    let after = std::fs::read(&path).unwrap();

    assert_ne!(before, after, "the overwrite must have taken effect");
    assert_eq!(TurboQuantIndex::load(&path).unwrap().len(), 200);
    assert!(strays(&dir).is_empty(), "temp files left: {:?}", strays(&dir));
    std::fs::remove_dir_all(&dir).ok();
}

/// A write to a path that cannot be created fails without disturbing
/// whatever is already there.
#[test]
fn a_failed_write_leaves_the_previous_file_intact() {
    let dir = temp_dir("failed");
    let path = dir.join("index.tv");
    let good = build(64, 96);
    good.write(&path).unwrap();
    let intact = std::fs::read(&path).unwrap();

    // A directory in place of the temp's parent: create_tmp cannot open.
    let missing = dir.join("nope").join("index.tv");
    assert!(good.write(&missing).is_err(), "writing into a missing dir must fail");

    assert_eq!(std::fs::read(&path).unwrap(), intact, "the good file changed");
    assert_eq!(TurboQuantIndex::load(&path).unwrap().len(), 96);
    assert!(strays(&dir).is_empty(), "temp files left: {:?}", strays(&dir));
    std::fs::remove_dir_all(&dir).ok();
}

/// Both durability levels round-trip an id-mapped index too.
#[test]
fn id_mapped_writes_round_trip_at_both_durability_levels() {
    let dir = temp_dir("idmap");
    let ids: Vec<u64> = (0..100u64).map(|i| i * 3 + 5).collect();
    let mut m = turbovec::IdMapIndex::new(64, 4).unwrap();
    m.add_with_ids(&rows(100, 64, 2), &ids).unwrap();

    for (name, d) in [("d.tvim", Durability::Durable), ("f.tvim", Durability::Fast)] {
        let p = dir.join(name);
        m.write_with_durability(&p, d).unwrap();
        let back = turbovec::IdMapIndex::load(&p).unwrap();
        assert_eq!(back.len(), 100);
        assert!(back.contains(5) && back.contains(5 + 3 * 99));
    }
    assert!(strays(&dir).is_empty(), "temp files left: {:?}", strays(&dir));
    std::fs::remove_dir_all(&dir).ok();
}
