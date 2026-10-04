//! `IdMapIndex::sync` — ids ride inside the v7 block units and header
//! tail, and the standard oracle is `to_bytes` equality (which covers
//! the id table byte-for-byte) plus id-level answers.

use std::path::PathBuf;

use turbovec::{IdMapIndex, TurboQuantIndex};

const DIM: usize = 64;

fn rows(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * DIM];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
    }
    for row in v.chunks_mut(DIM) {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    v
}

fn temp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-syncv7idm-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p.push("index.tvim");
    p
}

fn parity(a: &IdMapIndex, b: &IdMapIndex, queries: &[f32], k: usize) {
    assert_eq!(
        a.to_bytes(),
        b.to_bytes(),
        "synced-then-loaded IdMapIndex is not byte-identical to the live one"
    );
    let (sa, ia) = a.search(queries, k);
    let (sb, ib) = b.search(queries, k);
    assert_eq!(ia, ib, "ids disagree after a round trip");
    assert_eq!(sa, sb);
}

#[test]
fn idmap_sync_round_trips_with_ids_agreeing() {
    let path = temp("roundtrip");
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 50)).unwrap();
    let ids: Vec<u64> = (0..70u64).map(|i| i * 1000 + 7).collect();
    idx.add_with_ids(&rows(70, 51), &ids).unwrap();
    idx.sync(&path).unwrap();
    parity(&idx, &IdMapIndex::load(&path).unwrap(), &rows(8, 990), 10);

    // Interleave removals (middle, popping the tail, re-adding a freed
    // id) with adds across several syncs.
    assert!(idx.remove(7));
    assert!(idx.remove(34_007));
    idx.add_with_ids(&rows(3, 52), &[900_001, 900_002, 7]).unwrap();
    idx.sync(&path).unwrap();
    parity(&idx, &IdMapIndex::load(&path).unwrap(), &rows(8, 989), 10);

    assert!(idx.remove(900_002));
    assert!(idx.remove(69_007));
    idx.sync(&path).unwrap();
    let loaded = IdMapIndex::load(&path).unwrap();
    parity(&idx, &loaded, &rows(8, 988), 10);
    assert!(loaded.contains(7) && loaded.contains(900_001));
    assert!(!loaded.contains(900_002) && !loaded.contains(69_007));
}

#[test]
fn a_loaded_idmap_syncs_forward_incrementally() {
    let path = temp("forward");
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 53)).unwrap();
    idx.add_with_ids(&rows(64, 54), &(0..64u64).collect::<Vec<_>>()).unwrap();
    idx.sync(&path).unwrap();
    let full = std::fs::metadata(&path).unwrap().len();

    let mut loaded = IdMapIndex::load(&path).unwrap();
    loaded.add_with_ids(&rows(2, 55), &[100, 101]).unwrap();
    assert!(loaded.remove(3));
    loaded.sync(&path).unwrap();
    let grown = std::fs::metadata(&path).unwrap().len();
    assert!(
        grown - full < full / 2,
        "post-load sync appended {} bytes to a {full}-byte file",
        grown - full
    );
    parity(&loaded, &IdMapIndex::load(&path).unwrap(), &rows(8, 987), 10);
}

#[test]
fn the_two_index_types_refuse_each_others_sync_files() {
    let plain = temp("plain");
    let mut t = TurboQuantIndex::new(DIM, 4).unwrap();
    t.calibrate(&rows(1024, 56)).unwrap();
    t.add(&rows(10, 57));
    t.sync(&plain).unwrap();
    let err = IdMapIndex::load(&plain).unwrap_err();
    assert!(err.to_string().contains("TurboQuantIndex"), "{err}");

    let mapped = temp("mapped");
    let mut m = IdMapIndex::new(DIM, 4).unwrap();
    m.calibrate(&rows(1024, 58)).unwrap();
    m.add_with_ids(&rows(10, 59), &(0..10u64).collect::<Vec<_>>()).unwrap();
    m.sync(&mapped).unwrap();
    let err = TurboQuantIndex::load(&mapped).unwrap_err();
    assert!(err.to_string().contains("IdMapIndex"), "{err}");
}

/// An id-mapped index past the planes gate: its v8 file holds planes
/// units with the ids beside them, and incremental syncs (appends,
/// removals by id) reload with the ids agreeing.
#[test]
fn an_idmap_past_the_gate_syncs_planes_units() {
    let path = temp("idmap-planes");
    let queries = rows(8, 999);
    let n = 32_768 + 50;
    let ids: Vec<u64> = (0..n as u64).map(|i| i * 5 + 3).collect();
    let mut m = IdMapIndex::new(DIM, 4).unwrap();
    m.add_with_ids(&rows(n, 2), &ids).unwrap();
    m.sync(&path).unwrap();
    let raw = std::fs::read(&path).unwrap();
    assert_eq!(&raw[..4], b"TV8\0");
    assert_eq!(raw[7], 1, "planes units");
    let loaded = IdMapIndex::load(&path).unwrap();
    parity(&m, &loaded, &queries, 10);

    m.add_with_ids(&rows(40, 3), &(0..40).map(|i| 1_000_000 + i).collect::<Vec<u64>>()).unwrap();
    assert!(m.remove(ids[7]));
    assert!(m.remove(ids[32_768 + 2]));
    let before = std::fs::metadata(&path).unwrap().len();
    m.sync(&path).unwrap();
    let after = std::fs::metadata(&path).unwrap().len();
    assert!(after.saturating_sub(before) < before / 2, "the sync rewrote the file");
    let loaded = IdMapIndex::load(&path).unwrap();
    parity(&m, &loaded, &queries, 10);
    let _ = std::fs::remove_file(&path);
}
