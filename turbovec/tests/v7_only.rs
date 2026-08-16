use turbovec::TurboQuantIndex;
fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * dim];
    let mut s = seed | 1;
    for x in v.iter_mut() { s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5; }
    v
}
#[test]
fn to_bytes_round_trips_through_v7() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&rows(200, dim, 1));
    let q = rows(3, dim, 9);
    let before = idx.search(&q, 5);

    let bytes = idx.to_bytes();
    assert_eq!(&bytes[..4], b"TV7\0", "to_bytes must emit a v7 image");
    let back = TurboQuantIndex::from_bytes(&bytes).expect("from_bytes");
    assert_eq!(back.len(), 200);
    let after = back.search(&q, 5);
    assert_eq!(before.indices, after.indices, "indices differ after byte round-trip");
    assert_eq!(before.scores, after.scores, "scores differ after byte round-trip");
}

#[test]
fn write_then_load_round_trips_through_v7() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&rows(200, dim, 1));
    let q = rows(3, dim, 9);
    let before = idx.search(&q, 5);

    let mut p = std::env::temp_dir();
    p.push(format!("tv7-{}.tv", std::process::id()));
    idx.write(&p).unwrap();
    let raw = std::fs::read(&p).unwrap();
    assert_eq!(&raw[..4], b"TV7\0", "write must emit a v7 file");
    // The file and the in-memory image are the same builder, so they must
    // agree everywhere except the per-image nonce.
    let img = idx.to_bytes();
    assert_eq!(img.len(), raw.len(), "file and to_bytes differ in length");

    let back = TurboQuantIndex::load(&p).unwrap();
    let after = back.search(&q, 5);
    assert_eq!(before.indices, after.indices);
    assert_eq!(before.scores, after.scores);
    std::fs::remove_file(&p).ok();
}

#[test]
fn a_pre_v7_file_is_refused_with_an_actionable_error() {
    let mut p = std::env::temp_dir();
    p.push(format!("tv6-{}.tv", std::process::id()));
    // A v6 header: magic + version byte.
    std::fs::write(&p, {
        let mut v = b"TVPI".to_vec();
        v.push(6);
        v.extend_from_slice(&[0u8; 64]);
        v
    })
    .unwrap();
    let err = TurboQuantIndex::load(&p).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("version 6"), "should name the version: {msg}");
    assert!(msg.contains("v7"), "should say what is supported: {msg}");
    assert!(msg.contains("convert") || msg.contains("Convert"), "should be actionable: {msg}");

    // And something that is not an index at all reads differently.
    std::fs::write(&p, b"\x7fELF\x02\x01\x01\x00").unwrap();
    let msg = TurboQuantIndex::load(&p).unwrap_err().to_string();
    assert!(msg.contains("not a turbovec index"), "got: {msg}");
    std::fs::remove_file(&p).ok();
}

#[test]
fn id_map_round_trips_through_v7_bytes_and_files() {
    let dim = 64;
    let mut m = turbovec::IdMapIndex::new(dim, 4).unwrap();
    let ids: Vec<u64> = (100..300).collect();
    m.add_with_ids(&rows(200, dim, 3), &ids).unwrap();
    let q = rows(2, dim, 7);
    let before = m.search(&q, 5);

    let bytes = m.to_bytes();
    assert_eq!(&bytes[..4], b"TV7\0", "IdMapIndex::to_bytes must emit v7");
    let back = turbovec::IdMapIndex::from_bytes(&bytes).unwrap();
    assert_eq!(back.search(&q, 5), before, "byte round-trip changed results");

    let mut p = std::env::temp_dir();
    p.push(format!("tv7im-{}.tvim", std::process::id()));
    m.write(&p).unwrap();
    assert_eq!(&std::fs::read(&p).unwrap()[..4], b"TV7\0", "write must emit v7");
    let loaded = turbovec::IdMapIndex::load(&p).unwrap();
    assert_eq!(loaded.search(&q, 5), before, "file round-trip changed results");
    assert!(loaded.contains(150));
    std::fs::remove_file(&p).ok();
}

/// The three properties that pulled against each other while making v7
/// the only format, all holding at once.
#[test]
fn snapshots_are_deterministic_and_syncs_still_detect_a_foreign_writer() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&rows(300, dim, 11));

    let dir = std::env::temp_dir().join(format!("tv7prop-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let a = dir.join("a.tv");
    let b = dir.join("b.tv");

    // 1. to_bytes is a pure function of index state.
    assert_eq!(idx.to_bytes(), idx.to_bytes(), "to_bytes is not deterministic");

    // 2. write() produces exactly those bytes — nonce included.
    idx.write(&a).unwrap();
    assert_eq!(
        std::fs::read(&a).unwrap(),
        idx.to_bytes(),
        "write() and to_bytes() disagree"
    );
    // Two snapshots of the same index are identical files.
    idx.write(&b).unwrap();
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());

    // 3. A sync still refuses to patch a file another writer replaced.
    let mut synced = TurboQuantIndex::new(dim, 4).unwrap();
    synced.add(&rows(300, dim, 12));
    let p = dir.join("s.tv");
    synced.sync(&p).unwrap();
    // A claimed file carries a real nonce.
    let nonce = u64::from_le_bytes(std::fs::read(&p).unwrap()[11..19].try_into().unwrap());
    assert_ne!(nonce, 0, "sync must claim the file it owns");
    // Someone else replaces it wholesale.
    let mut other = TurboQuantIndex::new(dim, 4).unwrap();
    other.add(&rows(300, dim, 13));
    other.sync(&p).unwrap();
    let err = synced.sync(&p).unwrap_err();
    assert!(
        err.to_string().contains("another writer"),
        "a replaced file must not be patched: {err}"
    );

    // And a snapshot is loaded unbound, so the first sync to it claims it.
    let mut loaded = TurboQuantIndex::load(&a).unwrap();
    loaded.sync(&a).unwrap();
    let nonce = u64::from_le_bytes(std::fs::read(&a).unwrap()[11..19].try_into().unwrap());
    assert_ne!(nonce, 0, "the first sync must claim an unclaimed snapshot");
    std::fs::remove_dir_all(&dir).ok();
}

/// A lazily-constructed index — no dimension committed, never added to —
/// still round-trips, as it did under v6. Saving a store before its first
/// write is a real workflow.
#[test]
fn a_lazy_index_round_trips_as_the_dim_sentinel() {
    let idx = TurboQuantIndex::new_lazy(4).unwrap();
    let bytes = idx.to_bytes();
    let back = TurboQuantIndex::from_bytes(&bytes).expect("lazy round-trip");
    assert_eq!(back.len(), 0);
    assert_eq!(back.dim_opt(), None, "a lazy index must reload lazy");

    // Still usable: the first add commits the dimension.
    let mut back = back;
    back.add_2d(&rows(4, 32, 5), 32).unwrap();
    assert_eq!(back.len(), 4);
    assert_eq!(back.dim_opt(), Some(32));

    // And through a file.
    let mut p = std::env::temp_dir();
    p.push(format!("tv7lazy-{}.tv", std::process::id()));
    TurboQuantIndex::new_lazy(4).unwrap().write(&p).unwrap();
    let loaded = TurboQuantIndex::load(&p).unwrap();
    assert_eq!(loaded.len(), 0);
    assert_eq!(loaded.dim_opt(), None);
    std::fs::remove_file(&p).ok();
}
