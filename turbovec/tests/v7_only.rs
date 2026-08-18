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
    assert!(
        msg.contains("convert"),
        "a v5/v6 file converts forward, so say so: {msg}"
    );

    // Versions the v5 rotation change made undecodable get different
    // advice: nothing can read them, so re-saving is not an option.
    std::fs::write(&p, {
        let mut v = b"TVPI".to_vec();
        v.push(3);
        v.extend_from_slice(&[0u8; 64]);
        v
    })
    .unwrap();
    let msg = TurboQuantIndex::load(&p).unwrap_err().to_string();
    assert!(msg.contains("version 3"), "should name the version: {msg}");
    assert!(msg.contains("rebuild"), "pre-v5 must say rebuild: {msg}");
    assert!(
        !msg.contains("save it again"),
        "pre-v5 cannot be re-saved by any build: {msg}"
    );

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

/// A superblock can claim anything; the loader must refuse an absurd one
/// promptly rather than sizing an allocation from it.
///
/// The dim guard is two conditions — "not a multiple of 8" and "past
/// MAX_DIM" — and only the first had coverage: every malformed-dim test
/// used a dim that was also misaligned. A huge *aligned* dim went
/// straight through to the geometry math, which is how a hostile file
/// turns a load into an allocation the size of its claim.
#[test]
fn a_superblock_claiming_an_absurd_dim_is_refused_not_allocated() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&rows(64, dim, 4));
    let mut bytes = idx.to_bytes();

    // dim lives at offset 7..11. 2^24 is a multiple of 8 and far past
    // MAX_DIM, so only the second half of the guard can reject it.
    bytes[7..11].copy_from_slice(&(1u32 << 24).to_le_bytes());

    let started = std::time::Instant::now();
    let err = TurboQuantIndex::from_bytes(&bytes).expect_err("an absurd dim must not load");
    assert!(
        err.to_string().contains("dim"),
        "should name the dim: {err}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "refusal should be immediate, not an allocation"
    );
}

/// An image from a different v7 revision is refused by revision, not by
/// whatever a length guess makes of its bytes.
///
/// Revision 1 is a real input now: it is what every build before the
/// unclaimed-nonce change wrote, and the nonce field means something
/// different there.
#[test]
fn a_foreign_v7_revision_is_named_rather_than_misread() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&rows(32, dim, 5));
    let mut bytes = idx.to_bytes();
    bytes[4] = 1; // the previous revision

    let err = TurboQuantIndex::from_bytes(&bytes).expect_err("revision 1 must not load");
    let msg = err.to_string();
    assert!(
        msg.contains("revision") || msg.contains("unsupported"),
        "should name the revision rather than report a truncation: {msg}"
    );
}

/// An index at exactly `MAX_DIM` is legal and must round-trip.
///
/// The loader's bound is `dim > MAX_DIM`; at `>=` the largest supported
/// index stops loading, which no other test would notice — every other
/// dim in the suite is far below the cap.
#[test]
fn an_index_at_exactly_max_dim_round_trips() {
    let dim = turbovec::MAX_DIM;
    let mut idx = TurboQuantIndex::new(dim, 2).unwrap();
    idx.add(&rows(32, dim, 6));

    let bytes = idx.to_bytes();
    let back = TurboQuantIndex::from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("dim {dim} is legal and must load: {e}"));
    assert_eq!(back.len(), 32);
    assert_eq!(back.dim_opt(), Some(dim));

    let q = rows(1, dim, 7);
    assert_eq!(back.search(&q, 5).indices, idx.search(&q, 5).indices);
}
