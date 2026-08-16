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
