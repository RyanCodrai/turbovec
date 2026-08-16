use turbovec::TurboQuantIndex;
fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * dim];
    let mut s = seed | 1;
    for x in v.iter_mut() { s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5; }
    v
}
#[test]
fn alternating_write_and_sync_at_one_path() {
    let dim = 64;
    let dir = std::env::temp_dir().join(format!("zzalt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("i.tv");
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&rows(150, dim, 5));
    for round in 0..5u64 {
        idx.sync(&path).unwrap();
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes(), "r{round} sync/load");
        idx.add(&rows(9, dim, 60 + round));
        idx.sync(&path).unwrap();
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes(), "r{round} sync2/load");
        idx.write(&path).unwrap();
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes(), "r{round} write/load");
        idx.swap_remove(round as usize);
        idx.add(&rows(3, dim, 70 + round));
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn foreign_snapshot_over_a_synced_path_is_silently_clobbered() {
    let dim = 64;
    let dir = std::env::temp_dir().join(format!("zzfor-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("s.tv");
    let mut a = TurboQuantIndex::new(dim, 4).unwrap();
    a.add(&rows(300, dim, 12));
    a.sync(&p).unwrap();
    // Another, unrelated index snapshots over the same path.
    let mut other = TurboQuantIndex::new(dim, 4).unwrap();
    other.add(&rows(77, dim, 13));
    other.write(&p).unwrap();
    // Previously: Foreign -> error. Now?
    let r = a.sync(&p);
    println!("a.sync after foreign write -> {:?}", r.as_ref().map(|_| "Ok"));
    let back = TurboQuantIndex::load(&p).unwrap();
    println!("rows now {}", back.len());
    std::fs::remove_dir_all(&dir).ok();
}
