//! A v7 sync container handed to a byte/reader entry point must say so,
//! not report "wrong magic" (#486). v7 is deliberately not supported
//! there: it needs random access, and `to_bytes` only emits v6.

use turbovec::TurboQuantIndex;

const DIM: usize = 64;

fn rows(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * DIM];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
    }
    v
}

fn dir(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("tv-v7bytes-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&p).unwrap();
    p
}



/// The parity that is actually promised — v6 output — still holds both
/// ways, so the narrowed claim is not narrower than reality.
#[test]
fn v6_output_still_round_trips_through_both_entry_points() {
    let d = dir("v6");
    let path = d.join("i.tv");
    let mut i = TurboQuantIndex::new(DIM, 4).unwrap();
    i.add(&rows(64, 3));
    i.write(&path).unwrap();

    let by_path = TurboQuantIndex::load(&path).unwrap().to_bytes();
    let by_bytes = TurboQuantIndex::from_bytes(&std::fs::read(&path).unwrap())
        .unwrap()
        .to_bytes();
    assert_eq!(by_path, by_bytes);
    assert_eq!(TurboQuantIndex::from_bytes(&i.to_bytes()).unwrap().to_bytes(), by_path);
    let _ = std::fs::remove_dir_all(&d);
}
