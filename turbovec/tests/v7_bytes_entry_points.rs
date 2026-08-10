//! A v7 sync container handed to a byte/reader entry point must say so,
//! not report "wrong magic" (#486). v7 is deliberately not supported
//! there: it needs random access, and `to_bytes` only emits v6.

use turbovec::{IdMapIndex, TurboQuantIndex};

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

#[test]
fn a_synced_tv_file_rejected_by_from_bytes_names_the_reason() {
    let d = dir("tv");
    let path = d.join("i.tv");
    let mut i = TurboQuantIndex::new(DIM, 4).unwrap();
    i.add(&rows(64, 1));
    i.sync(&path).unwrap();

    // The same file opens fine by path.
    assert_eq!(TurboQuantIndex::load(&path).unwrap().len(), 64);

    let bytes = std::fs::read(&path).unwrap();
    let err = TurboQuantIndex::from_bytes(&bytes).unwrap_err().to_string();
    assert!(err.contains("v7 sync container"), "unhelpful: {err}");
    assert!(err.contains("load("), "should point at load(path): {err}");
    assert!(!err.contains("wrong magic"), "still the generic message: {err}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_synced_tvim_file_rejected_by_from_bytes_names_the_reason() {
    let d = dir("tvim");
    let path = d.join("i.tvim");
    let mut i = IdMapIndex::new(DIM, 4).unwrap();
    i.add_with_ids(&rows(64, 2), &(0..64u64).collect::<Vec<_>>()).unwrap();
    i.sync(&path).unwrap();
    assert_eq!(IdMapIndex::load(&path).unwrap().len(), 64);

    let bytes = std::fs::read(&path).unwrap();
    let err = IdMapIndex::from_bytes(&bytes).unwrap_err().to_string();
    assert!(err.contains("v7 sync container"), "unhelpful: {err}");
    assert!(!err.contains("wrong magic"), "still the generic message: {err}");
    let _ = std::fs::remove_dir_all(&d);
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
