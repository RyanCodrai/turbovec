//! The byte and path entry points must agree, whichever way an index
//! travels.
//!
//! This file was written when v7 was refused here (#486) — the reasoning
//! being that it needs random access and `to_bytes` only emitted v6.
//! Both halves of that are gone: the v7 parser works from a slice, and
//! every entry point now emits v7. What is left worth pinning is the
//! parity itself.

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



/// A file loaded from a path and the same bytes loaded directly must
/// produce the same index, and re-serializing either must reproduce the
/// same image.
#[test]
fn path_and_byte_entry_points_agree_on_a_v7_image() {
    let d = dir("parity");
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
    assert_eq!(&by_path[..4], b"TV7\0", "every entry point emits v7");
    let _ = std::fs::remove_dir_all(&d);
}
