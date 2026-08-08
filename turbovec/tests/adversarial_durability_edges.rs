//! Edge-condition durability probes: the header op-capacity boundary,
//! compaction, hostile destinations, and failure paths that must leave
//! the previous commit intact.

use std::path::PathBuf;

use turbovec::TurboQuantIndex;

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

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-advedge-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p
}

/// Distinct dirty slots on both sides of `MAX_OPS`: below it the header
/// carries them all, above it the sync falls back to a full rewrite.
/// Both must round-trip, and the header must not run past its slot.
#[test]
fn the_header_op_capacity_boundary_round_trips() {
    for n_dirty in [1023usize, 1024, 1025] {
        let dir = temp_dir(&format!("cap{n_dirty}"));
        let path = dir.join("index.tv");
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 60)).unwrap();
        // Well above the dirty count so no removal is popped by the
        // shrink and every one lands as an op in a committed block.
        idx.add(&rows(4096, 61));
        idx.sync(&path).unwrap();

        // Remove the tail each time so the hole is the only dirty slot
        // below the watermark, giving exactly `n_dirty` distinct ops.
        for i in 0..n_dirty {
            idx.swap_remove(i);
        }
        idx.sync(&path).unwrap();
        let back = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(
            back.to_bytes(),
            idx.to_bytes(),
            "{n_dirty} ops: reload does not match the live index"
        );

        // And the file stays usable for further incremental syncs.
        idx.add(&rows(40, 62));
        idx.swap_remove(7);
        idx.sync(&path).unwrap();
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes());
    }
}

/// A recalibration rewrites every stored code, so the sync compacts via
/// temp file + atomic rename. The old file must remain a complete index
/// until the rename, and no temp may survive a successful compaction.
#[test]
fn a_compaction_is_atomic_and_leaves_no_debris() {
    let dir = temp_dir("compact");
    let path = dir.join("index.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 63)).unwrap();
    idx.add(&rows(500, 64));
    idx.sync(&path).unwrap();
    let pre = std::fs::read(&path).unwrap();
    let pre_state = TurboQuantIndex::load(&path).unwrap().to_bytes();

    idx.calibrate(&rows(1024, 65)).unwrap();
    idx.sync(&path).unwrap();
    let post_state = TurboQuantIndex::load(&path).unwrap().to_bytes();
    assert_ne!(pre_state, post_state, "the recalibration changed nothing");
    assert_eq!(post_state, idx.to_bytes());

    let debris: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n != "index.tv")
        .collect();
    assert!(debris.is_empty(), "compaction left debris: {debris:?}");

    // A crash before the rename leaves exactly the old file; it must
    // still load and still sync forward.
    std::fs::write(&path, &pre).unwrap();
    let mut recovered = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(recovered.to_bytes(), pre_state);
    recovered.add(&rows(10, 66));
    recovered.sync(&path).unwrap();
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), recovered.to_bytes());
}

/// Incremental syncs write through a symlinked destination in place,
/// while a compaction renames over it. Whatever the choice, it must be
/// the same one every time — otherwise a compaction silently detaches
/// the link and every reader still following it sees a frozen index.
#[cfg(unix)]
#[test]
fn a_symlinked_destination_behaves_the_same_across_sync_and_compaction() {
    let dir = temp_dir("symlink");
    let target = dir.join("real.tv");
    let link = dir.join("link.tv");
    std::os::unix::fs::symlink(&target, &link).unwrap();

    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 67)).unwrap();
    idx.add(&rows(100, 68));
    idx.sync(&link).unwrap();
    let after_create = std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink();

    idx.add(&rows(10, 69));
    idx.sync(&link).unwrap();
    let after_incremental = std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink();

    idx.calibrate(&rows(1024, 70)).unwrap();
    idx.sync(&link).unwrap();
    let after_compaction = std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink();

    assert_eq!(
        (after_create, after_incremental),
        (after_incremental, after_compaction),
        "the destination alternates between the symlink and its target across \
         sync kinds (create={after_create}, incremental={after_incremental}, \
         compaction={after_compaction}); readers of the target see a frozen index"
    );
    assert_eq!(TurboQuantIndex::load(&link).unwrap().to_bytes(), idx.to_bytes());
}

/// A sync that cannot write must leave the previous commit intact and
/// must leave the index able to recover on the next attempt.
#[cfg(unix)]
#[test]
fn an_unwritable_destination_leaves_the_previous_commit_and_recovers() {
    use std::os::unix::fs::PermissionsExt;
    let dir = temp_dir("readonly");
    let path = dir.join("index.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 71)).unwrap();
    idx.add(&rows(100, 72));
    idx.sync(&path).unwrap();
    let good = TurboQuantIndex::load(&path).unwrap().to_bytes();

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
    idx.add(&rows(5, 73));
    let err = idx.sync(&path).unwrap_err();
    assert_eq!(
        TurboQuantIndex::load(&path).unwrap().to_bytes(),
        good,
        "a refused sync ({err}) damaged the committed file"
    );

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    idx.sync(&path).expect("the next sync must recover");
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes());
}

/// Arbitrary junk sitting at the destination must be replaced wholesale,
/// never treated as a container to sync into.
#[test]
fn junk_at_the_destination_is_replaced_not_extended() {
    for junk in [
        Vec::new(),
        b"TV7\0".to_vec(),
        b"TV7\0\x01\x04\x00".to_vec(),
        vec![0xABu8; 300_000],
    ] {
        let dir = temp_dir("junk");
        let path = dir.join("index.tv");
        std::fs::write(&path, &junk).unwrap();
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.calibrate(&rows(1024, 74)).unwrap();
        idx.add(&rows(100, 75));
        idx.sync(&path).unwrap();
        assert_eq!(
            TurboQuantIndex::load(&path).unwrap().to_bytes(),
            idx.to_bytes(),
            "junk of {} bytes was not cleanly replaced",
            junk.len()
        );
        idx.add(&rows(3, 76));
        idx.sync(&path).unwrap();
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes());
    }
}

/// Losing the destination's directory mid-life must surface as an error,
/// not as a silent no-op that reports success.
#[test]
fn a_vanished_destination_directory_is_an_error_then_recoverable() {
    let dir = temp_dir("vanish");
    let path = dir.join("sub").join("index.tv");
    std::fs::create_dir(path.parent().unwrap()).unwrap();
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 77)).unwrap();
    idx.add(&rows(100, 78));
    idx.sync(&path).unwrap();

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    idx.add(&rows(5, 79));
    assert!(idx.sync(&path).is_err(), "sync into a deleted directory reported success");

    std::fs::create_dir(path.parent().unwrap()).unwrap();
    idx.sync(&path).expect("recreating the directory must let the sync recover");
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes());
}
