//! `sync()` and the v7 append-only container.
//!
//! The contract under test: a sync writes bytes proportional to what
//! changed; a crash at any byte of a sync recovers the previous commit
//! exactly; nothing committed is ever overwritten (pinned literally, by
//! byte-comparing the file prefix across syncs); and a synced-then-
//! loaded index answers identically to the in-memory one it mirrors.

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

fn temp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-syncv7-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p.push("index.tv");
    p
}

fn search_parity(a: &TurboQuantIndex, b: &TurboQuantIndex, queries: &[f32], k: usize) {
    let ra = a.search(queries, k);
    let rb = b.search(queries, k);
    assert_eq!(ra.indices, rb.indices, "synced file answers differently");
    assert_eq!(ra.scores, rb.scores);
}

/// The whole lifecycle, interleaved: odd-size adds, removals hitting
/// committed rows, syncs between them, one reload in the middle to
/// prove the cursor survives a load. After every sync the file loads to
/// an index that answers exactly like the live one.
#[test]
fn interleaved_adds_removes_and_syncs_round_trip() {
    let path = temp("lifecycle");
    let queries = rows(16, 999);

    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 1)).unwrap();
    idx.add(&rows(100, 2));
    idx.sync(&path).unwrap();
    let full_size = std::fs::metadata(&path).unwrap().len();

    // Incremental round: 37 more rows, a removal inside the committed
    // region, a pure pop, and a removal whose filler is an unsynced row.
    idx.add(&rows(37, 3));
    idx.swap_remove(5);
    idx.swap_remove(idx.len() - 1);
    idx.swap_remove(70);
    let before = std::fs::metadata(&path).unwrap().len();
    idx.sync(&path).unwrap();
    let grew = std::fs::metadata(&path).unwrap().len() - before;
    assert!(
        grew < full_size / 2,
        "an incremental sync of ~37 rows wrote {grew} bytes against a \
         {full_size}-byte full file"
    );
    let loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.len(), idx.len());
    search_parity(&idx, &loaded, &queries, 10);
    assert_eq!(loaded.calibration_state(), idx.calibration_state());
    assert_eq!(loaded.tqplus_shift(), idx.tqplus_shift());

    // Continue from the RELOADED index: its cursor must let the next
    // sync append rather than rewrite.
    let mut idx = loaded;
    idx.add(&rows(64, 4));
    idx.swap_remove(0);
    let before = std::fs::metadata(&path).unwrap().len();
    let prefix: Vec<u8> = std::fs::read(&path).unwrap();
    idx.sync(&path).unwrap();
    let after_bytes = std::fs::read(&path).unwrap();
    assert!(after_bytes.len() as u64 > before, "sync did not append");
    // THE format rule, pinned literally: nothing committed is ever
    // overwritten — the old file is a byte-identical prefix of the new.
    assert_eq!(
        &after_bytes[..prefix.len()],
        &prefix[..],
        "a sync rewrote committed bytes"
    );
    let loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.len(), idx.len());
    search_parity(&idx, &loaded, &queries, 10);
}

/// Heavy churn: shrink far below the synced watermark, then grow back.
/// The dead committed region must be repaired by later segments.
#[test]
fn shrink_below_the_watermark_then_regrow() {
    let path = temp("shrink");
    let queries = rows(8, 998);
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 5)).unwrap();
    idx.add(&rows(200, 6));
    idx.sync(&path).unwrap();
    while idx.len() > 40 {
        idx.swap_remove(idx.len() / 2);
    }
    idx.sync(&path).unwrap();
    let loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.len(), 40);
    search_parity(&idx, &loaded, &queries, 5);

    idx.add(&rows(100, 7));
    idx.sync(&path).unwrap();
    let loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.len(), 140);
    search_parity(&idx, &loaded, &queries, 10);
}

/// A calibrate between syncs forces a compaction: the file is rewritten
/// (it may shrink, and the prefix changes), and everything still loads.
#[test]
fn calibrate_between_syncs_compacts() {
    let path = temp("compact");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 8)).unwrap();
    idx.add(&rows(150, 9));
    idx.sync(&path).unwrap();
    for _ in 0..30 {
        idx.swap_remove(0);
        idx.add(&rows(1, 10));
        idx.sync(&path).unwrap();
    }
    let churned = std::fs::metadata(&path).unwrap().len();

    idx.calibrate(&rows(1024, 11)).unwrap();
    idx.sync(&path).unwrap();
    let compacted = std::fs::metadata(&path).unwrap().len();
    assert!(
        compacted < churned,
        "the post-calibrate sync did not compact ({compacted} >= {churned})"
    );
    let loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.len(), idx.len());
    assert_eq!(loaded.tqplus_shift(), idx.tqplus_shift());
    search_parity(&idx, &loaded, &rows(8, 997), 10);
}

/// Sync to a v6 file's path: the first sync replaces it with v7, and a
/// v6 file still loads through the same `load`.
#[test]
fn v6_files_still_load_and_sync_forward() {
    let path = temp("v6fwd");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 12)).unwrap();
    idx.add(&rows(90, 13));
    idx.write(&path).unwrap();

    let mut loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.len(), 90);
    loaded.add(&rows(10, 14));
    loaded.sync(&path).unwrap();
    let again = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(again.len(), 100);
    search_parity(&loaded, &again, &rows(8, 996), 10);
}

/// The crash-safety contract, ported from the design prototype: a crash
/// at ANY byte of a sync recovers a valid prior commit — never garbage,
/// never a partial state. Exhaustive over the final sync's region.
#[test]
fn a_crash_at_any_byte_of_a_sync_recovers_the_previous_commit() {
    let path = temp("torn");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 15)).unwrap();
    idx.add(&rows(100, 16));
    idx.sync(&path).unwrap();
    idx.add(&rows(20, 17));
    idx.sync(&path).unwrap();
    let pre = std::fs::read(&path).unwrap();
    let pre_n = idx.len();

    // The sync to tear: an append AND removals (patch records).
    idx.add(&rows(40, 18));
    idx.swap_remove(3);
    idx.swap_remove(50);
    idx.sync(&path).unwrap();
    let post = std::fs::read(&path).unwrap();
    let post_n = idx.len();
    assert!(post.len() > pre.len());

    let torn = path.with_file_name("torn.tv");
    for cut in pre.len()..post.len() {
        std::fs::write(&torn, &post[..cut]).unwrap();
        let r = TurboQuantIndex::load(&torn)
            .unwrap_or_else(|e| panic!("cut={cut}: torn file failed to load: {e}"));
        assert_eq!(
            r.len(),
            pre_n,
            "cut={cut}: recovered neither the previous commit nor a whole one"
        );
    }
    // And the whole file recovers the new state.
    assert_eq!(TurboQuantIndex::load(&path).unwrap().len(), post_n);

    // Bit-rot inside the committed area is detected, never silently
    // served: flip one byte of a committed segment and load must fail.
    // ...whether it lands in the superblock (calibration/codebook) or
    // inside a committed segment's codes.
    for flip_at in [400usize, 1200] {
        let mut rotted = post.clone();
        rotted[flip_at] ^= 0xFF;
        std::fs::write(&torn, &rotted).unwrap();
        assert!(
            TurboQuantIndex::load(&torn).is_err(),
            "corrupted committed byte at {flip_at} loaded silently"
        );
    }
}

/// Byte cost: a 32-row batch syncs in kilobytes; the full file is tens
/// of times larger. (The de-risked figure at this dim: ~1.3 KB codes +
/// commit overhead.)
#[test]
fn sync_cost_is_proportional_to_the_change() {
    let path = temp("cost");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 19)).unwrap();
    idx.add(&rows(2000, 20));
    idx.sync(&path).unwrap();
    let full = std::fs::metadata(&path).unwrap().len();

    idx.add(&rows(32, 21));
    let before = std::fs::metadata(&path).unwrap().len();
    idx.sync(&path).unwrap();
    let batch_delta = std::fs::metadata(&path).unwrap().len() - before;

    idx.swap_remove(7);
    let before = std::fs::metadata(&path).unwrap().len();
    idx.sync(&path).unwrap();
    let remove_delta = std::fs::metadata(&path).unwrap().len() - before;

    assert!(
        batch_delta * 20 < full,
        "32-row sync wrote {batch_delta}B against a {full}B file"
    );
    assert!(
        remove_delta < 1024,
        "one removal synced {remove_delta}B (must be under 1 KB)"
    );
}
