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

/// The standard oracle: the loaded index must be byte-identical in
/// memory to the live one (`to_bytes` equality) — search parity alone
/// is provably too weak (it missed a stale-row corruption during
/// review) — plus answer-parity as a readable failure mode.
fn search_parity(a: &TurboQuantIndex, b: &TurboQuantIndex, queries: &[f32], k: usize) {
    assert_eq!(
        a.to_bytes(),
        b.to_bytes(),
        "synced-then-loaded index is not byte-identical to the live one"
    );
    let ra = a.search(queries, k);
    let rb = b.search(queries, k);
    assert_eq!(ra.indices, rb.indices, "synced file answers differently");
    assert_eq!(ra.scores, rb.scores);
}

/// The adversarial-review reproduction (Gap 1): a shrink below a block
/// boundary and a re-add INTO the dipped region, both between the same
/// pair of syncs. Without the low-watermark rule the re-added rows are
/// covered by neither a segment nor the commit tail, and the file
/// loads successfully with stale vectors.
#[test]
fn remove_then_readd_across_a_block_boundary_between_syncs() {
    let path = temp("gap1");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 30)).unwrap();
    idx.add(&rows(64, 31)); // 2 whole blocks, no tail
    idx.sync(&path).unwrap();

    while idx.len() > 24 {
        idx.swap_remove(0);
    }
    idx.add(&rows(8, 32)); // slots 24..31: inside the dipped region
    idx.sync(&path).unwrap();
    let loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(idx.to_bytes(), loaded.to_bytes());

    // The milder variant that search parity alone cannot see.
    let path2 = temp("gap1b");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 33)).unwrap();
    idx.add(&rows(64, 34));
    idx.sync(&path2).unwrap();
    idx.swap_remove(10);
    idx.add(&rows(1, 35));
    idx.sync(&path2).unwrap();
    let loaded = TurboQuantIndex::load(&path2).unwrap();
    assert_eq!(idx.to_bytes(), loaded.to_bytes());
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
    // And the post-reload sync stayed incremental: it must not have
    // rewritten the whole segment range into the log.
    assert!(
        (after_bytes.len() - prefix.len()) < prefix.len() / 2,
        "the sync after a reload rewrote the index into the log"
    );
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

/// The bit-rot half of the crash contract, exhaustive over EVERY byte
/// of a committed file. Every committed byte below the final commit
/// record is covered by a CRC whose failure refuses the load. A flip
/// inside the final commit record itself is indistinguishable from a
/// torn sync, so the stated contract applies: the loader falls back to
/// the previous commit — exactly that state, verified byte-for-byte —
/// or refuses. Nothing else is ever served.
#[test]
fn bit_rot_in_any_committed_byte_is_never_served() {
    let path = temp("rot");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 19)).unwrap();
    idx.add(&rows(70, 20));
    idx.sync(&path).unwrap();
    let prev_mem = TurboQuantIndex::load(&path).unwrap().to_bytes();

    idx.add(&rows(20, 21));
    idx.swap_remove(5);
    idx.sync(&path).unwrap();
    let cur_mem = TurboQuantIndex::load(&path).unwrap().to_bytes();
    let file = std::fs::read(&path).unwrap();
    let cmt_start = (0..=file.len() - 4)
        .rev()
        .find(|&i| &file[i..i + 4] == b"CMT1")
        .unwrap();

    let rot = path.with_file_name("rotted.tv");
    for at in 0..file.len() {
        let mut bytes = file.clone();
        bytes[at] ^= 1 << (at % 8);
        std::fs::write(&rot, &bytes).unwrap();
        match TurboQuantIndex::load(&rot) {
            Err(_) => {}
            Ok(got) => {
                let got_mem = got.to_bytes();
                assert!(
                    at >= cmt_start,
                    "flip at {at} (below the final commit at {cmt_start}) loaded silently"
                );
                assert_eq!(
                    got_mem, prev_mem,
                    "flip at {at} in the final commit served neither a refusal nor \
                     exactly the previous commit"
                );
                assert_ne!(got_mem, cur_mem, "sanity: prev and cur states differ");
            }
        }
    }
}

/// Gap 7: sync verifies its cursor against the file it points at.
/// sync -> write (v6) -> sync: the middle write replaces the container;
/// the next sync must notice and rebuild v7 rather than appending v7
/// records into a v6 file.
#[test]
fn sync_then_write_then_sync_rebuilds_the_container() {
    let path = temp("swsync");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 30)).unwrap();
    idx.add(&rows(50, 31));
    idx.sync(&path).unwrap();
    idx.write(&path).unwrap(); // v6 replaces the v7 file, cursor now stale
    idx.add(&rows(10, 32));
    idx.sync(&path).unwrap(); // must detect the swap and write v7 full
    let loaded = TurboQuantIndex::load(&path).unwrap();
    search_parity(&idx, &loaded, &rows(8, 995), 10);
}

/// Two indexes syncing one path: the second writer's full rewrite is
/// atomic and loadable, and the first writer's next sync refuses rather
/// than silently clobbering commits that are no longer its own.
#[test]
fn a_stale_cursor_refuses_to_clobber_another_writers_commits() {
    let path = temp("twowriters");
    let mut a = TurboQuantIndex::new(DIM, 4).unwrap();
    a.calibrate(&rows(1024, 33)).unwrap();
    a.add(&rows(40, 34));
    a.sync(&path).unwrap();

    // Writer B adopts the file and advances it.
    let mut b = TurboQuantIndex::load(&path).unwrap();
    b.add(&rows(20, 35));
    b.sync(&path).unwrap();

    // A's cursor now points into history that B superseded.
    a.add(&rows(5, 36));
    let err = a.sync(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("another writer"), "{err}");

    // Nothing was touched: the file still loads as B's state.
    let loaded = TurboQuantIndex::load(&path).unwrap();
    search_parity(&b, &loaded, &rows(8, 994), 10);
}

/// Trailing garbage from a torn sync (simulated) is shed by the next
/// sync from the matching cursor — never resurrected, never fatal.
#[test]
fn a_matching_cursor_sheds_torn_trailing_bytes() {
    let path = temp("shed");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 37)).unwrap();
    idx.add(&rows(40, 38));
    idx.sync(&path).unwrap();

    // A torn later sync: valid-looking prefix bytes, no commit.
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
    f.write_all(b"SEG1garbage-from-a-torn-sync\x00\x00\x00").unwrap();
    drop(f);

    idx.add(&rows(12, 39));
    idx.sync(&path).unwrap();
    let loaded = TurboQuantIndex::load(&path).unwrap();
    search_parity(&idx, &loaded, &rows(8, 993), 10);
}

/// Gap 3: sync shares write()'s temp-sibling protocol. A sync that
/// fails at the rename (destination occupied by a non-empty directory)
/// must remove its temp file — a crash-looping caller must not fill
/// the volume — and must leave the index unbound so a later sync to a
/// good path starts fresh.
#[test]
fn a_failed_first_sync_cleans_up_its_temp() {
    let path = temp("cleanup");
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("occupied"), b"x").unwrap();

    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 40)).unwrap();
    idx.add(&rows(40, 41));
    assert!(idx.sync(&path).is_err(), "rename over a non-empty dir must fail");

    let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp."))
        .collect();
    assert!(leftovers.is_empty(), "temp files leaked: {leftovers:?}");

    // The failed sync bound nothing; a good path syncs from scratch.
    let good = path.with_file_name("good.tv");
    idx.sync(&good).unwrap();
    let loaded = TurboQuantIndex::load(&good).unwrap();
    search_parity(&idx, &loaded, &rows(8, 992), 10);
}
