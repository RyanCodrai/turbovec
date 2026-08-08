//! Second-round durability probes, aimed at what the first round did
//! not reach: repeated rollbacks through the same generation, readers
//! racing a writer, and load's effect on the file it reads.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use turbovec::{IdMapIndex, TurboQuantIndex};

const DIM: usize = 64;
const SECTOR: usize = 512;

fn rows_d(dim: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * dim];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
    }
    for row in v.chunks_mut(dim) {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    v
}

fn rows(n: usize, seed: u64) -> Vec<f32> {
    rows_d(DIM, n, seed)
}

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-adv2-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p
}

/// The file after a power cut in which every changed sector below
/// `limit` landed and none above it did.
///
/// This is barrier-faithful for any sync: the repair batch and the
/// commit header both live in the header region, and the repair runs
/// first, so "the whole header region landed, no data did" is a state a
/// real crash can leave. It is also the state that strands a commit
/// whose delta names units that never arrived.
fn land_below(before: &[u8], after: &[u8], limit: usize) -> Vec<u8> {
    let mut out = before.to_vec();
    for s in 0..after.len().div_ceil(SECTOR) {
        let (a, b) = (s * SECTOR, ((s + 1) * SECTOR).min(after.len()));
        if a >= limit {
            break;
        }
        if out.len() < b {
            out.resize(b, 0);
        }
        out[a..b].copy_from_slice(&after[a..b]);
    }
    out
}

/// First byte of the block-unit region, from the format's geometry.
fn unit0(dim: usize, bw: usize, kind: u8) -> usize {
    const MAX_OPS: usize = 1024;
    let nl = 1usize << bw;
    let sb = 23 + (nl - 1) * 4 + nl * 4 + 4 + dim * 8 + 4;
    let row = dim / (8 / bw);
    let id1 = if kind == 1 { 8 } else { 0 };
    let hdr = 16
        + 31 * (row + 4 + id1)
        + 4
        + MAX_OPS * (5 + 1 + row + 4 + id1)
        + 4
        + MAX_OPS * 4
        + 12
        + 4;
    sb + 2 * hdr
}

/// Roll the file back again and again, through the same generation each
/// time. Every cycle re-enters the state where the next sync reuses a
/// generation whose rejected header is still on disk, so the repair
/// barrier has to fire repeatedly — and a repair that only worked once
/// (or that left the slot in a state the next repair mis-measures) shows
/// up here.
#[test]
fn repeated_rollbacks_through_one_generation_never_serve_a_third_state() {
    let dir = temp_dir("cycles");
    let path = dir.join("index.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 1)).unwrap();
    idx.add(&rows(200, 2));
    idx.sync(&path).unwrap();
    idx.swap_remove(10);
    idx.sync(&path).unwrap();

    let mut committed = vec![TurboQuantIndex::load(&path).unwrap().to_bytes()];
    for cycle in 0..8u64 {
        let before = std::fs::read(&path).unwrap();
        let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();

        let mut live = TurboQuantIndex::load(&path).unwrap();
        assert_eq!(live.to_bytes(), state_a);
        live.add(&rows(1 + cycle as usize, 10 + cycle));
        live.swap_remove(3);
        live.sync(&path).unwrap();
        let after = std::fs::read(&path).unwrap();
        let state_b = live.to_bytes();
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), state_b);

        // Crash: the header region landed, the units did not.
        let crashed = land_below(&before, &after, unit0(DIM, 4, 0));
        std::fs::write(&path, &crashed).unwrap();
        let got = TurboQuantIndex::load(&path)
            .unwrap_or_else(|e| panic!("cycle {cycle}: unloadable ({e})"))
            .to_bytes();
        assert!(
            got == state_a || got == state_b,
            "cycle {cycle}: rolled back to a state that was never committed"
        );
        committed.push(got.clone());

        // And the recovered file must still sync forward.
        let mut fwd = TurboQuantIndex::load(&path).unwrap();
        fwd.add(&rows(2, 900 + cycle));
        fwd.sync(&path).expect("forward sync after rollback");
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), fwd.to_bytes());
    }
}

/// The same cycle for the id-mapped container, checking ids rather than
/// only bytes.
#[test]
fn repeated_idmap_rollbacks_keep_the_id_table_consistent() {
    let dir = temp_dir("idcycles");
    let path = dir.join("index.tvim");
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 3)).unwrap();
    let ids: Vec<u64> = (0..200u64).map(|i| i * 7 + 1).collect();
    idx.add_with_ids(&rows(200, 4), &ids).unwrap();
    idx.sync(&path).unwrap();
    assert!(idx.remove(71));
    idx.sync(&path).unwrap();

    let mut next = 10_000u64;
    for cycle in 0..6u64 {
        let before = std::fs::read(&path).unwrap();
        let a = IdMapIndex::load(&path).unwrap();
        let (state_a, len_a) = (a.to_bytes(), a.len());

        let mut live = IdMapIndex::load(&path).unwrap();
        let new: Vec<u64> = (0..3u64).map(|i| next + i).collect();
        next += 3;
        live.add_with_ids(&rows(3, 20 + cycle), &new).unwrap();
        live.sync(&path).unwrap();
        let after = std::fs::read(&path).unwrap();
        let (state_b, len_b) = (live.to_bytes(), live.len());

        let crashed = land_below(&before, &after, unit0(DIM, 4, 1));
        std::fs::write(&path, &crashed).unwrap();
        let got = IdMapIndex::load(&path).unwrap_or_else(|e| panic!("cycle {cycle}: {e}"));
        let bytes = got.to_bytes();
        assert!(
            bytes == state_a || bytes == state_b,
            "cycle {cycle}: a third state"
        );
        // Whichever survived, the id table must agree with it.
        let expect_len = if bytes == state_a { len_a } else { len_b };
        assert_eq!(got.len(), expect_len, "cycle {cycle}: id count disagrees");
        for &id in &new {
            assert_eq!(
                got.contains(id),
                bytes == state_b,
                "cycle {cycle}: id {id} presence disagrees with the adopted commit"
            );
        }
    }
}

/// `load` must not touch the file it reads. A loader that repaired,
/// normalized, or otherwise rewrote what it found would turn every
/// reader into a writer — and, after a rollback, would let a reader
/// destroy a commit the writer still needs.
#[test]
fn load_never_modifies_the_file() {
    let dir = temp_dir("readonly-load");
    let path = dir.join("index.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 5)).unwrap();
    idx.add(&rows(300, 6));
    idx.sync(&path).unwrap();
    idx.swap_remove(9);
    idx.add(&rows(40, 7));
    idx.sync(&path).unwrap();

    // Also over a rolled-back file, where the loader has a rejected
    // header in front of it and every reason to want to tidy up.
    let clean = std::fs::read(&path).unwrap();
    let mut live = TurboQuantIndex::load(&path).unwrap();
    live.add(&rows(50, 8));
    live.sync(&path).unwrap();
    let after = std::fs::read(&path).unwrap();
    let rolled = land_below(&clean, &after, unit0(DIM, 4, 0));

    for (label, image) in [("clean", clean), ("rolled-back", rolled)] {
        std::fs::write(&path, &image).unwrap();
        let first = TurboQuantIndex::load(&path).unwrap().to_bytes();
        for round in 0..3 {
            let got = TurboQuantIndex::load(&path).unwrap().to_bytes();
            assert_eq!(got, first, "{label}: load {round} is not deterministic");
            assert_eq!(
                std::fs::read(&path).unwrap(),
                image,
                "{label}: load {round} rewrote the file"
            );
        }
    }
}

/// Readers racing a writer must never see a partial commit. A reader
/// pulls the header region before the units, and the writer lays down
/// units before the header, so every interleaving should resolve to a
/// commit that was durable at some point.
#[test]
fn a_reader_racing_a_writer_only_ever_sees_committed_states() {
    let dir = temp_dir("race");
    let path = dir.join("index.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 9)).unwrap();
    idx.add(&rows(500, 10));
    idx.sync(&path).unwrap();

    // Every commit is identified by its row count, which only grows.
    let committed = Arc::new(Mutex::new(vec![idx.len()]));
    let stop = Arc::new(AtomicBool::new(false));
    let reader = {
        let (path, committed, stop) = (path.clone(), committed.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
            while !stop.load(Ordering::Relaxed) {
                let Ok(l) = TurboQuantIndex::load(&path) else {
                    // A reader may lose the race with a compaction's
                    // rename; that is a missing file, not a torn one.
                    continue;
                };
                let n = l.len();
                let known = committed.lock().unwrap();
                assert!(
                    known.contains(&n),
                    "reader saw {n} rows, which was never committed (known: {known:?})"
                );
                drop(known);
                // The state must also be internally coherent: a torn
                // commit that slipped through would give a scale or an
                // id that does not belong to the rows it serves.
                let r = l.search(&rows(1, 77), 5);
                assert!(
                    r.scores.iter().all(|s: &f32| s.is_finite()),
                    "a raced load served non-finite scores"
                );
                seen.insert(n);
            }
            seen
        })
    };

    for round in 0..60u64 {
        idx.add(&rows(17, 100 + round));
        // Record the new count BEFORE it can be observed on disk.
        committed.lock().unwrap().push(idx.len());
        idx.sync(&path).unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    let seen = reader.join().expect("the reader must not have tripped");
    assert!(
        seen.len() > 1,
        "the reader never raced a commit (saw only {seen:?}); the test proved nothing"
    );
}

/// A rollback whose rejected header is carrying a full load of pending
/// redo ops — the repair has to measure and destroy the whole of it, not
/// just the short prefix an op-free header occupies.
#[test]
fn a_rollback_past_an_op_heavy_header_is_repaired_completely() {
    let dir = temp_dir("opheavy");
    let path = dir.join("index.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 30)).unwrap();
    idx.add(&rows(4096, 31));
    idx.sync(&path).unwrap();
    // Just under the header's op cap, scattered low in the file so none
    // is popped by the shrink: the rejected header will be near its
    // full width.
    for i in 0..1000usize {
        idx.swap_remove(i * 2 % 1500);
    }
    idx.sync(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();

    // The next sync materializes all of them, so its delta names many
    // units; land its header region only and the commit is stranded.
    let mut live = TurboQuantIndex::load(&path).unwrap();
    live.add(&rows(5, 32));
    live.sync(&path).unwrap();
    let after = std::fs::read(&path).unwrap();
    let state_b = live.to_bytes();

    let crashed = land_below(&before, &after, unit0(DIM, 4, 0));
    std::fs::write(&path, &crashed).unwrap();
    let rolled = TurboQuantIndex::load(&path).expect("must load");
    assert_eq!(rolled.to_bytes(), state_a, "the stranded commit must be rejected");

    // Sync forward from the rollback and confirm the result is one of
    // the two real states, never the rejected one.
    let mut fwd = rolled;
    fwd.add(&rows(7, 33));
    fwd.sync(&path).unwrap();
    let got = TurboQuantIndex::load(&path).unwrap().to_bytes();
    assert_eq!(got, fwd.to_bytes());
    assert_ne!(got, state_b, "the rejected commit came back");
}

/// The file is replaced underneath a bound index — what a lost rename
/// after a compaction, or an external restore, looks like. The sync must
/// refuse rather than write its generation into a file it does not own,
/// must not damage what is there, and must recover once the caller
/// re-loads.
#[test]
fn a_file_swapped_underneath_a_bound_index_is_refused_then_recoverable() {
    let dir = temp_dir("swapped");
    let path = dir.join("index.tv");
    let mut a = TurboQuantIndex::new(DIM, 4).unwrap();
    a.calibrate(&rows(1024, 40)).unwrap();
    a.add(&rows(120, 41));
    a.sync(&path).unwrap();

    // An unrelated index of the same shape, written by a different
    // writer: same geometry, different file identity.
    let mut b = TurboQuantIndex::new(DIM, 4).unwrap();
    b.calibrate(&rows(1024, 40)).unwrap();
    b.add(&rows(77, 42));
    let other = dir.join("other.tv");
    b.sync(&other).unwrap();
    std::fs::copy(&other, &path).unwrap();
    let planted = std::fs::read(&path).unwrap();

    a.add(&rows(4, 43));
    let err = a.sync(&path).expect_err("syncing into a foreign file must refuse");
    assert!(err.to_string().contains("another writer"), "{err}");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        planted,
        "the refused sync modified the foreign file"
    );

    // The documented way out: adopt the file.
    let mut adopted = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(adopted.to_bytes(), b.to_bytes());
    adopted.add(&rows(4, 44));
    adopted.sync(&path).expect("a loaded index must sync forward");
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), adopted.to_bytes());
}

/// Alternating the two writers at one path: `write` lays down a v6 file
/// where a sync container was, and `sync` must rebuild rather than
/// treat the foreign bytes as a container to append into.
#[test]
fn alternating_write_and_sync_at_one_path_never_mixes_the_formats() {
    let dir = temp_dir("alternate");
    let path = dir.join("index.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 50)).unwrap();
    idx.add(&rows(150, 51));

    for round in 0..5u64 {
        idx.sync(&path).unwrap();
        assert_eq!(
            TurboQuantIndex::load(&path).unwrap().to_bytes(),
            idx.to_bytes(),
            "round {round}: sync then load"
        );
        idx.add(&rows(9, 60 + round));
        idx.sync(&path).unwrap();
        assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes());

        idx.write(&path).unwrap();
        assert_eq!(
            TurboQuantIndex::load(&path).unwrap().to_bytes(),
            idx.to_bytes(),
            "round {round}: write then load"
        );
        idx.swap_remove(round as usize);
        idx.add(&rows(3, 70 + round));
    }
}

/// An index rebuilt from its own serialized bytes is unbound, so its
/// first sync must write a whole container rather than assume it owns
/// whatever is at the path.
#[test]
fn an_index_restored_from_bytes_syncs_over_a_stranger_safely() {
    let dir = temp_dir("frombytes");
    let path = dir.join("index.tv");

    let mut donor = TurboQuantIndex::new(DIM, 4).unwrap();
    donor.calibrate(&rows(1024, 80)).unwrap();
    donor.add(&rows(90, 81));
    donor.sync(&path).unwrap();

    let mut restored = TurboQuantIndex::from_bytes(&donor.to_bytes()).unwrap();
    assert_eq!(restored.to_bytes(), donor.to_bytes());
    restored.add(&rows(5, 82));
    restored.sync(&path).expect("an unbound index must write the container whole");
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), restored.to_bytes());

    // And it is bound afterwards: the next sync is incremental and still
    // round-trips.
    restored.swap_remove(2);
    restored.add(&rows(40, 83));
    restored.sync(&path).unwrap();
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), restored.to_bytes());
}
