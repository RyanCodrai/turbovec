//! Adversarial durability probes: a power-loss model coarser than the
//! in-crate torn-write harness.
//!
//! The in-crate harness tears a planned sync at every byte of every
//! write op, in a few orderings. This one is black-box and models what a
//! device actually does on power loss: each 512-byte sector of the file
//! independently either holds its pre-sync content or its post-sync
//! content. Arbitrary subsets, not prefixes.
//!
//! Every recovered file must load, and must load to exactly the previous
//! commit or the new one — and must still be writable afterwards.

use std::path::PathBuf;

use turbovec::{IdMapIndex, TurboQuantIndex};

const DIM: usize = 64;
const SECTOR: usize = 512;

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
    p.push(format!("turbovec-adv-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p.push("index.tv");
    p
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn bit(&mut self) -> bool {
        self.next() & 1 == 0
    }
}

/// Sectors whose content differs between the pre- and post-sync images.
fn dirty_sectors(a: &[u8], b: &[u8]) -> Vec<usize> {
    let n = a.len().max(b.len()).div_ceil(SECTOR);
    (0..n)
        .filter(|&s| {
            let (lo, hi) = (s * SECTOR, ((s + 1) * SECTOR).min(a.len().max(b.len())));
            let sa = a.get(lo..hi.min(a.len())).unwrap_or(&[]);
            let sb = b.get(lo..hi.min(b.len())).unwrap_or(&[]);
            sa != sb
        })
        .collect()
}

/// The file after a power cut in which exactly `landed` of the changed
/// sectors reached the platter.
fn hybrid(a: &[u8], b: &[u8], landed: &[usize]) -> Vec<u8> {
    let mut out = a.to_vec();
    for &s in landed {
        let lo = s * SECTOR;
        let hi = ((s + 1) * SECTOR).min(b.len());
        if lo >= b.len() {
            continue;
        }
        if out.len() < hi {
            out.resize(hi, 0);
        }
        out[lo..hi].copy_from_slice(&b[lo..hi]);
    }
    out
}

/// Run the sector-subset crash matrix over one sync.
///
/// `before`/`after` are the file images either side of the sync,
/// `state_a`/`state_b` the oracle states they must load to, and `load`
/// canonicalizes a candidate file.
fn crash_matrix(
    scratch: &std::path::Path,
    before: &[u8],
    after: &[u8],
    state_a: &[u8],
    state_b: &[u8],
    load: &dyn Fn(&std::path::Path) -> std::io::Result<Vec<u8>>,
    label: &str,
) {
    let dirty = dirty_sectors(before, after);
    assert!(!dirty.is_empty(), "{label}: the sync changed nothing");
    assert_ne!(state_a, state_b, "{label}: the sync changed no state");

    let mut cases: Vec<Vec<usize>> = Vec::new();
    // Every prefix and every suffix of the changed sectors.
    for k in 0..=dirty.len() {
        cases.push(dirty[..k].to_vec());
        cases.push(dirty[dirty.len() - k..].to_vec());
    }
    // Each sector alone, and each sector missing.
    for i in 0..dirty.len() {
        cases.push(vec![dirty[i]]);
        cases.push(
            dirty
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &s)| s)
                .collect(),
        );
    }
    // Random subsets.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ dirty.len() as u64);
    for _ in 0..400 {
        cases.push(dirty.iter().copied().filter(|_| rng.bit()).collect());
    }

    for landed in &cases {
        let file = hybrid(before, after, landed);
        std::fs::write(scratch, &file).unwrap();
        let got = load(scratch).unwrap_or_else(|e| {
            panic!("{label}: {} of {} sectors landed: unloadable ({e})", landed.len(), dirty.len())
        });
        let complete = landed.len() == dirty.len();
        if complete {
            assert_eq!(got, state_b, "{label}: a fully landed sync is not the new commit");
        } else {
            assert!(
                got == state_a || got == state_b,
                "{label}: {} of {} sectors landed: a third state",
                landed.len(),
                dirty.len()
            );
        }
        // Whatever it recovered to, the file must still be usable: load
        // it for real, sync it forward, and read it back.
        if landed.len() % 37 == 0 {
            let forward = scratch.with_extension("forward");
            std::fs::copy(scratch, &forward).unwrap();
            let mut idx = TurboQuantIndex::load(&forward).ok();
            if let Some(i) = idx.as_mut() {
                i.add(&rows(5, 4242));
                i.sync(&forward).unwrap_or_else(|e| {
                    panic!("{label}: recovered file refuses a forward sync ({e})")
                });
                let back = TurboQuantIndex::load(&forward).unwrap();
                assert_eq!(back.to_bytes(), i.to_bytes(), "{label}: forward sync lost state");
            }
        }
    }
}

fn tq_state(p: &std::path::Path) -> std::io::Result<Vec<u8>> {
    TurboQuantIndex::load(p).map(|i| i.to_bytes())
}

fn idm_state(p: &std::path::Path) -> std::io::Result<Vec<u8>> {
    IdMapIndex::load(p).map(|i| i.to_bytes())
}

/// A removal-heavy sync (pending redo ops in the header) torn at sector
/// granularity.
#[test]
fn a_removal_sync_survives_arbitrary_sector_loss() {
    let path = temp("removal");
    let scratch = path.with_file_name("scratch.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 1)).unwrap();
    idx.add(&rows(200, 2));
    idx.sync(&path).unwrap();

    let before = std::fs::read(&path).unwrap();
    let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();

    for i in [3usize, 40, 71, 100, 137] {
        idx.swap_remove(i);
    }
    idx.add(&rows(9, 3));
    idx.sync(&path).unwrap();
    let after = std::fs::read(&path).unwrap();
    let state_b = TurboQuantIndex::load(&path).unwrap().to_bytes();

    crash_matrix(&scratch, &before, &after, &state_a, &state_b, &tq_state, "removal");
}

/// The sync that MATERIALIZES the previous sync's pending ops — it
/// overwrites committed block units in place, which is the one moment
/// the file holds bytes no committed header describes.
#[test]
fn a_materializing_sync_survives_arbitrary_sector_loss() {
    let path = temp("materialize");
    let scratch = path.with_file_name("scratch.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 4)).unwrap();
    idx.add(&rows(256, 5));
    idx.sync(&path).unwrap();

    // Removals commit as pending ops; nothing in the unit region moves.
    for i in [1usize, 33, 65, 97, 129] {
        idx.swap_remove(i);
    }
    idx.sync(&path).unwrap();

    let before = std::fs::read(&path).unwrap();
    let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();

    // The next sync materializes them into their units.
    idx.add(&rows(1, 6));
    idx.sync(&path).unwrap();
    let after = std::fs::read(&path).unwrap();
    let state_b = TurboQuantIndex::load(&path).unwrap().to_bytes();

    crash_matrix(&scratch, &before, &after, &state_a, &state_b, &tq_state, "materialize");
}

/// Shrink below a committed block floor and regrow past it in one sync
/// interval: the regrown rows land inside units the file already holds,
/// so they can only travel as redo ops.
#[test]
fn a_shrink_and_regrow_within_one_sync_survives_sector_loss() {
    let path = temp("regrow");
    let scratch = path.with_file_name("scratch.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 7)).unwrap();
    idx.add(&rows(256, 8));
    idx.sync(&path).unwrap();

    let before = std::fs::read(&path).unwrap();
    let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();

    // 256 -> 192 (two whole blocks popped), then back to 256.
    for _ in 0..64 {
        idx.swap_remove(idx.len() - 1);
    }
    idx.add(&rows(64, 9));
    idx.sync(&path).unwrap();
    let after = std::fs::read(&path).unwrap();
    let state_b = TurboQuantIndex::load(&path).unwrap().to_bytes();

    crash_matrix(&scratch, &before, &after, &state_a, &state_b, &tq_state, "regrow");
}

/// Same, but the shrink happens in one sync and the regrow in the next —
/// the second sync rewrites units the FIRST one left stale.
#[test]
fn a_regrow_after_a_committed_shrink_survives_sector_loss() {
    let path = temp("regrow2");
    let scratch = path.with_file_name("scratch.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 10)).unwrap();
    idx.add(&rows(256, 11));
    idx.sync(&path).unwrap();
    for _ in 0..64 {
        idx.swap_remove(idx.len() - 1);
    }
    idx.sync(&path).unwrap();

    let before = std::fs::read(&path).unwrap();
    let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();

    idx.add(&rows(70, 12));
    idx.sync(&path).unwrap();
    let after = std::fs::read(&path).unwrap();
    let state_b = TurboQuantIndex::load(&path).unwrap().to_bytes();

    crash_matrix(&scratch, &before, &after, &state_a, &state_b, &tq_state, "regrow2");
}

/// The id-mapped container has ids in every structure the crash protocol
/// touches — units, header tail, redo ops — and no torn-sync coverage of
/// its own.
#[test]
fn an_idmap_sync_survives_arbitrary_sector_loss() {
    let path = temp("idmap");
    let scratch = path.with_file_name("scratch.tvim");
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 13)).unwrap();
    let ids: Vec<u64> = (0..200u64).map(|i| i * 31 + 5).collect();
    idx.add_with_ids(&rows(200, 14), &ids).unwrap();
    idx.sync(&path).unwrap();

    let before = std::fs::read(&path).unwrap();
    let state_a = IdMapIndex::load(&path).unwrap().to_bytes();

    for i in [0u64, 40, 71, 100, 137] {
        assert!(idx.remove(i * 31 + 5));
    }
    idx.add_with_ids(&rows(9, 15), &(9000..9009u64).collect::<Vec<_>>()).unwrap();
    idx.sync(&path).unwrap();
    let after = std::fs::read(&path).unwrap();
    let state_b = IdMapIndex::load(&path).unwrap().to_bytes();

    crash_matrix(&scratch, &before, &after, &state_a, &state_b, &idm_state, "idmap");
}

/// A recovered file must not just load — it must keep serving and keep
/// syncing. Roll back to the previous commit by hand, then drive the
/// index forward for several more syncs.
#[test]
fn a_rolled_back_idmap_keeps_syncing_forward() {
    let path = temp("idmap-forward");
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 16)).unwrap();
    idx.add_with_ids(&rows(100, 17), &(0..100u64).collect::<Vec<_>>()).unwrap();
    idx.sync(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    assert!(idx.remove(5));
    assert!(idx.remove(66));
    idx.sync(&path).unwrap();

    // Power cut: only the header slot's sectors landed, none of the data.
    let after = std::fs::read(&path).unwrap();
    let dirty = dirty_sectors(&before, &after);
    for keep in [0usize, 1, dirty.len() / 2] {
        let file = hybrid(&before, &after, &dirty[..keep.min(dirty.len())]);
        std::fs::write(&path, &file).unwrap();
        let mut back = IdMapIndex::load(&path).expect("rolled-back file must load");
        for round in 0..4u64 {
            back.add_with_ids(&rows(3, 18 + round), &[500 + round * 3, 501 + round * 3, 502 + round * 3])
                .unwrap();
            if back.contains(round) {
                assert!(back.remove(round));
            }
            back.sync(&path).expect("forward sync after rollback");
            let reread = IdMapIndex::load(&path).unwrap();
            assert_eq!(reread.to_bytes(), back.to_bytes(), "round {round} lost state");
        }
    }
}

/// Emptying the index and syncing, then regrowing from zero.
#[test]
fn syncing_down_to_empty_and_back_round_trips() {
    let path = temp("empty");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 19)).unwrap();
    idx.add(&rows(100, 20));
    idx.sync(&path).unwrap();
    while !idx.is_empty() {
        idx.swap_remove(idx.len() - 1);
    }
    idx.sync(&path).unwrap();
    assert_eq!(TurboQuantIndex::load(&path).unwrap().len(), 0);
    idx.add(&rows(70, 21));
    idx.sync(&path).unwrap();
    let back = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(back.to_bytes(), idx.to_bytes(), "regrow from empty lost state");
}

/// A mass removal exceeds the header's op capacity and falls back to a
/// full rewrite. That path is temp-file + atomic rename, so the only
/// two observable states are the old file and the new one — but the
/// index must also stay correct across it.
#[test]
fn a_mass_removal_falls_back_to_a_full_rewrite_and_stays_correct() {
    let path = temp("mass");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 22)).unwrap();
    idx.add(&rows(4096, 23));
    idx.sync(&path).unwrap();
    // Scatter more removals than one header can carry, low in the file
    // so none of them are popped by the shrink.
    for i in 0..1200usize {
        idx.swap_remove(i * 2 % 1500);
    }
    idx.sync(&path).unwrap();
    let back = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(back.to_bytes(), idx.to_bytes(), "mass-removal rewrite lost state");
    // And the file is still incrementally syncable afterwards.
    idx.add(&rows(10, 24));
    idx.sync(&path).unwrap();
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes());
}
