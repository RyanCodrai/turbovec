//! Chained-crash fuzz: many rounds of {mutate, sync, lose a random set
//! of sectors, reload}, against an independent in-memory model.
//!
//! The in-crate harness tears ONE sync from a clean base. This drives
//! the container through long histories where every commit may be a
//! recovery from the last crash, across all three bit widths and both
//! container kinds.

use std::path::PathBuf;

use turbovec::{IdMapIndex, TurboQuantIndex};

const SECTOR: usize = 512;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn row(dim: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut v: Vec<f32> = (0..dim)
        .map(|_| ((r.next() >> 40) as f32 / (1u64 << 23) as f32) - 0.5)
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

fn rows(dim: usize, n: usize, seed: u64) -> Vec<f32> {
    (0..n).flat_map(|i| row(dim, seed.wrapping_add(i as u64 * 7919))).collect()
}

fn temp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-advfuzz-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p.push("index.tv");
    p
}

/// Drop a random subset of the sectors that changed between `before` and
/// `after`, and write the result back over `path`.
fn crash(path: &std::path::Path, before: &[u8], after: &[u8], rng: &mut Rng) -> bool {
    let n = before.len().max(after.len()).div_ceil(SECTOR);
    let mut out = before.to_vec();
    let mut lost = false;
    for s in 0..n {
        let lo = s * SECTOR;
        let hi = ((s + 1) * SECTOR).min(after.len());
        if lo >= after.len() {
            continue;
        }
        // Two thirds of changed sectors land.
        if rng.next().is_multiple_of(3) {
            lost = true;
            continue;
        }
        if out.len() < hi {
            out.resize(hi, 0);
        }
        out[lo..hi].copy_from_slice(&after[lo..hi]);
    }
    if !lost {
        return false;
    }
    std::fs::write(path, &out).unwrap();
    true
}

/// One fuzz history over a plain index. The model is the previous
/// successfully-observed `to_bytes` image: after a crash the file must
/// load to either the pre-sync or post-sync image, and whichever it is,
/// the index reloaded from it must keep syncing forward forever.
fn fuzz_plain(dim: usize, bit_width: usize, seed: u64, rounds: usize) {
    let path = temp(&format!("plain-{dim}-{bit_width}-{seed}"));
    let mut rng = Rng::new(seed);
    let mut idx = TurboQuantIndex::new(dim, bit_width).unwrap();
    idx.calibrate(&rows(dim, 1024, seed ^ 0xC0FFEE)).unwrap();
    idx.add(&rows(dim, 40 + rng.below(200), seed));
    idx.sync(&path).unwrap();
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), idx.to_bytes());

    for round in 0..rounds {
        let before = std::fs::read(&path).unwrap();
        let state_a = TurboQuantIndex::load(&path).unwrap().to_bytes();
        assert_eq!(state_a, idx.to_bytes(), "round {round}: drifted before mutating");

        // A random mix of removals and adds.
        let n_rm = rng.below(6);
        for _ in 0..n_rm {
            if idx.len() > 1 {
                let i = rng.below(idx.len());
                idx.swap_remove(i);
            }
        }
        let n_add = rng.below(40);
        if n_add > 0 {
            idx.add(&rows(dim, n_add, seed.wrapping_add(round as u64 * 131)));
        }
        if n_rm == 0 && n_add == 0 {
            continue;
        }
        idx.sync(&path).unwrap();
        let after = std::fs::read(&path).unwrap();
        let state_b = idx.to_bytes();
        assert_eq!(
            TurboQuantIndex::load(&path).unwrap().to_bytes(),
            state_b,
            "round {round}: a clean sync does not reload"
        );

        // Crash on a third of rounds.
        if !rng.next().is_multiple_of(3) || !crash(&path, &before, &after, &mut rng) {
            continue;
        }
        let recovered = TurboQuantIndex::load(&path)
            .unwrap_or_else(|e| panic!("round {round}: crashed file is unloadable ({e})"));
        let got = recovered.to_bytes();
        assert!(
            got == state_a || got == state_b,
            "round {round}: crash recovered to a third state"
        );
        // Adopt whatever survived and keep going from there.
        idx = recovered;
    }
}

/// Same history for the id-mapped container, with the id table as an
/// extra oracle.
fn fuzz_idmap(dim: usize, bit_width: usize, seed: u64, rounds: usize) {
    let path = temp(&format!("idm-{dim}-{bit_width}-{seed}"));
    let mut rng = Rng::new(seed);
    let mut idx = IdMapIndex::new(dim, bit_width).unwrap();
    idx.calibrate(&rows(dim, 1024, seed ^ 0xBEEF)).unwrap();
    let mut next_id: u64 = 1;
    let n0 = 40 + rng.below(150);
    let ids: Vec<u64> = (0..n0 as u64).map(|i| i + next_id).collect();
    next_id += n0 as u64;
    idx.add_with_ids(&rows(dim, n0, seed), &ids).unwrap();
    idx.sync(&path).unwrap();
    let mut live: Vec<u64> = ids;

    for round in 0..rounds {
        let before = std::fs::read(&path).unwrap();
        let state_a = IdMapIndex::load(&path).unwrap().to_bytes();
        assert_eq!(state_a, idx.to_bytes(), "round {round}: drifted before mutating");
        let live_a = live.clone();

        let n_rm = rng.below(6);
        for _ in 0..n_rm {
            if live.len() > 1 {
                let k = rng.below(live.len());
                let id = live.swap_remove(k);
                assert!(idx.remove(id), "round {round}: remove({id}) missed");
            }
        }
        let n_add = rng.below(40);
        let new_ids: Vec<u64> = (0..n_add as u64).map(|i| next_id + i).collect();
        next_id += n_add as u64;
        if n_add > 0 {
            idx.add_with_ids(&rows(dim, n_add, seed.wrapping_add(round as u64 * 977)), &new_ids)
                .unwrap();
            live.extend_from_slice(&new_ids);
        }
        if n_rm == 0 && n_add == 0 {
            continue;
        }
        idx.sync(&path).unwrap();
        let after = std::fs::read(&path).unwrap();
        let state_b = idx.to_bytes();
        assert_eq!(
            IdMapIndex::load(&path).unwrap().to_bytes(),
            state_b,
            "round {round}: a clean idmap sync does not reload"
        );
        let live_b = live.clone();

        if !rng.next().is_multiple_of(3) || !crash(&path, &before, &after, &mut rng) {
            continue;
        }
        let recovered = IdMapIndex::load(&path)
            .unwrap_or_else(|e| panic!("round {round}: crashed idmap is unloadable ({e})"));
        let got = recovered.to_bytes();
        let expect_a = got == state_a;
        assert!(
            expect_a || got == state_b,
            "dim={dim} bw={bit_width} seed={seed} round={round}: crash recovered a third \
             state — {} rows, where the previous commit had {} and the new one {}",
            recovered.len(),
            live_a.len(),
            live_b.len()
        );
        live = if expect_a { live_a } else { live_b };
        // The id table must agree with whichever commit survived.
        assert_eq!(recovered.len(), live.len(), "round {round}: id count disagrees");
        for &id in &live {
            assert!(recovered.contains(id), "round {round}: lost id {id}");
        }
        idx = recovered;
    }
}

#[test]
fn plain_container_survives_chained_crashes_at_every_bit_width() {
    for bit_width in [2usize, 3, 4] {
        for (i, dim) in [64usize, 128, 256].into_iter().enumerate() {
            fuzz_plain(dim, bit_width, 0x1234 + (bit_width * 17 + i) as u64, 60);
        }
    }
}

#[test]
fn idmap_container_survives_chained_crashes_at_every_bit_width() {
    for bit_width in [2usize, 3, 4] {
        for (i, dim) in [64usize, 128, 256].into_iter().enumerate() {
            fuzz_idmap(dim, bit_width, 0x5678 + (bit_width * 19 + i) as u64, 60);
        }
    }
}

/// Two live handles bound to the same file. The second one must refuse
/// rather than write its own generation over the first's commits.
#[test]
fn a_second_writer_on_the_same_path_refuses_instead_of_clobbering() {
    let path = temp("two-writers");
    let dim = 64;
    let mut a = TurboQuantIndex::new(dim, 4).unwrap();
    a.calibrate(&rows(dim, 1024, 3)).unwrap();
    a.add(&rows(dim, 100, 4));
    a.sync(&path).unwrap();

    let mut b = TurboQuantIndex::load(&path).unwrap();
    a.add(&rows(dim, 5, 5));
    a.sync(&path).unwrap();
    let a_state = a.to_bytes();

    b.add(&rows(dim, 5, 6));
    let err = b.sync(&path).unwrap_err();
    assert!(
        err.to_string().contains("another writer"),
        "second writer did not refuse: {err}"
    );
    assert_eq!(
        TurboQuantIndex::load(&path).unwrap().to_bytes(),
        a_state,
        "the refused sync still damaged the file"
    );
    // And the first writer keeps working afterwards.
    a.add(&rows(dim, 3, 7));
    a.sync(&path).unwrap();
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), a.to_bytes());
}

/// An index restored from an older backup of its own file: the nonce
/// matches but the generation went backwards. It must not sync forward
/// over a state it cannot account for.
#[test]
fn a_restored_older_backup_is_not_synced_over_blindly() {
    let path = temp("backup");
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.calibrate(&rows(dim, 1024, 8)).unwrap();
    idx.add(&rows(dim, 100, 9));
    idx.sync(&path).unwrap();
    let backup = std::fs::read(&path).unwrap();

    for r in 0..4u64 {
        idx.add(&rows(dim, 40, 10 + r));
        idx.sync(&path).unwrap();
    }
    // Roll the file back under the live index's feet.
    std::fs::write(&path, &backup).unwrap();
    idx.add(&rows(dim, 3, 20));
    match idx.sync(&path) {
        Err(e) => assert!(e.to_string().contains("another writer"), "{e}"),
        Ok(()) => {
            // A full rewrite is also acceptable, as long as the file
            // that results is exactly the live index.
            assert_eq!(
                TurboQuantIndex::load(&path).unwrap().to_bytes(),
                idx.to_bytes(),
                "sync over a rolled-back file left a mixed state"
            );
        }
    }
}

/// Truncating the file to every plausible length must never produce a
/// file that loads to something other than a real commit.
#[test]
fn truncation_at_any_length_never_yields_a_bogus_index() {
    let path = temp("truncate");
    let scratch = path.with_file_name("scratch.tv");
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.calibrate(&rows(dim, 1024, 11)).unwrap();
    idx.add(&rows(dim, 200, 12));
    idx.sync(&path).unwrap();
    let state_a = idx.to_bytes();
    idx.swap_remove(3);
    idx.add(&rows(dim, 5, 13));
    idx.sync(&path).unwrap();
    let full = std::fs::read(&path).unwrap();
    let state_b = idx.to_bytes();

    let mut rng = Rng::new(99);
    for _ in 0..600 {
        let cut = rng.below(full.len() + 1);
        std::fs::write(&scratch, &full[..cut]).unwrap();
        if let Ok(loaded) = TurboQuantIndex::load(&scratch) {
            let got = loaded.to_bytes();
            assert!(
                got == state_a || got == state_b,
                "truncation to {cut} bytes loaded a state that was never committed"
            );
        }
    }
}
