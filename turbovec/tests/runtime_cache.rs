//! Runtime-cache sidecar behavior through the public API (issue #68).
//!
//! The in-crate unit tests (`src/runtime_cache.rs`) verify the format
//! and that loads actually seed the private caches; these tests pin the
//! externally observable contract:
//!   - `write` leaves exactly the index file plus one `.cache` sidecar.
//!   - Loads with a present, absent, or corrupted sidecar all succeed
//!     and return identical search results.
//!   - `load` never creates files (read-only deployments keep working).
//!   - The byte APIs (`to_bytes`/`from_bytes`) stay sidecar-free.
//!   - The full mutate → write → load cycle stays correct for
//!     `IdMapIndex`, including removals and post-load adds.

use std::path::{Path, PathBuf};

use turbovec::{IdMapIndex, TurboQuantIndex};

const DIM: usize = 32;
const N: usize = 70; // spans three 32-vector SIMD blocks

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-{}-{}", nonce, name));
    std::fs::create_dir(&p).unwrap();
    p
}

fn lcg_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n * dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 32) as u32 as f64 / 2_147_483_648.0 - 1.0) as f32
        })
        .collect()
}

fn dir_entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// The single sidecar file next to `index_path`, if any.
fn sidecar_of(dir: &Path) -> Option<PathBuf> {
    let caches: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".cache"))
        .collect();
    assert!(caches.len() <= 1, "at most one sidecar per backend expected");
    caches.into_iter().next()
}

#[test]
fn tv_write_creates_one_sidecar_and_all_load_paths_agree() {
    let dir = temp_dir("tv-sidecar");
    let path = dir.join("index.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add_2d(&lcg_vectors(N, DIM, 1), DIM).unwrap();
    idx.write(&path).unwrap();

    let entries = dir_entries(&dir);
    assert_eq!(entries.len(), 2, "index + one sidecar, got {entries:?}");
    let sidecar = sidecar_of(&dir).expect("write must create a sidecar");

    let queries = lcg_vectors(4, DIM, 2);
    // Baseline: byte API, which never sees sidecars.
    let baseline = TurboQuantIndex::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let base = baseline.search(&queries, 7);

    // Sidecar present.
    let loaded = TurboQuantIndex::load(&path).unwrap();
    let res = loaded.search(&queries, 7);
    assert_eq!(res.scores, base.scores);
    assert_eq!(res.indices, base.indices);

    // Sidecar corrupted (flip one payload byte).
    let mut bytes = std::fs::read(&sidecar).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x01;
    std::fs::write(&sidecar, &bytes).unwrap();
    let loaded = TurboQuantIndex::load(&path).unwrap();
    let res = loaded.search(&queries, 7);
    assert_eq!(res.scores, base.scores);
    assert_eq!(res.indices, base.indices);

    // Sidecar deleted — delete-safe by contract.
    std::fs::remove_file(&sidecar).unwrap();
    let loaded = TurboQuantIndex::load(&path).unwrap();
    let res = loaded.search(&queries, 7);
    assert_eq!(res.scores, base.scores);
    assert_eq!(res.indices, base.indices);

    // That load must not have re-created anything.
    assert_eq!(dir_entries(&dir), vec!["index.tv"], "load must never write");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tvim_mutation_cycle_stays_correct_with_sidecars() {
    let dir = temp_dir("tvim-cycle");
    let path = dir.join("index.tvim");
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    let ids: Vec<u64> = (1000..1000 + N as u64).collect();
    idx.add_with_ids(&lcg_vectors(N, DIM, 3), &ids).unwrap();
    idx.write(&path).unwrap();
    assert!(sidecar_of(&dir).is_some());

    let queries = lcg_vectors(4, DIM, 4);
    let check_parity = |dir: &Path, path: &Path| {
        let loaded = IdMapIndex::load(path).unwrap();
        let baseline = IdMapIndex::from_bytes(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(loaded.search(&queries, 5), baseline.search(&queries, 5));
        assert!(sidecar_of(dir).is_some());
        loaded
    };

    // Load → remove two ids → write → reload.
    let mut loaded = check_parity(&dir, &path);
    assert!(loaded.remove(1005));
    assert!(loaded.remove(1040));
    loaded.write(&path).unwrap();

    // Load → add more vectors after a seeded load → write → reload.
    let mut loaded = check_parity(&dir, &path);
    let more_ids: Vec<u64> = (5000..5010).collect();
    loaded
        .add_with_ids(&lcg_vectors(10, DIM, 5), &more_ids)
        .unwrap();
    loaded.write(&path).unwrap();

    let final_loaded = check_parity(&dir, &path);
    assert_eq!(final_loaded.len(), N - 2 + 10);
    assert!(final_loaded.contains(5003));
    assert!(!final_loaded.contains(1005));

    // Allowlist search still works through a seeded load.
    let allow = [5000u64, 5001, 1000];
    let (_, got_ids) = final_loaded.search_with_allowlist(&queries, 3, Some(&allow));
    assert!(got_ids.iter().all(|id| allow.contains(id)));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn empty_and_lazy_writes_leave_no_sidecar() {
    let dir = temp_dir("empty-lazy");

    let empty_path = dir.join("empty.tv");
    TurboQuantIndex::new(DIM, 4).unwrap().write(&empty_path).unwrap();

    let lazy_path = dir.join("lazy.tvim");
    IdMapIndex::new_lazy(3).unwrap().write(&lazy_path).unwrap();

    assert_eq!(
        dir_entries(&dir),
        vec!["empty.tv", "lazy.tvim"],
        "empty/lazy writes must not create sidecars",
    );

    // And they load back fine.
    assert_eq!(TurboQuantIndex::load(&empty_path).unwrap().len(), 0);
    assert_eq!(IdMapIndex::load(&lazy_path).unwrap().dim_opt(), None);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn byte_apis_are_unaffected_by_sidecars() {
    let dir = temp_dir("bytes-unaffected");
    let path = dir.join("index.tv");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add_2d(&lcg_vectors(N, DIM, 6), DIM).unwrap();
    idx.write(&path).unwrap();

    // to_bytes emits exactly the authoritative file — no sidecar data.
    assert_eq!(idx.to_bytes(), std::fs::read(&path).unwrap());

    // from_bytes of the same payload round-trips regardless of the
    // sidecar sitting next to the source file.
    let rebuilt = TurboQuantIndex::from_bytes(&idx.to_bytes()).unwrap();
    assert_eq!(rebuilt.len(), N);
    std::fs::remove_dir_all(&dir).ok();
}
