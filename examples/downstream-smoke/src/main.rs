//! Smoke test for the downstream `cargo add turbovec` experience.
//!
//! Exercises the public API end-to-end from outside the crate: construct,
//! add, prepare, search, mutate, serialize, and the id-addressed wrapper.
//! The crate has no native/BLAS dependency any more (removed with the v5
//! block-Hadamard rotation, #206 — on `main`, not yet in a tagged release),
//! so this binary links and runs with a plain `cargo build` and no
//! `extern crate blas_src;` ceremony.
//!
//! What this file is *for*, beyond linking: it is the only place the API is
//! used the way a dependent uses it — through `pub use`, with no access to
//! `pub(crate)` internals and no `#[cfg(test)]` helpers. Anything that only
//! breaks from that angle (a type that stops deriving `Debug`, an error
//! enum that stops being matchable, a trait bound that stops holding) is
//! invisible to `cargo test -p turbovec` and visible here (#351).

use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use turbovec::io::Durability;
use turbovec::{AddError, IdMapIndex, SearchError, TurboQuantIndex};

const DIM: usize = 64;
const N_DB: usize = 256;
const N_QUERIES: usize = 4;
const K: usize = 5;

fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32 / u64::MAX as f32) * 2.0 - 1.0
    };
    let mut out = vec![0.0f32; n * dim];
    for v in out.chunks_mut(dim) {
        for x in v.iter_mut() {
            *x = next();
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    out
}

/// Trait bounds a dependent's own types inherit by holding these.
///
/// A compile-time check with no runtime component: if `SearchResults` were
/// to lose `Debug` again (#351 finding 1) or an index were to stop being
/// `Send + Sync`, this function stops compiling.
fn trait_bounds() {
    fn shared<T: Send + Sync + Debug>() {}
    fn value<T: Debug + Clone + PartialEq>() {}
    fn boxed_error<T: std::error::Error + Send + Sync + 'static>() {}

    shared::<TurboQuantIndex>();
    shared::<IdMapIndex>();
    shared::<turbovec::SearchResults>();
    shared::<turbovec::IdSearchResults>();

    value::<turbovec::SearchResults>();
    value::<turbovec::IdSearchResults>();
    value::<AddError>();
    value::<SearchError>();

    boxed_error::<AddError>();
    boxed_error::<SearchError>();
    boxed_error::<turbovec::ConstructError>();
    boxed_error::<turbovec::FromPartsError>();

    // A downstream struct holding results must be able to derive.
    #[derive(Debug, Clone, PartialEq)]
    #[allow(dead_code)]
    struct Cached {
        positional: turbovec::SearchResults,
        by_id: turbovec::IdSearchResults,
    }
}

/// The headline concurrency claim, exercised the way a service would:
/// one shared index, readers in parallel, then a writer.
fn shared_index_across_threads(db: &[f32], queries: &[f32]) {
    let mut index = TurboQuantIndex::new(DIM, 4).expect("construct");
    index.add(db);
    index.prepare();
    let shared = Arc::new(RwLock::new(index));

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let handle = Arc::clone(&shared);
            let q = queries.to_vec();
            std::thread::spawn(move || {
                let guard = handle.read().expect("read lock");
                let r = guard.search(&q, K);
                assert_eq!(r.nq, N_QUERIES);
                assert_eq!(r.k, K);
                r.indices
            })
        })
        .collect();
    let rows: Vec<Vec<i64>> = readers.into_iter().map(|h| h.join().expect("join")).collect();
    assert!(
        rows.windows(2).all(|w| w[0] == w[1]),
        "concurrent searches disagreed"
    );

    // Writer through the same lock: an add after a search must invalidate
    // the caches the searches materialized, not serve stale ones.
    {
        let mut guard = shared.write().expect("write lock");
        guard.add(&unit_vectors(1, DIM, 99));
        assert_eq!(guard.len(), N_DB + 1);
        let r = guard.search(&queries[..DIM], K);
        assert_eq!(r.k, K);
    }
}

/// Error paths: the `Result`-returning API, matched the way a dependent
/// must match a `#[non_exhaustive]` enum — with a wildcard arm.
fn error_paths(db: &[f32]) {
    let mut index = TurboQuantIndex::new(DIM, 4).expect("construct");

    // Wrong dim against an index with a committed dim.
    match index.add_2d(db, DIM + 8) {
        Err(AddError::DimMismatch { existing, got }) => {
            assert_eq!((existing, got), (DIM, DIM + 8));
        }
        other => panic!("expected DimMismatch, got {other:?}"),
    }

    // A non-finite coordinate, matched with the wildcard arm that a
    // `#[non_exhaustive]` enum forces on every downstream `match`.
    let mut poisoned = db[..DIM].to_vec();
    poisoned[5] = f32::INFINITY;
    match index.add_2d(&poisoned, DIM) {
        Err(AddError::InvalidInputValue { coord_index, .. }) => assert_eq!(coord_index, 5),
        Err(other) => panic!("expected InvalidInputValue, got {other:?}"),
        Ok(()) => panic!("non-finite coordinate accepted"),
    }

    index.add(db);

    // The non-panicking search forms report, rather than abort the thread.
    match index.try_search(&db[..DIM + 1], K) {
        Err(SearchError::QueryBufferNotMultipleOfDim { .. }) => {}
        other => panic!("expected QueryBufferNotMultipleOfDim, got {other:?}"),
    }
    let mut bad = db[..DIM].to_vec();
    bad[3] = f32::NAN;
    match index.try_search(&bad, K) {
        Err(SearchError::InvalidQueryValue { .. }) => {}
        other => panic!("expected InvalidQueryValue, got {other:?}"),
    }
    assert!(
        index
            .try_search_with_mask(&db[..DIM], K, Some(&[true; 3]))
            .is_err(),
        "a mask shorter than the index must be rejected"
    );

    // Errors are usable as `anyhow`/`Box<dyn Error>` payloads, print, and
    // compare — the properties a service needs to map one onto a 400.
    let e = index.try_search(&db[..DIM + 1], K).unwrap_err();
    let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(e.clone());
    assert!(!boxed.to_string().is_empty());
    assert_eq!(e, index.try_search(&db[..DIM + 1], K).unwrap_err());
}

/// Mutation, masking, and the `k > len` clamp — none of which the original
/// smoke test reached, since it used K=5 against 256 vectors.
fn mutation_masking_and_clamping(db: &[f32]) {
    let mut index = TurboQuantIndex::new(DIM, 4).expect("construct");
    index.add(&db[..DIM * 3]);

    // k above len clamps, and the struct says so.
    let r = index.search(&db[..DIM], 10);
    assert_eq!(r.k, 3, "k must clamp to len");
    assert_eq!(r.indices.len(), 3);
    assert_eq!(r.indices_for_query(0).len(), 3);

    // A mask narrows the effective k further.
    let masked = index.search_with_mask(&db[..DIM], 10, Some(&[true, false, true]));
    assert_eq!(masked.k, 2, "k must clamp to the allowed count");
    assert!(!masked.indices.contains(&1), "masked-out slot was returned");

    // swap_remove moves the last vector into the freed slot.
    index.add(&db[DIM * 3..DIM * 5]);
    assert_eq!(index.len(), 5);
    index.swap_remove(0);
    assert_eq!(index.len(), 4);
    // Search still works against the mutated index — the packed/blocked
    // caches the searches above materialized must have been invalidated.
    let after = index.search(&db[..DIM], 4);
    assert_eq!(after.k, 4);
    assert!(after.indices.iter().all(|&i| (0..4).contains(&i)));
}

/// Serialization: both sinks, both durability levels, and `from_parts`.
fn serialization(db: &[f32], queries: &[f32]) {
    let mut index = TurboQuantIndex::new(DIM, 4).expect("construct");
    index.add(db);
    let baseline = index.search(queries, K);

    let dir = std::env::temp_dir();
    let durable = dir.join("turbovec_downstream_smoke_durable.tv");
    let fast = dir.join("turbovec_downstream_smoke_fast.tv");
    index.write(&durable).expect("write");
    index
        .write_with_durability(&fast, Durability::Fast)
        .expect("write_with_durability");
    let a = std::fs::read(&durable).expect("read durable");
    let b = std::fs::read(&fast).expect("read fast");
    assert_eq!(a, b, "Durability must not change the bytes");

    // The in-memory sink produces the same bytes as the file.
    let bytes = index.to_bytes();
    assert_eq!(bytes, a, "to_bytes must match the on-disk payload");

    let loaded = TurboQuantIndex::load(&durable).expect("load");
    let from_mem = TurboQuantIndex::from_bytes(&bytes).expect("from_bytes");
    for (label, idx) in [("load", &loaded), ("from_bytes", &from_mem)] {
        assert_eq!(idx.len(), N_DB, "{label}");
        assert_eq!(idx.dim_opt(), Some(DIM), "{label}");
        let r = idx.search(queries, K);
        assert_eq!(r.indices, baseline.indices, "{label}: results changed");
        assert_eq!(r.scores, baseline.scores, "{label}: scores changed");
    }
    let _ = std::fs::remove_file(&durable);
    let _ = std::fs::remove_file(&fast);

    // Rebuilding from raw parts is a public path too.
    let rebuilt = TurboQuantIndex::from_parts(
        Some(DIM),
        4,
        loaded.len(),
        loaded.packed_codes().to_vec(),
        loaded.scales().to_vec(),
        loaded.tqplus_shift().to_vec(),
        loaded.tqplus_scale().to_vec(),
    )
    .expect("from_parts");
    assert_eq!(rebuilt.len(), N_DB);
    assert_eq!(rebuilt.search(queries, K).indices, baseline.indices);
}

/// `IdMapIndex` — the API most applications actually want, and previously
/// untouched from the downstream build path.
fn id_map(db: &[f32], queries: &[f32]) {
    let ids: Vec<u64> = (0..N_DB as u64).map(|i| (i + 1) * 1000).collect();
    let mut index = IdMapIndex::new(DIM, 4).expect("construct");
    index.add_with_ids(db, &ids).expect("add_with_ids");
    assert_eq!(index.len(), N_DB);
    assert!(index.contains(1000));
    index.prepare();

    // Duplicate ids are rejected, with a matchable error.
    match index.add_with_ids(&db[..DIM], &[1000]) {
        Err(AddError::IdAlreadyPresent(1000)) => {}
        other => panic!("expected IdAlreadyPresent, got {other:?}"),
    }
    // Unlike TurboQuantIndex::add_2d, which documents a ragged buffer as a
    // caller-side bug and panics, the id-mapped form reports it.
    match index.add_with_ids_2d(&db[..DIM + 1], DIM, &[999]) {
        Err(AddError::VectorBufferNotMultipleOfDim { .. }) => {}
        other => panic!("expected VectorBufferNotMultipleOfDim, got {other:?}"),
    }
    assert_eq!(index.len(), N_DB, "a rejected add must not mutate the index");

    // Self-describing results carry nq and the effective k.
    let r = index.try_search(queries, K).expect("try_search");
    assert_eq!(r.nq, N_QUERIES);
    assert_eq!(r.k, K);
    assert_eq!(r.ids.len(), N_QUERIES * K);
    for q in 0..N_QUERIES {
        assert_eq!(r.ids_for_query(q).len(), K);
        assert!(r.ids_for_query(q).iter().all(|id| ids.contains(id)));
    }
    // The tuple form agrees with it.
    let (scores, tuple_ids) = index.search(queries, K);
    assert_eq!(scores, r.scores);
    assert_eq!(tuple_ids, r.ids);

    // Allowlist restriction, and its error cases.
    let allow = &ids[..3];
    let restricted = index
        .try_search_with_allowlist(queries, K, Some(allow))
        .expect("allowlist search");
    assert_eq!(restricted.k, 3, "k clamps to the allowlist size");
    assert!(restricted.ids.iter().all(|id| allow.contains(id)));
    assert!(matches!(
        index.search_with_allowlist(queries, K, Some(&[])),
        Err(SearchError::AllowlistEmpty)
    ));
    assert!(matches!(
        index.search_with_allowlist(queries, K, Some(&[7])),
        Err(SearchError::UnknownId(7))
    ));

    // Removal keeps every surviving id addressable.
    assert!(index.remove(1000));
    assert!(!index.remove(1000), "second remove must report absent");
    assert_eq!(index.len(), N_DB - 1);
    assert!(!index.contains(1000));
    let live: Vec<u64> = index.iter_ids().collect();
    assert_eq!(live.len(), N_DB - 1);
    assert!(!live.contains(&1000));
    assert!(index
        .try_search(queries, K)
        .expect("search after remove")
        .ids
        .iter()
        .all(|id| *id != 1000));

    // Round-trip through both sinks.
    let path = std::env::temp_dir().join("turbovec_downstream_smoke.tvim");
    index.write(&path).expect("write");
    let loaded = IdMapIndex::load(&path).expect("load");
    let from_mem = IdMapIndex::from_bytes(&index.to_bytes()).expect("from_bytes");
    let _ = std::fs::remove_file(&path);
    let baseline = index.try_search(queries, K).expect("baseline");
    for (label, idx) in [("load", &loaded), ("from_bytes", &from_mem)] {
        assert_eq!(idx.len(), N_DB - 1, "{label}");
        assert_eq!(idx.dim_opt(), Some(DIM), "{label}");
        assert!(!idx.contains(1000), "{label}: removed id came back");
        assert_eq!(
            idx.try_search(queries, K).expect("search").ids,
            baseline.ids,
            "{label}: results changed"
        );
    }

    // And it is shareable the same way the positional index is.
    let shared = Arc::new(RwLock::new(loaded));
    let handle = Arc::clone(&shared);
    let q = queries.to_vec();
    let n = std::thread::spawn(move || {
        handle
            .read()
            .expect("read lock")
            .try_search(&q, K)
            .expect("search")
            .ids
            .len()
    })
    .join()
    .expect("join");
    assert_eq!(n, N_QUERIES * K);
}

fn main() {
    let db = unit_vectors(N_DB, DIM, 1);
    let queries = unit_vectors(N_QUERIES, DIM, 2);

    trait_bounds();

    let mut index = TurboQuantIndex::new(DIM, 4).expect("construct");
    index.add(&db);
    index.prepare();

    let results = index.search(&queries, K);
    assert_eq!(results.nq, N_QUERIES);
    assert_eq!(results.k, K);
    for q in 0..N_QUERIES {
        let idxs = results.indices_for_query(q);
        assert_eq!(idxs.len(), K);
        for &i in idxs {
            assert!((0..N_DB as i64).contains(&i), "out-of-range index {i}");
        }
    }

    let tmp = std::env::temp_dir().join("turbovec_downstream_smoke.tv");
    index.write(&tmp).expect("write");
    let loaded = TurboQuantIndex::load(&tmp).expect("load");
    assert_eq!(loaded.len(), N_DB);
    assert_eq!(loaded.dim_opt(), Some(DIM));
    let _ = std::fs::remove_file(&tmp);

    shared_index_across_threads(&db, &queries);
    error_paths(&db);
    mutation_masking_and_clamping(&db);
    serialization(&db, &queries);
    id_map(&db, &queries);

    println!(
        "downstream-smoke: OK ({} db vectors, {} queries, top-{} search, write/load round-trip, \
         trait bounds, Arc<RwLock<_>> concurrency, error paths, mutation + masking + k-clamping, \
         to_bytes/from_bytes + Durability + from_parts, IdMapIndex)",
        N_DB, N_QUERIES, K
    );
}
