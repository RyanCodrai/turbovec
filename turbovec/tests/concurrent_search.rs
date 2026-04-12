//! Integration tests for concurrent search on a shared `TurboQuantIndex`.
//!
//! These tests exercise the `OnceLock`-based lazy cache initialisation
//! introduced to let `search` take `&self`. They verify that:
//!
//! 1. A shared `Arc<TurboQuantIndex>` can serve searches from many
//!    threads concurrently without external locking.
//! 2. Every concurrent thread that searches the same query against a
//!    shared index gets the same top-`k` back.
//! 3. `prepare()` is safe to call from multiple threads.
//! 4. A roundtrip through `write`/`load` produces the same top-`k` as
//!    the original in-memory index for the same queries.

use std::sync::Arc;
use std::thread;

use turbovec::TurboQuantIndex;

/// Deterministic pseudo-random vector generator so the tests are
/// reproducible without pulling an extra dev-dependency.
fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..(n * dim) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        // Centre on zero and scale modestly so rotation has room to move things.
        out.push(u * 2.0 - 1.0);
    }
    out
}

/// Build a small index ready for concurrent search.
fn build_index() -> TurboQuantIndex {
    let dim = 256;
    let bit_width = 4;
    let n = 1_024;

    let vectors = make_vectors(n, dim, 1);
    let mut index = TurboQuantIndex::new(dim, bit_width);
    index.add(&vectors);
    assert_eq!(index.len(), n);
    index
}

#[test]
fn search_is_deterministic_across_threads() {
    let index = Arc::new(build_index());

    // Warm up explicitly so no thread races on the lazy init path.
    index.prepare();

    let queries = make_vectors(4, index.dim(), 42);
    let k = 10;

    // Reference result from the main thread.
    let reference = index.search(&queries, k);
    let ref_indices: Vec<Vec<i64>> = (0..reference.nq)
        .map(|qi| reference.indices_for_query(qi).to_vec())
        .collect();
    let nq = reference.nq;

    // Every spawned thread should see the same indices for the same
    // queries — OnceLock caches must be visible to all readers.
    let mut handles = Vec::new();
    for _ in 0..16 {
        let index = Arc::clone(&index);
        let queries = queries.clone();
        let ref_indices = ref_indices.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..32 {
                let result = index.search(&queries, k);
                assert_eq!(result.nq, nq);
                assert_eq!(result.k, k);
                for (qi, expected) in ref_indices.iter().enumerate() {
                    assert_eq!(
                        result.indices_for_query(qi),
                        expected.as_slice(),
                        "top-k mismatch between threads for query {qi}"
                    );
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("search thread panicked");
    }
}

#[test]
fn lazy_init_is_safe_when_prepare_is_skipped() {
    // Deliberately do NOT call `prepare()` — the first concurrent
    // batch races through `get_or_init`, and OnceLock must let exactly
    // one thread initialise each cache while the others block briefly
    // then read the shared value.
    let index = Arc::new(build_index());
    let queries = make_vectors(2, index.dim(), 7);
    let k = 5;

    let mut handles = Vec::new();
    for _ in 0..32 {
        let index = Arc::clone(&index);
        let queries = queries.clone();
        handles.push(thread::spawn(move || {
            let r = index.search(&queries, k);
            // The only thing we can assert without a reference value
            // is that every call returns `nq * k` indices and no panic.
            assert_eq!(r.indices.len(), r.nq * r.k);
            assert_eq!(r.scores.len(), r.nq * r.k);
        }));
    }
    for h in handles {
        h.join().expect("search thread panicked");
    }
}

#[test]
fn prepare_is_idempotent_from_multiple_threads() {
    let index = Arc::new(build_index());

    let mut handles = Vec::new();
    for _ in 0..8 {
        let index = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            index.prepare();
            index.prepare();
        }));
    }
    for h in handles {
        h.join().expect("prepare thread panicked");
    }

    // The index must still be usable afterwards.
    let queries = make_vectors(1, index.dim(), 99);
    let r = index.search(&queries, 3);
    assert_eq!(r.k, 3);
}

#[test]
fn add_after_search_invalidates_blocked_cache() {
    // The `add` method must reset the `blocked` OnceLock so the next
    // search sees the extended vector set. If we forgot to invalidate,
    // the search would still score against the pre-`add` packed codes.
    let mut index = build_index();
    let queries = make_vectors(1, index.dim(), 3);
    let _before = index.search(&queries, 5);

    // Add 512 more vectors — roughly doubling the index.
    let more = make_vectors(512, index.dim(), 11);
    index.add(&more);
    assert_eq!(index.len(), 1_024 + 512);

    // The top-5 after `add` must come from an index of `len() = 1536`.
    // The only invariant we can check cheaply is that every returned
    // position is in-range for the new size.
    let after = index.search(&queries, 5);
    for &idx in after.indices_for_query(0) {
        assert!(idx >= 0, "negative index");
        assert!((idx as usize) < index.len(), "stale index out of range");
    }
}

#[test]
fn write_load_preserves_concurrent_search_results() {
    let index = build_index();
    let queries = make_vectors(3, index.dim(), 123);
    let k = 8;

    let before = index.search(&queries, k);

    let tmp = std::env::temp_dir().join(format!("turbovec_concurrent_{}.tv", std::process::id()));
    index.write(&tmp).expect("write");
    let reloaded = TurboQuantIndex::load(&tmp).expect("load");
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(reloaded.len(), index.len());
    assert_eq!(reloaded.dim(), index.dim());
    assert_eq!(reloaded.bit_width(), index.bit_width());

    // The reloaded index must produce the same top-k for the same
    // queries because rotation and centroids are deterministic
    // functions of `(dim, seed)` and `(bit_width, dim)`.
    let after = reloaded.search(&queries, k);
    for qi in 0..before.nq {
        assert_eq!(
            before.indices_for_query(qi),
            after.indices_for_query(qi),
            "roundtrip changed top-k for query {qi}"
        );
    }
}
