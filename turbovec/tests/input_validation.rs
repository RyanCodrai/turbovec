//! Input-validation tests for `TurboQuantIndex` and `IdMapIndex`.
//!
//! Before the validation was added, the encode pipeline silently
//! corrupted the index on degenerate inputs:
//!   - NaN coord poisoned `vec_scales[slot]` via `0 * NaN = NaN`,
//!     so the slot existed in `len()` but was unreachable through
//!     `search`.
//!   - +/-Inf coord followed the same NaN-poisoning path.
//!   - Magnitude `>= 1e16` overflowed the f32 sum-of-squares in the
//!     norm computation, storing `scale[i] = Inf` and incorrectly
//!     ranking the slot at the top of every query.
//!   - NaN/Inf query value produced arbitrary indices with NaN scores
//!     because the heap's `s > heap_min` comparison is false for NaN.
//!
//! These tests pin the typed-error behaviour on the `_2d` paths and
//! the panic behaviour on the flat / search paths.

use turbovec::{AddError, IdMapIndex, SearchError, TurboQuantIndex};

const DIM: usize = 64;

fn ok_vector() -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[0] = 1.0;
    v
}

// ---- TurboQuantIndex::add_2d returns typed error on bad input ----

#[test]
fn add_2d_rejects_nan_with_invalid_input_value_error() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    data[5] = f32::NAN;
    let err = idx.add_2d(&data, DIM).unwrap_err();
    match err {
        AddError::InvalidInputValue {
            vector_index,
            coord_index,
            value,
        } => {
            assert_eq!(vector_index, 0);
            assert_eq!(coord_index, 5);
            assert!(value.is_nan(), "expected NaN, got {value}");
        }
        other => panic!("expected InvalidInputValue, got {other:?}"),
    }
    // No vectors were actually added.
    assert_eq!(idx.len(), 0);
}

#[test]
fn add_2d_rejects_positive_infinity() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    data[10] = f32::INFINITY;
    let err = idx.add_2d(&data, DIM).unwrap_err();
    assert!(
        matches!(err, AddError::InvalidInputValue { coord_index: 10, .. }),
        "expected InvalidInputValue at coord 10, got {err:?}",
    );
}

#[test]
fn add_2d_rejects_negative_infinity() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    data[3] = f32::NEG_INFINITY;
    assert!(matches!(
        idx.add_2d(&data, DIM).unwrap_err(),
        AddError::InvalidInputValue { coord_index: 3, .. },
    ));
}

#[test]
fn add_2d_rejects_huge_magnitude_that_would_overflow_norm() {
    // 1e20 is finite but `(1e20)^2 = 1e40` overflows f32::MAX (~3.4e38)
    // — the scenario that previously stored `scale[i] = +Inf` and made
    // the slot incorrectly win every search.
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    data[2] = 1e20;
    let err = idx.add_2d(&data, DIM).unwrap_err();
    assert!(
        matches!(err, AddError::InvalidInputValue { coord_index: 2, .. }),
        "expected InvalidInputValue at coord 2, got {err:?}",
    );
}

#[test]
fn add_2d_accepts_values_just_under_the_magnitude_bound() {
    // 1e15 is well under 1e16, so it must be accepted. Pins the bound
    // explicitly so a future tightening doesn't silently break valid
    // (if unusual) callers.
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    data[1] = 1e15;
    idx.add_2d(&data, DIM).unwrap();
    assert_eq!(idx.len(), 1);
}

#[test]
fn add_2d_rejects_invalid_input_in_second_vector_of_batch() {
    // Multi-vector batch: the validation must report the *second* (or
    // later) vector, not just the first.
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = vec![0.0f32; 2 * DIM];
    data[0] = 1.0;
    data[DIM] = 1.0;
    data[DIM + 7] = f32::NAN;
    let err = idx.add_2d(&data, DIM).unwrap_err();
    match err {
        AddError::InvalidInputValue {
            vector_index: 1,
            coord_index: 7,
            ..
        } => {}
        other => panic!("expected vector 1 / coord 7, got {other:?}"),
    }
}

#[test]
fn add_2d_failure_does_not_commit_dim_on_lazy_index() {
    // A lazy index that rejects bad input on its first add must stay
    // uncommitted — otherwise a follow-up correct add at a different
    // dim would mis-report DimMismatch.
    let mut idx = TurboQuantIndex::new_lazy(4).unwrap();
    let mut bad = ok_vector();
    bad[0] = f32::NAN;
    assert!(idx.add_2d(&bad, DIM).is_err());
    assert_eq!(idx.dim_opt(), None, "dim should not have been committed");

    // A subsequent add at a different dim must succeed (no leftover
    // commitment from the failed call).
    let other_dim = 32;
    let mut clean = vec![0.0f32; other_dim];
    clean[0] = 1.0;
    idx.add_2d(&clean, other_dim).unwrap();
    assert_eq!(idx.dim_opt(), Some(other_dim));
}

// ---- TurboQuantIndex::add panics with a clear message on bad input ----

#[test]
#[should_panic(expected = "invalid input value")]
fn add_panics_on_nan_input() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    data[5] = f32::NAN;
    idx.add(&data);
}

#[test]
#[should_panic(expected = "invalid input value")]
fn add_panics_on_huge_magnitude_input() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    data[5] = 5e16;
    idx.add(&data);
}

// ---- search panics with a clear message on bad query ----

#[test]
#[should_panic(expected = "invalid query value")]
fn search_panics_on_nan_query() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add(&ok_vector());

    let mut query = vec![0.0f32; DIM];
    query[0] = f32::NAN;
    let _ = idx.search(&query, 1);
}

#[test]
#[should_panic(expected = "invalid query value")]
fn search_panics_on_infinity_query() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add(&ok_vector());

    let mut query = vec![0.0f32; DIM];
    query[0] = f32::INFINITY;
    let _ = idx.search(&query, 1);
}

#[test]
#[should_panic(expected = "invalid query value")]
fn search_panics_on_huge_magnitude_query() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add(&ok_vector());

    let mut query = vec![0.0f32; DIM];
    query[0] = 1e18;
    let _ = idx.search(&query, 1);
}

#[test]
fn search_on_lazy_uncommitted_skips_query_validation() {
    // A lazy-uncommitted search returns empty results without ever
    // looking at the query — so even a NaN-bearing query must not
    // panic. Pins that the early-return path bypasses validation.
    let idx = TurboQuantIndex::new_lazy(4).unwrap();
    let query = vec![f32::NAN; 8];
    let res = idx.search(&query, 1);
    assert_eq!(res.scores.len(), 0);
    assert_eq!(res.indices.len(), 0);
}

// ---- IdMapIndex inherits validation via the inner index ----

#[test]
fn id_map_add_with_ids_2d_rejects_nan_input() {
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    data[4] = f32::NAN;
    let err = idx.add_with_ids_2d(&data, DIM, &[1]).unwrap_err();
    assert!(
        matches!(err, AddError::InvalidInputValue { .. }),
        "expected InvalidInputValue, got {err:?}",
    );
    // IdMap tables must be untouched — pin the same partial-mutation
    // contract the fixed add_with_ids_2d enforces.
    assert_eq!(idx.len(), 0);
    assert!(!idx.contains(1));
}

/// `search_with_allowlist` returns `Result<_, SearchError>` and
/// `SearchError` has variants for both query-shape conditions, but they
/// used to be delivered as panics from the inner index instead (#412).
/// The realistic caller matches on the error and maps it to a 400; a
/// ragged request body took the thread down anyway. Both must now come
/// back as `Err`, with and without an allowlist — the allowlist is not
/// what makes the query well-shaped.
#[test]
fn id_map_search_with_allowlist_returns_query_shape_errors() {
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    idx.add_with_ids(&ok_vector(), &[1]).unwrap();

    let mut nan_query = vec![0.0f32; DIM];
    nan_query[3] = f32::NAN;
    let ragged = vec![0.0f32; DIM + 1];

    for allowlist in [None, Some(&[1u64][..])] {
        let err = idx
            .search_with_allowlist(&nan_query, 1, allowlist)
            .expect_err("a NaN coordinate must be reported, not panicked");
        assert!(
            matches!(err, SearchError::InvalidQueryValue { .. }),
            "expected InvalidQueryValue, got {err:?}",
        );
        let err = idx
            .search_with_allowlist(&ragged, 1, allowlist)
            .expect_err("a ragged query buffer must be reported, not panicked");
        assert!(
            matches!(err, SearchError::QueryBufferNotMultipleOfDim { .. }),
            "expected QueryBufferNotMultipleOfDim, got {err:?}",
        );
    }

    // The allowlist variants still work, so returning the query-shape
    // ones has not displaced them.
    assert!(matches!(
        idx.search_with_allowlist(&ok_vector(), 1, Some(&[])),
        Err(SearchError::AllowlistEmpty),
    ));
    assert!(matches!(
        idx.search_with_allowlist(&ok_vector(), 1, Some(&[99])),
        Err(SearchError::UnknownId(99)),
    ));
    // And a well-shaped query is unaffected.
    let (_, ids) = idx.search(&ok_vector(), 1);
    assert_eq!(ids, vec![1]);
}

/// The sibling `search` is still the panicking form, and its payload is
/// still the error's `Display`. Pinning the ragged message too: `search`
/// re-panics with `{e}`, so it must not regress into a `Debug` rendering
/// of the error behind an `.expect` string (#412).
#[test]
#[should_panic(expected = "not a multiple of dim")]
fn id_map_search_panics_on_ragged_query_with_the_descriptive_message() {
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    idx.add_with_ids(&ok_vector(), &[1]).unwrap();
    let _ = idx.search(&vec![0.0f32; DIM + 1], 1);
}

#[test]
#[should_panic(expected = "invalid query value")]
fn id_map_search_panics_on_nan_query() {
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    idx.add_with_ids(&ok_vector(), &[1]).unwrap();
    let mut query = vec![0.0f32; DIM];
    query[0] = f32::NAN;
    let _ = idx.search(&query, 1);
}

/// An empty query batch is a legal no-op, not a panic. The 2D
/// block-range tiling derives a divisor from the query-quad count, which
/// is zero for `nq == 0` — this pins the guard. Sized past
/// `SINGLE_QUERY_PARALLEL_MIN_BLOCKS` (derived, not hard-coded, so
/// raising the threshold cannot move the test off the path) and run on a
/// real multi-thread pool, since the 1-thread path short-circuits before
/// the division.
#[test]
fn search_with_zero_queries_is_a_no_op_not_a_panic() {
    const N: usize = (turbovec::search::SINGLE_QUERY_PARALLEL_MIN_BLOCKS + 25) * 32;
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = Vec::with_capacity(N * DIM);
    let mut s = 0x1234_5678u32;
    for _ in 0..N * DIM {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        data.push((s >> 8) as f32 / (1u32 << 24) as f32);
    }
    idx.add(&data);

    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let res = pool.install(|| idx.search(&[], 10));
    assert!(res.scores.is_empty(), "0 queries must yield no score rows");
    assert!(res.indices.is_empty(), "0 queries must yield no index rows");

    // Same through the id-map wrapper.
    let mut id_idx = IdMapIndex::new(DIM, 4).unwrap();
    let ids: Vec<u64> = (0..N as u64).collect();
    id_idx.add_with_ids(&data, &ids).unwrap();
    let (scores, got_ids) = pool.install(|| id_idx.search(&[], 10));
    assert!(scores.is_empty() && got_ids.is_empty());
}

/// `validation_parallelizes` must mark the exact boundary at which
/// `first_invalid_coord` splits across the current rayon pool. The Python
/// binding gates its fork-safe-pool routing on this predicate (issues #288,
/// #321), so the two must not drift: a buffer at the threshold folds on the
/// calling thread and is safe to validate inline, one float past it injects
/// work into whatever pool is current. If the underlying chunk length is
/// retuned, update this test and re-check every binding call site that
/// validates outside `with_pool`.
#[test]
fn validation_parallelizes_marks_the_chunk_boundary() {
    const CHUNK: usize = 64 * 1024;
    assert!(!turbovec::validation_parallelizes(0));
    assert!(!turbovec::validation_parallelizes(CHUNK - 1));
    assert!(!turbovec::validation_parallelizes(CHUNK));
    assert!(turbovec::validation_parallelizes(CHUNK + 1));
    assert!(turbovec::validation_parallelizes(CHUNK * 4));
}

/// A single maximum-width row must stay inside one validation chunk. The
/// binding's single-row add bypass no longer *relies* on this (it gates on
/// `validation_parallelizes` directly), but if this stops holding, one-row
/// adds start paying a pool `install` — a silent latency change worth
/// noticing deliberately rather than discovering in a benchmark.
#[test]
fn one_max_width_row_does_not_split_validation() {
    assert!(!turbovec::validation_parallelizes(turbovec::MAX_DIM));
}

// ---- #286: near-zero-norm vectors are accepted with documented semantics ----
//
// A vector whose norm is <= MIN_INPUT_NORM has no representable
// direction, so it is stored with scale 0 and scores exactly 0. That is
// deliberate, not a silent failure: 0 is the conventional cosine of a
// zero vector, and the framework integrations rely on it (a zero
// document embedding must be storable and rank last, not raise). These
// tests pin that contract so the behaviour cannot drift silently.

#[test]
fn zero_norm_vector_is_stored_and_counted() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    data.extend(vec![0.0f32; DIM]);
    idx.add_2d(&data, DIM).expect("zero-norm vectors are accepted");
    assert_eq!(idx.len(), 2);
}

#[test]
fn zero_norm_vector_scores_zero_and_ranks_below_real_vectors() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = ok_vector();
    // Coords ~1e-23 are finite but x*x underflows f32, so the norm is 0
    // — the exact input from #286.
    data.extend(vec![1e-23f32; DIM]);
    idx.add_2d(&data, DIM).unwrap();

    let res = idx.search(&ok_vector(), 2);
    assert_eq!(res.indices.len(), 2);
    let pos = res.indices.iter().position(|&i| i == 1).expect("slot 1 returned");
    assert_eq!(res.scores[pos], 0.0, "zero-norm vector must score exactly 0");
    assert_eq!(res.indices[0], 0, "the vector with a real direction ranks first");
}
