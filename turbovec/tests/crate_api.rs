//! The crate's surface as a downstream dependency (#351).
//!
//! Two things a `cargo add turbovec` user needs that nothing else in the
//! suite pins:
//!
//!   - `SearchResults` — the type *every* search returns — must implement
//!     the traits a downstream struct holding one needs in order to
//!     derive its own. Without `Debug` a caller cannot `#[derive(Debug)]`
//!     around it or `dbg!` it; without `Clone` it cannot be cached;
//!     without `PartialEq` it cannot appear in a user's `assert_eq!`.
//!     The trait-bound assertions below fail to *compile* if the derives
//!     are dropped, which is the only way to catch that class.
//!
//!   - The search path must have a non-panicking form. `add_2d` returns
//!     `AddError` and the Python binding pre-validates the same three
//!     query conditions and raises, but a Rust service handed a ragged
//!     buffer, a NaN coordinate or a stale mask had no option but an
//!     aborted request thread. `try_search` / `try_search_with_mask`
//!     return `SearchError` instead.

use turbovec::{SearchError, SearchResults, TurboQuantIndex};

const DIM: usize = 64;
const N: usize = 16;

fn index() -> TurboQuantIndex {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mut data = vec![0.0f32; N * DIM];
    for (i, row) in data.chunks_mut(DIM).enumerate() {
        row[i % DIM] = 1.0;
    }
    idx.add(&data);
    idx
}

fn query() -> Vec<f32> {
    let mut q = vec![0.0f32; DIM];
    q[0] = 1.0;
    q
}

// ---- SearchResults derives (#351 finding 1) ----

fn assert_debug<T: std::fmt::Debug>() {}
fn assert_clone<T: Clone>() {}
fn assert_partial_eq<T: PartialEq>() {}

#[test]
fn search_results_implements_the_downstream_derive_set() {
    // Compile-time half: these three lines are the test. If any derive
    // is removed from `SearchResults` this file stops compiling.
    assert_debug::<SearchResults>();
    assert_clone::<SearchResults>();
    assert_partial_eq::<SearchResults>();

    // Behavioural half: the derives have to mean something. A clone
    // compares equal to its source, `Debug` renders the shape fields,
    // and results of different shapes compare unequal.
    let idx = index();
    let res = idx.search(&query(), 4);
    let copy = res.clone();
    assert_eq!(res, copy);
    assert_eq!(copy.k, 4);

    let rendered = format!("{res:?}");
    assert!(
        rendered.contains("SearchResults") && rendered.contains("nq"),
        "Debug output should name the type and its fields, got {rendered}"
    );

    let narrower = idx.search(&query(), 2);
    assert_ne!(res, narrower);
}

#[test]
fn search_results_clone_is_independent_of_its_source() {
    let idx = index();
    let res = idx.search(&query(), 4);
    let mut copy = res.clone();
    copy.scores[0] = -1.0;
    assert_ne!(res.scores[0], copy.scores[0]);
}

// ---- try_search: the three panic conditions become errors (#351 finding 2) ----

#[test]
fn try_search_reports_a_ragged_query_buffer() {
    let idx = index();
    let ragged = vec![0.0f32; DIM + 1];
    match idx.try_search(&ragged, 4) {
        Err(SearchError::QueryBufferNotMultipleOfDim { queries_len, dim }) => {
            assert_eq!(queries_len, DIM + 1);
            assert_eq!(dim, DIM);
        }
        other => panic!("expected QueryBufferNotMultipleOfDim, got {other:?}"),
    }
}

#[test]
fn try_search_reports_a_non_finite_query_coordinate() {
    let idx = index();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e17] {
        let mut q = query();
        q[7] = bad;
        match idx.try_search(&q, 4) {
            Err(SearchError::InvalidQueryValue {
                query_index,
                coord_index,
                value,
            }) => {
                assert_eq!(query_index, 0);
                assert_eq!(coord_index, 7);
                assert!(
                    value.is_nan() == bad.is_nan() && (value.is_nan() || value == bad),
                    "reported value {value} should be the offending {bad}",
                );
            }
            other => panic!("expected InvalidQueryValue for {bad}, got {other:?}"),
        }
    }
}

#[test]
fn try_search_with_mask_reports_a_mask_length_mismatch() {
    let idx = index();
    let mask = vec![true; N + 3];
    match idx.try_search_with_mask(&query(), 4, Some(&mask)) {
        Err(SearchError::MaskLengthMismatch { expected, got }) => {
            assert_eq!(expected, N);
            assert_eq!(got, N + 3);
        }
        other => panic!("expected MaskLengthMismatch, got {other:?}"),
    }
}

#[test]
fn try_search_on_an_empty_index_still_reports_a_non_empty_mask() {
    // The empty-index early return sits before the mask build, so it
    // needs its own check — it is the one place a mask can be wrong
    // without the builder ever running.
    let idx = TurboQuantIndex::new(DIM, 4).unwrap();
    let mask = vec![true; 2];
    match idx.try_search_with_mask(&query(), 4, Some(&mask)) {
        Err(SearchError::MaskLengthMismatch { expected, got }) => {
            assert_eq!(expected, 0);
            assert_eq!(got, 2);
        }
        other => panic!("expected MaskLengthMismatch, got {other:?}"),
    }
}

// ---- try_search agrees with search wherever search does not panic ----

#[test]
fn try_search_returns_exactly_what_search_returns() {
    let idx = index();
    let mut queries = query();
    queries.extend(query());

    assert_eq!(idx.try_search(&queries, 4).unwrap(), idx.search(&queries, 4));

    let mask: Vec<bool> = (0..N).map(|i| i % 2 == 0).collect();
    assert_eq!(
        idx.try_search_with_mask(&queries, 4, Some(&mask)).unwrap(),
        idx.search_with_mask(&queries, 4, Some(&mask)),
    );

    // A lazy index with no committed dim: both forms return the same
    // empty shape rather than erroring, since there is no dim to
    // validate the buffer against.
    let lazy = TurboQuantIndex::new_lazy(4).unwrap();
    let empty = lazy.try_search(&queries, 4).unwrap();
    assert_eq!(empty, lazy.search(&queries, 4));
    assert_eq!((empty.nq, empty.k), (0, 0));
}

#[test]
fn try_search_errors_carry_the_panic_text_search_uses() {
    // `search_with_mask` panics with the error's `Display`, so the two
    // forms cannot drift apart in what they report.
    let idx = index();
    let mut q = query();
    q[7] = f32::NAN;
    let err = idx.try_search(&q, 4).unwrap_err();
    assert!(
        err.to_string().contains("invalid query value"),
        "got {err}"
    );

    let mask = vec![true; N + 1];
    let err = idx
        .try_search_with_mask(&query(), 4, Some(&mask))
        .unwrap_err();
    assert!(err.to_string().contains("mask length"), "got {err}");
}

#[test]
fn search_error_is_a_std_error_that_composes() {
    // The whole point of a typed error is that it reaches a `?` in a
    // downstream handler. `Box<dyn Error + Send + Sync>` is what
    // anyhow/thiserror-shaped code needs.
    fn handler(idx: &TurboQuantIndex, q: &[f32]) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        Ok(idx.try_search(q, 4)?.k)
    }
    let idx = index();
    assert_eq!(handler(&idx, &query()).unwrap(), 4);
    assert!(handler(&idx, &vec![0.0f32; DIM + 1]).is_err());
}
