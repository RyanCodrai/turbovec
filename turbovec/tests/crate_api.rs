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

/// The exact panic payload `f` produces, with the default hook muted.
///
/// `set_hook` is process-global — there is no thread-local form — and
/// the tests in this binary run concurrently, so while the hook is muted
/// a panic in any *other* test has its diagnostics swallowed. This repo
/// keeps its `FORCE_*` panic switches thread-local for exactly that
/// hazard (#373), but a hook has no such escape.
///
/// The mutex below does NOT close that window, and should not be read as
/// doing so: it only serializes callers of this helper against each
/// other, and there is currently one. Other tests never take it, so a
/// concurrent unrelated panic is still silenced. It is here so that the
/// window cannot be *nested* or widened if a second caller is added
/// later. The real mitigation is that the window is a few microseconds
/// around a call that is expected to panic; if a test in this file ever
/// fails mysteriously with no output, this is the first thing to suspect.
static HOOK_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn panic_payload<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> String {
    // A poisoned guard just means a previous holder panicked outside its
    // `catch_unwind`; the mutex protects no data, so recover and carry on.
    let _guard = HOOK_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    let e = caught.unwrap_err();
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("panic payload should be a string")
}

#[test]
fn the_panicking_forms_panic_with_exactly_the_error_display() {
    // Substring assertions are what let a false "panic messages are
    // identical" claim through review: `should_panic(expected = ...)`
    // matches on a substring, so it stayed green while the payload
    // changed from an `assert_eq!` rendering to the bare message. Pin
    // the *whole* payload, and pin it as equal to the `Display` of the
    // error the checked form returns for the same input — that
    // equality is the actual contract between the two forms.
    let idx = index();

    let ragged = vec![0.0f32; DIM + 1];
    assert_eq!(
        panic_payload(|| {
            idx.search(&ragged, 4);
        }),
        idx.try_search(&ragged, 4).unwrap_err().to_string(),
    );
    assert_eq!(
        panic_payload(|| {
            idx.search(&ragged, 4);
        }),
        "query buffer length 65 not a multiple of dim 64",
    );

    let mut nan = query();
    nan[7] = f32::NAN;
    assert_eq!(
        panic_payload(|| {
            idx.search(&nan, 4);
        }),
        idx.try_search(&nan, 4).unwrap_err().to_string(),
    );

    let mask = vec![true; N + 1];
    assert_eq!(
        panic_payload(|| {
            idx.search_with_mask(&query(), 4, Some(&mask));
        }),
        idx.try_search_with_mask(&query(), 4, Some(&mask))
            .unwrap_err()
            .to_string(),
    );
    assert_eq!(
        panic_payload(|| {
            idx.search_with_mask(&query(), 4, Some(&mask));
        }),
        "mask length 17 does not match index size 16",
    );
}

#[test]
fn search_results_partial_eq_follows_ieee_not_bit_equality() {
    // Pins what the type's docs claim about the derived `PartialEq`,
    // since "compares scores bitwise" is wrong in both directions.
    let base = SearchResults {
        scores: vec![0.0],
        indices: vec![0],
        nq: 1,
        k: 1,
    };

    // NaN: identical bits, not equal — so a NaN-carrying result is not
    // equal to its own clone. Assert the bits really are identical, or
    // the inequality would prove nothing about IEEE semantics.
    let nan = SearchResults {
        scores: vec![f32::NAN],
        ..base.clone()
    };
    let nan_clone = nan.clone();
    assert_eq!(
        nan.scores[0].to_bits(),
        nan_clone.scores[0].to_bits(),
        "the clone should be bit-identical, or this test proves nothing",
    );
    assert_ne!(nan, nan_clone);

    // Signed zero: different bits, equal.
    let neg_zero = SearchResults {
        scores: vec![-0.0],
        ..base.clone()
    };
    assert_eq!(base, neg_zero);
    assert_ne!(
        base.scores[0].to_bits(),
        neg_zero.scores[0].to_bits(),
        "the two zeros should differ bitwise, or this test proves nothing",
    );
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
