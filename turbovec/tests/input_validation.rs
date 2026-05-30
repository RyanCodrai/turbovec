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

use turbovec::{AddError, IdMapIndex, TurboQuantIndex};

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

#[test]
#[should_panic(expected = "invalid query value")]
fn id_map_search_panics_on_nan_query() {
    let mut idx = IdMapIndex::new(DIM, 4).unwrap();
    idx.add_with_ids(&ok_vector(), &[1]).unwrap();
    let mut query = vec![0.0f32; DIM];
    query[0] = f32::NAN;
    let _ = idx.search(&query, 1);
}
