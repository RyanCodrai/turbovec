//! Regression tests for [`TurboQuantIndex::suggested_params`] initialization.
//!
//! Exercises:
//!   - `suggested_params()` returns a dict-like struct
//!   - The returned struct has expected fields (ndim, metric, precision)
//!   - Values are of appropriate types/ranges
//!   - Params are stable after `build()` / `add()` is called

extern crate blas_src;

use turbovec::{IndexParameters, TurboQuantIndex};

const DIM: usize = 128;

/// Build a small index for testing.
fn build_index(dim: usize, bit_width: usize) -> TurboQuantIndex {
    let n = 64;
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut vectors = Vec::with_capacity(n * dim);
    for _ in 0..(n * dim) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        vectors.push(u * 2.0 - 1.0);
    }
    let mut index = TurboQuantIndex::new(dim, bit_width);
    index.add(&vectors);
    index
}

#[test]
fn test_suggested_params_returns_parameters_struct() {
    let index = build_index(DIM, 4);
    let params = index.suggested_params();
    // Verify it returns an IndexParameters struct (not void, not panic)
    let _ = format!("{:?}", params);
}

#[test]
fn test_suggested_params_keys_exist() {
    let index = build_index(DIM, 4);
    let params = index.suggested_params();
    // Expected keys: ndim, metric, precision
    assert!(
        params.ndim > 0,
        "ndim should be positive, got {}",
        params.ndim
    );
    assert_eq!(params.metric, "cosine");
    assert!((2..=4).contains(&params.precision));
}

#[test]
fn test_suggested_params_values_are_valid() {
    let index = build_index(DIM, 2);
    let params = index.suggested_params();
    // ndim should match the index dim
    assert_eq!(params.ndim, DIM);
    // metric should be cosine
    assert_eq!(params.metric, "cosine");
    // precision should be 2
    assert_eq!(params.precision, 2);
}

#[test]
fn test_suggested_params_after_build() {
    let mut index = TurboQuantIndex::new(DIM, 3);
    // Check before add
    let params_before = index.suggested_params();
    assert_eq!(params_before.ndim, DIM);

    // Add vectors and check again
    let n = 32;
    let mut vectors = vec![0.0f32; n * DIM];
    for v in &mut vectors {
        *v = 0.1;
    }
    index.add(&vectors);

    let params_after = index.suggested_params();
    // Params should be stable after build
    assert_eq!(params_after.ndim, params_before.ndim);
    assert_eq!(params_after.metric, params_before.metric);
    assert_eq!(params_after.precision, params_before.precision);
}

#[test]
fn test_suggested_params_different_bit_widths() {
    for bits in [2, 3, 4] {
        let index = build_index(DIM, bits);
        let params = index.suggested_params();
        assert_eq!(
            params.precision, bits,
            "precision should be {} for bit_width={}",
            bits, bits
        );
    }
}

#[test]
fn test_index_parameters_equality() {
    let index1 = build_index(DIM, 4);
    let index2 = build_index(DIM, 4);
    let params1 = index1.suggested_params();
    let params2 = index2.suggested_params();
    assert_eq!(params1, params2);
}

#[test]
fn test_index_parameters_clone() {
    let index = build_index(DIM, 4);
    let params = index.suggested_params();
    let cloned = params.clone();
    assert_eq!(params, cloned);
}