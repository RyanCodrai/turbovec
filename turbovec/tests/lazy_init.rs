//! Tests for lazy index construction on `TurboQuantIndex` and `IdMapIndex`.
//!
//! Invariants exercised:
//!   - `new_lazy(bit_width)` constructs an index with no committed dim.
//!   - `dim_opt()` returns `None` before the first add and `Some(d)` after.
//!   - `dim()` returns `0` as a sentinel before the first add.
//!   - `add_2d` / `add_with_ids_2d` lock the dim on first call and require
//!     a matching dim on subsequent calls.
//!   - `search` on a lazy uncommitted index returns empty results (no panic).
//!   - `prepare` on a lazy uncommitted index is a no-op.
//!   - `add` (the dim-implicit form) panics on a lazy uncommitted index.
//!   - File format: `write` from a lazy uncommitted index produces a file
//!     that loads back into a lazy uncommitted state; `write` from a
//!     committed index round-trips exactly.

extern crate blas_src;

use std::fs;
use turbovec::{IdMapIndex, TurboQuantIndex};

const DIM: usize = 64;

fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut uniform = || {
        let raw = (next() >> 40) as u32 | 1;
        raw as f32 / (1u32 << 24) as f32
    };
    let two_pi = 2.0_f32 * std::f32::consts::PI;
    let mut data = vec![0.0f32; n * dim];
    let mut i = 0;
    while i < data.len() {
        let u1 = uniform().max(1e-7);
        let u2 = uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = two_pi * u2;
        data[i] = r * theta.cos();
        if i + 1 < data.len() {
            data[i + 1] = r * theta.sin();
        }
        i += 2;
    }
    for row in 0..n {
        let row_slice = &mut data[row * dim..(row + 1) * dim];
        let norm: f32 = row_slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for x in row_slice.iter_mut() {
                *x *= inv;
            }
        }
    }
    data
}

// ---- TurboQuantIndex ----

#[test]
fn new_lazy_starts_with_no_dim() {
    let idx = TurboQuantIndex::new_lazy(4);
    assert_eq!(idx.dim_opt(), None);
    assert_eq!(idx.dim(), 0, "dim() returns 0 as sentinel");
    assert_eq!(idx.len(), 0);
    assert_eq!(idx.bit_width(), 4);
}

#[test]
fn add_2d_locks_dim_on_first_call() {
    let mut idx = TurboQuantIndex::new_lazy(4);
    let data = unit_vectors(3, DIM, 0xA00D_0001);
    idx.add_2d(&data, DIM);
    assert_eq!(idx.dim_opt(), Some(DIM));
    assert_eq!(idx.len(), 3);
}

#[test]
fn add_2d_subsequent_calls_must_match_dim() {
    let mut idx = TurboQuantIndex::new_lazy(4);
    let data1 = unit_vectors(2, DIM, 0xA00D_0002);
    idx.add_2d(&data1, DIM);
    let data2 = unit_vectors(2, DIM, 0xA00D_0003);
    idx.add_2d(&data2, DIM);
    assert_eq!(idx.len(), 4);
}

#[test]
#[should_panic(expected = "dim mismatch")]
fn add_2d_panics_on_dim_change() {
    let mut idx = TurboQuantIndex::new_lazy(4);
    let data = unit_vectors(1, DIM, 0xA00D_0004);
    idx.add_2d(&data, DIM);
    let wrong = unit_vectors(1, DIM * 2, 0xA00D_0005);
    idx.add_2d(&wrong, DIM * 2);
}

#[test]
#[should_panic(expected = "dim is not set")]
fn plain_add_panics_on_lazy_uncommitted() {
    let mut idx = TurboQuantIndex::new_lazy(4);
    let data = unit_vectors(1, DIM, 0xA00D_0006);
    idx.add(&data);
}

#[test]
fn search_on_lazy_uncommitted_returns_empty() {
    let idx = TurboQuantIndex::new_lazy(4);
    let queries = unit_vectors(2, DIM, 0xA00D_0007);
    let res = idx.search(&queries, 5);
    assert_eq!(res.scores.len(), 0);
    assert_eq!(res.indices.len(), 0);
    assert_eq!(res.k, 0);
}

#[test]
fn prepare_on_lazy_uncommitted_is_noop() {
    let idx = TurboQuantIndex::new_lazy(4);
    idx.prepare(); // should not panic
}

#[test]
fn write_load_round_trip_lazy_uncommitted() {
    let tmp = std::env::temp_dir().join("turbovec_lazy_uncommitted.tv");
    {
        let idx = TurboQuantIndex::new_lazy(4);
        idx.write(&tmp).unwrap();
    }
    let loaded = TurboQuantIndex::load(&tmp).unwrap();
    assert_eq!(loaded.dim_opt(), None);
    assert_eq!(loaded.len(), 0);
    assert_eq!(loaded.bit_width(), 4);
    fs::remove_file(&tmp).ok();
}

#[test]
fn write_load_round_trip_eager_index_still_works() {
    // Regression: the dim=0 sentinel logic must not affect normal indexes.
    let tmp = std::env::temp_dir().join("turbovec_lazy_eager.tv");
    {
        let mut idx = TurboQuantIndex::new(DIM, 4);
        idx.add(&unit_vectors(4, DIM, 0xA00D_0008));
        idx.write(&tmp).unwrap();
    }
    let loaded = TurboQuantIndex::load(&tmp).unwrap();
    assert_eq!(loaded.dim_opt(), Some(DIM));
    assert_eq!(loaded.len(), 4);
    fs::remove_file(&tmp).ok();
}

#[test]
fn write_load_round_trip_lazy_after_committed_add() {
    let tmp = std::env::temp_dir().join("turbovec_lazy_committed.tv");
    {
        let mut idx = TurboQuantIndex::new_lazy(2);
        idx.add_2d(&unit_vectors(3, DIM, 0xA00D_0009), DIM);
        idx.write(&tmp).unwrap();
    }
    let loaded = TurboQuantIndex::load(&tmp).unwrap();
    assert_eq!(loaded.dim_opt(), Some(DIM));
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded.bit_width(), 2);
    fs::remove_file(&tmp).ok();
}

// ---- IdMapIndex ----

#[test]
fn id_map_new_lazy_starts_with_no_dim() {
    let idx = IdMapIndex::new_lazy(4);
    assert_eq!(idx.dim_opt(), None);
    assert_eq!(idx.dim(), 0);
    assert_eq!(idx.len(), 0);
}

#[test]
fn id_map_add_with_ids_2d_locks_dim() {
    let mut idx = IdMapIndex::new_lazy(4);
    let data = unit_vectors(3, DIM, 0xA00D_0010);
    let ids: Vec<u64> = vec![10, 20, 30];
    idx.add_with_ids_2d(&data, DIM, &ids);
    assert_eq!(idx.dim_opt(), Some(DIM));
    assert_eq!(idx.len(), 3);
    assert!(idx.contains(20));
}

#[test]
#[should_panic(expected = "dim is not set")]
fn id_map_plain_add_with_ids_panics_on_lazy_uncommitted() {
    let mut idx = IdMapIndex::new_lazy(4);
    let data = unit_vectors(1, DIM, 0xA00D_0011);
    idx.add_with_ids(&data, &[42]);
}

#[test]
fn id_map_search_on_lazy_uncommitted_returns_empty() {
    let idx = IdMapIndex::new_lazy(4);
    let queries = unit_vectors(1, DIM, 0xA00D_0012);
    let (scores, ids) = idx.search(&queries, 5);
    assert!(scores.is_empty());
    assert!(ids.is_empty());
}

#[test]
fn id_map_write_load_round_trip_lazy_uncommitted() {
    let tmp = std::env::temp_dir().join("turbovec_idmap_lazy_uncommitted.tvim");
    {
        let idx = IdMapIndex::new_lazy(2);
        idx.write(&tmp).unwrap();
    }
    let loaded = IdMapIndex::load(&tmp).unwrap();
    assert_eq!(loaded.dim_opt(), None);
    assert_eq!(loaded.len(), 0);
    assert_eq!(loaded.bit_width(), 2);
    fs::remove_file(&tmp).ok();
}

#[test]
fn id_map_write_load_round_trip_lazy_after_committed_add() {
    let tmp = std::env::temp_dir().join("turbovec_idmap_lazy_committed.tvim");
    let ids: Vec<u64> = vec![100, 200, 300];
    {
        let mut idx = IdMapIndex::new_lazy(4);
        idx.add_with_ids_2d(&unit_vectors(3, DIM, 0xA00D_0013), DIM, &ids);
        idx.write(&tmp).unwrap();
    }
    let loaded = IdMapIndex::load(&tmp).unwrap();
    assert_eq!(loaded.dim_opt(), Some(DIM));
    assert_eq!(loaded.len(), 3);
    for &id in &ids {
        assert!(loaded.contains(id));
    }
    fs::remove_file(&tmp).ok();
}
