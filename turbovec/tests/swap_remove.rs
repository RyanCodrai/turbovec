//! Correctness of `TurboQuantIndex::swap_remove`.
//!
//! Invariants tested at the behaviour level (no private-state access):
//!   - `len()` decreases by one; return value is the position of the
//!     vector that moved into the deleted slot.
//!   - The deleted vector no longer appears as a self-query top-1.
//!   - Every non-deleted vector still self-queries to itself.
//!   - Deleting the last vector is a no-op swap (return == idx).
//!   - Out-of-bounds delete panics.
//!   - Cache invalidation: a search immediately after a delete reflects
//!     the new layout (no stale `OnceLock` blocked cache).

extern crate blas_src;

use turbovec::TurboQuantIndex;

fn gaussian_normalized(n: usize, dim: usize, seed: u64) -> Vec<f32> {
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
    for row_i in 0..n {
        let row = &mut data[row_i * dim..(row_i + 1) * dim];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    }
    data
}

#[test]
fn swap_remove_shrinks_length_and_returns_last_index() {
    let dim = 256;
    let data = gaussian_normalized(10, dim, 0xDE1E_7E00);
    let mut idx = TurboQuantIndex::new(dim, 4);
    idx.add(&data);
    assert_eq!(idx.len(), 10);

    let moved_from = idx.swap_remove(3);
    assert_eq!(moved_from, 9);
    assert_eq!(idx.len(), 9);
}

#[test]
fn swap_remove_last_is_no_swap() {
    let dim = 256;
    let data = gaussian_normalized(5, dim, 0xDE1E_7E01);
    let mut idx = TurboQuantIndex::new(dim, 4);
    idx.add(&data);

    // Removing the last element: moved_from == idx, no actual swap.
    let moved_from = idx.swap_remove(4);
    assert_eq!(moved_from, 4);
    assert_eq!(idx.len(), 4);
}

#[test]
fn search_after_swap_remove_reflects_new_layout() {
    // Cache-invalidation regression test: the OnceLock blocked cache
    // must be reset when swap_remove runs, otherwise search returns
    // scores against stale packed codes.
    let dim = 512;
    let n = 100;
    let data = gaussian_normalized(n, dim, 0xDE1E_7E02);

    let mut idx = TurboQuantIndex::new(dim, 4);
    idx.add(&data);

    // Prime the cache with a self-query.
    let q = &data[5 * dim..6 * dim];
    let res = idx.search(q, 1);
    assert_eq!(res.indices_for_query(0)[0], 5);

    // Delete vector 5. The vector at position (n-1)=99 moves into slot 5.
    let moved_from = idx.swap_remove(5);
    assert_eq!(moved_from, n - 1);
    assert_eq!(idx.len(), n - 1);

    // Self-query on the vector that USED to be at index 99: now expect
    // top-1 to come back as index 5 (the slot it moved into). This
    // requires the cache to have been rebuilt.
    let q_moved = &data[(n - 1) * dim..n * dim];
    let res = idx.search(q_moved, 1);
    assert_eq!(
        res.indices_for_query(0)[0],
        5,
        "cache invalidation failure: moved vector not found at its new slot"
    );
}

#[test]
fn deleted_vector_no_longer_returned() {
    let dim = 512;
    let n = 64;
    let data = gaussian_normalized(n, dim, 0xDE1E_7E03);

    let mut idx = TurboQuantIndex::new(dim, 4);
    idx.add(&data);

    // Remove vector 7. Query with it: top-1 must NOT be 7 (that slot
    // now holds what was vector 63).
    idx.swap_remove(7);
    let q = &data[7 * dim..8 * dim];
    let res = idx.search(q, idx.len());
    let indices = res.indices_for_query(0);
    assert_eq!(indices.len(), idx.len());
    assert!(
        !indices.contains(&7i64)
            || indices[0] != 7i64,
        "deleted vector appears as top-1 after swap_remove"
    );
}

#[test]
fn remaining_vectors_still_self_query_correctly() {
    // After deleting a few, every remaining vector must still find
    // itself as top-1.
    let dim = 384;
    let n = 80;
    let data = gaussian_normalized(n, dim, 0xDE1E_7E04);

    let mut idx = TurboQuantIndex::new(dim, 4);
    idx.add(&data);

    // Delete a handful of non-adjacent positions.
    // After each delete, len shrinks and the last slot's vector moves
    // into the deleted position. Track which original index lives
    // at each current slot.
    let mut live_at_slot: Vec<usize> = (0..n).collect();

    for &to_delete in &[10usize, 5, 40, 0] {
        let last = live_at_slot.len() - 1;
        let _moved = idx.swap_remove(to_delete);
        // Mirror the index's semantics in our tracker.
        live_at_slot.swap(to_delete, last);
        live_at_slot.pop();
    }

    // Now every slot should self-query back to its current slot.
    for (slot, &orig) in live_at_slot.iter().enumerate() {
        let q = &data[orig * dim..(orig + 1) * dim];
        let res = idx.search(q, 1);
        assert_eq!(
            res.indices_for_query(0)[0],
            slot as i64,
            "vector originally at {orig} (now slot {slot}) didn't self-query correctly"
        );
    }
}

#[test]
#[should_panic(expected = "out of bounds")]
fn swap_remove_out_of_bounds_panics() {
    let dim = 128;
    let data = gaussian_normalized(3, dim, 0xDE1E_7E05);
    let mut idx = TurboQuantIndex::new(dim, 4);
    idx.add(&data);
    idx.swap_remove(3); // valid range is 0..3
}
