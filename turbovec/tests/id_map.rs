//! Correctness tests for `IdMapIndex` — the stable-id wrapper.
//!
//! Invariants exercised:
//!   - `add_with_ids` panics on bad input (length mismatch, duplicate id).
//!   - `remove` returns true/false and keeps `len` consistent.
//!   - After `remove`, search doesn't return the removed id, and every
//!     remaining id still self-queries to itself.
//!   - Remove then re-add with the same id works.
//!   - Internal `slot_to_id` / `id_to_slot` tables stay consistent after
//!     a swap-and-pop (verified indirectly via search correctness).

extern crate blas_src;

use turbovec::IdMapIndex;

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
fn add_with_ids_updates_len_and_contains() {
    let dim = 128;
    let data = gaussian_normalized(5, dim, 0xA11D_0000);
    let mut idx = IdMapIndex::new(dim, 4);
    idx.add_with_ids(&data, &[100, 200, 300, 400, 500]);

    assert_eq!(idx.len(), 5);
    assert!(idx.contains(300));
    assert!(!idx.contains(999));
}

#[test]
fn search_returns_ids_not_slots() {
    let dim = 256;
    let data = gaussian_normalized(10, dim, 0xA11D_0001);
    let mut idx = IdMapIndex::new(dim, 4);
    let ids: Vec<u64> = (1_000_000..1_000_010).collect();
    idx.add_with_ids(&data, &ids);

    // Self-query each vector: expect the matching external id as top-1.
    for (i, &expected_id) in ids.iter().enumerate() {
        let q = &data[i * dim..(i + 1) * dim];
        let (_, got_ids) = idx.search(q, 1);
        assert_eq!(got_ids[0], expected_id);
    }
}

#[test]
fn remove_returns_false_for_missing_id() {
    let dim = 128;
    let data = gaussian_normalized(3, dim, 0xA11D_0002);
    let mut idx = IdMapIndex::new(dim, 4);
    idx.add_with_ids(&data, &[1, 2, 3]);

    assert!(!idx.remove(999));
    assert_eq!(idx.len(), 3);
}

#[test]
fn remove_existing_id_shrinks_and_hides_it() {
    let dim = 256;
    let data = gaussian_normalized(10, dim, 0xA11D_0003);
    let mut idx = IdMapIndex::new(dim, 4);
    let ids: Vec<u64> = (0..10).map(|i| i as u64 * 7 + 11).collect();
    idx.add_with_ids(&data, &ids);

    // Remove the third vector (id = 25, at slot 2).
    let target_id = ids[2];
    assert!(idx.remove(target_id));
    assert_eq!(idx.len(), 9);
    assert!(!idx.contains(target_id));

    // Its own vector should no longer be returned as a top-1 under its id.
    let q = &data[2 * dim..3 * dim];
    let (_, got_ids) = idx.search(q, 9);
    assert!(!got_ids.contains(&target_id));
}

#[test]
fn remaining_ids_still_self_query_after_mixed_removes() {
    let dim = 384;
    let data = gaussian_normalized(20, dim, 0xA11D_0004);
    let mut idx = IdMapIndex::new(dim, 4);
    let ids: Vec<u64> = (0..20).map(|i| i as u64 * 100 + 5).collect();
    idx.add_with_ids(&data, &ids);

    // Remove a few ids in different orders — some will trigger
    // swap-and-pop, some will be the last vector (no swap).
    idx.remove(ids[7]);   // middle
    idx.remove(ids[19]);  // last
    idx.remove(ids[0]);   // first

    assert_eq!(idx.len(), 17);
    assert!(!idx.contains(ids[7]));
    assert!(!idx.contains(ids[19]));
    assert!(!idx.contains(ids[0]));

    // Every surviving id still maps back to its own vector.
    for (i, &id) in ids.iter().enumerate() {
        if i == 0 || i == 7 || i == 19 {
            continue;
        }
        let q = &data[i * dim..(i + 1) * dim];
        let (_, got_ids) = idx.search(q, 1);
        assert_eq!(
            got_ids[0], id,
            "id {id} (row {i}) no longer self-queries correctly after remove",
        );
    }
}

#[test]
fn remove_then_re_add_same_id_is_allowed() {
    let dim = 128;
    let data = gaussian_normalized(5, dim, 0xA11D_0005);
    let mut idx = IdMapIndex::new(dim, 4);
    idx.add_with_ids(&data, &[1, 2, 3, 4, 5]);

    assert!(idx.remove(3));
    assert!(!idx.contains(3));

    // Re-add a new vector with id 3.
    let new_vec = gaussian_normalized(1, dim, 0xA11D_BEEF);
    idx.add_with_ids(&new_vec, &[3]);
    assert!(idx.contains(3));
    assert_eq!(idx.len(), 5);
}

#[test]
#[should_panic(expected = "already present")]
fn add_with_ids_rejects_duplicate_id() {
    let dim = 128;
    let data = gaussian_normalized(5, dim, 0xA11D_0006);
    let mut idx = IdMapIndex::new(dim, 4);
    idx.add_with_ids(&data[..2 * dim], &[1, 2]);
    // Same id "2" already present -> panic.
    idx.add_with_ids(&data[2 * dim..3 * dim], &[2]);
}

#[test]
#[should_panic(expected = "expected")]
fn add_with_ids_rejects_length_mismatch() {
    let dim = 128;
    let data = gaussian_normalized(5, dim, 0xA11D_0007);
    let mut idx = IdMapIndex::new(dim, 4);
    // 5 vectors, only 3 ids -> panic.
    idx.add_with_ids(&data, &[1, 2, 3]);
}

#[test]
fn write_and_load_round_trips() {
    let dim = 256;
    let data = gaussian_normalized(10, dim, 0xA11D_0100);
    let ids: Vec<u64> = (2000..2010).collect();

    let mut idx = IdMapIndex::new(dim, 4);
    idx.add_with_ids(&data, &ids);

    // Delete a few to exercise non-identity slot_to_id mapping.
    idx.remove(2003);
    idx.remove(2007);

    let tmp = std::env::temp_dir().join(format!("turbovec_idmap_{}.tvim", std::process::id()));
    idx.write(&tmp).expect("write failed");

    let restored = IdMapIndex::load(&tmp).expect("load failed");
    assert_eq!(restored.len(), 8);
    assert!(restored.contains(2000));
    assert!(!restored.contains(2003));
    assert!(!restored.contains(2007));

    // Every surviving id should still self-query to itself on the
    // restored index (exercising packed_codes + norms + slot_to_id
    // all round-trip correctly).
    for (i, &id) in ids.iter().enumerate() {
        if id == 2003 || id == 2007 {
            continue;
        }
        let q = &data[i * dim..(i + 1) * dim];
        let (_, got_ids) = restored.search(q, 1);
        assert_eq!(got_ids[0], id, "id {id} failed to self-query after reload");
    }

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn load_rejects_wrong_magic() {
    let tmp = std::env::temp_dir().join(format!(
        "turbovec_idmap_badmagic_{}.tvim",
        std::process::id()
    ));
    // Write a file that starts with the `.tv` format instead of `TVIM`.
    let dim = 64;
    let data = gaussian_normalized(2, dim, 0xA11D_0101);
    let mut inner = IdMapIndex::new(dim, 4);
    inner.add_with_ids(&data, &[1, 2]);
    // Use the inner TurboQuantIndex's write to produce a .tv file.
    // We can't do that directly since inner is private; simulate with
    // arbitrary bytes of the right shape.
    std::fs::write(&tmp, b"XXXX\x01").expect("write junk");
    let res = IdMapIndex::load(&tmp);
    assert!(res.is_err(), "load should reject file without TVIM magic");
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn empty_index_round_trip() {
    let dim = 128;
    let idx = IdMapIndex::new(dim, 4);

    let tmp = std::env::temp_dir().join(format!(
        "turbovec_idmap_empty_{}.tvim",
        std::process::id()
    ));
    idx.write(&tmp).expect("write failed");

    let restored = IdMapIndex::load(&tmp).expect("load failed");
    assert_eq!(restored.len(), 0);
    assert_eq!(restored.dim(), dim);
    assert_eq!(restored.bit_width(), 4);
    std::fs::remove_file(&tmp).ok();
}
