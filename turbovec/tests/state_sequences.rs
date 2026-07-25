//! Tests for operation sequences across the public API of
//! `TurboQuantIndex` and `IdMapIndex`.
//!
//! Most existing tests exercise operations in isolation; bugs that live
//! in the transition between two operations (cache invalidation, slot
//! reuse, persistence round-trip preserving post-mutation state) can
//! slip past those. The audit that surfaced the LlamaIndex intra-batch
//! `add()` corruption pointed at the same risk in the Rust core; these
//! tests pin the operation sequences most likely to harbour it.

use turbovec::{IdMapIndex, TurboQuantIndex};

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
    for row in data.chunks_mut(dim) {
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
fn second_add_after_search_lets_new_vectors_be_found() {
    // `add -> search -> add -> search`: the second add must invalidate
    // the blocked cache populated by the first search, so the new
    // vector is visible.
    //
    // The blocked layout packs vectors in blocks of 32 slots, padding
    // the final partial block. To make this test discriminate a stale
    // cache, the first add fills block 0 *exactly* (32 vectors) and the
    // second add lands at slot 32 — the first slot of a block the stale
    // cache does not have. A search against the stale single-block
    // layout can therefore never return slot 32, no matter how the
    // padding codes score. (An earlier 5+1 version of this test put the
    // new vector inside the stale block's padded slot range, where the
    // padding code could coincidentally win the self-query — it kept
    // passing with the invalidation in `add` disabled; see issue #191.)
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let first = unit_vectors(32, dim, 0x5001);
    idx.add(&first);
    // First search warms the blocked cache (exactly one full block).
    let _ = idx.search(&first[0..dim], 5);

    // Second add: distinctive vector at slot 32, beyond the warmed
    // cache's block coverage.
    let second = unit_vectors(1, dim, 0x5002);
    idx.add(&second);
    assert_eq!(idx.len(), 33);

    // Self-query the newly added vector — it must be top-1, proving
    // the second add invalidated the cache and rebuilt it including
    // the new block.
    let res = idx.search(&second, 1);
    assert_eq!(res.indices.len(), 1);
    assert_eq!(res.indices[0] as usize, 32, "new vector not findable after second add+search");
}

#[test]
fn add_swap_remove_add_then_self_query_finds_all_three_phases() {
    // Mixed shrink+grow sequence. Tests that swap_remove leaves enough
    // state intact for a subsequent add (no stale packed_codes length,
    // no stale n_vectors), and that self-query still works on every
    // surviving vector across all three phases.
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();

    let phase_a = unit_vectors(5, dim, 0x5003);
    idx.add(&phase_a);

    // Remove the middle vector. The last vector (idx 4) moves into slot 2.
    idx.swap_remove(2);
    assert_eq!(idx.len(), 4);

    // Add two more vectors.
    let phase_c = unit_vectors(2, dim, 0x5004);
    idx.add(&phase_c);
    assert_eq!(idx.len(), 6);

    // The two newest vectors must self-query to slots 4 and 5.
    let r0 = idx.search(&phase_c[0..dim], 1);
    assert_eq!(r0.indices[0] as usize, 4);
    let r1 = idx.search(&phase_c[dim..2 * dim], 1);
    assert_eq!(r1.indices[0] as usize, 5);
}

#[test]
fn swap_remove_after_load_produces_correct_search() {
    // `add -> write -> load -> swap_remove -> search`: a loaded index
    // has an empty blocked cache, but swap_remove still resets it. The
    // combined "load then mutate then search" path is untested.
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let data = unit_vectors(5, dim, 0x5005);
    idx.add(&data);

    let tmp = std::env::temp_dir().join(format!("turbovec_seq_load_swap_{}.tv", std::process::id()));
    idx.write(&tmp).unwrap();

    let mut loaded = TurboQuantIndex::load(&tmp).unwrap();
    std::fs::remove_file(&tmp).ok();

    // Remove slot 1 — slot 4 moves into slot 1.
    let moved = loaded.swap_remove(1);
    assert_eq!(moved, 4);
    assert_eq!(loaded.len(), 4);

    // Self-query the surviving 4 vectors via their now-correct slots.
    // Slot 0 still has vec 0; slot 1 has vec 4 (moved); slot 2,3 unchanged.
    let r0 = loaded.search(&data[0..dim], 1);
    assert_eq!(r0.indices[0] as usize, 0);
    let r4 = loaded.search(&data[4 * dim..5 * dim], 1);
    assert_eq!(r4.indices[0] as usize, 1, "moved vector should now be at slot 1");
}

#[test]
fn swap_remove_then_round_trip_matches_in_memory_search() {
    // `add -> swap_remove -> write -> load -> search` must match
    // `add -> swap_remove -> search`. Persistence captures the
    // post-removal packed_codes / scales, not the pre-removal state.
    // A regression that wrote stale tail bytes would diverge here.
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let data = unit_vectors(5, dim, 0x5006);
    idx.add(&data);
    idx.swap_remove(1);
    idx.swap_remove(2);
    let in_memory = idx.search(&data[0..dim], 3);

    let tmp = std::env::temp_dir().join(format!(
        "turbovec_seq_swap_roundtrip_{}.tv",
        std::process::id()
    ));
    idx.write(&tmp).unwrap();
    let loaded = TurboQuantIndex::load(&tmp).unwrap();
    std::fs::remove_file(&tmp).ok();

    let from_disk = loaded.search(&data[0..dim], 3);

    assert_eq!(in_memory.scores, from_disk.scores);
    assert_eq!(in_memory.indices, from_disk.indices);
    assert_eq!(in_memory.k, from_disk.k);
    assert_eq!(in_memory.nq, from_disk.nq);
}

#[test]
fn id_map_re_added_id_returns_new_vector_not_old() {
    // `add({id: 1, vec_a}) -> remove(1) -> add({id: 1, vec_b}) -> search(vec_b)`:
    // after re-adding the same id with a different vector, search for
    // the new vector must return id 1 and rank it top. The existing
    // `remove_then_re_add_same_id_is_allowed` only asserts `contains`.
    let dim = 128;
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let vec_a = unit_vectors(1, dim, 0x5007);
    idx.add_with_ids(&vec_a, &[42]).unwrap();
    assert!(idx.remove(42));

    let vec_b = unit_vectors(1, dim, 0x5008);
    idx.add_with_ids(&vec_b, &[42]).unwrap();

    let (_, ids) = idx.search(&vec_b, 1);
    assert_eq!(ids[0], 42, "self-query of re-added vector should return its id");
}

#[test]
fn prepare_then_add_invalidates_blocked_cache() {
    // `prepare -> add -> search`: prepare populates the blocked cache;
    // the subsequent add must still invalidate it (analogous to the
    // search -> add -> search path but with the cache pre-warmed via
    // prepare instead of an actual query).
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let first = unit_vectors(3, dim, 0x5009);
    idx.add(&first);
    idx.prepare();

    let second = unit_vectors(1, dim, 0x500A);
    idx.add(&second);

    let res = idx.search(&second, 1);
    assert_eq!(res.indices[0] as usize, 3, "new vector not findable after prepare+add");
}

#[test]
fn id_map_remove_last_then_add_keeps_slot_tables_consistent() {
    // `remove(last)` skips the swap branch (no `id_to_slot.insert` for
    // a moved id). Subsequent add+search must still produce correct
    // results — pinning that the no-swap branch left no stale
    // `slot_to_id` tail entry.
    let dim = 128;
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let data = unit_vectors(3, dim, 0x500B);
    idx.add_with_ids(&data, &[10, 20, 30]).unwrap();

    // Remove the LAST id; no swap occurs.
    assert!(idx.remove(30));
    assert_eq!(idx.len(), 2);

    // Add a fresh id — must land at slot 2 (the now-freed slot).
    let extra = unit_vectors(1, dim, 0x500C);
    idx.add_with_ids(&extra, &[40]).unwrap();

    // Self-query the new vector returns id 40.
    let (_, ids) = idx.search(&extra, 1);
    assert_eq!(ids[0], 40);

    // The two previously-existing ids still resolve.
    let (_, ids10) = idx.search(&data[0..dim], 1);
    assert_eq!(ids10[0], 10);
    let (_, ids20) = idx.search(&data[dim..2 * dim], 1);
    assert_eq!(ids20[0], 20);
}

#[test]
fn add_after_load_extends_index() {
    // `add -> write -> load -> add -> search`: a loaded index can be
    // extended via add and the new vectors join the search results.
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let first = unit_vectors(3, dim, 0x500D);
    idx.add(&first);

    let tmp = std::env::temp_dir().join(format!(
        "turbovec_seq_add_after_load_{}.tv",
        std::process::id()
    ));
    idx.write(&tmp).unwrap();
    let mut loaded = TurboQuantIndex::load(&tmp).unwrap();
    std::fs::remove_file(&tmp).ok();

    let second = unit_vectors(2, dim, 0x500E);
    loaded.add(&second);
    assert_eq!(loaded.len(), 5);

    // Self-query the newly added vector.
    let res = loaded.search(&second[0..dim], 1);
    assert_eq!(res.indices[0] as usize, 3, "new vector should be at slot 3 after load+add");
}

#[test]
fn prepare_then_swap_remove_invalidates_cache() {
    // Defensive: `prepare -> swap_remove -> search` must produce a
    // search that reflects the deletion.
    let dim = 128;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let data = unit_vectors(5, dim, 0x500F);
    idx.add(&data);
    idx.prepare();

    idx.swap_remove(1);  // slot 4 moves into slot 1
    assert_eq!(idx.len(), 4);

    // Self-query slot-1's old vector (data row 1) — must NOT return
    // slot 1 anymore (which now has data row 4).
    let res = idx.search(&data[dim..2 * dim], 1);
    assert_ne!(res.indices[0] as usize, 1, "deleted vector should not be retrievable after prepare+swap_remove");
}
