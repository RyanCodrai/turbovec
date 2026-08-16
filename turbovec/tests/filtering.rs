//! Correctness tests for slot-mask filtering on `TurboQuantIndex` and the
//! allowlist wrapper on `IdMapIndex`.
//!
//! Invariants exercised:
//!   - Masked search returns the same top-k as an unmasked search filtered
//!     post-hoc to the same allowed set (kernel parity).
//!   - `mask = None` and `mask = Some(all_true)` produce identical results.
//!   - Effective k shrinks to `n_allowed` when the mask is more selective.
//!   - Length / emptiness / unknown-id error paths panic.
//!   - `IdMapIndex.search_with_allowlist` returns only ids in the allowlist
//!     and never returns slot indices outside it.

use turbovec::{IdMapIndex, SearchError, TurboQuantIndex};

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

fn build_index(n: usize, dim: usize, seed: u64) -> TurboQuantIndex {
    let data = gaussian_normalized(n, dim, seed);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&data);
    idx
}

/// Reference top-k under a mask: score everything, filter to allowed slots,
/// take top-k by score.
fn reference_topk(
    idx: &TurboQuantIndex,
    query: &[f32],
    mask: &[bool],
    k: usize,
) -> (Vec<f32>, Vec<i64>) {
    let n = mask.len();
    let res = idx.search(query, n);
    let scores = &res.scores[..res.k];
    let indices = &res.indices[..res.k];
    let mut filtered: Vec<(f32, i64)> = scores
        .iter()
        .zip(indices.iter())
        .filter(|(_, &slot)| mask[slot as usize])
        .map(|(&s, &i)| (s, i))
        .collect();
    filtered.truncate(k);
    (
        filtered.iter().map(|p| p.0).collect(),
        filtered.iter().map(|p| p.1).collect(),
    )
}

#[test]
fn mask_matches_post_hoc_filter() {
    let dim = 128;
    let n = 256;
    let idx = build_index(n, dim, 0xF11D_0001);
    let query = gaussian_normalized(1, dim, 0xF11D_0002);

    // Allow every other slot.
    let mut mask = vec![false; n];
    for i in 0..n {
        if i % 2 == 0 {
            mask[i] = true;
        }
    }

    let masked = idx.search_with_mask(&query, 10, Some(&mask));
    let (ref_scores, ref_indices) = reference_topk(&idx, &query, &mask, 10);

    assert_eq!(masked.k, 10, "expected 10 results, got {}", masked.k);
    assert_eq!(&masked.scores[..], &ref_scores[..], "score mismatch");
    assert_eq!(&masked.indices[..], &ref_indices[..], "index mismatch");
    for &slot in &masked.indices {
        assert!(
            mask[slot as usize],
            "kernel returned disallowed slot {}",
            slot
        );
    }
}

#[test]
fn mask_none_equals_mask_all_true() {
    let dim = 64;
    let n = 200;
    let idx = build_index(n, dim, 0xF11D_0003);
    let query = gaussian_normalized(1, dim, 0xF11D_0004);

    let unfiltered = idx.search(&query, 20);
    let all_true = vec![true; n];
    let filtered = idx.search_with_mask(&query, 20, Some(&all_true));

    assert_eq!(unfiltered.k, filtered.k);
    assert_eq!(&unfiltered.scores[..], &filtered.scores[..]);
    assert_eq!(&unfiltered.indices[..], &filtered.indices[..]);
}

#[test]
fn effective_k_shrinks_when_allowlist_smaller_than_k() {
    let dim = 64;
    let n = 100;
    let idx = build_index(n, dim, 0xF11D_0005);
    let query = gaussian_normalized(1, dim, 0xF11D_0006);

    let mut mask = vec![false; n];
    mask[3] = true;
    mask[42] = true;
    mask[77] = true;

    let res = idx.search_with_mask(&query, 10, Some(&mask));
    assert_eq!(res.k, 3, "effective k should be 3 (popcount of mask)");
    assert_eq!(res.scores.len(), 3);
    assert_eq!(res.indices.len(), 3);
    for &slot in &res.indices {
        assert!(mask[slot as usize]);
    }
}

#[test]
fn all_false_mask_returns_empty_results() {
    let dim = 64;
    let n = 64;
    let idx = build_index(n, dim, 0xF11D_0007);
    let query = gaussian_normalized(1, dim, 0xF11D_0008);

    let mask = vec![false; n];
    let res = idx.search_with_mask(&query, 5, Some(&mask));
    assert_eq!(res.k, 0);
    assert!(res.scores.is_empty());
    assert!(res.indices.is_empty());
}

#[test]
#[should_panic(expected = "mask length")]
fn mask_length_mismatch_panics() {
    let dim = 64;
    let n = 50;
    let idx = build_index(n, dim, 0xF11D_0009);
    let query = gaussian_normalized(1, dim, 0xF11D_000A);

    let wrong_len_mask = vec![true; 10];
    let _ = idx.search_with_mask(&query, 5, Some(&wrong_len_mask));
}

#[test]
fn multi_query_batch_respects_mask() {
    // The x86 kernels batch queries in groups of 4. Make sure the mask is
    // honoured for every query in a multi-query batch, including the
    // non-power-of-4 tail.
    let dim = 128;
    let n = 256;
    let nq = 7;
    let idx = build_index(n, dim, 0xF11D_000B);
    let queries = gaussian_normalized(nq, dim, 0xF11D_000C);

    let mut mask = vec![false; n];
    for i in 0..n {
        if i % 3 == 0 {
            mask[i] = true;
        }
    }

    let res = idx.search_with_mask(&queries, 8, Some(&mask));
    assert_eq!(res.nq, nq);
    assert_eq!(res.k, 8);
    assert_eq!(res.indices.len(), nq * 8);

    for qi in 0..nq {
        let row_start = qi * res.k;
        let row = &res.indices[row_start..row_start + res.k];
        let scores_row = &res.scores[row_start..row_start + res.k];
        // Every returned slot must be in the allowed set.
        for &slot in row {
            assert!(
                mask[slot as usize],
                "query {qi}: kernel returned disallowed slot {slot}"
            );
        }
        // Scores are returned in descending order.
        for w in scores_row.windows(2) {
            assert!(w[0] >= w[1], "query {qi}: scores not descending: {scores_row:?}");
        }
        // The fused 4-query NEON kernel and the single-query tail kernel
        // produce scores that match within float rounding (~1e-4 relative).
        // The reference uses the single-query path; compare scores within
        // tolerance and indices exactly (assuming no tie-flips at this dim).
        let query_row = &queries[qi * dim..(qi + 1) * dim];
        let (ref_scores, ref_indices) = reference_topk(&idx, query_row, &mask, res.k);
        assert_eq!(row, &ref_indices[..], "query {qi} index mismatch");
        for (a, b) in scores_row.iter().zip(ref_scores.iter()) {
            assert!(
                (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0),
                "query {qi}: score {a} vs reference {b}",
            );
        }
    }
}

// ------------------- IdMapIndex allowlist -------------------

#[test]
fn allowlist_returns_only_listed_ids() {
    let dim = 128;
    let n = 100;
    let data = gaussian_normalized(n, dim, 0xF11D_1001);
    let ids: Vec<u64> = (0..n as u64).map(|i| 1000 + i).collect();
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    let query = gaussian_normalized(1, dim, 0xF11D_1002);
    let allowed: Vec<u64> = vec![1003, 1010, 1042, 1077, 1099];
    let (scores, returned_ids) = idx.search_with_allowlist(&query, 10, Some(&allowed)).unwrap();

    assert_eq!(scores.len(), allowed.len(), "effective k = allowlist len");
    assert_eq!(returned_ids.len(), allowed.len());
    for id in &returned_ids {
        assert!(
            allowed.contains(id),
            "kernel returned id {} not in allowlist",
            id
        );
    }
}

#[test]
fn allowlist_none_equivalent_to_plain_search() {
    let dim = 64;
    let n = 80;
    let data = gaussian_normalized(n, dim, 0xF11D_1003);
    let ids: Vec<u64> = (0..n as u64).map(|i| 7000 + i * 13).collect();
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    let query = gaussian_normalized(1, dim, 0xF11D_1004);
    let (s1, i1) = idx.search(&query, 5);
    let (s2, i2) = idx.search_with_allowlist(&query, 5, None).unwrap();
    assert_eq!(s1, s2);
    assert_eq!(i1, i2);
}

#[test]
fn empty_allowlist_is_an_error_not_a_panic() {
    let dim = 64;
    let data = gaussian_normalized(10, dim, 0xF11D_1005);
    let ids: Vec<u64> = (0..10).collect();
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    let query = gaussian_normalized(1, dim, 0xF11D_1006);
    assert_eq!(
        idx.search_with_allowlist(&query, 3, Some(&[])),
        Err(SearchError::AllowlistEmpty),
    );
}

#[test]
fn unknown_id_in_allowlist_is_an_error_not_a_panic() {
    let dim = 64;
    let data = gaussian_normalized(10, dim, 0xF11D_1007);
    let ids: Vec<u64> = (0..10).collect();
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    let query = gaussian_normalized(1, dim, 0xF11D_1008);
    assert_eq!(
        idx.search_with_allowlist(&query, 3, Some(&[5, 999])),
        Err(SearchError::UnknownId(999)),
    );
    // The index is still usable afterwards — the error is recoverable.
    assert_eq!(idx.search(&query, 3).1.len(), 3);
}

#[test]
fn block_skip_at_one_percent_selectivity_matches_post_filter() {
    // Block-level early exit (search.rs::block_has_allowed) skips entire
    // 32-vector SIMD blocks where no slot is allowed. At ~1% selectivity
    // the kernel skips the vast majority of blocks; this test confirms
    // the top-k returned by the masked path equals the top-k returned
    // by a full dense scan + post-hoc filter.
    let dim = 128;
    let n = 4096;  // 128 blocks of 32 — gives the skip path plenty to skip
    let data = gaussian_normalized(n, dim, 0xB10C_5417);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&data);
    idx.prepare();

    // Pick 1% of slots, scattered across the full range to mix in-block
    // hits and pure-zero blocks.
    let n_allowed = n / 100;
    let allowed_slots: Vec<usize> = (0..n_allowed).map(|i| (i * 97) % n).collect();
    let mut mask = vec![false; n];
    for &s in &allowed_slots {
        mask[s] = true;
    }

    let query = gaussian_normalized(1, dim, 0xB10C_5418);
    let k = 8;

    let masked = idx.search_with_mask(&query, k, Some(&mask));
    let dense = idx.search(&query, n);

    // Post-filter the dense top-k to the allowed set, then compare to masked.
    let dense_ids = dense.indices_for_query(0);
    let dense_scores = dense.scores_for_query(0);
    let mut expected: Vec<(f32, i64)> = dense_ids
        .iter()
        .zip(dense_scores.iter())
        .filter(|(&i, _)| mask[i as usize])
        .map(|(&i, &s)| (s, i))
        .collect();
    expected.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    expected.truncate(k);

    let masked_ids = masked.indices_for_query(0);
    let masked_scores = masked.scores_for_query(0);
    assert_eq!(masked_ids.len(), expected.len(), "result count");
    for (i, (exp_score, exp_id)) in expected.iter().enumerate() {
        assert_eq!(
            masked_ids[i], *exp_id,
            "rank {}: id mismatch (got {}, want {})",
            i, masked_ids[i], exp_id
        );
        assert!(
            (masked_scores[i] - exp_score).abs() < 1e-4,
            "rank {}: score mismatch (got {}, want {})",
            i, masked_scores[i], exp_score
        );
    }
}

#[test]
fn block_skip_at_extreme_selectivity_returns_only_allowed() {
    // 1 in 1000 allowed: the skip path will short-circuit virtually
    // every block. Verifies the returned ids are all in the allowlist
    // even when the mask is sparser than typical hybrid retrieval.
    let dim = 64;
    let n = 8192;
    let data = gaussian_normalized(n, dim, 0xB10C_5419);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&data);
    idx.prepare();

    let allowed_slots: Vec<usize> = vec![17, 533, 1024, 2500, 6700, 8000];
    let mut mask = vec![false; n];
    for &s in &allowed_slots {
        mask[s] = true;
    }

    let query = gaussian_normalized(1, dim, 0xB10C_541A);
    let results = idx.search_with_mask(&query, 4, Some(&mask));
    let ids = results.indices_for_query(0);

    assert_eq!(ids.len(), 4);
    for &id in ids {
        assert!(
            allowed_slots.contains(&(id as usize)),
            "returned id {} not in allowlist",
            id
        );
    }
}

#[test]
#[cfg(feature = "mask-skip-counter")]
fn block_skip_path_actually_fires_under_selective_mask() {
    // Direct activation test for the block-level early-exit path. The
    // correctness tests above would still pass if the block-skip guard
    // were deleted (the post-filter at heap-insert catches the same
    // vectors). This test reads `blocks_skipped_by_mask` before and
    // after a selective search and asserts the delta is non-zero,
    // proving the skip path executed.
    //
    // Robust to concurrent test interference: cargo test runs tests in
    // parallel and other selective-mask tests will also increment the
    // counter, but they can only push the delta UP, never down.
    use turbovec::search::{blocks_skipped_by_mask, reset_blocks_skipped_by_mask};

    let dim = 64;
    let n = 4096; // 128 blocks of 32
    let data = gaussian_normalized(n, dim, 0xC0DE_5417);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&data);
    idx.prepare();

    // Clustered allowlist: only the last 40 slots = ~2 blocks at the
    // tail. The remaining ~126 blocks have zero allowed slots and must
    // be short-circuited.
    let mut mask = vec![false; n];
    for slot in (n - 40)..n {
        mask[slot] = true;
    }

    let query = gaussian_normalized(1, dim, 0xC0DE_5418);

    reset_blocks_skipped_by_mask();
    // `None` means this build compiled the counter out; the test binary
    // enables `mask-skip-counter` through the self dev-dependency, so a
    // `None` here means that wiring broke rather than that no block was
    // skipped — which is exactly the distinction the Option exists for.
    let before = blocks_skipped_by_mask().expect("test build must enable mask-skip-counter");
    let _ = idx.search_with_mask(&query, 8, Some(&mask));
    let after = blocks_skipped_by_mask().expect("test build must enable mask-skip-counter");
    let delta = after - before;

    // Lower bound: at least one block must have been skipped, otherwise
    // the kernel never took the early-exit path during this search.
    assert!(
        delta > 0,
        "block-skip counter did not increment during selective search; \
         the early-exit path appears inactive (before={before}, after={after})"
    );

    // Tighter bound: with ~126 empty blocks and ~2 occupied, we expect
    // most-but-not-all blocks to be skipped. Allow generous slack since
    // BLOCK alignment may not match the allowlist boundary exactly.
    assert!(
        delta >= 50,
        "block-skip fired only {delta} times for a search where ~126 of \
         128 blocks have no allowed slots; some kernel variants may be \
         missing the guard or misaligning the mask-word check"
    );
}

#[test]
fn block_skip_with_all_slots_allowed_matches_unmasked() {
    // Defensive: when every bit is set, the mask path must produce the
    // same top-k as the no-mask path. Catches any block-skip logic that
    // accidentally short-circuits a fully-allowed block.
    let dim = 64;
    let n = 1024;
    let data = gaussian_normalized(n, dim, 0xB10C_541B);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&data);
    idx.prepare();

    let mask = vec![true; n];
    let query = gaussian_normalized(1, dim, 0xB10C_541C);
    let k = 16;

    let with_mask = idx.search_with_mask(&query, k, Some(&mask));
    let no_mask = idx.search(&query, k);

    assert_eq!(with_mask.indices_for_query(0), no_mask.indices_for_query(0));
    assert_eq!(with_mask.scores_for_query(0), no_mask.scores_for_query(0));
}

#[test]
fn allowlist_survives_swap_remove() {
    // After remove(), slots shift but external ids are stable. An allowlist
    // built against ids should keep working without rebuilding.
    let dim = 64;
    let n = 30;
    let data = gaussian_normalized(n, dim, 0xF11D_1009);
    let ids: Vec<u64> = (0..n as u64).map(|i| 5000 + i).collect();
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    let allowed: Vec<u64> = vec![5005, 5015, 5020];
    let query = gaussian_normalized(1, dim, 0xF11D_100A);

    let _before = idx.search_with_allowlist(&query, 3, Some(&allowed)).unwrap();
    // Removing an id NOT in the allowlist; the allowlist should remain valid.
    assert!(idx.remove(5025));
    let after = idx.search_with_allowlist(&query, 3, Some(&allowed)).unwrap();
    assert_eq!(after.1.len(), 3);
    for id in &after.1 {
        assert!(allowed.contains(id));
    }
}

/// Index large enough (>= `SINGLE_QUERY_PARALLEL_MIN_BLOCKS` blocks of 32
/// vectors) that a single-query search takes the block-parallel path.
/// That path used to be gated on `mask.is_none()`, so every masked
/// search fell back to the serial per-query path; it now carries a
/// word-aligned slice of the bitmap per range. These tests pin that the
/// parallel masked results are identical to the post-hoc-filtered
/// reference, and identical at any thread count.
///
/// Derived from the constant rather than hard-coded so raising the
/// threshold (#336) cannot silently move these tests off the path they
/// exist to cover.
const BLOCK_PARALLEL_N: usize =
    (turbovec::search::SINGLE_QUERY_PARALLEL_MIN_BLOCKS + 44) * 32;

fn assert_masked_matches_reference(mask: &[bool], k: usize, seed: u64) {
    let dim = 64;
    let idx = build_index(BLOCK_PARALLEL_N, dim, seed);
    let query = gaussian_normalized(1, dim, seed ^ 0xFFFF);
    let masked = idx.search_with_mask(&query, k, Some(mask));
    let (ref_scores, ref_indices) = reference_topk(&idx, &query, mask, k);
    assert_eq!(&masked.scores[..masked.k], &ref_scores[..], "score mismatch");
    assert_eq!(&masked.indices[..masked.k], &ref_indices[..], "index mismatch");
    for &slot in &masked.indices[..masked.k] {
        assert!(mask[slot as usize], "returned disallowed slot {slot}");
    }
}

#[test]
fn block_parallel_masked_all_true_matches_unmasked() {
    let dim = 64;
    let idx = build_index(BLOCK_PARALLEL_N, dim, 0xF11D_2001);
    let query = gaussian_normalized(1, dim, 0xF11D_2002);
    let mask = vec![true; BLOCK_PARALLEL_N];
    let unmasked = idx.search(&query, 20);
    let masked = idx.search_with_mask(&query, 20, Some(&mask));
    assert_eq!(unmasked.k, masked.k);
    assert_eq!(unmasked.scores, masked.scores);
    assert_eq!(unmasked.indices, masked.indices);
}

#[test]
fn block_parallel_masked_sparse_matches_reference() {
    // Sparse enough that whole 32-vector blocks are skipped, exercising
    // the range-relative block_has_allowed path.
    let mut mask = vec![false; BLOCK_PARALLEL_N];
    for i in (0..BLOCK_PARALLEL_N).step_by(97) {
        mask[i] = true;
    }
    assert_masked_matches_reference(&mask, 10, 0xF11D_2003);
}

#[test]
fn block_parallel_masked_dense_matches_reference() {
    let mut mask = vec![false; BLOCK_PARALLEL_N];
    for (i, m) in mask.iter_mut().enumerate() {
        *m = i % 3 != 0;
    }
    assert_masked_matches_reference(&mask, 25, 0xF11D_2004);
}

#[test]
fn block_parallel_masked_tail_and_head_only() {
    // Allowed slots confined to the first and last block: every range in
    // between skips wholesale, and the surviving ranges must still report
    // absolute slot indices.
    let mut mask = vec![false; BLOCK_PARALLEL_N];
    for m in mask.iter_mut().take(32) {
        *m = true;
    }
    for m in mask.iter_mut().skip(BLOCK_PARALLEL_N - 32) {
        *m = true;
    }
    assert_masked_matches_reference(&mask, 8, 0xF11D_2005);
}

#[test]
fn block_parallel_masked_results_are_thread_count_invariant() {
    let dim = 64;
    let idx = build_index(BLOCK_PARALLEL_N, dim, 0xF11D_2006);
    let query = gaussian_normalized(1, dim, 0xF11D_2007);
    let mut mask = vec![false; BLOCK_PARALLEL_N];
    for (i, m) in mask.iter_mut().enumerate() {
        *m = i % 5 != 0;
    }
    let default_threads = idx.search_with_mask(&query, 16, Some(&mask));
    for threads in [1usize, 2, 3, 7] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool build");
        let got = pool.install(|| idx.search_with_mask(&query, 16, Some(&mask)));
        assert_eq!(got.k, default_threads.k, "k differs at {threads} threads");
        assert_eq!(
            got.scores, default_threads.scores,
            "scores differ at {threads} threads"
        );
        assert_eq!(
            got.indices, default_threads.indices,
            "indices differ at {threads} threads"
        );
    }
}

#[test]
fn block_parallel_mask_allows_fewer_than_k() {
    // n_allowed < k: effective_k must clamp and no disallowed slot may leak.
    let dim = 64;
    let idx = build_index(BLOCK_PARALLEL_N, dim, 0xF11D_2008);
    let query = gaussian_normalized(1, dim, 0xF11D_2009);
    let mut mask = vec![false; BLOCK_PARALLEL_N];
    for i in [3usize, 4000, 9500] {
        mask[i] = true;
    }
    let masked = idx.search_with_mask(&query, 10, Some(&mask));
    assert_eq!(masked.k, 3);
    for &slot in &masked.indices[..masked.k] {
        assert!(mask[slot as usize], "returned disallowed slot {slot}");
    }
}

/// #349: an empty query batch must not panic. The divisor is
/// `(n_threads * TILES_PER_THREAD).div_ceil(n_quads)` in `n_block_ranges`, fed by
/// `n_quads = nq.div_ceil(QBS)`, which is zero for `nq == 0`.
///
/// The filtered forms reach that division too, but only on aarch64: the
/// x86 dispatch marks a masked search serial, so `n_block_ranges`
/// returns before dividing, while the aarch64
/// dispatch passes `serial = false` for masked and unmasked alike. So
/// pinning the unmasked form alone would leave the masked one covered on
/// exactly one target.
///
/// Runs on an explicit multi-thread pool: `n_threads == 1` short-circuits
/// ahead of the division, which would make every assertion here vacuous
/// on a single-core runner.
#[test]
fn empty_query_batch_is_not_a_panic_at_any_index_size() {
    let dim = 64;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    for &n in &[16usize, 8160, 8161, 9000] {
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.add_2d(&gaussian_normalized(n, dim, 0x3490 + n as u64), dim).unwrap();
        idx.prepare();
        let res = pool.install(|| idx.search(&[], 5));
        assert_eq!(res.nq, 0, "n={n}: empty batch must report nq = 0");
        assert!(res.indices.is_empty(), "n={n}: empty batch returned indices");

        // Same batch through the mask path.
        let mask = vec![true; n];
        let masked = pool.install(|| idx.search_with_mask(&[], 5, Some(&mask)));
        assert_eq!(masked.nq, 0, "n={n}: masked empty batch must report nq = 0");
        assert!(masked.indices.is_empty(), "n={n}: masked empty batch returned indices");

        let mut ids = IdMapIndex::new(dim, 4).unwrap();
        let id_vals: Vec<u64> = (0..n as u64).collect();
        ids.add_with_ids_2d(&gaussian_normalized(n, dim, 0x3491 + n as u64), dim, &id_vals)
            .unwrap();
        ids.prepare();
        let (scores, out_ids) = pool.install(|| ids.search(&[], 5));
        assert!(scores.is_empty() && out_ids.is_empty(), "n={n}: id-map empty batch");

        let allow: Vec<u64> = id_vals.iter().copied().take(3).collect();
        let (scores, out_ids) =
            pool.install(|| ids.search_with_allowlist(&[], 5, Some(&allow)).unwrap());
        assert!(
            scores.is_empty() && out_ids.is_empty(),
            "n={n}: id-map empty batch with an allowlist"
        );
    }
}

/// The block-parallel single-query path skips a block whose SIMD max
/// cannot beat the running heap minimum (#493). The prune is only sound
/// if it never changes a result, so compare each `k` against the full
/// ranking of the same index — taken at `k == n`, where the heap does not
/// fill until the very end and the prune therefore almost never fires.
///
/// Ties are the interesting case: with an evict-largest-index tie-break,
/// a prune that fired one block too eagerly would silently keep the wrong
/// one of two equal scores, which a score-only comparison would miss.
#[test]
fn block_parallel_prune_preserves_results_at_every_k() {
    let dim = 64;
    for &seed in &[0xBEEF_0001u64, 0xBEEF_0002, 0xBEEF_0003] {
        let idx = build_index(BLOCK_PARALLEL_N, dim, seed);
        let all = vec![true; BLOCK_PARALLEL_N];
        for &k in &[1usize, 10, 64, 100] {
            let query = gaussian_normalized(1, dim, seed ^ 0xABCD);
            let got = idx.search(&query, k);
            let (want_scores, want_indices) = reference_topk(&idx, &query, &all, k);
            assert_eq!(
                got.scores[..got.k].iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                want_scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                "scores differ from the full ranking at k={k}, seed={seed:#x}"
            );
            assert_eq!(
                &got.indices[..got.k],
                &want_indices[..],
                "indices differ from the full ranking at k={k}, seed={seed:#x}"
            );
        }
    }
}
