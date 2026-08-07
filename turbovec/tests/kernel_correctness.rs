//! Correctness harness for the SIMD scoring kernels.
//!
//! These tests are written to survive a change in the `BLOCK` constant:
//! they probe `n_vectors` values that straddle both `BLOCK=32` and
//! `BLOCK=64` tail boundaries, so a layout or tail-handling bug in
//! `pack::repack` or any of the per-arch kernels surfaces as an
//! assertion failure rather than silent recall drift.
//!
//! The invariants exercised are behaviour-level (no access to private
//! state), which means the same file runs unchanged against the AVX2,
//! NEON and eventual AVX-512 paths.

use std::sync::Arc;
use std::thread;

use turbovec::TurboQuantIndex;

/// Seeded gaussian normalized vectors via Box–Muller.
///
/// Normalized gaussian vectors have IP concentrated around zero with
/// spread `~1/sqrt(dim)`, so for `dim >= 256` the self-IP of 1.0
/// dominates any off-diagonal pair even after 2–4 bit quantization,
/// which makes the self-query invariant below robust.
fn gaussian_normalized(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    // Simple xorshift64 seeded from the input — avoids pulling rand_chacha
    // as a dev-dep and keeps the test reproducible regardless of rand
    // version drift.
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut uniform = || {
        // 24-bit mantissa → float in (0, 1]
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

/// `n_vectors` values that straddle both BLOCK=32 and BLOCK=64
/// boundaries, plus some that force multi-block scans.
const TAIL_SIZES: &[usize] = &[
    32, 33, 63, 64, 65, 96, 127, 128, 129, 160, 191, 192, 193, 256, 257, 500,
];

#[test]
fn self_query_returns_self_top1_4bit() {
    let dim = 512;
    let bits = 4;

    for &n in TAIL_SIZES {
        let data = gaussian_normalized(n, dim, 0x5EED_0000 ^ n as u64);
        let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
        idx.add(&data);
        assert_eq!(idx.len(), n);

        let nq = n.min(8);
        let q = &data[..nq * dim];
        let res = idx.search(q, 1);

        for qi in 0..nq {
            let top = res.indices_for_query(qi)[0];
            assert_eq!(
                top, qi as i64,
                "4-bit self-match failed: n={} qi={} got={}",
                n, qi, top
            );
        }
    }
}

#[test]
fn self_query_returns_self_top3_2bit() {
    // 2-bit quantization is coarser — allow the self-match to live in
    // the top 3 rather than strictly top 1. dim=512 keeps off-diagonal
    // IPs concentrated so this still catches any real bug.
    let dim = 512;
    let bits = 2;

    for &n in TAIL_SIZES {
        let data = gaussian_normalized(n, dim, 0xC0FF_EE00 ^ n as u64);
        let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
        idx.add(&data);

        let nq = n.min(8);
        let q = &data[..nq * dim];
        let k = 3.min(n);
        let res = idx.search(q, k);

        for qi in 0..nq {
            let top: &[i64] = res.indices_for_query(qi);
            assert!(
                top.contains(&(qi as i64)),
                "2-bit self-match failed: n={} qi={} top{}={:?}",
                n,
                qi,
                k,
                top
            );
        }
    }
}

#[test]
fn search_scores_are_sorted_descending() {
    // The heap path should return results in descending-score order.
    // A block handling or heap-min tracking bug often shows up as an
    // unsorted or truncated result set.
    let dim = 256;
    for bits in [2usize, 3, 4] {
        for &n in &[64usize, 100, 128, 200, 256, 500] {
            let data = gaussian_normalized(n, dim, 0xA11CE ^ (n as u64) ^ (bits as u64));
            let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
            idx.add(&data);

            let q = &data[..4 * dim];
            let k = 10.min(n);
            let res = idx.search(q, k);

            for qi in 0..4 {
                let scores = res.scores_for_query(qi);
                // Every returned score must be finite: `n >= k` here, so
                // no NEG_INFINITY heap padding may leak into the results
                // (a heap under-fill or kernel-tail bug would surface as
                // a non-finite score). Mirrors the finiteness checks the
                // id_map tests already apply.
                assert!(
                    scores.iter().all(|s| s.is_finite()),
                    "non-finite score leaked: bits={} n={} qi={} scores={:?}",
                    bits,
                    n,
                    qi,
                    scores
                );
                // Strict descending order — no escape hatch for
                // non-finite entries, which the finiteness assertion
                // above has already ruled out.
                for w in scores.windows(2) {
                    assert!(
                        w[0] >= w[1],
                        "scores not sorted desc: bits={} n={} qi={} window={:?}",
                        bits,
                        n,
                        qi,
                        w
                    );
                }
            }
        }
    }
}

#[test]
fn search_k_exceeding_len_clamps_to_len() {
    // `search(k > n_vectors)` with `0 < n < k` on the unmasked path:
    // the effective k is `min(k, n)`, so with n=3 and k=10 each query
    // must get back exactly 3 results — distinct, in-range, finite —
    // and `SearchResults::k` must report the clamped value 3, not the
    // requested 10.
    let dim = 512;
    let n = 3;
    let k = 10;
    let nq = 2;
    let data = gaussian_normalized(n, dim, 0x0EFF_EC7); // "effect(ive k)"
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&data);
    assert_eq!(idx.len(), n);

    let queries = gaussian_normalized(nq, dim, 0x0EFF_EC8);
    let res = idx.search(&queries, k);

    assert_eq!(res.nq, nq);
    assert_eq!(res.k, n, "SearchResults::k must be clamped to len()");
    assert_eq!(res.indices.len(), nq * n);
    assert_eq!(res.scores.len(), nq * n);

    for qi in 0..nq {
        let indices = res.indices_for_query(qi);
        assert_eq!(indices.len(), n);
        // All slots in 0..n, and all distinct — with k > n every stored
        // vector appears exactly once, no duplicates or padding slots.
        let mut seen = vec![false; n];
        for &slot in indices {
            assert!(
                (0..n as i64).contains(&slot),
                "qi={} returned out-of-range slot {}",
                qi,
                slot
            );
            assert!(
                !std::mem::replace(&mut seen[slot as usize], true),
                "qi={} returned duplicate slot {}",
                qi,
                slot
            );
        }
        // Scores stay finite and sorted — no NEG_INFINITY heap padding
        // leaks even though the heap was sized for k=10 candidates.
        let scores = res.scores_for_query(qi);
        assert!(
            scores.iter().all(|s| s.is_finite()),
            "qi={} non-finite score in {:?}",
            qi,
            scores
        );
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "qi={} scores not sorted desc: {:?}", qi, scores);
        }
    }
}

#[test]
fn search_is_deterministic_for_same_query() {
    let dim = 256;
    let bits = 4;
    for &n in &[64usize, 65, 127, 128, 129, 500] {
        let data = gaussian_normalized(n, dim, 0xD0D0_D0D0 ^ n as u64);
        let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
        idx.add(&data);

        let q = &data[..3 * dim];
        let r1 = idx.search(q, 10.min(n));
        let r2 = idx.search(q, 10.min(n));
        assert_eq!(
            r1.indices, r2.indices,
            "non-deterministic indices at n={}",
            n
        );
        assert_eq!(
            r1.scores, r2.scores,
            "non-deterministic scores at n={}",
            n
        );
    }
}

#[test]
fn single_query_matches_batched_query() {
    // Running one query on its own must produce the same top-k as
    // running it as part of a batch. Regressions here usually mean the
    // multi-query kernel branch has a per-query state bug.
    let dim = 256;
    let bits = 4;
    let n = 500;
    let data = gaussian_normalized(n, dim, 0x1234_5678);
    let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
    idx.add(&data);

    let batch = &data[..5 * dim];
    let k = 10;
    let batched = idx.search(batch, k);

    for qi in 0..5 {
        let single_q = &batch[qi * dim..(qi + 1) * dim];
        let single = idx.search(single_q, k);
        assert_eq!(
            batched.indices_for_query(qi),
            single.indices_for_query(0),
            "single-query vs batched mismatch at qi={}",
            qi
        );
        // Scores are compared with tolerance because the query rotation
        // is a GEMM whose accumulation order depends on the batch shape:
        // `(nq, dim) @ (dim, dim)` uses a different blocked reduction
        // than `(1, dim) @ (dim, dim)`, producing differences in the
        // low bits even though the algorithm is identical.
        let bs = batched.scores_for_query(qi);
        let ss = single.scores_for_query(0);
        for (i, (&b, &s)) in bs.iter().zip(ss.iter()).enumerate() {
            let tol = 1e-5_f32.max(1e-5_f32 * b.abs());
            assert!(
                (b - s).abs() <= tol,
                "single-query vs batched score diff > {} at qi={} rank={}: batched={} single={}",
                tol,
                qi,
                i,
                b,
                s
            );
        }
    }
}

#[test]
fn concurrent_search_matches_serial() {
    let dim = 256;
    let bits = 4;
    let n = 500;
    let data = gaussian_normalized(n, dim, 0xFACE_CAFE);
    let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
    idx.add(&data);
    let idx = Arc::new(idx);

    let q = gaussian_normalized(4, dim, 0xBEEF_0000);
    let expected = idx.search(&q, 10);
    let expected_indices: Vec<Vec<i64>> = (0..expected.nq)
        .map(|qi| expected.indices_for_query(qi).to_vec())
        .collect();

    let mut handles = Vec::new();
    for _ in 0..8 {
        let idx = Arc::clone(&idx);
        let q = q.clone();
        let expected_indices = expected_indices.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..16 {
                let r = idx.search(&q, 10);
                for (qi, exp) in expected_indices.iter().enumerate() {
                    assert_eq!(r.indices_for_query(qi), exp.as_slice());
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("worker panicked");
    }
}

#[test]
fn wide_single_thread_batch_scores_every_query() {
    // Regression for the thread-aware batch width (H124): single-threaded
    // batch searches widen to a 10-query batch when that saves a pass over
    // the code array, but only the permute-dot (4-bit vector-major) kernel
    // carries 10 query lanes. On 2/3-bit vector-major indexes the batch
    // lands in the 8-wide VNNI kernel instead, which scored lanes 0..8 and
    // silently dropped queries 8 and 9 of every batch — they came back as
    // NEG_INFINITY scores with id-0 padding. The width is now gated on the
    // permute-dot kernel actually taking the batch.
    //
    // A 1-thread pool with nq ∈ {10, 50} makes 10 the pass-saving width
    // (nq.div_ceil(10) < nq.div_ceil(8)); every query at every bit width
    // must come back fully scored. The other batched tests all use nq <= 8
    // and the ambient multi-thread pool, where the width is always 8 —
    // which is exactly how this went unseen. Runs on every arch; the
    // kernel it guards is x86 VBMI+VNNI.
    let dim = 512;
    let n = 500;
    let k = 5;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    for bits in [2usize, 3, 4] {
        let data = gaussian_normalized(n, dim, 0x81AC_4E55 ^ bits as u64);
        let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
        idx.add(&data);

        for nq in [10usize, 50] {
            let q = &data[..nq * dim];
            let res = pool.install(|| idx.search(q, k));

            for qi in 0..nq {
                let ids = res.indices_for_query(qi);
                let scores = res.scores_for_query(qi);
                assert_eq!(
                    ids.len(),
                    k,
                    "short result set: bits={bits} nq={nq} qi={qi}"
                );
                // A dropped query lane surfaces as NEG_INFINITY padding.
                assert!(
                    scores.iter().all(|s| s.is_finite()),
                    "dropped query lane: bits={bits} nq={nq} qi={qi} scores={scores:?}"
                );
                // Queries are the stored vectors themselves; even at 2 bits
                // the self-match sits comfortably inside the top 5 at
                // dim=512 (the dedicated 2-bit test above holds it to the
                // top 3).
                assert!(
                    ids.contains(&(qi as i64)),
                    "self-match missing: bits={bits} nq={nq} qi={qi} top{k}={ids:?}"
                );
            }
        }
    }
}

#[test]
fn multi_batch_accumulator_flush_at_high_dim() {
    // #307(3): every other searching test tops out at dim=512, i.e.
    // n_byte_groups = 256 = FLUSH_EVERY exactly, so `n_batches == 1` on
    // every arch and the flush/accumulate-reset logic never runs a second
    // time. dim 1024 and 2048 at 4-bit give 2 and 4 batches.
    //
    // This is also the test for the u16-overflow argument behind
    // `max_lut = 127`: without a flush every 256 groups a 512-group scan
    // accumulates up to 512*254 = 130048 into a u16 lane, which wraps and
    // destroys the self-match below.
    let bits = 4;
    for &dim in &[1024usize, 2048] {
        assert!(dim / 2 > 256, "dim={dim} must exceed one FLUSH_EVERY batch");
        for &n in &[64usize, 100] {
            let data = gaussian_normalized(n, dim, 0xF105_4000 ^ dim as u64 ^ n as u64);
            let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
            idx.add(&data);

            // nq=1 and nq=4 take different NEON/AVX2 kernels (single-query
            // block-parallel vs 4-query fused), each with its own batch loop.
            for &nq in &[1usize, 4, 8] {
                let q = &data[..nq * dim];
                let res = idx.search(q, 5);

                for qi in 0..nq {
                    assert_eq!(
                        res.indices_for_query(qi)[0],
                        qi as i64,
                        "self-match failed across a batch flush: dim={dim} n={n} nq={nq} qi={qi}"
                    );
                    let scores = res.scores_for_query(qi);
                    assert!(
                        scores.iter().all(|s| s.is_finite()),
                        "non-finite score: dim={dim} n={n} nq={nq} qi={qi} {scores:?}"
                    );
                    // A wrapped u16 accumulator produces a wildly off-scale
                    // self-score; the true value sits near cosine 1.0.
                    assert!(
                        (scores[0] - 1.0).abs() < 0.25,
                        "self-score {} off scale (wrap or missed flush?): \
                         dim={dim} n={n} nq={nq} qi={qi}",
                        scores[0]
                    );
                }
            }
        }
    }
}
