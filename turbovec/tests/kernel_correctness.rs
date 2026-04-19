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

// Pull the BLAS backend into the final test binary — ndarray's `.dot()`
// (used by `encode`) references cblas_sgemm, which only gets linked if
// something forces blas_src to be included in the final executable.
extern crate blas_src;

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
        let mut idx = TurboQuantIndex::new(dim, bits);
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
        let mut idx = TurboQuantIndex::new(dim, bits);
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
            let mut idx = TurboQuantIndex::new(dim, bits);
            idx.add(&data);

            let q = &data[..4 * dim];
            let k = 10.min(n);
            let res = idx.search(q, k);

            for qi in 0..4 {
                let scores = res.scores_for_query(qi);
                for w in scores.windows(2) {
                    assert!(
                        w[0] >= w[1] || !w[1].is_finite(),
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
fn search_is_deterministic_for_same_query() {
    let dim = 256;
    let bits = 4;
    for &n in &[64usize, 65, 127, 128, 129, 500] {
        let data = gaussian_normalized(n, dim, 0xD0D0_D0D0 ^ n as u64);
        let mut idx = TurboQuantIndex::new(dim, bits);
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
    let mut idx = TurboQuantIndex::new(dim, bits);
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
    let mut idx = TurboQuantIndex::new(dim, bits);
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
