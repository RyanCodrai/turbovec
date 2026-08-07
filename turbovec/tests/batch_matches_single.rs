//! Batch search must return exactly what per-query searches return.
//!
//! The batch dispatch tiles the (query, block) plane, chunks queries into
//! kernel-width batches, pads ragged tails, and (on some hosts) chooses its
//! batch width per call — every one of those seams is arithmetic that can
//! silently drop or misroute a query while each query's own scores stay
//! plausible. Comparing against the single-query path pins all of it at
//! once, at query counts chosen to land on the seams.

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

/// Query counts chosen for the seams, not coverage: 1 (the single-query
/// dispatch), 7 (below every batch width), 8 (exactly one classic batch),
/// 9 (one batch + a 1-lane tail), 23 (ragged at widths 8, 10 and 12), and
/// 50 (where a pass-saving width choice actually changes the pass count).
const SEAM_NQS: [usize; 6] = [1, 7, 8, 9, 23, 50];

#[test]
fn batch_results_match_single_query_results() {
    let (n, dim, k) = (9_000usize, 64usize, 10usize);
    let data = gaussian_normalized(n, dim, 0xBA7C_4);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&data);

    let queries = gaussian_normalized(50, dim, 0xBA7C_5);
    for &nq in &SEAM_NQS {
        let batch = idx.search(&queries[..nq * dim], k);
        assert_eq!(batch.nq, nq, "batch of {nq}: reported row count");
        assert_eq!(batch.scores.len(), nq * k, "batch of {nq}: flattened score count");
        assert_eq!(batch.indices.len(), nq * k, "batch of {nq}: flattened id count");
        for q in 0..nq {
            let single = idx.search(&queries[q * dim..(q + 1) * dim], k);
            assert_eq!(
                batch.indices[q * k..(q + 1) * k],
                single.indices[..],
                "batch of {nq}: ids for query {q} diverge from the single-query path",
            );
            assert_eq!(
                batch.scores[q * k..(q + 1) * k],
                single.scores[..],
                "batch of {nq}: scores for query {q} diverge from the single-query path",
            );
        }
    }
}
