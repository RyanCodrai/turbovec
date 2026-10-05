//! One query searched on a pool takes a different path through the staged
//! search than a batch: its first ranking pass runs inside the scan's own
//! workers (`in_range_refine`), and the shortlist it hands on is assembled
//! from candidates that pass already scored. #563's first version restored
//! the plain sign score there and again at assembly, so on a corpus with a
//! strong shared direction one query at a time ranked worse than the same
//! queries batched (summed top-100 overlap 13,377 against 15,687). This
//! pins the two paths to the same quality against float ground truth.
//!
//! On a host that scans the whole index instead (x86 without AVX-512 VBMI
//! and VNNI) both paths are that scan and the test holds trivially.

use turbovec::TurboQuantIndex;

const DIM: usize = 256;
const N: usize = 35_205;
const NQ: usize = 100;

fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n * dim)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 30) as f64 - 1.0) as f32
        })
        .collect()
}

fn normalise(v: &mut [f32], dim: usize) {
    for row in v.chunks_mut(dim) {
        let inv = 1.0 / row.iter().map(|x| x * x).sum::<f32>().sqrt();
        row.iter_mut().for_each(|x| *x *= inv);
    }
}

/// Every vector a common component plus a smaller individual one — the
/// geometry of domain-tuned embeddings (MedCPT: shared direction of norm
/// 0.8) — and queries beside database vectors, as real queries sit beside
/// their neighbours.
fn bunched() -> (Vec<f32>, Vec<f32>) {
    let shared = {
        let mut s = rows(1, DIM, 71);
        normalise(&mut s, DIM);
        s
    };
    let mut data = rows(N, DIM, 72);
    normalise(&mut data, DIM);
    for row in data.chunks_mut(DIM) {
        for (x, c) in row.iter_mut().zip(&shared) {
            *x = 0.8 * c + 0.6 * *x;
        }
    }
    normalise(&mut data, DIM);
    let noise = {
        let mut n = rows(NQ, DIM, 73);
        normalise(&mut n, DIM);
        n
    };
    let mut q = Vec::with_capacity(NQ * DIM);
    for i in 0..NQ {
        let src = &data[(i * 331 % N) * DIM..(i * 331 % N + 1) * DIM];
        q.extend(src.iter().zip(&noise[i * DIM..(i + 1) * DIM]).map(|(a, b)| a + 0.15 * b));
    }
    normalise(&mut q, DIM);
    (data, q)
}

/// Summed overlap of each query's returned ids with its float top-`k`.
fn overlap(data: &[f32], q: &[f32], k: usize, got: &[Vec<i64>]) -> usize {
    let mut total = 0;
    for (qi, ids) in got.iter().enumerate() {
        let qv = &q[qi * DIM..(qi + 1) * DIM];
        let mut s: Vec<(f32, usize)> = data
            .chunks(DIM)
            .enumerate()
            .map(|(i, x)| (x.iter().zip(qv).map(|(a, b)| a * b).sum::<f32>(), i))
            .collect();
        s.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
        let truth: std::collections::HashSet<i64> = s[..k].iter().map(|p| p.1 as i64).collect();
        total += ids.iter().filter(|i| truth.contains(i)).count();
    }
    total
}

#[test]
fn one_query_on_a_pool_ranks_as_well_as_a_batch() {
    let (data, q) = bunched();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for bits in [4usize, 2] {
        let mut ix = TurboQuantIndex::new(DIM, bits).unwrap();
        ix.calibrate(&data[..1024 * DIM]).unwrap();
        ix.add(&data);
        for k in [10usize, 100] {
            let (batched, single) = pool.install(|| {
                let r = ix.search(&q, k);
                let batched: Vec<Vec<i64>> =
                    (0..NQ).map(|i| r.indices[i * r.k..(i + 1) * r.k].to_vec()).collect();
                let single: Vec<Vec<i64>> =
                    (0..NQ).map(|i| ix.search(&q[i * DIM..(i + 1) * DIM], k).indices).collect();
                (batched, single)
            });
            let (b, s) = (overlap(&data, &q, k, &batched), overlap(&data, &q, k, &single));
            assert!(
                s * 100 >= b * 97,
                "{bits}-bit k={k}: one query at a time overlaps the float top-k {s} times, a batch {b}"
            );
        }
    }
}
