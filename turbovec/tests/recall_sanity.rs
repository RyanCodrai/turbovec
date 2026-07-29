//! Recall floor for the v5 block-Hadamard rotation.
//!
//! A broken rotation (one that fails to decorrelate coordinates, or
//! mis-rotates the query relative to the database) collapses recall
//! toward random (`k / n`). This test builds an index over lightly
//! clustered synthetic data and asserts recall@10 against exact-cosine
//! ground truth clears a healthy floor at dim 1536 & 1000 (the
//! collapse-to-8 weak-block case), 2- and 4-bit.
//!
//! A full head-to-head against the pre-change QR rotation (confirming
//! recall neutrality, not just a floor) was run out-of-tree; see the PR
//! description. This in-tree test is the CI regression guard.

use std::collections::HashSet;

use turbovec::TurboQuantIndex;

fn splitmix(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn gauss(s: &mut u64) -> f64 {
    let u1 = splitmix(s).max(1e-12);
    let u2 = splitmix(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Lightly clustered unit vectors (so exact NN structure exists and a
/// healthy recall floor is meaningful).
fn clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let centers: Vec<Vec<f64>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| gauss(&mut s)).collect())
        .collect();
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let c = &centers[i % n_clusters];
        let row = &mut out[i * dim..(i + 1) * dim];
        let mut nrm = 0.0f64;
        let mut tmp = vec![0.0f64; dim];
        for d in 0..dim {
            let v = c[d] + 0.7 * gauss(&mut s);
            tmp[d] = v;
            nrm += v * v;
        }
        let inv = 1.0 / nrm.sqrt().max(1e-12);
        for d in 0..dim {
            row[d] = (tmp[d] * inv) as f32;
        }
    }
    out
}

fn exact_topk(db: &[f32], q: &[f32], dim: usize, k: usize) -> Vec<HashSet<i64>> {
    let n = db.len() / dim;
    let nq = q.len() / dim;
    (0..nq)
        .map(|qi| {
            let qv = &q[qi * dim..(qi + 1) * dim];
            let mut scored: Vec<(f32, i64)> = (0..n)
                .map(|i| {
                    let dv = &db[i * dim..(i + 1) * dim];
                    (qv.iter().zip(dv).map(|(a, b)| a * b).sum(), i as i64)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            scored.into_iter().take(k).map(|(_, i)| i).collect()
        })
        .collect()
}

fn recall_at(dim: usize, bits: usize) -> f64 {
    let (n, nq, k) = (2000usize, 200usize, 10usize);
    let db = clustered(n, dim, 30, 0xC0FFEE ^ dim as u64);
    let q = clustered(nq, dim, 30, 0xBADF00D ^ dim as u64);
    let truth = exact_topk(&db, &q, dim, k);

    let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
    idx.add(&db);
    idx.prepare();
    let r = idx.search(&q, k);

    let mut hit = 0usize;
    for qi in 0..nq {
        let got: HashSet<i64> = r.indices_for_query(qi).iter().copied().collect();
        hit += got.intersection(&truth[qi]).count();
    }
    hit as f64 / (nq * k) as f64
}

#[test]
fn recall_clears_floor_across_cells_including_collapse_to_8() {
    // Floors are well below the ~0.85 (4-bit) / ~0.55 (2-bit) the pipeline
    // actually achieves, but far above the ~0.005 a random/broken
    // rotation would give — so they catch a broken transform without
    // being flaky. dim=1000 is the collapse-to-8 weak-block case.
    for &dim in &[1536usize, 1000] {
        let r4 = recall_at(dim, 4);
        assert!(r4 >= 0.70, "recall@10 4-bit dim={dim} = {r4:.3} below floor 0.70");
        let r2 = recall_at(dim, 2);
        assert!(r2 >= 0.40, "recall@10 2-bit dim={dim} = {r2:.3} below floor 0.40");
        // 3-bit had no accuracy assertion anywhere (#307): it appeared only
        // in shape, finiteness and codebook-property tests, so a 3-bit path
        // returning plausible-but-wrong rankings passed the whole suite.
        // Measured ~0.82; the floor sits between the 2- and 4-bit floors.
        let r3 = recall_at(dim, 3);
        assert!(r3 >= 0.58, "recall@10 3-bit dim={dim} = {r3:.3} below floor 0.58");
        assert!(
            r3 > r2,
            "3-bit recall {r3:.3} must beat 2-bit {r2:.3} at dim={dim}"
        );
    }
}
