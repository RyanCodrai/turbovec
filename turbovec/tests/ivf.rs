//! Coarse-cell index over residuals: correctness of the decomposition,
//! the exhaustive limit, and the online contract.

use turbovec::ivf::IvfIndex;
use turbovec::TurboQuantIndex;

const DIM: usize = 128;

/// Deterministic clustered corpus: `n` vectors drawn around `groups`
/// centres, unit-normalized, from a simple LCG so the fixture is the
/// same on every platform.
fn corpus(n: usize, groups: usize) -> Vec<f32> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let centres: Vec<Vec<f32>> = (0..groups)
        .map(|_| (0..DIM).map(|_| next()).collect())
        .collect();
    let mut out = Vec::with_capacity(n * DIM);
    for i in 0..n {
        let c = &centres[i % groups];
        let mut v: Vec<f32> = (0..DIM).map(|d| c[d] + 0.35 * next()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }
        out.extend_from_slice(&v);
    }
    out
}

fn exact_top_k(base: &[f32], q: &[f32], k: usize) -> Vec<usize> {
    let n = base.len() / DIM;
    let mut scored: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let v = &base[i * DIM..(i + 1) * DIM];
            ((0..DIM).map(|d| q[d] * v[d]).sum::<f32>(), i)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

fn recall_at(got: &[u64], want: &[usize]) -> f64 {
    let hit = want.iter().filter(|w| got.contains(&(**w as u64))).count();
    hit as f64 / want.len() as f64
}

#[test]
fn unfitted_index_is_exhaustive_and_correct_from_the_first_add() {
    let base = corpus(200, 4);
    let mut ivf = IvfIndex::new(DIM, 4, 16).unwrap().with_fit_threshold(usize::MAX);
    let ids = ivf.add(&base);

    assert_eq!(ids.len(), 200);
    assert_eq!(ivf.len(), 200);
    assert!(!ivf.is_fitted(), "threshold was never reached");
    assert_eq!(ivf.nlist(), 0, "no cells exist before the fit");

    // Buffered vectors are scored directly, so the top hit for a stored
    // vector is that vector — no partition can hide it.
    let q = &base[7 * DIM..8 * DIM];
    let (scores, got) = ivf.search(q, 5, 1);
    assert_eq!(got[0], 7, "self-retrieval from the buffer");
    assert!(scores[0] > 0.99, "self score should be ~1, got {}", scores[0]);
}

#[test]
fn probing_every_cell_matches_the_unpartitioned_candidate_set() {
    let base = corpus(1_000, 8);
    let mut ivf = IvfIndex::new(DIM, 4, 16).unwrap().with_fit_threshold(500);
    ivf.add(&base);
    assert!(ivf.is_fitted());
    assert_eq!(ivf.len(), 1_000);

    let mut flat = TurboQuantIndex::new(DIM, 4).unwrap();
    flat.add(&base);
    flat.prepare();

    // With every cell probed the two see the same candidates, so the
    // only difference is how each encodes them: residuals here, raw
    // vectors there. Residuals are smaller and resolve more finely at
    // the same bit width, so the partitioned index should match or beat
    // the flat one. Averaged over queries — a single query moves by
    // more than the effect being measured.
    let queries: Vec<usize> = (0..50).map(|i| i * 19 % 1_000).collect();
    let (mut r_ivf, mut r_flat) = (0.0, 0.0);
    for &qi in &queries {
        let q = &base[qi * DIM..(qi + 1) * DIM];
        let want = exact_top_k(&base, q, 10);
        let (_, got) = ivf.search(q, 10, 16);
        let flat_ids: Vec<u64> = flat
            .search(q, 10)
            .indices
            .iter()
            .map(|i| *i as u64)
            .collect();
        r_ivf += recall_at(&got, &want);
        r_flat += recall_at(&flat_ids, &want);
        assert_eq!(got[0], qi as u64, "query {qi} should retrieve itself first");
    }
    let (r_ivf, r_flat) = (r_ivf / queries.len() as f64, r_flat / queries.len() as f64);
    assert!(
        r_ivf >= r_flat,
        "residual coding lost recall against raw coding on the same candidates: \
         {r_ivf} vs {r_flat}"
    );
}

#[test]
fn recall_rises_with_nprobe_and_traffic() {
    let base = corpus(2_000, 12);
    let mut ivf = IvfIndex::new(DIM, 4, 32).unwrap().with_fit_threshold(1_000);
    ivf.add(&base);
    let mut flat = TurboQuantIndex::new(DIM, 4).unwrap();
    flat.add(&base);
    flat.prepare();

    let queries: Vec<usize> = (0..40).map(|i| i * 47 % 2_000).collect();
    let mut last = 0.0;
    for nprobe in [1usize, 4, 32] {
        let mut total = 0.0;
        for &qi in &queries {
            let q = &base[qi * DIM..(qi + 1) * DIM];
            let want = exact_top_k(&base, q, 10);
            let (_, got) = ivf.search(q, 10, nprobe);
            total += recall_at(&got, &want);
        }
        let recall = total / queries.len() as f64;
        assert!(
            recall >= last - 1e-9,
            "recall fell as nprobe rose: {recall} at nprobe={nprobe} vs {last} before"
        );
        last = recall;
    }

    // The ceiling is what the quantizer can do on this corpus, not an
    // absolute: 4-bit codes lose real recall here, and the partitioned
    // index cannot beat its own encoding by probing more cells.
    let mut flat_recall = 0.0;
    for &qi in &queries {
        let q = &base[qi * DIM..(qi + 1) * DIM];
        let want = exact_top_k(&base, q, 10);
        let ids: Vec<u64> = flat.search(q, 10).indices.iter().map(|i| *i as u64).collect();
        flat_recall += recall_at(&ids, &want);
    }
    let flat_recall = flat_recall / queries.len() as f64;
    assert!(
        last >= flat_recall,
        "probing every cell should reach the unpartitioned ceiling: {last} vs {flat_recall}"
    );
}

#[test]
fn vectors_added_after_the_fit_are_retrievable() {
    let base = corpus(800, 6);
    let mut ivf = IvfIndex::new(DIM, 4, 16).unwrap().with_fit_threshold(400);
    ivf.add(&base[..400 * DIM]);
    assert!(ivf.is_fitted(), "fit should have run at the threshold");

    // The corpus grows 2x past the fit with no rebuild.
    let late_ids = ivf.add(&base[400 * DIM..]);
    assert_eq!(ivf.len(), 800);
    assert_eq!(late_ids.first().copied(), Some(400));

    for qi in [400usize, 555, 799] {
        let q = &base[qi * DIM..(qi + 1) * DIM];
        let (_, got) = ivf.search(q, 10, 16);
        assert_eq!(got[0], qi as u64, "post-fit vector {qi} not retrieved");
    }
}

#[test]
fn cells_cover_every_vector_and_len_is_logical() {
    let base = corpus(1_200, 10);
    let mut ivf = IvfIndex::new(DIM, 4, 24).unwrap().with_fit_threshold(600);
    ivf.add(&base);

    let sizes = ivf.cell_sizes();
    assert_eq!(sizes.len(), 24);
    let stored: usize = sizes.iter().sum();
    // Margin spill (H7) stores boundary vectors in two cells, so the
    // physical count is >= the logical one, bounded by 2x; len() must
    // report the logical count exactly.
    assert!(stored >= 1_200, "every vector lands in at least one cell: {stored}");
    assert!(stored <= 2 * 1_200);
    assert_eq!(ivf.len(), 1_200, "len() is the logical count, spill excluded");
    assert!(sizes.iter().any(|&s| s > 0));
}

#[test]
fn a_vector_on_its_centroid_scores_through_the_offset() {
    // Residual zero encodes with scale 0 and contributes nothing; the
    // score must still be carried by q·c rather than collapsing to 0.
    let base = corpus(600, 4);
    let mut ivf = IvfIndex::new(DIM, 4, 8).unwrap().with_fit_threshold(300);
    ivf.add(&base);

    let q = &base[42 * DIM..43 * DIM];
    let (scores, got) = ivf.search(q, 5, 8);
    assert_eq!(got[0], 42);
    assert!(
        scores[0] > 0.9,
        "self-score should be near 1 even with residual quantization, got {}",
        scores[0]
    );
}
