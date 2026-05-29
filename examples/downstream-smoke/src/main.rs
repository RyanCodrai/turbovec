//! Smoke test for the downstream `cargo add turbovec` experience.
//!
//! Exercises the public API end-to-end (construct, add, prepare, search,
//! write, load) with no BLAS-related setup. If this binary links and
//! runs, the link-directive propagation from turbovec's build.rs is
//! working and downstream users won't hit the `cblas_sgemm` error.

use turbovec::TurboQuantIndex;

const DIM: usize = 64;
const N_DB: usize = 256;
const N_QUERIES: usize = 4;
const K: usize = 5;

fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32 / u64::MAX as f32) * 2.0 - 1.0
    };
    let mut out = vec![0.0f32; n * dim];
    for v in out.chunks_mut(dim) {
        for x in v.iter_mut() {
            *x = next();
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    out
}

fn main() {
    let db = unit_vectors(N_DB, DIM, 1);
    let queries = unit_vectors(N_QUERIES, DIM, 2);

    let mut index = TurboQuantIndex::new(DIM, 4).expect("construct");
    index.add(&db);
    index.prepare();

    let results = index.search(&queries, K);
    assert_eq!(results.nq, N_QUERIES);
    assert_eq!(results.k, K);
    for q in 0..N_QUERIES {
        let idxs = results.indices_for_query(q);
        assert_eq!(idxs.len(), K);
        for &i in idxs {
            assert!((0..N_DB as i64).contains(&i), "out-of-range index {i}");
        }
    }

    let tmp = std::env::temp_dir().join("turbovec_downstream_smoke.tv");
    index.write(&tmp).expect("write");
    let loaded = TurboQuantIndex::load(&tmp).expect("load");
    assert_eq!(loaded.len(), N_DB);
    assert_eq!(loaded.dim(), DIM);
    let _ = std::fs::remove_file(&tmp);

    println!(
        "downstream-smoke: OK ({} db vectors, {} queries, top-{} search, write/load round-trip)",
        N_DB, N_QUERIES, K
    );
}
