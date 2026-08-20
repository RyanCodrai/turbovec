//! H3 micro: what does one small TurboQuantIndex::search call cost?
//! A cell in the 500k index is ~707 vectors; audiences are ~23 queries.
use std::time::Instant;
use turbovec::TurboQuantIndex;
fn main() {
    let dim = 1536usize;
    let n = 707usize;
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let base: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    let mut ix = TurboQuantIndex::new(dim, 4).unwrap();
    ix.add(&base);
    ix.prepare();
    for nq in [1usize, 8, 23, 100] {
        let q: Vec<f32> = base[..nq * dim].to_vec();
        ix.search(&q, 10); // warm
        let reps = 200;
        let t = Instant::now();
        for _ in 0..reps { let _ = ix.search(&q, 10); }
        let per = t.elapsed().as_secs_f64() / reps as f64;
        println!("nq={nq:4}: {:8.1} us/call  ({:6.2} us/query)", per * 1e6, per * 1e6 / nq as f64);
    }
}
