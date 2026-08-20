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
    // H3b: where is the per-vector cliff? fixed nq, growing n.
    let nq = 23usize;
    for n2 in [64usize, 256, 707, 2048, 8192, 32768, 131072] {
        let mut state2 = 0xDEADBEEFCAFEBABEu64;
        let mut next2 = || {
            state2 = state2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state2 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let b2: Vec<f32> = (0..n2 * dim).map(|_| next2()).collect();
        let mut ix2 = TurboQuantIndex::new(dim, 4).unwrap();
        ix2.add(&b2);
        ix2.prepare();
        let q: Vec<f32> = b2[..nq * dim].to_vec();
        ix2.search(&q, 10);
        let reps = 100;
        let t = Instant::now();
        for _ in 0..reps { let _ = ix2.search(&q, 10); }
        let per = t.elapsed().as_secs_f64() / reps as f64 / nq as f64;
        println!("n={n2:7}: {:8.2} us/query  ({:6.2} ns/vector)", per * 1e6, per * 1e9 / n2 as f64);
    }
}
