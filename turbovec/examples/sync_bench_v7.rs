//! Sync latency per change shape, 50k x 512d 2-bit (every sync is
//! durable). Run: cargo run --release --example sync_bench_v7
use std::time::Instant;
use turbovec::TurboQuantIndex;

fn main() {
    let n = 50_000;
    let dim = 512;
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut rows = |k: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(k * dim);
        for _ in 0..k * dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5);
        }
        v
    };
    let dir = std::env::temp_dir().join("tv_sync_bench_v7");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bench.tv");

    let mut idx = TurboQuantIndex::new(dim, 2).unwrap();
    idx.add_2d(&rows(n), dim).unwrap();
    let t = Instant::now();
    idx.sync(&path).unwrap();
    println!("first sync (full write): {:.2} ms", t.elapsed().as_secs_f64() * 1e3);

    // 32-row append
    let mut best = f64::MAX;
    for _ in 0..5 {
        idx.add_2d(&rows(32), dim).unwrap();
        let t = Instant::now();
        idx.sync(&path).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("append 32 rows:  best {best:.2} ms");
    // single removal in a committed block
    let mut best = f64::MAX;
    for i in 0..5 {
        idx.swap_remove(100 + i);
        let t = Instant::now();
        idx.sync(&path).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("single removal:  best {best:.2} ms");
    // 200 scattered removals in one sync: all ride the header as redo
    // ops (capacity 1024), one fsync.
    let mut victims: Vec<usize> = (0..200).map(|i| 7 + i * 231).collect();
    victims.sort_unstable_by(|a, b| b.cmp(a));
    for v in victims {
        idx.swap_remove(v);
    }
    let t = Instant::now();
    idx.sync(&path).unwrap();
    println!("200 scattered removals, one sync: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    std::fs::remove_dir_all(&dir).ok();
}
