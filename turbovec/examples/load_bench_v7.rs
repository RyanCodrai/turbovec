//! Load-time A/B: v6 write()+load vs v7 sync()+load, 50k x 512d, 2-bit.
//! Run: cargo run --release --example load_bench_v7
use std::time::Instant;
use turbovec::TurboQuantIndex;

fn main() {
    let n = 50_000;
    let dim = 512;
    let mut rows = Vec::with_capacity(n);
    let mut state = 0x243F_6A88_85A3_08D3u64;
    for _ in 0..n {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5);
        }
        rows.push(v);
    }
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();

    let dir = std::env::temp_dir().join("tv_load_bench_v7");
    std::fs::create_dir_all(&dir).unwrap();
    let p6 = dir.join("a.tv6");
    let p7 = dir.join("a.tv7");

    let mut ix = TurboQuantIndex::new(dim, 2).unwrap();
    ix.add_2d(&flat, dim).unwrap();
    ix.write(&p6).unwrap();
    ix.sync(&p7).unwrap();
    println!(
        "file sizes: v6 {} KB, v7 {} KB",
        std::fs::metadata(&p6).unwrap().len() / 1024,
        std::fs::metadata(&p7).unwrap().len() / 1024
    );

    for (name, path) in [("v6", &p6), ("v7", &p7)] {
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            let l = TurboQuantIndex::load(path).unwrap();
            let dt = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(l.len(), n);
            if dt < best {
                best = dt;
            }
        }
        println!("{name} load best-of-7: {best:.2} ms");
    }
    // Component floors on the v7 file: raw read, and read+touch-every-byte.
    for _ in 0..3 {
        let t = Instant::now();
        let raw = std::fs::read(&p7).unwrap();
        let read_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut acc = 0u64;
        for c in raw.chunks_exact(8) {
            acc = acc.wrapping_add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let touch_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let copy = raw.clone();
        let copy_ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box((acc, copy));
        println!("read {read_ms:.2} ms, touch {touch_ms:.2} ms, clone {copy_ms:.2} ms");
    }
    std::fs::remove_dir_all(&dir).ok();
}
