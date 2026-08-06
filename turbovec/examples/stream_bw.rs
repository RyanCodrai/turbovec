//! What read bandwidth can this box actually sustain, and where does the
//! scan sit against it?
//!
//! This gates the largest outstanding idea. At `QBS = 4` a search makes
//! `nq/4 = 25` passes over the 76.8 MB code array, so it moves 1.92 GB per
//! search. FAISS and ScaNN both separate two limits we have folded into one
//! constant: how many queries fit in registers (FAISS instantiates 1-4,
//! ScaNN caps at 3) versus how many queries ride along per *pass over the
//! codes* (FAISS's default `qbs2 = 11`, decomposed 2+3+3+3). Raising the
//! second cuts DRAM traffic proportionally — 25 passes to 9.
//!
//! Whether that is worth anything depends entirely on whether the scan is
//! bandwidth-bound. If it is at ~90% of achievable read bandwidth, the win
//! is large and real. If it is well under, the bytes are already being
//! served from cache and fewer passes buys nothing.
//!
//! Two sizes matter, because they can differ per arch:
//!   * 77 MB  — the actual code array. The arm box has ~80 MB of L3, so
//!     this may be largely L3-resident there while overflowing the x86
//!     box's smaller L3. If so, the two arches are in different regimes
//!     and the same change is worth different amounts on each.
//!   * 512 MB — comfortably past any L3, so this is DRAM.
//!
//! Run: cargo run --release --example stream_bw

use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn measure(bytes: usize, threads: usize, reps: usize) -> f64 {
    let buf: Arc<Vec<u64>> = Arc::new(vec![1u64; bytes / 8]);
    let chunk = buf.len() / threads;

    // Warm the pages so the first pass is not paying faults.
    let mut warm = 0u64;
    for v in buf.iter().step_by(512) {
        warm = warm.wrapping_add(*v);
    }
    std::hint::black_box(warm);

    let t0 = Instant::now();
    let mut hs = Vec::with_capacity(threads);
    for t in 0..threads {
        let b = Arc::clone(&buf);
        hs.push(thread::spawn(move || {
            let lo = t * chunk;
            let hi = if t == threads - 1 { b.len() } else { lo + chunk };
            let mut acc = [0u64; 8];
            for _ in 0..reps {
                // Eight independent accumulators so the summation itself is
                // never the limit — this measures the memory system.
                for c in b[lo..hi].chunks_exact(8) {
                    for i in 0..8 {
                        acc[i] = acc[i].wrapping_add(c[i]);
                    }
                }
            }
            acc.iter().fold(0u64, |a, b| a.wrapping_add(*b))
        }));
    }
    let mut sink = 0u64;
    for h in hs {
        sink = sink.wrapping_add(h.join().unwrap());
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    (bytes * reps) as f64 / dt / 1e9
}

fn main() {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    println!("threads {threads}");
    println!();

    let small = measure(77 * 1024 * 1024, threads, 8);
    let large = measure(512 * 1024 * 1024, threads, 2);
    let single = measure(512 * 1024 * 1024, 1, 1);

    println!("read bandwidth");
    println!("  77 MB  (code-array sized)   {small:7.1} GB/s");
    println!("  512 MB (past any L3)        {large:7.1} GB/s");
    println!("  512 MB, single thread       {single:7.1} GB/s");
    println!();
    println!("A 77 MB figure well above the 512 MB figure means the code");
    println!("array is substantially cache-resident on this box, and cutting");
    println!("passes over it is worth much less than the traffic arithmetic");
    println!("suggests.");
    println!();
    println!("Scan traffic at QBS=4: nq/4 = 25 passes x 76.8 MB = 1.92 GB.");
    println!("Divide 1.92 by the measured search seconds for achieved GB/s,");
    println!("then compare against the figures above.");
}
