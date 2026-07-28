//! Insertion / removal micro-benchmark for hill-climbing the mutation
//! path, without needing the Python suite's OpenAI corpus or FAISS.
//!
//! The official numbers in the README come from `benchmarks/suite/` (real
//! embeddings, FAISS comparator, fixed shapes). This harness exists for the
//! inner loop of an optimization pass: it isolates the same four mutation
//! metrics on deterministic synthetic vectors so a hypothesis can be
//! measured in seconds on any machine, then confirmed on the official
//! environment.
//!
//! ```text
//! cargo run --release --example insert_bench -- --dim 1536 --bits 2
//! RAYON_NUM_THREADS=1 cargo run --release --example insert_bench
//! ```
//!
//! Metrics (median of `--repeats` runs, fresh index per timed run):
//!
//! * `bulk`   — one `add()` of `--n` vectors into an empty index. Includes
//!   the one-time rotation/codebook init and the TQ+ calibration fit.
//! * `warm`   — a second `add()` on that index: caches built, calibration
//!   frozen. The steady-state encode path.
//! * `single` — per-op latency of one-row `add()` on a warm index.
//! * `remove` — per-op latency of `IdMapIndex::remove` and
//!   `TurboQuantIndex::swap_remove`.
//! * `stages` — optional (`--stages`) breakdown of the one-time init costs
//!   (codebook, rotation build) and the rotation-only throughput, which is
//!   where most of the encode time goes.

use std::time::Instant;

use turbovec::{IdMapIndex, TurboQuantIndex};

/// Deterministic pseudo-random vectors (xorshift64*), unit-normalized so
/// the shape matches the embedding corpora the official suite uses.
fn synth(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut x = seed | 1;
    let mut v = vec![0.0f32; n * dim];
    for row in v.chunks_mut(dim) {
        for slot in row.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *slot = (x as f64 / u64::MAX as f64) as f32 - 0.5;
        }
        let norm = row.iter().map(|a| a * a).sum::<f32>().sqrt();
        for slot in row.iter_mut() {
            *slot /= norm;
        }
    }
    v
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

struct Args {
    dim: usize,
    bits: usize,
    n: usize,
    append: usize,
    singles: usize,
    removes: usize,
    repeats: usize,
    stages: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        dim: 1536,
        bits: 2,
        n: 100_000,
        append: 10_000,
        singles: 1_000,
        removes: 20_000,
        repeats: 5,
        stages: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let val = |i: usize| -> usize {
            argv.get(i + 1)
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| panic!("missing/invalid value for {}", argv[i]))
        };
        match argv[i].as_str() {
            "--dim" => { a.dim = val(i); i += 2; }
            "--bits" => { a.bits = val(i); i += 2; }
            "--n" => { a.n = val(i); i += 2; }
            "--append" => { a.append = val(i); i += 2; }
            "--singles" => { a.singles = val(i); i += 2; }
            "--removes" => { a.removes = val(i); i += 2; }
            "--repeats" => { a.repeats = val(i); i += 2; }
            "--stages" => { a.stages = true; i += 1; }
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

fn main() {
    let args = parse_args();
    let (dim, bits) = (args.dim, args.bits);
    eprintln!(
        "# dim={dim} bits={bits} n={} append={} threads={}",
        args.n,
        args.append,
        rayon::current_num_threads(),
    );

    let database = synth(args.n, dim, 0xD1536);
    let append = synth(args.append + args.singles, dim, 0xA55E3);
    let batch = &append[..args.append * dim];
    let singles = &append[args.append * dim..];

    let mut bulk = Vec::new();
    let mut warm = Vec::new();
    for _ in 0..args.repeats {
        let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
        let t0 = Instant::now();
        idx.add(&database);
        bulk.push(args.n as f64 / t0.elapsed().as_secs_f64());
        let t0 = Instant::now();
        idx.add(batch);
        warm.push(args.append as f64 / t0.elapsed().as_secs_f64());
    }

    // Single-row adds on a warm index (calibration frozen, caches built).
    let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
    idx.add(&database);
    let mut single = Vec::new();
    for _ in 0..args.repeats {
        let t0 = Instant::now();
        for i in 0..args.singles {
            idx.add(&singles[i * dim..(i + 1) * dim]);
        }
        single.push(t0.elapsed().as_secs_f64() / args.singles as f64 * 1e6);
    }

    // Removal: IdMapIndex::remove (id-map bookkeeping) vs the raw
    // swap_remove underneath it. Fresh index per timed run — removal
    // shrinks the index, so repeating on one index is not comparable.
    let mut id_remove = Vec::new();
    let mut swap_remove = Vec::new();
    let removes = args.removes.min(args.n);
    for r in 0..args.repeats {
        let ids: Vec<u64> = (0..args.n as u64).collect();
        let mut im = IdMapIndex::new(dim, bits).unwrap();
        im.add_with_ids(&database, &ids).unwrap();
        // Remove from the front: each remove swaps the tail vector in, so
        // no id in 0..removes has been moved out of the map yet.
        let t0 = Instant::now();
        for id in 0..removes as u64 {
            im.remove(id);
        }
        id_remove.push(t0.elapsed().as_secs_f64() / removes as f64 * 1e6);

        let mut ti = TurboQuantIndex::new(dim, bits).unwrap();
        ti.add(&database);
        let t0 = Instant::now();
        for _ in 0..removes {
            ti.swap_remove(0);
        }
        swap_remove.push(t0.elapsed().as_secs_f64() / removes as f64 * 1e6);
        let _ = r;
    }

    println!("{{");
    println!("  \"dim\": {dim},");
    println!("  \"bit_width\": {bits},");
    println!("  \"threads\": {},", rayon::current_num_threads());
    println!("  \"bulk_insert_vecs_per_sec\": {:.0},", median(bulk));
    println!("  \"warm_insert_vecs_per_sec\": {:.0},", median(warm));
    println!("  \"single_add_us\": {:.3},", median(single));
    println!("  \"idmap_remove_us\": {:.4},", median(id_remove));
    println!("  \"swap_remove_us\": {:.4}", median(swap_remove));
    println!("}}");

    if args.stages {
        // One-time init costs, measured separately: both are paid inside
        // the `bulk` number above and neither scales with n.
        let t0 = Instant::now();
        let mut probe = TurboQuantIndex::new(dim, bits).unwrap();
        probe.add(&database[..dim]);
        eprintln!("# first-add init + 1 row: {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
}
