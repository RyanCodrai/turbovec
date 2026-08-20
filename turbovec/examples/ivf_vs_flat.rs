//! main-vs-IVF: recall@10 and latency, reading pre-normalized data and
//! precomputed exact truth from flat files (see prep_truth.py).
use std::io::Write;
use std::time::Instant;
use turbovec::ivf::IvfIndex;
use turbovec::TurboQuantIndex;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: ivf_vs_flat <dir> <n>");
    let n: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let (dim, nq, k) = (1536usize, 500usize, 10usize);
    let base = read_f32(&format!("{dir}/base_{n}.f32"), n * dim);
    let queries = read_f32(&format!("{dir}/queries.f32"), nq * dim);
    let gt = read_i64(&format!("{dir}/gt_{n}.i64"), nq * k);

    let mut flat = TurboQuantIndex::new(dim, 4).unwrap();
    let t = Instant::now();
    flat.add(&base);
    flat.prepare();
    let build = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let fr = flat.search(&queries, k);
    let batch = t.elapsed().as_secs_f64();
    let rec: f64 = (0..nq).map(|qi| recall(fr.indices_for_query(qi).iter().map(|i| *i as u64), &gt[qi*k..(qi+1)*k], k)).sum::<f64>() / nq as f64;
    let t = Instant::now();
    for qi in 0..100 { let _ = flat.search(&queries[qi*dim..(qi+1)*dim], k); }
    let single = t.elapsed().as_secs_f64() / 100.0;
    let _ = std::io::stdout().flush();
    println!("flat 4-bit      : recall={rec:.4} batch={:7.0} QPS single={:6.2} ms build={build:.1}s",
             nq as f64 / batch, single * 1e3);

    let nlist = (n as f64).sqrt() as usize;
    let mut ivf = IvfIndex::new(dim, 4, nlist).unwrap().with_fit_threshold(40_000);
    let t = Instant::now();
    ivf.add(&base);
    let build = t.elapsed().as_secs_f64();
    for nprobe in [8usize, 16, 32, 64, 128, nlist] {
        let t = Instant::now();
        let (_, ids) = ivf.search(&queries, k, nprobe);
        let batch = t.elapsed().as_secs_f64();
        let rec: f64 = (0..nq).map(|qi| recall(ids[qi*k..(qi+1)*k].iter().copied(), &gt[qi*k..(qi+1)*k], k)).sum::<f64>() / nq as f64;
        let t = Instant::now();
        for qi in 0..50 { let _ = ivf.search(&queries[qi*dim..(qi+1)*dim], k, nprobe); }
        let single = t.elapsed().as_secs_f64() / 50.0;
        let _ = std::io::stdout().flush();
        println!("ivf nprobe={nprobe:4} : recall={rec:.4} batch={:7.0} QPS single={:6.2} ms build={build:.1}s{}",
                 nq as f64 / batch, single * 1e3, if nprobe == nlist { "  (all cells)" } else { "" });
    }
}

fn read_f32(p: &str, len: usize) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    assert_eq!(b.len(), len * 4, "{p}: wrong size");
    let mut v = vec![0f32; len];
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len()) };
    v
}
fn read_i64(p: &str, len: usize) -> Vec<u64> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    assert_eq!(b.len(), len * 8, "{p}: wrong size");
    let mut v = vec![0u64; len];
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len()) };
    v
}
fn recall(got: impl Iterator<Item = u64>, want: &[u64], k: usize) -> f64 {
    let g: Vec<u64> = got.collect();
    want.iter().filter(|w| g.contains(w)).count() as f64 / k as f64
}
