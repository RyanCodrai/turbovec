// P2 probe: where does the parallel codes read actually spend its time?
//
// Mirrors turbovec's read_range_parallel_transform (N scoped threads, one
// positioned read each, into a freshly-allocated destination) and compares
// it against variants that change only how the destination is obtained:
//
//   fresh      — Vec::with_capacity, exactly what the loader does today
//   pretouched — same, but every page written once before the read starts
//                (isolates first-touch fault cost from the copy itself)
//   hugepage   — anonymous mmap with MADV_HUGEPAGE, so the same span takes
//                ~512x fewer faults and far less dTLB pressure
//   reused     — one buffer allocated once and re-read into every rep
//                (the floor: no faults at all after the first rep)
//
// Build: rustc -O probe_p2.rs -o probe_p2 ; run: ./probe_p2 <file> [reps]
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::time::Instant;

const CHUNK_MIN: usize = 8 * 1024 * 1024;

fn read_into(f: &File, base: *mut u8, len: usize, off: u64, n_threads: usize) {
    let chunk = len.div_ceil(n_threads).max(CHUNK_MIN).next_multiple_of(4096);
    let n_chunks = len.div_ceil(chunk);
    #[derive(Clone, Copy)]
    struct P(*mut u8);
    unsafe impl Send for P {}
    unsafe impl Sync for P {}
    let p = P(base);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..n_threads.min(n_chunks) {
            s.spawn(|| {
                let p = p;
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n_chunks {
                        break;
                    }
                    let o = i * chunk;
                    let this = chunk.min(len - o);
                    let buf = unsafe { std::slice::from_raw_parts_mut(p.0.add(o), this) };
                    f.read_exact_at(buf, off + o as u64).unwrap();
                }
            });
        }
    });
}

fn mmap_anon(len: usize, huge: bool) -> *mut u8 {
    unsafe {
        let p = libc_mmap(len);
        if huge {
            madvise_huge(p, len);
        }
        p
    }
}

// Minimal libc bindings so the probe needs no crates.
unsafe extern "C" {
    fn mmap(
        addr: *mut core::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        off: i64,
    ) -> *mut core::ffi::c_void;
    fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
    fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 2;
const MAP_ANONYMOUS: i32 = 0x20;
const MADV_HUGEPAGE: i32 = 14;

unsafe fn libc_mmap(len: usize) -> *mut u8 {
    let p = unsafe {
        mmap(
            std::ptr::null_mut(),
            len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert!(p as isize != -1, "mmap failed");
    p as *mut u8
}
unsafe fn madvise_huge(p: *mut u8, len: usize) {
    let r = unsafe { madvise(p as *mut core::ffi::c_void, len, MADV_HUGEPAGE) };
    if r != 0 {
        eprintln!("note: MADV_HUGEPAGE rejected (THP likely disabled)");
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let reps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(15);
    let f = File::open(path).unwrap();
    let len = f.metadata().unwrap().len() as usize;
    // Read the codes span only: skip a nominal header, drop a nominal tail.
    let off = 4096u64;
    let len = len - 8 * 1024 * 1024 - off as usize;
    let nt = std::thread::available_parallelism().unwrap().get();
    println!("file={path} span={:.1} MB threads={nt} reps={reps}", len as f64 / 1e6);

    // Warm the page cache for the span.
    read_into(&f, mmap_anon(len, false), len, off, nt);

    let mut t = Vec::new();
    for _ in 0..reps {
        let mut v: Vec<u8> = Vec::with_capacity(len);
        let s = Instant::now();
        read_into(&f, v.spare_capacity_mut().as_mut_ptr() as *mut u8, len, off, nt);
        t.push(s.elapsed().as_secs_f64() * 1e3);
        unsafe { v.set_len(len) };
        std::hint::black_box(&v[0]);
    }
    println!("fresh      {:8.3} ms", median(t));

    let mut t = Vec::new();
    for _ in 0..reps {
        let mut v: Vec<u8> = Vec::with_capacity(len);
        let base = v.spare_capacity_mut().as_mut_ptr() as *mut u8;
        let mut i = 0;
        while i < len {
            unsafe { base.add(i).write(0) };
            i += 4096;
        }
        let s = Instant::now();
        read_into(&f, base, len, off, nt);
        t.push(s.elapsed().as_secs_f64() * 1e3);
        unsafe { v.set_len(len) };
        std::hint::black_box(&v[0]);
    }
    println!("pretouched {:8.3} ms   (excludes the pre-touch itself)", median(t));

    let mut t = Vec::new();
    for _ in 0..reps {
        let p = mmap_anon(len, true);
        let s = Instant::now();
        read_into(&f, p, len, off, nt);
        t.push(s.elapsed().as_secs_f64() * 1e3);
        unsafe { munmap(p as *mut core::ffi::c_void, len) };
    }
    println!("hugepage   {:8.3} ms", median(t));

    let p = mmap_anon(len, false);
    read_into(&f, p, len, off, nt);
    let mut t = Vec::new();
    for _ in 0..reps {
        let s = Instant::now();
        read_into(&f, p, len, off, nt);
        t.push(s.elapsed().as_secs_f64() * 1e3);
    }
    println!("reused     {:8.3} ms", median(t));
}
