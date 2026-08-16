//! Peak-heap gate on the v7 load path.
//!
//! `load` verifies the adopted commit's delta by collecting every unit
//! that commit's sync wrote into a `Vec<(usize, Vec<u8>)>` and then
//! copying all of it again into one contiguous digest buffer — on top of
//! the whole file, which `load` has already read into memory. A sync
//! that appended most of the index therefore makes the next load's peak
//! heap roughly three times the file rather than one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::PathBuf;

use turbovec::TurboQuantIndex;

thread_local! {
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

struct TrackingAlloc;

// SAFETY: every method forwards to `System` unchanged; the counters are
// `Cell`s in const-initialised thread-local storage, so touching them
// allocates nothing and cannot re-enter the allocator. `try_with`
// tolerates the TLS-teardown window.
unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size() as i64);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size() as i64);
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note(-(layout.size() as i64));
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size as i64 - layout.size() as i64);
        System.realloc(ptr, layout, new_size)
    }
}

fn note(delta: i64) {
    if !ARMED.try_with(|a| a.get()).unwrap_or(false) {
        return;
    }
    let _ = LIVE.try_with(|l| {
        let now = l.get() + delta;
        l.set(now);
        let _ = PEAK.try_with(|p| {
            if now > p.get() {
                p.set(now)
            }
        });
    });
}

#[global_allocator]
static ALLOC: TrackingAlloc = TrackingAlloc;

/// Peak live heap, in bytes, reached on this thread while `f` ran.
fn peak_bytes<T>(f: impl FnOnce() -> T) -> (T, i64) {
    LIVE.with(|l| l.set(0));
    PEAK.with(|p| p.set(0));
    ARMED.with(|a| a.set(true));
    let out = f();
    ARMED.with(|a| a.set(false));
    (out, PEAK.with(|p| p.get()))
}

const DIM: usize = 128;

fn rows(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * DIM];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
    }
    for row in v.chunks_mut(DIM) {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    v
}

fn temp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-advmem-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p.push("index.tv");
    p
}

#[test]
fn loading_after_a_large_append_does_not_triple_the_files_footprint() {
    let path = temp("bigdelta");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&rows(1024, 80)).unwrap();
    idx.add(&rows(1024, 81));
    idx.sync(&path).unwrap();
    let small = std::fs::metadata(&path).unwrap().len() as i64;

    // One incremental sync that appends nearly the whole index: its
    // delta descriptor names every unit it wrote.
    idx.add(&rows(300_000, 82));
    let ((), sync_peak) = peak_bytes(|| idx.sync(&path).unwrap());
    let full = std::fs::metadata(&path).unwrap().len() as i64;
    // A sync legitimately holds its write payload once, and this one
    // also materializes the index's packed codes. Anything past that is
    // the digest keeping a second copy of everything being written.
    let codes = (301_024 * DIM * 4 / 8) as i64;
    let sync_budget = full + codes + full / 8;
    assert!(
        sync_peak <= sync_budget,
        "sync peaked at {sync_peak} bytes writing a {full}-byte file ({:.2}x the \
         file, budget {sync_budget}); the appended payload is held twice",
        sync_peak as f64 / full as f64
    );
    let delta = full - small;
    assert!(delta > 8 << 20, "the append delta is too small to measure ({delta} bytes)");

    let (loaded, peak) = peak_bytes(|| TurboQuantIndex::load(&path).unwrap());
    assert_eq!(loaded.len(), 301_024);

    // The load already holds the whole file; a modest working margin on
    // top is expected. Buffering the delta twice is not.
    let budget = full + full / 2;
    assert!(
        peak <= budget,
        "load peaked at {peak} bytes for a {full}-byte file ({:.2}x); the \
         delta covering {delta} bytes is buffered, then copied again, on top \
         of the file image",
        peak as f64 / full as f64
    );
}

// ---------------------------------------------------------------------
// v6 load: memory must track declared content, not apparent file length
// (#487). `write()` always emits an exact-length file, so these only
// bite on a file some other writer produced, padded, or made sparse —
// and a sparse file makes a huge apparent length nearly free.
// ---------------------------------------------------------------------

/// Grow `path` to `len` without writing bytes (a hole).
fn make_sparse(path: &std::path::Path, len: u64) {
    let f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    f.set_len(len).unwrap();
}

const PADDED: u64 = 256 * 1024 * 1024;

#[test]
fn a_v6_index_padded_into_a_sparse_file_costs_its_content_not_its_length() {
    let path = temp("v6padded");
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add(&rows(64, 90));
    idx.write(&path).unwrap();
    let content = std::fs::metadata(&path).unwrap().len() as i64;
    make_sparse(&path, PADDED);

    let (loaded, peak) = peak_bytes(|| TurboQuantIndex::load(&path).unwrap());
    assert_eq!(loaded.len(), 64, "the padded file must still load correctly");

    // Unfixed: the tail is sized from the whole remainder and then copied
    // again, so peak is ~2x the padding (8.2 GB for a 4 GiB file).
    assert!(
        peak < content * 64 + (1 << 20),
        "load of a {PADDED}-byte sparse file with {content} bytes of content \
         peaked at {peak} bytes"
    );
}

#[test]
fn a_foreign_file_is_rejected_without_reading_it() {
    let path = temp("foreign");
    std::fs::write(&path, b"\x7fELF\x02\x01\x01\x00").unwrap();
    make_sparse(&path, PADDED);

    let (err, peak) = peak_bytes(|| TurboQuantIndex::load(&path).unwrap_err());
    assert!(
        err.to_string().contains("not a turbovec"),
        "unexpected error: {err}"
    );
    // Unfixed: the whole file is read before the magic is compared.
    assert!(
        peak < (1 << 20),
        "rejecting a {PADDED}-byte non-turbovec file peaked at {peak} bytes"
    );
}

#[test]
fn a_padded_id_map_file_costs_its_content_not_its_length() {
    let path = temp("tvimpadded");
    let mut idx = turbovec::IdMapIndex::new(DIM, 4).unwrap();
    idx.add_with_ids(&rows(64, 91), &(0u64..64).collect::<Vec<_>>()).unwrap();
    idx.write(&path).unwrap();
    let content = std::fs::metadata(&path).unwrap().len() as i64;
    make_sparse(&path, PADDED);

    let (loaded, peak) = peak_bytes(|| turbovec::IdMapIndex::load(&path).unwrap());
    assert_eq!(loaded.len(), 64, "the padded file must still load correctly");
    assert!(
        peak < content * 64 + (1 << 20),
        "load of a padded .tvim with {content} bytes of content peaked at {peak} bytes"
    );
}
