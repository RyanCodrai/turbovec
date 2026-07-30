//! Allocation-shape gates for the two hot paths in issue #409.
//!
//! These pin a *decision* — how many heap allocations a code path makes,
//! and whether that number scales with the number of vectors — not a
//! duration. A wall-clock ratio would be flaky on shared CI runners and
//! slow enough to threaten the mutation gate; an allocation count is
//! exact and costs microseconds.
//!
//! Counting is per-thread and opt-in, so these tests stay correct when
//! the harness runs them in parallel with each other and with the rest
//! of the suite: only the thread inside [`count_allocs`] is counted, and
//! every path measured here is serial on the calling thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use turbovec::TurboQuantIndex;

thread_local! {
    /// Allocations made by this thread while armed.
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    /// Whether this thread is currently counting.
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

struct CountingAlloc;

// SAFETY: every method forwards to `System` unchanged; the counter is a
// `Cell<u64>` in thread-local storage declared with `const` initialisers,
// so reading or writing it allocates nothing and cannot re-enter the
// allocator. `try_with` tolerates the window during TLS teardown.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A realloc that grows a buffer is exactly the cost `with_capacity`
        // removes, so it counts as an allocation event.
        bump();
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc_zeroed(layout)
    }
}

fn bump() {
    let armed = ARMED.try_with(|a| a.get()).unwrap_or(false);
    if armed {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

/// Run `f` and return the number of allocation events it made on this
/// thread.
fn count_allocs<R>(f: impl FnOnce() -> R) -> u64 {
    ALLOCS.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    let out = f();
    ARMED.with(|a| a.set(false));
    let n = ALLOCS.with(|c| c.get());
    drop(out);
    n
}

const DIM: usize = 64;
const BITS: usize = 4;

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

/// An index whose search caches — including the SIMD-blocked layout —
/// are cold, so `prepare` performs the full packed→blocked repack.
fn cold_index(n: usize) -> TurboQuantIndex {
    let mut built = TurboQuantIndex::new(DIM, BITS).unwrap();
    built.add(&unit_vectors(n, DIM, 7));
    TurboQuantIndex::from_parts(
        built.dim_opt(),
        built.bit_width(),
        built.len(),
        built.packed_codes().to_vec(),
        built.scales().to_vec(),
        built.tqplus_shift().to_vec(),
        built.tqplus_scale().to_vec(),
    )
    .expect("parts from a real index are consistent")
}

/// Building the blocked layout must not allocate per vector.
///
/// `extract_codes_flat` used to return `Vec<Vec<u8>>` — one heap
/// allocation per vector plus the outer pointer vector — so this count
/// grew linearly with the index length. With a single flat buffer the
/// count is identical for a small and a large index: the same fixed set
/// of rotation, codebook and layout buffers either way.
#[test]
fn repack_allocation_count_does_not_scale_with_vector_count() {
    let small = cold_index(64);
    let large = cold_index(4096);

    let a = count_allocs(|| small.prepare());
    let b = count_allocs(|| large.prepare());
    println!("prepare allocations: 64 vectors = {a}, 4096 vectors = {b}");

    assert_eq!(
        a, b,
        "warming a 4096-vector index allocated {b} times against {a} for a \
         64-vector one — the blocked repack is allocating per vector again"
    );
    // 4096 vectors would put the per-vector version above 4000; anything
    // in the low tens is the fixed rotation/codebook/layout set.
    assert!(
        b < 100,
        "prepare made {b} allocations, expected a small fixed number"
    );
}
