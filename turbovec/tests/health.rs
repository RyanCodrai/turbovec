//! `health()`: live searchable bytes over allocated bytes.
//!
//! Covers the four states the design asks for — fresh, thinned, drained,
//! all-empty — plus the two things a naive test misses: that rows stored
//! with a degenerate scale are counted as overhead, and that the
//! all-empty case does not divide by zero.
//!
//! Expected values are computed from the definition rather than pasted
//! from a run, so a change to what counts as overhead fails here instead
//! of being absorbed.

use turbovec::TurboQuantIndex;

const DIM: usize = 256;
const BLOCK: usize = 8192;

fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut out = vec![0.0f32; n * dim];
    for row in out.chunks_mut(dim) {
        let mut nrm = 0.0f64;
        for x in row.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((s >> 40) as f64 / (1u64 << 23) as f64) - 1.0;
            *x = v as f32;
            nrm += v * v;
        }
        let inv = 1.0 / (nrm.sqrt() + 1e-12);
        for x in row.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
    out
}

/// The definition, restated: live searchable rows' code bytes over every
/// byte the index holds for its rows, including one `(shift, scale)`
/// pair per block.
fn expected(live: usize, slots: usize, pairs: usize, dim: usize, bits: usize) -> f32 {
    let bytes_per_row = dim * bits / 8;
    let calib = pairs * 2 * dim * 4;
    (live * bytes_per_row) as f32 / (slots * bytes_per_row + calib) as f32
}

#[test]
fn a_fresh_index_is_all_payload_bar_its_calibration() {
    // The design's headline figure: a fresh index at an 8192-row block
    // size and 2 bits reads ~99.6%. The overhead is 2*dim floats per
    // block against 8192*dim/4 bytes of codes — 8/2048, i.e. 0.39%,
    // whatever the dim.
    let n = 100_000;
    let mut ix = TurboQuantIndex::new(DIM, 2).unwrap();
    ix.add(&rows(n, DIM, 7));

    let blocks = ix.sealed_blocks();
    assert_eq!(blocks, n / BLOCK);
    // One pair per sealed block, plus the open block's own.
    let want = expected(n, n, blocks + 1, DIM, 2);
    let got = ix.health();
    assert!(
        (got - want).abs() < 1e-6,
        "health {got} does not match the definition ({want})",
    );
    assert!(
        (got - 0.996).abs() < 0.001,
        "a fresh 8192/2-bit index should read ~99.6%, got {got}",
    );
}

#[test]
fn deleting_three_quarters_of_the_rows_shows_up_as_dead_weight() {
    // Nothing compacts across blocks — that would need the original
    // floats — so a deleted row keeps its bytes and health is what
    // surfaces the fact. This is the number the design predicts as
    // 98.5%; see the note at the bottom of this file for why it is
    // ~25% instead, and why 25% is the value consistent with the
    // design's own stated role for the metric.
    let n = 100_000;
    let mut ix = TurboQuantIndex::new(DIM, 2).unwrap();
    ix.add(&rows(n, DIM, 11));
    let fresh = ix.health();

    let mut removed = 0usize;
    let mut slot = 0usize;
    while removed < n * 3 / 4 {
        if slot >= ix.slot_capacity() {
            slot = 0;
        }
        if ix.slot_is_live(slot) {
            ix.swap_remove(slot);
            removed += 1;
        }
        slot += 4;
    }

    assert_eq!(ix.len(), n / 4);
    let got = ix.health();
    let want = expected(
        ix.len(),
        ix.slot_capacity(),
        ix.sealed_blocks() + 1,
        DIM,
        2,
    );
    assert!(
        (got - want).abs() < 1e-6,
        "health {got} does not match the definition ({want})",
    );
    assert!(
        got < fresh / 3.0,
        "75% of the rows are dead but health only moved {fresh} -> {got}",
    );
    assert!(
        (0.20..0.30).contains(&got),
        "a quarter of the rows live should read about a quarter, got {got}",
    );
}

#[test]
fn a_drained_index_reports_no_payload_and_an_empty_one_reports_no_waste() {
    // Two different zeroes, and they must not be confused. A drained
    // index still holds every slot it allocated, so none of what it
    // holds is payload. An index that never held anything has nothing
    // to waste — and is the case where a ratio divides by zero.
    let n = 3 * BLOCK;
    let mut drained = TurboQuantIndex::new(DIM, 2).unwrap();
    drained.add(&rows(n, DIM, 13));
    while drained.len() > 0 {
        let mut s = 0;
        while !drained.slot_is_live(s) {
            s += 1;
        }
        drained.swap_remove(s);
    }
    assert_eq!(drained.len(), 0);
    assert!(
        drained.slot_capacity() > 0,
        "the sealed blocks should still hold their slots",
    );
    assert_eq!(drained.health(), 0.0, "a drained index holds no payload");

    let empty = TurboQuantIndex::new(DIM, 2).unwrap();
    assert_eq!(empty.slot_capacity(), 0);
    let h = empty.health();
    assert!(h.is_finite(), "an index with nothing allocated divided by zero: {h}");
    assert_eq!(h, 1.0, "nothing allocated is nothing wasted");

    // Same for a lazy index, which has no dim to compute a row size with.
    let lazy = TurboQuantIndex::new_lazy(2).unwrap();
    assert_eq!(lazy.health(), 1.0);
}

#[test]
fn rows_that_no_search_can_return_are_overhead() {
    // A vector with no representable direction is stored with scale 0,
    // scores 0 against every query, and is returned only after every row
    // that does have a direction. It occupies its full code budget
    // regardless, so it is overhead — and a health function that only
    // counted dead *slots* would miss it entirely, since these rows are
    // live by every other measure.
    let n = 4 * BLOCK;
    let mut data = rows(n, DIM, 17);
    for x in data[..(n / 2) * DIM].iter_mut() {
        *x = 0.0;
    }
    let mut ix = TurboQuantIndex::new(DIM, 2).unwrap();
    ix.add(&data);

    assert_eq!(ix.len(), n, "degenerate rows are still stored and counted");
    assert_eq!(ix.slot_capacity(), n, "and still occupy their slots");
    let got = ix.health();
    let want = expected(n / 2, n, ix.sealed_blocks() + 1, DIM, 2);
    assert!(
        (got - want).abs() < 1e-6,
        "health {got} counts unsearchable rows as payload (definition says {want})",
    );
    assert!(
        got < 0.55,
        "half the rows score against nothing, so at most half is payload, got {got}",
    );
}

// ─── on the design's 98.5% ───────────────────────────────────────────────
//
// design.md predicts "75% uniform deletion gives 98.5%". The shipped
// number is ~25.2%, and the difference is not a rounding disagreement.
//
// 98.5% is what the arithmetic gives if a deleted row frees its bytes:
// 25k rows of codes against 25k rows of codes plus 13 blocks of
// calibration is 0.9836. The shipped model cannot free them — a sealed
// block keeps its extent, because shortening it would renumber every
// later slot — so the denominator stays at all 100k rows and the ratio
// is the live fraction, 0.252.
//
// The design's own table settles which is right. It lists cross-block
// compaction as an accepted limitation "surfaced via health". A metric
// that reads 98.5% while three quarters of the index's bytes are dead
// is not surfacing that limitation; it is hiding it. So 25% is the
// number consistent with what the design says health is *for*, and the
// 98.5% estimate looks like it was computed against a compacting model
// that was considered and rejected in the same document.
