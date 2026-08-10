//! The invalid-coordinate scan reports *which* coordinate was bad, and
//! that report is part of the panic contract callers read.
//!
//! Existing tests only assert that bad input is rejected, so every
//! arithmetic step that turns a flat offset into `(vector, coord)` —
//! the chunk base `ci * VALIDATE_CHUNK + j`, the `first / dim` and
//! `first % dim` split, and the leftmost-wins reduction — was untested,
//! and the magnitude predicate was never exercised at its boundary
//! (#463).

use turbovec::TurboQuantIndex;

const DIM: usize = 64;
/// Mirrors `encode::VALIDATE_CHUNK`, which is `pub(crate)`. A drift
/// shows up as the past-the-first-chunk case below losing its point,
/// which is why it asserts its own premise.
const VALIDATE_CHUNK: usize = 64 * 1024;

fn clean(n: usize) -> Vec<f32> {
    (0..n * DIM).map(|i| ((i % 7) as f32 * 0.1) - 0.3).collect()
}

/// Add `vals` and return the panic message.
fn add_panic(vals: Vec<f32>) -> String {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(move || {
        let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
        idx.add(&vals);
    });
    std::panic::set_hook(prev);
    let e = r.expect_err("invalid input must panic");
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("panic payload should be a string")
}

#[test]
fn the_reported_vector_and_coord_locate_the_bad_value() {
    // Several positions, including a coord that is not 0 and a vector
    // that is not 0, so a `/` vs `%` swap or a dropped divisor shows.
    for (vec_i, coord_i) in [(0usize, 0usize), (0, 5), (3, 0), (7, 63), (11, 40)] {
        let mut v = clean(16);
        v[vec_i * DIM + coord_i] = f32::NAN;
        let msg = add_panic(v);
        assert!(
            msg.contains(&format!("at vector {vec_i}, coord {coord_i}")),
            "position ({vec_i}, {coord_i}) misreported: {msg}"
        );
    }
}

/// The scan runs in parallel over fixed chunks and reduces by minimum
/// flat index, so a bad value past the first chunk exercises the chunk
/// base arithmetic that a single-chunk input never touches.
#[test]
fn a_bad_value_past_the_first_chunk_is_located_correctly() {
    let n_vectors = (VALIDATE_CHUNK / DIM) + 40;
    assert!(
        n_vectors * DIM > VALIDATE_CHUNK,
        "this case only bites when the input spans more than one chunk"
    );
    let vec_i = n_vectors - 7;
    let coord_i = 17;
    let flat = vec_i * DIM + coord_i;
    assert!(flat > VALIDATE_CHUNK, "the bad value must land past chunk 0");

    let mut v = clean(n_vectors);
    v[flat] = f32::INFINITY;
    let msg = add_panic(v);
    assert!(
        msg.contains(&format!("at vector {vec_i}, coord {coord_i}")),
        "past-chunk position misreported: {msg}"
    );
}

/// Two bad values: the earlier one wins, whichever chunk each lands in.
#[test]
fn the_leftmost_invalid_value_is_the_one_reported() {
    let n_vectors = (VALIDATE_CHUNK / DIM) + 40;
    let early = (3usize, 9usize);
    let late = (n_vectors - 2, 50usize);
    let mut v = clean(n_vectors);
    v[early.0 * DIM + early.1] = f32::NAN;
    v[late.0 * DIM + late.1] = f32::NAN;
    let msg = add_panic(v);
    assert!(
        msg.contains(&format!("at vector {}, coord {}", early.0, early.1)),
        "the leftmost invalid value must be reported, got: {msg}"
    );
}

/// The predicate is `!(|x| < max)`, so the bound itself is invalid and
/// the value just below it is fine. Nothing pinned that, which left
/// `<` free to become `<=`, `==` or `>`.
#[test]
fn the_magnitude_bound_is_exclusive() {
    const MAX: f32 = 1e16;
    let mut at_bound = clean(4);
    at_bound[2 * DIM + 1] = MAX;
    let msg = add_panic(at_bound);
    assert!(
        msg.contains("at vector 2, coord 1"),
        "|value| == 1e16 must be rejected: {msg}"
    );

    // Just inside: the largest f32 strictly below the bound.
    let mut inside = clean(4);
    inside[2 * DIM + 1] = f32::from_bits(MAX.to_bits() - 1);
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add(&inside);
    assert_eq!(idx.len(), 4, "a value just below the bound must be accepted");
}

/// Every non-finite shape is rejected, and negatives are judged on
/// magnitude — a dropped `!` or an `abs()` that stopped mattering would
/// let one of these through.
#[test]
fn every_non_finite_and_large_negative_is_rejected() {
    for (label, bad) in [
        ("NaN", f32::NAN),
        ("+inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
        ("-1e16", -1e16f32),
        ("-1e20", -1e20f32),
    ] {
        let mut v = clean(4);
        v[1 * DIM + 3] = bad;
        let msg = add_panic(v);
        assert!(
            msg.contains("at vector 1, coord 3"),
            "{label} must be rejected and located: {msg}"
        );
    }
}
