//! Determinism guarantees for the v5 block-Hadamard rotation and encode.
//!
//! These are the properties the QR rotation lacked (#206): the encoded
//! bytes must not depend on the ambient rayon thread count, and the exact
//! ChaCha8 stream that drives the rotation must be pinned so a future
//! `rand_chacha` release cannot silently change the wire format.

use std::sync::Arc;

use turbovec::rotation::Rotation;
use turbovec::TurboQuantIndex;

fn lcg_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n * dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 32) as u32 as f64 / 2_147_483_648.0 - 1.0) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Golden: pin the ChaCha8 stream / transform for fixed inputs.
// ---------------------------------------------------------------------------

/// Apply the rotation to `input` and return the raw f32 bit patterns.
fn rotate_bits(dim: usize, input: &[f32]) -> Vec<u32> {
    let mut v = input.to_vec();
    Rotation::new(dim).apply(&mut v);
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn golden_rotation_bits_pin_the_chacha_stream() {
    // dim = 8: a single Hadamard block, exercising the round-1/round-2
    // sign flips + intra-block butterfly + the global permutation.
    let input8: Vec<f32> = (0..8).map(|i| (i as f32) - 3.5).collect();
    assert_eq!(
        rotate_bits(8, &input8),
        GOLDEN_DIM8,
        "dim=8 rotation output changed — a rand_chacha stream change or a \
         transform edit would break every v5-encoded index. If this change \
         is intentional, it is a NEW format version, not an edit to v5.",
    );

    // dim = 24 = 8·3: three Hadamard blocks with a global inter-block
    // permutation, so a change to the permutation draw is caught here.
    let input24: Vec<f32> = (0..24).map(|i| ((i * 7 % 11) as f32) - 5.0).collect();
    assert_eq!(
        rotate_bits(24, &input24),
        GOLDEN_DIM24,
        "dim=24 rotation output changed — see the dim=8 note above.",
    );
}

// Golden f32 bit patterns — DO NOT CHANGE without bumping the format
// version. Generated on first run and pinned; the rotation is scalar and
// reduction-free, so these are bit-identical across platforms and thread
// counts.
const GOLDEN_DIM8: [u32; 8] = [
    1076887551, 1048575995, 1067450366, 3214934016, 1071644673, 3222274047, 3208642559,
    3231186944,
];
const GOLDEN_DIM24: [u32; 24] = [
    1071644671, 3217031167, 3232235520, 3230138368, 3224371200, 3227516928, 3224371200,
    3221225470, 1086324737, 1082654720, 3223322622, 3214934016, 1085800447, 1048576003,
    1067450368, 3214934016, 3208642556, 3221225470, 1048575995, 1076887552, 3224371200,
    3217031167, 3226468352, 1086324735,
];

// ---------------------------------------------------------------------------
// Thread-count determinism (the property v4 failed).
// ---------------------------------------------------------------------------

fn encode_in_pool(threads: usize, vectors: Arc<Vec<f32>>, dim: usize) -> (Vec<u8>, Vec<f32>) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.add(&vectors);
        (idx.packed_codes().to_vec(), idx.scales().to_vec())
    })
}

#[test]
fn encoded_bytes_identical_across_thread_counts() {
    // Large enough batch that rayon actually splits rows across the pool
    // and TQ+ calibration fits (>= TQPLUS_MIN_SAMPLES).
    let dim = 256;
    let vectors = Arc::new(lcg_vectors(2000, dim, 0xBADC0DE));

    let (codes1, scales1) = encode_in_pool(1, vectors.clone(), dim);
    for threads in [2usize, 4, 8] {
        let (codes_n, scales_n) = encode_in_pool(threads, vectors.clone(), dim);
        assert_eq!(
            codes1, codes_n,
            "packed codes differ between 1 and {threads} rayon threads — \
             the encode path is not thread-count-deterministic",
        );
        // Scales are computed per row and reduced in a fixed order, so
        // they must match bit-for-bit too.
        let bits1: Vec<u32> = scales1.iter().map(|s| s.to_bits()).collect();
        let bits_n: Vec<u32> = scales_n.iter().map(|s| s.to_bits()).collect();
        assert_eq!(
            bits1, bits_n,
            "per-vector scales differ between 1 and {threads} rayon threads",
        );
    }
}

#[test]
fn search_results_identical_across_thread_counts() {
    let dim = 256;
    let vectors = lcg_vectors(1500, dim, 0x5EED);
    let queries = lcg_vectors(8, dim, 0xF00D);

    let run = |threads: usize| -> (Vec<f32>, Vec<i64>) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
            idx.add(&vectors);
            let r = idx.search(&queries, 10);
            (r.scores, r.indices)
        })
    };

    let (s1, i1) = run(1);
    for threads in [2usize, 4, 8] {
        let (sn, i_n) = run(threads);
        assert_eq!(i1, i_n, "search indices differ at {threads} threads");
        let b1: Vec<u32> = s1.iter().map(|s| s.to_bits()).collect();
        let bn: Vec<u32> = sn.iter().map(|s| s.to_bits()).collect();
        assert_eq!(b1, bn, "search scores differ at {threads} threads");
    }
}
