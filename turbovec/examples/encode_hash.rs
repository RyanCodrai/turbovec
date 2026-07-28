//! Cross-platform encode fingerprint.
//!
//! Prints one line per (dim, bit_width) cell, each a hash of a stage of
//! the encode pipeline for a fixed, deterministic input. Two platforms
//! that agree on every line encode identical bytes for identical
//! vectors; a platform that differs says *which stage* diverged.
//!
//! ```text
//! cargo run --release --example encode_hash
//! ```
//!
//! This is the observable half of the v5/v6 determinism claim (#259).
//! The rotation is deterministic by construction (integer permutations
//! plus f32 add/sub/scale, no FMA, golden-pinned) and the per-vector
//! norm now has a frozen reduction order, but two inputs to the encode
//! can still only be *checked* across platforms, never proved on one
//! machine:
//!
//! 1. the Lloyd-Max codebook, computed at runtime from `statrs` Beta
//!    cdf/pdf — transcendentals, so cross-libm variance is possible;
//! 2. everything downstream of it.
//!
//! Hence the split: `boundaries`, `centroids`, `calibration`, `codes`,
//! `scales`, and `file` are hashed separately, so a divergence localizes
//! instead of just saying "the bytes differ". That split is what caught
//! the codebook boundaries diverging on all three platforms while the
//! centroids agreed — see `codebook::lloyd_max`.
//!
//! The hash is FNV-1a 64, not SHA-256: this compares outputs of the same
//! code across platforms, so it needs collision resistance against
//! accident, not against an adversary — and a cryptographic hash would
//! mean a new dependency for a diagnostic.

use turbovec::TurboQuantIndex;

/// Cells to fingerprint. Chosen to cover the block-size regimes the
/// rotation has and the Beta parameters the codebook is fit against:
/// `8·odd` dims collapse to a B=8 Hadamard block, powers of two use one
/// full-width block, and the two production dims sit in between.
const CELLS: &[(usize, usize)] = &[
    (200, 2),   // 8·25  -> B = 8, low dim (loosest Beta asymptotics)
    (768, 4),   // 256·3 -> B = 256
    (1024, 3),  // pure power of two -> B = 1024, odd bit width
    (1536, 2),
    (1536, 4),
    (3072, 4),
];

/// Enough vectors that the TQ+ calibration always fits (the identity
/// fallback below `TQPLUS_MIN_SAMPLES` would hide calibration drift).
const N: usize = 2_000;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fnv1a_f32(values: &[f32]) -> u64 {
    // Hash the bit patterns, not the printed values: a last-ulp
    // difference is exactly what this is looking for.
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fnv1a(&bytes)
}

/// Deterministic input vectors — a plain LCG so the fixture is identical
/// on every platform (no float RNG, no transcendentals).
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

fn main() {
    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "other"
    };
    eprintln!("# arch={arch} os={} n={N}", std::env::consts::OS);

    for &(dim, bits) in CELLS {
        // Seed varies per cell so two cells can't accidentally agree.
        let vectors = lcg_vectors(N, dim, 0x7B0_0000 ^ (dim as u64) << 8 ^ bits as u64);
        let mut index = TurboQuantIndex::new(dim, bits).unwrap();
        index.add(&vectors);

        // Boundaries and centroids are hashed separately: they fail
        // independently. The centroids survive the f64 Lloyd-Max
        // iteration's libm variance because the f32 cast absorbs it,
        // while the midpoints between them used to sit on an f32
        // rounding knife-edge and diverged on all three platforms
        // (#259). Splitting the column is what made that diagnosable.
        let (boundaries, centroids) = index.codebook_for_write();

        let mut calibration = index.tqplus_shift().to_vec();
        calibration.extend_from_slice(index.tqplus_scale());

        // One line per cell, one column per stage, so a diverging
        // platform names the stage rather than just "the bytes differ".
        println!(
            "dim={dim} bits={bits} boundaries={:016x} centroids={:016x} \
             calibration={:016x} codes={:016x} scales={:016x} file={:016x}",
            fnv1a_f32(&boundaries),
            fnv1a_f32(&centroids),
            fnv1a_f32(&calibration),
            fnv1a(index.packed_codes()),
            fnv1a_f32(index.scales()),
            fnv1a(&index.to_bytes()),
        );
    }
}
