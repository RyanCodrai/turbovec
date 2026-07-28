//! Golden pin for the Lloyd-Max codebook.
//!
//! The codebook is baked into every encoded vector: `boundaries` decides
//! each coordinate's code and `centroids` sets the reconstruction the
//! stored per-vector scale is computed against. It is *not* a frozen
//! table — it is recomputed at runtime from `statrs`'s Beta cdf/pdf,
//! which bottom out in `ln`/`exp`.
//!
//! Two different things can move it, and they need two different
//! defences:
//!
//! * **Across platforms**, for one build — different libms, different
//!   rounding. That is what the cross-OS fingerprint CI leg checks, and
//!   `codebook::lloyd_max` now derives boundaries from the already-
//!   rounded f32 centroids so the fragile step is gone.
//! * **Across builds**, on one platform — a `statrs` release that
//!   changes its special functions by an ulp, or a libm update. The
//!   fingerprint leg cannot see this: it compares three OSes *within a
//!   single locked build*, so a drift that moves every platform together
//!   passes it silently. `rand_chacha` is exact-pinned (`=0.3.1`)
//!   precisely because the rotation's byte stream is format-frozen;
//!   `statrs` is a caret range, so this test is the equivalent guard for
//!   the codebook.
//!
//! The constants below are not local observations — they are the values
//! Linux, macOS and Windows independently agreed on in the
//! `Encode fingerprint` CI legs, cross-checked against
//! `examples/encode_hash`, which uses this same hash over these same
//! cells.
//!
//! If this test fails, encoded bytes have moved for identical input.
//! That is a format change, not a rounding nit: decide it deliberately
//! and re-freeze, don't paper over it.

/// A minimal populated index for `dim`/`bits`.
///
/// The codebook is a pure function of `(bit_width, dim)` — the data does
/// not enter it — but `codebook_for_write` returns all-zero placeholders
/// for an *empty* index, so the index has to have something in it or the
/// assertions below hash zeros and pass vacuously. (They did, on the
/// first run of this test.) `assert_not_placeholder` is the standing
/// guard against that.
fn populated(dim: usize, bits: usize) -> turbovec::TurboQuantIndex {
    let mut index = turbovec::TurboQuantIndex::new(dim, bits).unwrap();
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ dim as u64;
    let vectors: Vec<f32> = (0..8 * dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 32) as u32 as f64 / 2_147_483_648.0 - 1.0) as f32
        })
        .collect();
    index.add(&vectors);
    index
}

/// Fail loudly on the placeholder codebook rather than hashing it.
fn assert_not_placeholder(boundaries: &[f32], centroids: &[f32], dim: usize, bits: usize) {
    assert!(
        centroids.iter().any(|c| *c != 0.0) && boundaries.iter().any(|b| *b != 0.0),
        "dim={dim} bits={bits}: got the all-zero placeholder codebook — the \
         index under test is empty, so this assertion would pass vacuously",
    );
}

/// FNV-1a 64 over the f32 bit patterns — byte-for-byte identical to
/// `examples/encode_hash`, so a failure here and a divergence there name
/// the same numbers.
fn fnv1a_f32(values: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for &b in v.to_bits().to_le_bytes().iter() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// `(dim, bit_width, boundaries_hash, centroids_hash)`.
///
/// Same six cells as `examples/encode_hash`: the rotation's block-size
/// regimes (`8·odd` → B=8, a pure power of two, the two production dims)
/// across a range of Beta shape parameters.
const GOLDEN: &[(usize, usize, u64, u64)] = &[
    (200, 2, 0x4fb0_378d_c1d8_6d75, 0xd43f_f871_2da1_9915),
    (768, 4, 0x2727_3d87_5c56_0161, 0x0970_4dad_5b65_a00d),
    (1024, 3, 0xd9ee_efae_66c3_4fd1, 0xfcf5_342e_5aef_b105),
    (1536, 2, 0xb1a9_7993_603c_7dcd, 0x1348_c9ae_60bb_dc3d),
    (1536, 4, 0x05f7_a92f_87ab_a9a1, 0x84f2_c25c_ecbf_6761),
    (3072, 4, 0x9a1b_3dca_8251_faad, 0x6a56_a173_8e12_e62d),
];

#[test]
fn codebook_bytes_are_frozen() {
    for &(dim, bits, want_boundaries, want_centroids) in GOLDEN {
        // Reached through the public surface: `codebook::codebook` is
        // crate-internal, but an index exposes the codebook it encoded
        // against, which is the thing that actually has to stay put.
        let index = populated(dim, bits);
        let (boundaries, centroids) = index.codebook_for_write();

        assert_eq!(boundaries.len(), (1 << bits) - 1);
        assert_eq!(centroids.len(), 1 << bits);
        assert_not_placeholder(&boundaries, &centroids, dim, bits);
        assert_eq!(
            fnv1a_f32(&centroids),
            want_centroids,
            "dim={dim} bits={bits}: Lloyd-Max centroids drifted. Every \
             stored scale is computed against these, so this re-encodes \
             every future vector differently from earlier builds. A \
             `statrs` upgrade or a libm change is the likely cause — the \
             cross-OS fingerprint leg cannot catch it, because it moves \
             all platforms together.",
        );
        assert_eq!(
            fnv1a_f32(&boundaries),
            want_boundaries,
            "dim={dim} bits={bits}: codebook boundaries drifted. These \
             decide every coordinate's code. See the centroid note above.",
        );
    }
}

/// The property that makes the codebook cross-platform reproducible:
/// boundaries are derived from the f32 centroids, not from the f64 ones.
///
/// Duplicated from the unit test in `codebook.rs` on purpose — this one
/// runs against the public surface, so it also covers the path an
/// embedder reaches through `codebook_for_write`.
#[test]
fn boundaries_are_midpoints_of_the_published_centroids() {
    for &(dim, bits, _, _) in GOLDEN {
        let index = populated(dim, bits);
        let (boundaries, centroids) = index.codebook_for_write();
        assert_not_placeholder(&boundaries, &centroids, dim, bits);
        for (i, b) in boundaries.iter().enumerate() {
            let expect = (centroids[i] + centroids[i + 1]) * 0.5;
            assert_eq!(
                b.to_bits(),
                expect.to_bits(),
                "dim={dim} bits={bits} boundary {i}",
            );
        }
    }
}
