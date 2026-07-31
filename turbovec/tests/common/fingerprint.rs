//! The encode fingerprint: one six-column hash row per `(dim, bits)` cell.
//!
//! This module is the single definition of the fingerprint, shared by its
//! two consumers so they can never drift apart:
//!
//! * `examples/encode_hash.rs` prints the rows. CI runs it on Linux,
//!   macOS and Windows and fails unless the three agree — that leg
//!   catches *platform-specific* divergence.
//! * `tests/encode_fingerprint.rs` compares the rows against frozen
//!   constants. That catches drift which moves every platform *together*
//!   — a `statrs` bump, a libm change, a retuned reduction order — which
//!   the cross-OS leg is structurally blind to (#346, #352).
//!
//! Both halves are needed and neither subsumes the other. Keeping the
//! fixture here rather than duplicating it means "the golden test pins
//! what the example prints" is enforced by the compiler instead of by a
//! comment.
//!
//! It lives under `tests/common/` (a subdirectory, so cargo does not
//! treat it as a test target of its own) and the example reaches it with
//! a `#[path]` module.

// The two consumers use different subsets: the example only formats
// rows, the golden test only reads columns. Neither is dead code from
// the crate's point of view, but each compilation unit sees the other's
// half as unused.
#![allow(dead_code)]

use turbovec::TurboQuantIndex;

/// Cells to fingerprint. Chosen to cover the block-size regimes the
/// rotation has and the Beta parameters the codebook is fit against:
/// `8·odd` dims collapse to a B=8 Hadamard block, powers of two use one
/// full-width block, and the two production dims sit in between.
///
/// The `8·odd` (B=8) regime is covered at *every* supported bit width,
/// not just one. It is the regime the rotation mixes least (#309), and
/// nothing about it is shared across bit widths downstream of the
/// rotation: the Lloyd-Max codebook is fit per `(bits, dim)`, the TQ+
/// calibration is fit against that codebook, and the packed layout gets
/// one bit-plane per bit. A drift confined to a weak-block dim at 3 or 4
/// bits was previously anchored by nothing — dims 768/1024/1536/3072 all
/// have B >= 256, and the one B=8 cell was 2-bit only (#352).
pub const CELLS: &[(usize, usize)] = &[
    (200, 2),   // 8·25  -> B = 8, low dim (loosest Beta asymptotics)
    (600, 3),   // 8·75  -> B = 8 at the odd bit width
    (768, 4),   // 256·3 -> B = 256
    (1000, 4),  // 8·125 -> B = 8 at 4 bits, the dim rotation.rs names
    (1024, 3),  // pure power of two -> B = 1024, odd bit width
    (1536, 2),  // 512·3 -> B = 512
    (1536, 4),
    (3072, 4),  // 1024·3 -> B = 1024
];

/// Enough vectors that the TQ+ calibration always fits (the identity
/// fallback below `TQPLUS_MIN_SAMPLES` would hide calibration drift).
pub const N: usize = 2_000;

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn fnv1a_f32(values: &[f32]) -> u64 {
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
pub fn lcg_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
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

/// Per-cell input seed. Varies per cell so two cells can't accidentally
/// agree. Part of the fixture: changing it invalidates every golden.
pub fn cell_seed(dim: usize, bits: usize) -> u64 {
    0x7B0_0000 ^ (dim as u64) << 8 ^ bits as u64
}

/// One hash per stage of the encode pipeline.
///
/// The stages fail independently, so they are hashed separately: a
/// divergence names the stage that drifted instead of just saying "the
/// bytes differ". That split is what localized the original #259 failure
/// to the boundary midpoints while everything downstream still agreed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fingerprint {
    pub boundaries: u64,
    pub centroids: u64,
    pub calibration: u64,
    pub codes: u64,
    pub scales: u64,
    pub file: u64,
}

impl Fingerprint {
    /// Columns in printed order, so a caller can diff them by name.
    pub fn columns(&self) -> [(&'static str, u64); 6] {
        [
            ("boundaries", self.boundaries),
            ("centroids", self.centroids),
            ("calibration", self.calibration),
            ("codes", self.codes),
            ("scales", self.scales),
            ("file", self.file),
        ]
    }

    /// The one-line form CI compares across operating systems.
    pub fn line(&self, dim: usize, bits: usize) -> String {
        format!(
            "dim={dim} bits={bits} boundaries={:016x} centroids={:016x} \
             calibration={:016x} codes={:016x} scales={:016x} file={:016x}",
            self.boundaries,
            self.centroids,
            self.calibration,
            self.codes,
            self.scales,
            self.file,
        )
    }

    /// The `GOLDEN` row literal for this cell, so a deliberate re-freeze
    /// is a copy-paste rather than a hand-transcription of six hex words.
    pub fn golden_literal(&self, dim: usize, bits: usize) -> String {
        format!(
            "    ({dim}, {bits}, 0x{:016x}, 0x{:016x}, 0x{:016x}, \
             0x{:016x}, 0x{:016x}, 0x{:016x}),",
            self.boundaries,
            self.centroids,
            self.calibration,
            self.codes,
            self.scales,
            self.file,
        )
    }
}

/// Encode the cell's fixture and hash every stage of the result.
///
/// Pure function of `(dim, bits)` given the frozen fixture above.
pub fn fingerprint(dim: usize, bits: usize) -> Fingerprint {
    let vectors = lcg_vectors(N, dim, cell_seed(dim, bits));
    let mut index = TurboQuantIndex::new(dim, bits).unwrap();
    index.add(&vectors);

    // Boundaries and centroids are hashed separately: they fail
    // independently. The centroids survive the f64 Lloyd-Max
    // iteration's libm variance because the f32 cast absorbs it,
    // while the midpoints between them used to sit on an f32
    // rounding knife-edge and diverged on all three platforms
    // (#259). Splitting the column is what made that diagnosable.
    let (boundaries, centroids) = index.codebook_for_write();

    // The all-zero placeholder codebook an *empty* index reports would
    // hash to a stable value and pass every assertion vacuously. The
    // index is populated above, so this is a standing guard, not a
    // live risk — but it is the failure mode that made the first
    // version of `codebook_determinism.rs` meaningless.
    assert!(
        centroids.iter().any(|c| *c != 0.0) && boundaries.iter().any(|b| *b != 0.0),
        "dim={dim} bits={bits}: got the all-zero placeholder codebook",
    );

    let mut calibration = index.tqplus_shift().to_vec();
    calibration.extend_from_slice(index.tqplus_scale());
    // An unfitted calibration is the identity (shift 0, scale 1) and
    // would pin nothing about `statrs`'s `inverse_cdf` — the exact drift
    // #346 is about. `N` is above `TQPLUS_MIN_SAMPLES`, so this must
    // have fitted.
    assert!(
        !calibration.is_empty()
            && (calibration.iter().any(|s| *s != 0.0 && *s != 1.0)),
        "dim={dim} bits={bits}: TQ+ calibration is empty or still the \
         identity — the calibration column would pin nothing",
    );

    Fingerprint {
        boundaries: fnv1a_f32(&boundaries),
        centroids: fnv1a_f32(&centroids),
        calibration: fnv1a_f32(&calibration),
        codes: fnv1a(index.packed_codes()),
        scales: fnv1a_f32(index.scales()),
        file: fnv1a(&index.to_bytes()),
    }
}
