//! Random orthogonal rotation matrix generation.
//!
//! Generates a deterministic orthogonal matrix via QR decomposition of
//! a seeded Gaussian random matrix. The rotation makes each coordinate
//! of a unit vector follow a known Beta distribution.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::ROTATION_SEED;

/// Generate a dim x dim orthogonal matrix (deterministic, seeded).
/// Returns row-major flat Vec<f32> of length dim*dim.
pub fn make_rotation_matrix(dim: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(ROTATION_SEED);

    // Generate random Gaussian matrix
    let mut g = faer::Mat::<f64>::zeros(dim, dim);
    for j in 0..dim {
        for i in 0..dim {
            g.write(i, j, rng.sample(StandardNormal));
        }
    }

    // Non-pivoted QR decomposition (deterministic)
    let qr = g.qr();
    let q_full = qr.compute_thin_q();
    let r = qr.compute_thin_r();

    // Sign correction: Q = Q * diag(sign(diag(R)))
    let mut q = q_full;
    for j in 0..dim {
        let sign = if r.read(j, j) >= 0.0 { 1.0 } else { -1.0 };
        if sign < 0.0 {
            for i in 0..dim {
                q.write(i, j, q.read(i, j) * sign);
            }
        }
    }

    // Convert to row-major f32
    let mut result = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[i * dim + j] = q.read(i, j) as f32;
        }
    }

    result
}

/// Number of rotation probe values stored in a v4 file header.
pub(crate) const N_PROBES: usize = 64;

/// Absolute tolerance for comparing stored rotation probes against the
/// rebuilt rotation.
///
/// The rebuilt matrix is not bit-identical across environments: faer's
/// QR gives results that differ with thread count and CPU architecture.
/// Measured on this codebase (aarch64 vs x86_64, 1 vs N rayon threads,
/// dim 1536/3072): 1-10 elements out of millions differ, each by
/// exactly 1 f32 ulp (max abs diff 1.2e-10). Real rotation drift — a
/// changed RNG stream, QR algorithm, or sign convention — perturbs
/// essentially every element at the ~1e-2 scale (element magnitudes are
/// ~1/√dim). 1e-4 sits >5 orders of magnitude above the benign noise
/// and ~2 below genuine drift, so 64 probes make misclassification in
/// either direction vanishingly unlikely.
pub(crate) const PROBE_TOLERANCE: f32 = 1e-4;

/// Fingerprint of a rotation matrix as stored in a v4 `.tv`/`.tvim`
/// header: an exact FNV-1a hash plus [`N_PROBES`] sampled element
/// values.
///
/// The rotation is rebuilt from a seed at load time, and the stored
/// codes silently decode wrong (recall → ~0) if the rebuild ever
/// produces a different — but still valid — matrix (e.g. a faer QR or
/// rand_distr sampling change). The fingerprint lets the loader detect
/// that *rotation drift* and error cleanly instead:
///
/// * `hash` — FNV-1a (64-bit) over the exact f32 bit patterns
///   (little-endian byte order, row-major element order). Equality is
///   proof of a bit-identical rebuild — the common same-environment
///   case.
/// * `probes` — the f32 values at [`probe_positions`]. When the hash
///   differs, the probes distinguish benign cross-environment build
///   noise (see [`PROBE_TOLERANCE`]) from genuine drift.
///
/// FNV-1a is a stable, dependency-free, non-cryptographic hash — the
/// fingerprint defends against accidental drift, not adversaries (an
/// attacker who can rewrite the fingerprint can rewrite the codes too).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RotationFingerprint {
    pub hash: u64,
    pub probes: [f32; N_PROBES],
}

impl RotationFingerprint {
    /// The fingerprint stored for an index with no vectors: no rotation
    /// is associated with the file, so all fields are zero and the
    /// loader skips verification.
    pub fn empty() -> Self {
        Self { hash: 0, probes: [0.0; N_PROBES] }
    }

    /// Fingerprint the given `dim`×`dim` row-major rotation matrix.
    pub fn compute(rotation: &[f32], dim: usize) -> Self {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = FNV_OFFSET;
        for &v in rotation {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        let positions = probe_positions(dim);
        let mut probes = [0.0f32; N_PROBES];
        for (p, &pos) in probes.iter_mut().zip(positions.iter()) {
            *p = rotation[pos];
        }
        Self { hash: h, probes }
    }

    /// Does the rebuilt rotation match this stored fingerprint?
    ///
    /// Exact hash equality passes immediately; otherwise every probe
    /// must be within [`PROBE_TOLERANCE`] of the rebuilt value (NaN in
    /// a stored probe never matches).
    pub fn matches(&self, rebuilt: &Self) -> bool {
        if self.hash == rebuilt.hash {
            return true;
        }
        self.probes
            .iter()
            .zip(rebuilt.probes.iter())
            .all(|(&stored, &fresh)| (stored - fresh).abs() <= PROBE_TOLERANCE)
    }
}

/// Deterministic probe positions (indices into the row-major `dim*dim`
/// rotation) for a given `dim`.
///
/// Part of the on-disk v4 format contract: files store the rotation
/// values at these positions, so this sequence must never change. It is
/// a fixed 64-bit LCG (Knuth's MMIX multiplier) seeded from `dim`,
/// taking the high 31 bits of each step modulo `dim*dim`. Positions are
/// spread across the whole matrix so drift confined to any region (e.g.
/// only late Householder columns) still lands on probes.
pub(crate) fn probe_positions(dim: usize) -> [usize; N_PROBES] {
    debug_assert!(dim > 0);
    let n = (dim as u64) * (dim as u64);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ (dim as u64);
    let mut out = [0usize; N_PROBES];
    for slot in &mut out {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *slot = ((state >> 33) % n) as usize;
    }
    out
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;

    #[test]
    fn probe_positions_are_a_stable_format_contract() {
        // These exact values are baked into every v4 file ever written
        // (the stored probes are the rotation elements at these
        // positions). If this test fails, the change breaks loading of
        // existing v4 files — do not update the expectations; revert
        // the change to `probe_positions`.
        // Expectations independently recomputed with a Python
        // implementation of the same LCG.
        let p32 = probe_positions(32);
        assert_eq!(&p32[..4], &[485, 198, 916, 474]);
        let p8 = probe_positions(8);
        assert_eq!(&p8[..4], &[6, 46, 0, 47]);
        // All positions in range for a selection of dims.
        for dim in [8usize, 32, 768, 16384] {
            for &p in probe_positions(dim).iter() {
                assert!(p < dim * dim);
            }
        }
    }

    #[test]
    fn fingerprint_is_deterministic_and_sensitive() {
        let rot = make_rotation_matrix(16);
        let a = RotationFingerprint::compute(&rot, 16);
        let b = RotationFingerprint::compute(&rot, 16);
        assert_eq!(a, b);

        // A 1-ulp style perturbation of a single element flips the hash
        // but stays within probe tolerance → still matches.
        let mut ulp = rot.clone();
        ulp[7] = f32::from_bits(ulp[7].to_bits() ^ 1);
        let c = RotationFingerprint::compute(&ulp, 16);
        assert_ne!(a.hash, c.hash);
        assert!(a.matches(&c), "ulp-level noise must be tolerated");

        // A structurally different rotation (sign convention change on
        // every column) must be rejected.
        let flipped: Vec<f32> = rot.iter().map(|v| -v).collect();
        let d = RotationFingerprint::compute(&flipped, 16);
        assert!(!a.matches(&d), "sign-flip drift must be detected");
    }
}
