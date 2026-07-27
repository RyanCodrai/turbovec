//! Deterministic orthogonal rotation via a globally-permuted
//! block-Hadamard transform.
//!
//! The rotation decorrelates coordinates so that each coordinate of a
//! unit vector follows the near-Gaussian marginal the Lloyd-Max codebook
//! is fit against. Unlike the dense QR-of-a-Gaussian rotation it replaces
//! (turbovec ≤ 0.9.0), this transform is **deterministic bit-for-bit**
//! across platforms, CPU architectures, and thread counts:
//!
//! * There is no matrix and no GEMM — the transform is applied in place
//!   to each row as sign flips, in-block Walsh-Hadamard butterflies, and
//!   integer permutations. Every arithmetic op is a plain `+`, `-`, or a
//!   single `*` by a fixed constant; the reduction (add) order is fixed
//!   and no fused-multiply-add is used, so a SIMD implementation would be
//!   obligated to match the scalar result exactly.
//! * The sign flips and permutations are drawn from ChaCha8, whose byte
//!   stream is a pure function of the seed — identical on every target.
//!
//! This closes issue #206: the old QR rotation read the global rayon
//! parallelism and used `faer`'s order-dependent parallel Householder
//! reduction plus a transcendental Ziggurat sampler, so its output
//! changed with `RAYON_NUM_THREADS` (dim ≥ 1536) and between libm
//! implementations (dim ≥ 3072). It also dispatched the rotate GEMM to a
//! per-OS BLAS backend, so the *encoded bytes* differed by platform. The
//! block-Hadamard transform removes all three causes by construction and
//! drops the 42 MB OpenBLAS dependency.
//!
//! # The transform (frozen wire-format invariant)
//!
//! Let `B` be the largest power-of-two divisor of `dim` (always ≥ 8,
//! since `dim` is a positive multiple of 8). One *round* is, in order:
//!
//! 1. a **global** ChaCha8-seeded Fisher-Yates permutation across all
//!    `dim` coordinates,
//! 2. a ChaCha8-seeded ±1 sign flip of every coordinate, and
//! 3. a normalized Walsh-Hadamard transform (× `1/√B`) applied
//!    independently to each contiguous `B`-coordinate block.
//!
//! The rotation is [`K`] = 2 rounds.
//!
//! **The permutation comes first, before every Hadamard.** This makes the
//! transform *order-invariant*: a `B`-block is never formed from
//! contiguous input coordinates. Importance-ordered embeddings —
//! matryoshka/MRL (e.g. OpenAI text-embedding-3, Nomic) and PCA-projected
//! vectors, whose energy decays monotonically with coordinate index — put
//! similar-energy coordinates next to each other, so a block built from
//! contiguous coordinates would group highly-correlated coordinates that
//! the small Walsh-Hadamard cannot decorrelate. At weak-block dims (`B =
//! 8`, i.e. `8·odd`) this measurably regressed recall versus the QR
//! rotation until the leading permutation was added; it scatters those
//! coordinates across blocks so the result no longer depends on the input
//! coordinate ordering. The global permutation between rounds also makes
//! two rounds mix across block boundaries — a single round leaves the
//! blocks independent and regresses recall; two rounds are statistically
//! indistinguishable from the old QR rotation's recall.
//!
//! Each of permutation, sign flip, and normalized Hadamard is orthogonal,
//! so their composition is orthogonal: the transform preserves L2 norm
//! (to f32 rounding) and its inverse is its transpose.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Number of block-Hadamard rounds.
///
/// DO NOT CHANGE — baked into every encoded vector. Two globally-permuted
/// rounds are the minimum that mixes across block boundaries; the value is
/// part of the v5 on-disk format contract, not a tunable.
pub const K: usize = 2;

/// ChaCha8 seed for the sign flips and permutations.
///
/// DO NOT CHANGE — baked into every encoded vector. The entire rotation is
/// a pure function of this seed and `dim`; changing it silently
/// invalidates every index ever written under the v5 format.
///
/// These 32 bytes are the `rand_core` 0.6 `seed_from_u64(42)` expansion,
/// frozen here as a literal so the wire format depends only on
/// `rand_chacha` (exact-pinned `=0.3.1`) and not on `rand_core`'s
/// unpinned seed-expansion algorithm. The golden-bytes tests in
/// `tests/rotation_determinism.rs` pin the resulting stream.
const ROTATION_SEED: [u8; 32] = [
    164, 143, 161, 123, 88, 50, 61, 10, 234, 184, 161, 204, 105, 1, 20, 184, 43, 140, 200,
    117, 24, 180, 247, 84, 141, 68, 110, 161, 228, 223, 32, 242,
];

/// Largest power-of-two divisor of `dim`.
///
/// `dim` is always a positive multiple of 8, so this is ≥ 8 and a power
/// of two. For a pure power-of-two `dim` it equals `dim` (one block); for
/// `8·odd` (e.g. 1000, 200) it collapses to 8.
fn block_size(dim: usize) -> usize {
    debug_assert!(dim > 0 && dim % 8 == 0);
    dim & dim.wrapping_neg()
}

/// A deterministic orthogonal rotation for a fixed `dim`.
///
/// Holds the per-round sign vectors and permutations precomputed from
/// [`ROTATION_SEED`]. Construction is `O(K · dim)`; [`Self::apply`]
/// rotates one `dim`-length row in place in `O(K · dim · log B)`.
#[derive(Debug, Clone)]
pub struct Rotation {
    dim: usize,
    block: usize,
    inv_sqrt_block: f32,
    /// Per-round ±1 sign flips, `K` vectors each of length `dim`.
    signs: Vec<Vec<f32>>,
    /// Per-round global permutations, `K` permutations each of length
    /// `dim` (`perm[i]` is the source coordinate for output slot `i`).
    perms: Vec<Vec<u32>>,
}

impl Rotation {
    /// Build the rotation for `dim` (a positive multiple of 8).
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0 && dim % 8 == 0, "rotation dim must be a positive multiple of 8");
        let block = block_size(dim);
        let inv_sqrt_block = 1.0 / (block as f32).sqrt();

        // A single ChaCha8 stream drives the whole construction. The draw
        // order — for each round: `dim` sign draws, then a Fisher-Yates
        // permutation — is part of the frozen format contract.
        let mut rng = ChaCha8Rng::from_seed(ROTATION_SEED);
        let mut signs = Vec::with_capacity(K);
        let mut perms = Vec::with_capacity(K);
        for _ in 0..K {
            let sign_row: Vec<f32> = (0..dim)
                .map(|_| if rng.next_u32() & 1 == 1 { -1.0 } else { 1.0 })
                .collect();
            signs.push(sign_row);
            perms.push(fisher_yates(dim, &mut rng));
        }

        Self { dim, block, inv_sqrt_block, signs, perms }
    }

    /// Vector dimensionality this rotation is built for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Apply the rotation to a single `dim`-length row in place.
    ///
    /// Reduction-free and scalar: fixed add order, no FMA, no rayon. Two
    /// calls on equal input produce bit-identical output regardless of the
    /// ambient thread count — this is the property the QR rotation lacked
    /// (#206).
    ///
    /// Panics if `row.len() != dim`.
    pub fn apply(&self, row: &mut [f32]) {
        // Scratch for the permutation step (`out[i] = row[perm[i]]`).
        let mut scratch = vec![0.0f32; self.dim];
        self.apply_with_scratch(row, &mut scratch);
    }

    /// [`Self::apply`] with a caller-provided scratch buffer, for hot loops
    /// that rotate many rows (encode, query batches) — reusing one scratch
    /// per rayon worker avoids an allocation per row. The scratch is fully
    /// overwritten before it is read, so its prior contents never influence
    /// the output and both entry points produce bit-identical results.
    ///
    /// Panics if `row.len() != dim` or `scratch.len() != dim`.
    pub fn apply_with_scratch(&self, row: &mut [f32], scratch: &mut [f32]) {
        assert_eq!(row.len(), self.dim, "rotation input row must have length dim");
        assert_eq!(scratch.len(), self.dim, "rotation scratch must have length dim");
        let dim = self.dim;
        let block = self.block;

        // The two buffers ping-pong: each round's permutation gathers from
        // one buffer into the other, and the sign flip + Walsh-Hadamard run
        // in the destination — no copy back. K = 2 (even), so the final
        // round lands the result in `row` where callers expect it.
        const _: () = assert!(K % 2 == 0, "ping-pong ends in `row` only for even K");
        let (mut input, mut output): (&mut [f32], &mut [f32]) = (row, scratch);

        for round in 0..K {

            // 1. Global permutation FIRST, with the sign flip fused into
            //    the gather (`out[i] = in[perm[i]] * sign[i]` — the same
            //    multiply the separate pass performed, so values are
            //    bit-identical). A permutation precedes *every*
            //    Walsh-Hadamard, including round 1, so a B-block is never
            //    formed from contiguous input coordinates. This makes the
            //    transform order-invariant: importance-ordered embeddings
            //    (matryoshka/MRL, PCA — energy monotonically ordered by
            //    coordinate index) otherwise co-locate similar-energy
            //    coordinates in one small block, which the block-Hadamard
            //    cannot decorrelate, regressing recall at weak-block (B=8,
            //    i.e. `8·odd`) dims. Permuting first scatters those
            //    coordinates across blocks.
            let perm = &self.perms[round];
            let sign_row = &self.signs[round];
            for ((dst, &src), &s) in
                output.iter_mut().zip(perm.iter()).zip(sign_row.iter())
            {
                *dst = input[src as usize] * s;
            }

            // 3. Normalized Walsh-Hadamard per B-block. The butterfly is
            //    the unnormalized transform (adds/subtracts only, fixed
            //    order); the single `1/√B` scale at the end makes it
            //    orthonormal. `√B` may be inexact in f32 (e.g. √8), but the
            //    scale is a fixed constant so the result is still
            //    deterministic.
            //
            //    The SIMD path below performs exactly the same adds,
            //    subtracts, and multiplies on the same operand pairs —
            //    butterflies within a stage are independent, so lane-
            //    parallel execution cannot reassociate anything and the
            //    output stays bit-identical to the scalar path (pinned by
            //    the golden-bytes tests in tests/rotation_determinism.rs).
            let mut offset = 0;
            while offset < dim {
                let blk = &mut output[offset..offset + block];
                wht_block(blk, block, self.inv_sqrt_block);
                offset += block;
            }

            // This round's output feeds the next round's gather.
            std::mem::swap(&mut input, &mut output);
        }
    }
}

/// Unnormalized Walsh-Hadamard butterfly over one `block`-length slice,
/// followed by the `1/√B` orthonormalization scale.
///
/// Scalar reference semantics: for each stage `len = 1, 2, 4, …, B/2`,
/// each pair `(blk[j], blk[j+len])` becomes `(a+b, a-b)`. The aarch64
/// path executes the identical operations 4 lanes at a time; butterflies
/// within a stage touch disjoint elements, so the results are
/// bit-identical to the scalar loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn wht_block(blk: &mut [f32], block: usize, inv_sqrt_block: f32) {
    use std::arch::aarch64::*;
    debug_assert!(block >= 8 && block.is_power_of_two());
    let p = blk.as_mut_ptr();

    unsafe {
        // Stage len=1: pairs are adjacent. vld2q deinterleaves 8 floats
        // into a = evens, b = odds; store back interleaved.
        let mut j = 0;
        while j < block {
            let ab = vld2q_f32(p.add(j));
            let s = vaddq_f32(ab.0, ab.1);
            let d = vsubq_f32(ab.0, ab.1);
            vst2q_f32(p.add(j), float32x4x2_t(s, d));
            j += 8;
        }

        // Stage len=2: layout per 8 floats is [a0 a1 b0 b1 a2 a3 b2 b3].
        let mut j = 0;
        while j < block {
            let q0 = vld1q_f32(p.add(j)); // [a0 a1 b0 b1]
            let q1 = vld1q_f32(p.add(j + 4)); // [a2 a3 b2 b3]
            let a = vcombine_f32(vget_low_f32(q0), vget_low_f32(q1));
            let b = vcombine_f32(vget_high_f32(q0), vget_high_f32(q1));
            let s = vaddq_f32(a, b);
            let d = vsubq_f32(a, b);
            vst1q_f32(p.add(j), vcombine_f32(vget_low_f32(s), vget_low_f32(d)));
            vst1q_f32(p.add(j + 4), vcombine_f32(vget_high_f32(s), vget_high_f32(d)));
            j += 8;
        }

        // Stages len >= 4: operand pairs are 4-aligned and disjoint.
        let mut len = 4;
        while len < block {
            let mut i = 0;
            while i < block {
                let mut j = i;
                while j < i + len {
                    let a = vld1q_f32(p.add(j));
                    let b = vld1q_f32(p.add(j + len));
                    vst1q_f32(p.add(j), vaddq_f32(a, b));
                    vst1q_f32(p.add(j + len), vsubq_f32(a, b));
                    j += 4;
                }
                i += 2 * len;
            }
            len <<= 1;
        }

        // Orthonormalization scale.
        let sv = vdupq_n_f32(inv_sqrt_block);
        let mut j = 0;
        while j < block {
            vst1q_f32(p.add(j), vmulq_f32(vld1q_f32(p.add(j)), sv));
            j += 4;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn wht_block(blk: &mut [f32], block: usize, inv_sqrt_block: f32) {
    let mut len = 1;
    while len < block {
        let mut i = 0;
        while i < block {
            for j in i..i + len {
                let a = blk[j];
                let b = blk[j + len];
                blk[j] = a + b;
                blk[j + len] = a - b;
            }
            i += 2 * len;
        }
        len <<= 1;
    }
    for x in blk.iter_mut() {
        *x *= inv_sqrt_block;
    }
}

/// Fisher-Yates shuffle of `0..dim` driven by `rng`.
///
/// Uses `next_u64() % (i + 1)` for the swap index — a fixed, portable
/// integer op. The residual modulo bias is negligible for `dim ≤ MAX_DIM`
/// and, being deterministic, is part of the frozen format contract rather
/// than a defect to correct.
fn fisher_yates(dim: usize, rng: &mut ChaCha8Rng) -> Vec<u32> {
    let mut perm: Vec<u32> = (0..dim as u32).collect();
    for i in (1..dim).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    perm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_is_largest_power_of_two_divisor() {
        assert_eq!(block_size(8), 8);
        assert_eq!(block_size(200), 8); // 8·25
        assert_eq!(block_size(768), 256); // 256·3
        assert_eq!(block_size(1000), 8); // 8·125
        assert_eq!(block_size(1536), 512); // 512·3
        assert_eq!(block_size(3072), 1024); // 1024·3
        assert_eq!(block_size(1024), 1024); // pure power of two
    }

    #[test]
    fn preserves_norm_and_is_deterministic() {
        for &dim in &[8usize, 200, 768, 1000, 1536] {
            let rot = Rotation::new(dim);
            let mut state = 0x1234_5678u64 ^ dim as u64;
            let mut v: Vec<f32> = (0..dim)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32
                })
                .collect();
            let before = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
            let orig = v.clone();
            rot.apply(&mut v);
            let after = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
            assert!(
                (before - after).abs() / before < 1e-4,
                "norm changed at dim={dim}: {before} -> {after}"
            );

            // Determinism: a second rotation of the same input matches
            // bit-for-bit.
            let mut again = orig;
            Rotation::new(dim).apply(&mut again);
            assert_eq!(v, again, "rotation not deterministic at dim={dim}");
        }
    }
}
