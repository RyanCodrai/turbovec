//! Bit-plane to SIMD-blocked layout repacking.
//!
//! Converts bit-plane packed codes into a layout optimised for SIMD scoring:
//! - x86: FAISS-style perm0-interleaved for AVX2 cross-lane compatibility
//! - ARM: Sequential layout for NEON

use crate::BLOCK;

/// Repack bit-plane codes into SIMD-blocked layout.
/// Returns (blocked_codes, n_blocks).
pub fn repack(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> (Vec<u8>, usize) {
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let blocked_size = n_blocks * n_byte_groups * BLOCK;
    let bytes_per_row = bits * bytes_per_plane;

    let perm0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

    // Step 1: Extract packed nibble bytes per vector per group
    let mut codes_flat = vec![vec![0u8; n_byte_groups]; n_vectors];
    for vec_idx in 0..n_vectors {
        for g in 0..n_byte_groups {
            let dim_start = g * codes_per_byte;
            let mut byte_val = 0u8;
            for c in 0..codes_per_byte {
                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                let mask = 1u8 << bit_in_byte;

                let mut code = 0u8;
                for p in 0..bits {
                    let plane_byte = packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
                    if plane_byte & mask != 0 {
                        code |= 1 << p;
                    }
                }

                let shift = if bits == 3 {
                    (codes_per_byte - 1 - c) * 4
                } else {
                    (codes_per_byte - 1 - c) * bits
                };
                byte_val |= code << shift;
            }
            codes_flat[vec_idx][g] = byte_val;
        }
    }

    // Step 2: Pack into platform-specific layout
    let blocked = pack_blocked(n_vectors, n_blocks, n_byte_groups, blocked_size, &codes_flat, &perm0);
    (blocked, n_blocks)
}

#[cfg(target_arch = "x86_64")]
fn pack_blocked(
    n: usize,
    n_blocks: usize,
    n_byte_groups: usize,
    blocked_size: usize,
    codes_flat: &[Vec<u8>],
    perm0: &[usize; 16],
) -> Vec<u8> {
    // FAISS layout: split each byte into hi/lo nibbles, interleave with perm0.
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for j in 0..16 {
                let va = base_vec + perm0[j];
                let vb = base_vec + perm0[j] + 16;
                let ba = if va < n { codes_flat[va][g] } else { 0 };
                let bb = if vb < n { codes_flat[vb][g] } else { 0 };
                blocked[out_offset + j] = (ba >> 4) | ((bb >> 4) << 4);
                blocked[out_offset + 16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }
    }
    blocked
}

/// Inverse of the `perm0` permutation used by the x86 `pack_blocked`:
/// `INV_PERM0[lane] == j` such that `perm0[j] == lane`, for `lane` in 0..16.
// Used by the x86 scalar fallback and by the round-trip test on every arch.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) const INV_PERM0: [usize; 16] =
    [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15];

/// Reconstruct the *sequential* code byte for vector `lane` (0..32) of a
/// block group from the x86 `perm0`-interleaved hi/lo-nibble layout that the
/// x86 [`pack_blocked`] produces. `group_off` is the byte offset of the group
/// within `blocked` (i.e. `block_offset + g * BLOCK`).
///
/// The x86 SIMD kernels read that interleaved layout natively, but the scalar
/// fallback ([`crate::search::score_query_into_heap`]) decodes one sequential
/// byte per vector. Without this de-interleave the scalar path — taken on
/// pre-AVX2 x86 / VMs without AVX2 — read the wrong bytes and returned
/// silently-wrong top-k results (issue #106). The returned byte is identical
/// to what the non-x86 sequential layout stores directly: high nibble = the
/// vector's "hi" code, low nibble = its "lo" code.
#[inline]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn deinterleave_x86_code_byte(blocked: &[u8], group_off: usize, lane: usize) -> u8 {
    let j = INV_PERM0[lane & 15];
    let hi_plane = blocked[group_off + j]; // byte holding hi-nibbles of two vectors
    let lo_plane = blocked[group_off + 16 + j]; // byte holding lo-nibbles
    let (hi, lo) = if lane < 16 {
        (hi_plane & 0x0F, lo_plane & 0x0F)
    } else {
        (hi_plane >> 4, lo_plane >> 4)
    };
    (hi << 4) | lo
}

#[cfg(not(target_arch = "x86_64"))]
fn pack_blocked(
    n: usize,
    n_blocks: usize,
    n_byte_groups: usize,
    blocked_size: usize,
    codes_flat: &[Vec<u8>],
    _perm0: &[usize; 16],
) -> Vec<u8> {
    // Sequential layout: each byte stored as-is, vectors in order.
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vi = base_vec + lane;
                if vi < n {
                    blocked[out_offset + lane] = codes_flat[vi][g];
                }
            }
        }
    }
    blocked
}

#[cfg(test)]
mod tests {
    use super::{deinterleave_x86_code_byte, BLOCK};

    /// Pack one 32-vector block exactly as the x86 `pack_blocked` does, then
    /// verify `deinterleave_x86_code_byte` recovers each vector's sequential
    /// code byte. This validates the issue-#106 scalar-fallback fix on every
    /// architecture (including ARM, where the x86 search path can't run) by
    /// exercising the layout math directly.
    #[test]
    fn deinterleave_x86_recovers_sequential_code_bytes() {
        let n_byte_groups = 5usize;
        // Deterministic pseudo-random code bytes for 32 vectors.
        let mut codes_flat = vec![vec![0u8; n_byte_groups]; BLOCK];
        let mut s = 0x1234_5678u32;
        for v in 0..BLOCK {
            for g in 0..n_byte_groups {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                codes_flat[v][g] = (s >> 24) as u8;
            }
        }

        let perm0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
        let mut blocked = vec![0u8; n_byte_groups * BLOCK];
        for g in 0..n_byte_groups {
            let out_offset = g * BLOCK;
            for j in 0..16 {
                let ba = codes_flat[perm0[j]][g];
                let bb = codes_flat[perm0[j] + 16][g];
                blocked[out_offset + j] = (ba >> 4) | ((bb >> 4) << 4);
                blocked[out_offset + 16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }

        for g in 0..n_byte_groups {
            for lane in 0..BLOCK {
                assert_eq!(
                    deinterleave_x86_code_byte(&blocked, g * BLOCK, lane),
                    codes_flat[lane][g],
                    "mismatch at lane {lane}, group {g}",
                );
            }
        }
    }
}
