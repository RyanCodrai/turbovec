//! Bit-plane to SIMD-blocked layout repacking.
//!
//! Converts bit-plane packed codes into a layout optimised for SIMD scoring:
//! - x86: FAISS-style perm0-interleaved for AVX2 cross-lane compatibility
//! - ARM: Sequential layout for NEON

use crate::BLOCK;

/// Repack bit-plane codes into SIMD-blocked layout.
/// Returns (blocked_codes, n_blocks).
///
/// Crate-internal: trusts `2 <= bits <= 4`, `dim` a multiple of 8, and
/// `packed_codes.len() == n_vectors * (dim/8) * bits`. A raw caller passing
/// `bits == 0` divides by zero and a short `packed_codes` reads out of
/// bounds — construct through
/// [`from_parts`](crate::TurboQuantIndex::from_parts) instead, which
/// validates these before the blocked layout is ever built.
pub(crate) fn repack(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> (Vec<u8>, usize) {
    let (n_blocks, n_byte_groups, blocked_size) = blocked_geometry(n_vectors, bits, dim);

    // Step 1: Extract packed nibble bytes per vector per group
    let codes_flat = extract_codes_flat(packed_codes, n_vectors, bits, dim);

    // Step 2: Pack into platform-specific layout
    let blocked = pack_blocked(n_vectors, n_blocks, n_byte_groups, blocked_size, &codes_flat, &PERM0);
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

/// The x86 in-block nibble-interleave permutation (see [`pack_blocked`]).
pub(crate) const PERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

/// Byte-group / block geometry shared by every layout function here.
/// Returns `(n_blocks, n_byte_groups, blocked_len)`.
pub(crate) fn blocked_geometry(n_vectors: usize, bits: usize, dim: usize) -> (usize, usize, usize) {
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    (n_blocks, n_byte_groups, n_blocks * n_byte_groups * BLOCK)
}

/// Extract per-vector code bytes (one byte per byte-group) from the
/// bit-plane packed rows — step 1 of every packed→blocked conversion.
fn extract_codes_flat(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> Vec<Vec<u8>> {
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let bytes_per_row = bits * bytes_per_plane;
    let mut codes_flat = vec![vec![0u8; n_byte_groups]; n_vectors];
    for (vec_idx, row) in codes_flat.iter_mut().enumerate() {
        for (g, byte_out) in row.iter_mut().enumerate() {
            let dim_start = g * codes_per_byte;
            let mut byte_val = 0u8;
            for c in 0..codes_per_byte {
                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                let mask = 1u8 << bit_in_byte;
                let mut code = 0u8;
                for p in 0..bits {
                    let plane_byte =
                        packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
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
            *byte_out = byte_val;
        }
    }
    codes_flat
}

/// Pack extracted code bytes into the *sequential* blocked layout — the
/// arch-neutral form the v6 file format persists: vectors in order inside
/// each 32-vector block, one code byte per lane. On non-x86 this is also
/// the layout the search kernel consumes.
fn pack_blocked_sequential(
    n: usize,
    n_blocks: usize,
    n_byte_groups: usize,
    blocked_size: usize,
    codes_flat: &[Vec<u8>],
) -> Vec<u8> {
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

/// Packed bit-plane rows → sequential blocked layout (the v6 file
/// payload). Arch-independent and deterministic: identical bytes on every
/// platform for the same packed codes.
pub(crate) fn repack_seq(packed_codes: &[u8], n_vectors: usize, bits: usize, dim: usize) -> Vec<u8> {
    let (n_blocks, n_byte_groups, blocked_size) = blocked_geometry(n_vectors, bits, dim);
    let codes_flat = extract_codes_flat(packed_codes, n_vectors, bits, dim);
    pack_blocked_sequential(n_vectors, n_blocks, n_byte_groups, blocked_size, &codes_flat)
}

/// Sequential blocked layout → packed bit-plane rows — the exact inverse
/// of [`repack_seq`]. Used to lazily rebuild `packed_codes` after a v6
/// load, only when a mutation or byte-serialization first needs them.
pub(crate) fn seq_to_packed(seq: &[u8], n_vectors: usize, bits: usize, dim: usize) -> Vec<u8> {
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let bytes_per_row = bits * bytes_per_plane;
    let mut packed = vec![0u8; n_vectors * bytes_per_row];
    for vec_idx in 0..n_vectors {
        let block_idx = vec_idx / BLOCK;
        let lane = vec_idx % BLOCK;
        let row = &mut packed[vec_idx * bytes_per_row..(vec_idx + 1) * bytes_per_row];
        for g in 0..n_byte_groups {
            let byte_val = seq[(block_idx * n_byte_groups + g) * BLOCK + lane];
            let dim_start = g * codes_per_byte;
            for c in 0..codes_per_byte {
                let shift = if bits == 3 {
                    (codes_per_byte - 1 - c) * 4
                } else {
                    (codes_per_byte - 1 - c) * bits
                };
                let code = (byte_val >> shift) & ((1u8 << bits) - 1);
                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                for p in 0..bits {
                    if code & (1 << p) != 0 {
                        row[p * bytes_per_plane + byte_in_plane] |= 1 << bit_in_byte;
                    }
                }
            }
        }
    }
    packed
}

/// Sequential blocked layout → the native layout the search kernel
/// reads, consuming the buffer. Non-x86: the sequential layout *is*
/// native — the buffer is returned untouched (zero-copy: a load hands
/// the file bytes straight to the search cache). x86: the per-block
/// `perm0` nibble interleave applied *in place* (each block's lanes are
/// loaded into registers before any store), run threaded with SIMD and
/// software prefetch — ~2 ms for 76.8 MB vs ~400 ms for a full repack
/// from bit-planes (see `scratch/hypothesis_log.md`).
pub(crate) fn seq_into_native(seq: Vec<u8>) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut buf = seq;
        interleave_blocks_x86_in_place(&mut buf);
        buf
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        seq
    }
}

/// Rebuild the *native* blocked layout for blocks `[block_start,
/// block_end)` from the packed bit-plane rows — the incremental-cache
/// primitive: a mutation recomputes only the blocks it touched instead
/// of discarding the whole cache. Lanes at or beyond `n_vectors` are
/// zero (matching the full repack exactly, so serialized bytes stay
/// deterministic).
pub(crate) fn repack_block_range(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
    block_start: usize,
    block_end: usize,
) -> Vec<u8> {
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let first_vec = block_start * BLOCK;
    let end_vec = (block_end * BLOCK).min(n_vectors);
    debug_assert!(
        first_vec <= n_vectors,
        "repack_block_range: block range starts beyond n_vectors"
    );
    let n_range = end_vec.saturating_sub(first_vec);
    // Extract only the range's rows (indices relative to the range).
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    let sub_packed = &packed_codes[first_vec * bytes_per_row..end_vec * bytes_per_row];
    let codes_flat = extract_codes_flat(sub_packed, n_range, bits, dim);
    let range_blocks = block_end - block_start;
    let blocked_size = range_blocks * n_byte_groups * BLOCK;
    pack_blocked(n_range, range_blocks, n_byte_groups, blocked_size, &codes_flat, &PERM0)
}

/// Native search layout → sequential blocked layout — [`seq_into_native`]'s
/// inverse. Lets the write path serialize a warm in-memory blocked cache
/// without a full O(n·dim) repack from bit-planes.
pub(crate) fn native_to_seq(blocked: &[u8]) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut out = vec![0u8; blocked.len()];
        deinterleave_blocks_x86(blocked, &mut out);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        blocked.to_vec()
    }
}

/// Threaded, SIMD, prefetching x86 in-place interleave: for each
/// 32-byte group, `buf[j] = (s[perm0[j]]>>4) | (s[perm0[j]+16] & 0xF0)`
/// and `buf[16+j] = (s[perm0[j]] & 0x0F) | ((s[perm0[j]+16] & 0x0F) << 4)`
/// where `s` is the block's pre-transform content. In-place is safe
/// because each block's 32 source bytes are read (into registers / a
/// stack copy) before any byte of the block is stored. Plain stores, not
/// streaming: the lines were just loaded, so they are already cache-hot
/// and owned.
#[cfg(target_arch = "x86_64")]
fn interleave_blocks_x86_in_place(buf: &mut [u8]) {
    use rayon::prelude::*;
    debug_assert_eq!(buf.len() % BLOCK, 0);
    // Chunk for parallelism; each chunk is block-aligned so lanes never
    // cross a chunk boundary. Serial for small payloads (thread-spawn
    // overhead dominates below ~4 MB — measured, see hypothesis log H2).
    const PAR_THRESHOLD: usize = 4 * 1024 * 1024;
    const CHUNK: usize = 2 * 1024 * 1024; // multiple of BLOCK
    if buf.len() >= PAR_THRESHOLD {
        buf.par_chunks_mut(CHUNK).for_each(interleave_chunk_x86);
    } else {
        interleave_chunk_x86(buf);
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn interleave_chunk_x86(buf: &mut [u8]) {
    if is_x86_feature_detected!("ssse3") {
        // SAFETY: gated on runtime SSSE3 detection.
        unsafe { interleave_chunk_ssse3(buf) }
    } else {
        let mut tmp = [0u8; BLOCK];
        for o in buf.chunks_exact_mut(BLOCK) {
            tmp.copy_from_slice(o);
            for j in 0..16 {
                let ba = tmp[PERM0[j]];
                let bb = tmp[PERM0[j] + 16];
                o[j] = (ba >> 4) | (bb & 0xF0);
                o[16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }
    }
}

/// SAFETY: caller must ensure SSSE3 is available. `buf.len()` is a
/// multiple of `BLOCK` (callers uphold this). Both 16-byte halves of a
/// block are loaded into registers before either store.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn interleave_chunk_ssse3(buf: &mut [u8]) {
    use std::arch::x86_64::*;
    let perm: [u8; 16] = std::array::from_fn(|j| PERM0[j] as u8);
    let permv = _mm_loadu_si128(perm.as_ptr() as *const __m128i);
    let lo_mask = _mm_set1_epi8(0x0Fu8 as i8);
    let hi_mask = _mm_set1_epi8(0xF0u8 as i8);
    let n = buf.len() / BLOCK;
    for i in 0..n {
        let p = buf.as_mut_ptr().add(i * BLOCK);
        // ~4 KB ahead: the measured sweet spot (hypothesis log H16).
        if (i + 128) * BLOCK < buf.len() {
            _mm_prefetch(buf.as_ptr().add((i + 128) * BLOCK) as *const i8, _MM_HINT_T0);
        }
        let lo16 = _mm_loadu_si128(p as *const __m128i);
        let hi16 = _mm_loadu_si128(p.add(16) as *const __m128i);
        let a = _mm_shuffle_epi8(lo16, permv);
        let b = _mm_shuffle_epi8(hi16, permv);
        let a_hi = _mm_and_si128(_mm_srli_epi16(a, 4), lo_mask);
        let out_hi = _mm_or_si128(a_hi, _mm_and_si128(b, hi_mask));
        let b_lo4 = _mm_and_si128(b, lo_mask);
        let out_lo = _mm_or_si128(_mm_and_si128(a, lo_mask), _mm_slli_epi16(b_lo4, 4));
        _mm_storeu_si128(p as *mut __m128i, out_hi);
        _mm_storeu_si128(p.add(16) as *mut __m128i, out_lo);
    }
}

#[cfg(target_arch = "x86_64")]
fn deinterleave_blocks_x86(blocked: &[u8], out: &mut [u8]) {
    use rayon::prelude::*;
    const PAR_THRESHOLD: usize = 4 * 1024 * 1024;
    const CHUNK: usize = 2 * 1024 * 1024;
    if blocked.len() >= PAR_THRESHOLD {
        out.par_chunks_mut(CHUNK)
            .zip(blocked.par_chunks(CHUNK))
            .for_each(|(o, b)| deinterleave_chunk_x86(b, o));
    } else {
        deinterleave_chunk_x86(blocked, out);
    }
}

#[cfg(target_arch = "x86_64")]
fn deinterleave_chunk_x86(blocked: &[u8], out: &mut [u8]) {
    if is_x86_feature_detected!("ssse3") {
        // SAFETY: gated on runtime SSSE3 detection.
        unsafe { deinterleave_chunk_ssse3(blocked, out) }
    } else {
        for (b, o) in blocked.chunks_exact(BLOCK).zip(out.chunks_exact_mut(BLOCK)) {
            for lane in 0..BLOCK {
                o[lane] = deinterleave_x86_code_byte(b, 0, lane);
            }
        }
    }
}

/// SAFETY: caller must ensure SSSE3 is available. Inverse of
/// [`interleave_chunk_ssse3`]: `ba = ((hi&0x0F)<<4) | (lo&0x0F)`,
/// `bb = (hi&0xF0) | (lo>>4)`, scattered back through `INV_PERM0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn deinterleave_chunk_ssse3(blocked: &[u8], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let inv: [u8; 16] = std::array::from_fn(|lane| INV_PERM0[lane] as u8);
    let invv = _mm_loadu_si128(inv.as_ptr() as *const __m128i);
    let lo_mask = _mm_set1_epi8(0x0Fu8 as i8);
    let hi_mask = _mm_set1_epi8(0xF0u8 as i8);
    let n = blocked.len() / BLOCK;
    let nt = out.as_ptr() as usize % 16 == 0;
    for i in 0..n {
        let b = blocked.as_ptr().add(i * BLOCK);
        if (i + 128) * BLOCK < blocked.len() {
            _mm_prefetch(blocked.as_ptr().add((i + 128) * BLOCK) as *const i8, _MM_HINT_T0);
        }
        let o = out.as_mut_ptr().add(i * BLOCK);
        let hi_plane = _mm_loadu_si128(b as *const __m128i);
        let lo_plane = _mm_loadu_si128(b.add(16) as *const __m128i);
        // ba[j] (vectors perm0[j], i.e. lanes 0..16 pre-permutation)
        let ba = _mm_or_si128(
            _mm_slli_epi16(_mm_and_si128(hi_plane, lo_mask), 4),
            _mm_and_si128(lo_plane, lo_mask),
        );
        // bb[j] (vectors perm0[j]+16)
        let bb = _mm_or_si128(
            _mm_and_si128(hi_plane, hi_mask),
            _mm_and_si128(_mm_srli_epi16(lo_plane, 4), lo_mask),
        );
        // seq[lane] = ba[INV_PERM0[lane]] / bb[INV_PERM0[lane]]
        let seq_lo = _mm_shuffle_epi8(ba, invv);
        let seq_hi = _mm_shuffle_epi8(bb, invv);
        if nt {
            _mm_stream_si128(o as *mut __m128i, seq_lo);
            _mm_stream_si128(o.add(16) as *mut __m128i, seq_hi);
        } else {
            _mm_storeu_si128(o as *mut __m128i, seq_lo);
            _mm_storeu_si128(o.add(16) as *mut __m128i, seq_hi);
        }
    }
    if nt {
        _mm_sfence();
    }
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

    fn pseudo_random_packed(n_vectors: usize, bits: usize, dim: usize) -> Vec<u8> {
        let bytes_per_row = bits * dim / 8;
        let mut s = 0x9E37_79B9u32;
        (0..n_vectors * bytes_per_row)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// `seq_to_packed` is the exact inverse of `repack_seq`, including at
    /// non-multiple-of-32 vector counts (padded tail lanes).
    #[test]
    fn repack_seq_roundtrips_through_seq_to_packed() {
        for (n, bits, dim) in [(7usize, 4usize, 64usize), (32, 2, 64), (100, 4, 96), (33, 2, 128)] {
            let packed = pseudo_random_packed(n, bits, dim);
            let seq = super::repack_seq(&packed, n, bits, dim);
            let back = super::seq_to_packed(&seq, n, bits, dim);
            assert_eq!(back, packed, "n={n} bits={bits} dim={dim}");
        }
    }

    /// The native layout produced by `repack` equals
    /// `seq_to_native(repack_seq(..))` — the v6 load path reconstructs
    /// exactly what the in-memory first-search rebuild would have built.
    #[test]
    fn seq_to_native_matches_repack() {
        for (n, bits, dim) in [(7usize, 4usize, 64usize), (100, 4, 96), (33, 2, 128), (1000, 4, 64)] {
            let packed = pseudo_random_packed(n, bits, dim);
            let (native, _) = super::repack(&packed, n, bits, dim);
            let seq = super::repack_seq(&packed, n, bits, dim);
            assert_eq!(super::seq_into_native(seq.clone()), native, "n={n} bits={bits} dim={dim}");
            assert_eq!(super::native_to_seq(&native), seq, "inverse n={n} bits={bits} dim={dim}");
        }
    }

    #[test]
    fn pseudo_random_helper_is_deterministic() {
        assert_eq!(pseudo_random_packed(3, 4, 64), pseudo_random_packed(3, 4, 64));
    }
}
