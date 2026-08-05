//! Bit-plane to SIMD-blocked layout repacking.
//!
//! Converts bit-plane packed codes into a layout optimised for SIMD scoring:
//! - x86: FAISS-style perm0-interleaved for AVX2 cross-lane compatibility
//! - ARM: Sequential layout for NEON

use crate::BLOCK;

/// Packed code bytes → the native search layout for this target.
///
/// x86 interleaves nibbles through `perm0`; every other target's native
/// layout *is* the sequential one, so it shares
/// [`pack_blocked_sequential`] rather than keeping a second copy of the
/// same loop. Deliberately a `cfg` on the call rather than a `cfg`-gated
/// function: a function compiled out on x86 cannot be covered by any test
/// the x86-only mutation gate runs, so it is reported uncovered forever
/// regardless of how well the logic is tested (#421). With no non-x86
/// function body there is nothing to mutate.
macro_rules! pack_blocked_native {
    ($n:expr, $n_blocks:expr, $n_byte_groups:expr, $blocked_size:expr, $codes_flat:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            pack_blocked($n, $n_blocks, $n_byte_groups, $blocked_size, $codes_flat, &PERM0)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            pack_blocked_sequential($n, $n_blocks, $n_byte_groups, $blocked_size, $codes_flat)
        }
    }};
}

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
    let blocked =
        pack_blocked_native!(n_vectors, n_blocks, n_byte_groups, blocked_size, &codes_flat);
    (blocked, n_blocks)
}

#[cfg(target_arch = "x86_64")]
fn pack_blocked(
    n: usize,
    n_blocks: usize,
    n_byte_groups: usize,
    blocked_size: usize,
    codes_flat: &[u8],
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
                let ba = if va < n { codes_flat[va * n_byte_groups + g] } else { 0 };
                let bb = if vb < n { codes_flat[vb * n_byte_groups + g] } else { 0 };
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

/// Write one vector's *sequential* code byte into the x86 native layout —
/// the exact inverse of [`deinterleave_x86_code_byte`]: nibble-merge the
/// byte into the two plane bytes that hold lane `lane`'s hi/lo nibbles,
/// preserving the partner lane's nibbles.
#[cfg(target_arch = "x86_64")]
pub(crate) fn write_x86_code_byte(blocked: &mut [u8], group_off: usize, lane: usize, code: u8) {
    let j = INV_PERM0[lane & 15];
    let hp = group_off + j;
    let lp = group_off + 16 + j;
    if lane < 16 {
        blocked[hp] = (blocked[hp] & 0xF0) | (code >> 4);
        blocked[lp] = (blocked[lp] & 0xF0) | (code & 0x0F);
    } else {
        blocked[hp] = (blocked[hp] & 0x0F) | (code & 0xF0);
        blocked[lp] = (blocked[lp] & 0x0F) | ((code & 0x0F) << 4);
    }
}

/// Copy vector `src_vec`'s code bytes into vector `dst_vec`'s lane across
/// every byte-group of the native blocked layout — the O(dim) primitive
/// that lets `swap_remove` maintain the cache without a block repack.
///
/// With `capture`, the moved row's *sequential* code bytes are also written
/// there. The move already computes exactly those bytes, one per byte
/// group, and would otherwise drop each into the destination lane and
/// forget it; handing them out costs a store per group and saves a later
/// reader from walking the whole 32-lane block to recover them (at dim 768,
/// a 12 KB strided read to collect 384 bytes). The bytes are the same
/// either way, because `write_x86_code_byte` is the exact inverse of the
/// de-interleave — what is captured is what a later read of `dst_vec`'s
/// lane returns.
///
/// `capture`, when given, must be exactly `n_byte_groups` long. The caller
/// sizes it so the loop below is a straight indexed store with no capacity
/// check and no temporary, which matters at one call per byte group per
/// removal: a `Vec::push` per byte, plus a `Vec` per removal and a copy out
/// of it, cost more than the whole capture is worth.
pub(crate) fn move_lane(
    blocked: &mut [u8],
    n_byte_groups: usize,
    src_vec: usize,
    dst_vec: usize,
    capture: Option<&mut [u8]>,
) {
    let (sb, sl) = (src_vec / BLOCK, src_vec % BLOCK);
    let (db, dl) = (dst_vec / BLOCK, dst_vec % BLOCK);
    debug_assert!(capture.as_ref().is_none_or(|c| c.len() == n_byte_groups));
    // Split the two loops rather than testing the option per byte: the
    // plain move is the common path and must stay branch-free.
    match capture {
        None => {
            for g in 0..n_byte_groups {
                let s_off = (sb * n_byte_groups + g) * BLOCK;
                let d_off = (db * n_byte_groups + g) * BLOCK;
                move_one(blocked, s_off, sl, d_off, dl);
            }
        }
        Some(out) => {
            for (g, slot) in out.iter_mut().enumerate() {
                let s_off = (sb * n_byte_groups + g) * BLOCK;
                let d_off = (db * n_byte_groups + g) * BLOCK;
                *slot = move_one(blocked, s_off, sl, d_off, dl);
            }
        }
    }
}

/// One byte group of a lane move, returning the byte that was moved.
#[inline(always)]
fn move_one(blocked: &mut [u8], s_off: usize, sl: usize, d_off: usize, dl: usize) -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        let code = deinterleave_x86_code_byte(blocked, s_off, sl);
        write_x86_code_byte(blocked, d_off, dl, code);
        code
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let code = blocked[s_off + sl];
        blocked[d_off + dl] = code;
        code
    }
}

/// Append `n_new` vectors' packed bit-plane rows to the native blocked
/// layout as direct lane writes, growing the buffer to the new geometry
/// (fresh bytes zeroed, so padding lanes match a from-scratch repack).
/// Existing lanes — including the partial tail block's — are untouched:
/// the cache's exact-bytes invariant carries them. Lets `add` append in
/// the v6-load window without materializing the packed prefix.
pub(crate) fn append_lanes(
    blocked: &mut Vec<u8>,
    packed_rows: &[u8],
    old_n: usize,
    n_new: usize,
    bits: usize,
    dim: usize,
) {
    let (_, n_byte_groups, new_len) = blocked_geometry(old_n + n_new, bits, dim);
    blocked.resize(new_len, 0);
    let codes_flat = extract_codes_flat(packed_rows, n_new, bits, dim);
    for i in 0..n_new {
        let row = &codes_flat[i * n_byte_groups..(i + 1) * n_byte_groups];
        let v = old_n + i;
        let (b, l) = (v / BLOCK, v % BLOCK);
        for (g, &code) in row.iter().enumerate() {
            let off = (b * n_byte_groups + g) * BLOCK;
            #[cfg(target_arch = "x86_64")]
            write_x86_code_byte(blocked, off, l, code);
            #[cfg(not(target_arch = "x86_64"))]
            {
                blocked[off + l] = code;
            }
        }
    }
}

/// Zero vector `vec_idx`'s code bytes across every byte-group — vacated
/// and padding lanes must be exactly zero so serialized cache bytes match
/// a from-scratch repack.
pub(crate) fn zero_lane(blocked: &mut [u8], n_byte_groups: usize, vec_idx: usize) {
    let (b, l) = (vec_idx / BLOCK, vec_idx % BLOCK);
    for g in 0..n_byte_groups {
        let off = (b * n_byte_groups + g) * BLOCK;
        #[cfg(target_arch = "x86_64")]
        write_x86_code_byte(blocked, off, l, 0);
        #[cfg(not(target_arch = "x86_64"))]
        {
            blocked[off + l] = 0;
        }
    }
}

/// The x86 in-block nibble-interleave permutation (see [`pack_blocked`]).
// Only the x86 layout permutes; other targets store lanes sequentially.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) const PERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

/// Byte-group / block geometry shared by every layout function here.
/// Returns `(n_blocks, n_byte_groups, blocked_len)`.
pub(crate) fn blocked_geometry(n_vectors: usize, bits: usize, dim: usize) -> (usize, usize, usize) {
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    (n_blocks, n_byte_groups, n_blocks * n_byte_groups * BLOCK)
}

/// Per-plane-byte extraction table: `lut[p][b]` scatters the 8 bits of
/// plane `p`'s byte `b` (dim-descending bit order) into the up-to-4
/// group bytes an 8-dim chunk produces, as a little-endian u32 — one
/// lookup per plane byte replaces the bit-by-bit gather (the mirror of
/// [`build_unpack_lut`] on the packing side).
fn build_extract_lut(bits: usize) -> [[u32; 256]; 4] {
    let codes_per_byte = 8 / bits;
    let field = if bits == 3 { 4 } else { bits };
    let mut lut = [[0u32; 256]; 4];
    for (p, plane) in lut.iter_mut().enumerate().take(bits) {
        for (b, e) in plane.iter_mut().enumerate() {
            let mut acc = 0u32;
            for j in 0..8usize {
                if b & (1 << (7 - j)) != 0 {
                    let out_byte = j / codes_per_byte;
                    let shift_in_byte = (codes_per_byte - 1 - (j % codes_per_byte)) * field;
                    acc |= 1u32 << (out_byte * 8 + shift_in_byte + p);
                }
            }
            *e = acc;
        }
    }
    lut
}

/// The extract LUT for each supported bit width, built once per process on
/// first use. The table is 4 KB and depends only on `bits`, so rebuilding it
/// inside [`extract_codes_flat`] charged every call — including the one-row
/// append the lazy-load `add` path takes — the full construction cost.
/// Indexed by `bits`; entries 0 and 1 are never read (`2 <= bits <= 4` is a
/// crate invariant, enforced by `from_parts`).
static EXTRACT_LUTS: [std::sync::OnceLock<[[u32; 256]; 4]>; 5] =
    [const { std::sync::OnceLock::new() }; 5];

/// Cached [`build_extract_lut`].
fn extract_lut(bits: usize) -> &'static [[u32; 256]; 4] {
    EXTRACT_LUTS[bits].get_or_init(|| build_extract_lut(bits))
}

/// Extract per-vector code bytes (one byte per byte-group) from the
/// bit-plane packed rows — step 1 of every packed→blocked conversion.
/// Branch-free: each 8-dim chunk is `bits` LUT lookups OR-ed together.
///
/// The result is one flat `n_vectors * n_byte_groups` buffer, row-major
/// with stride `n_byte_groups` — a single allocation instead of one per
/// vector plus an outer vector of pointers. The per-vector form was paid
/// in full by callers extracting a single row (#409).
pub(crate) fn extract_codes_flat(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> Vec<u8> {
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let bytes_per_row = bits * bytes_per_plane;
    let n_out = 8 / codes_per_byte;
    let lut = extract_lut(bits);
    let mut codes_flat = vec![0u8; n_vectors * n_byte_groups];
    if codes_flat.is_empty() {
        return codes_flat;
    }
    for (vec_idx, row) in codes_flat.chunks_exact_mut(n_byte_groups).enumerate() {
        let base = vec_idx * bytes_per_row;
        for c in 0..bytes_per_plane {
            let mut acc = 0u32;
            for (p, plane) in lut.iter().enumerate().take(bits) {
                acc |= plane[packed_codes[base + p * bytes_per_plane + c] as usize];
            }
            let le = acc.to_le_bytes();
            row[c * n_out..c * n_out + n_out].copy_from_slice(&le[..n_out]);
        }
    }
    codes_flat
}

/// Pack extracted code bytes into the *sequential* blocked layout — the
/// arch-neutral form the v6 file format persists: vectors in order inside
/// each 32-vector block, one code byte per lane. On non-x86 this is also
/// the layout the search kernel consumes.
pub(crate) fn pack_blocked_sequential(
    n: usize,
    n_blocks: usize,
    n_byte_groups: usize,
    blocked_size: usize,
    codes_flat: &[u8],
) -> Vec<u8> {
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vi = base_vec + lane;
                if vi < n {
                    blocked[out_offset + lane] = codes_flat[vi * n_byte_groups + g];
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
    // Rows are independent; parallelize over block-aligned row chunks so
    // each chunk reads whole blocks of `seq`. Serial for small payloads
    // (thread-spawn overhead dominates below ~4 MB, same threshold as
    // `interleave_blocks_x86_in_place`).
    const PAR_THRESHOLD: usize = 4 * 1024 * 1024;
    const ROWS_PER_CHUNK: usize = 512 * BLOCK;
    let lut = build_unpack_lut(bits);
    let unpack_rows = |first_vec: usize, rows: &mut [u8]| {
        for (r, row) in rows.chunks_exact_mut(bytes_per_row).enumerate() {
            unpack_row(seq, first_vec + r, row, bits, n_byte_groups, bytes_per_plane, &lut);
        }
    };
    if packed.len() >= PAR_THRESHOLD {
        use rayon::prelude::*;
        packed
            .par_chunks_mut(ROWS_PER_CHUNK * bytes_per_row)
            .enumerate()
            .for_each(|(ci, chunk)| unpack_rows(ci * ROWS_PER_CHUNK, chunk));
    } else {
        unpack_rows(0, &mut packed);
    }
    packed
}

/// Per-group-byte unpack table: entry `lut[b]` holds, for each plane `p`,
/// a `codes_per_byte`-bit field at offset `p * codes_per_byte` whose bit
/// `codes_per_byte - 1 - c` is bit `p` of the byte's `c`-th code. One
/// lookup replaces the bit-by-bit inner loop of the naive unpack — the
/// fields land in dim order, so a plane's output byte is just the fields
/// of its `8 / codes_per_byte` group bytes shifted into place.
fn build_unpack_lut(bits: usize) -> [u16; 256] {
    let codes_per_byte = 8 / bits;
    let mut lut = [0u16; 256];
    for (b, e) in lut.iter_mut().enumerate() {
        for c in 0..codes_per_byte {
            let shift = if bits == 3 {
                (codes_per_byte - 1 - c) * 4
            } else {
                (codes_per_byte - 1 - c) * bits
            };
            let code = (b >> shift) & ((1usize << bits) - 1);
            for p in 0..bits {
                if code & (1 << p) != 0 {
                    *e |= 1 << (p * codes_per_byte + (codes_per_byte - 1 - c));
                }
            }
        }
    }
    lut
}

/// Unpack one vector's bit-plane row from the sequential blocked layout —
/// the per-row body of [`seq_to_packed`]. Branch-free: one LUT lookup per
/// group byte, `8 / codes_per_byte` group bytes assembled per plane byte.
#[inline]
fn unpack_row(
    seq: &[u8],
    vec_idx: usize,
    row: &mut [u8],
    bits: usize,
    n_byte_groups: usize,
    bytes_per_plane: usize,
    lut: &[u16; 256],
) {
    let codes_per_byte = 8 / bits;
    let groups_per_out = 8 / codes_per_byte;
    let field_mask = (1u16 << codes_per_byte) - 1;
    let block_idx = vec_idx / BLOCK;
    let lane = vec_idx % BLOCK;
    let group_base = block_idx * n_byte_groups;
    debug_assert_eq!(n_byte_groups, bytes_per_plane * groups_per_out);
    for ob in 0..bytes_per_plane {
        let mut acc = [0u8; 4]; // one accumulator per plane; bits <= 4
        for q in 0..groups_per_out {
            let g = ob * groups_per_out + q;
            let byte_val = seq[(group_base + g) * BLOCK + lane];
            let e = lut[byte_val as usize];
            let sh = 8 - codes_per_byte * (q + 1);
            for (p, a) in acc.iter_mut().enumerate().take(bits) {
                *a |= (((e >> (p * codes_per_byte)) & field_mask) as u8) << sh;
            }
        }
        for (p, a) in acc.iter().enumerate().take(bits) {
            row[p * bytes_per_plane + ob] = *a;
        }
    }
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
    pack_blocked_native!(n_range, range_blocks, n_byte_groups, blocked_size, &codes_flat)
}

/// Byte `group` of lane `lane` in the sequential-blocked block starting
/// at `base` — the O(dim) row gather the non-x86 `seq_row` arm uses.
/// Kept cfg-free so every arch compiles and unit-tests the exact
/// arithmetic; x86's `seq_row` uses the nibble de-interleave instead.
// On x86 the lib target never calls this (`seq_row` de-interleaves
// nibbles instead); it exists there for the cross-arch unit test.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
pub(crate) fn seq_lane_byte(data: &[u8], base: usize, group: usize, lane: usize) -> u8 {
    data[base + group * BLOCK + lane]
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

/// Per-chunk native→sequential transform for the fused write path:
/// deinterleave a block-aligned chunk of the native cache into `out`
/// (resized to match). Chunk-local because the perm0 interleave never
/// crosses a 32-byte block, so per-chunk output is byte-identical to the
/// same range of a whole-buffer `native_to_seq`.
///
/// Deliberately SERIAL (`deinterleave_chunk_x86`, never the rayon-fanning
/// `deinterleave_blocks_x86`): callers are the scoped writer threads in
/// `write_atomic_parallel`, which provide the parallelism themselves and
/// have no rayon pool context — a parallel iterator here would inject
/// work into the global registry, which the Python bindings pin to a
/// one-thread sentinel whose contract is that it never receives work
/// (and whose sole worker is dead in a forked child — a `save()` there
/// would hang). Mirrors the load side, where `interleave_chunk_x86`
/// runs serially inside each reader thread.
#[cfg(target_arch = "x86_64")]
pub(crate) fn deinterleave_chunk_into(chunk: &[u8], out: &mut Vec<u8>) {
    debug_assert_eq!(chunk.len() % BLOCK, 0);
    out.resize(chunk.len(), 0);
    deinterleave_chunk_x86(chunk, out);
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

    /// Direct check on the sequential blocked layout's addressing: each
    /// vector's code byte for byte-group `g` lands at lane `v % 32` of
    /// block `v / 32`, and every lane at or beyond `n_vectors` is zero.
    ///
    /// Deliberately a tiny fixture asserted byte-for-byte rather than a
    /// round-trip: it pins the row stride (`v * n_byte_groups + g`) and
    /// the `vi < n` padding bound independently, so an off-by-one bound
    /// or a wrong stride fails here in microseconds instead of surviving
    /// as an inverse-of-itself round-trip.
    #[test]
    fn repack_seq_places_each_code_byte_at_its_lane_and_zeroes_padding() {
        // 33 vectors spills into a second block, so the tail padding is
        // exercised: lanes 1..32 of block 1 must be zero.
        let (n, bits, dim) = (33usize, 4usize, 64usize);
        let packed = pseudo_random_packed(n, bits, dim);
        let seq = super::repack_seq(&packed, n, bits, dim);
        let (n_blocks, n_byte_groups, blocked_len) = super::blocked_geometry(n, bits, dim);
        assert_eq!(seq.len(), blocked_len);
        assert_eq!(n_blocks, 2);

        // Independent reference for the per-vector code bytes: unpack each
        // vector's row straight from the bit-planes.
        let codes_per_byte = 8 / bits;
        let bytes_per_plane = dim / 8;
        let expected = |v: usize, g: usize| -> u8 {
            let mut byte = 0u8;
            for k in 0..codes_per_byte {
                let d = g * codes_per_byte + k;
                let mut code = 0u8;
                for p in 0..bits {
                    let bit = (packed[v * bits * bytes_per_plane + p * bytes_per_plane + d / 8]
                        >> (7 - (d % 8)))
                        & 1;
                    code |= bit << p;
                }
                byte |= code << ((codes_per_byte - 1 - k) * bits);
            }
            byte
        };

        for g in 0..n_byte_groups {
            for b in 0..n_blocks {
                for lane in 0..BLOCK {
                    let v = b * BLOCK + lane;
                    let got = seq[(b * n_byte_groups + g) * BLOCK + lane];
                    if v < n {
                        assert_eq!(got, expected(v, g), "vector {v}, group {g}");
                    } else {
                        assert_eq!(got, 0, "padding lane {lane} of block {b}, group {g}");
                    }
                }
            }
        }
    }

    /// `seq_to_packed` is the exact inverse of `repack_seq`, including at
    /// non-multiple-of-32 vector counts (padded tail lanes).
    #[test]
    fn repack_seq_roundtrips_through_seq_to_packed() {
        for (n, bits, dim) in [
            (7usize, 4usize, 64usize),
            (32, 2, 64),
            (100, 4, 96),
            (33, 2, 128),
            (50, 3, 64),
            (33, 3, 128),
        ] {
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

#[cfg(test)]
mod seq_lane_tests {
    use super::{seq_lane_byte, BLOCK};

    /// The lane gather's exact arithmetic, pinned on a synthetic
    /// two-block buffer where every byte encodes its own coordinates —
    /// any sign, stride, or operator slip lands on a different value.
    #[test]
    fn lane_gather_addresses_exactly() {
        let groups = 5;
        let block_bytes = groups * BLOCK;
        let data: Vec<u8> = (0..2 * block_bytes).map(|i| (i % 251) as u8).collect();
        for block in 0..2 {
            let base = block * block_bytes;
            for lane in [0usize, 1, 17, 31] {
                for g in 0..groups {
                    assert_eq!(
                        seq_lane_byte(&data, base, g, lane),
                        ((base + g * BLOCK + lane) % 251) as u8,
                        "block {block} lane {lane} group {g}"
                    );
                }
            }
        }
    }
}
