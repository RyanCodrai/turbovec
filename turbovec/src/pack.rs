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
    ($n:expr, $n_blocks:expr, $bits:expr, $n_byte_groups:expr, $blocked_size:expr, $codes_flat:expr) => {{
        // The encode path is the SECOND producer of the native layout
        // (the loader is the first), and both must emit whichever layout
        // the search dispatch will read. Getting this wrong does not
        // fail loudly: an index built by adding vectors would simply be
        // scored against a layout it is not in.
        if vm8_for($bits, $n_byte_groups) {
            let mut b = pack_blocked_sequential(
                $n, $n_blocks, $n_byte_groups, $blocked_size, $codes_flat);
            vector_major8_chunk(&mut b);
            b
        } else if vector_major_for($bits, $n_byte_groups) {
            let mut b = pack_blocked_sequential(
                $n, $n_blocks, $n_byte_groups, $blocked_size, $codes_flat);
            vector_major_chunk(&mut b);
            b
        } else {
            #[cfg(target_arch = "x86_64")]
            {
                pack_blocked($n, $n_blocks, $n_byte_groups, $blocked_size, $codes_flat, &PERM0)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                pack_blocked_sequential($n, $n_blocks, $n_byte_groups, $blocked_size, $codes_flat)
            }
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
        pack_blocked_native!(n_vectors, n_blocks, bits, n_byte_groups, blocked_size, &codes_flat);
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
    bits: usize,
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
                let code = read_code(blocked, bits, n_byte_groups, sb, g, sl);
                write_code(blocked, bits, n_byte_groups, db, g, dl, code);
            }
        }
        Some(out) => {
            for (g, slot) in out.iter_mut().enumerate() {
                let code = read_code(blocked, bits, n_byte_groups, sb, g, sl);
                write_code(blocked, bits, n_byte_groups, db, g, dl, code);
                *slot = code;
            }
        }
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
            write_code(blocked, bits, n_byte_groups, b, g, l, code);
        }
    }
}

/// Zero vector `vec_idx`'s code bytes across every byte-group — vacated
/// and padding lanes must be exactly zero so serialized cache bytes match
/// a from-scratch repack.
pub(crate) fn zero_lane(blocked: &mut [u8], bits: usize, n_byte_groups: usize, vec_idx: usize) {
    let (b, l) = (vec_idx / BLOCK, vec_idx % BLOCK);
    for g in 0..n_byte_groups {
        write_code(blocked, bits, n_byte_groups, b, g, l, 0);
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
pub(crate) fn seq_into_native(seq: Vec<u8>, bits: usize, n_byte_groups: usize) -> Vec<u8> {
    let mut buf = seq;
    apply_native_transform(&mut buf, bits, n_byte_groups);
    buf
}

/// Apply this geometry's stored-to-native transform in place, chunked for
/// parallelism. No-op when the target's native layout is the stored one.
///
/// Each chunk is block-aligned so lanes never cross a chunk boundary.
/// Serial for small payloads (thread-spawn overhead dominates below ~4 MB —
/// measured, see hypothesis log H2).
pub(crate) fn apply_native_transform(buf: &mut [u8], bits: usize, n_byte_groups: usize) {
    use rayon::prelude::*;
    debug_assert_eq!(buf.len() % BLOCK, 0);
    const PAR_THRESHOLD: usize = 4 * 1024 * 1024;
    const CHUNK: usize = 2 * 1024 * 1024; // multiple of BLOCK and VM_UNIT
    let Some(f) = native_transform(bits, n_byte_groups) else {
        return;
    };
    if buf.len() >= PAR_THRESHOLD {
        buf.par_chunks_mut(CHUNK).for_each(f);
    } else {
        f(buf);
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
    pack_blocked_native!(n_range, range_blocks, bits, n_byte_groups, blocked_size, &codes_flat)
}

/// Byte `group` of lane `lane` in the sequential-blocked block starting
/// at `base` — the O(dim) row gather the non-x86 `seq_row` arm uses.
/// Kept cfg-free so every arch compiles and unit-tests the exact
/// arithmetic; x86's `seq_row` uses the nibble de-interleave instead.
// On x86 the lib target never calls this (`seq_row` de-interleaves
// nibbles instead); it exists there for the cross-arch unit test.
/// Byte `lane`'s value for byte-group `group` in the sequential block
/// at `base` — the O(dim) row gather the non-x86 `seq_row` arm uses.
/// Kept cfg-free so every arch compiles and unit-tests the exact
/// arithmetic; x86's `seq_row` uses the nibble de-interleave instead.
// The lib target's callers vary by layout era (vm-layout arms gather
// differently), so this can be dead in any one build — it exists for
// the cross-arch unit test, which pins the arithmetic everywhere.
#[allow(dead_code)]
#[inline]
pub(crate) fn seq_lane_byte(data: &[u8], base: usize, group: usize, lane: usize) -> u8 {
    data[base + group * BLOCK + lane]
}

/// Native search layout → sequential blocked layout — [`seq_into_native`]'s
/// inverse. Lets the write path serialize a warm in-memory blocked cache
/// without a full O(n·dim) repack from bit-planes.
pub(crate) fn native_to_seq(blocked: &[u8], bits: usize, n_byte_groups: usize) -> Vec<u8> {
    if vm8_for(bits, n_byte_groups) {
        let mut out = blocked.to_vec();
        vector_major8_to_seq_chunk(&mut out);
        return out;
    }
    if vector_major_for(bits, n_byte_groups) {
        let mut out = blocked.to_vec();
        vector_major_to_seq_chunk(&mut out);
        return out;
    }
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
pub(crate) fn interleave_chunk_x86(buf: &mut [u8]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: gated on runtime AVX2 detection.
        unsafe { interleave_chunk_avx2(buf) }
    } else if is_x86_feature_detected!("ssse3") {
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

/// Two blocks per iteration on AVX2. The shuffle is per-128-bit-lane, so
/// the same 16-byte `perm0` vector serves both lanes; the only extra work
/// versus the SSSE3 kernel is four `permute2x128`s to gather the two
/// blocks' lo halves into one register and their hi halves into the
/// other, and to scatter the results back. Everything else — the shuffle,
/// the nibble merge, the loads and stores — happens once per two blocks
/// instead of once per block.
///
/// Bit-identical to [`interleave_chunk_ssse3`] by construction, and
/// `avx2_interleave_matches_ssse3` asserts it over a payload that
/// exercises both the paired path and the odd-block tail.
///
/// SAFETY: caller must ensure AVX2 is available. `buf.len()` is a
/// multiple of `BLOCK` (callers uphold this).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interleave_chunk_avx2(buf: &mut [u8]) {
    use std::arch::x86_64::*;
    let perm: [u8; 16] = std::array::from_fn(|j| PERM0[j] as u8);
    let permv = _mm256_broadcastsi128_si256(_mm_loadu_si128(perm.as_ptr() as *const __m128i));
    let lo_mask = _mm256_set1_epi8(0x0Fu8 as i8);
    let hi_mask = _mm256_set1_epi8(0xF0u8 as i8);
    let n = buf.len() / BLOCK;
    let pairs = n / 2;
    for i in 0..pairs {
        let p = buf.as_mut_ptr().add(i * 2 * BLOCK);
        // ~4 KB ahead, the same distance the SSSE3 kernel settled on.
        if (i * 2 + 128) * BLOCK < buf.len() {
            _mm_prefetch(buf.as_ptr().add((i * 2 + 128) * BLOCK) as *const i8, _MM_HINT_T0);
        }
        let v0 = _mm256_loadu_si256(p as *const __m256i);
        let v1 = _mm256_loadu_si256(p.add(BLOCK) as *const __m256i);
        // [b0.lo | b1.lo] and [b0.hi | b1.hi].
        let a = _mm256_shuffle_epi8(_mm256_permute2x128_si256(v0, v1, 0x20), permv);
        let b = _mm256_shuffle_epi8(_mm256_permute2x128_si256(v0, v1, 0x31), permv);
        let a_hi = _mm256_and_si256(_mm256_srli_epi16(a, 4), lo_mask);
        let out_hi = _mm256_or_si256(a_hi, _mm256_and_si256(b, hi_mask));
        let b_lo4 = _mm256_and_si256(b, lo_mask);
        let out_lo = _mm256_or_si256(
            _mm256_and_si256(a, lo_mask),
            _mm256_slli_epi16(b_lo4, 4),
        );
        // Back to block order: [b0.out_hi | b0.out_lo], [b1.out_hi | b1.out_lo].
        _mm256_storeu_si256(p as *mut __m256i, _mm256_permute2x128_si256(out_hi, out_lo, 0x20));
        _mm256_storeu_si256(
            p.add(BLOCK) as *mut __m256i,
            _mm256_permute2x128_si256(out_hi, out_lo, 0x31),
        );
    }
    if n % 2 == 1 {
        // SAFETY: AVX2 implies SSSE3, and this is the final whole block.
        unsafe { interleave_chunk_ssse3(&mut buf[pairs * 2 * BLOCK..]) }
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

    /// The AVX2 interleave processes two blocks per iteration and must
    /// agree with the SSSE3 kernel byte for byte — it is the same
    /// transform, only wider, and the two run on the same machines
    /// depending only on feature detection. The payload is an odd number
    /// of blocks so the tail path (which falls back to SSSE3) is covered
    /// alongside the paired path.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_interleave_matches_ssse3() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("ssse3") {
            return; // nothing to compare on this host
        }
        const N_BLOCKS: usize = 101;
        let mut s = 0x9E37_79B9u32;
        let src: Vec<u8> = (0..N_BLOCKS * BLOCK)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect();
        let mut wide = src.clone();
        let mut narrow = src.clone();
        // SAFETY: both gated on the detection above.
        unsafe { super::interleave_chunk_avx2(&mut wide) };
        unsafe { super::interleave_chunk_ssse3(&mut narrow) };
        assert_eq!(
            wide.iter().zip(&narrow).position(|(a, b)| a != b),
            None,
            "AVX2 interleave diverged from the SSSE3 kernel",
        );
        assert_ne!(wide, src, "fixture must actually be transformed");
    }

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

    /// The vm8 transform must place each sequential code byte exactly where
    /// `vm8_byte_index` says it lives — the SMMLA kernel and the write-path
    /// inverse both navigate by that formula, so the transform and the index
    /// map are pinned against each other byte-for-byte. Two units, so the
    /// `g / 8` unit stride is exercised, not just the intra-unit terms.
    #[test]
    fn vm8_transform_matches_its_byte_index_map() {
        use super::{vector_major8_chunk, vm8_byte_index, VM8_UNIT};
        let n_byte_groups = 16usize; // two vm8 units of one block
        let seq: Vec<u8> = (0..n_byte_groups * BLOCK).map(|i| (i % 251) as u8).collect();
        let mut vm = seq.clone();
        vector_major8_chunk(&mut vm);
        assert_ne!(vm, seq, "fixture must actually be transformed");
        for g in 0..n_byte_groups {
            for lane in 0..BLOCK {
                assert_eq!(
                    vm[vm8_byte_index(0, g, lane)],
                    seq[g * BLOCK + lane],
                    "mismatch at group {g}, lane {lane}",
                );
            }
        }
        // VM8_UNIT is the whole story of the `g / 8` term: byte-group 8 of
        // lane 0 must land exactly one unit after byte-group 0's.
        assert_eq!(vm8_byte_index(0, 8, 0) - vm8_byte_index(0, 0, 0), VM8_UNIT);
    }

    /// `vector_major8_to_seq_chunk` documents itself as the exact inverse of
    /// `vector_major8_chunk`; round-trip a multi-unit pseudo-random buffer
    /// so any slip in either direction's index arithmetic breaks the pair.
    #[test]
    fn vm8_to_seq_is_the_exact_inverse() {
        use super::{vector_major8_chunk, vector_major8_to_seq_chunk, VM8_UNIT};
        let mut s = 0x5F37_59DFu32;
        let seq: Vec<u8> = (0..3 * VM8_UNIT)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect();
        let mut buf = seq.clone();
        vector_major8_chunk(&mut buf);
        assert_ne!(buf, seq, "fixture must actually be transformed");
        vector_major8_to_seq_chunk(&mut buf);
        assert_eq!(buf, seq, "vm8 -> seq must invert seq -> vm8 exactly");
    }

    /// The layout predicates and the load-time transform must agree about
    /// what is in memory — the search dispatch navigates by the predicates,
    /// the loader by `native_transform`. On a host whose kernels read a
    /// vector-major layout the transform's output must match the byte map
    /// the corresponding predicate selects; a predicate that goes quiet
    /// (`vector_major_for` mutated to `false`) leaves the transform on the
    /// classic layout and this map check fails.
    #[test]
    fn native_transform_lands_bytes_where_the_selected_predicate_says() {
        use super::{apply_native_transform, vector_major_for, vm8_for, vm8_byte_index, vm_byte_index};
        let (bits, n_byte_groups) = (4usize, 16usize);
        let seq: Vec<u8> = (0..n_byte_groups * BLOCK).map(|i| (i % 249) as u8).collect();
        let mut native = seq.clone();
        apply_native_transform(&mut native, bits, n_byte_groups);
        if vm8_for(bits, n_byte_groups) {
            for g in 0..n_byte_groups {
                for lane in 0..BLOCK {
                    assert_eq!(native[vm8_byte_index(0, g, lane)], seq[g * BLOCK + lane]);
                }
            }
        } else if vector_major_for(bits, n_byte_groups) {
            for g in 0..n_byte_groups {
                for lane in 0..BLOCK {
                    assert_eq!(native[vm_byte_index(0, g, lane)], seq[g * BLOCK + lane]);
                }
            }
        } else if cfg!(target_arch = "x86_64") {
            // Classic x86: the perm0 nibble interleave, recovered per byte.
            for g in 0..n_byte_groups {
                for lane in 0..BLOCK {
                    assert_eq!(
                        deinterleave_x86_code_byte(&native, g * BLOCK, lane),
                        seq[g * BLOCK + lane],
                    );
                }
            }
        } else {
            assert_eq!(native, seq, "non-x86 classic layout is the stored one");
        }
        // The geometry gate is host-independent: a group count that is not
        // a multiple of 4 never takes the vector-major layout.
        assert!(!vector_major_for(bits, 3));
        assert!(!vm8_for(bits, 3));
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
            let (_, nbg, _) = super::blocked_geometry(n, bits, dim);
            assert_eq!(
                super::seq_into_native(seq.clone(), bits, nbg),
                native,
                "n={n} bits={bits} dim={dim}"
            );
            assert_eq!(
                super::native_to_seq(&native, bits, nbg),
                seq,
                "inverse n={n} bits={bits} dim={dim}"
            );
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

// =============================================================================
// Vector-major layout for the VNNI search kernel (x86_64)
// =============================================================================
//
// The `vpermb` + `vpdpbusd` kernel needs each aligned 4-byte group to belong
// to ONE vector, so that the dot product's 4-byte reduction sums four
// byte-groups' contributions for that vector rather than mixing four
// different vectors. That is the whole reason a dot-product instruction is
// usable here at all; see `benchmarks/hillclimb/LOG_search.md` (P11/P12).
//
// The permutation is local to 128 bytes — four byte-groups of 32 vectors —
// which divides every chunk size the loader uses, so it composes with the
// existing chunked/parallel read exactly as `interleave_chunk_x86` does.
//
// Within one 128-byte unit, source byte `j * 32 + v` (byte-group `j`, vector
// `v`) moves to `h * 64 + v_local * 4 + j`, where `h = v / 16` selects the
// 16-vector half that shares a zmm accumulator and `v_local = v % 16` is the
// dword lane within it. Unlike `interleave_chunk_x86` this moves whole bytes
// and never repacks nibbles, so the nibble meaning is unchanged: low = even
// dimension, high = odd.

/// Read vector `lane`'s code byte for byte-group `g` of block `b`.
///
/// Every in-place mutation path funnels through this and [`write_code`], so
/// the native layout is described in exactly one place. Adding a layout
/// means adding a branch here, not auditing `append_lanes`, `move_lane` and
/// `zero_lane` independently.
#[inline]
pub(crate) fn read_code(
    blocked: &[u8],
    bits: usize,
    n_byte_groups: usize,
    b: usize,
    g: usize,
    lane: usize,
) -> u8 {
    if vm8_for(bits, n_byte_groups) {
        return blocked[vm8_byte_index(b * n_byte_groups * BLOCK, g, lane)];
    }
    if vector_major_for(bits, n_byte_groups) {
        return blocked[vm_byte_index(b * n_byte_groups * BLOCK, g, lane)];
    }
    #[cfg(target_arch = "x86_64")]
    {
        deinterleave_x86_code_byte(blocked, (b * n_byte_groups + g) * BLOCK, lane)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        blocked[(b * n_byte_groups + g) * BLOCK + lane]
    }
}

/// Write vector `lane`'s code byte for byte-group `g` of block `b`.
/// See [`read_code`].
#[inline]
pub(crate) fn write_code(
    blocked: &mut [u8],
    bits: usize,
    n_byte_groups: usize,
    b: usize,
    g: usize,
    lane: usize,
    code: u8,
) {
    if vm8_for(bits, n_byte_groups) {
        blocked[vm8_byte_index(b * n_byte_groups * BLOCK, g, lane)] = code;
        return;
    }
    if vector_major_for(bits, n_byte_groups) {
        let i = vm_byte_index(b * n_byte_groups * BLOCK, g, lane);
        blocked[i] = code;
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        write_x86_code_byte(blocked, (b * n_byte_groups + g) * BLOCK, lane, code);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        blocked[(b * n_byte_groups + g) * BLOCK + lane] = code;
    }
}

/// Whether this process uses the vector-major code layout and the
/// dot-product search kernel that reads it.
///
/// The layout exists to serve a 4-byte-reducing integer dot product —
/// `vpdpbusd` on x86, `SDOT` on aarch64 — which sums four consecutive
/// byte-groups of one vector into that vector's own 32-bit lane. Both
/// instructions want the same thing in memory, so both arches share the
/// layout and differ only in the kernel that consumes it.
///
/// Decided once per process from CPU features, and it must be the SAME
/// answer at load time (which permutes the codes) and at search time
/// (which reads them) — otherwise one would write a layout the other
/// cannot read. A `OnceLock` guarantees that even if the environment
/// changes underneath us.
///
/// `TURBOVEC_NO_VECTOR_MAJOR=1` forces the classic layout, for A/B
/// measurement and as an escape hatch. `TURBOVEC_NO_VNNI=1` is accepted as
/// the older spelling from when this was x86-only.
///
/// Measured x1.233 on the x86 search cell (59.879 -> 48.576 ms at 200k x
/// 768 4-bit, nq=100); see `benchmarks/hillclimb/LOG_search.md` H21.
pub(crate) fn use_vector_major() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        if std::env::var("TURBOVEC_NO_VECTOR_MAJOR").is_ok_and(|v| v != "0")
            || std::env::var("TURBOVEC_NO_VNNI").is_ok_and(|v| v != "0")
        {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            // vbmi is no longer needed by the kernel itself — permute-dot
            // uses `vpshufb`, not `vpermb` — but the 2-bit path still
            // permutes, so the gate keeps it.
            is_x86_feature_detected!("avx512vbmi")
                && is_x86_feature_detected!("avx512vnni")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512f")
        }
        #[cfg(target_arch = "aarch64")]
        {
            // ARMv8.2-A dotprod. Mandatory from v8.4 and present on every
            // server core this targets, but optional in v8.2 itself, so it
            // is detected rather than assumed.
            std::arch::is_aarch64_feature_detected!("dotprod")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    })
}

/// The stored-to-native transform for this geometry: vector-major when the
/// dot-product kernel will read it, otherwise the arch's classic layout —
/// the perm0 interleave on x86, and identity on aarch64, whose classic
/// layout is the stored one. Chunk sizes used by the loader (2 MB, and 256
/// KB for the fused read) are multiples of every unit involved, so any of
/// them composes with chunked parallel reads.
pub(crate) fn native_transform(bits: usize, n_byte_groups: usize) -> Option<fn(&mut [u8])> {
    if vm8_for(bits, n_byte_groups) {
        return Some(vector_major8_chunk);
    }
    if vector_major_for(bits, n_byte_groups) {
        return Some(vector_major_chunk);
    }
    #[cfg(target_arch = "x86_64")]
    {
        Some(interleave_chunk_x86)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

/// Whether THIS index's geometry uses the vector-major layout.
///
/// The unit is 4 byte-groups, so a geometry with a group count that is not a
/// multiple of 4 keeps the classic layout and kernel. Both the load-time
/// transform and the search dispatch call this with the same arguments, so
/// they cannot disagree about what is in memory.
///
/// `bits` matters because the layout may only be *written* where a kernel
/// exists to *read* it. x86 has one at every supported width — permute-dot
/// at 4 bits, the `vpermb` LUT scan below that. aarch64 has only
/// permute-dot, which needs one code per nibble and so only applies at 4
/// bits; at 2 bits a nibble spans two dimensions and the nibble -> level map
/// stops being shared across them. Producing the layout there would leave
/// the codes in an order no NEON kernel reads, which does not fail loudly —
/// it silently mis-scores.
#[inline]
pub(crate) fn vector_major_for(bits: usize, n_byte_groups: usize) -> bool {
    let kernel_exists = cfg!(target_arch = "x86_64") || bits == 4;
    kernel_exists && use_vector_major() && n_byte_groups % 4 == 0
}

/// Byte index of vector `lane`'s code for byte-group `g`, in the
/// vector-major layout. Unlike the perm0 layout this is a whole byte and
/// needs no nibble surgery — it is the code byte exactly as stored.
#[inline]
pub(crate) fn vm_byte_index(block_base: usize, g: usize, lane: usize) -> usize {
    block_base + (g / 4) * 128 + (lane / 16) * 64 + (lane % 16) * 4 + (g % 4)
}

/// Bytes per vector-major unit: 4 byte-groups x 32 vectors.
pub(crate) const VM_UNIT: usize = 4 * BLOCK;

/// Bytes per *wide* vector-major unit: 8 byte-groups x 32 vectors.
///
/// The `vm8` variant exists to delete the two ZIPs from the aarch64 SMMLA
/// kernel, which P23 measured at x1.12. `SMMLA` reads bytes 0-7 of its B
/// operand as one vector's eight dimensions and 8-15 as the next vector's.
/// The 4-group unit puts *four* vectors in a 16-byte register, so bytes 0-7
/// straddle two of them and a ZIP is needed to regroup. Eight groups put two
/// vectors in the register instead — 8 byte-groups each — so the TBL output
/// *is* the operand.
///
/// The dimensions within an operand are the eight even (or eight odd) ones
/// rather than eight consecutive, because a byte still pairs dims `2g` and
/// `2g+1` — that pairing is the stored format. It costs nothing: `SMMLA`
/// sums over whatever index pairing A and B agree on, and A is built here
/// (see `search::build_smmla_a`), so it is matched rather than corrected.
pub(crate) const VM8_UNIT: usize = 8 * BLOCK;

/// Byte index of vector `lane`'s code for byte-group `g` in the `vm8`
/// layout: register `lane/2` holds lanes `2r`, `2r+1`, each contributing
/// eight consecutive byte-groups.
#[inline]
pub(crate) fn vm8_byte_index(block_base: usize, g: usize, lane: usize) -> usize {
    block_base + (g / 8) * VM8_UNIT + (lane / 2) * 16 + (lane % 2) * 8 + (g % 8)
}

/// Whether this build and CPU want the `vm8` layout.
///
/// aarch64 with i8mm only: it exists for the SMMLA kernel and no other
/// kernel reads it. On x86 the same arrangement would split each vector
/// across two dword lanes, doubling `vpdpbusd`'s accumulator count and
/// spilling — see LOG_search.md H41.
pub(crate) fn use_vm8() -> bool {
    cfg!(target_arch = "aarch64") && crate::search::have_i8mm_layout() && use_vector_major()
}

/// Whether THIS index's geometry uses `vm8`. Needs 8 groups per unit, so a
/// geometry that is a multiple of 4 but not 8 keeps the classic unit.
#[inline]
pub(crate) fn vm8_for(bits: usize, n_byte_groups: usize) -> bool {
    bits == 4 && use_vm8() && n_byte_groups % 8 == 0
}

/// Sequential blocked -> `vm8`, in place over whole [`VM8_UNIT`]s.
pub(crate) fn vector_major8_chunk(buf: &mut [u8]) {
    debug_assert_eq!(buf.len() % VM8_UNIT, 0);
    let mut tmp = [0u8; VM8_UNIT];
    for unit in buf.chunks_exact_mut(VM8_UNIT) {
        tmp.copy_from_slice(unit);
        for j in 0..8 {
            for v in 0..BLOCK {
                unit[(v / 2) * 16 + (v % 2) * 8 + j] = tmp[j * BLOCK + v];
            }
        }
    }
}

/// `vm8` -> sequential blocked. Exact inverse of [`vector_major8_chunk`].
pub(crate) fn vector_major8_to_seq_chunk(buf: &mut [u8]) {
    debug_assert_eq!(buf.len() % VM8_UNIT, 0);
    let mut tmp = [0u8; VM8_UNIT];
    for unit in buf.chunks_exact_mut(VM8_UNIT) {
        tmp.copy_from_slice(unit);
        for j in 0..8 {
            for v in 0..BLOCK {
                unit[j * BLOCK + v] = tmp[(v / 2) * 16 + (v % 2) * 8 + j];
            }
        }
    }
}

/// Sequential blocked -> vector-major, in place over whole `VM_UNIT`s.
pub(crate) fn vector_major_chunk(buf: &mut [u8]) {
    debug_assert_eq!(buf.len() % VM_UNIT, 0);
    let mut tmp = [0u8; VM_UNIT];
    for unit in buf.chunks_exact_mut(VM_UNIT) {
        tmp.copy_from_slice(unit);
        for j in 0..4 {
            for v in 0..BLOCK {
                unit[(v / 16) * 64 + (v % 16) * 4 + j] = tmp[j * BLOCK + v];
            }
        }
    }
}

/// Vector-major -> sequential blocked. Exact inverse of
/// [`vector_major_chunk`], used by the write path to reconstruct the stored
/// arch-neutral layout.
pub(crate) fn vector_major_to_seq_chunk(buf: &mut [u8]) {
    debug_assert_eq!(buf.len() % VM_UNIT, 0);
    let mut tmp = [0u8; VM_UNIT];
    for unit in buf.chunks_exact_mut(VM_UNIT) {
        tmp.copy_from_slice(unit);
        for j in 0..4 {
            for v in 0..BLOCK {
                unit[j * BLOCK + v] = tmp[(v / 16) * 64 + (v % 16) * 4 + j];
            }
        }
    }
}

#[cfg(test)]
mod vector_major_tests {
    use super::*;

    /// The transform must be a permutation and its inverse must restore the
    /// input exactly — anything else silently mis-scores every query.
    #[test]
    fn vector_major_round_trips() {
        for units in [1usize, 3, 8] {
            let n = units * VM_UNIT;
            let orig: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
            let mut buf = orig.clone();
            vector_major_chunk(&mut buf);
            assert_ne!(buf, orig, "transform should move bytes");
            vector_major_to_seq_chunk(&mut buf);
            assert_eq!(buf, orig, "inverse must restore the input exactly");
        }
    }

    /// Every source byte must appear exactly once: a permutation, not a
    /// gather that drops or duplicates lanes.
    #[test]
    fn vector_major_is_a_permutation() {
        let mut buf: Vec<u8> = (0..VM_UNIT).map(|i| i as u8).collect();
        vector_major_chunk(&mut buf);
        let mut seen = buf.clone();
        seen.sort_unstable();
        let want: Vec<u8> = (0..VM_UNIT).map(|i| i as u8).collect();
        assert_eq!(seen, want);
    }

    /// Byte `j*32 + v` must land where the kernel expects to read it:
    /// half `v/16`, dword lane `v%16`, byte position `j`.
    #[test]
    fn vector_major_places_bytes_where_the_kernel_reads_them() {
        let mut buf = vec![0u8; VM_UNIT];
        for j in 0..4 {
            for v in 0..BLOCK {
                buf[j * BLOCK + v] = (j * BLOCK + v) as u8;
            }
        }
        vector_major_chunk(&mut buf);
        for j in 0..4 {
            for v in 0..BLOCK {
                let at = (v / 16) * 64 + (v % 16) * 4 + j;
                assert_eq!(buf[at], (j * BLOCK + v) as u8, "j={j} v={v}");
            }
        }
    }
}
