//! Encode vectors: normalize, rotate, calibrate, quantize, bit-pack, scale.
//!
//! For each vector `v` with rotated unit form `u` and reconstructed
//! centroid vector `x_hat`, the stored scale is `||v|| / <u, x_hat>` —
//! the RaBitQ-style length-renormalization correction adapted to
//! turbovec's Lloyd-Max codebook. Applying this scale at the final
//! score-multiplication site in the SIMD kernel gives an unbiased
//! estimator of `<v, q>`.
//!
//! # TQ+ per-coordinate calibration
//!
//! After random rotation, each coord *should* follow the canonical
//! Beta((d-1)/2, (d-1)/2) marginal that Lloyd-Max was fit against. In
//! practice, anisotropic data leaves residual deviation per coord, and
//! the shared codebook then mis-fits. TQ+ corrects this with two free
//! parameters per coord — a `shift` and a `scale` — chosen to map the
//! empirical 5/95% quantiles of that coord onto the canonical Beta
//! marginal's 5/95% quantiles:
//!
//! ```text
//! u_calibrated[d] = (u_rot[d] + shift[d]) * scale_tq[d]
//! ```
//!
//! Quantization runs on `u_calibrated`; the search path applies the
//! inverse on the query side (`q_calib[d] = q_rot[d] / scale_tq[d]`)
//! plus a per-query bias correction `-<q_rot, shift>`. Net effect:
//! same kernel, same code, better-matched codebook.

use std::cmp::Ordering;

use rayon::prelude::*;
use statrs::distribution::{Beta, ContinuousCDF};

use crate::rotation::Rotation;

/// Parallel invalid-coordinate scan backing
/// [`crate::first_invalid_coord`]. Fixed chunks reduced by minimum flat
/// index, so the reported (vector, coord, value) is identical to a
/// left-to-right scan; the all-clean case (every call on the hot add
/// path) is one streaming pass split across the current rayon pool.
pub(crate) fn par_first_invalid_coord(
    values: &[f32],
    dim: usize,
    max_magnitude: f32,
) -> Option<(usize, usize, f32)> {
    const VALIDATE_CHUNK: usize = 64 * 1024;
    let first = values
        .par_chunks(VALIDATE_CHUNK)
        .enumerate()
        .filter_map(|(ci, chunk)| {
            first_invalid_in_chunk(chunk, max_magnitude).map(|j| ci * VALIDATE_CHUNK + j)
        })
        .min()?;
    let x = values[first];
    let vector_index = if dim == 0 { 0 } else { first / dim };
    let coord_index = if dim == 0 { first } else { first % dim };
    Some((vector_index, coord_index, x))
}

/// Position of the first invalid element in `chunk`, or `None`.
///
/// The predicate is `!(|x| < max_magnitude)` — identical to
/// `!x.is_finite() || x.abs() >= max_magnitude`: NaN fails every
/// comparison, ±Inf and over-magnitude values fail `<`. On aarch64 the
/// all-clean fast path tests 4 lanes per `vcalt`; a failing quad falls
/// back to a scalar scan so the reported index matches the scalar path
/// exactly.
#[cfg(target_arch = "aarch64")]
#[inline]
fn first_invalid_in_chunk(chunk: &[f32], max_magnitude: f32) -> Option<usize> {
    use std::arch::aarch64::*;
    let n = chunk.len();
    let quads = n / 4;
    unsafe {
        let bound = vdupq_n_f32(max_magnitude);
        for q in 0..quads {
            let x = vld1q_f32(chunk.as_ptr().add(q * 4));
            // Lane is all-ones iff |x| < bound (false for NaN/Inf/huge).
            let ok = vcaltq_f32(x, bound);
            if vminvq_u32(ok) == 0 {
                // Some lane failed — pinpoint with the scalar predicate.
                for j in q * 4..n {
                    let v = chunk[j];
                    if !(v.abs() < max_magnitude) {
                        return Some(j);
                    }
                }
                unreachable!("vector scan flagged a quad with no invalid element");
            }
        }
        for j in quads * 4..n {
            let v = chunk[j];
            if !(v.abs() < max_magnitude) {
                return Some(j);
            }
        }
    }
    None
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn first_invalid_in_chunk(chunk: &[f32], max_magnitude: f32) -> Option<usize> {
    chunk
        .iter()
        .position(|x| !x.is_finite() || x.abs() >= max_magnitude)
}

/// Quantile pair used to fit per-coord `(shift, scale)`.
const TQPLUS_P_LO: f64 = 0.05;
const TQPLUS_P_HI: f64 = 0.95;

/// Below this many input vectors, per-coord quantile estimates are too
/// noisy to be useful — fall back to identity calibration. Empirical
/// floor: at ~200 samples the calibration noise eats the precision gain
/// (4-bit vs 2-bit stddev becomes statistically indistinguishable). At
/// ~1000 samples calibration is stable enough that the 4-bit gain
/// reasserts itself; pick 1000 with a small safety margin.
const TQPLUS_MIN_SAMPLES: usize = 1000;

/// Encode n vectors of dimension dim.
///
/// `existing_calibration`, when `Some`, locks the (shift, scale_tq) used for
/// this batch — pass it on subsequent `.add()` calls so the new batch is
/// quantized with the same calibration as earlier data. When `None`, fits a
/// fresh calibration from this batch's empirical quantiles.
///
/// Appends the packed codes and per-vector scales for this batch to
/// `packed_out` / `scales_out` (existing contents untouched) and
/// returns (shift_fitted, scale_tq_fitted). The calibration pair is
/// non-empty only when this call fitted it (i.e.
/// `existing_calibration` was `None`); on the reuse path the caller
/// already owns the calibration and the returned pair is empty.
///
/// Crate-internal: trusts that `vectors.len() == n * dim`, that
/// `rotation`/`boundaries`/`centroids` are correctly shaped for `dim` and
/// `bit_width`, and (asserted below) that `dim` is a nonzero multiple of 8.
/// The high-level index types establish these before calling; external
/// callers build a validated index via
/// [`from_parts`](crate::TurboQuantIndex::from_parts) or
/// [`TurboQuantIndex::add`](crate::TurboQuantIndex::add) instead.
///
/// # Panics
///
/// Panics if `dim` is zero or not a multiple of 8 — the packed layout
/// allocates `dim / 8` bytes per bit-plane, so no other dim has a valid
/// layout. (`TurboQuantIndex` enforces the same rule at construction.)
pub(crate) fn encode(
    vectors: &[f32],
    n: usize,
    dim: usize,
    rotation: &Rotation,
    boundaries: &[f32],
    centroids: &[f32],
    bit_width: usize,
    existing_calibration: Option<(&[f32], &[f32])>,
    rotated_scratch: &mut Vec<f32>,
    packed_out: &mut Vec<u8>,
    scales_out: &mut Vec<f32>,
) -> (Vec<f32>, Vec<f32>) {
    // The packed layout allocates `dim / 8` bytes per bit-plane, so a dim
    // that is not a multiple of 8 has no valid layout: the tail
    // coordinates would write past the end of each plane (top plane
    // panics, lower planes silently corrupt the next plane's bytes —
    // #117). TurboQuantIndex enforces this at construction; enforce it
    // here too for direct callers of the public function.
    assert!(
        dim != 0 && dim % 8 == 0,
        "encode requires dim to be a nonzero multiple of 8, got {dim}",
    );
    // Norms only — the normalization scale rides the first rotation
    // gather (`Rotation::apply_scaled_into`), so the unit-normalized
    // intermediate copy of the batch is never materialized. The fused
    // path performs the identical multiplies in the identical order, so
    // encoded bytes are unchanged.
    let mut norms = vec![0.0f32; n];

    // Rotate each raw row into `rotated_buf` via the deterministic
    // block-Hadamard transform, applying 1/||v|| in the first gather.
    // Rows are independent so rayon splits them across cores; the
    // per-row transform is reduction-free (fixed add order, no FMA), so
    // the encoded bytes are identical regardless of how rows are
    // distributed across threads — the property the QR rotation lacked
    // (#206).
    let rotated_buf: &mut Vec<f32> = rotated_scratch;
    rotated_buf.clear();
    rotated_buf.reserve(n * dim);
    #[allow(clippy::uninit_vec)]
    // SAFETY: f32 has no invalid bit patterns, and every element is
    // written by apply_scaled_into (each output row is fully written)
    // before `rotated_buf` is read. Reusing the caller's scratch keeps
    // the allocation warm across adds instead of paying a fresh
    // multi-MB mmap + page-fault walk per call.
    unsafe {
        rotated_buf.set_len(n * dim);
    }
    // The norm is computed in the same per-row task as the rotation —
    // the row is already in cache, so the separate full-batch norms
    // pass disappears. Same per-row operations, identical bytes.
    rotated_buf
        .par_chunks_mut(dim)
        .zip(norms.par_iter_mut())
        .enumerate()
        .for_each_init(|| vec![0.0f32; dim], |scratch, (i, (dst_row, norm))| {
            let src = &vectors[i * dim..(i + 1) * dim];
            let n_val = simd_norm(src);
            *norm = n_val;
            let inv = if n_val > 1e-10 { 1.0 / n_val } else { 0.0 };
            rotation.apply_scaled_into(src, inv, dst_row, scratch)
        });
    let rotated: &[f32] = rotated_buf;

    // TQ+ per-coord (shift, scale) — fitted to empirical quantiles of the
    // rotated batch, or reused from a previous add for consistency across
    // incremental encodes.
    // Borrow an existing (frozen) calibration rather than cloning it —
    // the warm add path hits this on every call, and the caller already
    // owns the vectors. Freshly fitted calibration is owned here and
    // returned; on the borrow path the returned pair is empty and the
    // caller keeps its stored calibration unchanged.
    let fitted: (Vec<f32>, Vec<f32>);
    let (shift, scale_tq): (&[f32], &[f32]) = match existing_calibration {
        Some((s, sc)) => {
            assert_eq!(s.len(), dim, "existing shift length must equal dim");
            assert_eq!(sc.len(), dim, "existing scale_tq length must equal dim");
            fitted = (Vec::new(), Vec::new());
            (s, sc)
        }
        None => {
            fitted = compute_tqplus_calibration(rotated, n, dim);
            (&fitted.0, &fitted.1)
        }
    };

    // Precompute 1/scale_tq for the inner-product reconstruction inside the
    // fused per-row function. Avoids a divide per coord per vector.
    let inv_scale_tq: Vec<f32> = scale_tq.iter().map(|s| 1.0 / s).collect();

    // Hoist the reconstruction operand out of the per-row loop: for a
    // given code c and coordinate d, `centroids[c] * inv_scale_tq[d] -
    // shift[d]` is row-independent. The table entries are computed with
    // exactly the ops the kernel performed per element, so the
    // accumulated inner products — and the stored scales — are
    // bit-identical.
    // The table costs O(2^bits * dim) to build, so it only pays once the
    // batch is a few rows deep; below that the kernel computes the same
    // values inline (identical ops, identical results).
    const RECON_TABLE_MIN_ROWS: usize = 16;
    let centroid_orig: Option<Vec<f64>> = (n >= RECON_TABLE_MIN_ROWS).then(|| {
        let n_codes = 1usize << bit_width;
        let mut table = vec![0.0f64; n_codes * dim];
        for c in 0..n_codes {
            let row = &mut table[c * dim..(c + 1) * dim];
            for d in 0..dim {
                row[d] =
                    (centroids[c] as f64) * (inv_scale_tq[d] as f64) - (shift[d] as f64);
            }
        }
        table
    });

    let bytes_per_plane = dim / 8;
    let bytes_per_row = bit_width * bytes_per_plane;
    // Append-in-place: the batch's rows land directly at the tail of the
    // caller's buffers, so no per-call output allocation and no
    // extend_from_slice copy afterwards.
    let packed_old = packed_out.len();
    let scales_old = scales_out.len();
    #[cfg(target_arch = "aarch64")]
    {
        // The NEON kernel stores every byte of each packed row (one store
        // per plane per 8-coord chunk), so the zero-fill is dead work.
        // SAFETY: u8 has no invalid bit patterns, and fused_quantize_
        // scale_pack overwrites all bytes_per_row bytes of every row
        // before the region is read.
        packed_out.reserve(n * bytes_per_row);
        #[allow(clippy::uninit_vec)]
        unsafe {
            packed_out.set_len(packed_old + n * bytes_per_row);
        }
    }
    // The scalar fallback ORs bits into the packed row, so it needs the
    // zero-filled region.
    #[cfg(not(target_arch = "aarch64"))]
    packed_out.resize(packed_old + n * bytes_per_row, 0u8);
    scales_out.resize(scales_old + n, 0.0f32);
    let packed = &mut packed_out[packed_old..];
    let scales = &mut scales_out[scales_old..];

    // Monomorphized per bit-width so the per-plane pack loop and the
    // boundary scan fully unroll (bit_width is validated to {2, 3, 4}
    // at construction). Identical operations, identical bytes.
    match bit_width {
        2 => quantize_batch::<2>(
            packed, scales, rotated, shift, scale_tq, &inv_scale_tq,
            centroid_orig.as_deref(), boundaries, centroids, &norms, dim,
            bytes_per_row, bytes_per_plane,
        ),
        3 => quantize_batch::<3>(
            packed, scales, rotated, shift, scale_tq, &inv_scale_tq,
            centroid_orig.as_deref(), boundaries, centroids, &norms, dim,
            bytes_per_row, bytes_per_plane,
        ),
        4 => quantize_batch::<4>(
            packed, scales, rotated, shift, scale_tq, &inv_scale_tq,
            centroid_orig.as_deref(), boundaries, centroids, &norms, dim,
            bytes_per_row, bytes_per_plane,
        ),
        other => unreachable!("unsupported bit_width {other}"),
    }

    fitted
}

/// Quantize + pack the whole batch with a compile-time bit width.
#[allow(clippy::too_many_arguments)]
fn quantize_batch<const BITS: usize>(
    packed: &mut [u8],
    scales: &mut [f32],
    rotated: &[f32],
    shift: &[f32],
    scale_tq: &[f32],
    inv_scale_tq: &[f32],
    centroid_orig: Option<&[f64]>,
    boundaries: &[f32],
    centroids: &[f32],
    norms: &[f32],
    dim: usize,
    bytes_per_row: usize,
    bytes_per_plane: usize,
) {
    packed.par_chunks_mut(bytes_per_row)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(i, (packed_row, scale))| {
            let rot_orig = &rotated[i * dim..(i + 1) * dim];
            *scale = fused_quantize_scale_pack::<BITS>(
                rot_orig, shift, scale_tq, inv_scale_tq,
                centroid_orig, boundaries, centroids, norms[i],
                packed_row, dim, bytes_per_plane,
            );
        });
}

/// Per-coordinate TQ+ calibration. For each of the `dim` rotated coordinates,
/// computes `(shift, scale)` such that `(x + shift) * scale` maps the empirical
/// (P_LO, P_HI) quantiles onto the canonical Beta((dim-1)/2, (dim-1)/2)
/// marginal's quantiles. When the batch is too small or a coord is
/// degenerate (constant or near-constant), falls back to identity.
fn compute_tqplus_calibration(
    rotated: &[f32],
    n: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut shift = vec![0.0f32; dim];
    let mut scale = vec![1.0f32; dim];

    if n < TQPLUS_MIN_SAMPLES {
        // Identity calibration — not enough samples for reliable quantile
        // estimates. Index still works, just without the TQ+ recall gain
        // for this batch.
        return (shift, scale);
    }

    let a = (dim as f64 - 1.0) / 2.0;
    let beta = Beta::new(a, a).expect("Beta(a, a) is valid for a > 0");
    // Beta is on [0, 1]; canonical marginal is shifted to [-1, 1].
    let qc_lo = (2.0 * beta.inverse_cdf(TQPLUS_P_LO) - 1.0) as f32;
    let qc_hi = (2.0 * beta.inverse_cdf(TQPLUS_P_HI) - 1.0) as f32;
    let qc_span = qc_hi - qc_lo;

    let lo_idx = ((n as f64) * TQPLUS_P_LO) as usize;
    let hi_idx = (((n as f64) * TQPLUS_P_HI) as usize).min(n - 1);

    // Coords are independent, but gathering one column at a time strides
    // `dim * 4` bytes per element — every read is a fresh cache line, and
    // the whole n*dim batch is re-streamed once per coordinate. Instead,
    // fan out over TILES of coordinates: each tile makes one sequential
    // pass over the rows, scattering into `tile` contiguous column
    // buffers (each row contributes a contiguous 4*tile-byte read). The
    // collected values per coord are identical, so the selected quantiles
    // — and every downstream encoded byte — are unchanged.
    // Tile size trades streaming passes against parallelism: each tile
    // re-streams the whole rotated batch once, but tiles are also the
    // unit of fan-out. Pick the largest power-of-two tile (<= 256) that
    // still yields ~2 tiles per rayon worker; single-threaded runs get
    // the full 512. The choice only affects scheduling — the collected
    // values per coordinate, and every encoded byte, are identical for
    // any tile size.
    let workers = rayon::current_num_threads().max(1);
    let mut tile_size = 512usize;
    while tile_size > 32 && dim / tile_size < 2 * workers {
        tile_size /= 2;
    }
    shift
        .par_chunks_mut(tile_size)
        .zip(scale.par_chunks_mut(tile_size))
        .enumerate()
        .for_each(|(tile_idx, (sh_tile, sc_tile))| {
            let d0 = tile_idx * tile_size;
            let tile = sh_tile.len();
            let mut cols = vec![0.0f32; tile * n];
            for i in 0..n {
                let row = &rotated[i * dim + d0..i * dim + d0 + tile];
                for (c, &v) in row.iter().enumerate() {
                    cols[c * n + i] = v;
                }
            }
            for (c, (sh, sc)) in sh_tile.iter_mut().zip(sc_tile.iter_mut()).enumerate() {
                let coord = &mut cols[c * n..(c + 1) * n];
                // Only the two quantile order statistics are needed, so
                // two O(n) selects replace a full O(n log n) sort. Select
                // the upper quantile first, then the lower one within the
                // left partition — the values are identical to indexing a
                // fully sorted array.
                let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(Ordering::Equal);
                let (left, hi_val, _) = coord.select_nth_unstable_by(hi_idx, cmp);
                let qe_hi = *hi_val;
                let (_, lo_val, _) = left.select_nth_unstable_by(lo_idx, cmp);
                let qe_lo = *lo_val;
                let qe_span = qe_hi - qe_lo;
                if qe_span > 1e-6 {
                    *sc = qc_span / qe_span;
                    *sh = qc_lo / *sc - qe_lo;
                }
                // else: leave as (shift=0, scale=1) for this coord
            }
        });

    (shift, scale)
}

// ─── Norm and scale (aarch64) ────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn simd_norm(row: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let dim = row.len();
    let chunks = dim / 4;
    let mut acc = unsafe { vdupq_n_f32(0.0) };

    unsafe {
        for c in 0..chunks {
            let v = vld1q_f32(row.as_ptr().add(c * 4));
            acc = vfmaq_f32(acc, v, v);
        }
        let mut sum = vaddvq_f32(acc);
        for j in (chunks * 4)..dim {
            sum += row[j] * row[j];
        }
        sum.sqrt()
    }
}

// ─── Norm and scale (fallback) ───────────────────────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn simd_norm(row: &[f32]) -> f32 {
    row.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ─── Fused quantize + scale + pack (aarch64) ────────────────────────────────

/// Process one row: quantize calibrated rotated values against boundaries,
/// accumulate the centroid inner product *in original (uncalibrated) space*
/// for the scale correction, and pack the resulting codes.
///
/// The inner-product reconstruction undoes the calibration so the stored
/// `scale[i] = ||v|| / <u_rot[i], x_hat_orig[i]>` matches what the search
/// path will compute when scoring queries (which also apply the inverse
/// calibration):
///
/// ```text
/// x_hat_orig[d] = centroids[code[d]] / scale_tq[d] - shift[d]
/// inner        = sum_d u_rot[d] * x_hat_orig[d]
///              = sum_d u_rot[d] * inv_scale_tq[d] * centroids[code[d]]
///                - sum_d u_rot[d] * shift[d]
/// ```
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fused_quantize_scale_pack<const BITS: usize>(
    rot_orig: &[f32],
    shift: &[f32],
    scale_tq: &[f32],
    inv_scale_tq: &[f32],
    centroid_orig: Option<&[f64]>,
    boundaries: &[f32],
    centroids: &[f32],
    norm: f32,
    packed_row: &mut [u8],
    dim: usize,
    bytes_per_plane: usize,
) -> f32 {
    use std::arch::aarch64::*;

    let mut inner = 0.0f64;
    let chunks = dim / 8;

    unsafe {
        for c in 0..chunks {
            let offset = c * 8;
            // Boundary scan on the CALIBRATED rotated values — TQ+ moves
            // each coord's empirical distribution onto the canonical Beta
            // marginal that Lloyd-Max was fit against. Calibration is
            // computed inline — `(x + shift) * scale_tq` per element, the
            // same IEEE ops the old batch-materialized buffer stored — so
            // no n*dim intermediate is allocated, written, and re-read.
            let vals_lo = vmulq_f32(
                vaddq_f32(
                    vld1q_f32(rot_orig.as_ptr().add(offset)),
                    vld1q_f32(shift.as_ptr().add(offset)),
                ),
                vld1q_f32(scale_tq.as_ptr().add(offset)),
            );
            let vals_hi = vmulq_f32(
                vaddq_f32(
                    vld1q_f32(rot_orig.as_ptr().add(offset + 4)),
                    vld1q_f32(shift.as_ptr().add(offset + 4)),
                ),
                vld1q_f32(scale_tq.as_ptr().add(offset + 4)),
            );

            let mut acc_lo = vdupq_n_u32(0);
            let mut acc_hi = vdupq_n_u32(0);

            for bi in 0..(1usize << BITS) - 1 {
                let bv = vdupq_n_f32(boundaries[bi]);
                acc_lo = vaddq_u32(acc_lo, vshrq_n_u32::<31>(vcgtq_f32(vals_lo, bv)));
                acc_hi = vaddq_u32(acc_hi, vshrq_n_u32::<31>(vcgtq_f32(vals_hi, bv)));
            }

            let counts: [u8; 8] = [
                vgetq_lane_u32::<0>(acc_lo) as u8,
                vgetq_lane_u32::<1>(acc_lo) as u8,
                vgetq_lane_u32::<2>(acc_lo) as u8,
                vgetq_lane_u32::<3>(acc_lo) as u8,
                vgetq_lane_u32::<0>(acc_hi) as u8,
                vgetq_lane_u32::<1>(acc_hi) as u8,
                vgetq_lane_u32::<2>(acc_hi) as u8,
                vgetq_lane_u32::<3>(acc_hi) as u8,
            ];

            // Inner-product reconstruction in ORIGINAL space (see doc
            // comment): from the hoisted per-(code, coord) table when the
            // batch amortized building it, otherwise inline — the same
            // ops per element, so results are bit-identical.
            match centroid_orig {
                Some(table) => {
                    for k in 0..8 {
                        let d = offset + k;
                        inner += (rot_orig[d] as f64)
                            * table[counts[k] as usize * dim + d];
                    }
                }
                None => {
                    for k in 0..8 {
                        let d = offset + k;
                        let centroid_in_orig = (centroids[counts[k] as usize] as f64)
                            * (inv_scale_tq[d] as f64)
                            - (shift[d] as f64);
                        inner += (rot_orig[d] as f64) * centroid_in_orig;
                    }
                }
            }

            // Pack 8 codes into one byte per bit-plane (unchanged).
            let codes_vec = vld1_u8(counts.as_ptr());
            let weights: [u8; 8] = [128, 64, 32, 16, 8, 4, 2, 1];
            let wv = vld1_u8(weights.as_ptr());

            for p in 0..BITS {
                let mask = vdup_n_u8(1u8 << p);
                let hit = vcgt_u8(vand_u8(codes_vec, mask), vdup_n_u8(0));
                packed_row[p * bytes_per_plane + offset / 8] = vaddv_u8(vand_u8(hit, wv));
            }
        }
        // No tail loop: `encode` asserts dim % 8 == 0, so `chunks * 8 == dim`.
        // (The old tail branch could never work — `bytes_per_plane = dim / 8`
        // truncates, so tail coordinates have no bytes to land in; see #117.)
    }

    scale_from_inner(inner, norm)
}

/// Degeneracy threshold for the reconstruction inner product.
///
/// `inner = <u_rot, x_hat>` is computed against the *unit-normalized*
/// rotated vector, so it is already norm-relative (cosine-like, ≈ 1 for a
/// healthy reconstruction regardless of the vector's magnitude). Measured
/// healthy minima stay above ~0.56 (dim 8 at 2 bits, the coarsest
/// supported config; every other measured config is ≥ 0.70, and
/// in-distribution data under fitted calibration sits at ≈ 1.0). By
/// contrast, reconstructions of vectors a frozen calibration cannot
/// represent collapse below ~0.06 on their way to the sign flip (#116).
/// 0.1 splits that gap: healthy vectors are untouched (their encode
/// output stays bit-identical), while every stored scale is bounded by
/// `norm / EPS`, capping score inflation at 10× the vector's true
/// magnitude instead of the old ~1e10 blowup.
const DEGENERATE_INNER_EPS: f64 = 0.1;

/// Convert the reconstruction inner product `<u_rot, x_hat>` into the stored
/// per-vector correction scale `||v|| / inner`.
///
/// A small or negative `inner` means the quantized reconstruction points
/// away from (or nearly orthogonal to) the vector — the codebook cannot
/// represent it under the current (possibly frozen) calibration, and any
/// finite scale would inflate its scores by `1 / inner`. The old
/// `inner.max(1e-10)` clamp turned a negative `inner` into a ~1e10 scale
/// with a flipped sign, letting a single out-of-distribution vector falsely
/// dominate every top-k (#116); a purely non-positive test would have left
/// the same explosion reachable through the open window just above zero.
/// Degenerate reconstructions (`inner <= DEGENERATE_INNER_EPS`) store scale
/// 0 instead so the vector scores ~0 and ranks last; this also preserves
/// the zero-vector behavior the clamp originally guarded (`norm == 0` ⇒
/// `inner == 0` ⇒ scale 0). The comparison is written positively so a NaN
/// `inner` (reachable only via direct `encode` calls with non-finite input,
/// which the index-level API rejects) lands in the degenerate branch rather
/// than poisoning the stored scale. Both the SIMD path and the scalar
/// fallback route through this helper, so the two stay in agreement; for
/// `inner > EPS` the result is bit-identical to the previous code.
#[inline(always)]
fn scale_from_inner(inner: f64, norm: f32) -> f32 {
    if inner > DEGENERATE_INNER_EPS {
        norm / inner as f32
    } else {
        0.0
    }
}

// ─── Fused quantize + scale + pack (fallback) ───────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn fused_quantize_scale_pack<const BITS: usize>(
    rot_orig: &[f32],
    shift: &[f32],
    scale_tq: &[f32],
    inv_scale_tq: &[f32],
    centroid_orig: Option<&[f64]>,
    boundaries: &[f32],
    centroids: &[f32],
    norm: f32,
    packed_row: &mut [u8],
    dim: usize,
    bytes_per_plane: usize,
) -> f32 {
    let mut inner = 0.0f64;

    for j in 0..dim {
        let calib = (rot_orig[j] + shift[j]) * scale_tq[j];
        let mut code = 0u8;
        for bi in 0..(1usize << BITS) - 1 {
            if calib > boundaries[bi] { code += 1; }
        }
        // Same table-or-inline split as the aarch64 kernel; identical
        // ops either way, so results are bit-identical.
        let centroid_in_orig = match centroid_orig {
            Some(table) => table[code as usize * dim + j],
            None => {
                (centroids[code as usize] as f64) * (inv_scale_tq[j] as f64)
                    - (shift[j] as f64)
            }
        };
        inner += (rot_orig[j] as f64) * centroid_in_orig;

        let byte_pos = j / 8;
        let bit_pos = 7 - (j % 8);
        for p in 0..BITS {
            if code & (1 << p) != 0 {
                packed_row[p * bytes_per_plane + byte_pos] |= 1 << bit_pos;
            }
        }
    }

    scale_from_inner(inner, norm)
}
