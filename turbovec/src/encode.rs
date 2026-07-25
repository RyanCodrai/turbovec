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

use ndarray::ArrayView2;
use rayon::prelude::*;
use statrs::distribution::{Beta, ContinuousCDF};

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
/// Returns (packed_codes, scales, shift_used, scale_tq_used).
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
    rotation: &[f32],
    boundaries: &[f32],
    centroids: &[f32],
    bit_width: usize,
    existing_calibration: Option<(&[f32], &[f32])>,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
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
    let mut norms = vec![0.0f32; n];
    let mut unit_flat = vec![0.0f32; n * dim];

    // Normalize. Rows are independent so Rayon splits them across cores.
    norms.par_iter_mut()
        .zip(unit_flat.par_chunks_mut(dim))
        .enumerate()
        .for_each(|(i, (norm, unit_row))| {
            let row = &vectors[i * dim..(i + 1) * dim];
            let n_val = simd_norm(row);
            *norm = n_val;
            let inv = if n_val > 1e-10 { 1.0 / n_val } else { 0.0 };
            simd_scale(row, inv, unit_row);
        });

    // Rotate.
    let unit_mat = ArrayView2::from_shape((n, dim), &unit_flat).unwrap();
    let rot_mat = ArrayView2::from_shape((dim, dim), rotation).unwrap();
    let rotated_mat = unit_mat.dot(&rot_mat.t());
    let rotated = rotated_mat.as_slice().unwrap();

    // TQ+ per-coord (shift, scale) — fitted to empirical quantiles of the
    // rotated batch, or reused from a previous add for consistency across
    // incremental encodes.
    let (shift, scale_tq) = match existing_calibration {
        Some((s, sc)) => {
            assert_eq!(s.len(), dim, "existing shift length must equal dim");
            assert_eq!(sc.len(), dim, "existing scale_tq length must equal dim");
            (s.to_vec(), sc.to_vec())
        }
        None => compute_tqplus_calibration(rotated, n, dim),
    };

    // Materialize calibrated rotated values for the boundary-scan path.
    // Memory cost: n * dim * 4 bytes (= same as `unit_flat`). At d=3072,
    // n=100k this is 1.2 GB — still well under the rotate intermediate
    // and encoding is one-shot, so we eat the allocation rather than
    // recompute inline per row.
    let mut rotated_calib = vec![0.0f32; n * dim];
    rotated_calib.par_chunks_mut(dim).enumerate().for_each(|(i, calib_row)| {
        let orig_row = &rotated[i * dim..(i + 1) * dim];
        for d in 0..dim {
            calib_row[d] = (orig_row[d] + shift[d]) * scale_tq[d];
        }
    });

    // Precompute 1/scale_tq for the inner-product reconstruction inside the
    // fused per-row function. Avoids a divide per coord per vector.
    let inv_scale_tq: Vec<f32> = scale_tq.iter().map(|s| 1.0 / s).collect();

    let bytes_per_plane = dim / 8;
    let bytes_per_row = bit_width * bytes_per_plane;
    let mut packed = vec![0u8; n * bytes_per_row];
    let mut scales = vec![0.0f32; n];

    packed.par_chunks_mut(bytes_per_row)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(i, (packed_row, scale))| {
            let rot_orig = &rotated[i * dim..(i + 1) * dim];
            let rot_calib = &rotated_calib[i * dim..(i + 1) * dim];
            *scale = fused_quantize_scale_pack(
                rot_orig, rot_calib, &shift, &inv_scale_tq,
                boundaries, centroids, norms[i],
                packed_row, dim, bit_width, bytes_per_plane,
            );
        });

    (packed, scales, shift, scale_tq)
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

    // Each coord is independent — fan out over coords.
    shift.par_iter_mut().zip(scale.par_iter_mut()).enumerate().for_each(
        |(d, (sh, sc))| {
            let mut coord: Vec<f32> = (0..n).map(|i| rotated[i * dim + d]).collect();
            coord.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let qe_lo = coord[lo_idx];
            let qe_hi = coord[hi_idx];
            let qe_span = qe_hi - qe_lo;
            if qe_span > 1e-6 {
                *sc = qc_span / qe_span;
                *sh = qc_lo / *sc - qe_lo;
            }
            // else: leave as (shift=0, scale=1) for this coord
        },
    );

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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn simd_scale(row: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let dim = row.len();
    let chunks = dim / 4;
    let sv = unsafe { vdupq_n_f32(scale) };

    unsafe {
        for c in 0..chunks {
            let v = vld1q_f32(row.as_ptr().add(c * 4));
            vst1q_f32(out.as_mut_ptr().add(c * 4), vmulq_f32(v, sv));
        }
        for j in (chunks * 4)..dim {
            out[j] = row[j] * scale;
        }
    }
}

// ─── Norm and scale (fallback) ───────────────────────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn simd_norm(row: &[f32]) -> f32 {
    row.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn simd_scale(row: &[f32], scale: f32, out: &mut [f32]) {
    for j in 0..row.len() {
        out[j] = row[j] * scale;
    }
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
fn fused_quantize_scale_pack(
    rot_orig: &[f32],
    rot_calib: &[f32],
    shift: &[f32],
    inv_scale_tq: &[f32],
    boundaries: &[f32],
    centroids: &[f32],
    norm: f32,
    packed_row: &mut [u8],
    dim: usize,
    bits: usize,
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
            // marginal that Lloyd-Max was fit against.
            let vals_lo = vld1q_f32(rot_calib.as_ptr().add(offset));
            let vals_hi = vld1q_f32(rot_calib.as_ptr().add(offset + 4));

            let mut acc_lo = vdupq_n_u32(0);
            let mut acc_hi = vdupq_n_u32(0);

            for &b in boundaries {
                let bv = vdupq_n_f32(b);
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

            // Inner-product reconstruction in ORIGINAL space (see doc comment).
            for k in 0..8 {
                let d = offset + k;
                let centroid_in_orig =
                    (centroids[counts[k] as usize] as f64) * (inv_scale_tq[d] as f64)
                        - (shift[d] as f64);
                inner += (rot_orig[d] as f64) * centroid_in_orig;
            }

            // Pack 8 codes into one byte per bit-plane (unchanged).
            let codes_vec = vld1_u8(counts.as_ptr());
            let weights: [u8; 8] = [128, 64, 32, 16, 8, 4, 2, 1];
            let wv = vld1_u8(weights.as_ptr());

            for p in 0..bits {
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
fn fused_quantize_scale_pack(
    rot_orig: &[f32],
    rot_calib: &[f32],
    shift: &[f32],
    inv_scale_tq: &[f32],
    boundaries: &[f32],
    centroids: &[f32],
    norm: f32,
    packed_row: &mut [u8],
    dim: usize,
    bits: usize,
    bytes_per_plane: usize,
) -> f32 {
    let mut inner = 0.0f64;

    for j in 0..dim {
        let mut code = 0u8;
        for &b in boundaries {
            if rot_calib[j] > b { code += 1; }
        }
        let centroid_in_orig =
            (centroids[code as usize] as f64) * (inv_scale_tq[j] as f64)
                - (shift[j] as f64);
        inner += (rot_orig[j] as f64) * centroid_in_orig;

        let byte_pos = j / 8;
        let bit_pos = 7 - (j % 8);
        for p in 0..bits {
            if code & (1 << p) != 0 {
                packed_row[p * bytes_per_plane + byte_pos] |= 1 << bit_pos;
            }
        }
    }

    scale_from_inner(inner, norm)
}
