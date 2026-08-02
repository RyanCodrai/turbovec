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
//! parameters per coord — a `shift` and a `scale` — chosen to map that
//! coord's empirical quantiles onto the codebook's outermost centroids.
//! The probability level is taken from the codebook itself (see
//! [`tqplus_anchor`]) rather than fixed, because the level the outermost
//! centroid sits at moves with bit width:
//!
//! ```text
//! u_calibrated[d] = (u_rot[d] + shift[d]) * scale_tq[d]
//! ```
//!
//! Quantization runs on `u_calibrated`; the search path applies the
//! inverse on the query side (`q_calib[d] = q_rot[d] / scale_tq[d]`)
//! plus a per-query bias correction `-<q_rot, shift>`. Net effect:
//! same kernel, same code, better-matched codebook.

use rayon::prelude::*;
use statrs::distribution::{Beta, ContinuousCDF};

use crate::rotation::Rotation;

/// Chunk length of the invalid-coordinate scan. Public through
/// [`crate::validation_parallelizes`] because the Python binding must know
/// whether a given input splits: at or below one chunk the scan folds on the
/// calling thread and enters no pool, above it rayon injects work into
/// whatever pool is current (issue #288). Retuning this therefore changes
/// where the binding routes validation — the two move together.
pub(crate) const VALIDATE_CHUNK: usize = 64 * 1024;

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

#[cfg(target_arch = "x86_64")]
#[inline]
fn first_invalid_in_chunk(chunk: &[f32], max_magnitude: f32) -> Option<usize> {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { first_invalid_in_chunk_avx2(chunk, max_magnitude) }
    } else {
        first_invalid_in_chunk_scalar(chunk, max_magnitude)
    }
}

/// AVX2 all-clean fast path: 8 lanes per compare of `|x| < bound`
/// (NaN/Inf/huge all fail), with a scalar pinpoint on a failing group so
/// the reported index matches the scalar scan exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn first_invalid_in_chunk_avx2(chunk: &[f32], max_magnitude: f32) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = chunk.len();
    let groups = n / 8;
    let bound = _mm256_set1_ps(max_magnitude);
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    for g in 0..groups {
        let x = _mm256_loadu_ps(chunk.as_ptr().add(g * 8));
        let ok = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(x, abs_mask), bound);
        if _mm256_movemask_ps(ok) != 0xff {
            for j in g * 8..n {
                let v = chunk[j];
                if !(v.abs() < max_magnitude) {
                    return Some(j);
                }
            }
            unreachable!("vector scan flagged a group with no invalid element");
        }
    }
    for j in groups * 8..n {
        let v = chunk[j];
        if !(v.abs() < max_magnitude) {
            return Some(j);
        }
    }
    None
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn first_invalid_in_chunk(chunk: &[f32], max_magnitude: f32) -> Option<usize> {
    first_invalid_in_chunk_scalar(chunk, max_magnitude)
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn first_invalid_in_chunk_scalar(chunk: &[f32], max_magnitude: f32) -> Option<usize> {
    chunk
        .iter()
        .position(|x| !x.is_finite() || x.abs() >= max_magnitude)
}

/// Map a finite `f32` to a `u32` whose unsigned order matches the float's
/// numeric order, so quantile selection can run on integers.
///
/// Positives keep their bit pattern with the sign bit set; negatives are
/// complemented (their magnitude bits run backwards). The map is a
/// bijection on non-NaN floats — [`f32_from_sort_key`] inverts it — so
/// selecting rank `r` over the keys and converting back yields exactly
/// the float that selecting rank `r` over the values would.
#[inline(always)]
fn f32_sort_key(x: f32) -> u32 {
    let b = x.to_bits();
    b ^ ((((b as i32) >> 31) as u32) | 0x8000_0000)
}

/// Inverse of [`f32_sort_key`].
#[inline(always)]
fn f32_from_sort_key(k: u32) -> f32 {
    let mask = if k & 0x8000_0000 != 0 { 0x8000_0000 } else { 0xFFFF_FFFF };
    f32::from_bits(k ^ mask)
}

/// Whether this target's quantize kernel reads the hoisted
/// `centroid_orig` reconstruction table.
///
/// The x86 kernel computes the same values in registers (a lane permute
/// over `centroids`, then `* inv_scale_tq[d] - shift[d]`), so building
/// the table there is pure cost — an allocation plus `2^bits * dim` f64
/// writes per batch, up to 384 KB at dim 3072 / 4-bit. Every kernel
/// handles `None` by computing the same values inline, so switching the
/// table off is always safe: it changes how the numbers are obtained,
/// never what they are. The pre-AVX2 scalar fallback on x86 pays a
/// recompute per element as a result, which is the correct trade on a
/// path already an order of magnitude off the SIMD one.
#[cfg(target_arch = "x86_64")]
const KERNEL_USES_RECON_TABLE: bool = false;
#[cfg(not(target_arch = "x86_64"))]
const KERNEL_USES_RECON_TABLE: bool = true;

/// Batch size at or above which the reconstruction table is built.
///
/// Below it the kernel recomputes the same values inline. A pure
/// performance switch — see [`build_recon_table`] for why the two must
/// stay bit-identical. Two tests hold them to it, with different reach:
///
/// * `encode::simd_identity_tests::quantize_kernel_matches_scalar_bit_exactly`
///   forces *both* table settings explicitly and compares them against
///   each other, so the cross-path comparison runs on every
///   architecture.
/// * `the_recon_table_threshold_does_not_change_encoded_bytes`
///   (`tests/encode_fingerprint.rs`) drives the switch end-to-end
///   through `add`, on the batch depths this constant separates. It is
///   arch-conditional: `KERNEL_USES_RECON_TABLE` is `false` on x86_64,
///   so there both batches take the inline path and the test degrades
///   to a batch-depth consistency check rather than a cross-path one.
///
/// That end-to-end test lives outside the crate and needs the two batch
/// depths this constant separates, so the constant is exported rather
/// than copied: it reads `RECON_TABLE_MIN_ROWS - 1` and
/// `RECON_TABLE_MIN_ROWS` directly and moves with any change made here.
/// It previously kept a private `THRESHOLD` copy guarded by a
/// `const _: () = assert!(RECON_TABLE_MIN_ROWS == 16, ..)`, which caught
/// the constant moving out from under the copy but not the copy moving
/// on its own — lowering the copy to 8 put both depths below the
/// threshold and silently turned the test into an inline-vs-inline
/// comparison. With no copy there is nothing to drift (#410).
///
/// `#[doc(hidden)]`: exported for that test only, not part of the public
/// API. It is a performance threshold with no format meaning, and
/// nothing outside the crate should branch on it.
#[doc(hidden)]
pub const RECON_TABLE_MIN_ROWS: usize = 16;

/// One reconstruction: a codebook centroid moved back into original
/// space, in f64.
///
/// The single definition of the arithmetic [`build_recon_table`] hoists
/// and the kernels otherwise evaluate inline — **the same three
/// operations in the same order and the same widths**. The builder, the
/// scalar kernel and the aarch64 kernel all call *this*, so those
/// branches hold no expression of their own to reassociate: there is one
/// operation order, and
/// `recon_table_entries_are_bit_identical_to_the_inline_expression`
/// pins it (#410).
///
/// **That is a convention, not an enforced invariant.** Nothing stops a
/// later edit from re-inlining the arithmetic into a `None` branch, and
/// doing so reopens #410 exactly as it stood: the mirror test pins the
/// builder against a literal, so a re-inlined *and* reassociated branch
/// leaves the whole suite green while its reconstructions diverge from
/// the table — the cross-path test compares f32 outputs and absorbs a
/// sub-f32 difference. Keep the call; do not re-inline it.
///
/// The AVX2 kernel is the one caller that cannot use it — it evaluates
/// four reconstructions at a time in packed f64 registers
/// (`_mm256_sub_pd(_mm256_mul_pd(..), ..)`), so it stays hand-mirrored,
/// backstopped by the avx2-vs-scalar leg of
/// `quantize_kernel_matches_scalar_bit_exactly`, which runs both table
/// settings.
///
/// `#[inline(always)]` because the kernels call it per coordinate on
/// the inline path; it must cost nothing there.
#[inline(always)]
fn recon_entry(centroid: f32, inv: f64, sh: f64) -> f64 {
    (centroid as f64) * inv - sh
}

/// Build the hoisted reconstruction table the quantize kernels read.
///
/// Entry `[d * 2^bits + c]` is `centroids[c] * inv_scale_tq[d] -
/// shift[d]` in f64 — **the same three operations in the same order and
/// the same widths** the kernels apply inline when there is no table.
/// That identity is what makes `RECON_TABLE_MIN_ROWS` a pure
/// performance switch rather than a format one: a 15-row batch and a
/// 16-row batch must encode identically.
///
/// It is a named function, and not the inline closure it used to be,
/// so the tests can call *this* builder instead of writing their own. A
/// test that rebuilds the table itself cannot see a divergence between
/// the production builder and the kernels' inline expression — a
/// row-major/coordinate-major slip, or a reordering to
/// `(centroids[c] - sh * scale) * inv` — which is precisely the gap
/// #369 identified.
///
/// Two tests constrain it, neither completely:
///
/// * `quantize_kernel_matches_scalar_bit_exactly` runs both settings
///   through the kernels and compares packed bytes and stored scales.
///   Those are f32, so it catches a divergence large enough to move a
///   code or the rounded scale — but one that stays below f32
///   resolution survives it.
/// * `recon_table_entries_are_bit_identical_to_the_inline_expression`
///   pins the f64 entries, bit for bit, against a literal transcription
///   of the arithmetic. That used to be one-directional — it pinned
///   this builder, not the kernels, so reassociating a `None` branch
///   left the mirror passing against a literal that transcribed nothing
///   — but the builder and both scalar/aarch64 `None` branches now share
///   [`recon_entry`], so there is a single expression for it to pin
///   (#410). The AVX2 packed form is still hand-mirrored.
///
/// Layout is **coordinate-major** (`table[d * n_codes + code]`). The
/// kernel walks `d` in order and looks up one entry per coordinate, so
/// coordinate-major makes those lookups a single sequential stream —
/// 8 * n_codes * 8 bytes per chunk — instead of `n_codes` streams
/// strided `dim * 8` bytes apart. At 4 bits that is 16 concurrent
/// streams over a 2^bits * dim * 8 byte table (384 KB at dim 3072),
/// which outruns the L1 and the prefetcher; coordinate-major touches
/// each cache line once.
fn build_recon_table(
    bit_width: usize,
    dim: usize,
    centroids: &[f32],
    inv_scale_tq: &[f32],
    shift: &[f32],
) -> Vec<f64> {
    let n_codes = 1usize << bit_width;
    let mut table = vec![0.0f64; n_codes * dim];
    for d in 0..dim {
        let inv = inv_scale_tq[d] as f64;
        let sh = shift[d] as f64;
        let row = &mut table[d * n_codes..(d + 1) * n_codes];
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = recon_entry(centroids[c], inv, sh);
        }
    }
    table
}

/// Probability level the per-coord fit anchors on, derived from the
/// codebook rather than fixed.
///
/// The fit maps the empirical `p`-quantile of a coordinate onto the
/// canonical marginal's `p`-quantile. When the data already has the
/// canonical *shape* every `p` gives the same answer, so the choice only
/// bites on heavy-tailed coordinates — and there it decides which part of
/// the distribution is fitted exactly and which is sacrificed. Values past
/// the outermost centroid all collapse into one bucket with unbounded
/// error, so the right anchor is the point where the codebook stops:
/// `P(|x| <= c_outer)`, with `c_outer` the largest centroid magnitude.
///
/// That point moves with bit width — ~0.933 at 2 bits, ~0.984 at 3,
/// ~0.996 at 4 — which is why this cannot be a constant. The previous
/// fixed 0.95 was right only at 2 bits; at 3 and 4 it fitted an interior
/// quantile and stretched the tails far past the codebook's last level.
/// On lastfm-64 (per-coord kurtosis 26) the 4-bit fit scaled the data
/// 2x too far, putting the 99.9th percentile at |x| ~ 1.33 against an
/// outermost centroid of 0.313 and dropping R@10 from 0.4835 (identity)
/// to 0.1439; anchoring here gives 0.6022. See #454.
///
/// Returns `(p_lo, p_hi, qc_lo, qc_hi)`: the two probability levels and
/// the canonical values they are mapped onto (the codebook's edges).
fn tqplus_anchor(beta: &Beta, centroids: &[f32]) -> (f64, f64, f32, f32) {
    let c_outer = centroids.iter().fold(0.0f32, |acc, &c| acc.max(c.abs()));
    // Beta lives on [0, 1]; the canonical marginal is it shifted to [-1, 1].
    let p_hi = beta.cdf((f64::from(c_outer) + 1.0) / 2.0);
    (1.0 - p_hi, p_hi, -c_outer, c_outer)
}

/// Sample size at which a per-coord fit stops improving, quoted by the
/// public [`calibrate`](crate::TurboQuantIndex::calibrate) docs as the
/// size to aim for.
///
/// Advisory, not enforced: the calibration sample is the caller's to
/// choose, and a corpus smaller than this is better calibrated from all
/// of it than not at all. Below ~200 rows the per-coord quantile noise
/// eats the precision gain outright (4-bit and 2-bit reconstruction
/// stddev become statistically indistinguishable); by ~1000 the fit is
/// stable, and measured against a fit from all 100k rows of the same
/// corpus a 1000-row draw costs at most ~0.5 pp R@10 (gte-small-384
/// 0.9034/0.9027/0.9056 vs 0.9083/0.9082/0.9082 across three seeds;
/// OpenAI-1536 0.9655 vs 0.9661; GloVe-200 0.8746 vs 0.8757; SIFT-128
/// 0.8537 vs 0.8514).
///
/// What those numbers assume is that the draw is *random*. None of this
/// survives a sorted or clustered sample of the same size — see the
/// `calibrate` docs.
pub const RECOMMENDED_CALIBRATION_ROWS: usize = 1000;

/// Fewest rows a fit is structurally able to use: the anchor needs a
/// distinct low and high order statistic per coordinate, which needs two
/// rows. Everything above this is a quality judgement the caller owns —
/// see [`RECOMMENDED_CALIBRATION_ROWS`].
pub const MIN_CALIBRATION_ROWS: usize = 2;

/// Rotate `n` rows of `vectors` into `rotated_scratch` (resized to
/// `n * dim`), applying `1/||row||` in the first gather, and return the
/// per-row norms. Shared by [`encode`] and [`fit_calibration`] so both
/// see bit-identical rotated coordinates.
///
/// Rows are independent so rayon splits them across cores; the per-row
/// transform is reduction-free (fixed add order, no FMA), so the rotated
/// values are identical regardless of how rows are distributed across
/// threads — the property the QR rotation lacked (#206).
fn rotate_batch_into(
    vectors: &[f32],
    n: usize,
    dim: usize,
    rotation: &Rotation,
    rotated_scratch: &mut Vec<f32>,
) -> Vec<f32> {
    let mut norms = vec![0.0f32; n];
    rotated_scratch.clear();
    rotated_scratch.reserve(n * dim);
    // Each row is rotated into a per-worker staging row and then copied
    // into the Vec's spare capacity through `MaybeUninit`; the length is
    // set only after every row has been written (#292). The Vec never
    // claims uninitialized elements, so a panic mid-rotation leaves an
    // empty — not uninit-length — scratch for `add_2d`'s unwind guard to
    // store back. The staging copy costs ~1% of encode; writing the
    // rotation's final `permute_gather` output straight into the spare
    // capacity would remove it but needs a `MaybeUninit` variant of every
    // NEON/AVX/scalar gather kernel. Reusing the caller's scratch keeps
    // the allocation warm across adds instead of paying a fresh multi-MB
    // mmap + page-fault walk per call.
    //
    // Rows are independent so rayon splits them across cores; the per-row
    // transform is reduction-free (fixed add order, no FMA), so the
    // encoded bytes are identical regardless of how rows are distributed
    // across threads (#206). The norm is computed in the same per-row
    // task — the row is already in cache, so a separate full-batch norms
    // pass is unnecessary.
    rotated_scratch.spare_capacity_mut()[..n * dim]
        .par_chunks_mut(dim)
        .zip(norms.par_iter_mut())
        .enumerate()
        .for_each_init(
            || (vec![0.0f32; dim], vec![0.0f32; dim]),
            |(scratch, row), (i, (dst_row, norm))| {
                let src = &vectors[i * dim..(i + 1) * dim];
                let n_val = simd_norm(src);
                *norm = n_val;
                // Norms at or below MIN_INPUT_NORM have no representable
                // direction; scale 0 is the documented outcome (#286).
                let inv = if n_val > crate::MIN_INPUT_NORM { 1.0 / n_val } else { 0.0 };
                rotation.apply_scaled_into(src, inv, row, scratch);
                for (d, &s) in dst_row.iter_mut().zip(row.iter()) {
                    d.write(s);
                }
            },
        );
    // SAFETY: every one of the n*dim spare slots was written above.
    unsafe {
        rotated_scratch.set_len(n * dim);
    }
    norms
}

/// Fit a TQ+ calibration from `vectors` without encoding them.
///
/// The only place a calibration is ever produced. `encode` never fits:
/// it applies the pair it is handed, or identity when handed none, so
/// what an index is calibrated to is a function of the sample the caller
/// passed to [`TurboQuantIndex::calibrate`](crate::TurboQuantIndex::calibrate)
/// and of nothing else — not of batch sizes, arrival order, or which
/// rows happened to arrive first.
///
/// `n` must be at least [`MIN_CALIBRATION_ROWS`]; the caller checks.
pub(crate) fn fit_calibration(
    vectors: &[f32],
    n: usize,
    dim: usize,
    rotation: &Rotation,
    centroids: &[f32],
    rotated_scratch: &mut Vec<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let _norms = rotate_batch_into(vectors, n, dim, rotation, rotated_scratch);
    compute_tqplus_calibration(rotated_scratch, n, dim, centroids)
}

/// Encode n vectors of dimension dim.
///
/// `calibration` is the index's committed `(shift, scale_tq)` pair, or
/// `None` for an uncalibrated index — plain TurboQuant, which is
/// arithmetically the identity pair. **This function never fits.** A
/// calibration comes from exactly one place,
/// [`fit_calibration`], driven by an explicit
/// [`TurboQuantIndex::calibrate`](crate::TurboQuantIndex::calibrate)
/// call; encode applies whatever it is given. That is what makes the
/// encoded bytes a function of (rows, calibration) alone, and therefore
/// independent of how the rows were batched or ordered.
///
/// Appends the packed codes and per-vector scales for this batch to
/// `packed_out` / `scales_out` (existing contents untouched).
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
    calibration: Option<(&[f32], &[f32])>,
    rotated_scratch: &mut Vec<f32>,
    packed_out: &mut Vec<u8>,
    scales_out: &mut Vec<f32>,
) {
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
    // Rotate each raw row into the scratch buffer via the deterministic
    // block-Hadamard transform, applying 1/||v|| in the first gather.
    let norms = rotate_batch_into(vectors, n, dim, rotation, rotated_scratch);
    let rotated = std::mem::take(rotated_scratch);
    encode_prerotated(
        &rotated, &norms, n, dim, boundaries, centroids, bit_width, calibration, packed_out,
        scales_out,
    );
    *rotated_scratch = rotated;
}

/// [`encode`] from the rotation onwards: quantize, pack and score `n`
/// already-rotated unit rows with their norms.
///
/// Split out because the refit path
/// ([`TurboQuantIndex::calibrate`](crate::TurboQuantIndex::calibrate) on
/// a populated index) has no float32 originals to rotate — it
/// reconstructs the rotated rows from the stored codes instead. Both
/// paths must land in the same kernel or a refitted row would not be
/// bit-identical to the same row added under the same calibration, and
/// the fixed-point property the refit rests on would be untestable.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_prerotated(
    rotated: &[f32],
    norms: &[f32],
    n: usize,
    dim: usize,
    boundaries: &[f32],
    centroids: &[f32],
    bit_width: usize,
    calibration: Option<(&[f32], &[f32])>,
    packed_out: &mut Vec<u8>,
    scales_out: &mut Vec<f32>,
) {
    // An uncalibrated index is arithmetically the identity pair, and
    // materializing it here rather than branching in the kernel keeps
    // one quantize path for both cases — two dim-length vectors per
    // batch against n*dim of encode work. The kernels then do
    // `(x + 0.0) * 1.0`, which is exact for every finite x, so an
    // uncalibrated encode produces precisely the pre-TQ+ bytes.
    let identity;
    let (shift, scale_tq): (&[f32], &[f32]) = match calibration {
        Some((s, sc)) => {
            assert_eq!(s.len(), dim, "shift length must equal dim");
            assert_eq!(sc.len(), dim, "scale_tq length must equal dim");
            (s, sc)
        }
        None => {
            identity = (vec![0.0f32; dim], vec![1.0f32; dim]);
            (&identity.0, &identity.1)
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
    // bit-identical. See `build_recon_table` for the layout rationale.
    //
    // The table costs O(2^bits * dim) to build, so it only pays once the
    // batch is a few rows deep; below that the kernel computes the same
    // values inline (identical ops, identical results).
    let centroid_orig: Option<Vec<f64>> = (KERNEL_USES_RECON_TABLE
        && n >= RECON_TABLE_MIN_ROWS)
        .then(|| build_recon_table(bit_width, dim, centroids, &inv_scale_tq, shift));

    let bytes_per_plane = dim / 8;
    let bytes_per_row = bit_width * bytes_per_plane;
    // Append-in-place: the batch's rows land directly at the tail of the
    // caller's buffers, so no per-call output allocation and no
    // extend_from_slice copy afterwards.
    let packed_old = packed_out.len();
    let scales_old = scales_out.len();
    // Every quantize kernel — NEON, AVX2, and the scalar fallback —
    // stores whole bytes (one store per plane per 8-coord chunk)
    // rather than OR-ing bits into a pre-zeroed row, so the
    // bytes_per_row * n zero-fill is dead work. The rows land in the
    // Vec's spare capacity via `MaybeUninit` and the length is only set
    // after `quantize_batch` returns having written every row (#292).
    packed_out.reserve(n * bytes_per_row);
    scales_out.resize(scales_old + n, 0.0f32);
    let packed = &mut packed_out.spare_capacity_mut()[..n * bytes_per_row];
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
    // SAFETY: quantize_batch wrote all n*bytes_per_row spare bytes.
    unsafe {
        packed_out.set_len(packed_old + n * bytes_per_row);
    }

    // Test-only: unwind from HERE, with both output buffers already
    // extended by this batch, so `encode_and_append`'s guard has real
    // truncation to do. `TurboQuantIndex::force_encode_panic` fires
    // before this function runs, when the buffers are still at their
    // pre-call lengths and both `truncate` calls are no-ops — so it
    // cannot cover the truncation at all. See the switch's docs.
    #[cfg(test)]
    if FORCE_PANIC_AFTER_APPEND.with(|f| f.replace(false)) {
        panic!("forced post-append encode panic (test)");
    }
}

// Test-only switch that unwinds `encode` **after** it has appended this
// batch to `packed_out` / `scales_out`, so a caught panic finds both
// buffers longer than the caller left them.
//
// This is the only reachable shape of the partial-append failure the
// unwind guard in `encode_and_append` truncates for. Today's `encode`
// commits the packed length in one `set_len` at the very end (#292), so
// no *intermediate* point in this function leaves `packed_out` long —
// the guard's `truncate` is defense against a future encode that
// appends incrementally, and this switch is what pins it.
//
// Thread-local, never a process-global: `cargo test` runs a binary's
// tests concurrently in one process, and a global one-shot armed by one
// test can be consumed by another (#373). Armed and consumed on the same
// thread — this check is on the calling thread's side of every rayon
// split inside `quantize_batch`.
#[cfg(test)]
thread_local! {
    static FORCE_PANIC_AFTER_APPEND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// See [`FORCE_PANIC_AFTER_APPEND`].
#[cfg(test)]
pub(crate) fn force_panic_after_append(on: bool) {
    FORCE_PANIC_AFTER_APPEND.with(|f| f.set(on));
}

/// Quantize + pack the whole batch with a compile-time bit width.
#[allow(clippy::too_many_arguments)]
fn quantize_batch<const BITS: usize>(
    packed: &mut [std::mem::MaybeUninit<u8>],
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
        .for_each_init(
            || vec![0u8; bytes_per_row],
            |row_buf, (i, (packed_row, scale))| {
                let rot_orig = &rotated[i * dim..(i + 1) * dim];
                *scale = fused_quantize_scale_pack::<BITS>(
                    rot_orig, shift, scale_tq, inv_scale_tq,
                    centroid_orig, boundaries, centroids, norms[i],
                    row_buf, dim, bytes_per_plane,
                );
                // The kernels overwrite every byte of the row, so the
                // per-thread staging row publishes fully-written bytes
                // into the spare capacity (#292).
                for (d, &s) in packed_row.iter_mut().zip(row_buf.iter()) {
                    d.write(s);
                }
            },
        );
}

/// Per-coordinate TQ+ calibration. For each of the `dim` rotated coordinates,
/// computes `(shift, scale)` such that `(x + shift) * scale` maps the
/// empirical quantiles at [`tqplus_anchor`]'s probability levels onto the
/// codebook's outermost centroids. When the batch is too small or a coord is
/// degenerate (constant or near-constant), falls back to identity.
fn compute_tqplus_calibration(
    rotated: &[f32],
    n: usize,
    dim: usize,
    centroids: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut shift = vec![0.0f32; dim];
    let mut scale = vec![1.0f32; dim];

    debug_assert!(
        n >= MIN_CALIBRATION_ROWS,
        "fit needs two distinct order statistics per coordinate"
    );

    let a = (dim as f64 - 1.0) / 2.0;
    let beta = Beta::new(a, a).expect("Beta(a, a) is valid for a > 0");
    let (p_lo, p_hi, qc_lo, qc_hi) = tqplus_anchor(&beta, centroids);
    let qc_span = qc_hi - qc_lo;

    let lo_idx = ((n as f64) * p_lo) as usize;
    // `hi_idx > lo_idx` is what makes the second select below well-formed
    // (it partitions `hi_idx - lo_idx - 1` of the right side). The two
    // probabilities are far enough apart that only a pathological
    // codebook could collapse them, but the clamp makes that structural
    // rather than incidental.
    let hi_idx = (((n as f64) * p_hi) as usize).min(n - 1).max(lo_idx + 1);

    // Coords are independent, but gathering one column at a time strides
    // `dim * 4` bytes per element — every read is a fresh cache line, and
    // the whole n*dim batch is re-streamed once per coordinate. Instead,
    // fan out over TILES of coordinates: each tile makes one sequential
    // pass over the rows, scattering into `tile` contiguous column
    // buffers (each row contributes a contiguous 4*tile-byte read). The
    // collected values per coord are identical, so the selected quantiles
    // — and every downstream encoded byte — are unchanged.
    // Tile size is capped by the *write* working set of the transpose,
    // not by parallelism. The scatter below fills `tile` destination
    // columns concurrently, so it keeps `tile` cache lines live; each
    // line is revisited once per row, with every other live line touched
    // in between. Once `tile * 64` bytes exceeds L1 every store misses —
    // measured as a ~13% loss in cold bulk insert at 768 columns versus
    // 128. 128 columns is 8 KB of destination lines, comfortably
    // resident, and still leaves `dim / 128` tiles to fan out over.
    //
    // Below that ceiling the tile is halved further while it would leave
    // fewer than ~2 tiles per rayon worker. Total bytes read is the same
    // for any tile size (each rotated element is read exactly once per
    // full sweep) and the collected values per coordinate — hence every
    // encoded byte — are identical; only locality and scheduling change.
    let workers = rayon::current_num_threads().max(1);
    let mut tile_size = 128usize;
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
            // The column scratch is fully written by the scatter below
            // before anything reads it, so skip the zero-fill: at the
            // 128-coordinate tile ceiling and n 100k this is a ~51 MB
            // memset (and its page-fault walk) per tile, paid purely to
            // be overwritten.
            // The scatter writes into the spare capacity via
            // `MaybeUninit`; length is set only after every one of the
            // `tile * n` slots has been assigned (#292).
            let mut cols: Vec<u32> = Vec::with_capacity(tile * n);
            // Values are transposed as order-preserving integer keys, not
            // as floats: quickselect over `u32` uses the native `Ord`
            // and integer compares instead of a `partial_cmp` closure
            // that has to handle an unordered case which cannot occur
            // (`add` rejects non-finite input, and the rotation is
            // add/sub/scale, so every rotated coordinate is finite).
            // `f32_sort_key` is monotone on finite floats, so the element
            // selected at a given rank — and therefore the fitted
            // calibration and every encoded byte — is unchanged. The one
            // pair the orderings disagree on is -0.0 vs +0.0, which
            // `partial_cmp` calls equal; both map to the same quantile
            // arithmetic below (`x - -0.0` and `x - 0.0` agree for every
            // finite x), so that case is a wash too.
            {
                let spare = &mut cols.spare_capacity_mut()[..tile * n];
                for i in 0..n {
                    let row = &rotated[i * dim + d0..i * dim + d0 + tile];
                    for (c, &v) in row.iter().enumerate() {
                        spare[c * n + i].write(f32_sort_key(v));
                    }
                }
            }
            // SAFETY: the scatter above assigned all tile*n slots.
            unsafe {
                cols.set_len(tile * n);
            }
            for (c, (sh, sc)) in sh_tile.iter_mut().zip(sc_tile.iter_mut()).enumerate() {
                let coord = &mut cols[c * n..(c + 1) * n];
                // Only the two quantile order statistics are needed, so
                // two O(n) selects replace a full O(n log n) sort.
                // Select the LOW quantile first: its partition splits
                // off only ~5% of the data, so the second (high) select
                // runs over the ~95% right side — total elements
                // partitioned is the same, but the first partition's
                // pivot walks terminate sooner. Selected values are
                // identical to indexing a fully sorted array.
                let (_, lo_val, right) = coord.select_nth_unstable(lo_idx);
                let qe_lo = f32_from_sort_key(*lo_val);
                let (_, hi_val, _) = right.select_nth_unstable(hi_idx - lo_idx - 1);
                let qe_hi = f32_from_sort_key(*hi_val);
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

// ─── Per-vector norm ─────────────────────────────────────────────────────────

/// Number of independent accumulation chains in the canonical norm
/// reduction. Element `j` joins chain `j % NORM_CHAINS`.
///
/// DO NOT CHANGE without treating it as a format change: the reduction
/// order is part of the encode contract (see [`simd_norm`]). Eight is the
/// widest split every supported vector width can express exactly — one
/// `__m256`, two NEON `float32x4_t`, or eight scalars — and `dim` is
/// always a multiple of 8, so no index path ever reaches the tail.
const NORM_CHAINS: usize = 8;

/// Euclidean norm of one row, with a **fixed, architecture-independent
/// reduction order**.
///
/// This is the vector's `1/||v||` normalization factor, which rides the
/// first rotation gather — so its low bits reach every encoded byte. The
/// order is therefore part of the format contract, exactly like the
/// rotation's:
///
/// ```text
/// c[j % 8] += row[j] * row[j]      (separate multiply and add, never an FMA)
/// sum       = ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7))
/// norm      = sqrt(sum)
/// ```
///
/// The previous implementation was per-architecture and *not* equivalent
/// across them: aarch64 accumulated four chains with `vfmaq_f32` (a fused
/// multiply-add, one rounding instead of two) while every other target
/// summed left-to-right in a single chain. The two can disagree in the
/// last ulp, which is the remaining cross-platform encode input flagged in
/// the v5 determinism scope (#259, finding 1). Pinning one order closes it
/// by construction rather than by observation.
///
/// It is also considerably faster than the sequential sum it replaces on
/// x86: eight independent chains hide the f32 add latency that a single
/// accumulator serializes on.
///
/// `sqrt` is IEEE-754 correctly rounded (a hardware instruction on every
/// supported target), so it introduces no platform variance.
#[inline(always)]
fn simd_norm(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            return unsafe { norm_sq_avx(row) }.sqrt();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { norm_sq_neon(row) }.sqrt();
    }
    #[allow(unreachable_code)]
    {
        norm_sq_scalar(row).sqrt()
    }
}

/// Reference reduction. Every SIMD path must reproduce it bit-for-bit —
/// enforced by `norm_simd_matches_scalar_bit_exactly`.
#[inline]
fn norm_sq_scalar(row: &[f32]) -> f32 {
    let mut chains = [0.0f32; NORM_CHAINS];
    for (j, &x) in row.iter().enumerate() {
        chains[j % NORM_CHAINS] += x * x;
    }
    combine_norm_chains(&chains)
}

/// The frozen combine tree over the eight chains.
#[inline(always)]
fn combine_norm_chains(c: &[f32; NORM_CHAINS]) -> f32 {
    ((c[0] + c[1]) + (c[2] + c[3])) + ((c[4] + c[5]) + (c[6] + c[7]))
}

/// AVX: the eight chains are the eight lanes of one `__m256`. `mul` then
/// `add` as separate instructions — never `vfmadd` — so each element is
/// rounded twice, matching the scalar reference.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn norm_sq_avx(row: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = row.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + NORM_CHAINS <= n {
        let v = _mm256_loadu_ps(row.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
        i += NORM_CHAINS;
    }
    let mut chains = [0.0f32; NORM_CHAINS];
    _mm256_storeu_ps(chains.as_mut_ptr(), acc);
    // `dim` is a multiple of 8 on every index path, so this tail is dead
    // there; it keeps the helper correct for arbitrary lengths.
    while i < n {
        let x = *row.get_unchecked(i);
        chains[i % NORM_CHAINS] += x * x;
        i += 1;
    }
    combine_norm_chains(&chains)
}

/// NEON: the eight chains are two `float32x4_t` accumulators. `vmulq` +
/// `vaddq` rather than `vfmaq` — the fused form rounds once and would not
/// match the scalar reference (this is #259 finding 1).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn norm_sq_neon(row: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = row.len();
    let mut acc_lo = vdupq_n_f32(0.0);
    let mut acc_hi = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + NORM_CHAINS <= n {
        let lo = vld1q_f32(row.as_ptr().add(i));
        let hi = vld1q_f32(row.as_ptr().add(i + 4));
        acc_lo = vaddq_f32(acc_lo, vmulq_f32(lo, lo));
        acc_hi = vaddq_f32(acc_hi, vmulq_f32(hi, hi));
        i += NORM_CHAINS;
    }
    let mut chains = [0.0f32; NORM_CHAINS];
    vst1q_f32(chains.as_mut_ptr(), acc_lo);
    vst1q_f32(chains.as_mut_ptr().add(4), acc_hi);
    while i < n {
        let x = *row.get_unchecked(i);
        chains[i % NORM_CHAINS] += x * x;
        i += 1;
    }
    combine_norm_chains(&chains)
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

    let chunks = dim / 8;
    // Four fixed accumulation chains (two f64x2 registers): term j joins
    // chain j % 4; final combine ((a0 + a1) + (b0 + b1)). Deterministic,
    // mirrored exactly by the scalar fallback.
    let mut acc_a;
    let mut acc_b;
    unsafe {
        acc_a = vdupq_n_f64(0.0);
        acc_b = vdupq_n_f64(0.0);
    }

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

            if BITS == 4 {
                // Two-level scan: one compare against the median boundary
                // decides, per lane, which 7-boundary half to count.
                // code = 8*(x > b[7]) + sum over the selected half of
                // (x > b_k) — exactly the count the flat scan produces,
                // since x > b[7] implies x exceeds all of b[0..=7].
                let mid = vdupq_n_f32(boundaries[7]);
                let m_lo = vcgtq_f32(vals_lo, mid);
                let m_hi = vcgtq_f32(vals_hi, mid);
                acc_lo = vshlq_n_u32::<3>(vshrq_n_u32::<31>(m_lo));
                acc_hi = vshlq_n_u32::<3>(vshrq_n_u32::<31>(m_hi));
                for k in 0..7 {
                    let b_low = vdupq_n_f32(boundaries[k]);
                    let b_high = vdupq_n_f32(boundaries[8 + k]);
                    let bv_lo = vbslq_f32(m_lo, b_high, b_low);
                    let bv_hi = vbslq_f32(m_hi, b_high, b_low);
                    acc_lo =
                        vaddq_u32(acc_lo, vshrq_n_u32::<31>(vcgtq_f32(vals_lo, bv_lo)));
                    acc_hi =
                        vaddq_u32(acc_hi, vshrq_n_u32::<31>(vcgtq_f32(vals_hi, bv_hi)));
                }
            } else {
                for bi in 0..(1usize << BITS) - 1 {
                    let bv = vdupq_n_f32(boundaries[bi]);
                    acc_lo = vaddq_u32(acc_lo, vshrq_n_u32::<31>(vcgtq_f32(vals_lo, bv)));
                    acc_hi = vaddq_u32(acc_hi, vshrq_n_u32::<31>(vcgtq_f32(vals_hi, bv)));
                }
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
            // batch amortized building it, otherwise inline via
            // `recon_entry` — the same function `build_recon_table`
            // fills the table with, so the two are bit-identical by
            // construction rather than by inspection (#410).
            let mut terms = [0.0f64; 8];
            match centroid_orig {
                Some(table) => {
                    for k in 0..8 {
                        let d = offset + k;
                        terms[k] = (rot_orig[d] as f64)
                            * table[d * (1 << BITS) + counts[k] as usize];
                    }
                }
                None => {
                    for k in 0..8 {
                        let d = offset + k;
                        let centroid_in_orig = recon_entry(
                            centroids[counts[k] as usize],
                            inv_scale_tq[d] as f64,
                            shift[d] as f64,
                        );
                        terms[k] = (rot_orig[d] as f64) * centroid_in_orig;
                    }
                }
            }
            acc_a = vaddq_f64(acc_a, vld1q_f64(terms.as_ptr()));
            acc_b = vaddq_f64(acc_b, vld1q_f64(terms.as_ptr().add(2)));
            acc_a = vaddq_f64(acc_a, vld1q_f64(terms.as_ptr().add(4)));
            acc_b = vaddq_f64(acc_b, vld1q_f64(terms.as_ptr().add(6)));

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

    let inner = unsafe {
        (vgetq_lane_f64::<0>(acc_a) + vgetq_lane_f64::<1>(acc_a))
            + (vgetq_lane_f64::<0>(acc_b) + vgetq_lane_f64::<1>(acc_b))
    };
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

// ─── Fused quantize + scale + pack (x86_64) ─────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
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
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            fused_quantize_scale_pack_avx2::<BITS>(
                rot_orig, shift, scale_tq, inv_scale_tq, centroid_orig,
                boundaries, centroids, norm, packed_row, dim, bytes_per_plane,
            )
        }
    } else {
        fused_quantize_scale_pack_scalar::<BITS>(
            rot_orig, shift, scale_tq, inv_scale_tq, centroid_orig,
            boundaries, centroids, norm, packed_row, dim, bytes_per_plane,
        )
    }
}

/// AVX2 kernel mirroring the scalar path exactly: calibration
/// `(x + shift) * scale` and every boundary compare are element-wise
/// IEEE ops (8 lanes at a time), the reconstruction terms accumulate
/// into the same four f64 chains (term j -> chain j % 4, two __m128d
/// registers) with the same final combine, and the pack loop is the
/// scalar OR into the zeroed row — so the packed codes and stored
/// scales are bit-identical to the scalar kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_quantize_scale_pack_avx2<const BITS: usize>(
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
    use std::arch::x86_64::*;

    // Lane j of `acc4` is chain j: the two adds per chunk take terms
    // 0..3 then 4..7, so term t lands in chain t % 4 in the same order
    // as the scalar loop.
    let mut acc4 = _mm256_setzero_pd();
    let chunks = dim / 8;
    // The hoisted `centroid_orig` table is deliberately unused here: the
    // kernel selects `centroids[code]` with a lane permute and applies
    // `* inv_scale_tq[d] - shift[d]` in registers instead. The table
    // holds exactly those values — it is built with exactly these ops,
    // in this order — so the accumulated inner product is bit-identical,
    // but reading it costs a stream of the whole 2^bits * dim * 8 byte
    // table per row (384 KB at dim 3072, 4-bit; far past L1), where the
    // register form touches only the two dim-length f32 arrays the
    // kernel already walks. `encode` therefore skips building it on x86
    // entirely (see KERNEL_USES_RECON_TABLE).
    let _ = centroid_orig;
    // `centroids` has 2^BITS entries; a lane permute needs an 8-lane
    // source, so pad (BITS <= 3) or split into halves (BITS == 4).
    let mut cpad = [0.0f32; 8];
    let mut cpad_hi = [0.0f32; 8];
    for (i, slot) in cpad.iter_mut().enumerate() {
        *slot = centroids[i.min((1usize << BITS) - 1)];
    }
    if BITS == 4 {
        for (i, slot) in cpad_hi.iter_mut().enumerate() {
            *slot = centroids[8 + i];
        }
    }
    let cvec = _mm256_loadu_ps(cpad.as_ptr());
    let cvec_hi = _mm256_loadu_ps(cpad_hi.as_ptr());

    for c in 0..chunks {
        let offset = c * 8;
        // Calibrated values, 8 lanes: (x + shift) * scale_tq — the same
        // two IEEE ops per element as the scalar path.
        let vals = _mm256_mul_ps(
            _mm256_add_ps(
                _mm256_loadu_ps(rot_orig.as_ptr().add(offset)),
                _mm256_loadu_ps(shift.as_ptr().add(offset)),
            ),
            _mm256_loadu_ps(scale_tq.as_ptr().add(offset)),
        );

        // Boundary count per lane (acc -= cmp adds 1 where val > b).
        let mut acc = _mm256_setzero_si256();
        if BITS == 4 {
            let mid = _mm256_set1_ps(boundaries[7]);
            let m = _mm256_cmp_ps::<_CMP_GT_OQ>(vals, mid);
            acc = _mm256_slli_epi32::<3>(_mm256_srli_epi32::<31>(_mm256_castps_si256(m)));
            for k in 0..7 {
                let b_low = _mm256_set1_ps(boundaries[k]);
                let b_high = _mm256_set1_ps(boundaries[8 + k]);
                let bv = _mm256_blendv_ps(b_low, b_high, m);
                let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(vals, bv);
                acc = _mm256_sub_epi32(acc, _mm256_castps_si256(gt));
            }
        } else if BITS == 2 {
            let mid = _mm256_set1_ps(boundaries[1]);
            let m = _mm256_cmp_ps::<_CMP_GT_OQ>(vals, mid);
            acc = _mm256_slli_epi32::<1>(_mm256_srli_epi32::<31>(_mm256_castps_si256(m)));
            let bv = _mm256_blendv_ps(
                _mm256_set1_ps(boundaries[0]),
                _mm256_set1_ps(boundaries[2]),
                m,
            );
            let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(vals, bv);
            acc = _mm256_sub_epi32(acc, _mm256_castps_si256(gt));
        } else {
            for bi in 0..(1usize << BITS) - 1 {
                let bv = _mm256_set1_ps(boundaries[bi]);
                let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(vals, bv);
                acc = _mm256_sub_epi32(acc, _mm256_castps_si256(gt));
            }
        }
        // Pack: bit-plane p of this 8-coord chunk is one byte whose bit
        // (7 - k) is bit p of code k. `movemask_ps` gathers the sign bit
        // of each lane into a bit *in lane order* (lane k -> bit k), so
        // the lanes are reversed once up front; then each plane is a
        // shift-to-sign-bit plus a movemask. Same bits, same positions as
        // the scalar `byte |= ((code >> p) & 1) << (7 - k)` loop, minus
        // the 8 read-modify-writes per byte.
        let rev = _mm256_permutevar8x32_epi32(
            acc,
            _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0),
        );
        for p in 0..BITS {
            let bit = _mm256_sll_epi32(rev, _mm_cvtsi32_si128(31 - p as i32));
            let m = _mm256_movemask_ps(_mm256_castsi256_ps(bit)) as u8;
            *packed_row.get_unchecked_mut(p * bytes_per_plane + c) = m;
        }

        // Select centroids[code] per lane, widen to f64, and apply
        // `* inv_scale_tq[d] - shift[d]` — the same three IEEE ops, in
        // the same order, that build the hoisted table entry.
        let sel = if BITS == 4 {
            let low3 = _mm256_and_si256(acc, _mm256_set1_epi32(7));
            let lo = _mm256_permutevar8x32_ps(cvec, low3);
            let hi = _mm256_permutevar8x32_ps(cvec_hi, low3);
            let use_hi = _mm256_cmpgt_epi32(acc, _mm256_set1_epi32(7));
            _mm256_blendv_ps(lo, hi, _mm256_castsi256_ps(use_hi))
        } else {
            _mm256_permutevar8x32_ps(cvec, acc)
        };
        let x_lo = _mm256_sub_pd(
            _mm256_mul_pd(
                _mm256_cvtps_pd(_mm256_castps256_ps128(sel)),
                _mm256_cvtps_pd(_mm_loadu_ps(inv_scale_tq.as_ptr().add(offset))),
            ),
            _mm256_cvtps_pd(_mm_loadu_ps(shift.as_ptr().add(offset))),
        );
        let x_hi = _mm256_sub_pd(
            _mm256_mul_pd(
                _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(sel)),
                _mm256_cvtps_pd(_mm_loadu_ps(inv_scale_tq.as_ptr().add(offset + 4))),
            ),
            _mm256_cvtps_pd(_mm_loadu_ps(shift.as_ptr().add(offset + 4))),
        );
        let rot_lo = _mm256_cvtps_pd(_mm_loadu_ps(rot_orig.as_ptr().add(offset)));
        let rot_hi = _mm256_cvtps_pd(_mm_loadu_ps(rot_orig.as_ptr().add(offset + 4)));
        acc4 = _mm256_add_pd(acc4, _mm256_mul_pd(rot_lo, x_lo));
        acc4 = _mm256_add_pd(acc4, _mm256_mul_pd(rot_hi, x_hi));
    }

    // Fixed combine ((c0 + c1) + (c2 + c3)) — identical to the scalar
    // chains' combine.
    let mut chains = [0.0f64; 4];
    _mm256_storeu_pd(chains.as_mut_ptr(), acc4);
    let inner = (chains[0] + chains[1]) + (chains[2] + chains[3]);
    scale_from_inner(inner, norm)
}

// ─── Fused quantize + scale + pack (fallback) ───────────────────────────────

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[allow(clippy::too_many_arguments)]
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
    fused_quantize_scale_pack_scalar::<BITS>(
        rot_orig, shift, scale_tq, inv_scale_tq, centroid_orig,
        boundaries, centroids, norm, packed_row, dim, bytes_per_plane,
    )
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn fused_quantize_scale_pack_scalar<const BITS: usize>(
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
    // Four fixed chains mirroring the aarch64 kernel (chain j % 4;
    // combine ((c0 + c1) + (c2 + c3))).
    let mut chains = [0.0f64; 4];

    // One 8-coordinate chunk per iteration: the eight codes are resolved
    // first, then each bit-plane byte is *stored* whole. The previous
    // form OR-ed one bit at a time into a pre-zeroed row — 8 read-modify-
    // writes per byte plus a batch-wide memset. Same bits land in the
    // same positions, so the packed bytes are unchanged; `encode` no
    // longer has to zero the region first.
    let chunks = dim / 8;
    for c in 0..chunks {
        let offset = c * 8;
        let mut codes = [0u8; 8];
        for (k, code) in codes.iter_mut().enumerate() {
            let j = offset + k;
            let calib = (rot_orig[j] + shift[j]) * scale_tq[j];
            let mut v = 0u8;
            for bi in 0..(1usize << BITS) - 1 {
                if calib > boundaries[bi] { v += 1; }
            }
            *code = v;
            // Same table-or-inline split as the aarch64 kernel. Both
            // sides resolve to `recon_entry` — the table was built with
            // it — so they are bit-identical by construction (#410).
            let centroid_in_orig = match centroid_orig {
                Some(table) => table[j * (1 << BITS) + v as usize],
                None => {
                    recon_entry(centroids[v as usize], inv_scale_tq[j] as f64, shift[j] as f64)
                }
            };
            chains[j % 4] += (rot_orig[j] as f64) * centroid_in_orig;
        }
        for p in 0..BITS {
            let mut byte = 0u8;
            for (k, &code) in codes.iter().enumerate() {
                byte |= ((code >> p) & 1) << (7 - k);
            }
            packed_row[p * bytes_per_plane + c] = byte;
        }
    }
    // No tail loop: `encode` asserts dim % 8 == 0, so chunks * 8 == dim.

    let inner = (chains[0] + chains[1]) + (chains[2] + chains[3]);
    scale_from_inner(inner, norm)
}

#[cfg(test)]
mod simd_identity_tests {
    use super::*;
    use crate::codebook;
    use crate::rotation::Rotation;

    fn pseudo_rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        (0..n * dim)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x as f64 / u64::MAX as f64) as f32 - 0.5
            })
            .collect()
    }

    /// The arch quantize kernel must reproduce the scalar kernel
    /// bit-for-bit: identical packed codes and identical stored scales,
    /// for every bit width, on both the inline and the hoisted-table
    /// reconstruction paths.
    #[test]
    fn quantize_kernel_matches_scalar_bit_exactly() {
        crate::rotation::tests::require_simd_features();
        fn run<const BITS: usize>(dim: usize) {
            let rotation = Rotation::new(dim);
            let (boundaries, centroids) = codebook::codebook(BITS, dim);
            // A fitted-looking calibration (non-identity) to exercise
            // shift/scale arithmetic.
            let shift: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.001) - 0.01).collect();
            let scale_tq: Vec<f32> = (0..dim).map(|d| 1.0 + (d as f32 * 0.0005)).collect();
            let inv_scale_tq: Vec<f32> = scale_tq.iter().map(|s| 1.0 / s).collect();
            // The *production* builder, not a reimplementation of it.
            // Rebuilding the table here would make the table-vs-inline
            // comparison below self-fulfilling: it would compare the
            // test's idea of the table against the kernels, and a
            // divergence between `build_recon_table` and the kernels'
            // inline expression — the actual risk in #369 — would be
            // invisible.
            let table = build_recon_table(BITS, dim, &centroids, &inv_scale_tq, &shift);

            let n = 4;
            let raw = pseudo_rows(n, dim, 0xD1536 + BITS as u64);
            let bytes_per_plane = dim / 8;
            let bytes_per_row = BITS * bytes_per_plane;
            let mut scratch = vec![0.0f32; dim];

            for i in 0..n {
                let mut rot = vec![0.0f32; dim];
                let src = &raw[i * dim..(i + 1) * dim];
                let norm = src.iter().map(|x| x * x).sum::<f32>().sqrt();
                rotation.apply_scaled_into(src, 1.0 / norm, &mut rot, &mut scratch);

                // The scalar reference's result for each setting, kept
                // so the two settings can be compared against *each
                // other* after the loop and not only each against
                // itself (#369).
                let mut by_setting: Vec<(Vec<u8>, f32)> = Vec::new();

                for table_opt in [None, Some(table.as_slice())] {
                    let mut expect = vec![0u8; bytes_per_row];
                    let scale_ref = fused_quantize_scale_pack_scalar::<BITS>(
                        &rot, &shift, &scale_tq, &inv_scale_tq, table_opt,
                        &boundaries, &centroids, norm, &mut expect, dim,
                        bytes_per_plane,
                    );

                    // Every implementation this host can run, not just
                    // the dispatched one. (Unlike the rotation, the
                    // quantize dispatcher has no AVX-512 tier, so AVX2 is
                    // what it picks here — but asserting each kernel
                    // explicitly keeps the test honest if a tier is added,
                    // and pins the scalar reference either way.)
                    let mut paths: Vec<(&str, Vec<u8>, f32)> = Vec::new();
                    {
                        let mut p = vec![0u8; bytes_per_row];
                        let sc = fused_quantize_scale_pack::<BITS>(
                            &rot, &shift, &scale_tq, &inv_scale_tq, table_opt,
                            &boundaries, &centroids, norm, &mut p, dim,
                            bytes_per_plane,
                        );
                        paths.push(("dispatch", p, sc));
                    }
                    #[cfg(target_arch = "x86_64")]
                    if std::arch::is_x86_feature_detected!("avx2") {
                        let mut p = vec![0u8; bytes_per_row];
                        let sc = unsafe {
                            fused_quantize_scale_pack_avx2::<BITS>(
                                &rot, &shift, &scale_tq, &inv_scale_tq, table_opt,
                                &boundaries, &centroids, norm, &mut p, dim,
                                bytes_per_plane,
                            )
                        };
                        paths.push(("avx2", p, sc));
                    }

                    for (name, packed, scale) in &paths {
                        assert_eq!(
                            packed, &expect,
                            "BITS={BITS} row {i} table={} path={name} packed bytes diverge",
                            table_opt.is_some()
                        );
                        assert_eq!(
                            scale.to_bits(),
                            scale_ref.to_bits(),
                            "BITS={BITS} row {i} table={} path={name} scale diverges: \
                             {scale} vs {scale_ref}",
                            table_opt.is_some()
                        );
                    }
                    by_setting.push((expect, scale_ref));
                }

                // The cross-setting assertion. Each kernel reproducing
                // the scalar reference *within* a setting says nothing
                // about the two settings agreeing with each other —
                // and `RECON_TABLE_MIN_ROWS` flips between them purely
                // on batch depth, so if they disagree the same vector
                // encodes to different bytes depending on how many
                // neighbours it was added with. That is a format bug,
                // not a performance one.
                let (ref inline_packed, inline_scale) = by_setting[0];
                let (ref table_packed, table_scale) = by_setting[1];
                assert_eq!(
                    inline_packed, table_packed,
                    "BITS={BITS} row {i}: recon-table and inline paths packed \
                     different bytes. RECON_TABLE_MIN_ROWS would then be a \
                     format switch — a 15-row batch and a 16-row batch would \
                     encode the same vector differently.",
                );
                assert_eq!(
                    inline_scale.to_bits(),
                    table_scale.to_bits(),
                    "BITS={BITS} row {i}: recon-table and inline paths produced \
                     different stored scales ({inline_scale} vs {table_scale}). \
                     The table must apply exactly the ops the kernel applies \
                     inline, in the same order and the same widths.",
                );
            }
        }
        run::<2>(1536);
        run::<3>(128);
        run::<4>(1536);
        run::<2>(3072);
        run::<4>(3072);
    }

    /// Every entry of the hoisted table must be bit-identical to the
    /// expression the kernels evaluate inline when there is no table.
    ///
    /// `quantize_kernel_matches_scalar_bit_exactly` compares the two
    /// paths only through f32 outputs — packed codes and the stored
    /// scale. That is one rounding away from the f64 arithmetic, so a
    /// reassociation of the table entry to
    /// `(centroids[c] - sh * scale) * inv` perturbs a few thousand
    /// entries by a ulp, gets absorbed by the f32 rounding, and passes.
    /// The invariant [`build_recon_table`] documents is stronger than
    /// that: the *same three operations in the same order and the same
    /// widths*. This test asserts exactly that, in f64.
    ///
    /// The right-hand side is deliberately a literal transcription of
    /// the kernels' `None` branches (`fused_quantize_scale_pack_scalar`
    /// and the aarch64 kernel, and the register form the AVX2 kernel
    /// uses in place of the table). It is a reimplementation on purpose
    /// — an *independent* statement of the arithmetic is the only thing
    /// that can pin the builder's order, since a test that called the
    /// builder for both sides would be a tautology.
    ///
    /// Its reach is still exactly the builder — but there is much less
    /// left outside it. The scalar and aarch64 `None` branches used to
    /// carry their own copies of the arithmetic, which this test did not
    /// pin: reassociating one left this test passing against a literal
    /// that was no longer a transcription of anything. Both now call
    /// [`recon_entry`], the same function the builder calls, so there is
    /// a single operation order and pinning the builder reaches it.
    ///
    /// What that does **not** do is make the property enforced. It is a
    /// call-site convention: re-inline the arithmetic into a `None`
    /// branch and reassociate it, and this test goes green again while
    /// the kernel diverges from the table — the original #410. The
    /// expressions to keep in step by hand are therefore the AVX2 packed
    /// form, and any `None` branch a later change re-inlines.
    #[test]
    fn recon_table_entries_are_bit_identical_to_the_inline_expression() {
        fn run<const BITS: usize>(dim: usize) {
            let (_, centroids) = codebook::codebook(BITS, dim);
            // Non-identity shift/scale: with shift = 0 and scale = 1
            // every plausible reordering collapses to the same value,
            // exactly as in the end-to-end threshold test.
            let shift: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.001) - 0.01).collect();
            let inv_scale_tq: Vec<f32> =
                (0..dim).map(|d| 1.0 / (1.0 + (d as f32 * 0.0005))).collect();
            let table = build_recon_table(BITS, dim, &centroids, &inv_scale_tq, &shift);

            let n_codes = 1usize << BITS;
            assert_eq!(
                table.len(),
                n_codes * dim,
                "BITS={BITS} dim={dim}: table is not coordinate-major with \
                 2^BITS entries per coordinate",
            );
            for d in 0..dim {
                for c in 0..n_codes {
                    // The kernels' inline form, verbatim.
                    let want =
                        (centroids[c] as f64) * (inv_scale_tq[d] as f64) - (shift[d] as f64);
                    let got = table[d * n_codes + c];
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "BITS={BITS} dim={dim} d={d} code={c}: table entry \
                         {got:e} is not bit-identical to the inline \
                         expression {want:e}. RECON_TABLE_MIN_ROWS is then a \
                         format switch at f64 precision even if the packed \
                         bytes happen to round the same way today.",
                    );
                }
            }
        }
        run::<2>(1536);
        run::<3>(128);
        run::<4>(1536);
        run::<2>(3072);
        run::<4>(3072);
    }

    /// The integer sort key used for quantile selection must be a
    /// strictly order-preserving bijection on finite floats — otherwise
    /// the fitted calibration silently changes.
    #[test]
    fn f32_sort_key_is_order_preserving_and_invertible() {
        let mut vals: Vec<f32> = pseudo_rows(1, 4096, 0xC0FFEE);
        vals.extend_from_slice(&[
            0.0, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE,
            f32::MAX, f32::MIN, 1e-30, -1e-30, 3.5, -3.5,
        ]);
        for &x in &vals {
            assert_eq!(
                f32_from_sort_key(f32_sort_key(x)).to_bits(),
                x.to_bits(),
                "round-trip failed for {x}",
            );
        }
        for &a in &vals {
            for &b in &vals {
                // -0.0 vs 0.0 is the documented exception: numerically
                // equal, but the keys order them. Every other pair must
                // agree with the float ordering.
                if a == 0.0 && b == 0.0 {
                    continue;
                }
                assert_eq!(
                    f32_sort_key(a) < f32_sort_key(b),
                    a < b,
                    "key order disagrees for ({a}, {b})",
                );
            }
        }
    }

    /// Selecting a rank over the keys must return the same float as
    /// selecting the same rank over the values.
    #[test]
    fn key_selection_matches_float_selection() {
        for n in [1usize, 2, 17, 1000, 4096] {
            let vals = pseudo_rows(1, n, 0xBEEF ^ n as u64);
            for &rank in &[0usize, n / 20, n / 2, n - 1] {
                let mut by_val = vals.clone();
                let (_, v, _) = by_val.select_nth_unstable_by(rank, |a: &f32, b: &f32| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                let expect = *v;
                let mut by_key: Vec<u32> = vals.iter().copied().map(f32_sort_key).collect();
                let (_, k, _) = by_key.select_nth_unstable(rank);
                assert_eq!(
                    f32_from_sort_key(*k).to_bits(),
                    expect.to_bits(),
                    "n={n} rank={rank}",
                );
            }
        }
    }

    /// The per-vector norm feeds `1/||v||` into the first rotation
    /// gather, so its low bits reach every encoded byte: the SIMD
    /// reduction must reproduce the frozen scalar chain order exactly.
    /// A regression here is a silent cross-platform format divergence
    /// (#259 finding 1), not a rounding nit.
    #[test]
    fn norm_simd_matches_scalar_bit_exactly() {
        crate::rotation::tests::require_simd_features();
        // Multiples of 8 (every index path) plus non-multiples and
        // sub-chain lengths, which exercise the scalar tail.
        for len in [8usize, 16, 24, 64, 200, 768, 1000, 1536, 3072, 1, 5, 7, 9, 15] {
            for seed in [1u64, 0xD1536, 0xFFFF_FFFF] {
                let row = pseudo_rows(1, len, seed);
                let scalar = norm_sq_scalar(&row).sqrt();
                let mut paths: Vec<(&str, f32)> = vec![("dispatch", simd_norm(&row))];
                #[cfg(target_arch = "x86_64")]
                if std::arch::is_x86_feature_detected!("avx") {
                    paths.push(("avx", unsafe { norm_sq_avx(&row) }.sqrt()));
                }
                #[cfg(target_arch = "aarch64")]
                paths.push(("neon", unsafe { norm_sq_neon(&row) }.sqrt()));
                for (name, got) in &paths {
                    assert_eq!(
                        got.to_bits(),
                        scalar.to_bits(),
                        "len={len} seed={seed} path={name}: norm {got} != scalar {scalar}",
                    );
                }
            }
        }
    }

    /// The frozen reduction order, pinned against hand-computed values.
    /// Catches a change to `NORM_CHAINS` or to the combine tree — either
    /// of which silently re-encodes every future index differently from
    /// one built by an earlier build.
    #[test]
    fn norm_reduction_order_is_frozen() {
        // 16 elements: chains are (j, j+8) pairs.
        let row: Vec<f32> = (0..16).map(|i| (i as f32) + 1.0).collect();
        let mut chains = [0.0f32; NORM_CHAINS];
        for (j, &x) in row.iter().enumerate() {
            chains[j % NORM_CHAINS] += x * x;
        }
        let expect =
            (((chains[0] + chains[1]) + (chains[2] + chains[3]))
                + ((chains[4] + chains[5]) + (chains[6] + chains[7])))
                .sqrt();
        assert_eq!(simd_norm(&row).to_bits(), expect.to_bits());
        // sum of squares 1..16 = 1496; sqrt is exact enough to state.
        assert_eq!(NORM_CHAINS, 8, "NORM_CHAINS is part of the encode contract");
        assert!((simd_norm(&row) - 1496.0f32.sqrt()).abs() < 1e-3);
    }

    /// The vector validation predicate must agree with the scalar scan on
    /// clean input, and report the identical first-invalid index for NaN,
    /// +/-Inf, and over-magnitude values at every lane position.
    #[test]
    fn validation_matches_scalar_exactly() {
        let n = 100;
        let clean = pseudo_rows(1, n, 7);
        assert_eq!(
            first_invalid_in_chunk(&clean, 1e16),
            first_invalid_in_chunk_scalar(&clean, 1e16)
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e16, -2e16] {
            for pos in [0usize, 1, 7, 8, 15, 63, 64, 96, 99] {
                let mut v = clean.clone();
                v[pos] = bad;
                assert_eq!(
                    first_invalid_in_chunk(&v, 1e16),
                    first_invalid_in_chunk_scalar(&v, 1e16),
                    "bad={bad} pos={pos}"
                );
                assert_eq!(first_invalid_in_chunk(&v, 1e16), Some(pos));
            }
        }
    }
}

#[cfg(test)]
mod anchor_tests {
    use super::*;
    use crate::codebook;

    /// Heavy-tailed rotated coordinates: a Gaussian bulk with a small
    /// fraction of far outliers, which is what real embeddings with
    /// kurtosis >> 3 look like per coordinate (lastfm-64 measures ~26).
    fn heavy_tailed(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as f64 / u64::MAX as f64
        };
        let sd = 1.0 / (dim as f64).sqrt();
        (0..n * dim)
            .map(|_| {
                let u = next();
                let g = (next() - 0.5) * 2.0;
                // 2% of samples get a 12x tail kick.
                let scale = if u < 0.02 { 12.0 } else { 1.0 };
                (g * sd * scale) as f32
            })
            .collect()
    }

    /// Overload — how far the calibrated tail runs past the outermost
    /// centroid — must not get *worse* as bits are added.
    ///
    /// More bits mean a finer codebook, so spending them should reduce
    /// distortion. Under the fixed 5%/95% anchor the fitted scale was
    /// the same at every bit width, so the calibrated values were the
    /// same while the codebook's reach barely moved: the 99.9th
    /// percentile stayed ~4x past the last level at 4 bits. Anchoring on
    /// the codebook makes the ratio non-increasing.
    #[test]
    fn overload_does_not_worsen_with_more_bits() {
        let (n, dim) = (4000usize, 64usize);
        let rotated = heavy_tailed(n, dim, 0xC0FFEE_99);
        let mut prev = f32::INFINITY;
        for bits in [2usize, 3, 4] {
            let (_, centroids) = codebook::codebook(bits, dim);
            let c_outer = centroids.iter().fold(0.0f32, |a, &c| a.max(c.abs()));
            let (shift, scale) = compute_tqplus_calibration(&rotated, n, dim, &centroids);

            let mut cal: Vec<f32> = Vec::with_capacity(n * dim);
            for i in 0..n {
                for d in 0..dim {
                    cal.push(((rotated[i * dim + d] + shift[d]) * scale[d]).abs());
                }
            }
            cal.sort_by(|a, b| a.partial_cmp(b).expect("values are finite"));
            let p999 = cal[(cal.len() as f64 * 0.999) as usize];
            let ratio = p999 / c_outer;
            assert!(
                ratio <= prev,
                "bits={bits}: tail overload ratio rose to {ratio} from {prev} — \
                 adding bits made the fit worse, which is the #454 failure mode"
            );
            prev = ratio;
        }
    }

    /// The anchor probability is a function of the codebook, so it must
    /// move with bit width — that is the whole content of #454. Pinning
    /// the ordering (not the values, which depend on `dim`) keeps a
    /// future "simplify this to a constant" from passing silently.
    #[test]
    fn the_anchor_probability_widens_with_bit_width() {
        let dim = 1536usize;
        let a = (dim as f64 - 1.0) / 2.0;
        let beta = Beta::new(a, a).expect("Beta(a, a) is valid for a > 0");
        let mut prev = 0.0f64;
        for bits in [2usize, 3, 4] {
            let (_, centroids) = codebook::codebook(bits, dim);
            let (p_lo, p_hi, qc_lo, qc_hi) = tqplus_anchor(&beta, &centroids);
            assert!(
                p_hi > prev,
                "bits={bits}: anchor probability {p_hi} did not widen past {prev}"
            );
            assert!((p_lo + p_hi - 1.0).abs() < 1e-12, "anchor must stay symmetric");
            assert_eq!(qc_lo, -qc_hi, "canonical targets must stay symmetric");
            // The targets are the codebook's own edges, not an interior
            // quantile of the marginal.
            let c_outer = centroids.iter().fold(0.0f32, |a, &c| a.max(c.abs()));
            assert_eq!(qc_hi, c_outer);
            prev = p_hi;
        }
    }
}
