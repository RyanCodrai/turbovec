//! Encode vectors: normalize, rotate, quantize, bit-pack, compute per-vector scale.
//!
//! The returned `scales` array replaces the previous `norms` array. For each
//! vector `v` with rotated unit form `u` and reconstructed centroid vector
//! `x_hat`, the stored scale is `||v|| / <u, x_hat>` — the RaBitQ-style
//! length-renormalization correction adapted to turbovec's Lloyd-Max
//! codebook. Applying `scale` (instead of `||v||`) at the final
//! score-multiplication site in the SIMD kernel converts the biased
//! estimator `||v|| * <x_hat, y_rot>` into the unbiased estimator
//! `(||v|| / <u, x_hat>) * <x_hat, y_rot>`. When quantization is perfect
//! (`x_hat = u`), `<u, x_hat> = 1` and scale reduces to `||v||`.

use ndarray::ArrayView2;

/// Encode n vectors of dimension dim.
/// Returns (packed_codes as flat Vec<u8>, scales as Vec<f32>).
pub fn encode(
    vectors: &[f32],
    n: usize,
    dim: usize,
    rotation: &[f32], // dim x dim, row-major
    boundaries: &[f32],
    centroids: &[f32],
    bit_width: usize,
) -> (Vec<u8>, Vec<f32>) {
    let mut norms = vec![0.0f32; n];
    let mut unit_flat = vec![0.0f32; n * dim];

    // 1. Extract norms and normalize
    for i in 0..n {
        let row = &vectors[i * dim..(i + 1) * dim];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        norms[i] = norm;
        let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
        for j in 0..dim {
            unit_flat[i * dim + j] = row[j] * inv_norm;
        }
    }

    // 2. Rotate: rotated = unit @ rotation.T (BLAS-accelerated via ndarray)
    let unit_mat = ArrayView2::from_shape((n, dim), &unit_flat).unwrap();
    let rot_mat = ArrayView2::from_shape((dim, dim), rotation).unwrap();
    let rotated_mat = unit_mat.dot(&rot_mat.t());
    let rotated = rotated_mat.as_slice().unwrap();

    // 3. Quantize: for each boundary, codes += (rotated > boundary)
    let mut codes = vec![0u8; n * dim];
    for b in boundaries {
        for idx in 0..n * dim {
            if rotated[idx] > *b {
                codes[idx] += 1;
            }
        }
    }

    // 4. Compute per-vector correction scale = ||v|| / <u, x_hat>.
    //    inner = sum_j rotated[i,j] * centroids[codes[i,j]]
    //    scale[i] = norms[i] / inner   (with epsilon floor)
    let mut scales = vec![0.0f32; n];
    for i in 0..n {
        let row_start = i * dim;
        let mut inner = 0.0f64;
        for j in 0..dim {
            let r = rotated[row_start + j] as f64;
            let c = centroids[codes[row_start + j] as usize] as f64;
            inner += r * c;
        }
        // Floor at small positive value. With Lloyd-Max centroids each
        // coord-rotated-onto-centroid pair contributes non-negatively in
        // expectation (centroid sign matches input sign by construction),
        // so inner is essentially always positive — but guard anyway so a
        // pathological vector doesn't blow up the score.
        let inner = inner.max(1e-10) as f32;
        scales[i] = norms[i] / inner;
    }

    // 5. Bit-pack into bit-plane format
    let packed = pack_codes(&codes, n, dim, bit_width);

    (packed, scales)
}

/// Inverse of `pack_codes` — decode bit-plane packed codes back into a
/// flat `[u8; n * dim]` of integer codes. Used by the exact-math
/// debug search path to reconstruct x_hat without going through the
/// SIMD popcount kernel. Not on the hot path.
pub fn unpack_codes(packed: &[u8], n: usize, dim: usize, bits: usize) -> Vec<u8> {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    let mut codes = vec![0u8; n * dim];
    for i in 0..n {
        for j in 0..dim {
            let byte_pos = j / 8;
            let bit_pos = 7 - (j % 8);
            let mut code = 0u8;
            for p in 0..bits {
                let plane_byte = packed[i * bytes_per_row + p * bytes_per_plane + byte_pos];
                if plane_byte & (1 << bit_pos) != 0 {
                    code |= 1 << p;
                }
            }
            codes[i * dim + j] = code;
        }
    }
    codes
}

/// Pack quantized codes into bit-plane format.
fn pack_codes(codes: &[u8], n: usize, dim: usize, bits: usize) -> Vec<u8> {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    let mut packed = vec![0u8; n * bytes_per_row];

    for i in 0..n {
        for j in 0..dim {
            let code = codes[i * dim + j];
            let byte_pos = j / 8;
            let bit_pos = 7 - (j % 8);
            for p in 0..bits {
                if code & (1 << p) != 0 {
                    packed[i * bytes_per_row + p * bytes_per_plane + byte_pos] |= 1 << bit_pos;
                }
            }
        }
    }

    packed
}
