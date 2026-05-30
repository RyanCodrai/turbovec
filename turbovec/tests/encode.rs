//! End-to-end correctness tests for the encoding pipeline.
//!
//! `encode::encode` is the direct entry point for normalize ->
//! rotate -> quantize -> bit-pack -> compute correction scale.
//! Going through it (rather than through `TurboQuantIndex`) lets us
//! verify the low-level output shape and the per-vector scale value
//! without reaching into private state.

use turbovec::codebook::codebook;
use turbovec::encode::encode;
use turbovec::rotation::make_rotation_matrix;

fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..(n * dim) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (((state >> 32) as u32) & 0x007FFFFF) | 0x3F800000;
        let uniform = f32::from_bits(bits) - 1.0;
        out.push(uniform * 2.0 - 1.0);
    }
    out
}

#[test]
fn produces_expected_shape_for_bit_width_three() {
    // 3-bit is its own pack-layout branch (3 planes per row instead of
    // 2 or 4 per byte). The parametrised `produces_expected_shape` test
    // only covers 2 and 4 — pin 3 explicitly so a regression in the
    // 3-bit packing path doesn't slip through.
    let dim = 128;
    let n = 17;
    let rotation = make_rotation_matrix(dim);
    let (boundaries, centroids) = codebook(3, dim);
    let vectors = make_vectors(n, dim, 0);

    let (packed, scales, _, _) = encode(
        &vectors, n, dim, &rotation, &boundaries, &centroids, 3, None,
    );

    let bytes_per_row = 3 * (dim / 8);
    assert_eq!(packed.len(), n * bytes_per_row);
    assert_eq!(scales.len(), n);
}

#[test]
fn produces_expected_shape() {
    for &bit_width in &[2usize, 4] {
        let dim = 128;
        let n = 17;
        let rotation = make_rotation_matrix(dim);
        let (boundaries, centroids) = codebook(bit_width, dim);
        let vectors = make_vectors(n, dim, 0);

        let (packed, scales, _, _) = encode(
            &vectors, n, dim, &rotation, &boundaries, &centroids, bit_width, None,
        );

        let bytes_per_row = bit_width * (dim / 8);
        assert_eq!(
            packed.len(),
            n * bytes_per_row,
            "wrong packed length for bits={}, dim={}",
            bit_width,
            dim
        );
        assert_eq!(scales.len(), n);
    }
}

#[test]
fn scales_satisfy_rabitq_identity() {
    // Verify the mathematical identity that defines the correction:
    //     scale[i] = ||v_i|| / <u_i, x_hat_i>
    // where u_i is the rotated unit vector and x_hat_i is the
    // reconstructed-from-centroids vector. Both sides recoverable
    // here from the inputs.
    let dim = 128;
    let n = 10;
    let rotation = make_rotation_matrix(dim);
    let (boundaries, centroids) = codebook(4, dim);
    let vectors = make_vectors(n, dim, 0);

    let (_, scales, _, _) = encode(&vectors, n, dim, &rotation, &boundaries, &centroids, 4, None);

    // Reconstruct <u, x_hat> per vector and check scale = ||v|| / <u, x_hat>.
    for i in 0..n {
        let row = &vectors[i * dim..(i + 1) * dim];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv_norm = 1.0 / norm;

        // Rotate the unit vector: u_rot[k] = sum_j rotation[k*dim+j] * row[j] * inv_norm
        let mut u_rot = vec![0.0f32; dim];
        for k in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..dim {
                acc += rotation[k * dim + j] * row[j] * inv_norm;
            }
            u_rot[k] = acc;
        }

        // Quantize each coord and look up centroid (x_hat values).
        let mut inner = 0.0f64;
        for k in 0..dim {
            let mut code: usize = 0;
            for &b in &boundaries {
                if u_rot[k] > b {
                    code += 1;
                }
            }
            inner += (u_rot[k] as f64) * (centroids[code] as f64);
        }
        let expected_scale = norm as f64 / inner.max(1e-10);

        let rel_err = (scales[i] as f64 - expected_scale).abs() / expected_scale.abs().max(1e-10);
        assert!(
            rel_err < 1e-4,
            "scale identity broken at i={}: stored={}, expected={}, rel_err={}",
            i,
            scales[i],
            expected_scale,
            rel_err,
        );
    }
}

#[test]
fn deterministic_output() {
    let dim = 128;
    let n = 5;
    let rotation = make_rotation_matrix(dim);
    let (boundaries, centroids) = codebook(4, dim);
    let vectors = make_vectors(n, dim, 0);

    let (p1, s1, _, _) = encode(&vectors, n, dim, &rotation, &boundaries, &centroids, 4, None);
    let (p2, s2, _, _) = encode(&vectors, n, dim, &rotation, &boundaries, &centroids, 4, None);

    assert_eq!(p1, p2);
    assert_eq!(s1, s2);
}

#[test]
fn handles_zero_vector() {
    // A zero-norm vector must not produce NaN codes or NaN scales.
    let dim = 128;
    let rotation = make_rotation_matrix(dim);
    let (boundaries, centroids) = codebook(4, dim);
    let zeros = vec![0.0f32; dim];

    let (packed, scales, _, _) = encode(&zeros, 1, dim, &rotation, &boundaries, &centroids, 4, None);

    // ||v|| = 0 => scale = 0 / <u, x_hat>_floor = 0
    assert_eq!(scales[0], 0.0);
    assert!(scales[0].is_finite());
    let bytes_per_row = 4 * (dim / 8);
    assert_eq!(packed.len(), bytes_per_row);
}
