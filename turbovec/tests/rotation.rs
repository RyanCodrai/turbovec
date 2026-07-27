//! Correctness tests for the deterministic block-Hadamard rotation.
//!
//! The rotation is a globally-permuted block-Hadamard transform at k=2
//! rounds (see `turbovec::rotation`). These tests verify the mathematical
//! invariants the downstream quantizer depends on:
//!
//! - Orthonormality: the transform preserves L2 norms and inner products
//!   (an isometry), and its effective matrix has orthonormal columns.
//! - Determinism: two rotations of the same input are bit-identical.
//!
//! Coverage deliberately includes the `8·odd` dims (200, 1000), where the
//! block collapses to 8 and `1/√8` is not exactly representable in f32.

use turbovec::rotation::Rotation;

/// Seeded pseudo-random vector in [-1, 1], independent of any rng crate.
fn rand_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (((state >> 32) as u32) & 0x007F_FFFF) | 0x3F80_0000;
            (f32::from_bits(bits) - 1.0) * 2.0 - 1.0
        })
        .collect()
}

fn l2_norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64) * (y as f64)).sum()
}

fn rotated(dim: usize, v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    Rotation::new(dim).apply(&mut out);
    out
}

#[test]
fn preserves_norm_across_dims_including_collapse_to_8() {
    // 200 and 1000 are 8·odd, so the block collapses to 8 (1/√8 inexact).
    for &dim in &[8usize, 200, 768, 1000, 1536, 3072] {
        for seed in 0..4u64 {
            let x = rand_vec(dim, seed);
            let y = rotated(dim, &x);
            let nx = l2_norm(&x);
            let ny = l2_norm(&y);
            assert!(
                (nx - ny).abs() / nx < 1e-4,
                "norm changed at dim={dim}, seed={seed}: |x|={nx}, |Rx|={ny}"
            );
        }
    }
}

#[test]
fn preserves_inner_products_across_dims() {
    // An orthonormal transform preserves <x, y>; checking this on random
    // pairs certifies orthonormality in O(dim) even at large dims where a
    // full R^T R = I check would be O(dim^3).
    for &dim in &[8usize, 200, 768, 1000, 1536, 3072] {
        let x = rand_vec(dim, 1);
        let y = rand_vec(dim, 2);
        let rx = rotated(dim, &x);
        let ry = rotated(dim, &y);
        let before = dot(&x, &y);
        let after = dot(&rx, &ry);
        let scale = l2_norm(&x) * l2_norm(&y);
        assert!(
            (before - after).abs() / scale < 1e-4,
            "inner product not preserved at dim={dim}: <x,y>={before}, <Rx,Ry>={after}"
        );
    }
}

#[test]
fn effective_matrix_has_orthonormal_columns_small_dims() {
    // Full R^T R = I at small dims, including a collapse-to-8 case (24 =
    // 8·3, three Hadamard blocks) and a pure power-of-two (32).
    for &dim in &[8usize, 24, 32, 256] {
        // Column j of the effective matrix is R·e_j.
        let cols: Vec<Vec<f32>> = (0..dim)
            .map(|j| {
                let mut e = vec![0.0f32; dim];
                e[j] = 1.0;
                rotated(dim, &e)
            })
            .collect();
        for i in 0..dim {
            for j in i..dim {
                let d = dot(&cols[i], &cols[j]);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (d - expected).abs() < 1e-4,
                    "R^T R[{i}][{j}] = {d}, expected {expected} (dim={dim})"
                );
            }
        }
    }
}

#[test]
fn deterministic_for_same_dim() {
    for &dim in &[8usize, 128, 1000] {
        let x = rand_vec(dim, 7);
        let a = rotated(dim, &x);
        let b = rotated(dim, &x);
        assert_eq!(a.len(), b.len());
        for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "rotation[{i}] differs across calls at dim={dim}: {p} vs {q}"
            );
        }
    }
}
