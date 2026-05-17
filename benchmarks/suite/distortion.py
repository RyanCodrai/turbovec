#!/usr/bin/env python3
"""Distortion benchmark: measure quantization MSE and compare to paper bounds.

Measures the mean squared error introduced by TurboQuant quantization at
each bit width (2, 3, 4) across multiple dimensions. Compares against the
theoretical bounds from Section 4.4 of the TurboQuant paper (arXiv:2504.19874).

For unit vectors, MSE = 2(1 - <x, x_hat>) where x_hat is the quantized
reconstruction. Since turbovec scores via inner product on normalized vectors,
we measure distortion as: MSE = 2 * (1 - self_score).

Paper's reported MSE values (Table 1, d -> infinity):
  2-bit: ~0.117
  3-bit: ~0.03
  4-bit: ~0.009

Paper's upper bound: D_mse <= (3*pi/2) * (1/4^b)
  2-bit: 0.294
  3-bit: 0.074
  4-bit: 0.018
"""
import json
import os
import time

import numpy as np

from turbovec import TurboQuantIndex

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
RESULT_FILE = os.path.join(RESULTS_DIR, "distortion.json")
SEED = 42


def generate_unit_vectors(n, dim, seed=SEED):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    return vecs


def measure_distortion(vectors, dim, bit_width):
    n = len(vectors)
    index = TurboQuantIndex(dim, bit_width)
    index.add(vectors)

    # Self-search: each vector queries against itself in the index.
    # For a perfect reconstruction, score would be 1.0 (cosine of identical vectors).
    # The gap from 1.0 is the distortion.
    scores, indices = index.search(vectors, k=1)
    scores = np.array(scores).reshape(n)

    # MSE = ||x - x_hat||^2 = ||x||^2 + ||x_hat||^2 - 2<x, x_hat>
    # For unit vectors where x_hat is also approximately unit norm:
    # MSE ≈ 2(1 - <x, x_hat>) = 2(1 - score)
    mse_per_vector = 2.0 * (1.0 - scores)
    mean_mse = float(np.mean(mse_per_vector))
    std_mse = float(np.std(mse_per_vector))
    return mean_mse, std_mse


def paper_upper_bound(bit_width):
    """D_mse <= (3*pi/2) * (1/4^b) from Theorem 3.1."""
    return (3 * np.pi / 2) * (1.0 / (4 ** bit_width))


def paper_empirical_mse(bit_width):
    """Approximate empirical MSE values from paper's Table 1 (d -> inf)."""
    table = {2: 0.117, 3: 0.03, 4: 0.009}
    return table.get(bit_width)


def main():
    configs = [
        ("random_d768", 768, 10000),
        ("random_d1536", 1536, 10000),
        ("random_d3072", 3072, 5000),
    ]

    results = {}

    for name, dim, n_vectors in configs:
        print(f"\n{'='*60}")
        print(f"Dataset: {name} (n={n_vectors}, dim={dim})")
        print(f"{'='*60}")

        vectors = generate_unit_vectors(n_vectors, dim)

        for bit_width in [2, 3, 4]:
            key = f"{name}_{bit_width}bit"
            print(f"\n  {bit_width}-bit:")

            t0 = time.time()
            mean_mse, std_mse = measure_distortion(vectors, dim, bit_width)
            elapsed = time.time() - t0

            bound = paper_upper_bound(bit_width)
            empirical = paper_empirical_mse(bit_width)
            ratio_to_bound = mean_mse / bound

            print(f"    Measured MSE:      {mean_mse:.6f} (std: {std_mse:.6f})")
            print(f"    Paper empirical:   {empirical}")
            print(f"    Paper upper bound: {bound:.6f}")
            print(f"    Ratio to bound:    {ratio_to_bound:.3f}x")
            print(f"    Within bound:      {'YES' if mean_mse <= bound else 'NO'}")
            print(f"    Time:              {elapsed:.2f}s")

            results[key] = {
                "dataset": name,
                "dim": dim,
                "n_vectors": n_vectors,
                "bit_width": bit_width,
                "measured_mse": round(mean_mse, 6),
                "mse_std": round(std_mse, 6),
                "paper_empirical_mse": empirical,
                "paper_upper_bound": round(bound, 6),
                "ratio_to_bound": round(ratio_to_bound, 4),
                "within_bound": mean_mse <= bound,
            }

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Measured':>10} {'Paper':>10} {'Bound':>10} {'Ratio':>8}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for key, r in results.items():
        emp = r['paper_empirical_mse'] or 'N/A'
        print(f"{key:<25} {r['measured_mse']:>10.6f} {str(emp):>10} {r['paper_upper_bound']:>10.6f} {r['ratio_to_bound']:>7.3f}x")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULT_FILE}")


if __name__ == "__main__":
    main()
