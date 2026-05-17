#!/usr/bin/env python3
"""Generate distortion chart from benchmark results."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")


def main():
    with open(os.path.join(RESULTS_DIR, "distortion.json")) as f:
        results = json.load(f)

    bit_widths = [2, 3, 4]
    dims = [768, 1536, 3072]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {"768": "#2196F3", "1536": "#4CAF50", "3072": "#FF9800"}
    markers = {"768": "o", "1536": "s", "3072": "D"}

    for dim in dims:
        measured = []
        for bw in bit_widths:
            key = f"random_d{dim}_{bw}bit"
            measured.append(results[key]["measured_mse"])
        ax.plot(bit_widths, measured, marker=markers[str(dim)], color=colors[str(dim)],
                linewidth=2, markersize=8, label=f"Measured d={dim}")

    # Paper upper bound
    bounds = [(3 * np.pi / 2) * (1.0 / (4 ** b)) for b in bit_widths]
    ax.plot(bit_widths, bounds, "--", color="#F44336", linewidth=2,
            marker="x", markersize=10, label="Paper upper bound")

    # Paper empirical
    paper_emp = [0.117, 0.03, 0.009]
    ax.plot(bit_widths, paper_emp, "--", color="#9C27B0", linewidth=2,
            marker="^", markersize=8, label="Paper empirical (Table 1)")

    ax.set_xlabel("Bit width", fontsize=12)
    ax.set_ylabel("MSE (lower is better)", fontsize=12)
    ax.set_title("Quantization Distortion — Measured vs Paper Bounds\n"
                 "10K random unit vectors, self-retrieval score method",
                 fontsize=11)
    ax.set_xticks(bit_widths)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.8, 4.2)

    plt.tight_layout()
    out_path = os.path.join(DOCS_DIR, "distortion.svg")
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Chart saved to {out_path}")

    # Also save PNG for preview
    png_path = os.path.join(DOCS_DIR, "distortion.png")
    plt.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    print(f"PNG saved to {png_path}")


if __name__ == "__main__":
    main()
