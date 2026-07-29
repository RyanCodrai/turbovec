#!/usr/bin/env python3
"""Weighted harmonic mean of per-cell speedups vs pinned baselines.

Usage:
    python benchmarks/whm.py baseline.json current.json [--target OP]

Both JSON files map cell name -> median ms, cells being "<op>-<arch>" for
op in {search, save, load, insert, delete, load_search} and arch in
{arm, x86, arm_st, x86_st} — the _st variants run with RAYON_NUM_THREADS=1
(both modes are first-class: an MT win must not regress ST).
Prints per-cell speedups and the WHM; with --target, also verdicts the run:
the target op's cells must improve and every other cell must stay within
NOISE_TOLERANCE of baseline. Exit code 0 = win, 1 = no-win/regression.
"""

import argparse
import json
import sys

OPS = ["search", "save", "load", "insert", "delete", "load_search"]
ARCHES = ["arm", "x86", "arm_st", "x86_st"]

# Per-op weights in the objective. load_search is gate-only (weight 0):
# it exists to veto load/search cost-shifting, not to double-credit wins.
WEIGHTS = {
    "search": 3.0,
    "insert": 2.0,
    "delete": 2.0,
    "save": 1.0,
    "load": 1.0,
    "load_search": 0.0,
}

NOISE_TOLERANCE = 0.03  # non-target cells may regress at most 3%
MIN_IMPROVEMENT = 0.01  # target op must improve >1% (HM of its arm+x86 cells)


def cells():
    return [f"{op}-{arch}" for op in OPS for arch in ARCHES]


def whm(speedups):
    num = den = 0.0
    for cell, s in speedups.items():
        w = WEIGHTS[cell.rsplit("-", 1)[0]]
        num += w
        den += w / s
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("current")
    ap.add_argument("--target", choices=OPS)
    args = ap.parse_args()

    base = json.load(open(args.baseline))
    cur = json.load(open(args.current))

    speedups = {}
    for cell in cells():
        if cell not in base or cell not in cur:
            sys.exit(f"missing cell: {cell}")
        speedups[cell] = base[cell] / cur[cell]

    for cell in cells():
        print(f"{cell:18s} {base[cell]:10.3f} -> {cur[cell]:10.3f} ms   x{speedups[cell]:.3f}")
    print(f"{'WHM':18s} x{whm(speedups):.4f}")

    if args.target is None:
        return

    target_cells = [c for c in cells() if c.rsplit("-", 1)[0] == args.target]
    regressed = [
        c for c in cells()
        if c not in target_cells and speedups[c] < 1.0 - NOISE_TOLERANCE
    ]
    target_hm = len(target_cells) / sum(1.0 / speedups[c] for c in target_cells)
    print(f"{'target HM':18s} x{target_hm:.4f}")

    if regressed:
        print(f"VERDICT: FAIL — regressions outside noise: {', '.join(regressed)}")
        sys.exit(1)
    if any(speedups[c] < 1.0 for c in target_cells):
        print("VERDICT: FAIL — a target cell regressed")
        sys.exit(1)
    if target_hm < 1.0 + MIN_IMPROVEMENT:
        print(f"VERDICT: FAIL — target improvement x{target_hm:.4f} below 1% threshold")
        sys.exit(1)
    print("VERDICT: WIN")


if __name__ == "__main__":
    main()
