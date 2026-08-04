#!/usr/bin/env python3
"""Weighted harmonic mean of the eight persistence cells vs pinned baselines.

Cells are "<op>-<arch>" for op in {save_warm, save_mut, load, load_search}
and arch in {arm, x86}; weights 1 : 2 : 2 : 0. load_search is gate-only —
it exists to veto cost-shifting out of load and into the first search, not
to credit wins.

Usage:
    python whm_persist.py baseline.json current.json [--target OP]

With --target, verdicts the run: the target op's two cells must both
improve, their HM must clear 1%, and no other cell — including any _st
guard cells present in both files — may regress beyond noise.
Exit 0 = win, 1 = no-win/regression.
"""

import argparse
import json
import sys

OPS = ["save_warm", "save_mut", "load", "load_search"]
ARCHES = ["arm", "x86"]

WEIGHTS = {"save_warm": 1.0, "save_mut": 2.0, "load": 2.0, "load_search": 0.0}

NOISE_TOLERANCE = 0.03  # non-target cells may regress at most 3%
MIN_IMPROVEMENT = 0.01  # target op must improve >1% (HM of its arm+x86 cells)


def cells():
    return [f"{op}-{arch}" for op in OPS for arch in ARCHES]


def whm(speedups):
    num = den = 0.0
    for cell in cells():
        w = WEIGHTS[cell.rsplit("-", 1)[0]]
        num += w
        den += w / speedups[cell]
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

    # Guard cells: anything present in both files that is not an objective
    # cell (the _st variants, or cells carried over from another harness).
    guards = {c: base[c] / cur[c] for c in base
              if c in cur and c not in speedups}

    for cell in cells():
        print(f"{cell:20s} {base[cell]:10.3f} -> {cur[cell]:10.3f} ms   "
              f"x{speedups[cell]:.3f}")
    for cell in sorted(guards):
        print(f"{cell:20s} {base[cell]:10.3f} -> {cur[cell]:10.3f} ms   "
              f"x{guards[cell]:.3f}  (guard)")
    print(f"{'WHM':20s} x{whm(speedups):.4f}")

    if args.target is None:
        return

    target_cells = [c for c in cells() if c.rsplit("-", 1)[0] == args.target]
    checked = {**speedups, **guards}
    regressed = [c for c in checked
                 if c not in target_cells and checked[c] < 1.0 - NOISE_TOLERANCE]
    target_hm = len(target_cells) / sum(1.0 / speedups[c] for c in target_cells)
    print(f"{'target HM':20s} x{target_hm:.4f}")

    if regressed:
        print(f"VERDICT: FAIL — regressions outside noise: {', '.join(sorted(regressed))}")
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
