"""Score a candidate against the pinned baseline and call the verdict.

This script is the only authority on whether a candidate won. Prose in the log
is not — a section header once read "WIN" while this printed "not a win", and
the goal now names the script instead.

Objective: harmonic mean of the 8 per-cell speedups (baseline/candidate), equal
weights, across both arches. A win is HM > 1.01 with **no cell below 0.99**.

The 0.99 floor is not slack, it is calibration. A patch whose aarch64 hunk was
a comment — a byte-identical binary — measured x0.9938 on `arm nq1_mt`. A gate
at 1.00 fails an unchanged kernel, so it tests the machine, not the change. The
floor sits wider than every measured cell spread but still catches a real 1%
regression.

Per-arch HMs are printed alongside the 8-cell HM, so a one-arch result is never
dressed up as a two-arch one.

Usage:
  whm_2bit.py --base BASE_ARM BASE_X86 --cand CAND_ARM CAND_X86
              [--sweep-base B_ARM B_X86 --sweep-cand C_ARM C_X86]
              [--obs4-base B_ARM B_X86 --obs4-cand C_ARM C_X86]

Exit 0 only on VERDICT: WIN with every gate passing.
"""
import json
import sys

WIN_HM = 1.01
CELL_FLOOR = 0.99
SWEEP_FLOOR = 0.97
ARCHES = ("arm", "x86")
CELLS = ("nq1_st", "nq1_mt", "nq100_st", "nq100_mt")


def load(path, key):
    with open(path) as fh:
        return json.load(fh)[key]


def pairs(flag, key):
    """Both arches' files for `flag` as {cell_arch: ms}, or None if absent."""
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    out = {}
    for arch, path in zip(ARCHES, sys.argv[i + 1:i + 3]):
        for cell, ms in load(path, key).items():
            out[f"{cell}_{arch}"] = ms
    return out


def speedups(base, cand):
    return {c: base[c] / cand[c] for c in base if c in cand}


def hmean(xs):
    return len(xs) / sum(1.0 / x for x in xs)


if __name__ == "__main__":
    base, cand = pairs("--base", "cells"), pairs("--cand", "cells")
    if not base or not cand:
        sys.exit(__doc__)
    sp = speedups(base, cand)
    if len(sp) != 8:
        sys.exit(f"expected 8 cells, got {len(sp)}: {sorted(sp)}")

    print("cell            arm        x86")
    for cell in CELLS:
        a, x = sp[f"{cell}_arm"], sp[f"{cell}_x86"]
        flag = "  <-- below floor" if min(a, x) < CELL_FLOOR else ""
        print(f"  {cell:<12} x{a:.4f}    x{x:.4f}{flag}")

    hm = hmean(list(sp.values()))
    per = {a: hmean([sp[f"{c}_{a}"] for c in CELLS]) for a in ARCHES}
    worst = min(sp, key=sp.get)
    print(f"\n  arm 4-cell HM  x{per['arm']:.4f}")
    print(f"  x86 4-cell HM  x{per['x86']:.4f}")
    print(f"  8-cell HM      x{hm:.4f}   worst cell {worst} x{sp[worst]:.4f}")

    ok = hm > WIN_HM and sp[worst] >= CELL_FLOOR
    reasons = []
    if hm <= WIN_HM:
        reasons.append(f"HM x{hm:.4f} <= x{WIN_HM}")
    if sp[worst] < CELL_FLOOR:
        reasons.append(f"{worst} x{sp[worst]:.4f} < x{CELL_FLOOR}")

    sb, sc = pairs("--sweep-base", "points"), pairs("--sweep-cand", "points")
    if sb and sc:
        ssp = speedups(sb, sc)
        sworst = min(ssp, key=ssp.get)
        print(f"\n  sweep gate     worst {sworst} x{ssp[sworst]:.4f} (floor x{SWEEP_FLOOR})")
        if ssp[sworst] < SWEEP_FLOOR:
            ok = False
            reasons.append(f"sweep {sworst} x{ssp[sworst]:.4f} < x{SWEEP_FLOOR}")
    else:
        print("\n  sweep gate     NOT MEASURED — mandatory for dispatch-boundary changes")

    ob, oc = pairs("--obs4-base", "cells"), pairs("--obs4-cand", "cells")
    if ob and oc:
        o = speedups(ob, oc)
        print("\n  4-bit observation (recorded, never gated):")
        for cell in CELLS:
            print(f"    {cell:<12} arm x{o[f'{cell}_arm']:.4f}    x86 x{o[f'{cell}_x86']:.4f}")

    print(f"\nVERDICT: {'WIN' if ok else 'NOT A WIN'}" + ("" if ok else "  (" + "; ".join(reasons) + ")"))
    sys.exit(0 if ok else 1)
