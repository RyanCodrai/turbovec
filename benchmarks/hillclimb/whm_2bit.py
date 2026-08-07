"""Score a candidate against the pinned baseline and call the verdict.

Objective: harmonic mean of the 8 per-cell speedups (baseline/candidate),
equal weights, across both arches. A win is HM > 1.01 with no cell below 1.0.

Gates reported alongside, never folded into the score: the nq and N sweeps
(no point below 0.97) and the 4-bit observation cells, which are printed and
never acted on.

Usage:
  whm_2bit.py --base BASE_ARM BASE_X86 --cand CAND_ARM CAND_X86
              [--sweep-base B_ARM B_X86 --sweep-cand C_ARM C_X86]
              [--obs4-base B_ARM B_X86 --obs4-cand C_ARM C_X86]

Exit 0 only if the objective is a win and every gate passes.
"""
import json
import sys

WIN = 1.01
SWEEP_FLOOR = 0.97
ARCHES = ("arm", "x86")


def load(path, key):
    with open(path) as fh:
        return json.load(fh)[key]


def pairs(flag, key):
    """Both arches' files for `flag`, as {cell_arch: ms}, or None if absent."""
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    out = {}
    for arch, path in zip(ARCHES, sys.argv[i + 1:i + 3]):
        for cell, ms in load(path, key).items():
            out[f"{cell}_{arch}"] = ms
    return out


def speedups(base, cand):
    return {c: base[c] / cand[c] for c in sorted(base) if c in cand}


def hmean(xs):
    return len(xs) / sum(1.0 / x for x in xs)


def table(title, sp, floor):
    worst = min(sp.values())
    print(f"\n{title}")
    for c, s in sorted(sp.items(), key=lambda kv: kv[1]):
        print(f"  {c:<20} x{s:.4f}{'  <-- regression' if s < floor else ''}")
    return worst


if __name__ == "__main__":
    base, cand = pairs("--base", "cells"), pairs("--cand", "cells")
    if not base or not cand:
        sys.exit(__doc__)
    sp = speedups(base, cand)
    if len(sp) != 8:
        sys.exit(f"expected 8 cells, got {len(sp)}: {sorted(sp)}")

    worst = table("Objective cells (baseline/candidate)", sp, 1.0)
    hm = hmean(list(sp.values()))
    print(f"\n  HM = x{hm:.4f}   worst cell = x{worst:.4f}")

    ok = hm > WIN and worst >= 1.0
    sb, sc = pairs("--sweep-base", "points"), pairs("--sweep-cand", "points")
    if sb and sc:
        ssp = speedups(sb, sc)
        sworst = table("Sweep gate (floor x0.97)", ssp, SWEEP_FLOOR)
        ok = ok and sworst >= SWEEP_FLOOR
    else:
        print("\nSweep gate: NOT MEASURED — mandatory for any dispatch-boundary change")

    ob, oc = pairs("--obs4-base", "cells"), pairs("--obs4-cand", "cells")
    if ob and oc:
        table("4-bit observation (recorded, not gated)", speedups(ob, oc), float("-inf"))

    print(f"\nVERDICT: {'WIN' if ok else 'NOT A WIN'}")
    sys.exit(0 if ok else 1)
