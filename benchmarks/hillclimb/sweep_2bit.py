"""The boundary gates: an nq sweep and an N sweep, MT and ST.

The eight objective cells sample two query widths and one index size. Two
sample points cannot see a cliff between them, and the 4-bit climb was bitten
twice by exactly that — P27's small-N parallelism collapse and H90's nq=10 at
18.62 ms, both opened by changes the cells certified as wins. Anything that
moves a dispatch boundary (batch width, tile floor, parallelism gate, arm
selection) shows up here and nowhere else.

Gate: no point regressing more than 3% against its own pinned baseline.

Usage:  sweep_2bit.py [--bits {2,4}] [--reps N] [--out FILE]
"""
import json
import sys

from cells_2bit import DIM, K, N, arg, ensure_index, search_cell

NQ_POINTS = list(range(1, 17)) + [32, 64]
N_POINTS = [1_000, 8_192, 32_768, 200_000]


def sweep(bits, reps):
    path = ensure_index(bits)
    points = {}
    for st in (False, True):
        tag = "st" if st else "mt"
        # Best of three sub-runs per point, as the objective cells do. Without
        # it the small-nq points are worthless: a single-process sweep put
        # nq=2 MT at x0.56 against an unchanged binary, because at 0.5 ms a
        # perturbed process swamps the measurement (the H51 effect).
        for nq in NQ_POINTS:
            # per-query ms, so a cliff reads as a cliff rather than as slope
            # Nine sub-runs below nq=5. Those points are ~0.3 ms and the
            # thread-pool wakeup dominates them; three was not enough to reject
            # a perturbed process.
            subruns = 9 if nq <= 4 else 3
            ms = min(search_cell(path, nq, st, reps * (5 if nq <= 4 else 1))
                     for _ in range(subruns))
            points[f"nq{nq}_{tag}"] = ms / nq
        for n in N_POINTS:
            p = ensure_index(bits, n) if n != N else path
            points[f"n{n}_{tag}"] = min(search_cell(p, 100, st, reps)
                                        for _ in range(3))
    return points


if __name__ == "__main__":
    bits = arg("--bits", 2)
    points = sweep(bits, arg("--reps", 5))
    blob = json.dumps({"bits": bits, "dim": DIM, "k": K, "points": points})
    out = arg("--out", None, str)
    if out:
        with open(out, "w") as fh:
            fh.write(blob + "\n")
    print(blob)
