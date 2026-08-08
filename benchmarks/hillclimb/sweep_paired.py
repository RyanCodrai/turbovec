"""Paired per-point A/B sweep — the drift-cancelling gate instrument.

Pass-level ABBA measures all 88 points under one label, then all 88 under the
other, minutes apart; machine-state drift over that gap shows up as per-point
"structural" noise (a no-op read x0.82 on the worst point, 16/88 points past
3%). The 4-bit climb's H128/H129 lesson was ABBA *within pairs*: here each
point runs label A and label B back-to-back seconds apart, the ratio is taken
inside the pair, and slow drift cancels by construction. Three pairs per
point, in ABBA/BAAB order, median of the three paired ratios.

Usage: sweep_paired.py --a SO_A --b SO_B [--reps N] [--pairs N] [--out FILE]
(with --a == --b this measures its own noise floor, which is how the
instrument is validated before any candidate is scored.)
"""
import json
import shutil
import sys

from cells_2bit import DIM, K, N, arg, ensure_index, search_cell

NQ_POINTS = list(range(1, 17)) + [32, 64]
N_POINTS = [1_000, 8_192, 32_768, 200_000]
DEST = "/home/ryan/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so"


def measure(path, nq, st, reps, k=K):
    return search_cell(path, nq, st, reps)


def paired(so_a, so_b, reps, pairs):
    path = ensure_index(2)
    for n in N_POINTS:
        ensure_index(2, n)
    ratios = {}
    # Adaptive rep budget: every point gets >=100 ms of measured work per
    # process, which is the objective cells' own budget rule. The flat
    # `reps=5` the N-sweep points had been running gave n8192 five
    # milliseconds of work per process — the null failures were concentrated
    # on exactly the under-budgeted points, because they measured process
    # jitter, not the kernel.
    est = {}
    for nq in NQ_POINTS:
        est["nq%d" % nq] = (N, nq, max(0.05, 0.02 * nq))          # ms, rough
    est["n1000"], est["n8192"] = (1_000, 100, 0.3), (8_192, 100, 1.2)
    est["n32768"], est["n200000"] = (32_768, 100, 4.0), (200_000, 100, 20.0)
    jobs = []
    for name, (n, nq, ms) in est.items():
        r = max(5, int(100.0 / ms))
        jobs.append((name, n, nq, min(r, 800)))
    for st in (False, True):
        tag = "st" if st else "mt"
        for name, n, nq, r in jobs:
            p = ensure_index(2, n) if n != N else path
            rs = []
            for i in range(pairs):
                # Keyed by position, not by path: with --a == --b (the null
                # test) path-keying collapses the dict to one entry and every
                # ratio is exactly 1.0 — a vacuous control that in fact
                # "passed" once before this was caught.
                first_is_a = i % 2 == 0
                order = (so_a, so_b) if first_is_a else (so_b, so_a)
                got = []
                for so in order:
                    shutil.copyfile(so, DEST)
                    # min over three processes per side: pairing cancels the
                    # session-scale drift, but each process can still be
                    # individually perturbed (H51), and one bad process on one
                    # side poisons the whole pair — the first paired null
                    # failed at x0.55 on exactly that. Two noise sources, two
                    # defenses, both needed.
                    got.append(min(measure(p, nq, st, r) for _ in range(3)))
                va, vb = (got[0], got[1]) if first_is_a else (got[1], got[0])
                rs.append(va / vb)
            import statistics
            key = f"{name}_{tag}"
            ratios[key] = statistics.median(rs)   # a/b: >1 means b faster
    return ratios


if __name__ == "__main__":
    so_a = arg("--a", None, str)
    so_b = arg("--b", None, str)
    reps = arg("--reps", 5)
    pairs = arg("--pairs", 3)
    ratios = paired(so_a, so_b, reps, pairs)
    blob = json.dumps({"bits": 2, "ratios": ratios})
    out = arg("--out", None, str)
    if out:
        with open(out, "w") as fh:
            fh.write(blob + "\n")
    print(blob)
