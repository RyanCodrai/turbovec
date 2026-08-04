#!/usr/bin/env python3
"""Score a mutation hill-climb candidate against the pinned baseline.

Objective: weighted harmonic mean of the eight MT cells — {arm, x86} x
{bulk, append, single, remove} — with weights bulk 2, append 2, single 1,
remove 2. The `remove` cell speedup is the arithmetic mean of the `swap` and
`idremove` speedups. The `_st` cells are a hard gate, not part of the
objective: an ST regression is a failed hypothesis.

Usage: python whm_mutate.py BASELINE.json CANDIDATE.json [--target OP]
"""

import argparse
import json
import sys

OPS = ["bulk", "append", "single", "remove"]
WEIGHTS = {"bulk": 2, "append": 2, "single": 1, "remove": 2}
ARCHES = ["arm", "x86"]
MODES = ["", "_st"]
REGRESS = 0.99   # a target cell may not fall below this
NOISE = 0.97     # non-target cells must stay within this


def speedup(base, cand, op, arch):
    """base/cand for one op; `remove` averages its two sub-ops."""
    if op == "remove":
        vals = []
        for sub in ("swap", "idremove"):
            key = f"{sub}-{arch}"
            if key not in base or key not in cand:
                return None
            vals.append(base[key] / cand[key])
        return sum(vals) / len(vals)
    key = f"{op}-{arch}"
    if key not in base or key not in cand:
        return None
    return base[key] / cand[key]


def whm(pairs):
    """Weighted harmonic mean of (speedup, weight)."""
    num = sum(w for _, w in pairs)
    den = sum(w / s for s, w in pairs)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--target", choices=OPS, help="op this hypothesis targets")
    args = ap.parse_args()

    base = json.load(open(args.baseline))
    cand = json.load(open(args.candidate))

    rows, obj, flags = [], [], []
    for mode in MODES:
        for op in OPS:
            for arch in ARCHES:
                cell = f"{arch}{mode}"
                s = speedup(base, cand, op, cell)
                if s is None:
                    continue
                rows.append((op, cell, s))
                if mode == "":
                    obj.append((s, WEIGHTS[op]))
                is_target = args.target == op
                bar = REGRESS if is_target else NOISE
                if s < bar:
                    flags.append(f"{op}-{cell} x{s:.3f} (< {bar})")

    width = max(len(f"{op}-{cell}") for op, cell, _ in rows)
    for op, cell, s in rows:
        mark = " <-- target" if args.target == op else ""
        print(f"{op}-{cell:<{width - len(op)}} x{s:.3f}{mark}")

    print(f"\nWHM (8 MT cells): x{whm(obj):.4f}")

    if args.target:
        tgt_mt = [(s, 1) for op, cell, s in rows
                  if op == args.target and not cell.endswith("_st")]
        tgt_st = [(s, 1) for op, cell, s in rows
                  if op == args.target and cell.endswith("_st")]
        if tgt_mt:
            print(f"target {args.target} HM(arm,x86) MT: x{whm(tgt_mt):.4f}"
                  f"  (win bar: x1.01)")
        if tgt_st:
            print(f"target {args.target} HM(arm,x86) ST: x{whm(tgt_st):.4f}")

    # Sanity gate on the single-add cell. The absolute MT/ST ratio is NOT 1.0 on
    # baseline core — an MT 1-row add pays a pool handoff the ST sentinel pool
    # folds away, a stable ~1.35x on arm — so the gate is drift of the ratio away
    # from the baseline's own ratio, which is what a contaminated grid shows.
    for arch in ARCHES:
        cm, cs = cand.get(f"single-{arch}"), cand.get(f"single-{arch}_st")
        bm, bs = base.get(f"single-{arch}"), base.get(f"single-{arch}_st")
        if cm and cs and bm and bs:
            r_c, r_b = cm / cs, bm / bs
            drift = max(r_c, r_b) / min(r_c, r_b)
            verdict = ("ok" if drift < 1.15
                       else "CONTAMINATED — re-run this grid")
            print(f"sanity single-{arch} MT/ST {r_c:.3f} vs base {r_b:.3f} "
                  f"(drift {drift:.3f}) — {verdict}")

    if flags:
        print("\nFLAGS:")
        for f in flags:
            print(f"  {f}")
        sys.exit(1)
    print("\nno cell flagged")


if __name__ == "__main__":
    main()
