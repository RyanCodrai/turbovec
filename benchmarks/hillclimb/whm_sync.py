#!/usr/bin/env python3
"""Weighted harmonic mean of per-cell sync() speedups vs pinned baselines.

Usage:
    python benchmarks/hillclimb/whm_sync.py baseline.json current.json \\
        [--target {sync_append,sync_remove}]

Both JSON files map cell name -> median ms, cells being "<cell>-<arch>" for
arch in {arm, x86}. The objective is the weighted harmonic mean of the four
speedups over the two measured cells:

    sync_append  weight 1   sync of a 32-row append
    sync_remove  weight 1   sync of 1000 scattered removals

`sync_first` (the full write) and `sync_settle` (the follow-up sync that
materializes a removal's header ops) carry weight 0: they are gate cells.
They exist so that moving work *out* of a measured sync and *into* the full
write or the next sync reads as what it is — cost-shifting, not a win.

With --target, the run is verdicted: the target cell's arm+x86 harmonic mean
must clear MIN_IMPROVEMENT, neither of its cells may regress at all, and no
other cell — gates included — may regress by more than NOISE_TOLERANCE.
Exit code 0 = win, 1 = no-win/regression.

The fsync floor dominates both measured cells, so a margin inside fsync
variance is not a win however consistent it looks. Pass --fsync-floor with
the box's measured single-fsync cost to have the script report how much of
each cell is even addressable; a claimed gain that exceeds the non-fsync
remainder is a measurement artifact, not a speedup.
"""

import argparse
import json
import sys

MEASURED = ["sync_append", "sync_remove"]
GATES = ["sync_first", "sync_settle"]
ARCHES = ["arm", "x86"]

WEIGHTS = {"sync_append": 1.0, "sync_remove": 1.0, "sync_first": 0.0,
           "sync_settle": 0.0}

NOISE_TOLERANCE = 0.03   # non-target cells may regress at most 3%
MIN_IMPROVEMENT = 0.01   # target must improve >1% (HM of its arm+x86 cells)


def speedups(base, cur):
    """baseline_ms / current_ms per cell present in both, as a dict."""
    out = {}
    for cell, b in base.items():
        if cell.endswith("#incremental") or cell not in cur:
            continue
        c = cur[cell]
        if not isinstance(b, (int, float)) or not isinstance(c, (int, float)):
            continue
        if c <= 0:
            raise SystemExit(f"{cell}: non-positive current value {c}")
        out[cell] = b / c
    return out


def whm(sp):
    num = den = 0.0
    for cell, s in sp.items():
        w = WEIGHTS.get(cell.rsplit("-", 1)[0], 0.0)
        num += w
        den += w / s
    return num / den if den else float("nan")


def hmean(vals):
    return len(vals) / sum(1.0 / v for v in vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("current")
    ap.add_argument("--target", choices=MEASURED)
    ap.add_argument("--fsync-floor", type=float, default=None,
                    help="measured cost of one fsync on this box, in ms")
    args = ap.parse_args()

    base = json.load(open(args.baseline))
    cur = json.load(open(args.current))

    # An incremental cell that fell back to a full rewrite is not the cell
    # it claims to be; refuse to score the run at all.
    for k, v in cur.items():
        if k.endswith("#incremental") and v is False:
            raise SystemExit(f"REFUSING TO SCORE: {k} is false — that cell "
                             f"fell back to a full rewrite")

    sp = speedups(base, cur)
    if not sp:
        raise SystemExit("no comparable cells between the two files")

    print(f"{'cell':<26}{'base ms':>10}{'cur ms':>10}{'speedup':>10}")
    for cell in sorted(sp):
        w = WEIGHTS.get(cell.rsplit('-', 1)[0], 0.0)
        tag = "" if w else "   (gate)"
        print(f"{cell:<26}{base[cell]:>10.2f}{cur[cell]:>10.2f}"
              f"{sp[cell]:>9.3f}x{tag}")
    print(f"\nWHM (measured cells, weight 1:1): {whm(sp):.4f}x")

    if args.fsync_floor:
        print(f"\nAddressable headroom above the {args.fsync_floor:.2f} ms "
              f"fsync floor:")
        for cell in sorted(sp):
            if cell.rsplit("-", 1)[0] not in MEASURED:
                continue
            rem = base[cell] - args.fsync_floor
            pct = 100.0 * rem / base[cell] if base[cell] else 0.0
            print(f"  {cell:<24}{rem:>8.2f} ms  ({pct:.1f}% of the cell)")

    if not args.target:
        return

    tcells = [f"{args.target}-{a}" for a in ARCHES if f"{args.target}-{a}" in sp]
    if len(tcells) != len(ARCHES):
        raise SystemExit(f"target {args.target} is missing an arch: "
                         f"have {tcells}")
    thm = hmean([sp[c] for c in tcells])
    regressed = [c for c in tcells if sp[c] < 1.0]
    others = {c: s for c, s in sp.items() if c not in tcells}
    hurt = [c for c, s in others.items() if s < 1.0 - NOISE_TOLERANCE]

    print(f"\ntarget {args.target}: HM(arm,x86) = {thm:.4f}x")
    win = True
    if thm < 1.0 + MIN_IMPROVEMENT:
        print(f"  NO-WIN: below the {1 + MIN_IMPROVEMENT:.2f}x bar")
        win = False
    if regressed:
        print(f"  NO-WIN: target cell(s) regressed: "
              f"{', '.join(f'{c} {sp[c]:.3f}x' for c in regressed)}")
        win = False
    if hurt:
        kind = ["gate " if c.rsplit("-", 1)[0] in GATES else "" for c in hurt]
        print("  NO-WIN: " + ", ".join(
            f"{k}cell {c} regressed to {sp[c]:.3f}x"
            for k, c in zip(kind, hurt)))
        win = False
    print("\nVERDICT: " + ("WIN" if win else "NO-WIN"))
    sys.exit(0 if win else 1)


if __name__ == "__main__":
    main()
