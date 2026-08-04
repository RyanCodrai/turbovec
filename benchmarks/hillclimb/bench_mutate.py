#!/usr/bin/env python3
"""Live-index mutation hill-climb benchmark: one JSON of `<op>-<arch>` -> median ms.

Ops (all at N=200k, dim=768, 4-bit):
- bulk:    cold bulk insert — fresh empty IdMapIndex, one add_with_ids of N rows
- append:  warm append — another BATCH rows onto the index the bulk just built
- single:  single add — SINGLES individual 1-row add_with_ids calls, per-call median
- swap:    TurboQuantIndex.swap_remove x REMOVES on an N-row index
- idremove: IdMapIndex.remove x REMOVES by external id (includes the one-time
           slot-map build, which is the real first-remove cost on a live index)

The scored `remove` cell is the average of the `swap` and `idremove` speedups;
whm.py does that combination, this harness reports the two raw timings.

Modes: default is the multicore pool; --st pins RAYON_NUM_THREADS=1 (single-core
cells, suffixed `<op>-<arch>_st`). Both modes matter — an MT win must not tax ST.
`single` is the sanity gate: it takes no pool handoff, so single-ST and single-MT
must agree; if they don't, the grid is contaminated and needs a re-run.

Usage: python bench_mutate.py --arch {arm,x86} [--st] [--reps N] [--out FILE]
"""

import argparse
import json
import os
import statistics
import sys

if "--st" in sys.argv:  # must precede the extension's first pool build
    os.environ["RAYON_NUM_THREADS"] = "1"

import numpy as np
from turbovec import IdMapIndex, TurboQuantIndex

N, DIM, BITS = 200_000, 768, 4
BATCH = 10_000   # warm-append batch
SINGLES = 100    # 1-row adds, timed individually
REMOVES = 10_000  # removals per timed loop (sub-ms at 1k — too short to time)


def median_ms(xs):
    return statistics.median(xs) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["arm", "x86"])
    ap.add_argument("--st", action="store_true",
                    help="single-core mode (RAYON_NUM_THREADS=1)")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out")
    args = ap.parse_args()
    cell_arch = args.arch + ("_st" if args.st else "")

    import time
    rng = np.random.default_rng(0)
    base = rng.random((N, DIM), dtype=np.float32)
    base_ids = np.arange(N, dtype=np.uint64)
    warm = rng.random((BATCH, DIM), dtype=np.float32)
    singles = rng.random((SINGLES, DIM), dtype=np.float32)

    t_bulk, t_append, t_single, t_swap, t_idrm = [], [], [], [], []

    for rep in range(args.reps):
        # --- cold bulk insert + warm append + single add, one live index ---
        ix = IdMapIndex(dim=DIM, bit_width=BITS)
        t0 = time.perf_counter()
        ix.add_with_ids(base, base_ids)
        t_bulk.append(time.perf_counter() - t0)

        wid = np.arange(N, N + BATCH, dtype=np.uint64)
        t0 = time.perf_counter()
        ix.add_with_ids(warm, wid)
        t_append.append(time.perf_counter() - t0)

        for i in range(SINGLES):
            sid = np.array([N + BATCH + i], dtype=np.uint64)
            t0 = time.perf_counter()
            ix.add_with_ids(singles[i:i + 1], sid)
            t_single.append(time.perf_counter() - t0)

        # --- IdMapIndex.remove: first call builds the slot map ---
        t0 = time.perf_counter()
        for i in range(REMOVES):
            ix.remove(int(i))
        t_idrm.append(time.perf_counter() - t0)
        del ix

        # --- TurboQuantIndex.swap_remove on an equally sized index ---
        tq = TurboQuantIndex(dim=DIM, bit_width=BITS)
        tq.add(base)
        t0 = time.perf_counter()
        for _ in range(REMOVES):
            tq.swap_remove(0)
        t_swap.append(time.perf_counter() - t0)
        del tq

    r = {
        "bulk": median_ms(t_bulk),
        "append": median_ms(t_append),
        "single": median_ms(t_single),
        "swap": median_ms(t_swap),
        "idremove": median_ms(t_idrm),
    }
    cells = {f"{op}-{cell_arch}": ms for op, ms in r.items()}
    text = json.dumps(cells, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
