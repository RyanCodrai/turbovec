#!/usr/bin/env python3
"""Whole-file persistence hill-climb benchmark: `<op>-<arch>` -> median ms.

Ops (the four persistence cells):
- save_warm:   write() straight off a load, blocked cache warm and unmutated
- save_mut:    write() after one add — the post-mutation path
- load:        IdMapIndex.load() of the seeded file
- load_search: fresh subprocess doing load + first search (cost-shift gate)

Modes: default is the multicore pool; --st pins RAYON_NUM_THREADS=1 (cells
suffixed `<op>-<arch>_st`). The objective is the MT arm/x86 cells; the _st
cells are guard rails — a win there must not tax the single-core path.

Usage: python bench_persist.py --arch {arm,x86} [--st] [--reps N] [--out FILE]
       [--ops save_warm,load]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

if "--st" in sys.argv:  # must precede the extension's first pool build
    os.environ["RAYON_NUM_THREADS"] = "1"

import numpy as np
from turbovec import IdMapIndex

N, DIM, BITS = 200_000, 768, 4
CACHE = os.path.expanduser("~/.cache/turbovec-hillclimb")
PATH = os.path.join(CACHE, f"bench_{N}x{DIM}_{BITS}bit.tvim")
OPS = ["save_warm", "save_mut", "load", "load_search"]


def ensure_file():
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(PATH):
        return
    idx = IdMapIndex(dim=DIM, bit_width=BITS)
    rng = np.random.default_rng(0)
    idx.add_with_ids(rng.random((N, DIM), dtype=np.float32),
                     np.arange(N, dtype=np.uint64))
    idx.write(PATH)


def median_ms(xs):
    return statistics.median(xs) * 1e3


def child_load_search():
    q = np.random.default_rng(2).random((1, DIM), dtype=np.float32)
    t0 = time.perf_counter()
    ix = IdMapIndex.load(PATH)
    ix.search(q, k=10)
    print(time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["arm", "x86"])
    ap.add_argument("--st", action="store_true",
                    help="single-core mode (RAYON_NUM_THREADS=1)")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--out")
    ap.add_argument("--ops", help="comma-separated subset of " + ",".join(OPS))
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cell_arch = args.arch + ("_st" if args.st else "")
    wanted = set(args.ops.split(",")) if args.ops else set(OPS)

    if args.child:
        child_load_search()
        return

    ensure_file()
    rng = np.random.default_rng(1)
    one_q = rng.random((1, DIM), dtype=np.float32)
    out_dir = tempfile.TemporaryDirectory()  # auto-removed on exit — a 77 MB
    out_path = os.path.join(out_dir.name, "out.tvim")  # file per run leaks fast
    r = {}

    if "save_warm" in wanted:
        t = []
        for _ in range(args.reps):
            ix = IdMapIndex.load(PATH)
            ix.search(one_q, k=10)  # warm the blocked cache, no mutation
            time.sleep(0.15)  # drain the device queue between fsyncs
            t0 = time.perf_counter()
            ix.write(out_path)
            t.append(time.perf_counter() - t0)
        r["save_warm"] = median_ms(t)

    if "save_mut" in wanted:
        t = []
        for rep in range(args.reps):
            ix = IdMapIndex.load(PATH)
            ix.add_with_ids(one_q, np.array([20_000_000 + rep], dtype=np.uint64))
            time.sleep(0.15)
            t0 = time.perf_counter()
            ix.write(out_path)
            t.append(time.perf_counter() - t0)
        r["save_mut"] = median_ms(t)

    if "load" in wanted:
        t = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            IdMapIndex.load(PATH)
            t.append(time.perf_counter() - t0)
        r["load"] = median_ms(t)

    if "load_search" in wanted:
        t = []
        for _ in range(args.reps):
            cmd = [sys.executable, os.path.abspath(__file__), "--arch",
                   args.arch, "--child"]
            if args.st:
                cmd.append("--st")
            p = subprocess.run(cmd, capture_output=True, text=True, check=True,
                               env=os.environ.copy())
            t.append(float(p.stdout.strip()))
        r["load_search"] = median_ms(t)

    cells = {f"{op}-{cell_arch}": ms for op, ms in r.items()}
    text = json.dumps(cells, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
