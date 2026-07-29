#!/usr/bin/env python3
"""Six-op hill-climb benchmark: one JSON of `<op>-<arch>` -> median ms.

Ops: search, save, load, insert, delete, load_search.
- search: 100 queries, k=10, on a prepared index
- insert: add 1000 vectors then one search (repack cost stays visible)
- delete: remove 1000 ids then one search (same reason)
- save:   write() after a mutation (post-mutation save, the pain point)
- load:   IdMapIndex.load() of the seeded file
- load_search: fresh subprocess doing load + first search (cold-path gate)

Modes: default is the multicore pool; --st pins RAYON_NUM_THREADS=1 (single-core
cells, suffixed `<op>-<arch>_st`). Both modes matter — an MT win must not tax ST.

Usage: python bench_ops.py --arch {arm,x86} [--st] [--reps N] [--out FILE]
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
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cell_arch = args.arch + ("_st" if args.st else "")

    if args.child:
        child_load_search()
        return

    ensure_file()
    rng = np.random.default_rng(1)
    queries = rng.random((100, DIM), dtype=np.float32)
    one_q = queries[:1]
    batch = rng.random((1000, DIM), dtype=np.float32)
    out_dir = tempfile.TemporaryDirectory()  # auto-removed on exit — a 77 MB
    out_path = os.path.join(out_dir.name, "out.tvim")  # file per run leaks fast
    r = {}

    ix = IdMapIndex.load(PATH)
    ix.search(one_q, k=10)  # warm
    t = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        ix.search(queries, k=10)
        t.append(time.perf_counter() - t0)
    r["search"] = median_ms(t)

    t = []
    for rep in range(args.reps):
        ix = IdMapIndex.load(PATH)
        ids = np.arange(10_000_000 + rep * 1000, 10_001_000 + rep * 1000,
                        dtype=np.uint64)
        t0 = time.perf_counter()
        ix.add_with_ids(batch, ids)
        ix.search(one_q, k=10)
        t.append(time.perf_counter() - t0)
    r["insert"] = median_ms(t)

    t = []
    for rep in range(args.reps):
        ix = IdMapIndex.load(PATH)
        ids = np.arange(rep * 1000, (rep + 1) * 1000, dtype=np.uint64)
        t0 = time.perf_counter()
        for i in ids:
            ix.remove(int(i))
        ix.search(one_q, k=10)
        t.append(time.perf_counter() - t0)
    r["delete"] = median_ms(t)

    t = []
    for rep in range(args.reps):
        ix = IdMapIndex.load(PATH)
        ix.add_with_ids(one_q, np.array([20_000_000 + rep], dtype=np.uint64))
        time.sleep(0.15)  # drain device queue between fsyncs
        t0 = time.perf_counter()
        ix.write(out_path)
        t.append(time.perf_counter() - t0)
    r["save"] = median_ms(t)

    t = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        IdMapIndex.load(PATH)
        t.append(time.perf_counter() - t0)
    r["load"] = median_ms(t)

    t = []
    for _ in range(args.reps):
        cmd = [sys.executable, os.path.abspath(__file__), "--arch", args.arch,
               "--child"]
        if args.st:
            cmd.append("--st")
        p = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            env=os.environ.copy(),
        )
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
