#!/usr/bin/env python3
"""sync() hill-climb benchmark: one JSON of `<cell>-<arch>` -> median ms.

Cells, per arch:
  sync_append  — sync after adding 32 rows (one fresh block unit + header)
  sync_remove  — sync after 1000 scattered removals (ops ride the header)
  sync_first   — the first sync of a fresh path (full write)   [gate]
  sync_settle  — the follow-up sync that materializes those ops [gate]

The first two are the objective. The last two are recorded gates: a "win"
that moves work out of an incremental sync and into the full write or into
the next sync has shifted cost, not removed it, and these cells catch it.

Steady state. A removal marks the hole slot and the moved-from slot dirty;
1000 scattered removals leave ~995 slots live below the block floor, just
under the 1024-op header cap. Repeating that on an unsettled file pushes
carried ops past the cap (last round's units collide with this round's) and
sync falls back to a full rewrite — so each timed removal sync is followed
by an untimed-for-that-cell settle sync, which is itself timed as the gate.
Every rep therefore starts from pending = {}, exactly like the first one.

Path check, not timing check: an incremental sync writes in place, a full
rewrite lands via temp + atomic rename and so changes the inode. Each cell
records whether it stayed incremental; a cell that silently fell back to a
full rewrite is not measuring what it claims to.

Usage: python bench_sync.py --arch {arm,x86} [--st] [--reps N] [--out FILE]
"""

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

if "--st" in sys.argv:  # must precede the extension's first pool build
    os.environ["RAYON_NUM_THREADS"] = "1"

import numpy as np
from turbovec import IdMapIndex

N, DIM, BITS = 200_000, 768, 4
APPEND_ROWS = 32
REMOVALS = 1_000
CACHE = os.path.expanduser("~/.cache/turbovec-syncclimb")
SEED = os.path.join(CACHE, f"seed_{N}x{DIM}_{BITS}bit.tvim")


def ensure_seed():
    """A v6-written file of N rows with ids 0..N. Reused across runs."""
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(SEED):
        return
    idx = IdMapIndex(dim=DIM, bit_width=BITS)
    rng = np.random.default_rng(0)
    idx.add_with_ids(rng.random((N, DIM), dtype=np.float32),
                     np.arange(N, dtype=np.uint64))
    idx.write(SEED)


def median_ms(xs):
    return statistics.median(xs) * 1e3


def ino(path):
    return os.stat(path).st_ino


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["arm", "x86"])
    ap.add_argument("--st", action="store_true",
                    help="single-core mode (RAYON_NUM_THREADS=1)")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--out")
    ap.add_argument("--cells", help="comma-separated subset "
                                    "(sync_append,sync_remove,sync_first)")
    args = ap.parse_args()
    cell_arch = args.arch + ("_st" if args.st else "")
    wanted = set(args.cells.split(",")) if args.cells else None
    want = lambda c: wanted is None or c in wanted  # noqa: E731

    ensure_seed()
    rng = np.random.default_rng(1)
    add_batch = rng.random((APPEND_ROWS, DIM), dtype=np.float32)
    r, incremental = {}, {}

    # A scratch dir per run; the container is ~77 MB and a leaked one per
    # run filled the root disk during the six-op climb.
    work = tempfile.TemporaryDirectory(prefix="tv-syncclimb-")
    path = os.path.join(work.name, "bench.tvim")

    def fresh_container():
        """A file already in v7 sync form, index bound to it, ops settled."""
        shutil.copyfile(SEED, path)
        ix = IdMapIndex.load(path)
        ix.sync(path)          # first sync: v6 file -> sync container
        return ix

    # ── sync_first (gate): the full write ────────────────────────────────
    # Timed from a v6 file, which is the state `sync` promises to convert.
    if want("sync_first"):
        t = []
        for _ in range(args.reps):
            shutil.copyfile(SEED, path)
            ix = IdMapIndex.load(path)
            time.sleep(0.15)   # drain the device queue between fsyncs
            t0 = time.perf_counter()
            ix.sync(path)
            t.append(time.perf_counter() - t0)
        r["sync_first"] = median_ms(t)

    # ── sync_append: 32 rows, one fresh block unit + the commit header ───
    if want("sync_append"):
        ix = fresh_container()
        before = ino(path)
        t = []
        for rep in range(args.reps):
            ids = np.arange(10_000_000 + rep * APPEND_ROWS,
                            10_000_000 + (rep + 1) * APPEND_ROWS,
                            dtype=np.uint64)
            ix.add_with_ids(add_batch, ids)
            time.sleep(0.15)
            t0 = time.perf_counter()
            ix.sync(path)
            t.append(time.perf_counter() - t0)
        r["sync_append"] = median_ms(t)
        incremental["sync_append"] = ino(path) == before

    # ── sync_remove + sync_settle ───────────────────────────────────────
    # Sampled from the ids still alive, so every rep really performs 1000
    # removals — re-drawing from 0..N would hit already-removed ids, and
    # those calls are no-ops that quietly shrink the cell. Removals are
    # scattered in slot space by construction: swap_remove relocates the
    # tail row into each hole, so slot order stops tracking id order after
    # the first rep. ~0.5% of holes land above the post-shrink block floor
    # and drop out of the plan, leaving ~995 header ops — deliberately just
    # under the 1024-op cap, which is the regime this cell exists to cover.
    if want("sync_remove"):
        ix = fresh_container()
        before = ino(path)
        pick = np.random.default_rng(7)
        alive = np.arange(N, dtype=np.uint64)
        t, t_settle = [], []
        for rep in range(args.reps):
            take = pick.choice(len(alive), size=REMOVALS, replace=False)
            ids = alive[take]
            alive = np.delete(alive, take)
            for i in ids:
                ix.remove(int(i))
            time.sleep(0.15)
            t0 = time.perf_counter()
            ix.sync(path)                    # commits the ops in the header
            t.append(time.perf_counter() - t0)

            time.sleep(0.15)
            t0 = time.perf_counter()
            ix.sync(path)                    # materializes them into units
            t_settle.append(time.perf_counter() - t0)
        r["sync_remove"] = median_ms(t)
        r["sync_settle"] = median_ms(t_settle)
        incremental["sync_remove"] = ino(path) == before

    cells = {f"{c}-{cell_arch}": ms for c, ms in r.items()}
    out = dict(cells)
    for c, ok in incremental.items():
        out[f"{c}-{cell_arch}#incremental"] = ok
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    # A cell that fell back to a full rewrite is not the cell it claims to
    # be — fail loudly rather than record a number nobody can interpret.
    bad = [c for c, ok in incremental.items() if not ok]
    if bad:
        print(f"FAIL: fell back to a full rewrite: {', '.join(bad)}",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
