"""The eight objective cells: {arm, x86} x {ST, MT} x {nq=1, nq=100}.

This box measures four of them; the pair covers both arches. Every cell is
the search kernel and nothing else — no load, no process start. Both query
widths run against the same loaded index in the same process, so a kernel
change shows at full amplitude rather than being diluted by I/O.

nq=1 is not a scaled-down nq=100: the batched kernels amortize the nibble
unpack across the batch, and at nq=1 there is nothing to amortize over. A
result at one width says nothing about the other (see H42 in LOG_search.md).

Fork of `cells.py` with `--bits`, defaulting to 2. `--bits 4` produces the
4-bit observation run, which is recorded and never gated.

Usage:  cells_2bit.py [--bits {2,4}] [--reps N] [--out FILE]
"""
import json
import os
import statistics
import subprocess
import sys

DIM, N, K = 768, 200_000, 10
CACHE = os.environ.get("TURBOVEC_HILLCLIMB_CACHE",
                       os.path.expanduser("~/.cache/turbovec-hillclimb"))


def index_path(bits, n=N):
    return os.path.join(CACHE, f"cells_{n}_{bits}bit.tvim")


def ensure_index(bits, n=N):
    """Seeded build, so every box and every candidate scores the same data."""
    path = index_path(bits, n)
    if os.path.exists(path):
        return path
    import numpy as np
    from turbovec import IdMapIndex
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.default_rng(0)
    idx = IdMapIndex(dim=DIM, bit_width=bits)
    step = min(100_000, n)
    for s in range(0, n, step):
        idx.add_with_ids(rng.random((step, DIM), dtype=np.float32),
                         np.arange(s, s + step, dtype=np.uint64))
    idx.write(path)
    return path


def search_cell(path, nq, st, reps, k=K):
    """One warm search cell: `nq` queries, median of `reps`, in its own process."""
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = "1" if st else str(os.cpu_count())
    code = (
        "import statistics,time;import numpy as np;from turbovec import IdMapIndex;"
        f"idx=IdMapIndex.load({path!r});"
        f"q=np.random.default_rng(7).random(({nq},{DIM}),dtype=np.float32);"
        f"idx.search(q,k={k});ts=[]\n"
        f"for _ in range({reps}):\n"
        f"    t0=time.perf_counter();idx.search(q,k={k});ts.append(time.perf_counter()-t0)\n"
        "print(statistics.median(ts)*1e3)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-2000:])
    return float(out.stdout.strip().splitlines()[-1])


def measure(bits, reps):
    path = ensure_index(bits)
    cells = {}
    for st in (False, True):
        tag = "st" if st else "mt"
        # nq=1 is ~100x cheaper per call, so it gets proportionally more reps
        # — but reps do not fix its real variance. H51 saw a 0.955 ms reading
        # against a 0.54 ms median for the same build, and that reading was
        # itself the median of 55 reps: the whole *process* ran slow. Take the
        # best of three sub-runs, which rejects a perturbed process the way
        # extra reps cannot.
        cells[f"nq100_{tag}"] = search_cell(path, 100, st, reps)
        cells[f"nq1_{tag}"] = min(search_cell(path, 1, st, reps * 5)
                                  for _ in range(3))
    return cells


def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


if __name__ == "__main__":
    bits = arg("--bits", 2)
    cells = measure(bits, arg("--reps", 15))
    blob = json.dumps({"bits": bits, "dim": DIM, "n": N, "k": K, "cells": cells})
    out = arg("--out", None, str)
    if out:
        with open(out, "w") as fh:
            fh.write(blob + "\n")
    print(blob)
