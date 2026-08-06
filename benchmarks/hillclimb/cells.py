"""Measure this box's four cells: {ST, MT} x {nq=100, nq=1}, both warm.

Every cell is the search kernel and nothing else — no load, no process
start. Both query widths run against the same loaded index in the same
process, so a kernel change shows at full amplitude in all four rather than
being diluted by I/O.

nq=1 is not a scaled-down nq=100: the batched kernels amortize the nibble
unpack across the batch, and at nq=1 there is nothing to amortize over.
A result at one width says nothing about the other (see H42).

Usage:  cells.py [--reps N]
"""
import json
import os
import statistics
import subprocess
import sys
import time

DIM, BITS, N, K = 768, 4, 200_000, 10
IDX = os.path.expanduser("~/.cache/turbovec-hillclimb/cells_200k.tvim")


def ensure_index():
    if os.path.exists(IDX):
        return
    import numpy as np
    from turbovec import IdMapIndex
    os.makedirs(os.path.dirname(IDX), exist_ok=True)
    rng = np.random.default_rng(0)
    idx = IdMapIndex(dim=DIM, bit_width=BITS)
    for s in range(0, N, 100_000):
        idx.add_with_ids(rng.random((100_000, DIM), dtype=np.float32),
                         np.arange(s, s + 100_000, dtype=np.uint64))
    idx.write(IDX)


def search_cell(nq, st, reps):
    """One warm search cell: `nq` queries, k=10, median of `reps`."""
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = "1" if st else str(os.cpu_count())
    code = (
        "import os,statistics,time;import numpy as np;from turbovec import IdMapIndex;"
        f"idx=IdMapIndex.load({IDX!r});"
        f"q=np.random.default_rng(7).random(({nq},{DIM}),dtype=np.float32);"
        f"idx.search(q,k={K});ts=[]\n"
        f"for _ in range({reps}):\n"
        "    t0=time.perf_counter();idx.search(q,k=10);ts.append(time.perf_counter()-t0)\n"
        "print(statistics.median(ts)*1e3)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    return float(out.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    reps = 15
    if "--reps" in sys.argv:
        reps = int(sys.argv[sys.argv.index("--reps") + 1])
    ensure_index()
    cells = {}
    for st in (False, True):
        tag = "st" if st else "mt"
        # nq=1 is ~100x cheaper per call, so it gets proportionally more reps
        # — but reps do not fix its real variance. H51 saw a 0.955 ms reading
        # against a 0.54 ms median for the same build, and that reading was
        # itself the median of 55 reps: the whole *process* ran slow. nq=1 MT
        # is ~0.5 ms of work spread over 8 threads, so scheduling noise hits
        # the run, not the iteration. Take the best of three sub-runs, which
        # rejects a perturbed process the way extra reps cannot.
        cells[f"nq100_{tag}"] = search_cell(100, st, reps)
        cells[f"nq1_{tag}"] = min(search_cell(1, st, reps * 5) for _ in range(3))
    print(json.dumps(cells))
