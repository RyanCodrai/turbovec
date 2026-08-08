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

Every sample is recorded under `raw`, and any cell whose samples separate
into two clusters is reported under `modes` — see `modes()` for why a
summary statistic alone is not enough on this hardware.

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
        "print(min(ts)*1e3)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-2000:])
    return float(out.stdout.strip().splitlines()[-1])


def modes(samples, gap=0.03, min_side=2):
    """Split `samples` at their largest relative gap, if it looks like a mode.

    A cell that alternates between two operating points is not noisy in the
    way a spread implies — the samples cluster. P16 found x86 `nq100_st`
    landing at ~82 or ~98 ms and *locking* into one for a whole process, so a
    summary statistic silently reports whichever mode that run happened to
    draw. This finds the split rather than leaving it to whoever eyeballs the
    raw list, which is how it was found the first time.

    Returns `None` when the samples are one cluster, else the two clusters.
    The threshold is deliberately well above the ~1% run-to-run spread and
    well below the 18% band P16 measured.
    """
    xs = sorted(samples)
    if len(xs) < 2 * min_side:
        return None
    splits = [(xs[i + 1] / xs[i], i) for i in range(len(xs) - 1)]
    ratio, i = max(splits)
    if ratio - 1.0 < gap or i + 1 < min_side or len(xs) - i - 1 < min_side:
        return None
    return xs[:i + 1], xs[i + 1:]


def measure(bits, reps):
    path = ensure_index(bits)
    cells, raw = {}, {}
    for st in (False, True):
        tag = "st" if st else "mt"
        # Best of three sub-runs on every cell, not just nq=1.
        #
        # nq=1 always needed it: H51 saw a 0.955 ms reading against a 0.54 ms
        # median for the same build, itself the median of 55 reps — the whole
        # *process* ran slow, which extra reps cannot fix.
        #
        # At 2 bits nq=100 needs it too. x86 nq100_st is bimodal *within a
        # single process*, iterations landing at ~82 or ~98 ms, so the median
        # of 9 picks a mode by chance: three consecutive processes on one
        # unchanged build measured 83.1, 96.8, 84.1. An 18% band on an
        # objective cell makes every comparison noise. `min` selects the
        # unperturbed mode, which is the one a kernel change moves.
        raw[f"nq100_{tag}"] = [search_cell(path, 100, st, reps)
                               for _ in range(3)]
        cells[f"nq100_{tag}"] = min(raw[f"nq100_{tag}"])
        # Nine sub-runs on nq=1, not three. H6 ran a patch that touches only
        # x86-gated code and arm still read -8.6% on this cell — a control
        # channel showing the noise floor is ~8%, not the 2.5% the round
        # spread implied. The 4-bit climb reached the same place (H115) and
        # nine took its spread 7.8% -> 2.7%.
        raw[f"nq1_{tag}"] = [search_cell(path, 1, st, reps * 5)
                             for _ in range(9)]
        cells[f"nq1_{tag}"] = min(raw[f"nq1_{tag}"])
    return cells, raw


def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


if __name__ == "__main__":
    bits = arg("--bits", 2)
    cells, raw = measure(bits, arg("--reps", 15))
    # Every sample is kept, not just the summary. A capstone once read a 2.7%
    # regression on a cell that was flat, because one lucky baseline draw set
    # the minimum and the samples behind it had been discarded. Whatever
    # estimator the authority uses, the evidence for it survives the run.
    split = {c: {"lo": lo, "hi": hi}
             for c, ms in raw.items() if (m := modes(ms)) for lo, hi in [m]}
    blob = json.dumps({"bits": bits, "dim": DIM, "n": N, "k": K, "cells": cells,
                       "raw": raw, "modes": split})
    out = arg("--out", None, str)
    if out:
        with open(out, "w") as fh:
            fh.write(blob + "\n")
    print(blob)
