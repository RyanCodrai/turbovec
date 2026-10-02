"""4-bit round 2 cells: real embeddings, 16 cells per arch.

{1 thread, all threads} x {1,000-query batch, one query per call} x k in {10, 32, 64, 100},
ms per query on 100K OpenAI d=1536 (the official suite's split and seed). Each thread setting
runs in its own process (the pool size is read once). A cell is the minimum over REPS timed
passes; every sample is kept.

usage: cells_real.py [--bits 4] [--dim 1536] [--out FILE]      (driver)
"""
import json, os, subprocess, sys, time

KS = (10, 32, 64, 100)
REPS = int(os.environ.get("CELLS_REPS", "5"))
SINGLE_Q = 200


def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def worker(bits, dim):
    import numpy as np
    from turbovec import TurboQuantIndex
    v = np.load(os.path.expanduser(f"~/data/py-turboquant/openai-{dim}.npy"))
    idx = np.random.RandomState(42).permutation(len(v))
    db = v[idx[:100_000]].astype(np.float32)
    q = v[idx[100_000:101_000]].astype(np.float32)
    db /= np.linalg.norm(db, axis=-1, keepdims=True)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    ix = TurboQuantIndex(dim=dim, bit_width=bits)
    ix.add(db)
    ix.search(q[:1], k=64)
    ix.search(q, k=64)
    raw = {}
    for k in KS:
        b, s = [], []
        for _ in range(REPS):
            t0 = time.perf_counter(); ix.search(q, k=k); b.append((time.perf_counter() - t0))
            t0 = time.perf_counter()
            for j in range(SINGLE_Q):
                ix.search(q[j:j + 1], k=k)
            s.append((time.perf_counter() - t0) / SINGLE_Q * 1000)
        raw[f"batch_k{k}"], raw[f"single_k{k}"] = b, s
    print(json.dumps(raw))


if __name__ == "__main__":
    bits, dim = arg("--bits", 4), arg("--dim", 1536)
    if "--worker" in sys.argv:
        worker(bits, dim); sys.exit(0)
    cells, raw = {}, {}
    for tag, threads in (("st", "1"), ("mt", str(os.cpu_count()))):
        env = dict(os.environ, RAYON_NUM_THREADS=threads)
        out = subprocess.run([sys.executable, __file__, "--worker", "--bits", str(bits), "--dim", str(dim)],
                             capture_output=True, text=True, env=env)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-2000:])
        r = json.loads(out.stdout.strip().splitlines()[-1])
        for name, samples in r.items():
            raw[f"{name}_{tag}"] = samples
            cells[f"{name}_{tag}"] = min(samples)
    blob = json.dumps({"bits": bits, "dim": dim, "cells": cells, "raw": raw})
    out = arg("--out", None, str)
    if out:
        open(out, "w").write(blob + "\n")
    print(blob)
