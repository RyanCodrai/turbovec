"""Apple Silicon GPU vs CPU speed at dim=1536, 4-bit.

Compares the MLX-backed turbovec index against the Rust CPU index on
the same machine. Methodology matches the existing ARM/x86 speed
scripts: 100K database vectors and 1K queries sampled from OpenAI
DBpedia d=1536, k=64, 5 trials, median per-query latency.

The CPU baseline is single-threaded (RAYON_NUM_THREADS=1) so the
GPU/CPU ratio reflects per-core uplift, not multi-core.
"""
import json
import os
import time

import numpy as np

os.environ["RAYON_NUM_THREADS"] = "1"

from turbovec import TurboQuantIndex
from turbovec.mlx import TurboQuantIndex as MlxTurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
DIM, BIT_WIDTH = 1536, 4
N_DB, N_QUERY, K = 100_000, 1_000, 64
N_TRIALS = 5


def load_openai(dim, seed=42):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:N_DB]]
    queries = all_vecs[idx[N_DB : N_DB + N_QUERY]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database.astype(np.float32), queries.astype(np.float32)


def median_ms_per_query(fn, queries):
    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        fn(queries)
        times.append((time.perf_counter() - t0) / len(queries) * 1000)
    return sorted(times)[N_TRIALS // 2]


def main():
    database, queries = load_openai(DIM)
    print(f"loaded db={database.shape} queries={queries.shape}")

    cpu = TurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
    cpu.add(database)
    cpu.prepare()
    cpu.search(queries[:1], k=K)  # warmup
    cpu_ms = median_ms_per_query(lambda q: cpu.search(q, k=K), queries)
    print(f"turbovec CPU (single-threaded): {cpu_ms:.3f} ms/query")

    gpu = MlxTurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
    gpu.add(database)
    gpu.prepare()
    gpu.search(queries[:1], k=K)  # warmup + JIT prime
    gpu_ms = median_ms_per_query(lambda q: gpu.search(q, k=K), queries)
    print(f"turbovec.mlx (Apple GPU):       {gpu_ms:.3f} ms/query")

    print(f"speedup: {cpu_ms / gpu_ms:.2f}x")

    result = {
        "dim": DIM,
        "bit_width": BIT_WIDTH,
        "arch": "apple_gpu",
        "n_db": N_DB,
        "n_query": N_QUERY,
        "k": K,
        "n_trials": N_TRIALS,
        "cpu_ms_per_query": round(cpu_ms, 3),
        "gpu_ms_per_query": round(gpu_ms, 3),
        "speedup": round(cpu_ms / gpu_ms, 2),
    }
    out = os.path.join(
        os.path.dirname(__file__), "..", "results", "speed_d1536_4bit_apple_gpu.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
