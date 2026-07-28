import os
import platform
import sys
import tempfile
import time, json, numpy as np
from turbovec import TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
DIM, BIT_WIDTH = 1536, 4
N = 100_000
ARCH = "arm" if platform.machine() in ("arm64", "aarch64") else "x86"


def load_openai(dim, seed=42):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:N]]
    queries = np.ascontiguousarray(all_vecs[idx[N : N + 1]])
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database, queries


database, queries = load_openai(DIM)

# Build and persist once (untimed) — the benchmark measures the consumer
# side: persisted file -> answers, i.e. a service (re)start.
tq = TurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
tq.add(database)
path = os.path.join(tempfile.mkdtemp(), f"coldload_d{DIM}_{BIT_WIDTH}bit.tv")
tq.write(path)
del tq

# Cold load: load() + the first search on the loaded index (which pays
# any lazily built caches). Median of 5; each rep loads a fresh index
# object. The file sits in the OS page cache after rep 1, matching the
# common restart-with-warm-page-cache case; rep 1 is reported separately
# as the truly-cold sample.
load_times, first_search_times = [], []
for rep in range(5):
    t0 = time.perf_counter()
    idx = TurboQuantIndex.load(path)
    t1 = time.perf_counter()
    idx.search(queries, k=64)
    t2 = time.perf_counter()
    load_times.append((t1 - t0) * 1000)
    first_search_times.append((t2 - t1) * 1000)
    del idx

totals = sorted(l + s for l, s in zip(load_times, first_search_times))
result = {
    "dim": DIM,
    "bit_width": BIT_WIDTH,
    "arch": ARCH,
    "n_vectors": N,
    "load_ms_median": round(sorted(load_times)[2], 2),
    "first_search_ms_median": round(sorted(first_search_times)[2], 2),
    "cold_total_ms_median": round(totals[2], 2),
    "cold_total_ms_spread": round(totals[-1] - totals[0], 2),
    "cold_total_ms_rep1": round(load_times[0] + first_search_times[0], 2),
}
out = os.path.join(
    os.path.dirname(__file__), "..", "results", f"speed_load_d{DIM}_{BIT_WIDTH}bit_{ARCH}.json"
)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
if "--keep" not in sys.argv:
    os.remove(path)
