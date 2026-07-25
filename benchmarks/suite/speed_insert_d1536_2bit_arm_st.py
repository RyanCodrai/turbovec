import os
os.environ["RAYON_NUM_THREADS"] = "1"
import time, json, numpy as np
import faiss
from turbovec import TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
DIM, BIT_WIDTH = 1536, 2
N_APPEND, N_SINGLE = 10_000, 1_000

def load_openai(dim, seed=42):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    append = all_vecs[idx[100_000:100_000 + N_APPEND + N_SINGLE]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    append /= np.linalg.norm(append, axis=-1, keepdims=True)
    return database, append

database, append = load_openai(DIM)
batch = append[:N_APPEND]
singles = np.ascontiguousarray(append[N_APPEND:])
faiss.omp_set_num_threads(1)

# TurboQuant. add() is cumulative, so each run rebuilds a fresh index —
# looping the op on one index would not give comparable repeats.
# Bulk = one add() into an empty index: includes the one-time
# rotation/codebook init and the TQ+ calibration fit (the first add is
# >= 1000 vectors, so calibration is always fitted, never identity).
# Warm = a second add() on the same index: caches built, calibration
# frozen by the first add — isolates the steady-state encode path.
bulk_times, warm_times = [], []
for _ in range(5):
    tq = TurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
    t0 = time.perf_counter()
    tq.add(database)
    bulk_times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    tq.add(batch)
    warm_times.append(time.perf_counter() - t0)
tq_bulk = len(database) / sorted(bulk_times)[2]
tq_warm = len(batch) / sorted(warm_times)[2]

# Per-op latency of single-vector add() on the warm index. Includes the
# Python-call overhead a caller actually pays per add(); the index grows
# by N_SINGLE per run (<1% of 110K, negligible drift).
single_times = []
for _ in range(5):
    t0 = time.perf_counter()
    for i in range(N_SINGLE):
        tq.add(singles[i:i + 1])
    single_times.append((time.perf_counter() - t0) / N_SINGLE * 1e6)
tq_single_us = sorted(single_times)[2]

# FAISS PQ. train() is untimed (the one-time analogue of TQ's calibration
# fit); reset() drops stored codes but keeps the trained codebooks, so
# each run times add() into an empty trained index.
m_pq = DIM // 2
pq = faiss.IndexPQFastScan(DIM, m_pq, 4)
pq.train(database)
faiss_times = []
for _ in range(5):
    pq.reset()
    t0 = time.perf_counter()
    pq.add(database)
    faiss_times.append(time.perf_counter() - t0)
faiss_bulk = len(database) / sorted(faiss_times)[2]

result = {"dim": DIM, "bit_width": BIT_WIDTH, "arch": "arm", "threading": "st",
          "tq_bulk_insert_vecs_per_sec": round(tq_bulk),
          "tq_warm_insert_vecs_per_sec": round(tq_warm),
          "tq_single_add_us": round(tq_single_us, 2),
          "faiss_bulk_insert_vecs_per_sec": round(faiss_bulk)}
out = os.path.join(os.path.dirname(__file__), "..", "results", "speed_insert_d1536_2bit_arm_st.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
