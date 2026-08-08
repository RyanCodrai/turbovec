import os
os.environ["RAYON_NUM_THREADS"] = "1"
import time, json, numpy as np
import faiss
from turbovec import TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
DIM, BIT_WIDTH = 1536, 2
N_SINGLE, N_B100 = 1_000, 10

def load_openai(dim, seed=42):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    extra = all_vecs[idx[100_000:100_000 + N_SINGLE + N_B100 * 100]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    extra /= np.linalg.norm(extra, axis=-1, keepdims=True)
    return database, extra

database, extra = load_openai(DIM)
singles = np.ascontiguousarray(extra[:N_SINGLE])
batches = np.ascontiguousarray(extra[N_SINGLE:])
faiss.omp_set_num_threads(1)

# Online-update latency on a warm, populated 100K index (built untimed).
# Timed loops include the Python-call overhead a caller actually pays per
# add(). n=1 is the per-op latency of a single-vector add(); n=100 is a
# 100-vector batch, showing how far batching amortizes the per-call
# overhead. The index grows by N_SINGLE + N_B100 * 100 per run (~1% of
# 100K per run, negligible drift).
tq = TurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
tq.add(database)
single_times, b100_times = [], []
for _ in range(5):
    t0 = time.perf_counter()
    for i in range(N_SINGLE):
        tq.add(singles[i:i + 1])
    single_times.append((time.perf_counter() - t0) / N_SINGLE * 1e6)
    t0 = time.perf_counter()
    for i in range(N_B100):
        tq.add(batches[i * 100:(i + 1) * 100])
    b100_times.append((time.perf_counter() - t0) / N_B100 * 1e6)
tq_single_us = sorted(single_times)[2]
tq_b100_us = sorted(b100_times)[2]

# FAISS PQ: train() is untimed (the one-time analogue of TQ's calibration
# fit), then the same timed loops run against the trained, populated
# index. Sub-quantizer count matches TurboQuant's bit rate:
# m = (bits * dim) / 4.
m_pq = DIM // 2
pq = faiss.IndexPQFastScan(DIM, m_pq, 4)
pq.train(database)
pq.add(database)
faiss_single_times, faiss_b100_times = [], []
for _ in range(5):
    t0 = time.perf_counter()
    for i in range(N_SINGLE):
        pq.add(singles[i:i + 1])
    faiss_single_times.append((time.perf_counter() - t0) / N_SINGLE * 1e6)
    t0 = time.perf_counter()
    for i in range(N_B100):
        pq.add(batches[i * 100:(i + 1) * 100])
    faiss_b100_times.append((time.perf_counter() - t0) / N_B100 * 1e6)
faiss_single_us = sorted(faiss_single_times)[2]
faiss_b100_us = sorted(faiss_b100_times)[2]

result = {"dim": DIM, "bit_width": BIT_WIDTH, "arch": "x86", "threading": "st",
          "tq_single_add_us": round(tq_single_us, 2),
          "tq_batch100_add_us": round(tq_b100_us, 2),
          "faiss_single_add_us": round(faiss_single_us, 2),
          "faiss_batch100_add_us": round(faiss_b100_us, 2)}
out = os.path.join(os.path.dirname(__file__), "..", "results", "speed_insert_d1536_2bit_x86_st.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
