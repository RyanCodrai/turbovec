import os
os.environ["RAYON_NUM_THREADS"] = "1"
import time, json, numpy as np
from turbovec import IdMapIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
DIM, BIT_WIDTH = 1536, 2
N_SINGLE = 1_000

def load_openai(dim, seed=42):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    return database

database = load_openai(DIM)
ids = np.arange(len(database), dtype=np.uint64)

# Deterministic removal order. remove() shrinks the index, so each run
# rebuilds a fresh index (built untimed) — looping the op on one index
# would not give comparable repeats. Timed loops include the Python-call
# overhead a caller actually pays per op.
rng = np.random.RandomState(7)
remove_ids = [int(x) for x in rng.permutation(len(database))[:100 + N_SINGLE]]

# IdMapIndex.remove(id): O(1) swap-and-pop on the underlying index plus
# the id-map bookkeeping. n=100 is the first 100 removes on the fresh
# index, timed as one block — the latency a caller sees. n=1 is the
# per-op latency over the next N_SINGLE removes.
first100_times, single_times = [], []
for _ in range(5):
    im = IdMapIndex(dim=DIM, bit_width=BIT_WIDTH)
    im.add_with_ids(database, ids)
    t0 = time.perf_counter()
    for rid in remove_ids[:100]:
        im.remove(rid)
    t1 = time.perf_counter()
    for rid in remove_ids[100:]:
        im.remove(rid)
    t2 = time.perf_counter()
    first100_times.append((t1 - t0) * 1e6)
    single_times.append((t2 - t1) / N_SINGLE * 1e6)
tq_remove_100_us = sorted(first100_times)[2]
tq_remove_1_us = sorted(single_times)[2]

result = {"dim": DIM, "bit_width": BIT_WIDTH, "arch": "arm", "threading": "st",
          "tq_remove_1_us": round(tq_remove_1_us, 3),
          "tq_remove_100_us": round(tq_remove_100_us, 1)}
out = os.path.join(os.path.dirname(__file__), "..", "results", "speed_remove_d1536_2bit_arm_st.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
