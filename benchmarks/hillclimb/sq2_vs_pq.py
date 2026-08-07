"""H83: turbovec SQ2 latency at equal bytes to FAISS PQ4fs.

H82 established the two tie on recall at 384 B/vec (0.8940 vs 0.8995) and
H80 measured turbovec's scan 1.4x more efficient per byte. That implies SQ2
should win on time at matched recall — an inference, and the same shape of
claim H80 and H81 each got wrong. Measured here instead.
"""
import os, statistics, time
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, faiss
from turbovec import IdMapIndex

N, K, NQ, REPS = 200_000, 10, 1, 31
faiss.omp_set_num_threads(1)
X = np.load(os.path.expanduser("~/data/py-turboquant/openai-1536.npy"), mmap_mode="r")
X = np.array(X[: N + 1], dtype=np.float32, order="C")
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
q, X = X[N:], X[:N]
DIM = X.shape[1]

def bench(fn):
    fn(); ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3

print(f"OpenAI-{DIM}, N={N}, nq=1 ST")
for bits in (4, 2):
    ix = IdMapIndex(dim=DIM, bit_width=bits)
    ix.add_with_ids(X, np.arange(N, dtype=np.uint64))
    ix.search(q, k=K)
    print(f"  turbovec SQ{bits}   {DIM * bits // 8:5} B/vec  {bench(lambda: ix.search(q, k=K)):8.3f} ms")

for m in (DIM // 2, DIM // 4):
    fx = faiss.index_factory(DIM, f"PQ{m}x4fs", faiss.METRIC_INNER_PRODUCT)
    fx.train(X); fx.add(X)
    print(f"  faiss PQ{m}x4fs {m // 2:5} B/vec  {bench(lambda: fx.search(q, K)):8.3f} ms")
