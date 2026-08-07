"""H80: flat-vs-flat at nq=1 ST. Does FAISS scan 200k faster than turbovec?

P38 concluded the x0.42 competitive gap must be structural, on the premise
that FAISS is scanning fewer vectors. That premise is untested. This pits
turbovec against FAISS's *exhaustive* 4-bit scanner on identical data — no
IVF, no coarse quantizer, both touching all N.

If FAISS flat wins by ~2x, P38 is wrong and there is a kernel deficit.
"""
import os, statistics, time
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, faiss
from turbovec import IdMapIndex

N, DIM, K, REPS = 200_000, 768, 10, 31
faiss.omp_set_num_threads(1)
rng = np.random.default_rng(0)
X = rng.random((N, DIM), dtype=np.float32)
q = np.random.default_rng(7).random((1, DIM), dtype=np.float32)

tv = IdMapIndex(dim=DIM, bit_width=4)
tv.add_with_ids(X, np.arange(N, dtype=np.uint64))
tv.search(q, k=K)

def bench(fn):
    fn(); ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3

print(f"turbovec  4-bit flat   {bench(lambda: tv.search(q, k=K)):8.3f} ms")

for name, factory in (("SQ4", "SQ4"), ("SQ8", "SQ8")):
    try:
        ix = faiss.index_factory(DIM, factory, faiss.METRIC_INNER_PRODUCT)
        ix.train(X); ix.add(X)
        print(f"faiss     {name} flat    {bench(lambda: ix.search(q, K)):8.3f} ms")
    except Exception as e:
        print(f"faiss     {name}: {type(e).__name__}: {e}")

ix = faiss.index_factory(DIM, "PQ384x4fs", faiss.METRIC_INNER_PRODUCT)
ix.train(X); ix.add(X)
print(f"faiss     PQ4 fastscan {bench(lambda: ix.search(q, K)):8.3f} ms")
