"""H81: what does PQ's 2-dims-per-code cost in recall?

H80 showed FAISS wins nq=1 by storing 192 bytes/vector against turbovec's
384 - PQ384x4 packs two dims per 4-bit code where scalar packs one. The
speed follows from the bytes. The open question is the recall those bytes
buy, which this log has never measured for PQ (H50 refuted uniform/affine
*scalar*, a different construction).

Reports recall@10 against exact inner product for both, at their true sizes.
"""
import os, numpy as np, faiss
os.environ["RAYON_NUM_THREADS"] = "1"
from turbovec import IdMapIndex

N, DIM, K, NQ = 50_000, 768, 10, 200
faiss.omp_set_num_threads(1)
rng = np.random.default_rng(0)
X = rng.standard_normal((N, DIM)).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = rng.standard_normal((NQ, DIM)).astype(np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
truth = np.argpartition(-(Q @ X.T), K, axis=1)[:, :K]

def rec(ids):
    return np.mean([len(set(a) & set(b)) for a, b in zip(ids, truth)]) / K

tv = IdMapIndex(dim=DIM, bit_width=4)
tv.add_with_ids(X, np.arange(N, dtype=np.uint64))
_, ids = tv.search(Q, k=K)
print(f"turbovec SQ4   384 B/vec  recall@{K} = {rec(np.asarray(ids).reshape(NQ, K)):.4f}")

for name, dims_per_code in (("PQ384x4fs", 2), ("PQ192x4fs", 4)):
    m = DIM // dims_per_code
    ix = faiss.index_factory(DIM, f"PQ{m}x4fs", faiss.METRIC_INNER_PRODUCT)
    ix.train(X); ix.add(X)
    _, fids = ix.search(Q, K)
    print(f"faiss {f'PQ{m}x4fs':10} {m // 2:3} B/vec  recall@{K} = {rec(fids):.4f}"
          f"   ({dims_per_code} dims/code)")
