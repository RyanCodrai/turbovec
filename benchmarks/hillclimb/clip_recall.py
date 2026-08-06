"""H50: does a per-vector CLIPPED uniform codebook match Lloyd-Max at 4 bits?

P17 closed affine codebooks at ~2 recall points, but compared against plain
global min/max uniform. Weaviate, Lucene OSQ and RaBitQ all pair uniform
codes with rotation plus a PER-VECTOR clipped interval, chosen per row —
Weaviate sweeps clip factors {0.6,0.7,0.8,0.9} and keeps the best on the
actual entries, with 1.0 always a candidate so it is never worse than
unclipped. Its stated rationale is exactly our problem: "with only 16 code
points, spending them on the full [min,max] range wastes resolution on a few
outlier entries."

If this matches Lloyd-Max, both TBLs leave every kernel on both arches: the
code becomes the value and feeds the MAC directly.

Simulates the quantizer only — rotation is modelled as the Gaussianizing
transform turbovec's block-Hadamard achieves.
"""
import numpy as np

N, DIM, K, NQ, BITS = 20_000, 768, 10, 200, 4
L = 1 << BITS
rng = np.random.default_rng(0)
X = rng.standard_normal((N, DIM)).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = rng.standard_normal((NQ, DIM)).astype(np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
truth = np.argpartition(-(Q @ X.T), K, axis=1)[:, :K]

def recall(Xh):
    got = np.argpartition(-(Q @ Xh.T), K, axis=1)[:, :K]
    return np.mean([len(set(a) & set(b)) for a, b in zip(got, truth)]) / K

# Lloyd-Max on the pooled distribution, one shared codebook (what we ship).
s = np.sort(rng.standard_normal(200_000))
lev = np.quantile(s, (np.arange(L) + 0.5) / L).astype(np.float32)
for _ in range(50):
    edges = (lev[1:] + lev[:-1]) / 2
    idx = np.searchsorted(edges, s)
    for j in range(L):
        m = idx == j
        if m.any():
            lev[j] = s[m].mean()
sd = X.std(axis=1, keepdims=True)
codes = np.clip(np.searchsorted((lev[1:] + lev[:-1]) / 2, X / sd), 0, L - 1)
print(f"Lloyd-Max (shipped)        recall@{K} = {recall(lev[codes] * sd):.4f}")

# Uniform, per-vector clipped interval. Weaviate's grid plus 1.0.
best = None
for f in (0.6, 0.7, 0.8, 0.9, 1.0):
    lo, hi = X.min(1, keepdims=True) * f, X.max(1, keepdims=True) * f
    step = (hi - lo) / (L - 1)
    c = np.clip(np.round((X - lo) / step), 0, L - 1)
    r = recall(lo + c * step)
    print(f"  uniform clip={f:<4}          recall@{K} = {r:.4f}")
    best = r if best is None else max(best, r)

# Per-vector best-of-grid, chosen on reconstruction error like Weaviate does.
los, his = X.min(1, keepdims=True), X.max(1, keepdims=True)
err_best, out = None, None
for f in (0.6, 0.7, 0.8, 0.9, 1.0):
    lo, hi = los * f, his * f
    step = (hi - lo) / (L - 1)
    c = np.clip(np.round((X - lo) / step), 0, L - 1)
    rec = lo + c * step
    e = ((rec - X) ** 2).sum(1, keepdims=True)
    if err_best is None:
        err_best, out = e, rec
    else:
        m = (e < err_best).ravel()
        out[m], err_best[m] = rec[m], e[m]
print(f"uniform, per-vector best   recall@{K} = {recall(out):.4f}")
