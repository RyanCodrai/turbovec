import os, numpy as np
from turbovec import TurboQuantIndex
on = os.environ.get("TURBOVEC_2BIT_PLANES") == "1"
for dim in (256, 768):
    rng = np.random.default_rng(dim)
    n = 50_000
    db = rng.standard_normal((n, dim), dtype=np.float32); db /= np.linalg.norm(db, axis=1, keepdims=True)
    q = rng.standard_normal((500, dim), dtype=np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True)
    truth = np.argsort(-(q @ db.T), axis=1)[:, :10]
    ix = TurboQuantIndex(dim=dim, bit_width=2); ix.add(db)
    for k in (10, 100):
        s, i = ix.search(q, k=k)
        r1 = np.mean([t[0] in set(a) for t, a in zip(truth, i)])
        r10 = np.mean([len(set(t) & set(a[:10])) / 10 for t, a in zip(truth, i)])
        print(f"planes={int(on)} dim={dim} k={k} true-top1-in-k={r1:.3f} recall10@10={r10:.3f}")
