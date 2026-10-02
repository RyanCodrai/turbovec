import os, sys, numpy as np
from turbovec import TurboQuantIndex
on = os.environ.get("TURBOVEC_2BIT_PLANES") == "1"
out = {}
for dim in (64, 128, 256, 768, 1536):
    rng = np.random.default_rng(dim)
    n = 50_000
    db = rng.standard_normal((n, dim), dtype=np.float32); db /= np.linalg.norm(db, axis=1, keepdims=True)
    q = rng.standard_normal((500, dim), dtype=np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True)
    ix = TurboQuantIndex(dim=dim, bit_width=2); ix.add(db)
    s, i = ix.search(q, k=10)
    np.save(f"/tmp/rnd_{dim}_{int(on)}.npy", i)
    if on:
        e = np.load(f"/tmp/rnd_{dim}_0.npy")
        same = np.mean([(a == b).all() for a, b in zip(i, e)])
        ov = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(i, e)])
        print(f"dim={dim} identical_queries={same:.3f} overlap={ov:.4f}")
