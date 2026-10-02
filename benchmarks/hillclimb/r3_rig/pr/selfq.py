import os, sys, numpy as np
from turbovec import TurboQuantIndex
on = os.environ.get("TURBOVEC_2BIT_PLANES") == "1"
def run(name, db, mk):
    ix = TurboQuantIndex(dim=db.shape[1], bit_width=2); ix.add(db)
    for noise in (0.0, 0.1, 0.3):
        rng = np.random.default_rng(1)
        q = db[:2000] + noise * mk(rng, 2000, db.shape[1]); q = np.ascontiguousarray(q, dtype=np.float32)
        s, i = ix.search(q, k=10); i = np.array(i)
        one = np.array([np.array(ix.search(q[j:j+1], k=10)[1]).ravel() for j in range(300)])
        np.save(f"/tmp/self_{name}_{noise}_{int(on)}.npy", i); np.save(f"/tmp/self1_{name}_{noise}_{int(on)}.npy", one)
        msg = f"planes={int(on)} {name} noise={noise} self-is-top1 batched={np.mean(i[:,0]==np.arange(2000)):.4f} single={np.mean(one[:,0]==np.arange(300)):.4f}"
        if on:
            e = np.load(f"/tmp/self_{name}_{noise}_0.npy"); e1 = np.load(f"/tmp/self1_{name}_{noise}_0.npy")
            msg += f" | ids-identical-to-exact batched={np.mean((i==e).all(1)):.4f} single={np.mean((one==e1).all(1)):.4f} top1-same={np.mean(i[:,0]==e[:,0]):.4f}"
        print(msg, flush=True)
unit = lambda rng, n, d: (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(rng.standard_normal((n, d), dtype=np.float32))
v = np.load(os.path.expanduser("~/data/py-turboquant/openai-1536.npy"), mmap_mode="r")
db = np.ascontiguousarray(v[:100_000]).astype(np.float32); db /= np.linalg.norm(db, axis=1, keepdims=True)
run("openai1536", db, unit)
for d in (64, 768):
    run(f"random{d}", unit(np.random.default_rng(0), 50_000, d), unit)
