"""k sweep on the official corpus (OpenAI d, 100k db, 1000 queries): ms/query, median of 5."""
import os, sys, time, numpy as np
from turbovec import TurboQuantIndex
DIM = int(sys.argv[1])
v = np.load(os.path.expanduser(f"~/data/py-turboquant/openai-{DIM}.npy"))
idx = np.random.RandomState(42).permutation(len(v))
db = v[idx[:100_000]].astype(np.float32); q = v[idx[100_000:101_000]].astype(np.float32)
db /= np.linalg.norm(db, axis=-1, keepdims=True); q /= np.linalg.norm(q, axis=-1, keepdims=True)
ix = TurboQuantIndex(dim=DIM, bit_width=2); ix.add(db); ix.search(q[:1], k=64)
out = []
for k in (1, 10, 20, 32, 64, 100):
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); ix.search(q, k=k); ts.append((time.perf_counter() - t0))
    one = []
    for _ in range(3):
        t0 = time.perf_counter()
        for j in range(200): ix.search(q[j:j+1], k=k)
        one.append((time.perf_counter() - t0) / 200 * 1000)
    out.append(f"k={k}: batch {sorted(ts)[2]:.3f} single {sorted(one)[1]:.3f}")
print(f"dim={DIM} planes={os.environ.get('TURBOVEC_2BIT_PLANES','0')} threads={os.environ.get('RAYON_NUM_THREADS','all')} | " + " | ".join(out))
