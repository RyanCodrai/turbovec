"""TQ / TQ+ halves of benchmarks/suite/recall_d{DIM}_4bit.py (same split, seed, metric), without the FAISS leg."""
import os, sys, json, numpy as np
from turbovec import TurboQuantIndex
DIM = int(sys.argv[1]); K = 64; KS = [1, 2, 4, 8, 16, 32, 64]; SEED = 42
v = np.load(os.path.expanduser(f"~/data/py-turboquant/openai-{DIM}.npy"))
idx = np.random.RandomState(SEED).permutation(len(v))
db = v[idx[:100_000]].astype(np.float32); q = v[idx[100_000:101_000]].astype(np.float32)
db /= np.linalg.norm(db, axis=-1, keepdims=True); q /= np.linalg.norm(q, axis=-1, keepdims=True)
top1 = np.argmax(q @ db.T, axis=1)
def rec(ix):
    i = np.array(ix.search(q, k=K)[1]); return {str(k): round(float(np.mean([top1[j] in i[j, :k] for j in range(len(top1))])), 4) for k in KS}
a = TurboQuantIndex(DIM, bit_width=4); a.add(db)
b = TurboQuantIndex(DIM, bit_width=4); b.calibrate(db[np.random.RandomState(SEED).choice(len(db), 1024, replace=False)]); b.add(db)
print(json.dumps({"dim": DIM, "planes": os.environ.get("TURBOVEC_4BIT_PLANES", "0"), "tq": rec(a), "tqplus": rec(b)}))
