"""Large-k exploration on real embeddings with the knobbed h118 build.
usage: kx.py FILE NDB  (driver)  |  kx.py FILE NDB OUT (worker; config from env)"""
import os, sys, subprocess, time, json, numpy as np
FILE, NDB = sys.argv[1], int(sys.argv[2]); KS = (10, 32, 64, 100)
if len(sys.argv) > 3:
    from turbovec import TurboQuantIndex
    v = np.load(os.path.expanduser("~/data/py-turboquant/" + FILE), mmap_mode="r")
    db = np.ascontiguousarray(v[:NDB]).astype(np.float32); q = np.ascontiguousarray(v[-10000:]).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True); q /= np.linalg.norm(q, axis=1, keepdims=True)
    ix = TurboQuantIndex(db.shape[1], bit_width=2); ix.add(db); ix.search(q[:1], k=10)
    res = {}
    for k in KS:
        res[f"i{k}"] = np.array(ix.search(q, k=k)[1])
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); ix.search(q[:1000], k=k); ts.append(time.perf_counter() - t0)
        one = []
        for _ in range(3):
            t0 = time.perf_counter()
            for j in range(100): ix.search(q[j:j + 1], k=k)
            one.append((time.perf_counter() - t0) * 10)
        res[f"t{k}"] = np.array([sorted(ts)[1], sorted(one)[1]])
    np.savez(sys.argv[3], **res); sys.exit(0)
tmp = os.path.expanduser("~/hc/kx_tmp"); os.makedirs(tmp, exist_ok=True)
def run(name, env):
    out = f"{tmp}/{name}.npz"
    subprocess.run([sys.executable, __file__, FILE, str(NDB), out], env={**os.environ, **env}, check=True)
    return np.load(out)
ex = run("exact", {"TURBOVEC_2BIT_PLANES": "0"})
print(f"{FILE} th={os.environ.get('RAYON_NUM_THREADS','all')} exact      | " + " | ".join(f"k={k} batch {ex[f't{k}'][0]:.3f} single {ex[f't{k}'][1]:.3f}" for k in KS), flush=True)
for mult, tmult in ((128, 30), (96, 30), (64, 30), (48, 30), (32, 30), (64, 20), (48, 20), (32, 20), (48, 15), (32, 15), (24, 15)):
    r = run(f"m{mult}_t{tmult}", {"TURBOVEC_2BIT_PLANES": "1", "TURBOVEC_PLANES_MULT": str(mult), "TURBOVEC_PLANES_T_MULT": str(tmult)})
    print(f"{FILE} th={os.environ.get('RAYON_NUM_THREADS','all')} S={mult/10:g}k T={tmult/10:g}k | " + " | ".join(
        f"k={k} same {(r[f'i{k}'] == ex[f'i{k}']).all(axis=1).mean():.4f} batch {r[f't{k}'][0]:.3f} single {r[f't{k}'][1]:.3f}" for k in KS), flush=True)
