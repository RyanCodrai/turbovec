"""4-bit round 2 gate + k sweep in one: switch on vs off, same build, real embeddings.
usage: h1gate.py FILE NDB [worker OUT]   env BITS (default 4)"""
import os, sys, subprocess, time, numpy as np
FILE, NDB = sys.argv[1], int(sys.argv[2]); BITS = int(os.environ.get("BITS", "4"))
SW = "TURBOVEC_4BIT_PLANES" if BITS == 4 else "TURBOVEC_2BIT_PLANES"
if len(sys.argv) > 3:
    from turbovec import TurboQuantIndex
    v = np.load(os.path.expanduser("~/data/py-turboquant/" + FILE), mmap_mode="r")
    db = np.ascontiguousarray(v[:NDB]).astype(np.float32); q = np.ascontiguousarray(v[-10000:]).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True); q /= np.linalg.norm(q, axis=1, keepdims=True)
    res = {}
    for calib in (0, 1):
        ix = TurboQuantIndex(db.shape[1], bit_width=BITS)
        if calib: ix.calibrate(np.ascontiguousarray(db[:: NDB // 1024][:1024]))
        ix.add(db); ix.search(q[:1], k=10)
        for k in (1, 10, 100):
            s, i = ix.search(q, k=k); res[f"s{k}_{calib}"] = np.array(s); res[f"i{k}_{calib}"] = np.array(i)
        one = [ix.search(q[j:j + 1], k=10) for j in range(2000)]
        res[f"s1q_{calib}"] = np.array([np.array(o[0]).ravel() for o in one]); res[f"i1q_{calib}"] = np.array([np.array(o[1]).ravel() for o in one])
    np.savez(sys.argv[4], **res); sys.exit(0)
tmp = os.path.expanduser("~/hc/gate4_tmp"); os.makedirs(tmp, exist_ok=True)
runs = {}
for name, on in (("exact", "0"), ("planes", "1")):
    out = f"{tmp}/{name}.npz"
    subprocess.run([sys.executable, __file__, FILE, str(NDB), "worker", out], env={**os.environ, SW: on}, check=True)
    runs[name] = np.load(out)
ex, r = runs["exact"], runs["planes"]
for calib in (0, 1):
    parts = []
    for key in ("1", "10", "100", "1q"):
        ik, sk = (f"i{key}_{calib}", f"s{key}_{calib}")
        same = (r[ik] == ex[ik]).all(axis=1).mean(); m = r[ik] == ex[ik]
        bitw = (r[sk][m].view(np.uint32) == ex[sk][m].view(np.uint32)).mean()
        parts.append(f"{'single k=10' if key == '1q' else 'k=' + key}: ids {same:.4f} scores-bitwise {bitw:.6f}")
    print(f"{FILE} bits={BITS} N={NDB} calib={calib} | " + " | ".join(parts), flush=True)
