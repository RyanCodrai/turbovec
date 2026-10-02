"""Round-3 gate: planes-on results against the exact scan, same build, real embeddings.
usage: r3gate.py FILE NDB NQ  -> runs itself in subprocesses per mode (env is read once per process)."""
import os, sys, subprocess, json, numpy as np
FILE, NDB, NQ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
D = os.path.expanduser("~/data/py-turboquant")
if len(sys.argv) > 4:                       # worker
    from turbovec import TurboQuantIndex
    out, calib = sys.argv[4], sys.argv[5] == "1"
    v = np.load(os.path.join(D, FILE), mmap_mode="r"); dim = v.shape[1]
    db = np.ascontiguousarray(v[:NDB]).astype(np.float32); q = np.ascontiguousarray(v[-NQ:]).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True); q /= np.linalg.norm(q, axis=1, keepdims=True)
    ix = TurboQuantIndex(dim, bit_width=2)
    if calib: ix.calibrate(np.ascontiguousarray(db[:: NDB // 1024][:1024]))
    ix.add(db)
    res = {}
    for k in (1, 10, 100):
        s, i = ix.search(q, k=k); res[f"s{k}"] = np.array(s); res[f"i{k}"] = np.array(i)
    # one query at a time (the nq=1 paths), first 500 queries, k=10
    one = [ix.search(q[j:j + 1], k=10) for j in range(int(os.environ.get("NSINGLE", "500")))]
    res["s1q"] = np.array([np.array(o[0]).ravel() for o in one]); res["i1q"] = np.array([np.array(o[1]).ravel() for o in one])
    np.savez(out, **res); sys.exit(0)
tmp = os.path.expanduser("~/hc/gate_tmp"); os.makedirs(tmp, exist_ok=True)
for calib in (os.environ.get("GATE_CALIB", "0,1").split(",")):
    runs = {}
    P={"TURBOVEC_2BIT_PLANES": "1"}
    MODES={"exact": {}, "p128": P, "p10": {**P, "TURBOVEC_PLANES_MULT": "10", "TURBOVEC_PLANES_MIN": "1"},
           "p192": {**P, "TURBOVEC_PLANES_MULT": "192", "TURBOVEC_PLANES_MIN": "192"},
           "p256": {**P, "TURBOVEC_PLANES_MULT": "256", "TURBOVEC_PLANES_MIN": "256"},
           "t_k": {**P, "TURBOVEC_PLANES_T_MULT": "10", "TURBOVEC_PLANES_T_MIN": "1"},
           "t16": {**P, "TURBOVEC_PLANES_T_MULT": "15", "TURBOVEC_PLANES_T_MIN": "16"},
           "t24": {**P, "TURBOVEC_PLANES_T_MULT": "22", "TURBOVEC_PLANES_T_MIN": "24"},
           "t48": {**P, "TURBOVEC_PLANES_T_MULT": "45", "TURBOVEC_PLANES_T_MIN": "48"},
           "toff": {**P, "TURBOVEC_PLANES_T_MULT": "0", "TURBOVEC_PLANES_T_MIN": "0"}}
    names=os.environ.get("GATE_MODES", "exact,p10,p128,p192,p256").split(",")
    for name in names:
        env = MODES[name]
        out = f"{tmp}/{name}.npz"
        subprocess.run([sys.executable, __file__, FILE, str(NDB), str(NQ), out, calib], env={**os.environ, **env}, check=True)
        runs[name] = np.load(out)
    ex = runs["exact"]
    for name in names[1:]:
        r = runs[name]; parts = []
        for k in (1, 10, 100):
            same_ids = (r[f"i{k}"] == ex[f"i{k}"]).all(axis=1)
            same_set = np.array([set(a) == set(b) for a, b in zip(r[f"i{k}"], ex[f"i{k}"])])
            m = r[f"i{k}"] == ex[f"i{k}"]
            score_eq = (r[f"s{k}"][m].view(np.uint32) == ex[f"s{k}"][m].view(np.uint32)).mean()
            parts.append(f"k={k}: ids-identical {same_ids.mean():.4f} set-identical {same_set.mean():.4f} scores-bitwise {score_eq:.6f}")
        m1 = r["i1q"] == ex["i1q"]
        parts.append(f"single-query k=10: ids-identical {(m1.all(axis=1)).mean():.4f} scores-bitwise {(r['s1q'][m1].view(np.uint32) == ex['s1q'][m1].view(np.uint32)).mean():.6f}")
        print(f"{FILE} N={NDB} nq={NQ} calib={calib} {name} | " + " | ".join(parts), flush=True)
print("GATE_DONE")
