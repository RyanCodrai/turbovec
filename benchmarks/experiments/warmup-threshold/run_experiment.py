"""Warm-up threshold experiment for issue #434.

Measures search quality (recall vs exact float32 search) as a function of
the number of samples the TQ+ calibration is fitted from, using the real
turbovec encode/search path.

Requires a wheel built from a branch with the experiment-only
`TURBOVEC_TQPLUS_MIN_SAMPLES` env override in `turbovec/src/encode.rs`
(this branch). The first `add()` contains exactly `fit_n` rows and
crosses the threshold, so the calibration is fitted from those rows; the
rest of the data is added under the frozen calibration — which is also
exactly the state a saved-and-reloaded index is in. `fit_n = 0` keeps the
index below the threshold forever, i.e. identity calibration.

Insertion orders:
  shuf{0,1}  i.i.d. random order (2 seeds)
  pc1        sorted by 1st principal component (worst-case biased stream)
  norm       sorted by raw L2 norm (mild realistic bias)

Usage:
  python run_experiment.py prep-glove            # needs ~/data/py-turboquant/glove-200-angular.hdf5
  python run_experiment.py prep-openai           # streams 101k rows from HF (Qdrant dbpedia openai3-1536)
  python run_experiment.py prep-synth            # cone-structured synthetic data
  python run_experiment.py run DB.npy Q.npy OUT.csv 2,3,4 0,100,1000,100000
  python run_experiment.py report OUT.csv
"""
import os
import sys
import time

import numpy as np

WORK = os.path.expanduser("~/data/py-turboquant/warmup-threshold")


def prep_glove():
    import h5py

    os.makedirs(WORK, exist_ok=True)
    path = os.path.expanduser("~/data/py-turboquant/glove-200-angular.hdf5")
    with h5py.File(path, "r") as f:
        np.save(f"{WORK}/glove_db.npy", np.array(f["train"][:100_000], dtype=np.float32))
        np.save(f"{WORK}/glove_q.npy", np.array(f["test"][:1_000], dtype=np.float32))


def prep_openai():
    from datasets import load_dataset

    os.makedirs(WORK, exist_ok=True)
    name = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
    col = "text-embedding-3-large-1536-embedding"
    ds = load_dataset(name, split="train", streaming=True)
    rows = []
    for r in ds:
        rows.append(np.asarray(r[col], dtype=np.float32))
        if len(rows) == 101_000:
            break
    arr = np.stack(rows)
    np.save(f"{WORK}/openai_db.npy", arr[:100_000])
    np.save(f"{WORK}/openai_q.npy", arr[100_000:])


def prep_synth():
    os.makedirs(WORK, exist_ok=True)
    rng = np.random.default_rng(42)
    n, d, r = 101_000, 256, 24
    mean = rng.normal(0, 1.0, d).astype(np.float32)  # dominant cone axis
    u = rng.normal(0, 1, (d, r)).astype(np.float32) / np.sqrt(r)
    z = rng.standard_t(4, (n, r)).astype(np.float32)  # heavy-tailed factors
    x = mean + 0.8 * (z @ u.T) + rng.normal(0, 0.15, (n, d)).astype(np.float32)
    np.save(f"{WORK}/synth_db.npy", x[:100_000])
    np.save(f"{WORK}/synth_q.npy", x[100_000:])


def run(db_path, q_path, out_path, bits_arg, fitns_arg):
    db = np.load(db_path).astype(np.float32)
    queries = np.load(q_path).astype(np.float32)
    n_db, dim = db.shape
    n_q = queries.shape[0]
    k_true, k_search = 10, 100

    dbn = db / np.linalg.norm(db, axis=1, keepdims=True)
    qn = np.ascontiguousarray(queries / np.linalg.norm(queries, axis=1, keepdims=True))
    true10 = np.empty((n_q, k_true), dtype=np.int64)
    for s in range(0, n_q, 250):
        sc = qn[s:s + 250] @ dbn.T
        part = np.argpartition(-sc, k_true, axis=1)[:, :k_true]
        row = np.arange(sc.shape[0])[:, None]
        true10[s:s + 250] = part[row, np.argsort(-sc[row, part], axis=1)]

    orders = {f"shuf{s}": np.random.default_rng(s).permutation(n_db) for s in (0, 1)}
    sample = db[np.random.default_rng(7).choice(n_db, 20_000, replace=False)]
    _, _, vt = np.linalg.svd(sample - sample.mean(axis=0), full_matrices=False)
    orders["pc1"] = np.argsort(db @ vt[0])
    orders["norm"] = np.argsort(np.linalg.norm(db, axis=1))

    import turbovec

    def one(bits, perm, fit_n):
        os.environ["TURBOVEC_TQPLUS_MIN_SAMPLES"] = str(fit_n or 1_000_000_000)
        ix = turbovec.TurboQuantIndex(dim, bits)
        data = np.ascontiguousarray(db[perm])
        if fit_n:
            ix.add(data[:fit_n])
            assert ix.calibration_state == "fitted"
            if fit_n < n_db:
                ix.add(data[fit_n:])
        else:
            ix.add(data)
            assert ix.calibration_state == "warming_up"
        _, ids = ix.search(qn, k_search)
        orig = perm[ids]
        r10 = np.mean(
            [len(np.intersect1d(orig[i, :k_true], true10[i])) for i in range(n_q)]
        ) / k_true
        r100 = np.mean(
            [len(np.intersect1d(orig[i], true10[i])) for i in range(n_q)]
        ) / k_true
        return r10, r100

    with open(out_path, "w") as out:
        out.write("bits,order,fit_n,recall10,recall10at100,secs\n")
        for bits in (int(b) for b in bits_arg.split(",")):
            for oname, perm in orders.items():
                for fit_n in (int(f) for f in fitns_arg.split(",")):
                    t0 = time.time()
                    r10, r100 = one(bits, perm, fit_n)
                    dt = time.time() - t0
                    out.write(f"{bits},{oname},{fit_n},{r10:.4f},{r100:.4f},{dt:.1f}\n")
                    out.flush()
                    print(f"bits={bits} order={oname:5s} fit_n={fit_n:6d} "
                          f"R@10={r10:.4f} R10@100={r100:.4f} ({dt:.1f}s)", flush=True)
    print("done")


def report(csv_path):
    import csv as csvmod
    from collections import defaultdict

    rows = list(csvmod.DictReader(open(csv_path)))
    data = defaultdict(dict)
    for r in rows:
        key = (int(r["bits"]), "shuf" if r["order"].startswith("shuf") else r["order"])
        data[key].setdefault(int(r["fit_n"]), []).append(
            (float(r["recall10"]), float(r["recall10at100"]))
        )
    fit_ns = sorted({int(r["fit_n"]) for r in rows})
    bits_all = sorted({int(r["bits"]) for r in rows})
    for metric, mi in (("R@10", 0), ("R10@100", 1)):
        print(f"\n=== {metric} ===")
        print("bits group  " + "".join(f"{fn:>8}" for fn in fit_ns))
        for bits in bits_all:
            for grp in ("shuf", "pc1", "norm"):
                vals = [
                    np.mean([x[mi] for x in data[(bits, grp)][fn]])
                    if fn in data.get((bits, grp), {}) else float("nan")
                    for fn in fit_ns
                ]
                print(f"{bits}    {grp:5s} " + "".join(f"{v:8.4f}" for v in vals))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep-glove":
        prep_glove()
    elif cmd == "prep-openai":
        prep_openai()
    elif cmd == "prep-synth":
        prep_synth()
    elif cmd == "run":
        run(*sys.argv[2:7])
    elif cmd == "report":
        report(sys.argv[2])
    else:
        raise SystemExit(f"unknown command {cmd}")
