import os
import atexit, shutil, time, json, os.path, tempfile
import numpy as np
import faiss
from turbovec import TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
DIM, BIT_WIDTH = 3072, 4
N_MUTATE = 1_000

def load_openai(dim, seed=42):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    extra = all_vecs[idx[100_000:100_000 + N_MUTATE]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    extra /= np.linalg.norm(extra, axis=-1, keepdims=True)
    return database, extra

database, extra = load_openai(DIM)
query = np.ascontiguousarray(database[:1])
# FAISS threading is left at its default (all cores), matching the
# existing speed_*_mt.py cells. Do NOT call omp_set_num_threads(0)
# to mean "default": 0 is invalid OpenMP and libgomp clamps it to a
# single thread, which would silently benchmark single-threaded
# FAISS against multi-threaded turbovec.

tmpdir = tempfile.mkdtemp(prefix="tv-persist-")
# Index payloads here run to hundreds of MB; the official machines run
# all 16 cells in a loop, so leaving them behind adds up.
atexit.register(shutil.rmtree, tmpdir, True)
tv_path = os.path.join(tmpdir, "index.tv")
faiss_path = os.path.join(tmpdir, "index.faiss")

def build():
    idx = TurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
    idx.add(database)
    return idx

# ── Save ────────────────────────────────────────────────────────────────
# Two states, because they cost very differently:
#   warm  — a search has run, so the blocked layout cache is populated and
#           write serializes from it (a cheap inverse transform);
#   dirty — a mutation invalidated that cache, so write pays the full
#           O(n*dim) repack from the packed rows first.
# turbovec's write is fsync + atomic rename; FAISS write_index is a raw
# dump with neither. The durability difference is deliberate (#68) and is
# part of the gap below, not an artifact to tune away.
warm_write, dirty_write = [], []
for _ in range(5):
    idx = build()
    idx.search(query, k=10)          # populate the blocked cache
    t0 = time.perf_counter()
    idx.write(tv_path)
    warm_write.append(time.perf_counter() - t0)

    idx.add(extra)                   # invalidates the blocked cache
    t0 = time.perf_counter()
    idx.write(tv_path)
    dirty_write.append(time.perf_counter() - t0)
tv_write_warm = sorted(warm_write)[2]
tv_write_dirty = sorted(dirty_write)[2]

# ── Load ────────────────────────────────────────────────────────────────
# Page cache is warm throughout (the file was just written, and each run
# re-reads it) — this measures the format's parse + prepare cost, not
# storage. Bare load vs load-then-first-search separates the two: since
# format v6 the file is the search-ready index, so the gap between them
# is what the v6 work removed.
build().write(tv_path)
bare_load, load_search = [], []
for _ in range(5):
    t0 = time.perf_counter()
    TurboQuantIndex.load(tv_path)
    bare_load.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    loaded = TurboQuantIndex.load(tv_path)
    loaded.search(query, k=10)
    load_search.append(time.perf_counter() - t0)
tv_load = sorted(bare_load)[2]
tv_load_search = sorted(load_search)[2]

# ── Round trip ──────────────────────────────────────────────────────────
# The checkpoint/resume cycle an embedding store actually pays: mutate,
# save, reopen, serve the first query.
roundtrip = []
for _ in range(5):
    idx = build()
    idx.search(query, k=10)
    t0 = time.perf_counter()
    idx.add(extra)
    idx.write(tv_path)
    reopened = TurboQuantIndex.load(tv_path)
    reopened.search(query, k=10)
    roundtrip.append(time.perf_counter() - t0)
tv_roundtrip = sorted(roundtrip)[2]

# ── FAISS comparator ────────────────────────────────────────────────────
# Precision-matched IndexPQFastScan, identical config to the search
# and insert cells. FastScan uses nbits=4, so m sub-quantizers is
# 4*m bits per vector; turbovec at 4 bits is 4*dim. Matching
# those gives m = dim. (Do not reuse the ratios from the
# recall cells - those use IndexPQ at nbits=8, so their m is half
# this. Getting it wrong halves the FAISS index and silently
# compares against something that stores half the information.)
m_pq = DIM
pq = faiss.IndexPQFastScan(DIM, m_pq, 4)
pq.train(database)
pq.add(database)
fw, fr, frs = [], [], []
for _ in range(5):
    t0 = time.perf_counter()
    faiss.write_index(pq, faiss_path)
    fw.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    faiss.read_index(faiss_path)
    fr.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    reloaded = faiss.read_index(faiss_path)
    reloaded.search(query, 10)
    frs.append(time.perf_counter() - t0)
faiss_write = sorted(fw)[2]
faiss_read = sorted(fr)[2]
faiss_read_search = sorted(frs)[2]

result = {"dim": DIM, "bit_width": BIT_WIDTH, "arch": "x86", "threading": "mt",
          "n_vectors": len(database),
          "tv_file_bytes": os.path.getsize(tv_path),
          "tq_write_warm_ms": round(tv_write_warm * 1e3, 2),
          "tq_write_after_mutation_ms": round(tv_write_dirty * 1e3, 2),
          "tq_load_ms": round(tv_load * 1e3, 2),
          "tq_load_first_search_ms": round(tv_load_search * 1e3, 2),
          "tq_mutate_save_load_search_ms": round(tv_roundtrip * 1e3, 2),
          "faiss_write_ms": round(faiss_write * 1e3, 2),
          "faiss_read_ms": round(faiss_read * 1e3, 2),
          "faiss_read_first_search_ms": round(faiss_read_search * 1e3, 2)}
out = os.path.join(os.path.dirname(__file__), "..", "results",
                   "speed_persist_d3072_4bit_x86_mt.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
