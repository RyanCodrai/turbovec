import os
os.environ["RAYON_NUM_THREADS"] = "1"
import atexit, shutil, time, json, os.path, tempfile
import numpy as np
from turbovec import IdMapIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
DIM, BIT_WIDTH = 1536, 2
N_APPEND = 32
N_REMOVE = 1_000
REPS = 5

def load_openai(dim, seed=42):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    extra = all_vecs[idx[100_000:100_000 + N_APPEND]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    extra /= np.linalg.norm(extra, axis=-1, keepdims=True)
    return database, extra

database, extra = load_openai(DIM)
ids = np.arange(len(database), dtype=np.uint64)

tmpdir = tempfile.mkdtemp(prefix="tv-sync-")
# The container runs to hundreds of MB and the official machines run every
# cell in a loop, so leaving them behind fills the root disk.
atexit.register(shutil.rmtree, tmpdir, True)
seed_path = os.path.join(tmpdir, "seed.tvim")
tv_path = os.path.join(tmpdir, "index.tvim")

def build():
    idx = IdMapIndex(dim=DIM, bit_width=BIT_WIDTH)
    idx.add_with_ids(database, ids)
    return idx

build().write(seed_path)

def fresh():
    """A file already in v7 sync form, index bound to it, no pending ops."""
    shutil.copyfile(seed_path, tv_path)
    ix = IdMapIndex.load(tv_path)
    ix.sync(tv_path)
    return ix

def median(xs):
    return sorted(xs)[len(xs) // 2]

# ── First sync: the full write ──────────────────────────────────────────
# `sync` converts a v6 file into the sync container by rewriting it whole,
# so this cell costs what `write` costs. It is the reference the two
# incremental cells below are worth reading against, and the gate that
# stops an incremental win from being bought by making this slower.
first = []
for _ in range(REPS):
    shutil.copyfile(seed_path, tv_path)
    ix = IdMapIndex.load(tv_path)
    time.sleep(0.15)             # drain the device queue between fsyncs
    t0 = time.perf_counter()
    ix.sync(tv_path)
    first.append(time.perf_counter() - t0)
tv_sync_first = median(first)
tv_file_bytes = os.path.getsize(tv_path)

# ── Append: 32 rows, one fresh block unit plus the commit header ────────
ix = fresh()
append = []
for rep in range(REPS):
    new_ids = np.arange(10_000_000 + rep * N_APPEND,
                        10_000_000 + (rep + 1) * N_APPEND, dtype=np.uint64)
    ix.add_with_ids(extra, new_ids)
    time.sleep(0.15)
    t0 = time.perf_counter()
    ix.sync(tv_path)
    append.append(time.perf_counter() - t0)
tv_sync_append = median(append)

# ── Removal: 1000 scattered removals ────────────────────────────────────
# A removal is committed as a redo op riding the header; a later sync
# materializes it into its block unit. Both halves are timed, because a
# change that only moves work from the first into the second has shifted
# cost rather than removed it.
#
# Ids are drawn from the ones still alive: re-drawing from the original
# range would hit already-removed ids, and those calls are no-ops that
# would quietly shrink the cell. The settle sync also returns the file to
# an empty pending set each rep — repeating removals on an unsettled file
# pushes carried ops past the header's 1024-op cap and falls back to a
# full rewrite, which is a different measurement entirely.
ix = fresh()
pick = np.random.default_rng(7)
alive = ids.copy()
remove, settle = [], []
for _ in range(REPS):
    take = pick.choice(len(alive), size=N_REMOVE, replace=False)
    for i in alive[take]:
        ix.remove(int(i))
    alive = np.delete(alive, take)
    time.sleep(0.15)
    t0 = time.perf_counter()
    ix.sync(tv_path)             # commits the removals in the header
    remove.append(time.perf_counter() - t0)
    time.sleep(0.15)
    t0 = time.perf_counter()
    ix.sync(tv_path)             # materializes them into their units
    settle.append(time.perf_counter() - t0)
tv_sync_remove = median(remove)
tv_sync_settle = median(settle)

# No FAISS comparator: FAISS has no incremental save. `write_index` always
# writes the whole index, which is what the sync_first cell measures and
# what the speed_persist_* cells already compare against.
result = {"dim": DIM, "bit_width": BIT_WIDTH, "arch": "arm",
          "threading": "st",
          "n_vectors": len(database),
          "n_append": N_APPEND, "n_remove": N_REMOVE,
          "tv_file_bytes": tv_file_bytes,
          "tq_sync_first_ms": round(tv_sync_first * 1e3, 2),
          "tq_sync_append_ms": round(tv_sync_append * 1e3, 2),
          "tq_sync_remove_ms": round(tv_sync_remove * 1e3, 2),
          "tq_sync_settle_ms": round(tv_sync_settle * 1e3, 2)}
out = os.path.join(os.path.dirname(__file__), "..", "results",
                   "speed_sync_d1536_2bit_arm_st.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
