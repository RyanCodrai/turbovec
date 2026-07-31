import os, time, json, numpy as np
from turbovec import IdMapIndex, TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
DIM, BIT_WIDTH = 1536, 2
N_REMOVE = 50_000

def load_openai(dim, seed=42):
    all_vecs = np.load(os.path.join(DATA_DIR, f"openai-{dim}.npy"))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    return database

database = load_openai(DIM)
ids = np.arange(len(database), dtype=np.uint64)

# Deterministic removal order. remove() shrinks the index, so each run
# rebuilds a fresh index (built untimed) — looping the op on one index
# would not give comparable repeats. Timed loops include the Python-call
# overhead a caller actually pays per op.
rng = np.random.RandomState(7)
remove_ids = [int(x) for x in rng.permutation(len(database))[:N_REMOVE]]
# swap_remove targets slots. Calibration is fitted per block of rows, so
# the vector that fills the hole comes from the target's own block and a
# removal need not shrink the slot space — which means a slot below
# len() can be dead, and the live slots can run past len(). Neither
# `len(database) - i` nor `len(tq)` bounds them.
#
# Resolve the draws against a real index on an untimed dry run: draw a
# slot, walk forward to the next live one, remove it, record it. Every
# timed repeat starts from an identical fresh index and replays that
# exact list, so each recorded slot is live at the point it is used and
# the timed loop does the same work it always did — one swap_remove per
# iteration, nothing else. The probe walk consumes no randomness, so the
# RNG stream is what it was and `remove_ids` above is unchanged.
def _live_swap_slots():
    probe = TurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
    probe.add(database)
    slots = []
    for _ in range(N_REMOVE):
        capacity = probe.slot_capacity
        slot = int(rng.randint(0, capacity))
        while not probe.slot_is_live(slot):
            slot = (slot + 1) % capacity
        probe.swap_remove(slot)
        slots.append(slot)
    return slots

swap_slots = _live_swap_slots()

# IdMapIndex.remove(id): O(1) swap-and-pop on the underlying index plus
# the id-map bookkeeping (hash-map remove/insert + slot_to_id fix-up).
idmap_times = []
for _ in range(5):
    im = IdMapIndex(dim=DIM, bit_width=BIT_WIDTH)
    im.add_with_ids(database, ids)
    t0 = time.perf_counter()
    for rid in remove_ids:
        im.remove(rid)
    idmap_times.append(time.perf_counter() - t0)
idmap_t = sorted(idmap_times)[2]

# TurboQuantIndex.swap_remove(idx): the raw swap-and-pop (the last vector
# moves into the freed slot — order is not preserved). Baseline that
# isolates the cost of the id-map layer above.
swap_times = []
for _ in range(5):
    tq = TurboQuantIndex(dim=DIM, bit_width=BIT_WIDTH)
    tq.add(database)
    t0 = time.perf_counter()
    for slot in swap_slots:
        tq.swap_remove(slot)
    swap_times.append(time.perf_counter() - t0)
swap_t = sorted(swap_times)[2]

result = {"dim": DIM, "bit_width": BIT_WIDTH, "arch": "arm", "threading": "mt",
          "tq_idmap_remove_us_per_op": round(idmap_t / N_REMOVE * 1e6, 3),
          "tq_idmap_removes_per_sec": round(N_REMOVE / idmap_t),
          "tq_swap_remove_us_per_op": round(swap_t / N_REMOVE * 1e6, 3)}
out = os.path.join(os.path.dirname(__file__), "..", "results", "speed_remove_d1536_2bit_arm_mt.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
