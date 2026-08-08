"""Correctness oracle: bitwise-identical scores, ids and tie-break order.

The score is trivially hackable by trading accuracy, so parity is the gate
that makes it mean anything. Scores are hashed as raw float bits, not
compared with a tolerance, and the id sequence is hashed in order — `e7e507e`
reverted an otherwise-working change for transposing two ids.

Ties are the part an approximate check misses, so the fixture deliberately
contains duplicated rows: any reordering within a tie group changes the
digest.

Usage:
  parity_2bit.py --pin FILE          write the reference digests
  parity_2bit.py --check FILE        compare against them (exit 1 on drift)
"""
import hashlib
import json
import os
import sys

import numpy as np
from turbovec import IdMapIndex

DIM, N, DUPES, NQ, K = 768, 20_000, 2_000, 64, 64
CACHE = os.environ.get("TURBOVEC_HILLCLIMB_CACHE",
                       os.path.expanduser("~/.cache/turbovec-hillclimb"))


def fixture(bits):
    """Seeded index whose last `DUPES` rows repeat the first ones, so the
    top-k contains exact ties and their order is part of the digest."""
    path = os.path.join(CACHE, f"parity_{N}_{bits}bit.tvim")
    if os.path.exists(path):
        return IdMapIndex.load(path)
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = rng.random((N, DIM), dtype=np.float32)
    rows[-DUPES:] = rows[:DUPES]
    idx = IdMapIndex(dim=DIM, bit_width=bits)
    idx.add_with_ids(rows, np.arange(N, dtype=np.uint64))
    idx.write(path)
    return idx


def digest(bits):
    idx = fixture(bits)
    q = np.random.default_rng(7).random((NQ, DIM), dtype=np.float32)
    scores, ids = idx.search(q, k=K)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(scores, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(ids, dtype=np.uint64).tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    digests = {f"{b}bit": digest(b) for b in (2, 4)}
    if "--pin" in sys.argv:
        with open(sys.argv[sys.argv.index("--pin") + 1], "w") as fh:
            json.dump(digests, fh, indent=2)
        print(json.dumps(digests, indent=2))
    elif "--check" in sys.argv:
        with open(sys.argv[sys.argv.index("--check") + 1]) as fh:
            pinned = json.load(fh)
        bad = {k: (pinned.get(k), v) for k, v in digests.items() if pinned.get(k) != v}
        if bad:
            print("PARITY BROKEN:", json.dumps(bad, indent=2))
            sys.exit(1)
        print("parity ok:", json.dumps(digests))
    else:
        print(json.dumps(digests, indent=2))
