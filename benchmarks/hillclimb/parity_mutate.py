#!/usr/bin/env python3
"""Correctness oracle for the mutation hill-climb.

Drives the same mutation sequence the benchmark does, then emits a JSON of
- sha256 of `to_bytes()` after each mutation stage (the byte-level oracle), and
- the full top-k scores and ids of a fixed query set (search parity).

Run it on the baseline core, keep the output, and diff it against the candidate's
output. Any difference is a correctness failure — mutation speed is never traded
for it. Deliberately smaller than the benchmark so it runs in seconds.

Usage: python parity_mutate.py --out FILE
"""

import argparse
import hashlib
import json

import numpy as np
from turbovec import IdMapIndex, TurboQuantIndex

N, DIM, BITS = 20_000, 768, 4
BATCH, SINGLES, REMOVES = 1_000, 50, 500


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    base = rng.random((N, DIM), dtype=np.float32)
    warm = rng.random((BATCH, DIM), dtype=np.float32)
    singles = rng.random((SINGLES, DIM), dtype=np.float32)
    queries = rng.random((20, DIM), dtype=np.float32)
    out = {}

    ix = IdMapIndex(dim=DIM, bit_width=BITS)
    ix.add_with_ids(base, np.arange(N, dtype=np.uint64))
    out["after_bulk"] = sha(ix.to_bytes())
    out["len_after_bulk"] = len(ix)

    ix.add_with_ids(warm, np.arange(N, N + BATCH, dtype=np.uint64))
    out["after_append"] = sha(ix.to_bytes())

    for i in range(SINGLES):
        ix.add_with_ids(singles[i:i + 1],
                        np.array([N + BATCH + i], dtype=np.uint64))
    out["after_singles"] = sha(ix.to_bytes())
    out["len_after_singles"] = len(ix)

    # Search parity on the fully mutated index, before any removal.
    scores, ids = ix.search(queries, k=10)
    out["search_scores"] = [[float(x) for x in row] for row in scores]
    out["search_ids"] = [[int(x) for x in row] for row in ids]

    for i in range(REMOVES):
        ix.remove(int(i))
    out["after_idremove"] = sha(ix.to_bytes())
    out["len_after_idremove"] = len(ix)

    scores, ids = ix.search(queries, k=10)
    out["post_remove_scores"] = [[float(x) for x in row] for row in scores]
    out["post_remove_ids"] = [[int(x) for x in row] for row in ids]

    # `to_bytes()` must survive a round trip unchanged.
    out["roundtrip"] = sha(IdMapIndex.from_bytes(ix.to_bytes()).to_bytes())

    tq = TurboQuantIndex(dim=DIM, bit_width=BITS)
    tq.add(base)
    out["tq_after_bulk"] = sha(tq.to_bytes())
    for _ in range(REMOVES):
        tq.swap_remove(0)
    out["tq_after_swap"] = sha(tq.to_bytes())
    out["tq_len"] = len(tq)
    scores, ids = tq.search(queries, k=10)
    out["tq_scores"] = [[float(x) for x in row] for row in scores]
    out["tq_ids"] = [[int(x) for x in row] for row in ids]

    text = json.dumps(out, indent=2, sort_keys=True)
    print(sha(text.encode()))  # one-line digest for quick eyeballing
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
