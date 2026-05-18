"""
Speed-test the block-level mask early-exit: 100K vectors, only the last 1K
slots allowed.

100K / 32 per block = 3125 blocks. The last 1K vectors occupy ~32 blocks
at the end. With block-skip active, ~3093 of 3125 blocks (~99%) should be
short-circuited. Without it (main), the masked search pays the full
unmasked SIMD cost.

Run this script twice — once on each wheel — to see the before/after.
"""

import os
import time

import numpy as np
from turbovec import TurboQuantIndex

DIM = 1536
N_DB = 100_000
N_ALLOWED = 1_000
N_QUERIES = 100
K = 10
SEED = 42
WARMUP = 3
REPEATS = 5


def main() -> None:
    rng = np.random.RandomState(SEED)
    database = rng.standard_normal((N_DB, DIM)).astype(np.float32)
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)

    # Allow only the last 1K slots.
    mask = np.zeros(N_DB, dtype=bool)
    mask[N_DB - N_ALLOWED:] = True

    index = TurboQuantIndex(DIM, bit_width=4)
    index.add(database)
    index.prepare()

    print(f"=== block-skip selectivity benchmark ===")
    print(f"  db={N_DB}, dim={DIM}, queries={N_QUERIES}, k={K}")
    print(f"  allowed slots: {N_ALLOWED} (last {N_ALLOWED}; "
          f"{N_ALLOWED / N_DB * 100:.1f}% of index)")
    print(f"  blocks total: {(N_DB + 31) // 32}, "
          f"blocks containing allowed slots: ~{(N_ALLOWED + 31) // 32}")
    print()

    for _ in range(WARMUP):
        index.search(queries, K)
        index.search(queries, K, mask=mask)

    unmasked_times = []
    masked_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        index.search(queries, K)
        unmasked_times.append((time.perf_counter() - t0) * 1000 / N_QUERIES)

        t0 = time.perf_counter()
        index.search(queries, K, mask=mask)
        masked_times.append((time.perf_counter() - t0) * 1000 / N_QUERIES)

    unmasked_ms = sorted(unmasked_times)[REPEATS // 2]
    masked_ms = sorted(masked_times)[REPEATS // 2]

    print(f"  unmasked search:  {unmasked_ms:.3f} ms / query (median of {REPEATS})")
    print(f"  masked search:    {masked_ms:.3f} ms / query (median of {REPEATS})")
    print(f"  speedup (unmasked / masked): {unmasked_ms / masked_ms:.2f}x")

    if masked_ms < unmasked_ms * 0.5:
        print("  -> block-skip appears active (>2x speedup at 1% selectivity)")
    elif masked_ms < unmasked_ms * 0.95:
        print("  -> some speedup but not large; block-skip may be partial or "
              "post-kernel scan is dominant")
    else:
        print("  -> no measurable speedup; block-skip likely not active "
              "(post-filter only)")


if __name__ == "__main__":
    main()
