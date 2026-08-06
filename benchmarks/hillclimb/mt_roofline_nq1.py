"""Is single-threaded search compute-bound or memory-bound?

The op-count model says the x86 kernel should run about twice as fast as it
measures. Either the model is wrong or something outside the issue ports is
binding. This sweeps N *downwards* from the benchmark point: at small N the
whole code array is L2/L1-resident, so if cost per (query x vector) drops
sharply as N shrinks, the 200k number is paying for memory rather than for
arithmetic — and further kernel work on ST is mistargeted.

Reports achieved code-array bandwidth alongside, since that is the quantity
a memory bound would cap.

codes = N * dim/2 bytes: 5k=1.9MB 25k=9.6MB 100k=38MB 200k=77MB 400k=154MB
"""
import os
import statistics
import time

pass

import numpy as np  # noqa: E402
from turbovec import IdMapIndex  # noqa: E402

DIM, BITS, NQ, K = 768, 4, 1, 10
print(f"{'N':>9} {'codes':>8} {'ms':>9} {'ns/(q*vec)':>11} {'GB/s':>7}")
for n in (5_000, 12_500, 25_000, 50_000, 100_000, 200_000, 400_000):
    rng = np.random.default_rng(0)
    idx = IdMapIndex(dim=DIM, bit_width=BITS)
    step = 100_000
    for s in range(0, n, step):
        m = min(step, n - s)
        idx.add_with_ids(rng.random((m, DIM), dtype=np.float32),
                         np.arange(s, s + m, dtype=np.uint64))
    q = rng.random((NQ, DIM), dtype=np.float32)
    idx.search(q, k=K)
    # Hold total scanned work roughly constant so every point gets a
    # comparable amount of time under the clock.
    reps = max(3, min(41, int(200_000 * 9 / n)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        idx.search(q, k=K)
        ts.append(time.perf_counter() - t0)
    ms = statistics.median(ts) * 1e3
    code_bytes = n * DIM / 2
    # The scan reads the whole code array once per batch of 8 queries.
    passes = 1
    gbs = code_bytes * passes / (ms * 1e-3) / 1e9
    print(f"{n:>9} {code_bytes/1e6:>7.0f}M {ms:>9.2f} "
          f"{ms*1e6/(NQ*n):>11.4f} {gbs:>7.1f}", flush=True)
    del idx
