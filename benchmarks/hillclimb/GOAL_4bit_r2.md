# 4-bit search hill-climb, round 2 — goal

Make 4-bit search faster. Score: harmonic mean of per-cell speedups at
`bit_width=4` on real embeddings (100K OpenAI d=1536) —
`{arm, x86} x {1 thread, all threads} x {1,000-query batch, one query per
call} x k in {10, 32, 64, 100}`, 32 cells, equal weights, against a
baseline pinned at main once PR #549 has merged.

A win is HM > x1.01 with every cell >= x0.99, `cargo test -p turbovec`
green, RAM per vector unchanged, and the file format unchanged.

Results stay exact, or pass the probabilistic gate: on OpenAI d=1536,
OpenAI d=3072 and mpnet d=768, at least 99.9% of queries return the same
ids as the exact scan at k = 1, 10 and 100, and returned scores are the
exact 4-bit scores.

All builds, tests, probes and measurements run on the GCP benchmark boxes,
never on Ryan's machine.

Each hypothesis gets a smoke (< 3 min); only a passing smoke gets one soak
(< 15 min) to confirm it.

Every hypothesis is logged with its measurements and verdict, win or not.
Done at 20 consecutive non-wins; a win resets the count.
