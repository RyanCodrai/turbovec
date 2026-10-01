# 2-bit search hill-climb, round 3 — goal

Make 2-bit search faster. Score: harmonic mean of 8 per-cell speedups at
`bit_width=2` — `{arm, x86} x {ST, MT} x {nq=1, nq=100}`, k=10, N=200k,
dim=768, equal weights, against a baseline pinned at the round-2 HEAD.

A win is HM > x1.01 with every cell >= x0.99, `cargo test -p turbovec`
green, and RAM per vector unchanged.

Results stay exact, or pass the probabilistic gate: on real embeddings, at
least 99.9% of queries return the same ids as the exact scan, and returned
scores are the exact 2-bit scores.

All builds, tests, probes and measurements run on the GCP benchmark boxes,
never on Ryan's machine.

Each hypothesis gets a smoke (< 3 min); only a passing smoke gets one soak
(< 15 min) to confirm it.

Every hypothesis is logged with its measurements and verdict, win or not.
Done at 20 consecutive non-wins; a win resets the count.
