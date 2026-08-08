# 2-bit search hill-climb — goal

Make 2-bit search faster. Score: harmonic mean of 8 per-cell speedups at
`bit_width=2` — `{arm, x86} x {ST, MT} x {nq=1, nq=100}`, k=10, N=200k,
dim=768, equal weights, against a baseline pinned at the climb HEAD.

A win is HM > x1.01 with no cell regressing. Gates: bitwise-identical scores,
ids and tie-break order; `cargo test -p turbovec` green; no point of the nq
sweep (1..16, 32, 64) or N sweep (1k, 8k, 32k, 200k) regressing >3%.

4-bit does not gate anything. Measure and record it on each win; never drop a
2-bit win for it.

Every hypothesis is logged with its measurements and verdict, win or not.
Done at 20 consecutive non-wins; a win resets the count.
