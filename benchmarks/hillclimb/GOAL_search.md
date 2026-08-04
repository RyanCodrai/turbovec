# Search hill-climb — goal

Maximize the WHM of 20 search cells — shapes `q1` (nq=1, k=10, weight 3), `q10`
(nq=10, weight 2), `mask` (nq=100, 50% mask, weight 2), `k100` (nq=100, k=100,
weight 2) and `q100` (nq=100, k=10, weight 1, anchor) — each measured on arm,
x86, arm_st and x86_st, at N=200k, dim=768, 4-bit. The last climb only ever
measured `q100` and proved the kernels are at roofline there, so treat them as
fixed and attack the shape space around them; its three search wins were all
schedule wins, H24 being 8% sitting in a tile-count accident specific to nq=100.
A win is the target shape's 4-cell HM > x1.01 with no target cell regressing,
every other cell within 3%, bitwise parity of scores, ids and tie-break order on
both arches (a hard gate — `e7e507e` reverted a working change for transposing
two ids), `cargo test -p turbovec` green, and no regression at (1536, 4-bit) or
(768, 2-bit). Claimed wins need an interleaved A/B, since x86 search is bimodal
at 67–117 ms. Stop at 20 consecutive non-wins, probe- and argument-refuted
included. H4/H5, H9, H12/H37, H13, H38, H39/H40 and H45 are already refuted but
only at `q100`, so re-opening one at a new shape is fine if you say which
measurement stopped covering it. In scope: the binding's `search` entry down
through prep, rotation, LUT, scheduling, the NEON and AVX2/AVX-512 kernels,
mask/allowlist and top-k. Out: format changes, recall trades, re-encodes —
correctness and determinism are never traded for speed.
