# TQ-IVF hill-climb — recall × speed HM at 500k

Score: HM(recall/0.9254, QPS/3,333) vs the frozen baseline, best point of
the nprobe sweep {8,16,32,64,128}. Harness: `examples/ivf_vs_flat` on the
frozen files `base_500000.f32` / `queries.f32` / `gt_500000.i64`
(OpenAI-1536, 500 queries, exact truth). Batched MT on the M3 Max.
Win = reproduced HM > 1.01× best confirmed; suite green; all-cells
ceiling within 0.002 of 0.9698.

## Baseline — commit 3b673846 (2026-08-20)

| nprobe | recall@10 | batch QPS | single ms |
|---|---|---|---|
| 8   | 0.8328 | 5,493 | 1.07 |
| 16  | 0.8820 | 5,305 | 1.15 |
| 32  | 0.9254 | 3,333 | 1.29 |
| 64  | 0.9462 | 2,262 | 1.57 |
| 128 | 0.9600 | 1,229 | 2.09 |

Anchor HM = 1.0 at nprobe=32. Diagnostics: build 408 s (scalar
assignment), all-cells ceiling 0.9698. Reference points at the same
bit rate: flat TQ 0.9730 @ 1,777; FAISS IVFPQFastScan (M=1536)
0.9146 @ 9,143 at nprobe=32 — lower recall per probe than us at every
nprobe (−1.1 to −1.7pp), 2.7–4× our throughput at equal nprobe.

Streak: 0 non-improving since last win.

## H1 — per-cell call overhead, not kernel work, dominates batch search

**Hypothesis.** An ideal-work model says IVF at nprobe=32 scores ~4.5%
of the flat scan's query-vector pairs, so it should run near
0.045^-1 × the flat batch rate ≈ 40k QPS. Measured: 3,333. A ~12×
gap between work done and time taken cannot be kernel cost; it must
live in the per-cell plumbing — per-cell `search()` call overheads
(validation, query prep, result alloc, per-row sort) × 707 cells, the
audience gather copies, or nested rayon (our par-iter over cells
around a kernel that may itself parallelize).

**Prediction.** Phase timing will show scan-call time far exceeding
the kernel's byte budget, concentrated in per-call fixed costs; fixing
the top contributor is worth a large slice of the 12×.

**Test.** Instrument `IvfIndex::search` phases (cell ranking, audience
build, per-cell scan, merge) under an env flag; run the frozen sweep.

**Result (H1).** CONFIRMED as diagnosis: profile shows rank=39.3ms
(52%), scan=32.7ms, audience=3.8ms, merge=0.4ms of a 76ms nq=500
call at nprobe=8. Scan also ~10x its byte budget (~46us fixed cost
per probed cell).

## H1-fix — tile the rank matmul (8 queries per centroid stream)

**Hypothesis.** Rank is bandwidth-bound on re-streamed centroids
(2.2 GB for a 1.1 GFLOP product); tiling 8 queries per centroid row
cuts traffic 8x and most of the 39ms.

**Result: NEUTRAL.** rank unchanged (~40ms every sweep point);
recall bit-identical as designed; np32 QPS 3,488 vs 3,333 (+4.6%,
within scan noise). The traffic model is refuted — 8 query rows fit
L1, so re-streaming was not the cost. Rank's 40ms at 28 GFLOPS
effective points at the scalar inner loop itself (no FMA
vectorization across d). Not counted as a win. Streak: 1.

**Note.** Climb re-anchors on GCP (turbovec-bench, c3-standard-8,
Sapphire Rapids) from here — M3 numbers above are the local
prototype record, not the score baseline.

## H2 — rank is dependency-chain-bound: one serial f32 accumulator

**Hypothesis.** H1-fix refuted the traffic model, so the 40ms rank at
~28 effective GFLOPS must be the reduction itself: a single `s +=`
chain is non-associative f32, the compiler may not re-associate it,
so the loop runs at add-latency, not FMA throughput. Eight
independent accumulators (`dot8`) re-associate explicitly and let the
loop vectorize. Same chain sits in `nearest_centroid`, i.e. inside
the 409s build.

**Prediction.** Rank falls several-fold; build assignment falls with
it; recall moves only by f32 summation-order noise (different
rounding, same mathematical value) — ids essentially unchanged.

**Test.** GCP sweep on the frozen files, before/after commits at the
same box. (M3 numbers retired; GCP baseline table below.)

## GCP baseline — c40ef25 on turbovec-bench (c3-standard-8), 2026-08-20

| nprobe | recall@10 | batch QPS | single ms |
|---|---|---|---|
| 8   | 0.8328 | 3,317 | 1.17 |
| 16  | 0.8820 | 2,069 | 1.43 |
| 32  | 0.9254 | 1,203 | 1.92 |
| 64  | 0.9462 |   664 | 2.83 |
| 128 | 0.9600 |   353 | 4.62 |

Anchor HM = 1.0 at nprobe=32 (0.9254, 1,203 QPS). flat = 0.9730 @
1,242. Recall bit-identical to the M3 run at every point. Diagnostics:
build 749s; all-cells 0.9698 @ 67. Profile at np32: scan=351ms,
rank=47ms, audience=12ms — scan dominates on this box (4 physical
cores), rank is a constant 47ms tax.
