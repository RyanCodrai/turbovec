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

## H2 — RESULT: WIN #1 (confirmed, commit 32909dbf)

rank 47 -> 13.3ms (3.5x); build 749 -> 149s (5x); np32 1,203 -> 1,312
QPS at recall +0.0002 (f32 reorder noise, as predicted); all-cells
0.9690, within gate. Reproduced: np8 4,245/4,256, np32 1,312/1,314.
Score: best-point HM 1.357 -> 1.434 (x1.057 > 1.01). Note the
score's best point is np8; anchor-point (np32) HM is 1.043 —
both recorded so the operating-point drift stays visible.
Streak: 0.

## H3 (pre-registered) — per-cell scan carries ~0.5ms of fixed cost per call

**Hypothesis.** GCP np32 scan = 351ms for 1.13e7 query-vector pairs;
the flat scan does 2.5e8 pairs in 402ms, so IVF's scan runs ~19x
worse per pair. That is ~0.47ms of overhead per cell call (707 calls,
~23-query audience each) — three orders above the ~0.5us the cell's
bytes justify. Candidates: per-call query validation + LUT prep,
result allocation/sort, and rayon task-per-cell scheduling nested
inside the kernel's own parallelism (4 physical cores, 707 outer
tasks, inner multi-query path may subdivide further).

**Prediction.** Timing one `cells[c].search` call in isolation will
show most of the 0.47ms is outside the scoring loop; eliminating the
top contributor (likely: one kernel invocation over a merged layout,
or clamping nested parallelism) recovers a large multiple of scan.

**Test after H2 lands** — one variable at a time.

## H3 — RESULT: WIN #2 (confirmed, commit d17fa5a6)

Per-query prep (rotation + LUT build, ~89us) was rebuilt per probed
cell; cells share one rotation/codebook so it is byte-identical —
hoisted to once per batch. np32 scan 350 -> 37ms. QPS: np32
1,312 -> 7,306 (5.6x), np128 359 -> 3,276 (9.1x), all-cells 67 -> 742
(11x). Recall bit-identical at every point; suite green. Reproduced:
np32 7,306/7,296, np8 10,462/10,254. Score: best-point HM
1.434 -> 1.718 (x1.198), best point now np32. Cumulative np32:
1,203 -> 7,296 QPS at identical recall (6.1x). Streak: 0.

## H4 (pre-registered) — audience build does a full sort where a partial select suffices

**Hypothesis.** Post-H3, audience build is a flat ~12ms at every
nprobe: each query fully sorts all 707 cell scores to take the top
nprobe. `select_nth_unstable` is O(nlist) against O(nlist log nlist),
and only the selected nprobe need ordering (probe order matters only
for the wave-less audience map, which is order-insensitive — only
membership matters).

**Prediction.** Audience ~12ms → ~4ms; np32 total 64 → ~56ms
(~1.14x); larger relative gain at np8/16 where audience is a bigger
share. Recall unchanged: same selected set, order-free consumption.

**Test after H3 reproduction.**

## H4 — RESULT: WIN #3 (confirmed, np32 8,548/8,576; np8 13,482/13,466)

audience 12.3 -> 1.3ms (predicted 4); recalls bit-identical; np32
7,306 -> 8,548 QPS; np8 10,462 -> 13,482. Best-point HM
1.718 -> 1.754. Build unchanged. Streak: 0.

## H5 (pre-registered) — rank runs at <20% of f32 peak; register-block it

**Hypothesis.** Rank is a 500x1536x707 f32 GEMM at 82 GFLOPS on a
~450 GFLOPS 4-core AVX-512 part. dot8 fixed the dependency chain but
still streams both operands per (query, cell) pair with no register
reuse. A 4-query x 4-cell microkernel (16 accumulators, each centroid
load reused 4x, each query load reused 4x) should approach 3x.

**Prediction.** rank 13.3 -> ~4-5ms; np32 total 54.5 -> ~46ms
(~1.18x on the anchor point, more at np8 where rank is 37% of the
call). Recall moves only by f32 reorder noise within the gates.

**Test after H4 reproduction.**

## H5 — RESULT: REFUTED (x0.4 — rank 13.3 -> 95ms)

Rank-1 updates into a memory-resident accumulator array serialize on
the L1 store-to-load forwarding chain: every d iteration re-reads and
re-writes acc[c], the memory analogue of the register chain H2
removed. 11 GFLOPS — worse than dot8. Recall bit-identical; reverted
to the H4 rank. Rule refined: breaking a dependency chain only counts
if the accumulators live in REGISTERS across the reduction. Streak: 1.

## H6 (pre-registered) — same transpose, register-tiled accumulators

**Hypothesis.** Loop order (c-tile, then d) with a 16-wide c-tile of
accumulators held in registers for the whole d reduction: per (query,
c-tile), acc never touches memory until the final store. The
transpose makes the per-d loads contiguous within the tile.

**Prediction.** rank -> 4-6ms (the H5 prediction, on the corrected
mechanism); np32 ~1.15x. Recall within f32 reorder noise.

## H7 (pre-registered) — margin-based spill: the first recall-side lever

**Hypothesis.** Every win so far is speed-side; the recall column is
untouched since baseline. SOAR/SPANN-style redundancy: at insert,
a vector whose top-2 centroid scores are within a margin is stored in
BOTH cells (residual per host cell). Boundary vectors are exactly the
ones cell pruning loses — measured cell recall is the binding ceiling
(0.9256 at np32 vs 0.9698 all-cells). Spilling ~25-35% of vectors
lifts recall-at-fixed-nprobe substantially while scan traffic grows
only by the spill fraction; the HM should gain more from the recall
numerator at low nprobe than it loses to the fatter cells.

**Design notes.** Merge must dedup by id (a vector's two host cells
can both be probed): keep max score. Ceiling gate unchanged —
all-cells probing sees each vector at least once; dedup keeps
results well-formed. Diagnostics to report: spill fraction, memory
ratio, per-nprobe recall shift.

**Prediction.** At margin ~0.9 (relative), np16 recall 0.882 -> ~0.93
and np32 0.926 -> ~0.95 at ~0.75x the QPS of the same nprobe —
net HM gain if recall lifts >4pp where speed ratio is already >5.

**Test after H6's verdict.**

## H6 — RESULT: NEUTRAL (x0.99; np32 8,249 vs 8,548)

Register-tiled rank measured ~16ms — no better than dot8's 13.3.
Probable cause: the workspace pins target-cpu=x86-64-v2 (128-bit
SSE), so a 64-float tile is the entire xmm register file and spills —
H5's failure mode reintroduced by ISA width. Rank at 13-16ms may also
sit near the LLC bandwidth for the 4.3MB transpose per query batch.
Reverted to the H4 rank. Learning: on this build target, wide
register tiles are not available; rank's remaining headroom needs
either target-feature widening (a build-config question, out of climb
scope) or a quantized coarse stage. Streak: 2.

## H7 — RESULT: NEUTRAL (best x1.0014 vs bar x1.01), streak 3

Spill works as designed — recall +3.6pp at np8, +2.8pp at np16,
ceiling restored to flat parity (0.9698) — but fatter cells cost
~13% QPS and the HM prices the trade almost exactly even: tau scan
{0.02: 1.747, 0.035: 1.753, 0.05: 1.757} vs bar 1.772. Learning: on
this metric spill is a recall<->speed *converter*, not a win; kept in
tree (default tau 0.05) because the recall column matters outside the
climb and the metric cost is ~zero.

## H8 (pre-registered) — cell granularity: nlist 707 -> 1414

**Hypothesis.** nlist=sqrt(N) was inherited, never tested. Finer
cells select candidates more precisely (higher cell recall per byte
scanned); the costs that used to forbid large nlist are gone (H2-H4:
rank 13ms, audience 1.4ms). Doubling nlist halves mean cell size, so
matched-traffic points double nprobe; recall at matched traffic
should rise while scan stays ~flat and rank doubles from a small
base.

**Prediction.** Best-point HM gains if recall-at-matched-traffic
rises >1pp; build assignment doubles (~150 -> 300s, diagnostic only).

**Test:** harness reads TV_IVF_NLIST; sweep nlist=1414 at nprobe
{16,32,64,128,256}.

## H8 — RESULT: REFUTED (best HM 1.708 vs confirmed 1.754), streak 4

Finer cells DO improve recall per byte (np32@1414 0.9236 vs np16@707
0.9100 at matched traffic, +1.4pp) but probing 2x as many half-size
cells doubles per-cell call constants: QPS at matched traffic falls
~28% and the HM loses. Build doubles (310s). Learning: per-cell fixed
cost is the binding constraint on granularity; until it falls,
nlist=sqrt(N) stands. Points directly at kernel-ranked cells (H9).

## H9 (pre-registered) — coarse ranking through the cells' own kernel

**Hypothesis.** The hoisted LUTs (H3) are index-agnostic: one shared
rotation and codebook, so they score ANY same-shape index — including
a 707-row TurboQuantIndex over the centroids themselves. Replace the
exact rank (13.3ms) + audience select (1.4ms) with one kernel pass
over the centroid index (top-nprobe per query directly), then exact
q.c re-offsets for the selected cells only (~0.7ms of dot8). Probe
selection becomes quantized (a boundary-order perturbation — spill
already covers boundaries); offsets stay exact, so scores remain
exact-decomposed.

**Prediction.** rank+audience 14.7 -> ~6ms; np16 ~10,000 -> ~12,000
QPS; recall shifts only at probe-selection boundaries (<0.3pp).
Best-point HM > 1.79. Ceiling gate: nprobe=nlist selects all cells,
results identical.
