# Live-index mutation hill-climb — results log

Objective and rules: `GOAL_mutate.md`. Bench: `bench_mutate.py` (N=200k, dim=768,
4-bit). Smoke = 5 reps both arches; soak = 15 reps.

Non-win streak: 1

## Rig

`turbovec-bench-arm-mutate` (c4a-standard-8) is up. `turbovec-bench-mutate`
(c3-standard-8) could not be created yet: the `C3_CPUS` quota for us-central1 is
24 and all 24 are held by three other goals' rigs (`turbovec-bench-persist`,
`-search`, `-sync`). A self-limiting retry is running. Until it lands, hypotheses
are smoked on ARM only and no win can be confirmed — a win needs both arches.

## Baseline (ARM, 15 reps, core `c8d7ec02` + harness)

| cell | MT | ST |
|---|---|---|
| bulk | 357.39 ms | 1000.47 ms |
| append | 16.42 ms | 48.27 ms |
| single | 0.014264 ms | 0.010534 ms |
| swap (x10k) | 0.7736 ms | 0.7919 ms |
| idremove (x10k) | 4.0127 ms | 3.9575 ms |

Correctness oracle digest: `4b91af2b40558e8e9a5296460fb97af84aef681e15f796581a9fe717e7108095`.

Per-unit costs this implies, which is where the headroom is: a bulk row costs
~1.8 us, an append row ~1.6 us — so append is essentially pure encode and shares
the bulk bottleneck. A *single* add costs 14.3 us, ~8x the encode of the one row
it contains, so that cell is nearly all fixed overhead. A `swap_remove` is 77 ns
against `IdMapIndex.remove`'s 409 ns.

The single-add MT/ST ratio is x1.35 on baseline core and reproduced exactly
across two independent runs, so it is a real cost (an MT 1-row add pays a
`with_pool` install the ST sentinel pool folds away), not a contaminated grid.
The sanity gate was rewritten to test drift of that ratio rather than its
distance from 1.0.

## Hypotheses

### H1 — fuse the rotation into the quantize pass (target: bulk, append)

`encode` ran two parallel passes: `rotate_batch_into` wrote every rotated row
into an `n * dim` f32 buffer — 614 MB for a 200k x 768 add — and `quantize_batch`
then streamed it back. Rows are independent in both, so the rotation can move
inside the quantize loop and produce each row into a per-worker `dim`-length
buffer that stays in L1. Same per-row ops in the same order, so encoded bytes are
unchanged. Profile motivating it (perf, ARM, bulk): `quantize_batch` 24.5%,
`rotate_batch_into` 18.9% + `apply_scaled_into` 10.7%, `par_first_invalid_coord`
3.7%, `__pi_clear_page` 2.6% (the 614 MB buffer's page faults).

Implemented as a `RowSource` enum so the refit path — which has no float32
originals and hands over rows reconstructed from stored codes — keeps the staged
form, as does `fit_calibration`, which needs the whole rotated batch resident to
take per-coordinate order statistics. The add path then has no encode scratch at
all, so its retention/shrink bookkeeping and the six `add_2d`-driven tests that
pinned it were removed; `retain_scratch` and its unit test stay for the
calibration path.

- Correctness: `cargo test -p turbovec --release` fully green (115 lib + all
  integration binaries), golden-byte tests included. Oracle digest on ARM
  `4b91af2b…` — **identical to baseline**, so the bytes really are unchanged.
- Smoke (5 reps, ARM): bulk 359.23 vs 357.39 (x0.995), append 16.90 vs 16.42
  (x0.972), bulk_st 1002.00 vs 1000.47 (x0.998), append_st 49.17 vs 48.27
  (x0.982). single and both removes unchanged.
- **Verdict: NO WIN.** Not smoked on x86 and not soaked — there is nothing to
  confirm. The 1.2 GB of round-tripped memory traffic was not on the critical
  path: the batch is compute-bound in the rotation and quantize arithmetic
  (55% of the profile between them), and the streaming write/read the fusion
  removes was already being absorbed by prefetch. Eliminating it buys footprint
  (614 MB of RSS and its page-fault walk), not time.
- Refutes, for later hypotheses: bulk and append are **not** bandwidth-bound at
  this shape. A win on those cells has to remove arithmetic or improve SIMD
  efficiency inside `apply_scaled_into` / `fused_quantize_scale_pack`, not move
  data around.
