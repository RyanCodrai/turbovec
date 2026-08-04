# Live-index mutation hill-climb — results log

Objective and rules: `GOAL_mutate.md`. Bench: `bench_mutate.py` (N=200k, dim=768,
4-bit). Smoke = 5 reps both arches; soak = 15 reps.

Non-win streak: 0 (H3 is the most recent win on ARM)

**Confirmation status:** H2 and H3 both pass smoke and soak on ARM with the
correctness oracle bit-identical, but neither is a *confirmed* win under this
goal's rules, which score the harmonic mean of a target's arm **and** x86 cells.
The x86 box does not exist yet (quota — see Rig). Both are staged, measured, and
waiting on that one number.

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
The sanity gate therefore measures how far the ratio sits from parity
*relative to the baseline's own offset*, so a candidate that closes the gap
reads as healthy rather than as a contaminated grid (H3 closes it entirely).

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

### H2 — latch the `slots_ready` probe in `IdMapIndex.remove` (target: remove)

Profiling the removal loop in isolation (`perf -D`, so the setup add is not
sampled) put `IdMapIndex::remove` at 21.4%, hashbrown `insert` at 11.0%, and —
the tell — `pthread_mutex_lock` at 3.3% plus `__aarch64_ldadd4_rel` at 4.5%.
That lock traffic is the binding, not the core: every `remove` ran
`py.detach(|| lock_read(&self.inner).slots_ready())` before taking the write
lock, so each removal paid a GIL release/reacquire and a read lock to ask
whether the id→slot map was built.

That question only ever answers false→true. `id_to_slot` is a `OnceLock` that is
only `get_or_init`'d, and the Python `IdMapIndex` is `frozen` with its `inner`
`RwLock` built once per object and never reassigned — so the answer is monotonic
per index and can be latched in an `AtomicBool`. The latch is only ever a
short-circuit to the same `true` the probe would have returned, so which path a
removal takes is unchanged. `Relaxed` suffices: a lost race re-probes, which is
what every call was already doing.

- Correctness: oracle digest `4b91af2b…` — identical to baseline.
- Smoke (5 reps, ARM): idremove 3.469 vs 4.013 (x1.157), swap flat.
- Soak (15 reps, ARM): remove-arm **x1.080**, remove-arm_st **x1.083**; bulk
  x1.014, append x1.008, single x0.994, all `_st` cells up. WHM x1.0273. No cell
  flagged.
- Correctness: `cargo test -p turbovec --release` green; 340 binding tests pass.
- **Verdict: WIN on ARM, x86 pending.** Above the x1.01 bar on the target with
  neither sub-op regressing.

### H3 — single-row bypass for `IdMapIndex.add_with_ids` (target: single)

Profiling the single-add loop the same way was decisive: `arch_local_irq_enable`
18.2%, `el0_svc_common` 15.7%, crossbeam epoch pin 11.0%, deque `steal` 8.0%,
`sched_yield` 4.6%, `wait_until_cold` 3.7% — the encode itself
(`quantize_batch` + `rotate_batch_into`) was ~6%. A 1-row add was spending its
time waking eight rayon workers to do one row of work.

`TurboQuantIndex::add` already had exactly the bypass for this (#321, #392): a
one-row encode's rayon bridges have length 1 and fold on the calling thread, so
the `install` buys nothing and costs the wakeup, and it is skipped unless the
row count or the input-validation scan would actually split. That gate was never
applied to `IdMapIndex::add_with_ids`, which is the method this cell — and any
incremental ingest through the id-mapped store — actually calls. Mirrored it,
`validation_parallelizes` term and all; the id-side work (presence checks, table
updates) is serial and allocates no rayon jobs, so it does not change the gate.

- Correctness: oracle digest `4b91af2b…` — identical to baseline.
- Smoke (5 reps, ARM): single 0.006357 vs 0.014264 (x2.244).
- Soak (15 reps, ARM, cumulative with H2): single-arm **x2.254**, single-arm_st
  **x1.658**, remove x1.083, bulk x0.998, append x1.004, every `_st` cell up.
  WHM (8 MT cells) **x1.1136**. No cell flagged.
- Correctness: `cargo test -p turbovec --release` green; 340 binding tests pass.
- The sanity gate corroborates the diagnosis rather than firing on it: the
  single-add MT/ST ratio goes 1.354 → 0.996. The gap *was* the pool install, and
  removing it put MT and ST on top of each other. The gate was rewritten to
  measure distance from parity, since a candidate moving the ratio toward 1.0 is
  the healthy direction, not a contaminated grid.
- **Verdict: WIN on ARM, x86 pending.**

### Note on where the remaining headroom is not

H1 refuted the bandwidth story for `bulk`/`append`, and those two cells have not
moved since (x0.998 / x1.004 cumulative). Both wins so far came from the same
place — per-call binding overhead that is invisible at batch sizes and dominant
at small ones — and `swap_remove`, which never had a probe or a pool handoff, is
flat at 77 ns throughout. The cells still at x1.0 are the ones whose cost is real
kernel arithmetic.
