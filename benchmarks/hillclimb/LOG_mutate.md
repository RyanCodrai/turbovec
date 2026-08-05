# Live-index mutation hill-climb — results log

Objective and rules: `GOAL_mutate.md`. Bench: `bench_mutate.py` (N=200k, dim=768,
4-bit). Smoke = 5 reps both arches; soak = 15 reps.

Non-win streak: 0 (H4 is the most recent win on ARM)

**Confirmation status: all three wins are confirmed on both arches.** The x86 box
came up after ~1 h of retrying (see Rig) and was measured against its own pinned
baseline. Final objective across the eight MT cells: **WHM x1.6441**, with all 16
cells (MT and ST, both arches) improved and none flagged.

| target | arm | x86 | HM(arm,x86) | bar |
|---|---|---|---|---|
| bulk | x1.946 | x1.804 | **x1.872** | x1.01 |
| append | x1.871 | x2.544 | **x2.155** | x1.01 |
| single | x2.180 | x3.882 | **x2.792** | x1.01 |
| remove | x1.086 | x1.017 | **x1.051** | x1.01 |

ST cells: bulk x1.174 / x1.342, append x1.158 / x1.617, single x1.597 / x2.590,
remove x1.085 / x1.052 — so no win is an MT-only win, which was the hard rule.

Both sanity gates read ok, and they corroborate rather than merely pass: the
single-add MT/ST ratio goes 1.354 → 0.992 on arm and 1.554 → 1.037 on x86. Both
arches carried the same pool-install signature and both lost it to H3.

One honest caveat: `swap-x86` is x0.982, the only sub-op below parity. It is
inside the noise gate, no change touched it (it has never had a probe, a pool
handoff or a wrapper), and its arm twin is x1.014 — so this reads as noise, not a
regression. It is a sub-op of the `remove` cell, which scores x1.017 on x86.

The correctness oracle digest is `4b91af2b…` on **both** arches, baseline and
candidate alike — the encoded bytes are identical across architectures and
unchanged by every hypothesis in this log.

## Rig

`turbovec-bench-arm-mutate` (c4a-standard-8) and `turbovec-bench-mutate`
(c3-standard-8), both built from boot-disk images of the masters.

The x86 half was blocked for the first hour of the climb: the `C3_CPUS` quota for
us-central1 is 24 and all 24 were held by three other goals' rigs
(`turbovec-bench-persist`, `-search`, `-sync`). A self-limiting retry got a slot
on attempt 15, on-spec as a c3-standard-8 — so no rig substitution was needed and
every number here is from the specified pair. H1–H4 were developed and smoked on
ARM during that window; each was then measured on x86 against its own pinned
x86 baseline before anything was called confirmed.

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

### H4 — native batch validation in the interruptible add wrapper (target: bulk, append)

H1 said bulk was compute-bound, and the phase instrumentation agreed about the
*core*: of a 200k add, `add_with_ids_2d` accounted for 199 ms (inner encode
189.6, id validation 3.3, map inserts 5.6, slot extend 0.1). But the cell was
354 ms. The missing 155 ms was above the core entirely.

Two things came out of chasing it. First, a Python add is not one call: the
interruptibility wrapper (#216, `BATCH_CHUNK_SIZE = 4096`) slices it, so a 200k
add arrives as 49 calls of 4096 rows. Second, and the actual finding, the slicing
is not what costs — the whole-batch pre-validation that licenses the slicing is.
Measured directly by varying the knob: bulk 354.4 ms at the default vs **109.1 ms
with chunking off**, append 16.43 vs 7.17.

That pre-validation exists for a real reason — a batch the core would reject has
to fail atomically rather than commit its early slices — but all three of its
checks restate something the core already does, in the most expensive way
available from numpy:

| check | old form | cost |
|---|---|---|
| finite values | `np.all(np.abs(a) < 1e16)` | materializes an abs array and a bool array the size of the batch, ~1.5 GB of temporaries for this shape, and reads every coordinate even when the first is bad |
| no duplicate ids | `np.unique(ids).size == ids.shape[0]` | sorts the whole id array (visible as `unique_numeric` in the profiles) |
| no id already present | `any(int(h) in index for h in ids)` | a Python loop: one GIL round trip and one index lock **per id** |

Replaced by two native predicates with identical answers: `_all_finite`, which
calls the core's own `first_invalid_coord` (one parallel pass, no temporaries,
short-circuits), and `IdMapIndex::batch_addable`, which answers both id
conditions in one short-circuiting pass under one read lock. Atomicity,
fall-through and error paths unchanged — a failing condition still delegates the
whole batch to the raw kernel with the original arrays.

Using the core predicate instead of a numpy restatement also removes a
duplication that had to be kept in step by hand; the two can no longer disagree
about what "acceptable" means.

- Correctness: oracle digest `4b91af2b…` — identical to baseline. Core suite
  green; all 340 binding tests pass — and that suite's own runtime fell from
  148 s to 19 s, which is the win showing up somewhere entirely independent.
- Smoke (5 reps, ARM): bulk 184.10 vs 357.39 (x1.941), append 8.72 vs 16.42
  (x1.883).
- Soak (15 reps, ARM, cumulative): bulk **x1.946**, append **x1.871**, single
  x2.180, remove x1.086; ST cells x1.174 / x1.158 / x1.597 / x1.085. WHM (8 MT
  cells) **x1.5921**. No cell flagged, sanity gate ok.
- **Verdict: WIN on ARM, x86 pending.**
- Left on the table deliberately: bulk is still 184 ms against 109 ms with
  chunking disabled entirely. The rest of that gap is the per-slice snapshot
  copy and pool handoff, and closing it means either weakening Ctrl-C latency
  (a user-visible property) or holding the GIL across the encode (a concurrency
  regression, #289). Neither is a free win, so neither was taken.

### Note on where the remaining headroom is not

H1 refuted the bandwidth story for the *core* encode, and that verdict stands —
but it was answering the wrong question, because the core encode was never where
the bulk cell's time was going. Every win so far has come from the same place:
per-call overhead in the binding and its Python wrapper, not kernel arithmetic.
H2 and H3 found it at small batch sizes (a probe and a pool install per call);
H4 found the same shape at large ones (whole-batch validation done in numpy
temporaries and Python loops).

`swap_remove` is the control that never moved: 77 ns throughout, because it
never had a probe, a pool handoff, or a wrapper. The remaining honest targets
are the per-slice snapshot copy and the encode kernels themselves.

### H5 — snapshot the batch once by moving chunking into the kernel (target: bulk, append)

**Measured, confirmed on ARM, and deliberately NOT landed in the PR.** Kept on
branch `perf/mutate-h5`.

After H4 the bulk cell was 184 ms against a 109 ms ceiling (what the same add
costs with chunking disabled outright). The gap is copying: the wrapper
snapshots the whole batch in Python so every slice reads one coherent version
(#108), and then the kernel snapshots *each slice again*, because it releases
the GIL and another Python thread could write to the source. Two full copies —
1.2 GB for a 200k x 768 add. Measured directly: `np.array` of this batch is
37.1 ms on arm, and the per-slice copies total the same volume again.

Taking the one snapshot on the Rust side of the boundary, with the GIL still
held, gives the same coherence guarantee for one copy — the slices are then read
from memory Python cannot reach. Validation moves onto the snapshot, which is
strictly stronger: it is provably the same bytes the slices encode. The slicing
loop and its signal check move into the kernel, so a `KeyboardInterrupt` still
lands within one slice and still leaves earlier slices committed.

- Correctness: oracle digest `4b91af2b…` — identical to baseline.
- Soak (15 reps, ARM, cumulative): bulk **x3.006** (357.39 → 118.90 ms), append
  x2.066, single x2.274, remove x1.203; ST x1.302 / x1.266 / x1.663 / x1.151.
  WHM (8 MT cells) **x1.8736**, against x1.5921 for the landed set.
- **Blocker — why it is not in the PR.** Two tests fail:
  `test_add_with_ids_always_chunks` and
  `test_add_with_ids_cancel_commits_completed_slices`. Neither is a behaviour
  regression — they drive the raw kernel through `__wrapped__` and count
  per-slice calls, and H5 moves that loop inside the kernel, so the seam they
  observe no longer exists. The behaviour they protect (a cancel mid-batch
  commits the completed slices and no more) still holds.

  The reason that is not a licence to rewrite them: those `__wrapped__`-based
  tests are deliberately the *deterministic, cross-platform* coverage for
  chunking. The real-SIGINT tests beside them are skipped on Windows, and say so
  in their skip reason. Deleting the seam would leave the interruptibility
  feature with no coverage at all on Windows, and no deterministic coverage
  anywhere. Replacing it needs a test-only hook into the kernel's slice loop —
  a design decision about shipped API surface that belongs in its own change,
  not appended to a perf PR.

So: a x1.53 further improvement on the heaviest cell, sitting behind a test-design
question rather than a technical one.
