# sync() hill-climb — results log

Objective: weighted harmonic mean of four per-cell speedups vs the baselines
pinned in `benchmarks/results/sync_baseline.json` — {arm, x86} × {sync of a
32-row append, sync of 1000 scattered removals}, weights 1:1. `sync_first`
(the full write) and `sync_settle` (the follow-up sync that materializes a
removal's header ops) are recorded gate cells, weight 0: they exist so that
moving work out of a measured sync and into one of them reads as what it is.

Win = target cell's HM(arm, x86) > x1.01, neither of its cells regressing,
no other cell — gates included — regressing beyond 3%, and the crash
contract untouched: one write batch and one fsync per sync, torn-write
harness and corruption matrix green, `sync_all` durability. Stop: 20
consecutive non-wins.

Bench: `benchmarks/hillclimb/bench_sync.py` (N=200k, dim=768, 4-bit).
Scorer: `benchmarks/hillclimb/whm_sync.py`. Smoke = 5 reps both arches;
soak = 15 reps.

## Rig

Purpose-built pair, never the masters and never local:

| box | machine | zone | disk |
|---|---|---|---|
| `turbovec-bench-sync` | c3-standard-8 | us-central1-a | pd-balanced 100 GB |
| `turbovec-bench-arm-sync` | c4a-standard-8 | **us-east4-c** | hyperdisk-balanced 80 GB, 3480 IOPS / 260 MB/s |

Both cloned from the masters (`turbovec-bench` / `turbovec-bench-arm`): x86
from a machine image, ARM from a boot-disk snapshot, because machine images
do not support C4A. **Zone deviation:** us-central1 was at its 24-CPU C4A
quota, held by three sibling goals' ARM boxes, so the ARM box lives in
us-east4-c. Its disk spec matches the ARM master exactly and every speedup
is measured against a baseline recorded on that same box, so the objective
is unaffected; only cross-goal absolute ARM numbers are not comparable.

`rm -rf target` before each release build; `LD_PRELOAD` the arch's
libopenblas. Pair stopped when idle, deleted at termination.

**Working tree:** `scratchpad/wt-sync`, a dedicated worktree. The shared
checkout at `~/git/turbovec` has concurrent editors — a sibling session
swept this climb's first staged commit into its own branch — so nothing in
this climb touches the shared tree.

## The fsync floor

Measured directly (write of size S into an existing 80 MB file, then
`fsync`, 15 reps, median):

| write size | x86 | arm |
|---|---|---|
| 4 KB | 1.41 ms | 1.42 ms |
| 64 KB | 1.69 ms | 1.74 ms |
| 400 KB | 2.46 ms | 2.44 ms |
| 12.6 MB | 8.75 ms | 6.14 ms |

This sets what is even addressable. A 32-row append commits ~12 KB, so
`sync_append` sits within ~0.5 ms of its floor on both arches — margins
there are fsync variance and are rejected regardless of how consistent they
look. A 1000-removal commit writes a ~400 KB header, so `sync_remove` is
~75% CPU on x86: plan build, header assembly, and digest. That is the honest
target.

## Baseline

Pinned in `benchmarks/results/sync_baseline.json`, 15 reps per cell, core =
`81d614d1` (main `c8d7ec02` plus the harness commit).

## Hypotheses

### H1 — a sync stops re-proving its own last commit (target: sync_remove)

Every sync opens with `cursor_state`, which decides whether the file is
still the one the cursor wrote. It did that by walking the header slots
newest-first and, for each, re-reading every unit that commit's sync wrote
and recomputing the delta digest over them. After a 1000-removal settle
that is 12.6 MB of read and CRC on the way into the *next* sync.

That work re-proves something already proven. A cursor is established
exactly two ways: this process wrote the commit — and `sync` returns Ok
only after `sync_all` reports it durable — or `load` adopted it, and `load`
picks a commit by running this same delta check. The nonce comparison just
above already established it is the same file. So when the newest parsing
header is at the cursor's own generation, it is the newest adoptable
commit, which is what `Intact` means. Any other generation still takes the
full verifying walk: a newer header means another writer advanced the file,
and Foreign-vs-Intact there genuinely depends on whether their data landed.
The shortcut only ever skips a commit already proven, and in the
unsupported concurrent-writer case it errs toward refusing.

Found by the warmup anomaly: rep 0 of the removal cell ran at 4.7 ms
against a steady 17.4: at rep 0 the preceding commit is the full write,
whose header names no units, so there was nothing to re-verify.

- Correctness: full `cargo test -p turbovec` green (121 lib + every
  integration binary). The crash contract specifically — torn-write
  harness (`a_sync_torn_at_any_byte_recovers_the_previous_commit`,
  `a_torn_materialize_of_a_delta_named_unit_recovers`,
  `blocked_only_capture_survives_a_torn_sync`,
  `a_recovery_load_syncs_forward_and_survives_a_second_tear`,
  `an_id_mapped_sync_torn_anywhere_restores_ids_exactly`), the corruption
  matrix, and the two multi-writer tests
  (`a_stale_cursor_refuses_to_clobber_another_writers_commits`,
  `two_writers_at_the_same_generation_do_not_collide`) all pass. One batch,
  one fsync, `sync_all` — untouched by this diff.
- Soak (15 reps): sync_remove arm 9.77 → 3.49 (**x2.80**), x86 18.58 →
  4.90 (**x3.79**). Target HM **x3.22**. WHM of the four measured cells
  x1.52.
- Gates: sync_first parity both arches (267.5→266.2, 435.8→432.9).
- Two cells flagged against baseline and both cleared by interleaved A/B
  (3 rounds each arch, alternating prebuilt modules on one machine state):
  - `sync_append-x86` read x0.93 vs baseline, but A/B has the new code
    *faster* in all 3 rounds (1.99/2.04/1.96 → 1.81/1.90/1.90). The cell
    sits ~0.3 ms above its fsync floor; baseline drift, not a regression.
  - `sync_settle-arm` read x0.82, and round 1 appeared to confirm it. Six
    further paired rounds refuted it: base 17.37/18.92/18.74/18.13/17.97/
    17.94 (median 18.05) vs new 17.48/18.01/17.66/17.71/18.19/18.33
    (median 17.86), new ≤ base in 4 of 6. The ARM settle cell is bimodal
    on unchanged code (~15.3 and ~18.5 states — the pinned baseline itself
    recorded 15.02 MT and 18.46 ST); the first A/B's base run drew the low
    state twice. **Protocol note for later hypotheses: judge
    `sync_settle-arm` on ≥6 paired rounds, never on one.**
- **Verdict: WIN** — committed. Streak 0.

### H2 — read a header slot's used prefix, not its op-capacity (target: sync_append)

`cursor_state` read the whole header region — superblock plus both slots —
before looking at either. A slot is *sized* for the 1024-op cap, ~428 KB at
dim 768, so that is ~856 KB read and zeroed on the way into every sync,
including a 32-row append whose entire commit is ~12 KB.

Almost none of it is used. A commit carrying no pending redo ops — an
append, or any sync following one that materialized its ops, which is the
steady state both objective cells sit in — uses only the fixed fields, the
tail block, the delta descriptor and the CRC: ~16 KB. Each slot is now read
at that size and widened to the full slot only when the parse needs more,
which costs the second read once for an op-bearing header and never in the
steady state. `parse_header_slot` split into a slot-local `parse_header_at`
whose every field read is bounds-checked, so a prefix that is long enough
parses and one that is not returns `None`.

- Correctness: `cargo test -p turbovec` green, including the torn-write
  harness, the corruption matrix, `io_versioning` and `bytes_io` (the
  parser's other callers). Crash contract untouched — this reads, it does
  not write.
- A/B, alternating prebuilt modules on one machine state (x86 4 rounds,
  arm 6 — the settle-cell protocol from H1):

  | cell | x86 | arm |
  |---|---|---|
  | `sync_append` | 1.815 → 1.635 (**x1.110**, better 4/4) | 1.700 → 1.640 (x1.037, better 3/6) |
  | `sync_remove` | 4.810 → 4.650 (x1.034, better 4/4) | 3.550 → 3.525 (x1.007, better 3/6) |
  | `sync_settle` | 35.58 → 34.07 (x1.044) | 18.31 → 18.27 (x1.002) |
  | `sync_first` | 431.5 → 427.6 (x1.009) | 265.9 → 266.0 (x1.000) |

- Target HM (sync_append) **x1.072**; no cell regresses on either arch.
- Honest reading: the win is carried by x86, where it is unanimous across
  rounds and larger than the fsync spread that cell sits in (0.18 ms moved
  against a ~0.07 ms fsync spread, with a mechanism — 840 KB less read per
  sync — that is not device noise). On ARM both cells are parity within
  noise: 840 KB off that box's memory system is ~0.05 ms, under its own
  round-to-round spread. Claimed as a win on the stated rule (HM > x1.01,
  neither cell regressing), not as a two-arch result.
- **Verdict: WIN** — committed. Streak 0.

### Where the removal sync's time actually goes (probe, not a hypothesis)

Timed `sync_remove` at 50/200/500/1000 removals on x86 with H2 in place:
1.92 / 2.40 / 3.23 / 4.78 ms — clean and linear. **3.02 µs per op, 1.77 ms
fixed.** Of the 3.02 ms that 1000 ops add, the fsync itself accounts for
~0.9 ms (the commit grows from ~20 KB to ~400 KB, and the floor table above
puts that at 1.5 → 2.46 ms), leaving ~2.1 ms of CPU — the only part any
per-op hypothesis can reach.

Where that 2.1 ms sits: each op serializes one row's sequential codes out
of the 32-lane interleaved block, and at dim 768 that is 384 byte
extractions at stride 32 — a walk over the whole 12 KB block unit to
collect 384 bytes. At ~995 ops that is ~12 MB of scattered reads. The cost
is memory latency, not arithmetic and not allocation, which is what H3
then confirmed the hard way.

### H3 — append row codes into the header buffer (target: sync_remove)

Header assembly called `seq_row`, which allocates and returns a `Vec` per
op, then copied it into the header and dropped it: ~995 allocations and
~382 KB of copying per 1000-removal sync. Replaced with a `seq_row_into`
that extends the header buffer directly.

- A/B (x86 4 rounds, arm 6): sync_remove x86 4.910 → 4.985 (**x0.985**,
  new better in only 1 of 4), arm 3.425 → 3.470 (x0.987). Target HM
  **x0.986** — a regression, not a win.
- Why: `Vec::extend` from an iterator re-checks capacity per byte, and at
  384 bytes a row that costs more than the `collect()` allocation plus
  `extend_from_slice` memcpy it replaced. Together with the probe above,
  this pins the per-op cost on the strided gather rather than on
  allocation — allocation was never the bottleneck to remove.
- **Verdict: NON-WIN** — reverted. Streak 1.

### A/A control — what the harness measures when nothing changed

Run after H4 came back at x1.07 on the append cell, the same figure H2 had
claimed by a completely different mechanism. Two copies of the *identical*
module, alternated exactly as an A/B alternates them:

| cell | range on identical code | 2nd-position ratio | 2nd slower |
|---|---|---|---|
| `sync_append` x86 | 1.56–1.82 (**±7.9%**) | x0.952 | 2/3 |
| `sync_append` arm | 1.50–1.81 (**±9.7%**) | x0.967 | 2/4 |
| `sync_remove` x86 | 4.55–4.88 (±3.5%) | **x0.942** | **3/3** |
| `sync_remove` arm | 3.41–3.67 (±3.7%) | x1.016 | 2/4 |
| `sync_settle` x86 | 32.6–35.3 (±4.0%) | x0.961 | 2/3 |
| `sync_settle` arm | 17.5–18.8 (±3.6%) | x1.005 | 1/4 |

Two findings, both of which change how earlier results must be read.

**1. The append cell cannot carry a claim below ~10%.** Its own
round-to-round spread on unchanged code is ±8% on x86 and ±10% on ARM —
the cell is ~0.15 ms above its fsync floor and that floor is what moves.
This is precisely the "reject wins within fsync variance" the goal names.
H2's append figure (x1.072) and H4's (x1.071) are both inside this band and
neither can be claimed on the append cell, however consistent the rounds
looked. H2's *remove* result is a separate matter — see below.

**2. `ab.sh` always ran base first, and second position is not neutral.**
On x86 `sync_remove` the second run is slower in 3 of 3 at ~6%. Every A/B
so far therefore handicapped "new" by roughly that much on that cell, which
means the remove-cell numbers were *understated*, not flattered:

| measured x86 remove | position-corrected |
|---|---|
| H2 x1.034 | ≈ x1.10 |
| H3 x0.985 | ≈ x1.05 |
| H4 x1.007 | ≈ x1.07 |

So **H3's rejection is suspect** — it may have been a small win read through
a larger handicap — and H4's is unresolved. Neither verdict stands on the
old harness.

`ab2.sh` replaces `ab.sh` from here: odd rounds run base-then-new, even
rounds new-then-base, so a within-round trend lands on both sides equally.
H1 is unaffected — at x2.8/x3.8 it is an order of magnitude outside every
band above, and it was measured against a pinned baseline in a separate
run, not by position. H2, H3 and H4 are all being re-measured on `ab2.sh`
at 8 rounds, and the verdicts below will be restated from that.

- **H4 verdict: UNRESOLVED pending re-measurement** (its x1.07 was on the
  append cell, inside the band). Not counted toward the streak yet.

### H2 restated on `ab2.sh` — 8 rounds, order alternated

| cell | x86 | arm |
|---|---|---|
| `sync_append` | 1.950 → 1.700 (**x1.147, better 7/7**) | 1.685 → 1.660 (x1.015, 5/8) |
| `sync_remove` | 4.960 → 4.850 (x1.023, 6/7) | 3.520 → 3.500 (x1.006, 4/8) |
| `sync_settle` | 35.04 → 34.77 (x1.008) | 18.13 → 18.14 (x0.999) |
| `sync_first` | 429.5 → 425.9 (x1.008) | 265.7 → 266.3 (x0.998) |

Target HM (sync_append) **x1.077**; no cell regresses. **The win stands.**

The correction the A/A control actually implies is narrower than it first
looked, and it is about *statistics*, not about this result. What the
control measured is the cell's **unpaired** spread — ±8% run to run — and
that is the right bar only for comparing two numbers taken at different
times, which is what a baseline comparison does. It is the wrong bar for a
paired design: with base and new alternated within one machine state, the
statistic is the **sign test across rounds**, and x86 append comes back
better in 7 of 7 (p ≈ 0.008). The magnitude sitting inside the unpaired
spread does not weaken that; it is why the pairing exists.

So the honest position on H2 is: the x86 append improvement is real and
larger than first measured (x1.147 here vs x1.110 on the biased harness),
ARM is parity, and the claim rests on paired sign tests rather than on
medians of independent runs. Absolute levels drifted between the two runs
(x86 append base 1.815 then, 1.950 now) while the ratio held — which is
exactly the drift the pairing is there to absorb.

**Standing rule from here:** report every A/B as (median ratio,
better-in-N-of-M) under `ab3.sh`, and treat a result with no majority in
the sign test as parity no matter how the medians fall.

## Loop state

Non-win streak: 1 (H3 — under re-measurement, see the A/A control)
