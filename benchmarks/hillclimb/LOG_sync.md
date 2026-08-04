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

## Loop state

Non-win streak: 0
