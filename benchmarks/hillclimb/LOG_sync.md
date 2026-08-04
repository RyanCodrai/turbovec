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

_(none yet — baseline recording in progress)_

## Loop state

Non-win streak: 0
