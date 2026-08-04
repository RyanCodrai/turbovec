# Whole-file persistence hill-climb — results log

Objective and rules: `GOAL_persist.md`. Bench:
`benchmarks/hillclimb/bench_persist.py` (N=200k, dim=768, 4-bit), scored by
`whm_persist.py` (save_warm 1, save_mut 2, load 2, load_search 0).

Rig: `turbovec-bench-persist` (c3-standard-8, pd-balanced) and
`turbovec-bench-arm-persist` (c4a-standard-8, hyperdisk-balanced), both in
`pydocs-prod`/`us-central1-a`.

Non-win streak: 0

## Rig notes

- Machine images are not supported for the c4a (Axion/ARM64) master —
  `gcloud compute machine-images create` refuses `generation: GEN_4,
  cpu_vendor: MARVELL, architecture: ARM64`. The ARM clone was therefore
  built from a boot-disk snapshot of `turbovec-bench-arm`
  (`turbovec-bench-arm-snap`) with the same disk type and size, which is
  content-identical to what a machine image would have carried. The x86
  clone came from the machine image `turbovec-bench-mi` as specified.
- The shared checkout at `/Users/ryan/git/turbovec` has a concurrent
  editor (another climb's session moved `HEAD` and reset this branch
  under it). All work for this climb happens in a private worktree; the
  branch is only ever advanced from there.
- Neighbour VMs from other concurrent climbs (`*-search`, `*-mutate`,
  `*-sync`) are running in the same zone. Treat single-round differences
  with suspicion and confirm wins with interleaved A/B rounds.

## Baseline

15 reps/arch at `ea020e35` (harness commit; library code identical to main
`c8d7ec02`), pinned in `benchmarks/results/persist_baseline.json`:

| cell        |    arm |    x86 | arm_st | x86_st |
|-------------|-------:|-------:|-------:|-------:|
| save_warm   | 256.06 | 381.99 | 257.39 | 383.96 |
| save_mut    | 259.75 | 383.17 | 260.13 | 384.05 |
| load        |   2.46 |   8.71 |   2.80 |   8.92 |
| load_search |   7.85 |  19.22 |  10.95 |  25.28 |

Structural readings from the baseline itself:

- Save is device-bound on both arches — 77 MB at 300 MB/s (ARM,
  hyperdisk-balanced) and 203 MB/s (x86, pd-balanced). `save_mut` costs
  only ~1.3 ms more than `save_warm`: a one-row add lazy-appends to the
  blocked cache, so the post-mutation save takes the same warm-borrow
  path and is *not* a distinct re-materialization cost at this shape.
- `_st` ≈ MT on every persistence cell because `RAYON_NUM_THREADS` does
  not reach the persistence paths at all: both the parallel reader and
  the parallel writer spawn `std::thread` scoped threads sized by
  `available_parallelism()`, not rayon workers. The `_st` cells are still
  kept as guards, but a persistence change is expected to move them
  together with their MT twins.
- x86 `load` is 3.5x ARM `load` for identical bytes; the x86 read fuses
  `interleave_chunk_x86` into it while ARM's stored layout is already
  native. That gap is the load cell's headroom, and it is on the
  heavier-weighted arch pair.

## Hypotheses
