# Live-index mutation hill-climb — goal

Maximize the weighted harmonic mean of eight per-cell speedups vs the pinned
baseline: `{arm, x86}` x `{bulk, append, single, remove}`, weights bulk 2,
append 2, single 1, remove 2. `remove` is the mean of the `swap_remove` and
`IdMapIndex.remove` speedups. Every win must hold at `RAYON_NUM_THREADS=1` as
well as multi-threaded — an ST regression is a failed hypothesis, not a
trade-off. Correctness is never traded: the `to_bytes()` digests and the search
top-k of `parity_mutate.py` must match the baseline exactly, and no other
benchmark cell may regress beyond noise.

A win is >1% on the harmonic mean of the target op's arm+x86 cells with neither
of them regressing. Loop: hypothesize -> smoke (<3 min, both boxes) ->
soak-confirm (<15 min) only on a passing smoke. Every hypothesis is logged with
its measurements and verdict, pass or fail. Stop after 20 consecutive winless
hypotheses; any win resets the counter.

## Rig

Measure only on this goal's own pair — `turbovec-bench-mutate` (c3-standard-8)
and `turbovec-bench-arm-mutate` (c4a-standard-8), both in `pydocs-prod`
/ `us-central1-a`, built from images of the masters `turbovec-bench` /
`turbovec-bench-arm`. Never measure locally or on the masters. `rm -rf target`
before each release build; `LD_PRELOAD` the arch's libopenblas. Stop the pair
when idle, delete it at termination.

Note: the ARM master is a c4a (GEN_4 / Marvell), which GCP machine images do not
support, so both boxes are built from boot-disk images instead — same contents,
supported path.

## Harness

- `bench_mutate.py` — the five raw timings, MT and `--st`, at N=200k, dim=768,
  4-bit.
- `whm_mutate.py` — scores a candidate against the baseline; gates the `_st`
  cells and the single-add sanity ratio.
- `parity_mutate.py` — the correctness oracle.
- `data/base_*_all.json`, `data/parity_base_*.json` — the pinned baselines.

## Sanity gate

Single-add takes no pool handoff, so a contaminated grid shows up as the
single-add MT/ST ratio drifting. The ratio is *not* 1.0 on baseline core: an MT
1-row add pays a `with_pool` install that the ST sentinel pool folds away, a
stable x1.35 on arm (reproduced exactly across two independent runs). So the
gate is drift of that ratio away from the baseline's own ratio, not its distance
from 1.0.
