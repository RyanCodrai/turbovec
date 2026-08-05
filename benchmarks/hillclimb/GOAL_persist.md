# Whole-file persistence hill-climb — goal

Maximize the WHM of the eight persistence cells — `save_warm` (weight 1),
`save_mut` (weight 2), `load` (weight 2) and `load_search` (weight 0,
gate-only) measured on `arm` and `x86` — against the baselines pinned in
`benchmarks/results/persist_baseline.json`. Scope is whole-file
persistence: `write()` / `write_with_durability` down through the atomic
temp-file protocol, and `load()` down through the v6 fast path, on both
arches. `load_search` carries no weight; it exists solely to veto a
"win" that shifts cost out of `load` and into the first search.

A win is: the target op's `arm`+`x86` HM > x1.01, neither of its two
cells regressing, every other measured cell (including the `_st` guard
cells) within 3%, `cargo test -p turbovec` green, and `to_bytes`
round-trip equality preserved. The durability floor — payload fsync
before an atomic rename, plus the parent-directory fsync — is never
traded for speed, and neither is the temp-file protocol that keeps a
reader from ever seeing a torn index.

Measurement happens only on this goal's own GCP pair,
`turbovec-bench-persist` / `turbovec-bench-arm-persist` in
`pydocs-prod` / `us-central1-a`, cloned from the masters
`turbovec-bench` / `turbovec-bench-arm`. Never on the masters, never
locally. `rm -rf target` before each release build; `LD_PRELOAD` the
arch's libopenblas.

Loop: hypothesize → smoke (<3 min, both boxes) → soak-confirm (<15 min)
on a passing smoke only. Every hypothesis is logged in `LOG_persist.md`
with its measurements and verdict, pass or fail. Stop after 20
consecutive winless hypotheses; any win resets the counter.

## Prior art this climb inherits

The six-op climb (`LOG.md`, H15/H16/H19/H22/H23/H26/H28/H29/H31/H33/H34/
H41/H42) and the earlier save- and coldload-specific climbs already
refuted, at the save/load cells: `fallocate` before the write,
`sync_file_range` writeback during it, `fdatasync` instead of `fsync`,
writer-thread counts above 4, fixed 2/4/8 MB write chunks, 4 MB load
read chunks on both arches, tail-after-codes ordering, bulk u64 tail
serialization, and mmap-based load. Re-opening any of them requires
saying which measurement stopped covering it — the new `save_warm` cell
and the ARM-side asymmetries are legitimate grounds; a bare retry is
not.
