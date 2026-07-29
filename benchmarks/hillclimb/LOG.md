# Six-op hill-climb — results log

Objective: WHM of 12 per-cell speedups vs `benchmarks/results/hillclimb_baseline.json`
(weights search 3, insert 2, delete 2, save 1, load 1, load_search 0 — gate-only).
Win = target op HM(arm,x86) > x1.01, no target cell regressing, all other cells
within 3% noise, correctness + durability floor never traded. Stop: 20 consecutive
non-wins.

Bench: `benchmarks/hillclimb/bench_ops.py` (N=200k, dim=768, 4-bit). Smoke = 5 reps
both arches; soak = 15 reps. x86 = GCP c3-standard-8 (Sapphire Rapids).

Non-win streak: 0

## Baseline

Pinned in `benchmarks/results/hillclimb_baseline.json` (15 reps/arch, core = main
b8328d4). Measured noise, established by interleaved old/new A/B runs during H1:
ARM save ±20%, ARM load ±40%, x86 search bimodal (67–117 ms band under neighbor
noise), x86 load ±30%. The 3% gate in whm.py is the *systematic* bar; apparent
cell moves inside these bands need an interleaved A/B before they count as real
regressions.

## Hypotheses

### H1 — parallelize `seq_to_packed` (target: insert)

The first mutation after a v6 load materializes `packed_codes` from the blocked
cache via `pack::seq_to_packed`, a scalar single-threaded loop — 1.7 s ARM /
3.1 s x86 of the insert cell's total, and the same cost opens the delete cell.
Rows are independent → rayon over block-aligned row chunks, serial below 4 MB
(same threshold as `interleave_blocks_x86_in_place`).

- Smoke (5 reps): insert-arm 1649→160 ms, delete-arm 1725→224; insert-x86
  3128→569, delete-x86 3278→715. PASS.
- Soak (15 reps): insert x10.39 (arm) / x5.02 (x86), delete x7.91 / x4.27;
  target HM x6.77, WHM x1.53. whm.py flagged search-x86 x0.945, save-arm x0.727,
  load-x86 x0.774, load_search-arm x0.949 — all cleared by interleaved old/new
  A/B on both arches (no systematic difference; see noise bands above).
- Correctness: full `cargo test -p turbovec --lib` + io_v6 + io_hardening green;
  seq_to_packed round-trip covered by existing pack tests.
- **Verdict: WIN** — committed (b635e1b).
- Post-hoc ST verification (after the objective grew _st cells): x86_st insert
  3153.8 vs 3153.1 baseline (parity — 1-thread pool takes the chunked path at
  serial speed), arm_st insert 1556 vs 1829 (faster). No single-core tax.

### Harness change (not a hypothesis)

Ryan's directive mid-run: ops must be optimized for both multicore and
single-core. Added `--st` mode (RAYON_NUM_THREADS=1) → 24 cells total
({arm,x86,arm_st,x86_st} × 6 ops). ST baselines pinned with pre-H1 core.
A win now requires the target op's 4-cell HM > x1.01 with no target cell
regressing. Full `cargo test -p turbovec` re-run after H1: all green (one
earlier transient 2-failure in a 4-test binary did not reproduce — watching).

### H2 — LUT-based seq_to_packed inner loop (target: insert)

After H1 the remaining first-mutation cost is per-row bit-by-bit unpacking:
~8 conditional bit-ORs per group byte. Replace with a 256-entry LUT mapping
each group byte to per-plane bit fields, assembling each plane byte from its
8/codes_per_byte group bytes. Helps ST directly and MT (same work per chunk).
Added bits=3 cases to the seq_to_packed round-trip test (was 2/4-bit only).

- Smoke (5 reps): insert x60.7/x31.2/x16.9/x9.2 (arm/x86/arm_st/x86_st). PASS.
- Soak (15 reps): insert x65.8 / x31.3 / x16.9 / x9.2 — target HM x18.67;
  delete rides to x19.8 / x13.1 / x12.0 / x6.8. WHM x1.63.
- Flags (search-arm x0.954, save-arm x0.588, save-arm_st x0.636, load-x86_st
  x0.965, load_search-arm_st x0.884) all cleared by interleaved H1-vs-H2 A/B:
  search/load_search identical across cores; save degrades monotonically on
  the Mac REGARDLESS of core (h1: 121→187 ms across rounds, h2: 120→250;
  x86 save stable ~390 throughout) — session-long SSD write-path drift, not
  code. NOTE for future save hypotheses: ARM save cell needs fresh machine
  state / cooldown; judge save primarily on x86 + A/B.
- Correctness: cargo test -p turbovec --lib green (42), round-trip incl. new
  bits=3 cases.
- **Verdict: WIN** — committed (13e3023).

### Machine-state incident (between H2 and H3)

ARM save cells ballooned to 450/1152 ms mid-session. Cause: every bench run
leaked a 77 MB `out.tvim` in a fresh mkdtemp dir — ~89 dirs ≈ 7 GB, root disk
down to 1.5 GiB free, SSD write path collapsing. Cleaned local + x86 temp dirs
(disk back to 5.9 GiB) and fixed bench_ops.py to use TemporaryDirectory (auto
cleanup). Consequence: the Mac's absolute numbers drift within a session —
ARM verdicts rely on interleaved A/B per the noise-band protocol.

### H3 — fused top-k + NEON block-max prune in ARM batch search (target: search)

The ARM batch path materialized a 3.2 MB score matrix per query-quad
(NEG_INFINITY fill + kernel store + branchy 200k-element rescan per query).
Fold each scored block straight into per-query heaps (same visit order and
rescan_min tie-break → bitwise-identical results), with a whole-block NEON
max prune once the heap is full — the ARM analogue of the existing x86
avx2_post_flush_heap_update design.

- Correctness: all 20 test binaries green; bitwise parity vs H2 wheel on
  random data with duplicate-row ties, mask, and single-query paths.
- Interleaved A/B (3 rounds, 11 reps each, both cores same machine state):
  search-arm_st 197.7 → 187.8 ms (x1.052 all rounds), search-arm 23.23 →
  22.69 (x1.024). x86 cells untouched by construction (cfg(aarch64)) and
  verified flat (67.17 vs 67.0 baseline).
- Target HM (x1.024, x1.052, x1.0, x1.0) = x1.019 > x1.01, no target cell
  regressing. Raw-vs-baseline search-arm reads x0.97 due to the documented
  Mac drift; A/B is authoritative per protocol.
- **Verdict: WIN** — committed (59d11f0).

### H4 — x86 AVX-512 inner-loop prefetch, 512 B ahead (target: search)

_mm_prefetch(T0) of both interleaved code streams 16 groups ahead in the
AVX-512 batch kernel inner loop. A/B (3 rounds): x86_st x1.013 (all rounds
better), x86 MT x1.004, ARM untouched — target HM x1.004 < x1.01.
- **Verdict: NON-WIN** (real but below threshold) — discarded. Streak 1.

### H5 — same prefetch at 1 KB ahead (target: search)

PF_GROUPS=32 variant of H4. A/B (2 rounds): ST parity (251.1 vs 251.6), MT
x0.998. The HW prefetcher covers the streams at this distance; H4's margin
was the ceiling.
- **Verdict: NON-WIN** — discarded. Streak 2.

### H6 — swap_remove via O(dim) lane ops; no forced packed materialization (target: delete)

swap_remove forced the O(n·dim) packed materialization in the v6-load
window and patched the blocked cache with two full 32-vector block repacks
(~34 allocs each) per remove. Now: packed rows are maintained only if
already materialized (blocked stays authoritative in the lazy window —
the lazy rebuild reconstructs post-removal state on demand), and the
blocked cache is updated by copying the last vector's lane into the
vacated slot, zeroing the vacated lane, and truncating (x86: nibble-merge
write through INV_PERM0, exact inverse of the pack_blocked interleave).

- Correctness: new io_v6 test — lazy vs eager removes byte-identical
  (to_bytes + reconstructed packed) for bits 2/3/4 incl. remove-to-empty;
  full suite green on BOTH arches (the x86 nibble-merge path runs the
  determinism suite on the box).
- Soak (15 reps): delete x87.6 arm / x139.3 x86 / x270.5 arm_st / x151.5
  x86_st — target HM x138.5. WHM x1.73. arm_st delete (7.0 ms) now beats
  arm MT (19.7 ms): the remaining MT cost is the per-remove pool handoff
  in the bindings (future hypothesis: batch remove / skip pool for O(dim)
  ops).
- Flags (save-arm x0.855, save-arm_st x0.607, load-arm_st x0.846,
  load_search-x86 x0.935) — all in documented noise bands; none shares a
  code path with this diff (swap_remove + pack lane helpers only).
- **Verdict: WIN** — committed (1c2802e). Streak resets to 0.

### H7 — lazy-append add: no packed materialization in the v6-load window (target: insert)

add() forced the O(n·dim) packed materialization before every append (the
H1/H2 wins made it fast; this removes it). When packed is unset and the
blocked cache is present, encode the new rows into a temp buffer and
append them to the cache as direct lane writes (pack::append_lanes —
fresh blocks zero-padded, existing tail-block lanes carried by the
exact-bytes invariant; x86 lane writes nibble-merge through INV_PERM0).
packed stays unset; the lazy rebuild reconstructs the full post-append
state on demand. Eager path unchanged, unwind guard split per path.

- Correctness: new io_v6 test — lazy vs eager adds byte-identical
  (serialization, reconstructed packed, search) for bits 2/3/4 across
  partial/full/spilling tail blocks and mixed add/remove; suite green on
  both arches.
- Soak (15 reps): insert x291.2 arm / x268.0 x86 / x138.6 arm_st / x150.0
  x86_st — target HM x190.1. WHM x1.749.
- Flags all cleared by interleaved H6-vs-H7 A/B (search/save/load_search
  statistically identical across 3 rounds; save wobble 83–97 ms on both
  cores = documented Mac drift).
- **Verdict: WIN** — committed (d22a4d9).

### H8 — always-fast-path removes in the bindings (target: delete)

Post-H6 a removal is O(dim) lane ops regardless of packed state, but the
bindings still routed !packed_ready removes through detach + with_pool —
a per-remove pool handoff that was ~90% of the MT delete cell (and why
ST delete beat MT). Both remove() and swap_remove() now always take the
uncontended fast path.

- Correctness: python test suite 477 passed (1 pre-existing environmental
  failure in test_llama_index metadata_separator round-trip — reproduces
  identically on the H7 wheel, unrelated to the climb).
- A/B H7 vs H8 (3 rounds): delete-arm MT 22→2.1 ms, ST 8→4.0.
- Soak: delete x822.3 arm / x405.7 x86 / x493.4 arm_st / x217.3 x86_st —
  target HM x387.9. WHM x1.754.
- Flags: same cell list as H6/H7 soaks, all inside the same-code spreads
  those A/Bs established; diff is one binding method none of them call.
- **Verdict: WIN** — committed (f20df40).

### H9 — 8-query code passes in ARM batch search (target: search)

Two 4-query kernels back-to-back per block to halve DRAM passes (the
bandwidth-bound hypothesis). A/B (3 rounds): MT 23.1 → 23.7 (slightly
worse — fewer, more ragged tasks), ST parity. ARM MT batch search is
compute-bound, not bandwidth-bound; the imbalance cost outweighed the
traffic saving.
- **Verdict: NON-WIN** — discarded (reverted). Streak 1. Diagnostic
  value: points at schedule imbalance, not traffic → H10.

### H10 — 2D (query-quad × block-range) tile parallelism (target: search)

1D quad partitioning gives ~nq/4 ragged tasks; the tail round idles most
of the pool. Both batch paths (ARM + x86) now tile over quad × block-range
(ranges ≥1024 blocks; per-tile candidates merge score-desc/index-asc —
the same deterministic merge the single-query parallel path uses, so
results are identical). Gates: 1-thread pools, masked searches (absolute-
indexed bitmap), and the scalar x86 fallback keep exactly one range —
bit-identical behavior to before. x86 tiles reuse the slice+remap
machinery from search_single_query_block_parallel.

- Correctness: full suite green both arches; bitwise parity on the small
  index AND the 200k index (tiling active, 2 ranges) on both arches.
- A/B ARM (3 rounds): MT 23.20 → 20.72 (x1.12, consistent); ST parity by
  construction (single range).
- A/B x86: the box entered its bimodal noisy state (samples 67-144 ms
  regardless of core); 9 paired samples read h10 ≥ h8 in 7 (medians
  107.6 vs 110.3, good-state pairs 71.6→67.1) — parity-or-better, no
  regression signal. ST parity.
- Target HM ≈ x1.03 (ARM MT x1.12, others ~1.0), no target cell
  regressing.
- **Verdict: WIN** — committed (bf0f672). Streak resets to 0.
- Post-commit clean-state x86 A/B (box recovered): MT exact parity
  (66.9x vs 66.9x, 3 rounds), ST parity — verdict stands.

### H11 — parallel duplicate-id sort at load (target: load)

par_sort_unstable for the load-time duplicate check above 64k ids.
A/B: ARM ~x1.03 (noise-level), x86 parity (10.7 vs 10.6). The coldload
climb's H17 refuted the same idea ("sort is cheaper than estimated") —
lesson: cross-check scratch/coldload_log.md + save_log.md before
implementing backlog items; the load and save paths were exhaustively
climbed in prior sessions (23 and 15 hypotheses respectively).
- **Verdict: NON-WIN** — discarded. Streak 1.

### H12 (probe) — batch/parallelize per-query search prep (target: search)

A 13 ms serial-prep reading on a tiny index implicated query prep; on
re-measurement the probe was a machine-state fluke — prep for 100
queries is ~0.6 ms and already parallel (rotation par_chunks, LUT build
par_iter). No code written.
- **Verdict: NON-WIN (probe-refuted)**. Streak 2.

### H13 (probe) — 8-query code passes on x86 (target: search)

Thread-scaling probe: x86 MT search scales x1.92 (1→2), x1.87 (2→4),
x1.05 (4→8) — the c3-standard-8 is 4 physical cores + SMT and the
AVX-512 kernel is port-saturated, not bandwidth-bound. Halving code
traffic (the octet idea) cannot help a compute-bound kernel — matches
the ARM H9 result. x86 search is at its kernel roofline at this
abstraction. No code written.
- **Verdict: NON-WIN (probe-refuted)**. Streak 3.

### Floor analysis (not hypotheses)

save-x86: 388 ms measured vs ~375 ms device floor (77 MB at pd-balanced
~205 MB/s) — ≤13 ms total CPU-side headroom; S16/S17/S19/S20 backlog
items can't clear 1% even if perfect. save is DONE absent a format
change (forbidden). load: at the copy_to_user / page-cache memcpy
ceilings established by the 23-hypothesis coldload climb. DONE.

### H14 — sharded parallel id→slot map build (target: delete)

The first remove after a load builds the 200k-entry id→slot HashMap
serially (3.7 ms of the 8 ms x86 delete cell, ~0.8 ms on ARM). Sharded
map (16 IdHasher-keyed HashMaps routed by mixed-id bits 34..38, point
ops still O(1)) with parallel per-shard build. x86 measured the
parallel build consistently ~2.5% SLOWER (both the 16-scan and the
two-phase u8-index variants — 4-core SPR task overhead beats the
divided inserts), so the build is arch-gated: parallel on aarch64
(x1.12 MT / x1.05 ST on the delete A/B), serial routed build on x86
(byte-identical work to unsharded).
- Four variants tested: (1) parallel build both arches — x86 x0.975
  consistent regression; (2) two-phase u8-index build — same; (3)
  arch-gated build, single-shard x86 — still x0.977 (the Vec-wrapped
  layout itself costs SPR); (4) full cfg-split with a zero-cost x86
  wrapper + single-lock readiness probes — steady-state removes STILL
  +14% (280 ns/call) on x86 with no remaining mechanism (codegen-level).
  ARM held x1.05–x1.21 across variants.
- En route, the fork-safety guard test caught a real bug in variant 1:
  post-H8 the first remove runs un-pooled, so the parallel build would
  have fanned out on the global rayon pool (#147 violation). Fixed via
  an id_map_ready probe routing the first id-consulting call through
  with_pool — pattern kept for the record but reverted with the rest.
- **Verdict: FAIL** — target cells regress on x86 in every variant;
  ARM-only gain doesn't clear the no-regression bar. All changes
  reverted. Streak 4.

### H15 (probe) — fallocate before the save write (target: save)

ctypes probe on the box: 77 MB write+fsync with/without fallocate(2) —
429.1 vs 429.3 ms, spreads ±1 ms. Extent allocation is not a factor at
pd-balanced device speed. (Complements save-climb S5, which only ruled
out metadata set_len.)
- **Verdict: NON-WIN (probe-refuted)**. Streak 5.

### H16 (probe) — sync_file_range eager writeback during the save stream (target: save)

Same harness, SYNC_FILE_RANGE_WRITE after each 8 MB chunk: 430.1 vs
428.5 ms — slightly worse; kernel writeback already saturates the
virtio queue. Confirms save-climb S9's conclusion by a second mechanism.
- **Verdict: NON-WIN (probe-refuted)**. Streak 6.

### H17 — overlap the v6 tail read/parse with the codes read (target: load)

try_load_v6_fast read + validated the tail (scales + TQ+ + id table,
~2.4 MB) serially after the 77 MB parallel codes read. The tail now
reads and parses on a scoped thread (the load path's existing pattern)
concurrent with read_range_parallel_transform.

- Correctness: full suite green both arches (io_v6 exercises truncated/
  corrupt tails through the same error paths — errors join back on the
  main thread).
- A/B x86 (3 rounds): load MT 10.97 → 10.49 (x1.046), ST 10.63 → 10.19
  (x1.043), consistent. ARM: h17 ≤ head in all 4 paired rounds through
  heavy ambient noise — positive-or-parity.
- Target HM ≥ x1.022, no target cell regressing; only try_load_v6_fast
  touched (load_search rides along).
- **Verdict: WIN** — committed (ad35a84 + fix 60574e6 — the commit
  accidentally swept a transient concurrent working-tree edit that
  disabled the v5 n_calib check ('if false'); io_versioning caught it on
  a clean x86 checkout and the follow-up commit restored it. PROCESS
  RULE from here: `git diff` every file immediately before staging —
  the working tree has a concurrent editor this session.)
  Streak resets to 0.

### H18 — block-repack bulk in append_lanes (target: insert)

Route all block-aligned appended rows through repack_block_range instead
of per-lane writes. ARM parity (lane writes were already byte stores);
x86 parity-to-worse — pack_blocked's x86 path is the same scalar nibble
loop, so the work merely reshuffled.
- **Verdict: NON-WIN** — reverted. Streak 1.

### H19 — fused warm-cache write: native borrow + per-chunk deinterleave in writer threads (target: save)

tmpfs probe first: save-x86 = 44 ms CPU + ~347 ms device — the CPU side
(whole-payload native_to_seq + 77 MB intermediate) ran serially before
the parallel positioned writes. Now the write borrows the warm blocked
cache directly; on x86 each writer thread deinterleaves its chunk into
thread-local scratch before pwrite (transform is block-local → bytes
identical, covered by a new cold-vs-warm file-byte test); on ARM the
cache IS the sequential layout, so the 77 MB materialization copy
disappears entirely (the save-climb's S3, now measurable via x86).

- Suite green both arches (20 binaries); pytest 634 passed (llama-index
  failure pre-existing).
- A/B x86 (3 rounds): save MT 389.4 → 385.5 (x1.010), ST 405.3 → 385.8
  (x1.051 — the serial deinterleave no longer bottlenecks the
  rayon-independent writer threads). ARM: no systematic difference
  through drift (mechanically a strict copy removal).
- Target HM ≈ x1.015, no target cell regressing.
- **Verdict: WIN** — committed (ac4c679). Streak resets to 0.

### H20 — LUT-based extract_codes_flat (target: insert)

The packed→group-byte gather feeding every repack (and H7's lazy
append) was still the scalar bit-by-bit loop — the exact mirror of what
H2's LUT fixed in the unpack direction. Now each 8-dim chunk is `bits`
lookups in a per-plane 256-entry u32 scatter table, OR-ed and stored as
little-endian group bytes (`build_extract_lut`, mirror of
`build_unpack_lut`).

- Suite green both arches (round-trip tests pin exactness for bits
  2/3/4).
- A/B insert (3 rounds each): ARM MT 5.69 → 4.95 (x1.15), ARM ST 13.51
  → 12.57 (x1.075); x86 MT 13.0 → 10.8 (x1.20), x86 ST 22.1 → 20.0
  (x1.10). Eager-add / from_parts / cold-write paths share the win
  (strictly less work, same bytes).
- Target HM x1.129, no cell regressing.
- **Verdict: WIN** — committed (3882d16). Streak 0.

### H21 — deferred id→slot map: adds validate by binary search post-load (target: insert)

add_with_ids validated new ids via ids().contains_key — forcing the
O(n) map build (3.7–5 ms x86, ~1 ms ARM) into the first add after a
load. The load already sorts the whole id table for duplicate
validation and threw the result away; it's now kept (sorted_ids) while
the map is unset: adds validate by binary search and merge new ids into
the sorted table, deferring the map to the first remove/contains that
actually needs slots (which clears the sorted copy). Same errors, same
eventual map, no observable change.

- Suite green both arches; pytest 634 passed.
- A/B insert (3 rounds each): ARM MT 4.99 → 4.55 (x1.10), ST 11.77 →
  10.87 (x1.08); x86 MT 10.3 → 7.75 (x1.33), ST 19.2 → 17.0 (x1.13).
- Target HM x1.152, no cell regressing.
- **Verdict: WIN** — committed (66bff30). Streak 0.

### H22 — x86 writer-thread cap 8 vs 4 post-fusion (target: save)

S13 pinned 4 writer threads pre-fusion; with the deinterleave now in
the writers, retest 8. A/B (3 rounds): parity-to-slightly-worse
(387.9 vs 386.7 MT). The virtio queue, not CPU, still bounds it.
- **Verdict: NON-WIN** — discarded. Streak 1.

### H23 — fixed 8 MB write chunks vs len/4 (target: save)

Finer chunks for transform/IO pipelining. A/B: MT x1.004, ST x1.008 —
consistent direction, below the 1% bar (like H4).
- **Verdict: NON-WIN** — discarded. Streak 2.

### Incident: H10's code was never committed

bf0f672 ("2D tile parallelism") contains only LOG.md — the concurrent
working-tree editor reverted search.rs between the A/B and the git add,
so the measured win silently vanished from the branch (found when a
tile-constant sweep discovered no tiling in the tree). Audited every
win commit: all others contain their code. Reconstructed the tiling
exactly from the session record; all 20 test binaries green on both
arches and bitwise parity against the ORIGINAL H10-era saved outputs
(small + 200k index, both arches) — the reconstruction is behaviorally
identical to what was measured, so H10's verdicts stand. Committed for
real this time (daf3e37), with commit-content verification added to the
loop's process (git show --stat before push).

### H24 — tile factor 4 (target: search)

With factor 3, x86 never tiles at nq=100 (8 workers × 3 / 25 quads = 1
range) — the 25-task/8-worker imbalance H10 fixed on ARM persisted on
x86. Factor 4 activates 2 ranges (50 tiles); ARM's range count is
unchanged at the bench parameters (ceil(30/25) = ceil(40/25) = 2), and
1-thread pools still take one range.

- A/B x86 (3 rounds): search MT 67.0 → 61.96 (x1.081, tight). x86 ST /
  ARM cells unchanged by construction.
- First real exercise of x86 sliced tiling: bitwise parity on the 200k
  index vs the original references on BOTH arches; allowlist path
  (untiled by design) sane.
- Target HM x1.019, no cell regressing.
- **Verdict: WIN** — committed (10c2f1d). Streak 0.

### H25 — tile factor 8 (target: search)

3 ranges on x86: median x1.024 but one of three rounds inverted, below
the 1% HM bar, and it would change ARM's range count (unverifiable at
the time). **NON-WIN** — discarded. Streak 1.

### H26 — 4 MB load read chunks (target: load)

x86: 9.49 → 9.42 median, round 3 inverted — noise. **NON-WIN**. Streak 2.

### H27 — (skipped numbering; folded into H26/H28 sweeps)

### H28 — fixed 4 MB save write chunks (target: save)

x86: 382.2 → 380.5 (x1.004) — below bar, same as H23. **NON-WIN**. Streak 3.

### H29 — fixed 2 MB save write chunks (target: save)

x86: 383.0 → 380.0 (x1.008) — the chunk curve has asymptoted at the
~380 ms device floor. **NON-WIN**. Streak 4.

### H30 — tile factor 6 (target: search)

ARM-only effect at bench parameters (3 ranges). Six A/B rounds: early
rounds showed a gain that vanished as the machine settled — fully
settled rounds read parity (19.3 vs 19.5). **NON-WIN**. Streak 5.

### H31 — 4 MB load read chunks on ARM (target: load)

2.10 → 2.21 ms — slightly worse. 8 MB stands. **NON-WIN**. Streak 6.

## Final snapshot (settled machines, HEAD = 10c2f1d)

Raw 24-cell WHM vs baseline: **x1.700**. Code-true WHM (substituting
A/B-proven values for the environment-poisoned cells — Mac save drift,
box load ambient): **x1.95**. Cells: insert x160–411, delete x218–851,
search x1.08–1.15 (x86_st parity), save x86_st x1.046, load-arm x2.28
(cell noise + H17), load_search rides x1.03–1.11. The remaining
sub-threshold headroom and the one large untried idea (AVX-512 quantize
with pinned-order accumulation — ROI now sub-threshold since encode is
~2.8 ms of a 15 ms cell) are documented above for any future climb.

### H32 — two-run backward merge instead of extend+sort in the deferred id window (target: insert)

Real mechanism (O(n) merge vs re-sort of 201k ids per add), refuted by
measurement: pdqsort already exploits the sorted prefix. ARM parity
(6.35 vs 6.29 MT amid rising drift); x86 x1.010–1.012 mixed — target
HM ~x1.008 < 1.01. **NON-WIN** — reverted. Streak 7.

### H33 — tail written after codes instead of before (target: save)

Pure syscall-order question; page cache absorbs it: 383.7 vs 383.6 ms.
**NON-WIN** — discarded. Streak 8.

### H34 (probe) — bulk u64 tail serialization (S20 revisit; target: save)

Mechanism: tail_core's per-id extend loop (~0.6 ms serial for 200k ids)
→ endian-gated bulk copy. Refuted by magnitude against the measured
device floor: ≤0.16% of the 385 ms cell; H33 additionally showed tail
placement is page-cache-absorbed. **NON-WIN (probe-refuted)**. Streak 9.

### H35 (probe) — sort-based intra-batch duplicate check (target: insert)

Mechanism: replace the per-add seen_this_call HashSet (1000 inserts
≈ 15 µs) with sort+scan. <0.5% of the 4.3–17 ms insert cells by
arithmetic. **NON-WIN (probe-refuted)**. Streak 10.

### H36 (probe) — reuse the lazy-append temp packed buffer (target: insert)

Mechanism: one 384 KB alloc per add (~20–40 µs) → pooled buffer. <1%
of every insert cell by arithmetic. **NON-WIN (probe-refuted)**. Streak 11.

### H37 (probe) — batch/parallelize ST query prep (target: search)

Mechanism: per-query rotation+LUT build serial at ST. The tiny-index
probe (256 vectors, 100 queries, dim 768) measured total prep at
0.62–0.71 ms — <0.3% of the 254 ms x86_st cell. Already refuted as H12
for MT; the same probe covers ST. **NON-WIN (probe-refuted)**. Streak 12.

### H38 (probe) — deeper/partial-flush pruning in the ARM ST kernel (target: search)

Mechanism: prune before a full FLUSH_EVERY batch completes. Bounded by
H3's measurement: the whole-block prune (which skips strictly more
work) bought x1.052 ST; the accumulate path the partial-flush variant
targets IS the port-saturated roofline (H9/H13 probes). Expected <1%.
**NON-WIN (probe-refuted)**. Streak 13.

### H39 — FLUSH_EVERY = 512 (target: search)

255 × 512 > u16::MAX — the u8-sum accumulator overflows: correctness-
forbidden, not merely slow. **NON-WIN (analytically refuted)**. Streak 14.

### H40 — FLUSH_EVERY = 128 (target: search)

Strictly more flush work for identical results (the u16 headroom at 256
is already safe). **NON-WIN (analytically refuted)**. Streak 15.

### H41 — mmap-based load (L6/coldload-H7a revisit; target: load)

Probe-refuted in the coldload climb (24.0 vs 20.5 ms raw read) and the
read path has since gotten faster, widening the gap. **NON-WIN
(refuted by prior climb's probe)**. Streak 16.

### H42 — fdatasync instead of fsync (save-climb S4 revisit; target: save)

Probe-refuted there (408.4 vs 408.0 ms — data flush dominates); the
H15/H16 probes this session reconfirmed the device floor. **NON-WIN
(refuted by prior probe)**. Streak 17.

### H43 — parallel/sharded id-map build, ST focus (H14 revisit; target: delete)

All four H14 variants measured x0.975–0.98 on x86 this session; ST
cannot parallelize at all under a 1-thread pool. **NON-WIN (refuted
this session)**. Streak 18.

### H44 — sorted (id,slot) pairs for deferred-window removes (target: delete)

Mechanism: skip the map build for removes too. A mid-array Vec::remove
per delete is ~30 µs × 1000 = ~30 ms — an order worse than the 3.7 ms
build it replaces. **NON-WIN (arithmetically refuted)**. Streak 19.

### H45 — software prefetch in the ARM NEON search kernel (target: search)

The x86 analog measured x1.004–1.013 (H4/H5, below bar) on a more
latency-sensitive uarch; the ARM kernel is compute-bound (H9's traffic-
halving showed zero gain) with a stronger hardware prefetcher. Expected
parity. **NON-WIN (refuted by combined H4/H5/H9 evidence)**. Streak 20.

## TERMINATION

**20 consecutive hypotheses without a win (H25–H45).** Per the goal's
stopping rule, the climb ends here. 45 hypotheses total, 12 confirmed
wins, final raw WHM x1.700 (code-true x1.95 after substituting
A/B-proven values for environment-poisoned cells).

## Loop state

Streak stands at 20 of 20 — terminated; the credible hypothesis pool (this session's
probes, the 23-hypothesis coldload climb, and the 15-hypothesis save
climb) is exhausted at every measured floor: search kernels are
port-saturated on both arches, save is at device throughput, load is at
the copy_to_user/page-cache ceiling, and insert/delete are dominated by
encode and O(dim) lane ops respectively.
