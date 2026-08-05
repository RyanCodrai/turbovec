# Search-latency hill-climb — results log

Cells: `{arm, x86} x {search, load_search}`, weights 3:1. `search` is
nq=100, k=10 at N=200k, dim=768, 4-bit; `load_search` is a fresh process
doing `IdMapIndex.load()` + one k=10 query. A win is >1% on the harmonic
mean of the target's arm and x86 cells with neither regressing.

Rig: `turbovec-bench-search` (c3-standard-8, Sapphire Rapids) and
`turbovec-bench-arm-search` (c4a-standard-8, Axion), pydocs-prod,
us-central1-a, built from boot-disk images of the masters. Note ARM
machine images are unsupported by GCP, so both boxes come from
`gcloud compute images create --source-disk` rather than machine images.
`rm -rf target` before every release build; LD_PRELOAD the arch's
libopenblas; one process per cell.

## Baseline (HEAD = c8d7ec02, three interleaved rounds each)

| cell | r1 | r2 | r3 | baseline |
|---|---|---|---|---|
| search-arm | 40.394 | 40.335 | 40.462 | **40.394 ms** |
| search-x86 | 61.685 | 61.757 | 61.740 | **61.740 ms** |
| load_search-arm | 9.952 | 9.999 | 8.574 | **9.952 ms** |
| load_search-x86 | 20.286 | 21.112 | 20.715 | **20.715 ms** |

`search` is remarkably stable here (spread <0.5%); the previous climb's
bimodal 67–117 ms x86 search is not reproducing on this box at this HEAD.
`load_search` is the noisy cell (~14% spread on arm), so it needs more
reps and interleaving to call.

## Orientation probes (not hypotheses — no streak effect)

### P1 — is `search` DRAM-bandwidth-bound on re-streaming the codes?

nq=100 makes 25 passes over the 76.8 MB code array (the kernels batch
QBS=4 queries per pass on both arches), which is ~1.9 GB per search — on
its face a bandwidth story. Refuted: cost per (query·vector) *improves*
monotonically as the index outgrows cache.

| N | codes | arm ns/(q·vec) | x86 ns/(q·vec) |
|---|---|---|---|
| 8 192 | 3.1 MB | 2.819 | 4.800 |
| 32 768 | 12.6 MB | 2.257 | 3.606 |
| 200 000 | 76.8 MB | **2.002** | **3.077** |

The DRAM-resident point is the *cheapest* per unit work, so the scan is
compute-bound, not bandwidth-bound. Widening the kernel's query batch
(QBS 4→8) to halve code traffic is therefore refuted before it is built.
This independently reconfirms the previous climb's port-saturation
finding for these kernels.

### P2 — how big is the fixed, index-size-independent part of a search?

Extrapolating the same sweep down to a one-block index:

| N | arm ms | x86 ms |
|---|---|---|
| 32 | 0.658 | 1.408 |
| 1 024 | 0.776 | 1.647 |
| 200 000 | 39.775 | 61.541 |

Fixed overhead is **0.66 ms arm (1.7% of the cell) / 1.41 ms x86 (2.3%)**.
That is the entire addressable surface on the `search` side, and only if
prep were driven to zero — which it cannot be, since rotation and LUT
build are real work. Ceiling well under 2%.

### P3 — what is that fixed cost made of?

Sweeping nq at N=32, so the scan is negligible:

| nq | arm µs/query | x86 µs/query |
|---|---|---|
| 1 | 31.9 | 44.6 |
| 25 | 7.8 | 15.2 |
| 100 | 5.3 | 12.0 |
| 200 | 4.9 | 12.1 |

Essentially all per-query (flat cost ~0.02 ms), i.e. rotation + TQ+
calibration + LUT build. x86 pays **2.4x** arm per query despite being
the faster box elsewhere — `build_query_neon_lut_from_slice` is shared by
both arches and has no AVX path. Vectorizing it is a genuine x86-only
win worth ~0.7 ms, but that is HM(1.000, 1.011) ≈ x1.006 across the two
`search` cells — **below the 1% bar on its own**, and recorded here as
the reason the search side is not the place to spend hypotheses.

### P4 — where does the `load_search` cell actually go?

Phase breakdown inside a fresh process, page cache warm (median of 5):

| phase | arm | x86 |
|---|---|---|
| load | 5.25 ms | 14.6 ms |
| first search | 1.50 ms | 2.96 ms |
| cell total | 6.80 ms | 18.5 ms |
| warm search (nq=1) | 1.04 ms | 2.44 ms |

**load is 75–80% of the cell.** Note load costs ~1.7x more here than the
warm-harness `load` op (2.99 ms arm / 9.79 ms x86) measures: that op
loads 11 times in one process, so the allocator hands back the same
77 MB block and the kernel never re-zeroes it. The cold cell pays that
zeroing every time, and the cold path is what this goal measures.

### P5 — ceiling on the cold load path: anonymous read vs mmap

Fresh process per data point, page cache warm, both modes made to
consume every byte (so mmap cannot win by deferring faults — the cell
would pay them in its first search anyway):

| mode | arm: available / +consume | x86: available / +consume |
|---|---|---|
| read into fresh 77 MB buffer | 8.30 ms / 52.6 ms | 32.8 ms / 69.0 ms |
| mmap the file | 0.025 ms / 45.7 ms | 0.029 ms / 39.9 ms |

Consuming-inclusive, mmap saves **~6.9 ms on arm and ~29 ms on x86** of
pure zero-plus-copy traffic. This is the largest lever the cell offers,
and it is the basis of H1.

The previous climb refuted mmap (its H41: 24.0 vs 20.5 ms) — but it
measured the *warm* `load` op, the exact regime in which the allocator
has already absorbed the zeroing that mmap exists to avoid. That
measurement stopped covering the question the moment the target cell
became cold load→first-search, so H1 re-opens it.

## Hypotheses

### H1 — map the code section instead of reading it into a fresh buffer (target: load_search)

Mechanism from P4/P5: the cold load allocates a 77 MB anonymous buffer
(the kernel zeroes it) and then overwrites every byte via `copy_to_user`.
Mapping the file skips both. Implemented as `io::CodeBuf`
(`Owned(Vec<u8>) | Mapped{Arc<Mmap>,start,len}`) with `Deref<[u8]>` and a
`to_mut()` that promotes to owned before any in-place edit, so mutation
semantics are unchanged. Only taken where the stored layout is already
the kernel layout — every non-x86 target — because x86 must run
`interleave_chunk_x86` over the bytes and has to materialize them anyway.

Smoke (2 rounds, reps=15, vs baseline):

| cell | baseline | H1 | ratio |
|---|---|---|---|
| load_search-arm | 9.952 | 5.611 / 5.697 | **x1.77** |
| load_search-x86 | 20.715 | 19.166 / 20.114 | x1.05 (within cell noise; x86 takes the read path) |
| search-arm | 40.394 | 41.942 / 41.957 | **x0.963** |
| search-x86 | 61.740 | 61.954 / 61.879 | x0.997 (noise) |

The target cell moved exactly as predicted, but `search-arm` regressed
**3.9%** — tight and repeatable, not noise. Cause: an anonymous buffer
gets transparent huge pages for free, a file mapping is 4 KB-paged, so
the scan's TLB reach collapses from ~38 entries to ~19,700 over the
77 MB it walks 25 times per search. Trading 1.55 ms of a weight-3 cell
for 4.34 ms of a weight-1 cell is a WHM gain on arm (x1.087) but the
acceptance rule is "neither regressing", and `search-arm` regresses.
**NON-WIN as measured** — carried into H2 rather than discarded, since
the regression has a single identified cause with a direct remedy.
Streak 1.

### H2 — H1 plus `MADV_HUGEPAGE` on the mapping (target: load_search)

Same change, asking the kernel for huge pages back on the mapped range
so the scan keeps its TLB reach. Advisory only: kernels built without
read-only THP for page cache reject it and the mapping stays correct,
just 4 KB-paged.

| cell | baseline | H2 |
|---|---|---|
| search-arm | 40.394 | 42.133 / 41.691 (**x0.965**) |
| load_search-arm | 9.952 | 5.699 / 5.517 (x1.77) |

No effect: these boxes run Debian 12, whose kernel will not back a
read-only page-cache mapping with huge pages (`MADV_HUGEPAGE` on a file
mapping needs read-only THP for filesystems, and khugepaged collapse is
asynchronous even where it is built in). The `search-arm` regression is
unchanged. **NON-WIN.** Streak 2.

The pair H1/H2 establishes the real constraint: on this kernel a code
buffer can have a cheap cold load (mapped, 4 KB pages) or a fast
repeated scan (anonymous, huge pages), not both at once. H3 stops
treating that as a static choice.

### H3 — map on load, copy to owned memory once the copy has paid for itself (target: load_search) — **WIN**

`io::CodeBuf` keeps the mapping from H1;
`TurboQuantIndex::promote_mapped_codes` copies it into owned memory, and
the Python binding calls that after `SEARCHES_BEFORE_PROMOTE = 4`
searches on the same index. The threshold is read straight off the
measurements rather than picked: the copy costs ~5 ms and mapped pages
cost ~1.55 ms per search, so four searches is break-even. A caller that
loads, queries once and drops never reaches it and keeps the whole load
win; a long-lived index pays it once and scans huge pages forever after.

Soak — 3 interleaved rounds, reps=15, one process per cell:

| cell | baseline | H3 rounds | median | ratio |
|---|---|---|---|---|
| search-arm | 40.394 | 40.188 / 40.654 / 40.325 | 40.325 | x1.002 |
| load_search-arm | 9.952 | 5.493 / 5.674 / 5.921 | 5.674 | **x1.754** |
| search-x86 | 61.740 | 61.736 / 61.671 / 61.784 | 61.736 | x1.000 |
| load_search-x86 | 20.715 | 19.659 / 21.312 / 20.671 | 20.671 | x1.002 |

Target cell `load_search`: HM(1.754, 1.002) = **x1.276**, far above the
x1.01 bar, with neither target cell regressing and both `search` cells at
parity. x86 is unchanged by construction — it must run
`interleave_chunk_x86` over the codes, so it never takes the mapped path
and only the arm cell moves.

`cargo test -p turbovec` green (all 20 suites, 121 unit + integration).

**Rejected on the non-target cells.** Baseline vs H3 on arm, reps=9:

| cell | baseline | H3 | ratio |
|---|---|---|---|
| insert-arm | 2.849 | 11.610 | **x0.245** |
| delete-arm | 3.140 | 12.351 | **x0.254** |
| save-arm | 169.313 | 168.408 | x1.005 |
| load-arm | 3.769 | 2.006 | x1.88 |

`insert` and `delete` time a mutation with the load *excluded*, so
deferring the 77 MB materialization to the first mutation drops the
entire copy inside their timed region — +8.76 ms, almost exactly one
single-threaded 77 MB memcpy. The total work is unchanged (the baseline
pays it during the untimed load), but the goal's rule is that no other
cell regresses, and these regress 4x. **NON-WIN — reverted.** Streak 3.

This closes the mapping line rather than just this variant. Any scheme
that maps and materializes later moves cost out of `load` and into
whichever operation touches the codes first, and the `insert`/`delete`
cells are defined to exclude load from their timing — so the cost has
nowhere to land that is not someone's regression. Making the promotion
copy parallel (~8.8 ms → ~2 ms) would shrink but not remove it. The
only escape would be a segmented buffer (mapped prefix + owned suffix)
that mutation paths can extend without materializing, which is a format-
adjacent redesign well outside a latency hill-climb.

Recorded for whoever revisits: the win it was buying was real and large
(load_search-arm x1.5–1.75 across all rounds measured, load-arm x1.88),
and it is available to any caller that can promise read-only use.

### H4 (probe-refuted) — warm the rayon pool during the load's I/O wait (target: load_search)

P4 showed the first search in a fresh process costs ~0.45 ms more than a
warm one (arm 1.50 vs 1.04), and the load path reads with raw scoped
threads rather than the rayon pool, so lazy pool construction looked like
the cause — and overlapping it with the read would have been nearly free.

Probing it found a much larger effect than expected: doing *any* work
before the load drops the first search from ~1.48 to ~0.80 ms **and every
later search from ~1.05 to ~0.59 ms**, which a one-off pool build cannot
explain. Three candidate mechanisms, tested in turn:

| pre-warm | search0 | steady | verdict |
|---|---|---|---|
| none (`raw`) | 1.28–1.93 | ~1.05 | — |
| 32-vector index built + searched | 0.77–0.84 | ~0.59 | recovers it fully |
| 4096-vector index | 0.83–1.21 | ~0.55 | no better than 32 — not encode volume |
| **90 ms busy-wait, no turbovec work at all** | **1.01–1.04** | **~0.58** | recovers it fully |

Huge pages were ruled out directly: `smaps_rollup` shows the code buffer
is 92% `AnonHugePages` in *both* the warm and cold cases, so page size is
not the variable. The busy-wait row settles it — the premium is **CPU
frequency ramp-up**. A fresh process starts on a downclocked core and the
first few milliseconds of any work clock it up.

So ~0.45 ms of every `load_search` measurement is DVFS, not code, and
nothing in the pool, allocator or page-table layer can remove it. Burning
CPU to pre-ramp would "improve" the cell while making real cold starts
worse. **NON-WIN (probe-refuted).** Streak 4.

## Where the two cells now stand

Both target cells have been traced to a floor:

* **`search`** is compute-bound at the kernel (P1: per-unit cost *falls*
  as the index outgrows cache, so it is not bandwidth-limited), which
  reconfirms the previous climb's port-saturation result on this newer
  HEAD. Everything outside the kernel is 1.7% (arm) / 2.3% (x86) of the
  cell (P2), it is all per-query prep (P3), and the one clear inefficiency
  in it — the LUT builder having no AVX path, costing x86 2.4x arm per
  query — is worth ~1.1% on one arch, i.e. HM x1.006, below the bar.
* **`load_search`** is 75–80% load (P4). Load is at the roofline for a
  read into a fresh anonymous buffer, the only way past it is mapping
  (P5, worth x1.5–1.75 on the cell), and mapping is closed by the
  non-regression rule because the deferred materialization lands in
  `insert`/`delete` (H1–H3). The rest of the cell is a first search whose
  premium over steady state is CPU frequency ramp (H4).

### H5 — raise the block-axis tile target so the schedule stops leaving a ragged wave (target: search) — **WIN**

The kernels being at roofline (P1) does not make the *schedule* optimal,
and the previous climb's only large search win was a scheduling accident
at this exact shape. At nq=100 on 8 workers the batch dispatch built
25 query-quads x 2 block-ranges = 50 tiles: 6.25 waves of work rounded up
to 7, so a whole wave runs mostly idle. The tile target
(`n_threads * 4` in `n_block_ranges`) is what held the count down; the
`min_tile_blocks` cap would have allowed 7 ranges.

Swept interleaved via a temporary `TV_TILE_MULT` override (one build, so
the values cannot differ by anything but the schedule), 3 rounds x
reps=11, then soaked 5 rounds x reps=15. Medians:

| target | ranges | tiles | search-arm | search-x86 |
|---|---|---|---|---|
| 4 (base) | 2 | 50 | 41.43 | 61.85 |
| 8 | 3 | 75 | 39.69 | 60.53 |
| 16 | 6 | 150 | 37.64 | 60.27 |
| **32** | 7 (at cap) | 175 | **37.50** | **60.04** |

At 32: **search-arm x1.105, search-x86 x1.030, HM x1.066** — six times
the bar, neither cell regressing. 32 beat 16 on both arches and the x86
samples do not overlap (60.02–60.06 vs 60.23–60.29).

Bitwise parity is the hard gate here, since the range count decides how
many per-range top-k heaps get merged. Verified on both arches against
the target=4 reference over nq ∈ {1,4,25,100,257} x k ∈ {1,10,100} plus
masked and deliberately tied-score shapes — **40 result arrays, bitwise
identical at every target swept** (8, 16, 32).

Non-target cells, target 4 → 16:

| cell | arm | x86 |
|---|---|---|
| insert | 2.942 → 2.880 | 5.530 → 5.583 |
| delete | 3.184 → 3.188 | 7.902 → 7.773 |
| load_search | 9.720 → 8.053 | 19.428 → 18.817 |

All parity or better — expected, since those cells search with nq=1,
which takes the single-query path and never builds these tiles.

Shapes below nq≈64 are untouched: there `min_tile_blocks` already
decided the count under the old target, and the count still collapses to
1 once `n_quads` alone exceeds the target. `RAYON_NUM_THREADS=1` returns
1 range as before.

### H6 — lower `MIN_TILE_BLOCKS` so the split can go finer than H5's cap (target: search)

H5's tile-target sweep improved monotonically all the way to the
`min_tile_blocks` cap and stopped there, so it never demonstrated a peak
— only that it ran out of room. If the cap was the binding constraint,
lowering it should buy more. Swept via a `TV_MIN_TILE_BLOCKS` override in
one build, 3 rounds x reps=11, with the unmodified value measured twice
per round as the control:

| cap | ranges | search-arm | search-x86 |
|---|---|---|---|
| 1024 (control) | 7 | 36.64 | 60.13 |
| 512 | 11 | 36.99 | 60.99 |
| 256 | 11 | 36.89 | 61.13 |
| 128 | 11 | 36.83 | 61.13 |

Refuted, and it says something useful: with the target at 32 the cap is
**no longer** what binds — at 512 and below the target itself yields 11
ranges, so every row past 512 is the same schedule. Going 7 → 11 ranges
costs rather than pays (x86 −1.7%, arm −0.6%), because past the point
where the workers are balanced the extra ranges only duplicate per-range
top-k work. H5 did not stop short of a peak; it landed on one. 1024 is
the right cap. `cargo test -p turbovec` green at 1024, 256 and 64 (446
passed, 0 failed each). **NON-WIN.** Streak 1.

### H7 — enumerate tiles block-range-major instead of query-quad-major (target: search)

Tiles are currently emitted quad-outer, so the ~8 tiles in flight at any
moment sit in *different* block ranges and stream disjoint slices of the
77 MB code array. Block-outer keeps concurrent tiles inside one range,
where they share those bytes in L2/L3. Identical tile set — only the
order rayon draws from — so results cannot change.

Sweep (3 rounds x reps=11), medians against the interleaved control:

| order | search-arm | search-x86 |
|---|---|---|
| quad (control) | 36.64 | 60.13 |
| block | 36.17 | 59.76 |

arm x1.013, x86 x1.006 — marginally under the x1.01 bar, but consistent:
every block-order sample beat every control sample on both arches.

Soak, 6 interleaved rounds x reps=15, medians:

| order | search-arm | search-x86 |
|---|---|---|
| quad (control) | 36.731 | 60.139 |
| block | 36.220 | 59.787 |
| | x1.0141 | x1.0059 |

**HM x1.00998** — under the bar by 0.002%. Not noise (all 6 x86 block
samples below all 6 controls), but it does not clear x1.01. See H9 for
the definitive re-measurement.

### H8 — block-major order combined with finer ranges (target: search)

H7's mechanism predicts its own improvement: if the win comes from
concurrent workers sharing a range's slice of the code array in cache,
then shrinking that slice should deepen it. H6 had found extra ranges
useless, but H6 ran under *quad* order, where concurrent tiles sit in
different ranges and there is no slice to share — so the two knobs are
only meaningful jointly. Swept block-order across (tile target, cap)
pairs, 3 rounds x reps=11:

| config | ranges | slice | search-arm | search-x86 |
|---|---|---|---|---|
| shipped (quad, 32, 1024) | 7 | — | 37.113 | 60.203 |
| block (32, 1024) | 7 | 11.0 MB | 36.872 | **59.789** |
| block (64, 256) | 21 | 3.7 MB | **36.260** | 60.635 |
| block (128, 128) | 41 | 1.9 MB | 37.391 | 62.655 |
| block (256, 64) | 82 | 0.95 MB | 37.371 | 62.617 |

The mechanism is real on arm — 21 ranges beats 7 (x1.023 vs shipped) —
but the two arches want opposite things: x86 *degrades* monotonically
with range count (60.6 at 21 ranges, 62.7 at 41, ~4% worse than
shipped), because its per-range top-k duplication costs more than the
cache sharing returns. No single pair wins on both, and the shared
optimum is just H7's block(32, 1024).

Arch-conditional constants could capture arm's extra 1.6%, but that is a
different and uglier change than a tile-ordering fix, and it would need
its own justification rather than riding along here. **NON-WIN.**
Streak 3.

### H9 — block-major tile order, re-measured to settle H7 (target: search) — **WIN**

H7's soak landed at x1.00998 against a x1.01 bar: under it by 0.002%, on
an effect that was plainly not noise. Rather than accept a coin-flip
verdict, re-measured at higher precision — 10 interleaved rounds x
reps=21, committed to in advance as the deciding run whichever way it
went.

| order | search-arm | search-x86 |
|---|---|---|
| quad (control) | 37.458 | 60.147 |
| block | 36.755 | 59.878 |
| ratio | x1.0191 | x1.0045 |
| sample range | quad 36.98–37.83, block 36.51–37.34 | quad 60.11–60.24, block 59.68–60.04 |

**HM x1.01175 — over the bar.** Stated plainly: the two soaks bracket
the threshold (x1.00998 and x1.01175, best estimate ~x1.011), so this is
a marginal ~1% win, not a comfortable one. What decides it in favour of
shipping is that the effect is consistently signed across 16 rounds on
two arches, and the change costs nothing — it is the order a `flat_map`
emits pairs in, with no extra work, no new state, and no correctness
surface beyond the parity gate.

Parity verified as for H5: bitwise identical to the old order on both
arches across all 40 result arrays (nq ∈ {1,4,25,100,257} x
k ∈ {1,10,100}, plus masked and tied-score shapes).

Final build, vs the recorded baselines:

| cell | baseline | now | ratio |
|---|---|---|---|
| search-arm | 40.394 | 37.065 | x1.090 |
| search-x86 | 61.740 | 59.718 | x1.034 |
| load_search-arm | 9.952 | 8.346 | x1.19 |
| load_search-x86 | 20.715 | 20.808 | x0.996 (cell noise) |

Non-target cells on the shipped build: insert-arm 2.819 (base 2.849),
delete-arm 3.093 (3.140), insert-x86 5.629 (5.530), delete-x86 7.749
(7.902). The one that reads slightly high, insert-x86, has spanned
5.530–5.629 across every measurement this session and cannot be affected
by this change on mechanism — insert searches with nq=1, which takes the
single-query path and never builds these tiles.

**Goal metric across the four cells (weights 3:1): WHM x1.067.**
Streak 0.

### H10 (probe-refuted) — size the pool by physical cores, not logical (target: search)

x86 search is 63% slower than arm despite AVX-512, and the boxes differ:
c3-standard-8 is 4 physical cores with 2 threads each, c4a-standard-8 is
8 real cores. `turbovec-python` builds its own pool from
`available_parallelism()` (logical), so 8 workers share 4 cores' SIMD
ports — plausibly contention rather than parallelism, and the thread
count is turbovec's own decision, not just an environment artifact.

Refuted, in the opposite direction — hyperthreads help:

| threads | search-x86 | search-arm |
|---|---|---|
| 8 | **59.68–59.90** | **36.76–37.20** |
| 6 | 60.28–61.01 | 48.57–49.17 |
| 4 | 61.15–64.15 | 71.38–71.94 |

Pinning to 4 would cost 2–7%. It also explains the arch gap with no
mystery left: x86 loses only ~2% going 8→4 threads, i.e. search
saturates at 4 physical cores, while arm scales near-linearly across 8
real ones (x1.32 at 6 threads, x1.93 at 4 — both essentially ideal).
The gap is core count, not a defect. **NON-WIN.** Streak 1.

### P6 — is the cost per query, or per pass over the code array?

Both kernels score QBS=4 queries per pass, so nq=100 makes 25 passes.
P1 had refuted widening that batch, but P1 tested *cache residency* —
whether the scan is bandwidth-bound — which is a different question from
how a pass's cost splits between shared and per-query work. Re-probed
properly at saturated nq, where parallel efficiency is not the variable
(an earlier small-nq attempt was confounded by an under-filled pool):

| nq | quads | search-x86 | search-arm |
|---|---|---|---|
| 93 | 24 | 57.191 | 34.790 |
| 97 | 25 | 59.583 | 36.176 |
| 100 | 25 | 59.822 | 37.347 |
| 101 | 26 | **62.283** | 37.490 |
| 105 | 27 | 64.999 | 39.602 |

x86 **steps with the pass count**: 97 and 100 sit within 0.4% of each
other on the same 25 quads, then 101 jumps 4.1% by adding the 26th.
That splits as ~2.38 ms per pass against ~0.08 ms per extra query — so
at nq=100 the passes are essentially the whole cell, and QBS 4→8 (25
passes → 13) predicts ~x1.5.

arm shows no quantum at all: a flat ~373 µs/query at every point,
per-query work dominating. So the same change predicts ~nothing there.

### H11 — carry two quads per tile, sharing each block's codes (target: search)

The cheap way to halve the passes without a new kernel: have a tile
carry 8 queries and call the existing 4-query kernel twice per block,
while that block's ~12 KB of codes are still in L1. Implemented for the
aarch64 dispatch (the x86 kernel loops over blocks internally, so the
trick does not apply there without changing the kernel itself).

arm, reps=15: **37.44–37.80 vs ~37.07 shipped — a regression.** Halving
the quad count also halves the tile count (175 → 91), coarsening exactly
the schedule granularity H5 won by fixing. Isolating that by restoring
the tile count via the cap:

| config | ranges | tiles | search-arm |
|---|---|---|---|
| shipped (4q, cap 1024) | 7 | 175 | 36.76–37.07 |
| 8q, cap 1024 | 7 | 91 | 37.58 |
| 8q, cap 512 | 13 | 169 | 36.26 |
| 8q, cap 256 | 20 | 260 | 36.26 |

With the schedule restored the 8-query tile reaches 36.26 — but that is
**exactly** what H8 measured for 4-query tiles at 21 ranges (36.260).
The entire gain is the finer range split, which x86 rejects (H8); the
pass-sharing itself contributes nothing on arm, precisely as P6 predicts.
`cargo test -p turbovec` green (446 passed). **NON-WIN — reverted.**
Streak 2.

P6 leaves a genuine, well-evidenced opportunity that this hypothesis did
not reach: an **8-query AVX-512 kernel** for x86, predicted ~x1.5 on
`search-x86` and ~nothing on arm, for HM ~x1.2. It needs a real kernel
rewrite (8 live accumulator sets against 32 zmm registers — the risk is
that it spills and gives the win back), not a loop restructure, so it is
the next substantial piece of work rather than something to bolt on here.

## Loop state

Streak 2 of 50. Two confirmed wins (H5, H9).
