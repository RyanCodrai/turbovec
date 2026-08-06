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

**Rig note (mid-climb):** the x86 box moved to **us-central1-c** after I
deleted it trying to enable a virtual PMU — which c3-standard-8 does not
support on either the v1 or beta API — and then hit a c3 stockout in
us-central1-a and -b. The move is benign: on the shipped build the new
host measures 59.934/59.963/59.955/59.990 (median **59.958**) against
59.878 on the old one, a 0.13% difference, so the recorded baselines
still apply and no cell needed re-baselining. It also means **no
hardware-counter attribution is available on this rig at any zone** —
every `perf` event reads `<not supported>` — which is why the x86
mechanism hypotheses below had to be settled by A/B rather than by
counters.

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

> **This reading was wrong, and P7 corrects it.** The flatness between 97
> and 100 is not evidence that per-query work is cheap: the x86 dispatch
> *pads* a partial quad (`pad_qi`) and the kernel loops `for qi in 0..4`
> unconditionally, so a quad carrying one real query does four queries'
> work. Queries 98–100 were already being paid for at nq=97. The
> conclusion drawn here — ~2.38 ms per pass against ~0.08 ms per query,
> and therefore ~x1.5 from QBS 4→8 — does not follow from this data.

arm shows no quantum at all: a flat ~373 µs/query at every point. arm's
dispatch has a real tail path (the single-query kernel per leftover
query) rather than padding, so its numbers mean what they appear to.

### P7 — the shared/per-query split, measured without the padding artifact

Re-ran the same sweep with the AVX-512 kernel's `qi` loops temporarily
bounded by the real query count instead of the padded 4 (probe only —
the epilogue still assumes 4, so this was never a shippable state):

| nq | quads | search-x86 |
|---|---|---|
| 93 | 24 | 56.541 |
| 97 | 25 | 58.658 |
| 100 | 25 | 60.211 |
| 101 | 26 | 61.216 |
| 105 | 27 | 63.680 |

Now the two axes separate. nq=97→100 holds the pass count at 25 and adds
3 query-scans: +1.553 ms, so a query-scan costs **~0.50 ms**. nq=100→101
adds one pass and one scan: +1.005 ms, so a shared pass costs
**~0.50 ms**. Fitting `cost = a·passes + b·queries` over all five points
gives a ≈ 0.52, b ≈ 0.49 — consistent.

At nq=100 that is 25 × 0.50 = 12.5 ms shared against 100 × 0.50 = 50 ms
per-query: the shared part is **21% of the cell, not 88%**.

### H12 (probe-refuted) — 8-query AVX-512 kernel (target: search)

QBS 4→8 halves the shared passes only, so P7 caps it at
12.5 → 6.25 ms, i.e. **~x1.066 on x86 and ~nothing on arm** (HM ~x1.03).
That would clear the bar if it came for free. It does not:

`accus` is already `[[__m512; 4]; 4]` — **16 zmm live across the whole
group loop** — plus codes/LUT/result temporaries, putting the 4-query
kernel near the 32-register ceiling already. That is almost certainly why
QBS=4 was chosen. Eight queries need 32 zmm for accumulators alone, so
the kernel would spill in its hottest loop, and the spill traffic would
plausibly cost more than the 6.25 ms on offer. A 256-bit single-block
variant keeps register pressure flat but doubles the shuffle instruction
count, trading the same win away from the other side.

Refuted on the arithmetic rather than built: a multi-hour unsafe-SIMD
rewrite whose entire upside is 6.25 ms, against a register budget that
says it will not survive the attempt. **NON-WIN (probe-refuted).**
Streak 3.

The general lesson worth keeping: on x86 this workload is ~80% per-query
scan work, and the per-query work is what the kernels already spend
their register budget on. Sharing more across queries is bounded at 21%
of the cell before it starts costing more than it saves.

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

P6 appeared to leave a large opportunity here — an 8-query AVX-512
kernel, predicted ~x1.5 on `search-x86`. P7 shows that prediction rested
on the padding artifact; the real ceiling is ~x1.066, and H12 records why
it is not worth taking.

### H13 — arch-specific tile granularity: a finer split for NEON only (target: search)

H8 found the two arches want opposite granularity and I set it aside as
"a different and uglier change". That judgement was too quick: the two
dispatches are already separate `cfg` bodies with separate kernels and
separate batch constants, so giving each its own tile target and cap is
the natural structure, not a hack. Implemented by passing the pair into
`n_block_ranges` (which already took `min_tile_blocks`), with NEON on
(64, 256) → 21 ranges and x86 unchanged on (32, 1024) → 7.

Re-checking H8's arithmetic first: its arm medians were 37.113 → 36.189,
x1.0255, which harmonic-means with an unchanged x86 to x1.0126 — over the
bar. That is what justified building it. But H8's numbers came from a
3-round sweep interleaved with three other configs across a box that
drifted, so the candidate needed its own A/B.

8 interleaved rounds x reps=21, one build, both configs behind an env
override so nothing but the constants differ:

| config | median | range |
|---|---|---|
| shipped (32, 1024) | 36.738 | 36.52–36.96 |
| H13 (64, 256) | **36.125** | 36.03–36.24 |

**arm x1.0170, x86 x1.0000 → HM x1.0084.** Under the bar.

The arm gain is not in doubt — every one of the 8 candidate samples beats
every one of the 8 control samples, and the effect reproduces H8 and the
H11 cap sweep, which both landed on ~36.2 by different routes. It is just
smaller than H8's noisier estimate (x1.017, not x1.0255), and a gain on
one arch alone harmonic-means to 0.84% across the pair. x86 was verified
untouched (59.999 / 60.252 / 60.069 against 59.878 shipped — inside its
noise band, and structurally guaranteed by `cfg(target_arch)`).

`cargo test -p turbovec` green (446 passed). **NON-WIN — reverted.**
Streak 4.

Worth flagging for whoever sets the next goal: this is the second real
arm-only improvement the pair-HM rule has discarded (H8 being the first
sighting of the same effect). A per-arch bar, or a rule that credits the
weighted objective rather than the per-hypothesis pair mean, would take
~1.7% on `search-arm` that is sitting there fully measured.

### H14 — vectorizable fast path for the 4-bit LUT build (target: search)

P3's standing anomaly: prep costs 12.0 µs/query on x86 against 4.9 on
arm, though x86 is the faster box elsewhere. Reading the builder explains
why it might not vectorize — at `bits == 4` each 16-entry sub-table is
just `q * centroids[0..16]`, but it is written as a nested loop whose
inner trip count is a runtime value and whose every entry carries a shift
and a mask. Replaced that with the straight scalar-times-vector form for
the 4-bit case.

Measured prep (N=32 so the scan is negligible), against P3:

| | P3 | H14 |
|---|---|---|
| x86 | 12.03 µs/query | 12.46 / 11.98 |
| arm | 5.31 µs/query | 6.10 / 5.62 |

**No change on either arch.** Either LLVM already vectorized the original
(the trip count is loop-invariant and small, so it can peel it), or prep
is dominated by the rotation rather than the LUT build. Since P2 bounds
all of prep at 1.3–2.0% of the cell, the question is not worth another
hypothesis either way. **NON-WIN — reverted.** Streak 5.

Worth recording as a near-miss on correctness: the first cut gated the
fast path on `codes_per_nibble == 1`, which is also true at `bits == 3` —
where `code_mask` folds the nibble into 0..8 and `centroids` holds 8
entries, so `centroids[..16]` panics. `cargo test` caught it
(`pipeline_self_score_is_unbiased`). The gate has to be `bits == 4`. A
4-bit-only benchmark would never have exercised it.

### H15 — arch-specific tile granularity, re-run under the search-only goal — **IMPROVEMENT (arm)**

H13 measured this and the old pair-mean rule rejected it. The goal now
targets search alone, so it was re-tested rather than assumed: the NEON
dispatch takes (64, 256) -> 21 block ranges, x86 keeps (32, 1024) -> 7.

Smoke (2 rounds): control 36.72/36.48 vs 36.03/35.87 - signal.

Soak, 8 interleaved rounds x reps=21, one build, both configs behind a
switch:

| | median | range |
|---|---|---|
| control (7 ranges) | 36.603 | 36.37-36.90 |
| H15 (21 ranges) | **36.127** | 36.00-36.32 |

**search-arm x1.0132**, every candidate sample below every control
sample. Third independent measurement of this effect (H8, H13, H15), all
landing on ~36.1-36.2.

Shape check - the win is not an artefact of the nq=100 tile count, and
is in fact larger at nq=10:

| shape | control | H15 | |
|---|---|---|---|
| nq=1 | 0.615 | 0.633 | separate single-query path; within noise |
| nq=10 | 4.522 | 4.238 | **x1.067** |
| nq=100 | 36.649 | 35.998 | **x1.018** |

Correctness: **bitwise identical across all 40 result arrays** (nq in
{1,4,25,100,257} x k in {1,10,100}, masked and tied-score shapes), both
arches. `cargo test -p turbovec` 446 passed, 0 failed.

**On the bar, stated plainly:** this clears 1% on the arch it targets
(x1.0132) but the joint harmonic mean of the two search cells moves
x1.0066, since x86 is untouched by construction (`cfg(target_arch)`).
Shipped because it is a strict improvement - one cell faster, the other
byte-identical and unchanged, nothing regressed at any shape - but the
joint number is recorded here so the distinction is not buried.

### P8 — what is the AVX-512 kernel's actual ceiling on this machine?

The previous climb's "kernels are port-saturated" was inherited, not
checked here, and my own arithmetic contradicted it: ~240M `vpshufb` at
nq=100 against a nominal 1/cycle port-5 budget looked like ~40%
utilisation, i.e. 2.5x of headroom. One of the two had to be wrong.

Measured it directly (`turbovec/examples/kernel_roofline.rs`): the
kernel's exact inner sequence — load codes, split nibbles, two LUT
shuffles, u16 accumulate — over an L1-resident buffer, so only issue rate
is under test.

| | shuffles/s per core |
|---|---|
| microbenchmark, single thread, L1-resident | **1.26 G/s** |
| real kernel (240M / (0.060 s x 4 physical cores)) | **1.00 G/s** |

**The kernel runs at 79% of what this instruction sequence can achieve on
this machine.** The previous climb was right and I was wrong: the error
was assuming `vpshufb` issues at 1/cycle. The microbenchmark reaches only
0.47 shuffles/cycle at 2.7 GHz, because the loop is bounded by the
surrounding loads, ANDs and adds — the paper port figure was never the
ceiling. The real number is 79%, and with HT contention on 4 cores the
true headroom is smaller still.

Consequences for where x86 effort can go:

* Micro-optimising this loop is capped at ~20%, and only by *removing*
  instructions — scheduling and layout cannot help a loop already at 79%
  of its own issue rate. This retires the whole class of x86
  microarchitectural hypotheses (LUT interleaving, broadcast hoisting,
  prefetch, unrolling), which is the class that had a 0-for-4 record here
  anyway.
* The shuffle count is algorithmically fixed. A 512-bit shuffle performs
  64 byte-lookups, and the algorithm needs `nq * N * (dim/2) * 2`
  lookups; no instruction selection changes that, including VBMI's
  `vpermb` (64-entry tables, still 64 lookups per instruction). Going
  faster on x86 means issuing fewer lookups — pruning blocks or filtering
  candidates — not a faster inner loop.

The same arithmetic on arm points the other way: ~960M `tbl` ops /
(0.0361 s x 8 cores) ~ 3.3 G/s/core ~ 1.1 tbl/cycle against NEON's 2/cycle,
so roughly **55%** — materially more headroom than x86, and consistent
with all three improvements so far having landed on arm.

### H16 (arithmetically refuted) — exact block pruning via a scale bound (target: search)

P8 said further x86 gains need *fewer lookups*, not a faster loop, which
points at pruning: skip a block when its provable upper bound cannot beat
the current k-th best. Sound and exact, using a stored per-block max
scale.

Refuted by arithmetic before building. With 384 groups each contributing
0..127, a block's integer sum concentrates at ~24,384 +/- 705, while the
bound — every group taking its maximum — is 384 * 127 = 48,768. The
top-10 threshold over 200k vectors sits near 27,344, far below the bound,
so it never fires. A partial-prefix variant is no better: bounding the
remainder after P groups gives P*63.5 + (384-P)*127, which only drops
below the threshold at P ~ 340 of 384 — pruning that begins after 89% of
the work is done. **NON-WIN (arithmetically refuted, unbuilt).**

The concentration is the whole problem: sums of many bounded terms are
tightly clustered, so any bound built from per-term maxima is loose by
~2x exactly where it needs to be tight.

### P9 — the NEON kernel's real ceiling, measured the same way

P8 left arm looking like the opportunity: ~1.1 `tbl`/cycle against
NEON's paper 2/cycle, i.e. ~55% and lots of headroom. But that is the
same species of paper-number inference that had just been wrong about
x86, so it was measured rather than believed
(`turbovec/examples/kernel_roofline_neon.rs`).

| | per core |
|---|---|
| microbenchmark, single thread, L1-resident | **4.00 G tbl/s** |
| real kernel (960M / (0.0361 s x 8 cores)) | **3.32 G tbl/s** |

**arm runs at 83% of achievable — slightly closer to its ceiling than
x86's 79%.** The "55%" was wrong for the same reason the x86 "40%" was:
the loop is bounded by its loads, ANDs, widening adds and accumulator
pressure, not by the table-lookup unit.

### Where this leaves the search cell

Both kernels now have a *measured* ceiling rather than an inherited
claim, and both sit at roughly 80% of it. That bounds inner-loop work at
~1.20x (arm) and ~1.27x (x86) even if the remaining stall were entirely
removed — and the residual is streaming 77 MB rather than an L1-resident
buffer, so the achievable part is smaller still.

The lookup count itself is fixed at `nq * N * (dim/2) * 2` by the
algorithm, independent of instruction selection: a 512-bit shuffle and a
NEON `tbl` both perform a fixed number of byte-lookups, and neither VBMI
`vpermb` (64-entry tables) nor SVE2 `TBL` on Neoverse V2 (2x128-bit, so
16 lookups per instruction as now) changes the count. H16 closes exact
pruning as the way to reduce it.

So the three improvements this climb found were all schedule-level
(H5, H9, H15), the schedule now measures at a joint peak, and the kernels
are within ~20% of a measured hardware limit.

### P10 — the ceiling measured while streaming, not from L1

P8/P9 compared the real kernels against an L1-resident microbenchmark,
which flatters nothing but is not the workload: the real scan streams
77 MB. Re-ran the same sequences over a full-size buffer.

| | L1-resident | streaming 77 MB | real kernel | real / streaming |
|---|---|---|---|---|
| x86 | 1.27 G shuf/s | 1.09 | 1.00 | **92%** |
| arm | 3.78 G tbl/s | 3.56 | 3.32 | **93%** |

Streaming costs 14% on x86 and 6% on arm relative to L1 — and against
that like-for-like ceiling both kernels run at **92–93% of achievable**.

This closes the kernel direction properly. The chain of estimates went
2.5x headroom (paper port numbers) → ~20% (L1 microbenchmark) → **~7–8%**
(streaming microbenchmark), and only the last one compares like with
like. Each refinement moved the answer toward "there is nothing here",
and the first two were wrong in the same direction for the same reason:
an assumed ceiling rather than a measured one.

What that leaves, honestly:

* Inner loop: ~7–8%, and that residual is the difference between a
  synthetic loop and one doing real top-k, masking and block bookkeeping.
* Lookup count: fixed by the algorithm (H16 closes exact pruning; P8
  closes instruction selection).
* Schedule: three improvements taken (H5, H9, H15), now at a joint peak
  across both arches.
* Prep: 1.3–2.0% of the cell, unimprovable in practice (P2/P3/H14).

Going materially faster from here needs a change this goal excludes — a
different format (fewer groups per vector), or accepting approximation.
Within the current design the search cell is within ~8% of a measured
hardware limit.

### H17 (sized, refuted) — halve the flush cadence on x86 (target: search)

`FLUSH_EVERY = 256` gives 2 flush batches per block at dim=768. arm cannot
raise it — its u8 pre-add allows 254/group, so 512 * 254 overflows u16
(the previous climb's H39) — but x86 accumulates u8 lookups straight into
i16 lanes, bounded by `FLUSH_EVERY * max_lut` = 512 * 127 = 65,024, which
fits. So x86 could run 1 batch instead of 2.

Sized without building the variant, by measuring the slope in the
direction that *is* safe: 128 gives 3 batches and 64 gives 6, so the cost
per extra batch predicts the saving from one fewer.

| flush | batches | search-x86 |
|---|---|---|
| 256 | 2 | 60.222 |
| 128 | 3 | 60.634 |
| 64 | 6 | 62.829 |

The 2→3 slope is **0.41 ms per batch**, so 2→1 is worth ~0.41 ms of
60.2 — **0.68%**, under the bar. (The 3→6 slope is steeper at 0.73
ms/batch, so extrapolating from the far end would have overstated it.)

It would also cost something real: arm and x86 currently flush on the
same boundaries, which is what makes their f32 accumulation round
identically. Diverging the cadence buys 0.68% and gives up cross-arch
numerical equivalence. **NON-WIN (sized, not built).** Streak 2.

### H18 (probe-refuted) — per-tile allocation churn (target: search)

H15 raised the tile count to 21 ranges x 25 quads = 525, and each tile
allocates its own heap vectors — roughly 7,000 allocations per search,
which at ~60 ns each would be ~0.4 ms of 36 ms (~1.2%), just over the bar.

Sized with no code change at all, by swapping the allocator: if malloc
traffic mattered, jemalloc would show it.

| | search-arm |
|---|---|
| glibc | 37.19 / 37.45 / 37.21 / 37.19 |
| jemalloc | 37.64 / 37.49 / 37.50 / 37.58 |

jemalloc is **0.7% slower**, not faster. A wholesale allocator swap moving
nothing in the favourable direction bounds the gain from removing those
allocations at below noise — the arithmetic overestimated because these
are small, same-sized, immediately-reused blocks that glibc's thread cache
serves without touching the general path. **NON-WIN (probe-refuted).**
Streak 3.

Corroborating evidence already on record: H15 tripled the tile count
(7 -> 21 ranges) and therefore tripled this allocation traffic, and still
came out 1.3% ahead.

### H19 — do the three shipped improvements generalise, or are they tuned-shape artefacts?

All three (H5, H9, H15) are *schedule* changes tuned at
N=200k / dim=768 / 4-bit / nq=100, and a tile schedule can win at one
shape by accident of its tile count. Measured the shipped build against
the pre-climb baseline (c8d7ec02) across a shape grid, same box, same
session, baseline rebuilt from a detached worktree.

arm, nq=100, k=10, median of 7:

| shape | baseline | shipped | ratio |
|---|---|---|---|
| 200k x 768 x 4b (tuned) | 42.252 | 36.841 | **x1.147** |
| 200k x 1536 x 4b | 88.200 | 77.994 | **x1.131** |
| 500k x 768 x 4b | 104.062 | 97.199 | x1.071 |
| 50k x 768 x 4b | 10.288 | 9.735 | x1.057 |
| 200k x 384 x 4b | 19.834 | 19.055 | x1.041 |
| 200k x 768 x 2b | 19.967 | 19.233 | x1.038 |

**No regression at any shape**, and the gain survives a 10x range in N,
a 4x range in dim, and a change of bit width. The tuned shape gains most,
as expected of tuned constants, but 1536-dim gains nearly as much and
nothing falls below +3.8%.

Also worth recording: the tuned cell reads **x1.147** here against a
baseline rebuilt and measured in the same session, versus the x1.12 this
log had been quoting against the original recorded baseline (40.394 ms).
The same-session figure is the trustworthy one — the recorded baseline
was taken on a different boot of the box weeks of wall-clock earlier.
The headline arm improvement is x1.147, not x1.12.

The same grid on x86, baseline rebuilt and measured on that box in the
same session:

| shape | baseline | shipped | ratio |
|---|---|---|---|
| 200k x 768 x 4b (tuned) | 61.943 | 59.555 | **x1.040** |
| 200k x 1536 x 4b | 120.700 | 116.708 | x1.034 |
| 200k x 768 x 2b | 31.965 | 30.847 | x1.036 |
| 200k x 384 x 4b | 31.283 | 30.477 | x1.027 |
| 500k x 768 x 4b | 152.557 | 148.949 | x1.024 |
| 50k x 768 x 4b | 17.366 | 17.251 | x1.007 |

**12 shapes across two arches, no regression anywhere.** x86 also reads
higher than previously quoted: x1.040 rather than x1.034.

Both headline numbers had been understated for the same reason — they
compared against baselines recorded on a different boot of boxes that
have since been stopped, restarted and (for x86) moved zone. Every
individual hypothesis in this log was judged by same-session interleaved
A/B precisely to avoid that, and the cumulative figure simply had not
been held to the same standard. Corrected:

**arm x1.147, x86 x1.040.**

### H20 — is H15's tile constant over-fit to the 8-core box?

H19 validated across index shapes but not thread counts, and the tile
target scales with `n_threads`. Interleaved A/B of the shipped NEON pair
against the previous one at 2, 4 and 8 threads, 3 rounds:

| threads | old (7 ranges) | new (21 ranges) | ratio |
|---|---|---|---|
| 2 | 158.5 | 145.2 | x1.09 |
| 4 | 77.2 | 71.4 | x1.08 |
| 8 | 39.5 | 36.7 | x1.08 |

Consistent across a 4x range of thread counts. Not over-fit. (A
validation, not a new improvement.)

### P11 — a legal layout for VNNI dot products on x86, and it is worth 1.54x

I had ruled out `vpdpbusd` (and ARM `udot`) as structurally illegal: they
reduce four adjacent bytes, and adjacent code bytes belong to four
different database vectors. That reasoning was right about the
instruction and **wrong about it being unfixable** — it is a property of
the layout, not the algorithm. Research surfaced the fix.

**The layout.** For a block of 16 vectors and 8 consecutive
subquantizers, store 64 bytes where byte `v*4 + j` holds vector `v`'s
codes for subquantizers `s0+j` (low nibble) and `s4+j` (high nibble).
Now each aligned 4-byte group belongs to ONE vector, so `vpdpbusd`'s
4-byte reduction sums four subquantizer contributions for that vector —
per-lane semantics preserved at dword granularity.

**Why `vpermb` is the enabler.** `vpshufb` applies one 16-byte table per
128-bit lane, so it cannot give byte position `j` a different LUT.
`vpermb` takes a 6-bit index, so `(j << 4) | code` selects from a 64-byte
table holding four consecutive 16-entry LUTs — which is the existing LUT
array unchanged, at the same cost per uops.info (1 uop, p5, 1/cycle,
identical to `vpshufb`).

Measured on the bench box, both sequences L1-resident over identical
work (`turbovec/examples/vnni_probe.rs`):

| sequence | time |
|---|---|
| current: 2x `vpshufb` + u8 add + 2x widening add | 0.079 s |
| proposed: 2x `vpermb` + 2x `vpdpbusd` | **0.051 s** |
| | **x1.539** |

Ops per query per 64 code bytes go 7.75 -> 4.75, with `vpermb` on p5 and
`vpdpbusd` on p0 — near-perfect port balance where today both the shuffle
and part of the widening contend for p5.

Three consequences beyond the op count:
* The u16 flush disappears. u32 accumulators cannot overflow at any
  realistic dim, so `FLUSH_EVERY` and its rounding go away (which also
  retires H17).
* Register pressure collapses: one accumulator per query per 16 vectors,
  against 16 live zmm today. This is what blocked H12's 8-query kernel.
* LUT entries could use the full 0..255 rather than the current 127 cap,
  since there is no u8 pre-add — slightly better score resolution.

**No on-disk format change is needed.** x86 already permutes the stored
layout at load time via `interleave_chunk_x86`; this is a different
in-memory permutation, built once per load.

**One thing it does change: results.** Today's scores accumulate through
a periodic f32 flush; exact u32 accumulation rounds differently — more
accurately, but not bit-identically. Every improvement in this log so far
has been bitwise-identical to its predecessor, so this is a deliberate
departure and is called out rather than buried.

### P12 — pricing the two kernel redesigns, measured rather than estimated

Research produced two candidate inner loops. Both were measured with the
same harness as P8/P10 (L1-resident, identical work), rather than trusted
as op-count estimates — the last three static estimates in this log were
all wrong in the optimistic direction.

**x86** (`turbovec/examples/vnni_probe.rs`):

| sequence | time | vs current |
|---|---|---|
| current: 2x `vpshufb` + u8 add + 2x widening add | 0.078 s | — |
| deferred: u8 accumulate, widen every 4 groups | 0.059 s | **x1.311** |
| vector-major: 2x `vpermb` + 2x `vpdpbusd` | 0.051 s | **x1.520** |

**arm** (`turbovec/examples/kernel_roofline_neon.rs`):

| sequence | rate | vs current |
|---|---|---|
| current | 3.78 G tbl/s | — |
| deferred: u8 accumulate, widen every 4 groups | 4.15 G tbl/s | **x1.098** |

The deferred-widening idea was predicted at ~1.5x. It is **x1.31 on x86
and x1.10 on arm** — a real asymmetry, and the predicted reason holds:
arm's `UADDW` is 2 ops issuing on all four V pipes, where x86 spends four
(and + shift + two `vpaddw`). My own op-count arithmetic for arm said
~1.11x, which was right; the 1.5x estimate was not.

What each costs:

* **Deferred widening** needs the LUT cap dropped from 127 to 31 so four
  groups' u8 sums cannot overflow. That raises LUT quantisation error
  from ~0.85% to ~3.4% of the score sigma — it reorders genuine near-ties
  only, but it is not bitwise identical. No layout or format change, and
  it also removes `FLUSH_EVERY` entirely (at cap <= 85 all 384 groups fit
  u16), retiring H17's machinery.
* **Vector-major + `vpermb`/`vpdpbusd`** costs no accuracy at all — u32
  accumulation is *more* exact than today's periodic f32 flush, and the
  LUT cap could rise to 255. It needs a different in-memory permutation
  (no on-disk change: x86 already permutes at load via
  `interleave_chunk_x86`), a new kernel, and runtime VBMI+VNNI detection
  with the existing kernel as fallback. It is x86-only — the arm analogue
  needs a 64-entry table, and `vqtbl4q_u8` at 2/3 throughput makes it
  strictly worse than the current NEON loop.

Neither is bitwise identical to the shipped build, which every
improvement in this log so far has been. That is the decision this log
cannot make on its own.

### P13 — the VNNI kernel validated at full scale, before integration

P11/P12 measured the sequence L1-resident over 4 KB. The real scan streams
76.8 MB, which P10 showed costs x86 a further 14%, and the microbenchmark
omitted the epilogue and top-k. So the claim was re-measured over a
full-size code array (`turbovec/examples/vnni_fullscan.rs`), single
threaded, both layouts, 4 queries:

| | time | |
|---|---|---|
| current (`vpshufb` + widening adds) | 0.011 s | |
| vector-major (`vpermb` + `vpdpbusd`) | **0.008 s** | **x1.388** |
| layout transform | 0.009 s | one-off at load, vs 0.011 s per scan |

Reproducible across runs (x1.387 / x1.388). The decay from x1.52 to
x1.388 is the memory system, exactly as P10 predicted — so x1.388 is the
honest figure to integrate against, not x1.52.

An unplanned correctness signal came out of it: the two scans' checksums
are **bit-identical** (38926883714) despite being computed over different
layouts with different accumulator widths (u16 pairs vs u32 lanes). With
the per-lane harness in `vector_major_check.rs`, the layout arithmetic is
now confirmed two independent ways.

Caveat carried forward: this scan is single-threaded and excludes the
per-block epilogue and top-k, which both paths share. Those dilute the
ratio, so the real cell should land below x1.388 — nearer x1.25-1.3 once
the ~21% non-kernel share of the cell (P7) is accounted for. On the x86
cell that is roughly 60 ms -> ~47 ms, moving x86 from x1.040 to about
x1.32 against the original baseline.

### H21 — vector-major layout + `vpermb`/`vpdpbusd` kernel on x86 — **IMPROVEMENT**

The change P11-P13 measured, now integrated and measured on the real cell.

**search-x86: 59.879 -> 48.576 ms, x1.233.** Eight interleaved rounds from
one build, every VNNI sample below every classic sample (47.67-49.87 vs
59.79-60.02). Against the original baseline that is **x1.271** (61.740 ->
48.576). The projection from P13 was x1.25-1.3.

arm is unchanged: the analogue needs a 64-entry table and `vqtbl4q_u8`'s
2/3 throughput makes it worse than the current NEON loop (P12).

**Correctness.** Determinism holds — repeated queries return byte-identical
results — and recall@10 against float ground truth is **0.7375 on both
paths, identical**. Top-k ids are identical, set agreement 1.0. Scores are
NOT bit-identical: max absolute difference 4.6e-05, because the kernel
accumulates exactly in u32 where the classic one rounds through f32 every
256 byte-groups. It is the more accurate of the two, and this is the
disclosed departure from bitwise stability that every earlier improvement
in this log preserved.

Full suite green on both paths (133 passed, 0 failed each) and on aarch64
(453 passed).

**What integration actually cost.** The native layout turned out to be an
implicit, file-wide assumption in six places, and making it a runtime
choice forced every one into the open. The tests found them in three
rounds — 17 failures, then 5, then 1:

* two *producers*: the loader (`seq_into_native`) and the encode path
  (`pack_blocked_native!`). Missing the second meant an index built by
  adding vectors was perm0 while one loaded from disk was vector-major.
* three *mutators*: `append_lanes`, `move_lane`, `zero_lane`.
* two *readers*: the scalar search fallback and the removal-capture path.

None would have failed loudly in production; all would have silently
mis-scored. They now all route through `read_code`/`write_code` and the
single `vnni_layout_for()` gate, which requires the CPU features *and* a
byte-group count divisible by 4. The permanent obligation this creates —
that anything touching native codes goes through those accessors rather
than indexing directly — is the real price of the x1.23, and is worth
weighing against it.

The last failure was a false positive: a one-time `env::var` in the gate's
`OnceLock`, charged to whichever allocation measurement ran first. Fixed by
warming the lazy state in the test rather than relaxing its assertion.

No on-disk format change: x86 already permuted the stored layout at load,
and this simply permutes it differently.

**Shape grid, x86, VNNI enabled by default** (the earlier H19 grid predated
the kernel and so validated the classic path only):

| shape | pre-climb | sched only | +vnni | total |
|---|---|---|---|---|
| 200k x 768 x 4b | 61.94 | 59.55 | **51.30** | x1.207 |
| 200k x 1536 x 4b | 120.70 | 116.71 | **95.65** | x1.262 |
| 500k x 768 x 4b | 152.56 | 148.95 | **125.66** | x1.214 |
| 200k x 768 x 2b | 31.97 | 30.85 | **25.77** | x1.241 |
| 50k x 768 x 4b | 17.37 | 17.25 | **14.24** | x1.219 |
| 200k x 384 x 4b | 31.28 | 30.48 | **26.00** | x1.203 |

Gains at every shape, no regression anywhere, across 10x in N, 4x in dim
and a bit-width change. 2-bit working confirms the divisibility gate
handles the halved byte-group count.

This grid builds its indexes through the *encode* path rather than by
loading files, so it independently exercises the producer that the first
integration attempt missed. Its tuned-cell figure (x1.207) sits below the
dedicated A/B (x1.271) because it uses 7 reps against 15 and a fresh index
per shape; the interleaved A/B is the more precise measurement.

### P14 — 8 queries per pass, unlocked by the VNNI kernel

H12 refuted a wider query batch because 8 queries needed 32 zmm of u16
accumulators against 16 already live. The VNNI kernel accumulates in u32,
one register per query per 16 vectors — 8 zmm at QBS=4 — so QBS=8 needs
16, which now fits. The constraint that killed H12 was a property of the
old accumulator design, not of the machine.

Probed at equal total query-work (`vnni_probe.rs`):

| | time | per query |
|---|---|---|
| VNNI, 4 queries/pass | 0.051 s | — |
| VNNI, 8 queries/pass | 0.085 s (2x the work) | **x1.20** |

Two reasons not to take x1.20 at face value:

* the probe shares one LUT register across both nibbles where the real
  kernel loads `tlo` and `thi` separately. Both probe variants do this, so
  the ratio is roughly fair, but the absolute figure is optimistic.
* halving the quad count halves the tile count, and the H5/H15 schedule
  constants are tuned for `n_quads = nq/4`. H11 showed on arm that exactly
  this interaction can swallow the whole gain — there, restoring the tile
  count revealed the pass-sharing itself was worth nothing.

So this is a promising lead requiring its own schedule retune, not a
drop-in x1.20. Recorded rather than built.

### H23 — 8 queries per pass on the VNNI kernel (target: search)

P14's probe measured x1.20 per query for 8 queries per pass, and the VNNI
kernel's u32 accumulators make it register-feasible where H12 could not.
Implemented and A/B'd from one build, 6 interleaved rounds:

| | median |
|---|---|
| 4 queries/pass | 50.593 |
| 8 queries/pass | 50.731 |
| | **x0.997 — parity** |

**NON-WIN — reverted.** The kernel is correct (133 tests pass at QBS=8);
it is simply not faster.

The reason is a conflict this log has now hit three times. Halving the
queries per pass halves the quad count (25 -> 13) and therefore the tile
count, coarsening the schedule. On arm, H11 could restore the tile count
by splitting the block axis further and the pass-sharing then showed as
worth exactly nothing. On x86 that escape does not exist: H6 measured x86
*degrading* as block ranges increase, so the granularity cannot be bought
back at any price. Pass-sharing and schedule granularity trade directly
against each other, and here they cancel.

Worth stating plainly: the probe said x1.20 and the cell said x1.00. The
probe measured the kernel loop in isolation, where halving passes is a
pure win; it could not see the schedule interaction, which is the whole
effect. That is the fourth time this session an isolated-loop estimate has
overstated a cell result (P12's x1.5 -> x1.31, P11's x1.52 -> x1.23,
deferred widening's claimed x1.5 -> x1.10 on arm).

### P15 — what a uniform codebook actually costs in recall

The largest remaining lever was replacing the shared Lloyd-Max codebook
with a uniform one, which turns the LUT lookup into a plain integer dot
product (`UDOT` / `vpdpbusd`) on both arches at the same 4 bits per
dimension — so no extra RAM. The literature priced the accuracy cost at
"MSE x1.21, about 0.15 bit", which is a claim about a Gaussian coordinate,
not a measurement of this pipeline.

Measured directly, with no kernel or encoder work: quantize the same
rotated unit-norm data both ways, score asymmetrically (exact query,
quantized database), compare recall@10 against exact float ground truth.
N=20k, dim=768, nq=200, k=10, 16 levels.

| codebook | recall@10 | coord MSE |
|---|---|---|
| Lloyd-Max | **0.8435** | 1.244e-05 |
| uniform | **0.8225** | 1.504e-05 |
| | **-0.0210** | **x1.209** |

The MSE ratio lands essentially on the predicted x1.21, so that estimate
was sound. The translation into *recall* is the number that matters
though, and 2.1 points is larger than "0.15 bit" makes it sound — about
2.5% of retrieval quality, permanently, on every query, to buy a claimed
(still unmeasured) ~2.5x.

Against the alternatives, with a +25% RAM increase ruled out by the
maintainer:

| | speed | recall | RAM | format |
|---|---|---|---|---|
| uniform-4 + UDOT | ~2.5x claimed | **-0.021** | same | re-encode |
| 1-bit prefilter | ~4x claimed | ~-0.02 | +25% | additive |
| 5-bit uniform | ~2.5x claimed | better | +25% | re-encode |

Uniform-4 is the only one respecting the RAM constraint, and it costs
about the same recall as the 1-bit prefilter while being slower and
requiring a format break. **Not recommended** — recorded with its measured
price so the decision does not have to be made on a literature estimate.

### H24 (probe-refuted) — deferred u8 widening (target: search, arm)

P12 measured deferred u8 accumulation at x1.10 on arm. Note it is now
arm-only: the x1.31 measured on x86 was against the classic kernel, and
the VNNI kernel that replaced it accumulates in u32 with no widening step
to defer.

It requires the per-entry LUT cap to drop from 127 to 31, so four groups'
u8 sums cannot overflow before widening. That quantizes the *query-side
lookup table* rather than the database codes — a different quantity from
P15's codebook, computed per query rather than baked into stored vectors,
so it was worth pricing separately rather than assuming it behaves the
same.

| LUT cap | recall@10 |
|---|---|
| 127 (current) | 0.8285 |
| 63 | 0.7750 |
| **31** | **0.6625** |
| 15 | 0.4170 |

**16.6 recall points for x1.10.** Eight times the cost of the uniform
codebook for a fifth of the speed. **NON-WIN (probe-refuted).** Streak 2.

Method caveat: this simulation uses one global LUT scale where
`build_query_neon_lut_from_slice` computes `max_span` per query, so it is
harsher than the real pipeline. The gradient is measured consistently
though and is steep — even 127 -> 63 costs 5.4 points — so the conclusion
does not depend on the absolute values. It also retroactively explains why
the code treats the u8 pre-add bound as binding: 127 is load-bearing, not
a conservative default.

**The inverse is a latent recall gain.** The VNNI kernel has no u8 pre-add
and accumulates in u32, so it could carry a cap of 255. Extrapolating this
curve upward, that is free accuracy on x86 — currently forgone to keep the
two arches numerically equivalent. Not a speed change, so out of scope for
this goal, but worth recording as an available improvement.

## Reference baseline (2026-08-06, both boxes same boot)

| cell | ms |
|---|---|
| search-arm | 36.740 |
| search-x86 | 49.847 |

Measured at `36052be4` after a clean rebuild on both boxes. Both are
~2% slower than the figures quoted from earlier boots (36.1 / 48.6),
which is boot-to-boot variation on shared GCP hardware, not a
regression. It is the reason every verdict here rests on a same-boot
A/B rather than a comparison against a recorded number: the 1%
improvement gate is smaller than the drift between boots.

## P18b — permute-dot shipped on x86: x1.359 MT, x1.111 ST, recall +1.35

The kernel from P18 wired into the real search path
(`search_multi_query_permute_dot` in `turbovec/src/search.rs`) and measured
end to end rather than on an inner loop. Baseline is branch head
`f35b1d1c`, so this multiplies with the four improvements already shipped
rather than replacing any of them.

Both extensions were built from one tree with one toolchain, differing only
by this change, and the `.so` was swapped between rounds so the two arms
alternate under identical machine conditions.

x86 (c3-standard-8), search cell, N=200k dim=768 4-bit nq=100 k=10,
reps=21, 6 interleaved rounds, medians:

| threads | base | permute-dot | |
|---|---|---|---|
| 8 (MT) | 50.24 ms | 36.96 ms | **x1.359** |
| 1 (ST) | 219.73 ms | 197.85 ms | **x1.111** |

Recall@10 against exact inner-product truth on the same 200k index, 200
queries: **0.8640 -> 0.8775**, +1.35 points. That matches
`permute_dot_recall.py`'s +1.3 prediction, which is the part worth keeping:
the accuracy model was right, so the gain is understood rather than a
property of this dataset.

MT gained far more than the P18 inner-loop microbenchmark predicted
(x1.144). The microbenchmark gave every query its own private LUT array and
so understated the win: in the real search the per-query LUTs are 12 KB
each and eight of them share cache with the code stream, which permute-dot
deletes outright — its per-query state is 768 bytes of int8 weights.

Two things fell out of the change rather than being designed in:

- **No accumulator flush.** `vpdpbusd` reduces into i32 and the widest
  possible sum over 768 dimensions is `768 * 255 * 127` ~ 2.5e7, three
  orders of magnitude inside i32. `FLUSH_EVERY` and the 7-bit LUT cap it
  forced both leave this path, so the held-off 127 -> 255 cap question is
  moot here rather than pending.
- **`avx512vbmi` is no longer required by the kernel** — `vpermb` is gone
  and `vpshufb` is plain AVX-512BW. The *layout* gate still asks for vbmi;
  widening it is a separate change and is not made here.

One test changed. `scalar_fallback_matches_simd_topk` asserted the scalar
fallback returns an identical top-k to SIMD; at 4 bits those are now two
different quantizations of the same score, so identity would have pinned
the LUT's rounding error as the specification. It now asserts what the test
was for — that the fallback still lands in the same neighbourhood (>=75%
slot agreement) and does not *beat* the kernel it stands in for, both
scored against exact float top-k.

Unrelated pre-existing failure, confirmed identical at `f35b1d1c`:
`allocation_hot_paths::repack_allocation_count_does_not_scale_with_vector_count`
(11 allocations at 4096 vectors against 0 at 64). Not touched here.

## P18c — permute-dot on arm, and the whole branch measured against main

The arm half needed the vector-major layout, until now x86-only. `SDOT`
reduces four s8 x s8 products into one 32-bit lane exactly as `vpdpbusd`
does, so both arches want the same bytes in the same order and now share
the layout, differing only in the kernel that reads it. `SDOT` goes in as
inline asm because `vdotq_s32` is still unstable (rust-lang/rust#117224).

**Everything below is against `origin/main` (`f6e8275d`)**, three arms
interleaved within each round — main, branch head, branch head +
permute-dot — so machine drift hits all three equally. Only the `.so` is
swapped; the python package is byte-identical across all three commits.
Search cell, N=200k dim=768 4-bit nq=100 k=10, reps=21, 6 rounds, medians.

| | main | head | +permute-dot | head vs main | total |
|---|---|---|---|---|---|
| x86 MT | 61.88 ms | 49.50 ms | 36.84 ms | x1.250 | **x1.680** |
| x86 ST | 240.55 ms | 217.70 ms | 192.14 ms | x1.105 | **x1.252** |
| arm MT | 41.67 ms | 36.47 ms | 30.50 ms | x1.143 | **x1.366** |
| arm ST | 312.20 ms | 312.33 ms | 245.41 ms | x1.000 | **x1.272** |

The arm ST column is the useful one for reading the rest: the previously
shipped arm work is a *pure* ST no-op (x1.000, not merely small), because
H5/H9/H15 are all rayon tiling changes and `n_block_ranges` returns 1 when
`n_threads == 1`. Permute-dot is the first arm change that touches the
scalar path at all, and it is worth more single-threaded (x1.273) than
multi-threaded (x1.196) — the reverse of x86, where deleting per-query LUT
traffic pays most when eight cores contend for it.

**Scores are now bit-identical across arches.** Verified, not asserted:
same md5 over the score bytes and the id bytes for a 20k x 768 index built
from a fixed seed on both boxes. Both kernels accumulate the same integers;
x86's +128 level-table bias is cancelled exactly by the accumulator seed,
in integers, before anything reaches f32. Previously the two diverged by
4.6e-05.

Recall@10 against exact inner-product truth, same box, same index file:
x86 0.8640 -> 0.8775, arm 0.7505 -> 0.7900. The two *baselines* differ
because the cached benchmark index on each box was built three weeks apart
by different code, so only the within-box deltas mean anything here; the
cross-arch recall comparison does not, and the parity check above is what
establishes the arches agree.

Extending the layout to aarch64 surfaced two latent bugs, both invisible
while arm's native layout happened to equal its stored layout:

- `write_to_writer` and the `.tvim` writer both short-circuited on "off
  x86 the cache is already sequential, write it straight out". That put
  native bytes in the file, which the loader then transformed again — 7
  v7 sync/delta failures. Both now ask whether the cache is vector-major
  rather than inferring it from the target arch.
- The layout gate keyed only on `n_byte_groups`, so it engaged at 2 bits
  too, where permute-dot does not apply: a nibble spans two dimensions
  there and the nibble -> level map stops being shared across them. That
  silently mis-scored 2-bit indexes. `vector_major_for` now takes `bits`
  and states the invariant — the layout is only ever *written* where a
  kernel exists to *read* it.

Both were found by test failures, not by review, and both are the same
class of mistake: an arch-shaped assumption stated as a comment rather
than as a predicate.

## H28 — x86 query batch 4 -> 8: x1.433 ST, x1.116 MT

Found by asking why arm ST was slower than x86 ST despite arm winning MT,
which turned into a question about how often each arch streams the code
array. Sweeping nq single-threaded answers it directly — time is a
staircase in nq, flat within a batch and stepping when a new pass is
needed:

| nq | 1 | 4 | 5 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| x86 | 6.02 | 7.75 | **13.66** | 15.61 | **21.37** | 23.28 | 31.01 |
| arm | 4.13 | 10.03 | 14.28 | 20.17 | 24.32 | 30.26 | 40.61 |

x86 steps at nq=5, 9, 13: the batch is 4. Reading off the steps, one pass
over the code array costs ~5.9 ms and each extra query inside a batch ~0.6
ms — so at nq=100 the 25 passes are ~147 ms of the 192 ms total.

**H23 measured a batch of 8 at parity (x0.997) and kept 4. That result was
voided by permute-dot rather than confirmed by it.** H23 ran against the
`vpermb` kernel, where every extra query in a batch cost a 128-byte LUT
load per byte-group; widening the batch bought fewer passes at the price of
proportionally more table traffic, and the two cancelled. Permute-dot's
per-query cost inside a batch is an 8-byte broadcast, so passes are now
nearly free to amortize and the trade is no longer a trade. The kernel
already accepted 8 (`nq.min(8)`); only the dispatch constant said 4.

Predicted from the staircase model: `13 * 5.9 + 100 * 0.6` = 137 ms.
Measured 137.15. Interleaved, 3 rounds, medians:

| x86 | batch 4 | batch 8 | |
|---|---|---|---|
| ST | 196.51 ms | 137.15 ms | **x1.433** |
| MT | 36.89 ms | 33.07 ms | **x1.116** |

Recall unchanged at 0.8775 and scores unchanged — batching changes how many
times the scan streams the array, not what it computes.

The general lesson is worth more than the constant: **a refuted hypothesis
is only refuted against the kernel it was measured on.** H23 was correct
when it was run. Nothing flagged it for retest when the cost model
underneath it changed, and it sat as a settled null for eight hypotheses.

Two leads left open:

- **arm shows no staircase at all** — near-linear at ~2.0 ms/query, so its
  4-query batch is barely amortizing. Whatever x86 gained here, arm has
  not yet. Worth finding out why before assuming it is the same fix.
- **x86 beyond 8** needs kernel work, not a constant: `acc` is
  `[[__m512i; 2]; 8]`, so 16 of 32 zmm at nq=8. 16 queries would need 32.

## H29 — the same batch widening on arm: refuted, x0.906 MT

H28's win on x86 does not transfer, and the prediction that it would was
wrong by more than the effect it predicted.

The reasoning was that arm's near-linear nq curve meant its 4-query batch
was under-amortizing the *unpack*, not the memory passes: per 16-byte code
register the kernel issues 5 shared ops (load, AND, SHR, two `TBL`) plus 2
`SDOT` per query, so 8 useful MACs of 13 ops at 4 queries against 16 of 21
at 8 — predicted x1.24. Independent-`SDOT` throughput measured 9.30 G/s
(3.11/cycle at 2.993 GHz) against the kernel's 4.80 G/s, so the headroom
was real.

Measured at 8 queries, interleaved, 3 rounds:

| arm | batch 4 | batch 8 | |
|---|---|---|---|
| ST | 251.19 ms | 253.54 ms | x0.991 |
| MT | 30.50 ms | 33.67 ms | **x0.906** |

**Cause: NEON has 32 vector registers and the accumulators alone want 64.**
A block is 32 vectors = 8 accumulator registers per query, so 8 queries
need 64 before a single weight, level table or code register. The op-count
model was right about the instruction mix and silently omitted the register
file; the spills cost more than the amortized unpack saved. x86 had room
for the same change because `vpdpbusd` accumulates 16 vectors per zmm — 2
registers per query, 16 at NQ=8, half of its 32.

This is structural, not an implementation detail: no scheduling of the
current accumulator layout makes 8 queries fit. A variant that could is
processing 8 vectors per pass instead of 32 (2 accumulators per query = 16
at NQ=8) with `SDOT`-by-element weights (one register per query per two
q4 units = 8), totalling 24 and leaving 8 for the unpack. Untested, and the
ceiling is still only the x1.24 above — so it is a rewrite of the inner
loop for a fifth, with spill risk that already bit once here.

Reverted to 4.

## H30 — arm: score the block in halves. x1.160 MT, x1.086 ST

H29's failure named the constraint, and the constraint applied to the
*shipped* kernel too, not only to the 8-query version that failed.

A whole block is 32 vectors = 8 accumulator registers per query. At NQ=4
that is 32 — the entire NEON register file — before the level table, the
mask, the code register, the two TBL results and 8 weight registers. The
shipped kernel was already spilling; H29 only made an existing problem
worse, which is why its op-count model predicted a gain and measured a
loss.

Scoring the block in halves holds 4 accumulators per query, 16 at NQ=4, and
leaves room for the rest of the working set. The two halves read disjoint
64-byte runs of each 128-byte vector-major unit, so every byte of the block
is still read exactly once across the pair.

Interleaved, 3 rounds, medians:

| arm | full block | halves | |
|---|---|---|---|
| ST | 252.08 ms | 232.02 ms | **x1.086** |
| MT | 30.98 ms | 26.70 ms | **x1.160** |

The op *ratio* gets slightly worse, not better — the weight registers
reload once per half, so useful-MAC share falls from 57% to 53% of issued
SIMD ops. It still wins, which is the evidence that spills were the binding
cost rather than something merely correlated with block width. Recall is
unchanged at 0.7900 and scores stay bit-identical to x86 (same md5), as a
pure scheduling change must.

Quarter-blocks are not worth trying: at 2 accumulators per query the
working set already fits at halves, so the only change would be reloading
the weights four times instead of twice.

## H31 — indexed SDOT weights: parity on its own, but the enabler for H32

`SDOT` by element takes its second operand from one 4-byte group of a
register, so a single register carries two byte-group quads of query
weights instead of one quad needing two broadcasts. That is 4 weight
registers at NQ=4 instead of 8, and one load instead of four — about 10%
fewer issued ops by count.

Measured, interleaved, 3 rounds: ST x0.998, MT x0.997. **Nothing.**

The reason is worth keeping: the weight setup was already hoisted out of
the inner `i` loop, so those ops issue on the load pipes and overlap with
SIMD work. They were never on the critical path, and removing work that is
not on the critical path buys exactly zero. The binding resource is SIMD
issue *inside* the `i` loop — 5 shared unpack ops per 8 SDOT at NQ=4.

Kept anyway, because halving the weight registers is what lets H32 fit.

## H32 — arm: 8 queries on quarter-blocks. x1.371 MT, x1.346 ST

The only lever H31 left was raising SDOT per code register, i.e. more
queries per pass — which is what H29 tried and lost to spills. The register
budget is what changed:

| | acc/query | acc at NQ=8 | weights | total |
|---|---|---|---|---|
| H29 (full block, broadcast weights) | 8 | 64 | 16 | **80** |
| H30 (halves, broadcast weights) | 4 | 32 | 16 | **48** |
| H32 (quarters, indexed weights) | 2 | 16 | 8 | **24** |

24 of 32, leaving room for the level table, mask, code register and the two
TBL results. Eight queries fit for the first time, and the useful-MAC share
of issued SIMD ops goes from 53% to ~70%: at NQ=8 a code register's 5
shared unpack ops carry 16 SDOT instead of 8.

Interleaved, 3 rounds, medians:

| arm | H30 | H32 | |
|---|---|---|---|
| ST | 230.78 ms | 171.46 ms | **x1.346** |
| MT | 26.65 ms | 19.44 ms | **x1.371** |

Recall unchanged, scores still bit-identical to x86 (same md5).

H29 was right about the direction and wrong about the budget. What made the
difference was not a better idea but counting the register file before
predicting, twice — once to explain H29's loss (H30), once to make the
same change fit (H32).

## P19 — SMMLA (i8mm) is available and 2.27x SDOT's MAC rate

Probe only; no kernel written. `SMMLA Vd.4S, Vn.16B, Vm.16B` multiplies a
2x8 int8 matrix by an 8x2 and accumulates a 2x2 int32 result — 32 MACs per
instruction against `SDOT`'s 16 — and Neoverse V2 has i8mm.

Measured on c4a, eight independent accumulators:

| | rate | MAC rate |
|---|---|---|
| SDOT | 9.30 G/s | 148.8 GMAC/s |
| SMMLA | 10.57 G/s | **338.1 GMAC/s** |

x2.27 on MACs, and SMMLA issues *faster* per instruction than SDOT, not
slower — so the wider instruction costs nothing in issue rate.

Its shape is the scan's shape: 2 queries x 2 database vectors x 8
dimensions. The mapping works out with no extra shuffling beyond one ZIP:

- **B operand** (2 vectors x 8 dims). After the TBL pair,
  `vzip1q_s8(vhi, vlo)` interleaves them into exact dimension order,
  because the high nibble is the even dim and the low nibble the odd. Bytes
  0-7 of the result are vector 0's 8 dims, bytes 8-15 vector 1's.
  `vzip2q_s8` gives vectors 2 and 3.
- **A operand** (2 queries x 8 dims). Needs `build_permute_dot` to store
  weights in dimension order rather than today's `[4 lo][4 hi]` split — a
  change to the per-query build, not to the kernel.
- **Op count per code register at NQ=8**: 7 shared (load, AND, SHR, 2 TBL,
  2 ZIP) + 8 SMMLA = 15 ops for 256 MACs, against today's 5 + 16 SDOT = 21
  ops for the same 256. ~x1.4 on issued ops.
- **Registers**: one accumulator per (query pair, vector pair) = 16 at NQ=8
  on quarter-blocks, plus 4 weight registers instead of 8. ~26 of 32, so it
  fits — the constraint that decided H29/H30/H32 is satisfied *before*
  writing anything this time.

The cost is a real rewrite: the 2x2 accumulator tile means the epilogue has
to scatter results across queries and lanes rather than storing a lane-major
vector, and the weight layout change touches the per-query build. Not
attempted here.

## H33 — SMMLA permute-dot kernel on arm: **x1.369 ST / x1.266 MT** (confirmed)

P19's design, built. Six interleaved rounds, reps=21, three arms alternating
within each round; the `sdot` and `smmla` arms are the *same binary* with
`TURBOVEC_NO_I8MM` picking the kernel at runtime, so nothing but the kernel
differs between them.

| | main | sdot (H32) | smmla | vs sdot | vs main |
|---|---|---|---|---|---|
| arm ST | 312.59 ms | 167.96 ms | **122.70 ms** | **x1.369** | **x2.548** |
| arm MT | 41.65 ms | 19.20 ms | **15.17 ms** | **x1.266** | **x2.746** |

Bit-identical output, not a precision trade: score md5 `5939c346...`, id md5
`940512f7...`, recall 0.8030 — the same values the SDOT kernel and x86
produce. `SMMLA` sums eight int8 products where `SDOT` sums four, and both
sums are exact in i32, so the arithmetic is the same arithmetic.

Two things made it work, both known in advance from P19:

- **The nibbles are already in operand shape.** `vlo`/`vhi` byte `4u+v` is
  vector `4k+u`'s dimension `2v+1`/`2v`, so one `vzip1q_s8(vhi, vlo)` lays
  vectors `4k` and `4k+1` down as eight consecutive dimensions each — the
  column-major 8x2 operand `SMMLA` wants — and `vzip2q_s8` does the same for
  `4k+2` and `4k+3`. Two ZIPs per code register replace 16 SDOT with 8 SMMLA
  at NQ=8.
- **The register budget was an input, not a post-mortem.** 16 accumulators
  (4 query pairs x 4 vector pairs on a quarter-block), 4 A-operands, level
  table, mask, five transients: ~26 of 32. H29 died on exactly this and was
  only diagnosed afterwards.

One thing did not come from the design and mattered: **hoisting the weight
reshape out of the block loop.** `SMMLA` has no indexed form, so its A
operand must be a full 16-byte register per query pair per quad, rebuilt
from the interleaved `weights` layout. Done per quarter that is ~4 extra ops
per code register and eats most of the win (op model x1.21); done once per
tile into a scratch buffer it is ~2 (x1.35). The measured x1.37 ST matches
the hoisted model.

MT gains less than ST (x1.266 vs x1.369), the same shape every arm-side
compute win in this log has shown: at 8 threads the kernel sits closer to
memory-bound, so removing issue slots returns less.

## P20 — the x86 op model never matched the emitted code

Three hypotheses about why x86 ST sat at 30% of its VNNI ceiling, all
refuted by measurement, and all built on an op count of the *source*:

- **Memory bandwidth.** Refuted. `st_roofline.py` sweeps N downward: cost
  per (query x vector) is flat from 38 MB (L3-resident) to 154 MB (firmly
  DRAM), 6.47 -> 6.76 ns on x86 and 6.09 -> 5.95 on arm. The hardware
  prefetcher keeps a sequential stream fed well below the DRAM roof. The
  ~7.5 vs 8.0 GB/s the two arches showed was coincidence, not a shared wall.
- **MAC saturation.** Refuted, and the opposite of true. `vnni_peak.rs`
  measures 5.96 G/s independent VPDPBUSD. **The rates below were corrected
  in P22 — the clock is 2.98 GHz, not 2.517, so the peak is 2.0/cycle and
  the kernel sat at ~0.76/cycle, 38% of peak.** The conclusion (the MAC
  units are not the constraint) is unchanged.
- **Top-k epilogue.** Refuted. `split_probe.py` fits ns against dim at fixed
  N, so the intercept is everything charged per block: `ns = 0.008655*dim +
  0.1545` on x86, 2% of total, and negative-to-zero on arm. k=1 and k=10
  cost the same (131.94 vs 132.92 ms).

The disassembly settled it in one read. The model said 76 uops per
byte-group quad; the machine was issuing roughly 176, because
`search_multi_query_permute_dot` took a runtime batch width, so `acc[qi]`
was a runtime index, so LLVM spilled all 16 accumulators to the stack and
never unrolled the query loop:

    cmp  %r8,%rbx / je ...          <- bounds check, per query
    mov  0x0(%r13,%r8,8),%r9        <- chase pds[qi]
    mov  0x8(%r9),%r9               <- chase .weights.ptr
    vmovdqa64 (%rdi),%zmm2          <- reload accumulator from stack
    vpdpbusd (%r9,%rcx,8){1to16},%zmm0,%zmm2
    vpdpbusd 0x4(%r9,%rcx,8){1to16},%zmm1,%zmm2
    vmovdqa64 %zmm2,(%rdi)          <- store it back

64 vector memory ops per quad doing no arithmetic. One theory did survive
contact: the weight broadcast folds into VPDPBUSD's `{1to16}` embedded
operand, so there was never a GPR round-trip to fix.

**The lesson is the entry.** Three hypotheses were modelling the
abstraction while measuring the machine, and only the disassembly
reconciled them. Read the emitted code before modelling a kernel's op mix —
especially when a measurement misses a model by 2x or more.

## H34 — x86 const-generic batch width: **x1.318 ST / x1.618 MT** (confirmed)

Give x86 the compile-time `NQ` arm has had since H32. A fixed-size
`[&QueryPermuteDot; NQ]` makes every `acc[qi]` a constant index, the query
loop unrolls, and the accumulators stay in zmm1..zmm15 — the reload/store
pair around every pair of MACs disappears. LLVM then hoists the weight
broadcasts across both block halves unprompted, which was the next
hand-optimization on the list.

Six interleaved rounds, reps=21, three arms alternating within each round:

| | main | spilled | H34 | vs spilled | vs main |
|---|---|---|---|---|---|
| x86 MT | 61.97 ms | 32.97 ms | **20.37 ms** | **x1.618** | **x3.042** |
| x86 ST | 242.01 ms | 136.29 ms | **103.42 ms** | **x1.318** | **x2.340** |

Bit-identical: score md5 `5939c346...`, recall 0.8030, unchanged from arm.

MT gains more than ST — the reverse of every arm result in this log. The
spill traffic hit L1/L2 per query, and at 8 threads that capacity is shared,
so deleting it is worth more under load.

**arm has had a const-generic NQ since H32 and x86 did not, and I built the
arm one for register-budget reasons without noticing it was also what forced
the unroll.** A structural advantage on one arch is worth checking against
the other explicitly; it will not announce itself.

This also puts an asterisk on **H28** (x86 batch 4 -> 8, x1.433). Eight
queries x two accumulators could never have stayed in registers, so that
result was measured entirely in the spilled regime. The right width now that
the accumulators are real registers is an open question — see the queue.

## P21 — the arm kernel has no mirror of H34

Queue item 2 after H34: read the emitted aarch64 the way P20 read the x86,
since a structural difference between the arches had just been worth x1.6.
There is none. The `SMMLA` q4 loop is clean:

    ldp  q24, q25, [x10, #-16]      <- both code registers, one instruction
    ldp  q29, q30, [x11, #-32]      <- A operands, paired
    and / ushr / and / ushr         <- nibble split, 2 per code register
    tbl x4                          <- shared level permute
    zip1 / zip2 x2                  <- B operands in dimension order
    smmla x16                       <- accumulators v0-v7, v16-v23
    subs / b.ne

No stack traffic in the loop at all: 16 accumulators in registers, 4 A
operands, level table in v9, mask in v8. The const-generic `NQ` is doing its
job. **Null result, and worth the twenty minutes** — it retires "arm has the
same bug" rather than leaving it as a standing maybe.

What the disassembly does say is where arm's remaining headroom is. Per q4
per quarter-block: 32 instructions, of which 16 are `SMMLA` — half the
instruction stream is unpack and operand movement. Against P19's measured
3.53 SMMLA/cycle ceiling the kernel runs at ~1.31/cycle, **37% of peak**,
which is the same shape x86 showed at 30% before H34 but without a spill to
explain it.

Raising that ratio means more queries per unpack, and the register file says
no: NQ=12 needs 24 accumulators on quarter-blocks. Eighth-blocks would cut
accumulators to `NQ/2 * 2` and fit NQ=12 in ~26 registers, but the A
operands then reload 8 times per q4 instead of 4, and the arithmetic comes
out at ~10% fewer ops per MAC — inside the noise this log resolves at, for a
full kernel rewrite. Not attempted; recorded so the next attempt starts from
the cost rather than rediscovering it.

## H35 — x86 batch width, re-tested with the accumulators in registers: null

H28 measured 4 -> 8 as x1.433, but entirely in the spilled regime where
register pressure was free, so it had to be re-run after H34. Five
interleaved rounds, four widths alternating within each round:

| | NQ=4 | **NQ=8** | NQ=12 | NQ=16 |
|---|---|---|---|---|
| x86 MT | x0.816 | — | x0.733 | x0.690 |
| x86 ST | x0.636 | — | x0.902 | x0.905 |

8 stays. It was right for the wrong reason before and is right for the right
reason now: at NQ=8 the kernel holds 16 accumulators plus the 16 broadcast
registers LLVM materializes, which is exactly 32 zmm. 12 and 16 spill
straight back into the H34 pathology.

## H36 — x86 half-blocks to unlock a wider batch: **refuted**

The direct transplant of arm's H30/H32. A block is 32 vectors = two 16-lane
accumulators per query, so a batch pins `2*NQ` registers; scoring one half
at a time holds `NQ`, which is what let arm widen from 4 queries to 8. The
two halves read disjoint 64-byte runs of each 128-byte unit, so no code byte
is read twice.

Against full-block NQ=8:

| | hb8 | hb12 | hb16 | hb24 |
|---|---|---|---|---|
| x86 MT | x0.811 | x0.860 | x0.864 | x0.566 |
| x86 ST | x0.802 | x0.980 | **x1.050** | x0.796 |

Half-blocks cost ~20% at equal width and the wider batches they permit do
not repay it. Only ST at NQ=16 beats the incumbent, by 5%, while giving up
14% MT — a net loss on the goal metric. Reverted.

Why it worked on arm and not here: arm's registers are 128-bit, so a
32-vector block needs *eight* accumulators per query and the batch was
genuinely register-starved. x86's are 512-bit and need two, so NQ=8 already
fits — the half-block split buys register room that was not the binding
constraint, and charges a second pass over the block's loop overhead and
weight broadcasts for it.

**An optimization is not portable across arches just because the kernels
are analogous.** H30/H32 transferred as an *idea* (H33 came from the same
register-budget thinking and won); this one transferred as a *mechanism* and
did not.

Caveat on the numbers: hb* was measured in its own sweep, so the comparison
to full-block NQ=8 is cross-run rather than interleaved. Within-sweep
ordering (hb16 best on ST, hb24 worst everywhere) is solid; the ~20% hb8
gap is far outside observed drift but is not an interleaved figure.

## H37 — re-tune the tiling now that a block is 1.6x cheaper: null

Same reasoning that made H35 worth running. `TILES_PER_THREAD` and
`MIN_TILE_BLOCKS` were swept before H34, when scanning a block cost 1.6x
what it costs now. Range count trades tail balance against duplicating a
`k`-entry heap per range, and H34 made the scan cheaper without touching the
heap, so the balance point should have moved toward *fewer* ranges.

It has not moved. Eight interleaved rounds, reps=21, one binary with
`TV_X86_CAP` selecting the count:

| ranges | median | min | max |
|---|---|---|---|
| **7 (shipped)** | **20.267 ms** | 20.044 | 20.485 |
| 13 | 20.343 ms | 20.314 | 20.383 |

13 is 0.37% *slower* and the distributions overlap. Shipped value stands.
arm's own knob was swept too (`TV_NEON_MULT` 16/32/64/128/256 -> 16.14 /
15.64 / 15.79 / 15.85 / 15.83): the shipped 64 and the best 32 differ by
0.95%, under the 1% bar and inside the noise established below.

**The smoke run produced a better noise estimate than the soak did.**
`MULT` 32/64/128/256 all resolve to the *same* 7 ranges — the
`min_tile_blocks` cap binds first — and measured 20.58 / 20.84 / 20.37 /
20.42. **Four provably identical configurations spanning 2.3%** at reps=11.
Nothing below ~2.5% is visible at smoke resolution, which is why every
result in this log goes through interleaved rounds; a 1% win, which the goal
counts, simply cannot be seen without the soak gate.

H37 also explains where that noise lives: at 7 ranges the spread is 2.2%
and at 13 it is 0.34%. The variance is the ragged final wave of a coarse
schedule, and more ranges buy predictability without buying throughput.

## H38 — GFNI affine shift on x86: **x1.056 MT / x1.020 ST** (confirmed)

The high nibble was `_mm512_srli_epi16(c, 4)` then `_mm512_and_si512(.., m0f)`
— two instructions, the AND needed only because a 16-bit shift drags
neighbouring bits into positions 4..7. A logical shift is linear over GF(2),
so `vgf2p8affineqb` does shift-and-mask in one op, per byte, with no
follow-up. One of 22 ops per half-quad, ~4.5%.

Six interleaved rounds, reps=21, three arms alternating within each round:

| | main | H34 | H38 | vs H34 | vs main |
|---|---|---|---|---|---|
| x86 MT | 61.92 ms | 20.48 ms | **19.40 ms** | **x1.056** (no overlap) | **x3.192** |
| x86 ST | 241.15 ms | 101.21 ms | **99.22 ms** | x1.020 (overlaps) | **x2.431** |

**Only the MT figure is claimed.** MT is the goal cell and its samples
separate cleanly; the ST arms overlap, so x1.020 is a median difference this
rig cannot resolve — consistent with H37's finding that nothing under ~2.5%
is visible without separation.

Matrix is `0x1020408000000000`: the identity is `0x0102040810204080`
(output bit `i` is `parity(A[7-i] & x)`), and shifting right by 4 keeps only
the rows mapping input bits 4..7 to output bits 0..3. Bit-identical on the
first run — score md5 `5939c346...`, recall 0.8030.

**GFNI and AVX-512 VNNI are different generations.** Cascade Lake has VNNI
without GFNI, so this cannot ride the gate that selects permute-dot itself.
The kernel is now generated twice from one macro, differing only in
`target_feature` and the high-nibble expression, chosen once by
`is_x86_feature_detected!("gfni")`.

`TURBOVEC_NO_GFNI` forces the baseline kernel. Without it that path is
unreachable on every machine this is developed or benchmarked on — both
bench boxes and the dev machine have GFNI — so it would be present but
untestable, which is how a fallback rots unnoticed. The suite runs green
through both (133/133 each) and the two produce identical bytes.

A note on the risk taken: a wrong affine matrix does not crash, it produces
plausible-but-wrong nibbles. This was only safe to attempt because the md5
parity check already existed to catch it.

## H39 — unroll the arm q4 loop by two: **refuted, x0.971 MT / x0.909 ST**

P21's disassembly showed the `q4` loop rolled — one quad's body ending in
`subs`/`b.ne`. Two instructions of overhead per quad, and worse, no
scheduling window: each quad's TBL and ZIP feed SMMLA in the same iteration,
so their latency cannot hide behind anything. Unrolling by two should let
quad `q4+1`'s loads issue while `q4`'s permutes are in flight.

It made things worse, cleanly and reproducibly. Eight interleaved rounds:

| | rolled | unrolled x2 | |
|---|---|---|---|
| arm MT | 15.534 ms (15.46-15.69) | 16.000 ms (15.97-16.14) | **x0.971** |
| arm ST | 126.197 ms (125.1-127.5) | 138.873 ms (137.1-139.3) | **x0.909** |

No overlap in either mode, so the regression is real, not drift.

**The obvious explanation is wrong.** I predicted the unroll would cost no
registers, and it did not: `objdump` shows *zero* stack references in the
unrolled body, so there is no spill. The kernel got 3% slower on MT and 9%
slower on ST while issuing strictly fewer instructions and touching no
memory it was not already touching.

Mechanism unidentified. Candidates not yet tested: the unrolled body may
exceed a fetch or loop-buffer window on V2; LLVM appears to have unrolled
past the requested factor (48 SMMLA per cluster where 32 was asked for),
so the emitted body is larger than intended; or the two-quad stride pattern
disturbs the prefetcher. Recorded as refuted with the mechanism open rather
than closed with a guess — the guess I did have was checked and was false.

Worth noting the asymmetry: ST regressed 3x harder than MT. Whatever binds
here binds less when eight threads are competing, which is the opposite of
a memory effect and points back at the core's front end.

## H40 — arm batch width, re-tested after SMMLA changed the budget: null

Same move as H35 on x86. H32 fixed the permute-dot batch at 8 for the
**SDOT** kernel, where each query held two accumulators on a quarter-block.
SMMLA holds two per query *pair*, so the register arithmetic that chose 8 no
longer applies and the width had to be re-derived rather than inherited.

Smoke, reps=11, one build with `TV_PD_QBS` selecting the width:

| QBS | 4 | **8** | 12 | 16 |
|---|---|---|---|---|
| arm MT | 20.41 ms | **15.51 ms** | 20.56 ms | 20.26 ms |

8 stays, and by a margin far outside the 2.3% noise floor — no soak needed,
which is what the smoke gate is for. Score md5 identical at all four widths.

The shape is the register file again, from both sides. At 4 the unpack
amortizes over too few queries. At 12 the accumulators alone are 24 (6 pairs
x 4 vector pairs) and at 16 they are 32, before A operands, level table or
mask — both spill, and both land back at roughly the cost of NQ=4.

So the two arches now agree on 8 for opposite reasons: x86 because 16
accumulators plus 16 broadcasts is exactly 32 zmm (H35), arm because 16
accumulators plus 4 A operands and the fixed pair leaves ~6 spare of 32.

## P22 — research: the V2 and SPR ceilings, and a measurement of mine that was wrong

Three research passes against vendor documentation and uops.info, after the
in-code ideas ran out. Two corrections to this log and one closed door.

### My x86 clock and peak figures were wrong

`vnni_peak.rs` recovers the clock from a dependent VPDPBUSD chain as
`rate x latency`, and I used latency 5. uops.info gives **6** for the
accumulator operand on Emerald Rapids (Raptor Cove, same port layout as
SPR; uops.info has no SPR column and its "ICL" column is *client* Ice Lake
with one 512-bit FMA — not applicable).

The tell was in the output all along: **2.37 VPDPBUSD/cycle is
architecturally impossible.** VPDPBUSD zmm is `1*p05`, TP 0.50 — exactly two
ports, so 2.00/cycle is a hard ceiling. With latency 6 everything
reconciles: clock = 0.50 x 6 = **2.98 GHz**, which matches GCP's published
"sustained all-core turbo 3.0 GHz" for C3, and the independent rate becomes
5.96 / 2.98 = **1.99/cycle**, i.e. the hardware maximum.

Corrected: x86 runs at ~3.0 GHz and the kernel was at **~0.76 VPDPBUSD/cycle
= 38% of peak**, not 30%. Also: **SPR has no fixed AVX-512 licence offsets**
(Chips and Cheese measured 3.8 GHz with 512-bit vectors; LLVM #102047
confirms), so the 2.517 GHz I attributed to licensing was never a real
effect — it was my arithmetic.

*A microbenchmark that returns an impossible number is still wrong when the
number is merely implausible rather than absurd.* The earlier `.rept` probe
returning 17 GHz got discarded instantly; 2.37/cycle sat in the log for
several hypotheses because it looked reasonable.

### arm: SMMLA peak is 4/cycle, and the roofline is 7.0 cycles

Arm Neoverse V2 SWOG Issue 3.0 (PJDOC-466751330-593177), Tables 3-16/3-18,
corroborated against LLVM's `AArch64SchedNeoverseV2.td`:

| group | latency | throughput | pipes |
|---|---|---|---|
| SMMLA / SDOT | 3 (1 accum) | **4** | all V |
| ZIP1/2, UZP, TRN, **TBX** | 2 | **4** | all V |
| **TBL** (1-2 table regs) | 2 | **2** | **V01 only** |
| AND/ORR/EOR | 2 | 4 | all V |
| **USHR** and all basic shifts | 2 | **2** | **V13 only** |
| SVE **TBL (Z-form)** | 2 | **4** | all V |
| SVE2 BDEP/BEXT | 6 | 1/2 | V1 only |

So P19's measured 3.53 SMMLA/cycle was **88% of a documented 4/cycle peak**,
not an implausible outlier — my suspicion of that number was misplaced.

The kernel's 28 V-µops per iteration give a roofline of 28/4 = **7.0
cycles**; measured is 12.2, so **57% of roofline**. TBL's `V01` restriction
and USHR's `V13` restriction are both provably *not* binding at this µop
count (V0+V1 have 14 slots in 7 cycles for 4 TBLs). Loads (2 cycles) and
front end (4.1 cycles) are slack too. **The gap is not pipe capacity** —
which also means H39's unrolling had no pipe-level reason to help, and its
failure is less surprising than it looked.

Two documented substitutions that move TBL off `V01` to all four pipes:
SVE `TBL Zd.B, {Zn.B}, Zm.B`, and Neon `TBX`, which is bit-identical to TBL
whenever every index is 0-15 — guaranteed here by the `AND #0x0f` and the
`USHR #4`. Neither is expected to pay while TBL is not binding; recorded for
when the µop count drops far enough that it is.

### x86: what is actually available

- **VPSHUFB zmm is `1*p5`, 1/cycle** — and so is VPERMB, with worse latency.
  Every 512-bit byte shuffle is port-5-only. VGF2P8AFFINEQB (H38) is `1*p0`,
  so it did not relieve p5; it moved work off it.
- **Port model per 64-byte code register at NQ=8**: 20 p0/p5 µops -> >=10
  cycles, ceiling 1.6 VPDPBUSD/cycle. **17 load µops** (1 code + 16 query
  broadcasts) -> >=8.5 cycles on two 512-bit-capable load ports. One load
  per MAC is the structural problem, not the shuffles.
- **GFNI cannot replace the LUT.** VGF2P8AFFINEQB is a GF(2)-*affine* map
  per byte; an arbitrary 16-entry Lloyd-Max codebook is not GF(2)-affine.
  It could be skipped entirely only for a codebook of the form
  `level[i] = a*i + b` — which is exactly the uniform codebook **P17 closed
  at ~2 recall points**. That door is shut from both sides now.
- **AMX is impractical here, for reasons other than the one I assumed.**
  LDTILECFG's 204 cycles amortizes fine. The real blocks: TDPBSSD issues on
  **port 5**, the same port as the unpack, so they cannot overlap; there is
  **no register path between tmm and zmm**, so a 4-bit kernel must unpack in
  AVX-512, store to scratch, and TILELOADD it back; and AMX **power-gates**,
  with TDPBSSD latency measured rising from 50 cycles warm to 20 000 cold
  and requiring >=6% utilization to stay warm (Kalyanapu et al., IEEE CAL
  24(1) 2025). Break-even is driven by *query batch size* — below ~16
  queries it never pays, at 32+ the matmul saturates — so it is not a
  fit for nq=100 in batches of 8.

## P23 — pricing the ZIPs before rebuilding the layout to remove them

P22's research found that KleidiAI's i8mm int4 kernel emits **zero ZIPs**:
it pairs dimension `k` with `k+16` inside a byte rather than adjacent dims,
so mask-and-shift alone yields SMMLA-ready dimension runs. ggml pays the
same ZIPs turbovec does and its documented fix is the same repack.

Before building that, price it. The probe deletes the two ZIPs and feeds
`vhi`/`vlo` straight to SMMLA — **wrong results on purpose**, same loads,
TBLs and SMMLAs, only the reorder gone:

| | with ZIPs | ZIPs deleted | ceiling |
|---|---|---|---|
| arm ST | 124.94 ms | 111.75 ms | **x1.118** |
| arm MT | 15.44 ms | 13.69 ms | **x1.128** |

~12%, against a roofline prediction of x1.167 for dropping 4 of 28 V-µops.
Worth building.

**The first run of this probe was void and nearly got logged.** The patch
assertion failed on indentation, the deploy ran anyway, and both arms were
the same binary — which produced five rounds agreeing to within 0.5%, a
result indistinguishable from a genuine null. The rerun counts `zip`
instructions in both binaries (1435 vs 1427, exactly the 8 in the two
kernels) before measuring. *Verify the arms differ before trusting a null;
"no effect" and "the experiment did not happen" look identical.*

### Why the ZIP cannot be removed without moving bytes

Worth recording so this is not re-litigated. SMMLA reads bytes 0-7 of its B
operand as column 0 and 8-15 as column 1, and each column must be **one
vector's** eight dimensions. The vector-major unit currently puts 4 vectors
x 4 byte-groups in a 16-byte register, so `vhi` holds
`[v0d0,v0d2,v0d4,v0d6, v1d0,...]` — bytes 0-7 straddle *two* vectors. No
permutation of the A operand fixes that, because A/B pairing is free only in
the dimension index, not across columns. Re-pairing which dims share a byte
does not help either: at 4 vectors per register the result still needs one
ZIP per B operand, just at 32-bit granularity.

The fix is 2 vectors x 8 byte-groups per 16-byte register, with byte
`8u+v` holding dim `base+v` high and `base+8+v` low. Then `vhi` *is* the B
operand for dims 0-7 and `vlo` for dims 8-15, with no reorder at all.

**This is not a file-format change.** turbovec stores sequential,
arch-neutral bytes and builds the vector-major arrangement at load time via
`pack::native_transform`, so the layout that feeds the kernel is already an
in-memory-only concern. Memory cost is zero — still two codes per byte.

The cost lands on x86 instead: `vpdpbusd` wants four consecutive byte-groups
of one vector per dword lane with lane `i` = vector `i`, and the new
arrangement puts each vector across two lanes. Either x86 gains a
lane-pair reduction in its epilogue, or the native transform becomes
arch-specific — which the sequential on-disk format permits, and which
`vector_major_for` is already shaped for.

### No hardware counters on either box

`perf` installs on the arm box but every event reads `<not supported>`,
matching the x86 box. Confirmed for both arches: nothing in this climb can
be attributed by PMU, which is why every mechanism question here is settled
by A/B or by disassembly.

## H41 — the `vm8` layout: **x1.060 MT, ST at parity** (confirmed)

P23's design, built. Two vectors x eight byte-groups per 16-byte register
instead of four x four, so after the nibble split the TBL output *is* an
`SMMLA` B operand and the two ZIPs disappear. Not a file-format change:
turbovec stores sequential arch-neutral bytes and builds the native
arrangement at load, so this is `pack::native_transform` and the kernel.
aarch64-with-i8mm only — see below for why x86 cannot take it.

Eight interleaved rounds, reps=21, three arms alternating:

| | main | 4-group | **vm8** | vs 4-group | vs main |
|---|---|---|---|---|---|
| arm MT | 42.40 ms | 15.413 ms | **14.547 ms** | **x1.0595** (no overlap) | **x2.915** |
| arm ST | 321.87 ms | 124.723 ms | **125.120 ms** | x0.9968 (overlaps) | **x2.573** |

Bit-identical, and verified through the *write* path this time: the cached
parity index was deleted first, and the file md5 came back `fd476bbb...`,
identical to pre-vm8. A vm8 machine still writes the same stored bytes as
any other. Score md5 `5939c346...`, recall 0.8030.

### It took two cuts, and the first one failed the same way three earlier
### hypotheses did

The first cut kept quarter-blocks. It measured **+3.0% MT / -4.1% ST** —
a regression on ST, which fails the bar regardless of MT. `objdump` said
why: zero ZIPs in the hot loop, exactly as designed, but

    stp q6, q3, [sp, #864]     <- accumulators to the stack, in-loop
    ldp q16, q4, [sp, #864]
    movi v6.16b, #0xf          <- mask rematerialized three times

`vm8` needs **two** A operands per query pair, the even-dim and odd-dim
halves, where the 4-group kernel needed one. At NQ=8 that takes the A
registers from 4 to 8, and 16 accumulators on top overflows 32. Eighth-
blocks halve the accumulators to 8; 8 + 8 plus level table, mask and three
transients fits, and ST recovered from -4.1% to parity.

**Third time this session the register file was the real constraint**
(H29, H36, now H41) and the second time removing instructions was paid back
in spills. The op count is never the whole budget: *deleting work from the
inner loop only helps if the working set still fits.*

P23's probe priced the ZIPs at x1.12 and the shipped kernel realizes x1.06
on MT and nothing on ST. The probe measured the ceiling of removing two
instructions; it could not price the two extra A-operand registers their
removal requires, because the probe kept using the old A operands. **A
speed-of-light probe bounds the win, it does not predict it** — it deletes
a cost without paying for what makes the deletion legal.

### Why x86 keeps the 4-group layout

`vpdpbusd` reduces four bytes per dword lane and the epilogue assumes lane
`i` is vector `i`. Under `vm8` each vector spans two dword lanes, which
doubles the accumulators per query from 2 to 4 — at NQ=8 that is 32 zmm
before the level table, and H35/H36 already showed what that costs. The
sequential on-disk format is what makes an arch-specific in-memory layout
safe here.

## H42 — arm single-query was 1.5x SLOWER than main, and nobody measured it

**Found only because the goal changed.** Nine improvements shipped on nq=100
evidence alone. At nq=1 the arm branch was *worse than main*:

| nq=1 | main | branch (H41) | |
|---|---|---|---|
| arm MT | 0.628 ms | 0.917 ms | **x0.68** |
| arm ST | 4.107 ms | 6.185 ms | **x0.66** |
| x86 MT | 2.458 ms | 1.160 ms | x2.12 |
| x86 ST | 9.439 ms | 5.261 ms | x1.79 |

x86 carried over because `vpdpbusd` takes its width from the *vector*
dimension — 16 database vectors per instruction — so one query still fills
the machine. `SMMLA` takes half its width from the *query* dimension, and at
nq=1 that half is empty.

Bisected across the shipped builds at nq=1 ST: main 4.15, pre-permute-dot
head 4.13, H33 (SMMLA) 6.28, H41 (vm8) 6.27. The regression arrives with the
SMMLA kernel and vm8 is innocent.

### The cause was not the wasted MACs

A lone query rides SMMLA as a duplicated pair, so lanes 2/3 of every tile
repeat lanes 0/1 and half the MACs are discarded. That is *not* what cost:
per byte of codes the batched and single paths issue the same instructions,
which is exactly why the instruction-count model said they should be equal
and the clock said otherwise.

The cause is **instruction-level parallelism**. `SMMLA` has latency 3
(1 on the accumulator, per the V2 SWOG) and needs several independent chains
to stay fed. The batched kernel holds 16 accumulators — 4 query pairs x 4
vector pairs — and saturates. Routing one query through it at `NQ=2, NP=1`
leaves **two chains**, and the loop goes latency-bound.

With a single query pair the register budget is nearly empty, so the fix is
a dedicated kernel holding all 16 vector-pair accumulators of the block:
16 chains, 18 of 32 registers.

| nq=1 arm ST | main | H41 | **H42** |
|---|---|---|---|
| | 4.147 / 4.197 / 4.147 | 6.239 / 6.327 / 6.186 | **4.141 / 4.165 / 4.188** |

Parity with main restored. nq=100 is untouched by construction and measures
so: 14.50/14.47/14.58 before, 14.48/14.42/14.54 after.

**Two lessons, and the second is the expensive one.**

*Register pressure and ILP pull in opposite directions.* Every arm hypothesis
in this log (H29, H30, H32, H36, H41) was about fitting the working set into
32 registers, and the answer was always "hold fewer accumulators". At nq=1
the constraint inverts: registers are free and accumulators are the scarce
resource, because they are the only source of independent chains. The same
number that was too big at NQ=8 is too small at NQ=1.

*A metric that omits a shape will not protect it.* The old goal weighted
`{search, load_search}` 3:1 and every hypothesis here was measured on the
batch shape alone, so a 46% regression on the other one survived nine
consecutive "confirmed improvements" — each of which was correctly measured,
bit-identical, and genuinely better at what it was measured on. Nothing
caught it because nothing looked. The goal now weights the two shapes
equally and requires both to be measured per hypothesis.

## P24 / H43 — x86 nq=1 leaves cache badly; prefetch does not fix it

**P24.** Sweeping N at nq=1 splits the two arches:

| N | codes | arm ns/(q*vec) | x86 ns/(q*vec) |
|---|---|---|---|
| 100k | 38 MB | 20.92 | **14.51** |
| 200k | 77 MB | 21.15 | **25.59** |
| 400k | 154 MB | 20.80 | **31.45** |

arm is flat at 18-20 GB/s from 2 MB to 154 MB — compute-bound at every size.
x86 degrades **2.2x per vector** once the array leaves cache. The same sweep
at nq=100 was flat on both (P20): eight queries per pass give the memory
system time to keep up, and at nq=1 there is almost no arithmetic per byte,
so the load stream is fully exposed. The bound this log refuted at nq=100 is
real at nq=1 — *a refutation is scoped to the shape it was measured at*, the
same trap H23 set on x86 and H42 set on arm.

**H43 — software prefetch: refuted.** Two implementations:

- Whole next block issued up front (192 lines): **-35% at nq=100** (19.3 ->
  26.1 ms) and worse at nq=1 too. Floods the load ports; a strawman.
- One line per 64-byte load, 8 quads ahead — matching the consumption rate:
  **neutral at both shapes.** nq=1 ST 5.98/5.89/6.10 against 5.90/6.08/5.88;
  nq=100 MT 19.87/19.98/19.85 against 20.07/19.78/19.89. Overlapping.

The hardware prefetcher already has this stream. Software prefetch adds no
memory-level parallelism it was not already extracting, so the nq=1 cliff is
a latency/bandwidth limit rather than a prefetch-distance problem. Reverted.

Worth keeping: **the first implementation would have "refuted" the
hypothesis for the wrong reason.** Prefetching 192 lines per block tests
"does flooding the load ports hurt", not "is the load stream exposed". The
competent version had to be measured before the idea could be discarded.

## Metric change — 8 cells, harmonic mean of per-cell speedups

After H42 found a 46% arm nq=1 regression that nine "confirmed
improvements" had not looked at, the goal was rewritten:

> {arm, x86} x {ST, MT} x {nq=100, nq=1}, all 8 cells weighted equally.
> Score is the harmonic mean of the 8 per-cell speedups against
> origin/main. Test every hypothesis at all 8 cells.

Two properties are load-bearing, both learned the hard way here:

- **Harmonic, not arithmetic.** A regressing cell contributes a large `1/s`
  and drags the score down instead of being averaged away by seven wins.
  arm nq=1 at x0.68 costs ~0.12 on the harmonic mean and only ~0.04 on the
  arithmetic one.
- **Speedups, not raw times.** The cells span 0.6 ms to 320 ms; a mean over
  absolute times is decided almost entirely by the smallest cell. Each cell
  is normalized against main first, which is what makes "weighted equally"
  true rather than nominal.

`cells.py` measures one box's four cells and `score_cells.py` computes the
figure and flags any cell below 0.99x.

### Baseline: where the branch stands

| cell | main | now | speedup |
|---|---|---|---|
| arm nq100 MT | 42.18 ms | 14.48 ms | x2.912 |
| arm nq100 ST | 318.65 ms | 124.96 ms | x2.550 |
| **arm nq1 MT** | 0.639 ms | 0.647 ms | **x0.988** |
| **arm nq1 ST** | 4.206 ms | 4.151 ms | **x1.013** |
| x86 nq100 MT | 61.98 ms | 19.45 ms | x3.187 |
| x86 nq100 ST | 243.11 ms | 101.27 ms | x2.401 |
| x86 nq1 MT | 2.452 ms | 1.192 ms | x2.056 |
| x86 nq1 ST | 9.479 ms | 5.409 ms | x1.753 |

**Harmonic mean x1.7691** (arithmetic x2.1074).

The metric names its own next target without argument: the two arm nq=1
cells are the only ones near parity, and moving them from ~1.0 to ~1.75
would take the score from 1.769 to **2.18** — a 23% gain from two cells.
Nothing else on the board is worth a fifth of that. Under the old nq=100
metric these two cells were invisible; under this one they are the whole
opportunity.

## H44 — fewer accumulators to stop the nq=1 spill: **refuted, x0.69**

The arm single-query kernel spills: `objdump` on the H42 build shows **13
`stp` against 20 SMMLA** in the loop window. LLVM fully unrolls the 16-way
inner loop and hoists its sixteen code loads, so sixteen accumulators plus
sixteen code registers overflow the file. Removing the spill should recover
the gap to main.

Halving to eight chains over two half-block passes made it far worse:

| nq=1 arm ST | main | H42 (16 acc, spilling) | H44 (8 acc, clean) |
|---|---|---|---|
| | 3.948 / 3.974 / 4.004 | 3.999 / 3.975 / 3.984 | **5.904 / 5.737 / 5.702** |

x0.69 — a bigger loss than the spill ever cost. Reverted.

**The spill was the lesser evil, and the ILP arithmetic says why.** SMMLA is
latency 3 on four pipes, so saturating it needs ~12 independent chains in
flight. Sixteen accumulators clear that with room; eight do not, and the
loop goes latency-bound exactly as H42's two-chain version did. The stack
traffic costs less than the stalls it would have removed.

So the arm nq=1 kernel sits between two walls: below ~12 chains it is
latency-bound, at 16 chains it spills, and there is no count in between that
both fits the register file and feeds four pipes. That is a genuine
structural limit of SMMLA at one query, not a tuning miss.

Consistent with the instruction-density argument: at nq=1 the SMMLA kernel
and the classic LUT kernel both achieve ~4.57 vector-dims per instruction
(7 instructions per 16-byte register, 32 vector-dims), which is why they
measure within 1% of each other. Beating main at nq=1 needs a kernel with a
*different* density, not a better-tuned version of this one.

## H45 — cap the nq=1 load pressure, keep the chains: **HM x1.769 -> x1.851**

H44 concluded the arm nq=1 kernel "sits between two walls: below ~12 chains
it is latency-bound, at 16 chains it spills, and there is no count in
between". **That conclusion was wrong**, and both measurements it rested on
were right. It assumed the only knob was chain count. There is a second one:
*accumulator* pressure and *load* pressure are separable.

Sixteen accumulators stay live — that part H44 got right — but the sixteen
code loads are consumed four at a time instead of being hoisted together, so
peak pressure is 16 + 4 rather than 16 + 16. The spill goes away and the ILP
stays.

| nq=1 arm ST | main | H42 | **H45** |
|---|---|---|---|
| | 4.024 / 3.992 / 3.967 | 3.964 / 3.955 / 3.960 | **3.570 / 3.627 / 3.634** |

Full 8-cell score, bit-identical throughout (`5939c346...`), 127/127 green:

| cell | main | now | speedup |
|---|---|---|---|
| arm nq100 MT | 41.428 ms | 14.323 ms | x2.892 |
| arm nq100 ST | 310.760 ms | 121.921 ms | x2.549 |
| **arm nq1 MT** | 0.602 ms | 0.545 ms | **x1.105** (was x0.988) |
| **arm nq1 ST** | 4.074 ms | 3.636 ms | **x1.121** (was x1.013) |
| x86 nq100 MT | 61.978 ms | 19.449 ms | x3.187 |
| x86 nq100 ST | 243.108 ms | 101.269 ms | x2.401 |
| x86 nq1 MT | 2.452 ms | 1.192 ms | x2.056 |
| x86 nq1 ST | 9.479 ms | 5.409 ms | x1.753 |

**Harmonic mean x1.8506**, up from x1.7691. The batch cells are unchanged —
the single-query kernel shares no code with them — and both arm nq=1 cells
beat main for the first time on this branch.

**The lesson is about the shape of a refutation, not about registers.** H44
measured two points, 8 chains and 16 chains, and drew a line through them:
"there is no count that both fits and feeds four pipes." The data supported
that. What it did not support was the unstated premise that chain count was
the only axis — and the fix moves along a different one entirely, changing
*when* the loads happen rather than *how many* accumulators exist. A
refutation is only as wide as the axis it was measured on; H23, H42 and now
H44 have all been narrowed the same way.

## P25 — the arm unpack is already the industry sequence; H46 — group size is tuned

**P25 (research).** Checked against LLVM's Arm-authored
`AArch64SchedNeoverseV2.td`, KleidiAI, llama.cpp and XNNPACK:

- **The 7-instruction sequence is not reducible for an arbitrary codebook.**
  llama.cpp's IQ4_NL kernel (`ggml/src/ggml-cpu/arch/arm/repack.cpp:465`,
  a non-uniform signed codebook like ours) is instruction-for-instruction
  identical: `ldr` + `ushr` + `and` + 2x`vqtbl1q` + 2 MACs. Independent
  convergence on the same 5 non-MAC instructions.
- **LUTI (FEAT_LUT) is not in V2's feature set** (`AArch64Processors.td:383`).
- **SVE2 BEXT/BDEP are present but useless**: latency 6, 2 µops, **V1 only**
  (0.5/cycle) against `AND` at latency 2, 1 µop, 4/cycle — and they produce
  one plane, not two.
- **TBX needs no MOV.** It writes only in-range lanes, and the AND/USHR
  guarantee every index is 0-15, so the destination is architecturally dead;
  any register will do. But single-table TBX (4/cycle, all pipes) optimizes
  **V01, which is not the binding port** — the bound is total issue width,
  6 µops / 4 pipes = 1.5 cycles per 16 bytes. No published kernel uses TBX
  for this, including Arm's own.
- **SVE `tbl z, {z}, z` is 2c/1µop on all four pipes** and a bit-exact
  drop-in at VL=128. A free hedge, worth taking only if the µop mix ever
  makes V01 bind (e.g. 2-bit codes with 4 TBLs).
- **KleidiAI's tighter ratios come entirely from tile width**, not from a
  better unpack: its 2-bit kernel spends *10* non-MAC instructions per 16
  bytes but feeds 32 SDOTs (M=8), for 0.31 non-MAC per MAC against our 2.5.
  Its int4 kernel reaches 2 unpack instructions only by exploiting a
  *uniform* codebook (values stay x16, folded into the dequant scale) — the
  option P17 closed for us on recall.

**The number that matters: at nq=1 the issue-bound floor is 1.5 cycles per
16 bytes and we measure 2.27.** So single-query is at 66% of its own
ceiling, and the remaining ~1.5x is stalls, not instruction count. The
unpack is done; scheduling is not.

**H46 — load-group size sweep: null.** H45 chose 4 loads in flight by
guess. Sweeping (nq=1 arm ST, 3 rounds):

| group | 2 | **4** | 8 |
|---|---|---|---|
| | 4.094 / 4.113 / 4.074 | **3.582 / 3.652 / 3.663** | 3.702 / 3.673 / 3.653 |

4 and 8 are indistinguishable; 2 starves the pipeline. The guess was already
at the optimum, and the plateau from 4 to 8 says load pressure is no longer
what binds — consistent with P25's stall diagnosis.

## H47 — software-pipeline the nq=1 loads: null

P25 put nq=1 at 66% of its issue-bound floor (2.27 cycles per 16 bytes
against 1.5), and H46 showed load *count* in flight is not binding. That
leaves load *latency*: the chain is load(6) -> AND/USHR(2) -> TBL(2) ->
SMMLA(3), 13 cycles deep. Issuing group `g+1`'s loads before consuming
group `g` should hide the first 6.

| nq=1 arm ST | H45 | H47 |
|---|---|---|
| | 3.694 / 3.642 / 3.612 | 3.603 / 3.637 / 3.617 |

Fully overlapping. V2's out-of-order window already covers this distance, so
hand-pipelining hands it nothing it had not already done. Reverted.

That closes the cheap explanations for the remaining 0.76 cycles per
register: not instruction count (P25), not load count (H46), not load
latency (H47), not accumulator count (H44/H45). What is left is either a
port interaction the µop model does not capture — V1 is shared between
TBL's `V01` and USHR's `V13`, the one contention the simple
`6 µops / 4 pipes` bound ignores — or something that needs hardware
counters, which neither box has.

Testing the V1 hypothesis means SVE `tbl z, {z}, z` (all four pipes) in
place of NEON `tbl` (V01). P25 rates it a free drop-in at VL=128 but
predicts no gain, on the grounds that total issue width binds rather than
V01. That prediction rests on the same µop model that has now failed to
explain the gap four times, so it is worth measuring rather than trusting —
but it needs SVE inline asm, since the Rust intrinsics are unstable, and
that is a larger change than the remaining evidence justifies mid-session.

## P26 — how production systems actually make nq=1 fast (and where turbovec stands)

Source read across FAISS, ScaNN, Qdrant, Lucene, RaBitQ, Weaviate, SVS,
SimSIMD and QuickerADC.

### turbovec's nq=1 kernel is already the densest shipped

| kernel | instructions per (vec,dim) pair |
|---|---|
| **turbovec (this branch)** | **7 per 32 pairs** |
| Qdrant TurboQuant NEON `query4bit/arm.rs` | 11 per 32 pairs |
| FAISS `kernels_simd256.h` NQ=1 | 12 per 64 pairs (same density) |
| ScaNN `lut16_avx2.inc` | same density |

Qdrant's is the direct comparable — 4-bit, **Lloyd-Max non-uniform
codebook, one query at a time**, the same problem — and it costs 11 where we
cost 7 (they pay 2 ZIPs for dim order and 4 SDOTs for a split-precision
query). There is no kernel win left to copy, which confirms H44/H46/H47 from
the outside.

### Nobody beats the shuffle for a non-uniform codebook

Every system with a genuinely fast nq=1 path got there by **giving up the
non-uniform codebook**: RaBitQ (`ip16_fxu4_avx512` — no shuffle at all),
Weaviate `rq4.go`, Lucene OSQ. The count is an ISA floor: a shuffle emits W
bytes per instruction, so 2W nibbles need 2 shuffles, and `TBL` zeroes rather
than wraps out-of-range indices, so the low-nibble `AND` is mandatory.

**Worth re-testing P17 against a stronger baseline.** P17 closed affine
codebooks at ~2 recall points, but it compared against *plain global
min/max* uniform. Weaviate, Lucene and RaBitQ all pair uniform codes with
**rotation plus a per-vector clipped interval** — Weaviate's `rq4ClipFactors
= {0.6,0.7,0.8,0.9}`, chosen per vector, with the stated rationale "with only
16 code points, spending them on the full [min,max] range wastes resolution
on a few outlier entries". That is a materially different proposition from
what P17 refuted, and if it holds the entire TBL pair disappears.

### The three levers that are actually pulled at nq=1

1. **DB-axis unrolling, capped ~4 accumulator groups.** FAISS caps `NQ x BB
   <= 4` (`IndexFastScan.cpp:532`); ScaNN swaps the query axis for
   `kNumDatapoints` ∈ {256,128,32}; Qdrant runs 4 chains and *measured*
   "~7-20% over a 1x version". **We already do this** — H45's 16 chains is
   the same lever, further than any of them.
2. **Cross-partition NTA prefetch** (ScaNN `PrefetchStrategy::kSmart`,
   `kPrefetchBytesAhead = 768`), issued into the *next partition* while
   scanning the current one. H43 refuted same-stream prefetch on x86; this
   is a different thing and untested here.
3. **Early-abandon against the heap bound** — QuickerADC's per-32-vector
   `_mm512_cmplt_epi16_mask`, FAISS's new `Panorama.h` (level-wise
   progressive distances, claimed up to 28.9x) and `PdxLayout.h` (claimed
   40%). **At nq=1 the winning move in current research is to touch fewer
   bytes, not fewer instructions per byte** — and turbovec does no pruning
   at all inside a block.

### One x86-specific idea worth sizing

QuickerADC (`VecProductQuantizer.h`) keeps an arbitrary codebook and still
cuts the lookup, by indexing a table with *packed* bits directly. Transplanted
to 4-bit on AVX-512 VBMI: build `T[byte] = q[2d]*level[lo] + q[2d+1]*level[hi]`
— the packed byte **is** the index, so unpack cost is zero and two dimensions
arrive pre-summed, evaluated as 2x`vpermi2b` + blend. ~3-4 instructions per
64 bytes against FAISS/ScaNN's 12. Costs: query LUT build grows 8x, tight
8-bit accumulator range, Ice Lake+. No NEON analogue (a 4+4 pair index needs
256 entries; `vqtbl4q` tops out at 64). Flagged by the agent as inference
from the QuickerADC mechanism, not a citation.

## H48 — prefetch on arm this time: refuted, x0.98

H43 refuted software prefetch on **x86**. arm was never tested, and this log
has now been caught three times treating a refutation as wider than its
measurement (H23, H42, H44), so it was worth the ten minutes.

`prfm pldl1keep` over the next q8 unit's 256 bytes, nq=1 arm ST:

| | H45 | H48 |
|---|---|---|
| | 3.634 / 3.613 / 3.629 | 3.696 / 3.677 / 3.759 |

Consistently ~2% worse. arm's hardware prefetcher already saturates this
stream — consistent with P24, where arm held a flat 18-20 GB/s from 2 MB to
154 MB while x86 fell off a cliff. Reverted. Prefetch is now closed on both
arches, each on its own measurement.

## Why early-abandon does not transfer to turbovec

P26's third lever — prune against the heap bound (QuickerADC, FAISS
`Panorama.h`, `PdxLayout.h`) — is where current nq=1 research puts its
effort, and it does **not** apply here. Recording the reasoning so it is not
re-derived:

Progressive pruning needs a *tight* bound on the contribution of the
dimensions not yet scanned. Panorama gets one by ordering dimensions by
energy, so the first slice carries most of the signal and the residual norm
is small. turbovec applies a **deterministic block-Hadamard rotation**
before quantizing, whose entire purpose is to spread energy evenly across
dimensions — precisely so no coordinate dominates. After that rotation every
dimension carries the same expected energy, so the residual bound over the
unscanned half is `sum |w_d| * max_level` across ~384 dimensions: far too
loose to ever fire.

The two are in direct opposition: the rotation that makes 4-bit
quantization accurate is the same property that makes progressive pruning
useless. Adopting Panorama would mean giving up the rotation, which costs
recall — the same trade P17 already refused.

Cheap block-level skips are similarly closed: a bound needs either per-block
statistics we do not store or a coarse pre-pass that costs what it saves.

## H49 — more accumulator chains for x86 nq=1: refuted, x0.95

arm's single-query path was latency-bound on two accumulator chains (H42),
and x86's `search_multi_query_permute_dot::<1>` has exactly the same shape:
two accumulators against VPDPBUSD's latency 5 on two ports, which wants ~10
chains. Splitting each into `P=4` independent partials (eight chains, six
extra registers, which NQ=1 has spare) should have transferred the fix.

| nq=1 x86 ST | H38 | H49 |
|---|---|---|
| | 5.400 / 5.294 / 5.274 | 5.693 / 5.447 / 5.854 |

x0.95. Refuted, and the reason is already in this log: **P24 showed x86
nq=1 is memory-bound above cache** (14.5 -> 31.5 ns/vec from 38 MB to
154 MB) while arm was flat. Adding chains cannot help a loop waiting on
DRAM, and the extra partial-summing in the epilogue is pure cost.

The same defect, diagnosed identically, has opposite answers on the two
arches because they are limited by different resources at the same shape.
That is the H36 lesson again — an idea transfers, a mechanism does not —
and it is now the fifth time this log has recorded a cross-arch transplant
failing for a reason that was already measured and written down here.

## H50 — a clipped uniform codebook costs **0.85 recall points, not 2**

P26 flagged that P17 refuted affine codebooks against the wrong baseline:
it compared Lloyd-Max to *plain global min/max* uniform, while Weaviate,
Lucene OSQ and RaBitQ all use rotation plus a **clipped** interval, on the
stated grounds that "with only 16 code points, spending them on the full
[min,max] range wastes resolution on a few outlier entries".

Simulated at 4 bits, N=20k, dim=768, k=10 (`clip_recall.py`):

| codebook | recall@10 |
|---|---|
| **Lloyd-Max (shipped)** | **0.8335** |
| uniform, clip=0.6 | 0.7705 |
| uniform, clip=0.7 | 0.8145 |
| **uniform, clip=0.8** | **0.8250** |
| uniform, clip=0.9 | 0.8245 |
| uniform, clip=1.0 (what P17 tested) | 0.8215 |
| uniform, per-vector best-of-grid | 0.8245 |

**The gap is 0.85 points, not ~2.** A single fixed clip factor recovers more
than half of what P17 measured, and the plain-uniform row (0.8215) reproduces
P17's baseline, so the two experiments agree where they overlap.

Second, cleaner result: **per-vector clip selection buys nothing.** Choosing
the factor per row on reconstruction error (0.8245) is no better than one
global 0.8 (0.8250). Weaviate's per-vector sweep does not pay for
inner-product recall, so this direction needs no per-vector metadata, no
format change, and no extra bytes — just a different codebook constant.

### What it would buy

An affine codebook makes the code *be* the value, so **both TBLs leave every
kernel on both arches**: 7 instructions per 16 bytes becomes 5, a 29%
reduction in the inner loop, at every one of the 8 cells. P25 established
the TBLs are otherwise irreducible, and P26 that every system with a fast
nq=1 path bought it exactly this way.

### DECIDED: closed. Recall is not traded.

Ryan, on seeing the 0.85: **"don't trade recall. Recall is more important."**
The ruling was never about the number 2 — it is that recall is not currency.
A speed probe to price the win was started and abandoned unmeasured, because
the answer does not depend on the size of the gain.

**This closes the whole family permanently**, and more firmly than P17 did:
uniform, split, bit-linear, and now clipped-uniform, at 4 bits, whether the
clip is global or per-vector. Nobody should re-open it with a better clip
grid or a cleverer interval search — the objection is to the trade, not to
the margin.

The consequence for every future kernel hypothesis: **the two TBLs are
permanent.** P25 established they are irreducible for an arbitrary codebook
on V2, P26 that every competitor with a faster nq=1 path bought it by
abandoning the non-uniform codebook, and this closes that purchase for
turbovec. The arm inner loop is 7 instructions per 16 bytes and that is the
floor, for good. Future gains must come from scheduling, layout, or work
avoidance — never from the unpack.

## H51 — finer block split when the query axis is one quad: null

H37 swept the tiling knobs at nq=100 only, and "a refutation is only as wide
as the axis it was measured on" has now caught this log five times, so nq=1
deserved its own sweep. At nq=1 there is a single query quad, so the block
axis must supply all the parallelism, and the `min_tile_blocks` cap — tuned
in H15 against the 13-quad case — leaves the workers coarse.

Smoke (nq=1 arm MT, `TV_NEON_CAP`): 64 -> 0.520, 128 -> 0.518, **256
(shipped) -> 0.543**, 512 -> 0.543, 1024 -> 0.538. A 4.6% win, above the
noise floor. Halving the cap only when `n_quads == 1` cannot touch nq=100,
where the condition is false.

Six interleaved rounds killed it:

| nq=1 arm MT | main | H45 | H51 |
|---|---|---|---|
| median | 0.584 ms | **0.541 ms** | 0.546 ms |

H51 is 1% *slower* than H45 and the distributions overlap heavily. The smoke
signal was noise that happened to land above the noise floor — which is what
the two-gate rule exists to catch, and the fourth time in this log the smoke
and the soak have disagreed.

Also worth recording: the first soak attempt showed a single 0.955 ms sample
against a 0.54 ms median for the *same* build. nq=1 MT is the noisiest cell
on the board — 0.5 ms of work across 8 threads — and needs interleaved
rounds rather than the 2-round cell harness to resolve anything.

## H52 — SVE `TBL` to dodge the V01 restriction: blocked by the toolchain

P25's one surviving suggestion was SVE `tbl z, {z}, z` — 2c, 1 µop, **all
four V pipes**, against NEON `TBL`'s 2/cycle on `V01` only, and a bit-exact
drop-in at V2's 128-bit VL. It is the only untested lever on the V1
contention that the `6 µops / 4 pipes` bound ignores (TBL is `V01`, USHR is
`V13`, and V1 is in both).

Unreachable from stable Rust. `asm!` has no `z` template modifier:

    error: invalid asm template modifier for this register class
       "tbl {o:z}.b, {{{a:z}.b}}, {b:z}.b"
             ^^^^^

The aarch64 `vreg` class offers `v/b/h/s/d/q` only, and the SVE `zreg` class
is not stable. The remaining route is binding fixed registers (`out("v0")`
and writing `z0` literally), which forces a MOV per operand per use — three
instructions where TBL is one. That is strictly worse than the single
instruction it would save.

Recorded as **blocked, not refuted**: the hardware advantage is documented
and real, and this becomes available the moment Rust stabilizes SVE register
classes. It also explains P26's negative finding that no shipped kernel uses
this trick — Arm's own assembly is hand-written, where the modifier problem
does not exist, and they still chose plain `TBL`.

## H53 — quarter-blocks for the batched vm8 kernel: refuted, x0.965

H41 chose eighth-blocks because quarter-blocks spilled: 16 accumulators plus
8 A operands. H45 later showed accumulator and *load* pressure are separable
— the spill came from transient loads — so quarter-blocks deserved a retry,
and they halve the epilogue's 2x2 scatter from 8 runs per block to 4.

| arm nq=100 ST | eighth (shipped) | quarter |
|---|---|---|
| | 120.46 / 120.10 / 121.06 / 119.92 | 124.77 / 124.51 / 125.21 / 124.56 |

x0.965, no overlap. Eighth-blocks stand.

**H45's lesson does not generalize backwards.** Separating load pressure from
accumulator pressure rescued the *single-query* kernel, where one query pair
means 8 free registers and 16 chains to gain. In the batched kernel at NQ=8
the accumulators alone are 16 and the A operands 8, so quarter-blocks are
over the file with or without the load trick — and the halved epilogue, ~3%
of runtime, cannot pay for it. The right generalization is narrower than it
looked: *when registers are scarce, cap the transients; when they are
plentiful, add chains.* Which of those applies is set by the batch width,
not by the kernel.

## H54 — interleave blocks for x86 nq=1: **HM x1.851 -> x1.928**

P24 measured x86 nq=1 as memory-bound once the code array leaves cache
(14.5 -> 31.5 ns/vec, 38 MB -> 154 MB) while arm stays flat. Two attacks on
it failed — H43 (prefetch) and H49 (more accumulator chains) — and **both
failed for the same reason, which neither entry named: neither adds a memory
*stream*.** One query walks one sequential stream, so outstanding misses are
capped at whatever a single stream sustains. Prefetch issues the same stream
earlier; extra chains give the ALUs more to do while still waiting on it.

Interleaving `BLK` blocks inside the quad loop walks BLK independent
streams. At NQ=8 the registers are full and BLK=1; at NQ=1 only two
accumulators are live, so BLK=4 costs 6 registers of 32.

| nq=1 x86 ST | H38 | H54 |
|---|---|---|
| | 5.376 / 5.318 / 5.655 / 5.972 | **4.113 / 4.124 / 4.359 / 4.186** |

Full 8-cell, bit-identical (`5939c346...`), 133/133 green:

| cell | main | now | speedup |
|---|---|---|---|
| arm nq100 MT | 41.428 ms | 14.323 ms | x2.892 |
| arm nq100 ST | 310.760 ms | 121.921 ms | x2.549 |
| arm nq1 MT | 0.602 ms | 0.545 ms | x1.105 |
| arm nq1 ST | 4.074 ms | 3.636 ms | x1.121 |
| x86 nq100 MT | 61.984 ms | 19.507 ms | x3.178 |
| x86 nq100 ST | 241.659 ms | 100.343 ms | x2.408 |
| **x86 nq1 MT** | 2.460 ms | 1.082 ms | **x2.273** (was x2.056) |
| **x86 nq1 ST** | 9.468 ms | 4.201 ms | **x2.254** (was x1.753) |

**Harmonic mean x1.9281**, from x1.8506.

**Two refutations pointed at the answer and neither was read that way.** H43
and H49 both sit in this log as "x86 nq=1 does not respond to X", and the
common factor — X did not change the number of memory streams — only became
visible when a third attempt named the *resource* instead of the technique.
The lesson is not "try more things": it is that a refutation should record
**which resource it failed to move**, not merely that it failed. H43 and H49
recorded only the latter.

The first implementation of this hypothesis compiled and would have measured
as a no-op — it stepped the block loop by BLK but still processed the
sub-blocks one after another, leaving one stream in flight. Caught by reading
it back before spending a measurement: P23's lesson applied before the fact
rather than after.

## H55 — the same stream argument, on arm nq=1 (next up, designed)

arm nq=1 is now the worst pair on the board (x1.105 MT, x1.121 ST) and H54's
mechanism has not been tried there. P24 measured arm as **flat** across N —
18-20 GB/s from 2 MB to 154 MB — which says it is not *bandwidth*-bound. It
does not say a second stream cannot help, because H54's win on x86 was about
outstanding misses and latency, not about bandwidth headroom. Treating P24 as
closing this would be the scoped-refutation error this log has now recorded
six times.

Design, so it starts clean:

- `score_block_vm8_single` currently owns all 16 vector-pair accumulators of
  one block. Two blocks at 8 accumulators each keeps the 16 chains H45 showed
  are needed (H44 proved 8 stall) while walking two streams.
- The caller must pass **two** block offsets and two `out` rows; today it
  passes one of each, at both arm single-query sites. That is the part that
  makes this more than a kernel edit, and why it was not attempted mid-context
  rather than left half-applied.
- Register budget: 16 accumulators + 2 A operands + level table + mask + 4
  transients per stream ~= 26 of 32. Fits, but it is the same margin that bit
  H41's first cut, so check `objdump` for `stp` in the loop before trusting a
  null.

If arm behaves like x86 the two cells move from ~1.11 to ~1.4+, which would
take the harmonic mean from x1.928 to roughly x2.05.

## Loop state

Streak 0 — H54 landed, taking the 8-cell harmonic mean from x1.851 to
x1.928. Before it: H46, H47, H51 (null), H48, H49, H53 (refuted), H50
(closed by Ryan: recall is not traded), H52 (blocked on stable Rust), and
H45, which took the 8-cell harmonic mean from x1.769 to
x1.851. H43 and H44 (both refuted) preceded it. H42 repaired an nq=1 regression that H33 introduced and
nine hypotheses' worth of nq=100 measurement never saw. H41 landed before
it; H39 (refuted) and H40 (null) preceded that.

**The goal now weights nq=100 batch search and load + single search equally,
and every hypothesis must be measured at both shapes.** Before that: H35 (null),
H36 (refuted), H37 (null), with P21 (null probe) among them. Before that: H33 and H34 both landed.
P18 on both arches,
H28/H34 on x86, H30/H32/H33 on arm. Nine confirmed improvements: H5, H9,
H15, H21, P18, H33, H34, H38, H41. H19/H20 are validations rather than
changes.

Shipped on PR #485, against `origin/main`, six interleaved rounds with all
three arms alternating inside each round:

| | main | now | total |
|---|---|---|---|
| x86 ST | 241.15 ms | 99.22 ms | **x2.431** |
| x86 MT | 61.92 ms | 19.40 ms | **x3.192** |
| arm ST | 321.87 ms | 125.12 ms | **x2.573** |
| arm MT | 42.40 ms | 14.55 ms | **x2.915** |

Recall is up on both arches and cross-arch scores are bit-identical,
verified through both the `add` and `load` paths by md5.

Read absolute ms within a row, not across arm runs from different sessions:
the arm box's `main` baseline has ranged 310 -> 329 ms as it warmed. Every
ratio here is from arms alternating inside one round, so the ratios hold.

Where the two arches now sit, after H33 on arm and H34 on x86: x86 takes ST
back (103.42 vs 122.70) and leads on nothing else — arm holds MT 15.17
against 20.37. Both moved a long way this session and neither is at a wall.

Next, in order:

Both follow-ups from H34 are now closed: the x86 batch width is confirmed
at 8 (H35, null) and arm has no mirror of the spill (P21, null). H36 tried
to widen x86 via half-blocks and lost.

Both kernels now sit at a similar place — x86 ~30% and arm ~37% of their
respective MAC ceilings, with clean unspilled inner loops. The remaining
lever on both is the unpack-to-MAC ratio, and on both the register file is
what caps it:

1. **Layout change to drop the unpack.** arm spends 16 of 32 instructions
   per quad on nibble splitting and operand movement. Packing codes so
   nibbles arrive in dimension order would delete the ZIPs, but it changes
   the on-disk format and needs a repack path — a much larger piece of work
   than anything in this log so far, and it should be costed before it is
   attempted.
2. **AMX on x86** is the only remaining wider-MAC option and looks
   unpromising: per-tile setup does not amortize over a 32-vector block.
   AVX-512 has no int8 matrix-multiply instruction.
3. Nothing cheap is left. The next real gain likely costs a format change.

Pre-existing and untouched:
`allocation_hot_paths::repack_allocation_count_does_not_scale_with_vector_count`
fails identically on main, on both arches, and with the layout disabled.

**Standing constraints.** No RAM increase (+25% ruled out, which closes
the 1-bit prefilter sidecar and the 5-bit uniform codebook). The LUT cap
127 -> 255 change is on hold. **Recall costs of ~2 points are not
acceptable at any speed** — combined with P17, which shows ~2 points is a
floor for the entire dot-product-compatible codebook family rather than an
opening bid, this closes uniform, split and bitlinear together and retires
"replace the LUT with a dot product" as a direction on both arches.

Priced and rejected, with measurements rather than estimates:

| lever | speed | price | status |
|---|---|---|---|
| uniform-4 codebook | est. x2.5-3.5 | -0.021 recall | **ruled out** |
| split / bitlinear codebook | est. x2.5 | -0.0175 recall | **ruled out** |
| deferred u8 widening (cap 31) | x1.10 arm | -0.166 recall | refuted |
| 1-bit prefilter sidecar | est. x4 | +25% RAM | ruled out |
| 5-bit uniform codebook | est. x2.5 | +25% RAM | ruled out |
| 8 queries per pass | parity | — | refuted |
| LUT cap 127 -> 255 | none (recall gain) | — | on hold |

Closed by measurement: both kernels run at 92-93% of their own
instruction sequence's streaming ceiling; the lookup count is
algorithmically fixed; exact pruning is dead to score concentration
(any bound ignoring fraction f of energy has slack >= sqrt(f), and the
block-Hadamard rotation deliberately flattens energy so f < 16/D);
prep is 1.3-2.0% and unimprovable; huge pages are already in effect
(the code buffer is 92% `AnonHugePages` warm and cold); the schedule
sits at a joint two-arch peak.

### H25 (refuted) — NEON `TBX` in place of `TBL` (target: search, arm)

The Neoverse V2 optimisation guide prices the two lookup instructions
very differently:

| instruction | latency | throughput | pipes |
|---|---|---|---|
| ASIMD `TBL`, 1-2 tables | 2 | 2 | V01 |
| ASIMD `TBX`, 1 table | 2 | **4** | **V (all four)** |
| SVE `TBL` / `TBX` | 2 | 4 | V |

They differ only in the out-of-range case: `TBL` writes zero, `TBX`
leaves the destination byte unchanged. Our indices are nibbles, always
0-15, so they are never out of range and the two are **semantically
identical here** — at twice the documented throughput on twice the pipes.

P9 had concluded the loop is bound by its loads, ANDs and widening adds
rather than by the lookup unit, which reads as "a faster TBL cannot
help". That misses the mechanism worth testing: those ANDs and adds also
need V pipes, and `TBL` monopolises V01. Spreading lookups across all
four pipes relieves the exact pressure P9 identified as binding.

| variant | L1-resident | streaming 77 MB |
|---|---|---|
| `TBL` (current) | 3.79 G/s | 3.61 G/s |
| `TBX`, zero fallback | 3.14 (**x0.83**) | 3.04 (**x0.84**) |
| `TBX`, table as own fallback | 2.71 (**x0.72**) | 2.65 (**x0.73**) |

**NON-WIN (refuted).** Streak 3.

**Cause, from the disassembly** rather than inferred: 16 `tbx` in the
inner loop are accompanied by **17 `movi` and 9 `mov`**. `TBX` reads its
destination register, so the compiler must materialise the fallback
operand into that register before every lookup. `TBL` overwrites its
destination unconditionally and needs no such setup.

The second variant tested the obvious repair — the fallback value is
irrelevant to us, so pass a register that is already live (the table
itself) and no `movi` should be needed. It came back **worse**, x0.72:
using the table as the fallback makes the destination alias a live input,
which costs the register allocator more than the `movi` did.

**Generalisable:** `TBX`'s 4/cycle is unreachable for any *pure lookup*
workload. It is priced for the merge case, where the destination already
holds data you intend to keep; when you only want a lookup, the
destination setup is pure overhead. That closes `TBX` as a family, not
just this instance.

**Independently corroborated after the fact.** x265 PR #901 changed its
`rev16`/`rev32` helpers *from* `TBX` *to* `TBL` with the note that "`TBL`
has slightly better performance on some CPUs" — the same direction,
reached separately, and nobody there investigated why.

A literature sweep also established that the V2 pipe asymmetry is
**documentation-only**: it appears in the V2 and Cortex-X3 SWOGs, was
transcribed into `AArch64SchedNeoverseV2.td` by its Arm author (D151894,
validated only against SPEC, not per-instruction), and has never been
independently measured on V2, X3, Graviton 4, Grace, X4 or V3. The
restriction *was* measured to be real one generation earlier
(insn_bench_aarch64 on Graviton 3 / Neoverse V1: `tbl` 2.00/cyc against
`ext` 4.00/cyc on the same harness, and 4.00 on Apple M1, so the tool
does resolve 4-wide where the hardware allows). What was never checked is
the V2-specific *asymmetry* — new in V2/X3, gone again in V3 — which is
exactly the part this hypothesis rested on. These numbers appear to be
the first measurement of it, and it does not deliver.

### H26 (refuted) — paired layout + `vqtbl2q_u8` + `vpadalq_u8` (target: search, arm)

The strongest ARM lead this round, and the direct analogue of the x86
win. Research on the ARM 4-bit PQ paper (arXiv 2203.02505) and its Faiss
implementation found that Faiss spends **4 ops per 16 lanes** on the
accumulate chain where ours spends 3 — and that `vpadalq_u8` (`UADALP`)
would spend **1**, adding adjacent byte lanes pairwise straight into u16.
Neither the paper nor Faiss explores it.

`UADALP` is semantically wrong under the current layout, where adjacent
byte lanes are different database vectors. It becomes exactly right under
a layout where lanes 2i and 2i+1 hold vector i's codes for two
*consecutive byte-groups*, because then the pairwise fold sums two
dimensions of one vector — which is what the score wants. The two lanes
then need two different 16-entry LUTs, which `vqtbl2q_u8` supplies from
one 32-entry table indexed by `(parity << 4) | code`; the guide prices
TBL with "1 or 2 table" registers identically, so the wider table is free.

Arithmetic per 32 bytes of codes, per query: current 4 TBL + 2 `vaddq_u8`
+ 4 `vaddw_u8` = 10 ops for 32 vectors x 1 group; paired 4 TBL2 + 2
`vaddq_u8` + 2 `vpadalq_u8` = 8 ops for 16 vectors x 2 groups. Same work,
20% fewer per-query ops, 38 against 44 including the shared split.

| | L1-resident | streaming 77 MB |
|---|---|---|
| current | 3.79 G/s | 3.61 G/s |
| paired + UADALP | 3.38 (**x0.89**) | 3.26 (**x0.90**) |

**NON-WIN (refuted).** Streak 4.

**Cause.** The op count was right and irrelevant. It counted arithmetic
and ignored LUT traffic:

| | vectors x groups per iteration | LUT bytes |
|---|---|---|
| current | 32 x 1 | 32 |
| paired | 16 x 2 | **64** |

Pairing halves the vectors per register, so each loaded LUT entry is
amortised over 8 vectors instead of 16 — **twice the LUT load bytes for
the same work**. P9 had already established this loop is bound by its
loads. The change traded 14% off the resource that is not binding for
100% more of the resource that is.

**This retroactively explains the x86 win**, which had been treated as a
one-off. `vpermb` pays exactly the same halving penalty, but `vpdpbusd`
reduces *four* byte-groups per instruction, which buys the LUT
amortisation back. `vpadalq_u8` folds only two, so there is nothing to
pay it back with.

**The rule, which prices a family rather than an instance:** any layout
change that reduces vectors-per-register must buy back at least as many
byte-groups per instruction, or it loses on LUT traffic no matter what it
saves in arithmetic. On ARM that requires a >=4-way reduction instruction
operating on lookup results, and NEON has none — `UDOT` reduces 4 but
consumes raw bytes, not table outputs, which is why it is only reachable
via the uniform-codebook route already priced at -0.021 recall.

### H27 (refuted, and it closes the direction) — SVE `TBL` for lookup bandwidth

The last remaining route to more lookup bandwidth on arm, and a clean
test rather than a retest of H25: SVE `TBL` is destination-only like NEON
`TBL`, so it carries none of `TBX`'s register-initialisation penalty. The
V2 guide prices it at throughput 4 on all four V pipes against NEON
`TBL`'s 2 on V01. Third-party evidence pointed the same way — isa-l PR
#367 measured an SVE rewrite of TBL-heavy Galois-field kernels 28-32%
faster than NEON on Graviton 4, the same core.

Measured in isolation (`turbovec/examples/sve_tbl_probe.rs`): eight
mutually independent lookups per iteration sharing one table and one
index register, so nothing is serialised by a data dependency and the
only limit is issue capacity. V2 implements SVE at VL=128, so a `z`
register is the same 16 bytes as a `v` register — like for like.

| | rate | per cycle |
|---|---|---|
| NEON `tbl v.16b, {v.16b}, v.16b` | 11.97 G/s | **4.00** |
| SVE `tbl z.b, {z.b}, z.b` | 11.97 G/s | **4.00** |

Core clock measured at 2.993 GHz in the same session with a dependent
add chain, so the per-cycle figures are exact rather than nominal.

**x1.000. NON-WIN (refuted).** Streak 5.

**The measurement contradicts the documentation, in the informative
direction.** The guide says NEON `TBL` is throughput 2, restricted to
V01. It is not: it achieves **4/cycle**, the full vector issue width, and
SVE `TBL` has nothing to add because there is no restriction to escape.

That single number closes the whole lookup-bandwidth direction on arm and
retroactively explains the previous two refutations, which had been
recorded with separate local causes:

* **H25** was never an upgrade. `TBX`'s documented 4/cycle looked like
  2x more lookup throughput; NEON `TBL` was already at 4/cycle, so the
  ceiling `TBX` promised was the one already in hand. It could only lose,
  and the `movi` per lookup is how it lost.
* **H26** assumed lookups were worth economising. They are not — they are
  free at the margin, running at maximum issue rate.
* **P9** reached "the loop is not bound by the table-lookup unit" from
  the opposite direction, by measuring the whole kernel. This confirms it
  from first principles at the instruction level.

**Novel as far as the literature goes.** A dedicated sweep found no
published TBL microbenchmark on Neoverse V2, Cortex-X3, Graviton 4,
Grace, Cortex-X4 or V3 — the pipe claim exists only in the V2 and X3
optimisation guides and in LLVM's V2 scheduling model, which was
transcribed from them by its Arm author and validated only against SPEC.
The V01 restriction was independently measured to be real one generation
earlier on Neoverse V1. It does not hold here.

Caveat worth keeping: this is Google Axion, a V2-based core that Google
may have customised, so the result is a statement about our target
hardware rather than about every V2 implementation. That is the statement
the decision needs.

**Why the guide is wrong, established independently of the measurement.**
A row-by-row diff of the V1, V2 and V3 optimisation guides: in V1 every
`TBL` and `TBX` row reads `V01`, symmetric. In V2, every `TBX` row and the
SVE `TBL` row receive the same edit — throughput doubled, `V01` -> `V` —
while the three ASIMD `TBL` rows are character-for-character identical to
V1. In V3 the ASIMD `TBL` rows finally receive exactly that edit and the
asymmetry disappears. That is the signature of a stale table row left
behind when the lookup crossbar was widened, and it is internally
incoherent besides: `TBX` is a strict superset of `TBL` (same crossbar
plus a merge), so hardware issuing `TBX` on four pipes can issue `TBL` on
four; and V2's SVE is 128-bit on the same pipes, so `tbl z0.b, z1.b, z2.b`
and `tbl v0.16b, {v1.16b}, v2.16b` are bit-identical work. The measurement
above is what that hypothesis predicts.

`TBX`'s destructive destination was also confirmed as a genuine 2-cycle
loop-carried chain that the renamer does not elide even when the
destination is architecturally dead — consistent with H25's disassembly,
and it means no amount of unrolling would have rescued it.

### P18 (candidate) — constant nibble->level permute, then `SDOT`

**P17's conclusion was too strong, and this is the variant it missed.**

The requirement for a dot-product kernel is not that the reconstruction
levels be uniform. It is that the level be reachable from the stored nibble
more cheaply than a per-dimension per-query table. turbovec's codebook is
**shared across every dimension**, so nibble -> level is a *fixed 16-entry
permute*: query-independent, dimension-independent, register-resident for
the whole scan. Apply it to nibbles that already have to be unpacked, and

    score = sum_d q[d] * C[code[d]]

is a plain dot product over the permuted bytes — **with the full Lloyd-Max
codebook intact**. Nothing about the codebook changes, so P17's ~2-point
floor does not apply.

**Accuracy improves.** Today's LUT rounds the *product* `q[d]*C[c]` to 7
bits per entry (cap 127). This quantises `q` and `C` to 8 bits separately
and accumulates the products exactly in s32.

| scorer | recall@10 | vs shipping |
|---|---|---|
| float (ceiling) | 0.8435 | — |
| `lut7` (ships today) | 0.8280 | — |
| **permute-dot, int8 query** | **0.8410** | **+0.0130** |
| permute-dot, int16 query | 0.8430 | +0.0150 |

**The structural difference from every other variant tried:** the permute
is shared across queries, where today's lookup is per-query because the LUT
bakes in `q[d]`. Per 32 bytes of codes, four queries: current 4 shared + 40
per-query = 44 ops; this 6 shared + 16 per-query = 22. It also passes H26's
rule — there is no per-query LUT to re-read, only 16 bytes of int8 query
weights per group against the current 32-byte LUT slice, so LUT traffic
*falls*.

Measured interleaved in one process, 5 rounds, medians, query weights
loaded per group so the variant is not flattered:

| | L1-resident | streaming 77 MB |
|---|---|---|
| control | 3.78 G/s | 3.36 G/s |
| permute-dot, 8 acc/query | 3.99 (x1.056) | 3.60 (x1.071) |
| **permute-dot, 4 acc/query** | **4.55 (x1.203)** | **3.99 (x1.187)** |

**x1.187 streaming, and the accumulator count matters more than anything
else.** Eight `int32x4_t` per query covers a full 32-vector block but at
four queries that is 32 live vector registers against NEON's 32 total, so
it spills; tiling half a block at a time (four per query, 16 live) nearly
triples the gain for identical work and traffic.

Note the op count predicted x2 and the measurement gives x1.19 — the loop
is not purely issue-bound, consistent with P16. Recording the gap rather
than the prediction.

**x86 gains more, and needs no format change to test** — it already has
the vector-major layout from H21. The structural reason is that the
shipping kernel's `vpermb` is per-query only because the table it permutes
*is* that query's LUT; under permute-dot it becomes query-independent and
leaves the per-query path entirely. Per 64 bytes per query, control is
2 x 64-byte LUT load + 2 `vpermb` + 2 `vpdpbusd`; candidate is one 4-byte
broadcast + 2 `vpdpbusd`, so LUT traffic falls from 128 bytes to 8.

`vpdpbusd` multiplies unsigned by signed and both operands here are
signed, so the levels are stored offset by +128. The resulting
`128 * sum_d q[d]` term is a per-query constant, independent of which
database vector is scored, so it shifts every score equally and cannot
change a ranking.

Measured the same way (`turbovec/examples/x86_permute_dot.rs`),
interleaved, 5 rounds, medians:

| | L1-resident | streaming 77 MB |
|---|---|---|
| x86 | **x1.487** | x1.115 |
| arm | x1.203 | x1.187 |

**Measured at one thread and at full width**, since the goal metric is
multi-threaded but a win has to hold single-threaded too — an ST
regression means the hypothesis failed however good MT looks. Streaming
77 MB, interleaved, 5 rounds, medians:

| | 1 thread | 8 threads |
|---|---|---|
| arm | x1.176 | **x1.192** |
| x86 | x1.110 | **x1.144** |

**The win holds MT on both arches and is slightly larger there**, which is
the opposite of the usual concern that a compute saving evaporates once
threads saturate memory. It follows from P16: MT runs at 59% of available
bandwidth against ST's 91%, so there is more headroom at full width for a
compute saving to show. arm's control also scales 3.42 -> 29.22 G/s across
8 cores (x8.5), so it is not bandwidth-starved at full width either.

The earlier single-threaded-only figures are why the x1.487 L1 number
should not be quoted as the expected gain: that harness is issue-limited
with the data in cache, while the real scan streams. The honest range is
the table above, and only the end-to-end A/B settles where in it the cell
lands. Recorded explicitly because this log's isolated-loop estimates have
run optimistic five times, and this is the shape of error behind them.

**Not yet a confirmed improvement.** This is the microbenchmark, not the
cell; four earlier estimates in this log ran optimistic against the real
search. What it needs before it can be believed:

* an end-to-end A/B on both boxes, since the kernel is ~90% of arm search
  time and the epilogue and top-k are unchanged;
* a vector-major layout on arm — `SDOT`'s 4-byte reduction needs four
  consecutive dimensions of one vector in four adjacent bytes. x86 already
  has exactly this layout from H21, so the same change should apply there
  by replacing `vpermb` with a shared `vpshufb` pair;
* the same score-change disclosure H21 needed. Scores move (they get *more*
  accurate), so this is a deliberate departure from bitwise stability.

**Premise verified against the real pipeline, not just the simulation.**
`centroids` is a single shared 16-entry codebook (`search.rs:1541`),
applied identically for every dimension as `q_rot_row[dim] *
centroids[code]`. So the nibble->level map really is constant across all
768 dimensions and the fixed permute exists.

The simulation had missed `tqplus_shift` / `tqplus_scale`, the
per-coordinate calibration threaded into `search` and non-empty for v3+
indexes. They make the reconstruction per-dimension affine,
`a[d]*C[code] + s[d]`, which is free rather than fatal:

    sum_d q[d]*(a[d]*C[code[d]] + s[d])
      = sum_d (q[d]*a[d]) * C[code[d]]      <- dot product, folded weights
      + sum_d q[d]*s[d]                     <- per-query constant

The scale folds into the int8 query weights at build time and the shift
collapses to one scalar per query — exactly the folds the current LUT
already performs. So the scheme covers v3+ indexes too, and TQ+
calibration costs it nothing.

**Cross-arch numerics improve.** Today the VNNI path accumulates exactly
in u32 while the classic path rounds through f32 every 256 byte-groups,
leaving a max score difference of 4.6e-05 between arches. Permute-dot
computes `sum_d qi[d]*Ci[code[d]]` as exact integer arithmetic on both,
so given the same int8 quantisation the two agree bitwise. x86's `+128`
offset (needed because `vpdpbusd` is unsigned x signed where `SDOT` is
signed x signed) contributes exactly `128 * sum_d q[d]`, an integer
constant per query, so subtracting it recovers the identical value with
no rounding.

Sanity check on the layout, since it is the part that could silently be
wrong: four adjacent bytes of one vector hold dimension pairs {0,1} {2,3}
{4,5} {6,7}, so the lo-nibble register holds dimensions 0,2,4,6 and the hi
register 1,3,5,7. `SDOT` therefore sums four genuine dimensions of one
vector into that vector's own dword lane, with the query weights in the
matching interleaved order. Correct, and it costs only a reordering of the
query weight vector at build time.

### P17 — can a dot-product-friendly codebook beat uniform?

P15 measured a uniform 4-bit codebook at -0.021 recall and framed that as
the price of turning the score into an integer dot product. But uniform is
a far stronger constraint than the dot product needs. The score is
`sum_d q[d] * C[code[d]]`; for it to be computable without a lookup table
`C` need not be uniform, only **linear in something accumulable**:

| family | reconstruction | free shape params | kernel |
|---|---|---|---|
| uniform | `a*c + b` | 1 | one dot product over raw codes |
| split | `s1*(c>>2) + s2*(c&3) + b` | 2 | two dot products over 2-bit streams |
| bitlinear | `sum_k w_k*bit_k(c) + b` | 4 | four dot products over *binary* streams |

Uniform is the special case `w_k = a*2^k`, so bitlinear can only do
better. It looked genuinely promising: subset sums of four weights are
binomially dense in the middle, which is the same qualitative shape
Lloyd-Max wants (spacing 0.010 near zero widening to 0.024 at the tails).

Fit by constrained Lloyd — assign to nearest achievable level, re-fit the
free parameters by weighted least squares inside the family, repeat — from
62 starting points per family, because the uniform grid is a fixed point
of the iteration and starting there guarantees finding it.

| family | recall@10 | delta | coord MSE |
|---|---|---|---|
| Lloyd-Max (needs LUT) | 0.8435 | — | 1.000x |
| uniform | 0.8235 | -0.0200 | 1.206x |
| split | 0.8185 | -0.0250 | 1.202x |
| bitlinear | 0.8260 | **-0.0175** | **1.193x** |

**Four free parameters recover about 1% of the gap.** Every family
converges to a near-arithmetic grid from every start.

**The result worth keeping: the dot-product constraint is, in effect, the
uniform constraint.** ~2 recall points is a floor on this whole family,
not an opening bid, and no cleverness in the codebook reduces it. That
retires "find a smarter dot-product-compatible codebook" as a direction.

Note on boundaries, which was the original hypothesis and is wrong: for
any *fixed* level set the MSE-optimal decision boundaries are exactly the
midpoints, which `encode` already uses. There is no separate boundary
freedom to exploit — the entire question was how much shape the level set
can carry, and the answer is almost none.

### P16 — is the scan bandwidth-bound? (gate for wider query blocking)

Research into how FAISS and ScaNN structure their scans surfaced a
distinction we had folded into one constant. Both separate **how many
queries fit in registers** (FAISS instantiates kernels for 1-4, ScaNN
caps at 3 with the comment "register spilling happens when kNumQueries >
3" — our `QBS = 4` is right here) from **how many queries ride along per
pass over the codes**, where FAISS's default is `qbs2 = 11`, decomposed
2+3+3+3, via four back-to-back kernel calls against the same `codes`
pointer.

At `QBS = 4` a search makes `nq/4 = 25` passes over 76.8 MB = **1.92 GB
per search**. At 11 it would be 9 passes, 0.70 GB — a 2.75x cut. That is
only worth anything if the scan is actually waiting on memory, so measure
that first (`turbovec/examples/stream_bw.rs`) rather than build the
three-axis blocking the change would need.

| | scan achieves | available at 77 MB | available at 512 MB | utilisation |
|---|---|---|---|---|
| arm | 52.3 GB/s | 192.5 GB/s | 174.6 | **27%** |
| x86 | 38.5 GB/s | 65.6 GB/s | 48.6 | **59%** |

**Neither box is bandwidth-bound, so the idea is dead at the gate.**

Both L3s hold the entire code array — arm 80 MiB, x86 105 MiB against
76.8 MB — which is why the 77 MB read rate beats the 512 MB rate on both.
The codes are already served from L3 across passes, so the traffic the
change would remove was never being paid for at DRAM prices.

**This finally gives H23 a cause.** Eight queries per pass was recorded as
"cancels against schedule granularity", which described the result
without explaining it. The actual reason: total work is fixed at
`nq * N * (dim/2) * 2` lookups independent of QBS, and the kernel is
compute-bound, so re-blocking queries cannot change what the machine has
to do. Any QBS variant was always going to measure parity.

It also sharpens what P10's "92-93% of streaming ceiling" means. That
ceiling was the same *instruction sequence* run over 77 MB — a combined
compute-and-memory figure, not a bandwidth figure. Being at 92-93% of it
while at 27% of pure read bandwidth confirms the binding constraint is
issue rate, not the memory system.

**What this leaves on arm.** Lookups are at maximum issue rate, the
accumulate chain is already leaner than Faiss's (3 ops per 16 lanes
against 4), H26's rule rules out relayouts that reduce vectors per
register, and the kernel measures at 93% of its streaming ceiling. The
arm inner loop is closed. Remaining arm ideas must reduce *work*, not
schedule it better — which is the uniform-codebook family, priced at
-0.021 recall, or nothing.
