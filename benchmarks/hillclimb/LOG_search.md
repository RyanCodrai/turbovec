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

**Corrected on working it through: blocked, and the design above was wrong.**

The register budget in that sketch was miscounted. arm's single-query kernel
holds **16 accumulators for one block** — all 16 vector pairs — because H44
proved 8 chains stall and H45 proved 16 are needed. Two blocks at 16 each is
32 accumulators before the A operands, level table, mask or any transient.

Eight per block (16 total) does fit, but then each block covers only 8 of its
16 vector pairs per pass, so the loop must run twice per block over registers
0-7 then 8-15. That reads each byte exactly once — the halves are disjoint
bytes of the 256-byte unit, not a re-read — but it walks each block's array
with a **stride**, touching bytes 0-127 of every unit and skipping the rest.
That is the access pattern H44 measured at **x0.69**.

So the asymmetry that made H54 work on x86 does not exist here: x86's
single-query kernel had only **2** accumulators live, leaving 30 registers to
spend on extra streams. arm's already spends 16 on the chains it needs. The
same idea, the same shape, and the binding resource is different — which is
the rule H53 wrote down (*check which resource was binding in the source
context before transplanting*) applied one hypothesis later, this time before
the measurement rather than after.

Recorded as **blocked by the register file**, not refuted: no measurement was
taken, because the arithmetic decides it. The x86-side lever remains
available (BLK is a const generic there and only 4 is tried; 2 and 8 are
not).

## H56 — sweep the x86 stream depth: null, 4 was already right

H54 picked `BLK=4` without sweeping. nq=1 x86 ST, 3 rounds, medians:

| BLK | 1 (H38) | 2 | **4** | 8 | 16 |
|---|---|---|---|---|---|
| | 5.41 ms | 5.32 ms | **4.04 ms** | 4.12 ms | 4.46 ms |

4 stands; the guess was at the knee.

The shape is the interesting part. **BLK=2 is barely better than BLK=1**
(5.32 vs 5.41) and then 4 drops 24%. A second stream buys almost nothing
while a fourth is transformative, which is not what a simple "more
outstanding misses is better" story predicts — it looks like a threshold in
the core's fill-buffer or stride-detector rather than a linear return. 16
regresses again, presumably as the streams start evicting each other.

That also retro-explains H43: prefetching *one* stream harder was never
going to reach the knee, no matter the distance or the lookahead.

## H57 — more chains for the arm batched kernel: null

Eighth-blocks give `NP*2 = 8` accumulator chains at NQ=8, and P25 puts SMMLA
at latency 3 on four pipes — wanting ~12 to saturate. Quarter-blocks would
give 16 and spill (H41, reconfirmed by H53), so partials were the way to get
chains without accumulators: split even-dim and odd-dim into separate
partials, so the two SMMLAs of one code register stop chaining on each other.
16 chains for 8 registers and no extra loads.

| arm nq=100 ST | shipped | H57 |
|---|---|---|
| median | 119.8 ms | 120.3 ms |

Overlapping. Null.

**The negative is worth more than the change would have been.** The two
SMMLAs per register genuinely were dependent — `acc = smmla(acc, ..)` twice
— so if the batched kernel were latency-bound at 8 chains, breaking that
chain had to help. It did not, so **arm at nq=100 is not latency-bound**,
and the ~12-chain figure that correctly explained H42 and H44 does not
transfer to this shape.

That matters for the map: at nq=1 arm is latency-bound (H42, H44, H45 all
turn on chain count) and at nq=100 it is not. Same kernel family, same
instruction, opposite constraint — decided by how much independent work the
batch already supplies. It is the third distinct instance in this log of a
resource binding at one query width and not the other (H49 memory, H55
registers, now H57 latency).

## Verified state at `da0198de`

Both boxes resynced to HEAD, parity re-checked (`5939c346...`, recall
0.8030), all eight cells re-measured with the hardened nq=1 harness:

| cell | main | now | speedup |
|---|---|---|---|
| arm nq100 MT | 41.342 ms | 14.279 ms | x2.895 |
| arm nq100 ST | 308.037 ms | 120.832 ms | x2.549 |
| arm nq1 MT | 0.593 ms | 0.533 ms | x1.112 |
| arm nq1 ST | 4.053 ms | 3.581 ms | x1.132 |
| x86 nq100 MT | 61.984 ms | 19.507 ms | x3.178 |
| x86 nq100 ST | 241.659 ms | 100.343 ms | x2.408 |
| x86 nq1 MT | 2.460 ms | 1.082 ms | x2.273 |
| x86 nq1 ST | 9.468 ms | 4.201 ms | x2.254 |

**Harmonic mean x1.9353** (arithmetic x2.2252). Eleven confirmed
improvements; arm nq1 MT is the weakest cell and the standing target.

### The session's structural finding

**The same kernel family is limited by different resources at different
query widths, and that is what decides whether a fix transfers.**

| | arm | x86 |
|---|---|---|
| nq=1 | latency-bound on chains (H42/H44/H45) | memory-stream-bound (H54) |
| nq=100 | *not* latency-bound (H57) | compute-bound (P20) |
| nq=1 spare registers | none — 16 chains in use (H55) | 30 — only 2 accumulators |

Every transplant that failed today failed on this: H36 (half-blocks x86),
H49 (chains x86), H53 (quarter-blocks batched), H55 (streams arm), H57
(partials batched). Both that succeeded — H45 and H54 — worked because the
binding resource was named *before* the technique was copied.

Operating rule, earned five times over: **before transplanting a fix,
identify which resource was binding in the source context and check whether
it binds in the target.** A refutation should record which resource it
failed to move, not merely that it failed — H43 and H49 recorded only the
latter, and between them they already contained H54's answer.

## P27 — a single-query cliff below 1024 blocks (outside the goal cells)

Sweeping N at nq=1 **multi-threaded** on arm, chasing the weakest cell:

| N | codes | ms | ns/(q*vec) | GB/s |
|---|---|---|---|---|
| 5 000 | 2 MB | 0.10 | 20.18 | 19.0 |
| 12 500 | 5 MB | 0.21 | 16.90 | 22.7 |
| 25 000 | 10 MB | **0.39** | 15.54 | 24.7 |
| 50 000 | 19 MB | **0.17** | 3.44 | 111.6 |
| 100 000 | 38 MB | 0.28 | 2.84 | 135.2 |
| 200 000 | 77 MB | 0.58 | 2.91 | 132.2 |
| 400 000 | 154 MB | 1.12 | 2.81 | 136.9 |

**A 25k index answers a single query in 0.39 ms; a 50k index answers the
same query in 0.17 ms.** Twice the data, 2.3x less time — an absolute
regression, not just a worse rate.

`SINGLE_QUERY_PARALLEL_MIN_BLOCKS` is tied to `MIN_TILE_BLOCKS = 1024`, so
below ~33k vectors a single query never enters the pool and runs on one
core. The constant's doc comment justifies the tie — "a single query must
not be routed into the pool at a size where the same work, batched, would
not have been worth splitting (#336)" — but the two decisions are not the
same: a batch has a query axis to parallelize over and a single query has
only the block axis. The floor that is right for one is a cliff for the
other.

Also refutes an L3 story I was about to assert: the rate is **flat at
~135 GB/s from 38 MB to 154 MB**, well past Axion's 80 MB L3, so the arm
nq=1 MT cell is not cache-capacity-bound.

**Not fixed here, and it does not move the goal metric** — all eight cells
are at N=200k, far above the threshold. Recorded because it is a real
user-facing cliff for small indexes, and because chasing the goal's weakest
cell is what surfaced it. The fix is to give the single-query gate its own
floor rather than inheriting the batch tile size; the risk is thread-pool
overhead dominating at genuinely small N, which is what the constant was
protecting against, so it needs its own sweep from ~64 blocks upward.

## H58 — pipeline the A-operand loads: null

H47 pipelined the *code* loads and found the out-of-order window already
covered them. The A operands looked like a different case: both feed all 16
SMMLA pairs of a q8 unit, so at latency 6 they sit at the head of the
iteration with nothing able to issue before them, where a code load feeds
only its own register.

| nq=1 arm ST | shipped | H58 |
|---|---|---|
| median | 3.653 ms | 3.703 ms |

Consistently ~1.4% slower. Null, and the reasoning was wrong: the loads for
q8+1 can issue during q8 regardless of who wrote the source order, because
they depend on nothing in the iteration. Carrying them in registers across
the loop just extends two live ranges.

**Every load-scheduling explanation for arm nq=1's 66% is now eliminated**:
instruction count (P25), load count (H46), code-load latency (H47),
A-operand latency (H58), prefetch (H48), chain count (H44/H45). The gap is
not in the load path.

## P28 — P20's "nq=100 is compute-bound" has expired

Re-running P20's N-sweep on the current x86 kernel, nq=100 ST:

| N | codes | P20 (pre-H34) | now |
|---|---|---|---|
| 100k | 38 MB | 6.47 ns/(q*vec) | 3.97 |
| 200k | 77 MB | 6.65 (+3%) | **4.96 (+25%)** |
| 400k | 154 MB | 6.76 (+4%) | **5.32 (+34%)** |

P20 concluded "both kernels are compute-bound; prefetch is off the table"
and that conclusion shaped six later hypotheses. It was correct when
measured. **It is no longer true.** H34, H38 and H54 made the kernel ~2.4x
faster without changing how many bytes it reads, so the same code array now
arrives too slowly, and crossing out of cache costs 34% instead of 4%.

*A performance fact has a shelf life set by the code it was measured on.*
This log has repeatedly caught refutations that were too narrow in **scope**
(H23, H42, H44, H49, H54); this is the first that went stale in **time**. A
measurement that pins a bottleneck is invalidated by any change that moves
the other side of the balance — and every confirmed win does exactly that.

Consequence: the x86 nq=100 cells now sit in the regime where H54's stream
argument applies, and H54 is the only lever this log has found for it.
Whether it fits is a register question — at NQ=8 the kernel holds 16
accumulators and BLK=2 needs 32 — so the honest options are NQ=4 with BLK=2
(16 accumulators, but H35 measured NQ=4 alone at x0.636, a deficit streams
are unlikely to repay) or half-blocks with BLK=2 (H36 measured x0.80 alone).
Both start from a large hole; recorded with the arithmetic so the next
attempt starts from it rather than rediscovering it.

Also worth re-testing on this basis: **H35's batch-width sweep and H43's
prefetch refutation were both measured before H54**, and both were about the
memory side on x86.

## H59 — prefetch on x86, re-tested after P28: **HM x1.935 -> x1.985**

**The same code H43 refuted.** Byte for byte: one line per 64-byte load at
an 8-quad lookahead. H43 measured it neutral at both shapes and closed
prefetch on x86. P28 then showed the cell it was measured in had changed
underneath — H34, H38 and H54 made the kernel 2.4x faster without changing
its byte traffic, so crossing out of cache went from costing 4% to 34%.

Re-run in the new regime, three interleaved rounds, medians:

| x86 cell | H54 | **H59** | |
|---|---|---|---|
| nq=100 ST | 102.57 ms | **97.27 ms** | +5.2% |
| nq=1 ST | 4.385 ms | **3.696 ms** | **+18.6%** |
| nq=1 MT | 1.052 ms | **0.998 ms** | +5.2% |
| nq=100 MT | 19.447 ms | 19.363 ms | +0.4% |

All four x86 cells improve. Full 8-cell, bit-identical (`5939c346...`),
133/133 green:

| cell | main | now | speedup |
|---|---|---|---|
| arm nq100 MT | 41.342 ms | 14.279 ms | x2.895 |
| arm nq100 ST | 308.037 ms | 120.832 ms | x2.549 |
| arm nq1 MT | 0.593 ms | 0.533 ms | x1.112 |
| arm nq1 ST | 4.053 ms | 3.581 ms | x1.132 |
| x86 nq100 MT | 61.984 ms | 19.363 ms | **x3.201** |
| x86 nq100 ST | 241.659 ms | 97.267 ms | **x2.484** |
| x86 nq1 MT | 2.460 ms | 0.998 ms | **x2.465** |
| x86 nq1 ST | 9.468 ms | 3.696 ms | **x2.562** |

**Harmonic mean x1.9846**, from x1.9353.

### This is the most expensive lesson in the log

H43 was a correct measurement, competently implemented, and it closed a
direction for **sixteen hypotheses**. The code was right the whole time; the
machine's balance was not yet in a state where it paid. Nothing about H43
was wrong except the assumption that a refutation stays refuted.

Combined with H23 (refuted against a kernel that was later replaced), H42
(refuted at a query width nobody re-checked) and H49/H54 (refuted without
naming the resource), the pattern is now unambiguous and worth stating as a
standing rule for this log:

> **Re-test refuted hypotheses after any confirmed win that moves the
> resource they were measured against.** A refutation is a measurement of a
> moment, not a property of the idea.

Concretely owed a re-test on the same grounds: **H35** (x86 batch width) and
**H36** (x86 half-blocks), both measured pre-H54 on the memory side, and
**H46/H48** on arm if arm ever crosses into the same regime.

## H60 — x86 batch width, re-tested under the new bottleneck: H35 stands

H59's rule names H35 as owed a re-test, and batch width is the most
plausible flip: NQ=16 halves the sweeps over the code array (6.25 vs 12.5 at
nq=100), which is exactly the resource P28 showed had become binding. If
memory now dominates, the wider batch's halved traffic should outweigh the
spills that sank it before.

nq=100 x86 ST, three rounds:

| NQ | **8 (shipped)** | 12 | 16 |
|---|---|---|---|
| | **99.04 / 99.67 / 100.03** | 108.05 / 110.79 / 108.09 | 106.33 / 106.33 / 106.60 |

H35 stands, unchanged. NQ=16 is 6.5% worse despite reading half the bytes.

**The register cliff is steeper than the memory saving.** At NQ=16 the
accumulators alone are 32 zmm before the broadcasts, so the kernel spills on
every iteration of the hot loop — a per-quad cost — while the traffic saving
is amortized across a whole sweep. Halving something you touch once per pass
cannot pay for doubling something you touch 96 times per block.

Worth recording as the counterweight to H59: **re-testing after a bottleneck
shift is obligatory, not automatically productive.** Two hypotheses were
owed the same re-test on identical reasoning; one flipped decisively (H43 ->
H59, +2.5% on the score) and one did not move at all. The rule earns its
keep on the first and costs little on the second, which is the trade that
makes it worth following.

H36 (x86 half-blocks) remains owed the same treatment.

## H61 — half-blocks + NQ=16 on x86: designed, not yet attempted

The last hypothesis owed a re-test under H59's rule, and the most promising
of the three. H36 measured `hb16` — half-blocks at NQ=16 — at **x1.050 ST**
against full-block NQ=8, losing only on MT (x0.864). Two things have changed
since:

- **P28**: memory became binding at nq=100, and NQ=16 halves the sweeps over
  the code array (6.25 vs 12.5), which is the resource that now costs.
- **H60**: NQ=16 with *full* blocks loses 6.5% because 32 accumulators spill
  every quad. Half-blocks hold one accumulator per query instead of two, so
  NQ=16 needs 16 — the exact register room the spill needs.

So the two failures compose into a candidate: H36's ST win was measured
before memory mattered, and H60's loss is caused by the thing half-blocks
fix.

**Not attempted here, and the reason is mine rather than the code's**: H54
added a `BLK` dimension to the accumulator array (`[[[_; 2]; NQ]; BLK]`), so
the half-block restructure now has to thread through that too, and I do not
have the context left to do it and verify it properly. A half-applied
version of this change compiles and silently mis-scores — the failure mode
H41's `native_to_seq` and P23's void probe both had — so it is better left
clean than left broken.

For whoever picks it up: `BLK=1` on the batched path, so the half-block
split only has to be correct for the `sub == 0` case; the two halves read
disjoint 64-byte runs of each 128-byte unit, so no byte is read twice; and
the prefetch from H59 must move inside the half loop with `h` in its offset.

## P29 — P28 has no arm twin: arm is still compute-bound at nq=100

P28's mechanism was general — a kernel that gets faster without changing its
byte traffic eventually outruns memory — so arm was owed the same re-check.
Its `st_roofline` flatness was last measured pre-H33/H41, since when the arm
kernel got ~2.5x faster.

| N | codes | arm ns/(q*vec) |
|---|---|---|
| 100k | 38 MB | 6.22 |
| 200k | 77 MB | 6.13 |
| 400k | 154 MB | **5.69** |

**Still flat, and if anything faster at 400k.** arm sustains ~8 GB/s at
nq=100 and never leaves the compute-bound regime. So:

- **Prefetch is closed on arm for good**, at both query widths and for a
  reason rather than a measurement: there is no memory cliff to hide. H48's
  refutation at nq=1 was not a scoping accident after all.
- **The asymmetry is arm's own speed.** x86 crossed into memory-bound
  because H34/H38/H54 made it fast enough to outrun its memory system. arm's
  nq=100 ST is 122.7 ms against x86's 97.3 — arm simply does not demand
  bytes fast enough to hit the wall, which is the same fact stated as a
  weakness.

**x86 has now overtaken arm at nq=100 ST** (97.3 vs 122.7 ms) having been
behind it for most of this climb. arm still leads nq=100 MT (14.3 vs 19.4).

The consequence for the remaining search: arm's four cells cannot be helped
by anything memory-shaped, and its unpack is at the ISA floor (P25) with the
codebook permanently non-uniform (H50). What is left for arm is scheduling
inside a loop already measured at 66% of its issue bound, where six distinct
explanations have now been eliminated.

## P30 — why arm is behind x86 at nq=100, and why that is structural

P29 left arm 26% slower than x86 at nq=100 ST (122.7 vs 97.3 ms). Working
out whether that is a fixable inefficiency or a property of the ISA:

**Peak MAC rate is identical.** arm: 4 SMMLA/cycle x 32 MACs = 128
MACs/cycle. x86: 2 VPDPBUSD/cycle x 64 MACs = 128 MACs/cycle. Four 128-bit
pipes against two 512-bit ports come out level.

**Achieved**: arm 41.8 MACs/cycle (33% of peak), x86 52.7 (41%).

**The gap is unpack amortization, and it follows from register width.** The
nibble unpack costs a fixed ~5 instructions per register regardless of what
that register holds:

| | register | MACs per unpack | ops per MAC |
|---|---|---|---|
| arm | 128-bit | 256 | **0.051** |
| x86 | 512-bit | 1024 | **0.036** |

x86 spreads the same five instructions over four times the data. That is not
something a better arm kernel can recover: the unpack is at the ISA floor
(P25, confirmed against llama.cpp's identical sequence), the codebook is
permanently non-uniform (H50), and SVE is 128-bit on V2 so it offers no
width (P22).

**So arm's nq=100 cells are close to their structural limit**, and the
26% deficit is the price of 128-bit registers on a workload whose fixed cost
is per-register rather than per-byte. The remaining arm headroom is the
scheduling gap at nq=1 — 66% of the issue bound, with six explanations
eliminated (instruction count, load count, code-load latency, A-operand
latency, prefetch, chain count) and no seventh candidate that does not need
either hardware counters (unavailable on both boxes, P23) or SVE register
classes (unstable in Rust, H52).

Stated plainly so the next session does not re-derive it: **arm nq=100 is
near its ceiling, arm nq=1 has a real but unexplained 34%, and both x86
widths have moved into a memory regime where H59's prefetch already
collected the available win.**

## H62 — sweep the prefetch distance: **HM x1.985 -> x2.041**

H59 inherited H43's lookahead of 8 quads without sweeping it — the one
parameter a prefetch actually has. nq=100 x86 ST:

| PF quads | 2 | 4 | **8** | 16 | **32** | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|
| ms | 97.3 | 97.3 | 96.4 | 82.8 | **75.4** | 76.5 | 76.0 | 77.6 |

**32 is worth x1.28 over the 8 that shipped**, and the plateau from 32 to
128 says it is the knee rather than the edge of the sweep.

But 32 measured as a **regression at nq=1** (-5.7% ST, -7.0% MT) while
winning +24.9% at nq=100 — so the distance is not a property of the machine,
it is a property of the access pattern. nq=100 sweeps the code array 12.5
times and a 4 KB lookahead stays useful; nq=1 sweeps it once, so the same
depth runs ahead of what the scan will reach before eviction. The shipped
value now depends on `NQ`: 8 at nq=1, 32 otherwise.

Full 8-cell, bit-identical (`5939c346...`), 133/133 green:

| cell | main | now | speedup |
|---|---|---|---|
| arm nq100 MT | 41.342 ms | 14.279 ms | x2.895 |
| arm nq100 ST | 308.037 ms | 120.832 ms | x2.549 |
| arm nq1 MT | 0.593 ms | 0.533 ms | x1.112 |
| arm nq1 ST | 4.053 ms | 3.581 ms | x1.132 |
| x86 nq100 MT | 61.984 ms | 18.827 ms | **x3.292** |
| x86 nq100 ST | 241.659 ms | 75.156 ms | **x3.215** |
| x86 nq1 MT | 2.460 ms | 0.999 ms | x2.462 |
| x86 nq1 ST | 9.468 ms | 3.578 ms | **x2.646** |

**Harmonic mean x2.0414**, from x1.9846. First reading above x2.

**A parameter inherited from a refuted experiment is not a tuned
parameter.** H59 was a real win and I shipped it carrying H43's arbitrary
constant, which turned out to be leaving 28% on the table in the cell H59
was specifically fixing. The re-test rule this log earned says to re-run
refuted *hypotheses* after a bottleneck moves; the corollary it missed is to
re-tune their *constants*, since those were chosen under the old bottleneck
too.

## H63 — TBL -> TBX on arm: null, and P25's model survives its own test

`TBX` with one table register is latency 2, throughput **4**, **all four V
pipes**; `TBL` is throughput 2, **V01 only** (SWOG Table 3-18). TBX differs
only in leaving out-of-range lanes untouched instead of zeroing them, and
our indices are always 0-15 by construction, so it is bit-identical. One
intrinsic swap, no SVE, no asm — `vqtbx1q_s8(idx, levels, idx)`, passing the
index vector as the dead fallback so no new register is needed.

Verified emitted: 72 `tbx` in the binary, none of the hot-loop `tbl` left.
Output bit-identical (`5939c346...`), 127/127 green.

| arm ST | TBL | TBX |
|---|---|---|
| nq=1 | 3.666 / 3.658 / 3.678 | 3.714 / 3.655 / 3.654 |
| nq=100 | 122.41 / 121.69 / 122.30 | 119.45 / 120.93 / 123.83 |

Null at both widths — nq=100 shows +1.1% on medians but the ranges overlap.
Reverted.

**This is the cleanest confirmation the log has that arm is µop-count-bound
rather than port-bound.** P25 predicted exactly this: the bound is 6 µops
over 4 pipes = 1.5 cycles, and V01 has 14 slots in that window for 4 TBLs,
so freeing them changes nothing. Two independent research passes reached the
same prediction and the measurement now agrees with both. The 34% gap is not
`V01`, and that is now a measured fact rather than a modelled one.

### Corrections to P26 from the second research pass

- **turbovec is not "instruction-for-instruction identical" to llama.cpp.**
  Counted at master: turbovec **4.57** (vector,dim) pairs per instruction,
  `ggml_gemv_q4_0_4x8_q8_0` 3.76, `ggml_gemv_iq4_nl_4x4_q8_0` **3.37** — and
  IQ4_NL is the true analogue since it also does arbitrary-codebook TBL
  dequant. We are ~36% denser, not equal. P26 recorded the weaker claim.
- **Against published measured numbers we are well ahead**: best measured
  int8 scan on Neoverse-class arm is ~7.6-8.8 pairs/cycle (Kuffo & Boncz,
  DaMoN'25, arXiv:2505.07621 Table 6; SimSIMD on Graviton3); turbovec is at
  ~14.1.
- **Arm's own team abandoned arbitrary 4-bit codebooks for this reason.**
  arXiv:2501.00032 (Gope, Mansell, Loh, Bratt) interleaves nibbles offline
  and XORs with `0x8888` so runtime dequant is a shift and a mask with **no
  TBL at all**, folding the x16 into the FP scale — 190 -> 571 tok/s on
  64-core Graviton. It needs a uniform quantizer, so H50 closes it here; the
  notable part is that Arm's engineers judged two TBLs not worth paying at 4
  bits and reserve them for 2-3 bits.
- **KleidiAI has 26 micro-kernels at M-tile 1 and zero use i8mm** — the
  lowest M-tile with any i8mm kernel is 4. llama.cpp gates SMMLA behind
  `if (nrc == 2)` and falls through to SDOT at `nrc == 1`. Both confirm that
  SMMLA-at-M=1 is a road the field has not taken, which is what turbovec's
  duplicated-pair single-query kernel does.

## P31 — llvm-mca cannot answer the arm question, and nearly gave a wrong answer

The research's most actionable suggestion was a static analyzer, since
neither box exposes a PMU (P23). Tried it on the nq=1 inner loop
(`arm_nq1_loop.s`):

- **llvm-14**: `'neoverse-v2' is not a recognized processor` — the V2
  scheduling model landed later. Using `neoverse-v1` as a proxy gives
  Block RThroughput **5.0** cycles and 11000 µops for 7000 instructions
  (V1 cracks SMMLA into 2 µops), against 2.27 measured. Not a usable proxy.
- **llvm-16**: accepts `-mcpu=neoverse-v2` and reports Block RThroughput
  **2.5** cycles per 16 bytes — which, against 2.27 measured, would have
  said *the loop is already at its ceiling and the 34% gap never existed*.

**That conclusion would have been wrong, and I was one commit from it.** The
resource names in the pressure table are `N2UnitV0`, `N2UnitV1`, `N2UnitS`,
`N2UnitM0` — **llvm-16 silently aliases `neoverse-v2` to the Neoverse N2
model**, which has *two* vector pipes where V2 has four. The 2.5-cycle
figure is for the wrong core, and it is 2.5 rather than 3.0 only because N2
also models SMMLA differently.

Two things worth keeping:

1. **A tool accepting a `-mcpu` string is not evidence it models that CPU.**
   llvm-mca printed no warning; the only tell was the resource *names* in a
   view I had to ask for separately. The V2 model (`AArch64SchedNeoverseV2.td`,
   which P22 cited from LLVM main) needs **LLVM 17+**, not available in
   Debian 12's archive.
2. **The 1.5-cycle floor is still unverified in either direction.** It came
   from my own µop arithmetic over the SWOG tables, and nothing has now
   confirmed *or* refuted it. H63 established the gap is not `V01`; that is
   the only hard fact about it.

### Resolved: llvm-18 from apt.llvm.org, and **the 34% gap never existed**

Installed LLVM 18 and re-ran. Confirmed genuine this time — the resource
table names `V2UnitV0..V2UnitV3` (four vector pipes) and dispatch width 16,
against llvm-16's two-pipe `N2Unit*`.

| | cycles per 16 bytes |
|---|---|
| my hand-derived floor (6 µops / 4 pipes) | 1.50 |
| **llvm-18 Neoverse V2 model** | **2.015** |
| **measured** | **2.27** |

**arm nq=1 is at 89% of its modelled ceiling, not 66%.** The 1.5 figure was
my own arithmetic over the SWOG tables and it was wrong: dividing µops by
pipe count ignores everything the real model accounts for — `TBL` pinned to
`V01` while `USHR` needs `V13` (both contend on V1), the load on `L01`, and
dispatch grouping. The true headroom was always ~11%, not ~34%.

**Eight hypotheses were aimed at a number I computed rather than measured.**
H44, H45, H46, H47, H48, H57, H58 and H63 all targeted "the missing 34%".
H45 found a real 10% inside it — the spill was genuine — and the remaining
seven found nothing because after H45 there was little left to find. Each
was individually well-reasoned and correctly measured; the error was
upstream of all of them, in a back-of-envelope bound I never checked against
a model or a tool.

*Derive a ceiling, then verify it before spending hypotheses on the gap.*
The verification here cost one `apt-get install` and ten minutes, after
eight hypotheses had been spent.

The arm nq=1 thread is closed on this basis: ~11% remains against a static
model, the loop is µop-count-bound (H63 measured it is not `V01`), the
unpack is at the ISA floor (P25), and the codebook is permanently
non-uniform (H50). `llvm-mca -mcpu=neoverse-v2` under LLVM 18 is now the
right first stop for any future arm kernel question, and `arm_nq1_loop.s`
is checked in for it.

## P32 — both kernels are now at or past their static-model throughput

P31's lesson applied to x86, where the bounds behind P22 and P28 ("~16
cycles per quad") were also hand-derived and never checked. Modelled the
shipped nq=100 inner loop (`x86_nq100_loop.s`) with llvm-18:

| | cycles per byte-group quad |
|---|---|
| llvm-18 `-mcpu=sapphirerapids` | **34.1** (17.03 per 64-byte half) |
| **measured** (75.156 ms nq=100 ST) | **30.1** |

**x86 runs ~13% faster than its own model predicts.** The model charges 37
µops for 21 instructions — each `vpdpbusd` with a `{1to16}` memory operand
is cracked into two — and evidently overstates that cost on real Sapphire
Rapids. Either way there is no modelled headroom to chase.

With P31, both arches are now measured against real scheduling models rather
than my arithmetic:

| | modelled | measured | position |
|---|---|---|---|
| arm nq=1 | 2.015 cyc/16B | 2.27 | 89% of model |
| x86 nq=100 | 34.1 cyc/quad | 30.1 | **113% of model** |

**The kernel-tuning phase is over.** Thirteen confirmed improvements took
the harmonic mean from x1.0 to x2.041, and both inner loops now sit at or
beyond what a static analyzer says the hardware allows. What remains is not
a faster loop:

- **arm**: ~11% against its model, in a loop that is µop-count-bound (H63),
  at the ISA floor for the unpack (P25), with the codebook permanently
  non-uniform (H50) and register width structurally capping amortization
  (P30).
- **x86**: nothing by this model. The wins that remain would come from
  reading fewer bytes or doing less work per byte — both closed by H50 —
  or from an algorithmic change outside this kernel.

Anything further should be aimed at the *algorithm* (fewer vectors scanned,
different index structure) rather than the scan, or at the small-N
parallelism cliff P27 found, which is a real user-facing win that this
goal's cells cannot see.

## P33 — arm nq=1 is **bandwidth**-bound, and P24 was misread

Modelled a full q8 unit rather than the idealised 7-instruction block —
2 A-operand loads, four groups of four code registers, loop overhead
(`arm_nq1_q8.s`), which is the shape the kernel actually runs:

| | cycles per 16 bytes |
|---|---|
| idealised block (P31) | 2.015 |
| **full q8 unit, llvm-18 V2** | **1.878** (µops/cycle 3.93, near the 4-pipe limit) |
| measured | 2.27 |

Then the number that matters. At 1.878 cycles per 16 bytes the kernel wants
**8.5 bytes/cycle**. Measured throughput is 18.5 GB/s at 2.993 GHz =
**6.2 bytes/cycle**. *The memory system cannot feed the loop*, and the
shortfall — 6.2 against 8.5 — is 27%, which is the gap.

**P24 was misread, by me, for eleven hypotheses.** It measured arm flat at
18-20 GB/s from 2 MB to 154 MB and I recorded that as "compute-bound at
every size". Flatness across cache levels does not mean compute-bound — it
means *single-core streaming bandwidth is the same whichever level serves
it*, which is the normal case when the limit is outstanding misses rather
than cache capacity. The correct reading was: arm sustains ~6.2 B/cycle no
matter where the data lives, and the kernel needs 8.5.

This retroactively explains **every** null on this cell — H46 (load count),
H47 and H58 (load pipelining), H57 (chains), H63 (TBX), and the H44/H45
chain-count results beyond the spill H45 genuinely fixed. None of them
changed how many bytes per cycle the core can pull.

It also **reopens H55**, which was closed on register arithmetic without a
measurement. H54 won on x86 by adding memory *streams*, and this says arm
nq=1 has the same disease. H55's blocker stands — 16 accumulators are needed
for ILP and two blocks would want 32 — but that arithmetic assumed the
compute bound was binding. If the core is starved at 6.2 B/cycle, **fewer
chains may be affordable**: at 8 chains the compute bound rises to roughly
2.4 cycles/16B, still under what memory can supply, and H44 measured 8
chains at x0.69 *in a regime where compute was thought to bind*. That
experiment is owed a re-run under the correct model — the exact re-test
discipline H59 established.

## H64 — two blocks, eight chains on arm nq=1: designed, kernel written, not landed

P33's correction makes this the highest-value open hypothesis. The kernel
wants 8.5 B/cycle and the core delivers 6.2, so compute has slack: eight
chains model at ~2.4 cycles/16B, still inside what memory can supply. Spend
that slack on what x86 showed is actually scarce (H54, x1.29): independent
memory streams.

Kernel written and compiling — two blocks in flight, eight accumulators
each, 16 chains total across two streams, ~24 registers. It is the **callers**
that block it: both arm single-query sites loop one block per iteration with
a single `[[f32; BLOCK]; 1]` output row, and this needs them stepping by two
with two rows.

### Landed, measured, **refuted at x0.57**

| nq=1 arm ST | 16 chains, 1 stream | 8 chains, 2 streams |
|---|---|---|
| | 3.673 / 3.684 / 3.670 | **6.470 / 6.345 / 6.434** |

Correct throughout — 127/127, `score md5 5939c346...`, recall 0.8030 through
a fresh write path — and **43% slower**. Worse even than H44's x0.69, which
had the same eight chains without the strided halves.

**This refutes the stream argument for arm and closes P33's reopening of
H55 for good**, on the falsification condition the design set in advance.

It also bounds P33 itself. The bandwidth diagnosis — 8.5 B/cycle wanted
against 6.2 delivered — may well be right about *why* the loop sits at 2.27
rather than 1.878. But the inference drawn from it, that compute therefore
has slack worth trading, is **wrong in practice**: eight chains do not model
at 2.4 cycles/16B in the real loop, they collapse. Bandwidth being the
binding constraint does not mean the compute side is free to degrade, because
the two are not independent — fewer chains means less memory-level
parallelism too, and the same accumulators that feed the pipes are what keep
loads in flight.

*A resource diagnosis says what is scarce. It does not license spending
whatever else looks abundant.* Three of this log's failures now share that
shape: H36 (register room bought at the cost of a second pass), H49 (chains
added where memory bound), and now H64.

For the attempt:
- Kernel signature gains `nb: usize` and `out: &mut [[f32; BLOCK]; 2]`;
  the `half` loop stays outermost, `blk` inside `q8`, `r` in `0..8`
  indexing `half * 8 + r`.
- Both call sites are `for b in ..` loops that must become `step_by(2)`
  with `nb = 2.min(range_blocks - b)`, and the per-lane heap fold below each
  must run for both rows.
- H44 measured eight chains at **x0.69** — under one stream and the wrong
  model. If H64 does not beat that, the stream argument is refuted for arm
  and P33's reopening of H55 closes for good.
- Gate on `load_parity.py` (`5939c346...`) before timing: two blocks with a
  shared `raw` buffer is exactly where an indexing slip stays silent.

## P34 — the vPMU is reachable, but only by rebuilding the box

Every mechanism question left on arm needs hardware counters, and P23
recorded them as unavailable: `perf` installs but every event reads
`<not supported>` on both boxes. That was true of the *running* instances;
it is not the whole story.

`gcloud` (SDK 519) exposes **`--performance-monitoring-unit`** — but on
`instances create` only. It is absent from `instances update`, so the vPMU
cannot be toggled on an existing VM:

    gcloud compute instances create --help | grep performance-monitoring-unit
      --performance-monitoring-unit=PERFORMANCE_MONITORING_UNIT     # present
    gcloud compute instances update --help | grep performance-monitoring-unit
      (nothing)

So unlocking counters means **recreating** `turbovec-bench-arm-search`
(c4a-standard-8, us-central1-a, 34.28.97.62) from its boot-disk image with
the flag set. The rig note at the top of this log records that both boxes
were built that way already — `gcloud compute images create --source-disk`,
since ARM machine images are unsupported — so the procedure exists and is
known-good.

**Not done here.** It is destructive to a working rig mid-climb: the
instance carries `~/tv-hc`, the venv, the cached 200k index files and every
`so_*.so` baseline this log compares against, and a fresh ephemeral IP would
break the scripts. The right time is at the start of a session, not the end
of one.

What it would buy: attribution for the one number this log could never
explain. P33 infers arm nq=1 is bandwidth-starved (8.5 B/cycle wanted, 6.2
delivered) from a static model plus a throughput measurement; a single
`perf stat` on stall cycles and L1/L2 miss counts would confirm or kill that
inference directly, and H64's refutation makes the inference load-bearing —
it is the only remaining account of the gap, and the one experiment it
motivated failed.

Values to try: `ARCHITECTURAL` is the portable baseline; `STANDARD` and
`ENHANCED` expose progressively more counters. For arm, `ENHANCED` is what
would give the Neoverse V2 stall-slot events worth having.

## P35 — hardware counters, at last: P33 confirmed, P23 superseded

P34 found `--performance-monitoring-unit` is create-time only. The
non-destructive move is a **second** box rather than rebuilding the rig:

    gcloud compute instances create turbovec-bench-arm-pmu \
      --zone=us-central1-a --machine-type=c4a-standard-8 \
      --image=turbovec-bench-arm-img --boot-disk-type=hyperdisk-balanced \
      --performance-monitoring-unit=standard

(`enhanced` is rejected on c4a — x86 only. Lowercase values.) Live at
**35.232.9.182**, the working rig untouched. Counters report real numbers:

**P23 was wrong as stated.** "No hardware counters on either box" was an
instance-configuration fact recorded as a hardware one, and it stood for
twelve hypotheses. Every mechanism question on arm was answered by
inference because of a flag nobody checked.

### The measurement

`perf stat` over the nq=1 scan (arm, ST):

| | |
|---|---|
| IPC | **3.53** (4-wide machine) |
| **backend cycles idle** | **34.88%** |
| frontend cycles idle | 2.78% |
| L1-dcache-load-misses | 265,815,952 |

**This confirms P33 directly.** 34.9% backend idle against 2.8% frontend is
the signature of a core waiting on data, not on instruction supply or
decode — and 34.9% is the gap this log spent eleven hypotheses hunting. The
front end is essentially never the problem, which retroactively explains
H39's unrolling loss and H47/H58's pipelining nulls: all three targeted
instruction supply, which was never idle.

It also **vindicates H64's refutation rather than contradicting it**. The
core is memory-stalled, but H64 showed you cannot buy that back by trading
chains for streams, because chains *are* what keep loads outstanding. Both
facts hold: the stall is real, and the obvious lever for it is not available.

### Immediately qualified: nq=100 has the *same* profile

Ran the same `perf stat` over nq=100 ST:

| | nq=1 | nq=100 |
|---|---|---|
| IPC | 3.53 | **3.53** |
| backend cycles idle | 34.88% | **35.69%** |
| frontend cycles idle | 2.78% | 3.23% |

**Identical, and that weakens the bandwidth reading rather than confirming
it.** nq=100 amortizes the code array across 8 queries per sweep, so its
bytes-per-unit-compute is ~8x lower than nq=1. If the backend stalls were
memory, the two figures should differ sharply. They do not.

So "backend cycles idle" on this PMU is not a memory counter — it covers
execution-resource stalls too (SMMLA latency and issue among them), and the
more parsimonious reading of a constant 35% across an 8x change in memory
intensity is **execution-side, not bandwidth**.

**P33 is therefore not confirmed, and the paragraph above overstated it.**
What the counters establish is narrower and still useful: the front end is
never the constraint (2.8-3.2% idle at both widths), IPC is 3.53 of 4, and
whatever the backend waits on is *the same thing at both query widths*. That
last fact is new and is not explained by any account in this log — P24/P29
said compute at nq=100, P33 said bandwidth at nq=1, and a single shared
cause fits neither.

### Resolved: `stall_backend_mem` splits it

The PMU exposes `stall_backend_mem`, which attributes backend stalls to
memory specifically. Both widths, ST:

| | cycles | `stall_backend` | **`stall_backend_mem`** | non-memory backend |
|---|---|---|---|---|
| nq=1 | 11.09e9 | 35.8% | **13.9%** | **21.9%** |
| nq=100 | 4.64e9 | 43.2% | **18.4%** | **24.8%** |

**Memory is real but secondary at both widths.** It accounts for 14-18% of
cycles — roughly 40% of the backend stalls — while the larger share, 22-25%
of cycles, is non-memory execution-resource stalls. The proportions barely
move across an 8x change in memory intensity, which is why the generic
counter looked flat.

So the final attribution of arm's gap, after this log spent eleven
hypotheses on it:

- **~22% of cycles: execution-resource stalls.** The dominant term, stable
  across query width. Consistent with a loop whose SMMLA and TBL chains are
  latency- and port-limited rather than starved — and with H63, which found
  the gap is not `V01` specifically, and H64, which found chains cannot be
  traded away.
- **~14% of cycles: memory.** P33's story, at roughly 40% of the size it
  claimed. It was the right mechanism at the wrong magnitude, which is
  exactly why H64 (built entirely on it) failed at x0.57.
- **~3% frontend.** Never the constraint, at either width.

*The counters did not confirm any hypothesis in this log; they replaced all
of them with a split none had proposed.* P24/P29 said compute at nq=100,
P33 said bandwidth at nq=1, and the truth is both, in fixed proportion, at
both widths. Eleven hypotheses argued over which single mechanism it was
because none of them could see 14 and 22 as separate numbers.

*Recorded as a correction to the entry directly above, written minutes
earlier.* A counter whose name matches a hypothesis is not evidence for it;
`stalled-cycles-backend` sounds like memory and is not.

### What this unlocks

Every future arm kernel question can now be attributed instead of inferred,
on a box that costs nothing to keep and does not disturb the baselines. The
immediate follow-ups: `perf stat` the nq=100 kernel to see whether its
backend idle differs (P29 says arm never leaves the compute-bound regime
there — now checkable), and record whether the 34.88% moves under the
kernels this log already refuted, which would say whether they failed for
the reason claimed.

## P36 — arm is at 88% issue utilization; the cells are done

One number from P35's data settles what the split means in practice.
**IPC is 3.53 on a 4-wide machine — 88% of issue capacity** — at *both*
query widths. The 22% execution-resource stalls are not a fixable
inefficiency; they are what 88% utilization looks like from the other side.

Three independent routes now agree the arm kernel is essentially done:

| route | position |
|---|---|
| llvm-18 V2 model (P33) | 1.878 vs 2.27 cyc/16B = **83%** |
| IPC / issue width (P36) | 3.53 / 4 = **88%** |
| stall attribution (P35) | 3% frontend, 22% execution, 14% memory |

The residual is memory stalls that overlap imperfectly with execution, and
every mechanism for attacking those has been measured and refuted: streams
(H64, x0.57), chains (H44/H57), prefetch (H48), load scheduling
(H46/H47/H58), ports (H63). The unpack is at the ISA floor (P25) and the
codebook is permanently non-uniform (H50).

**Closing position on the goal's eight cells.** x86 measures *above* its own
static model (P32, 113%); arm sits at 83-88% of three different ceilings
with no untried mechanism. Thirteen confirmed improvements took the harmonic
mean from x1.0 to **x2.0414**. Further gains need one of:

1. **Algorithmic** — scanning fewer vectors, a different index structure.
   Outside this kernel and outside this goal's framing.
2. **P27's small-N cliff** — a real 2.3x user-facing regression below ~33k
   vectors that all eight cells are blind to, since they sit at N=200k.
3. **SVE register classes in stable Rust** (H52) — would enable the one
   documented lever left, and is a toolchain wait rather than work.

## H65 — x86 stream depth, re-swept after prefetch landed: **BLK 4 -> 8**

H56 swept `BLK` and found 4 and 8 indistinguishable. That was **before** H59
and H62 added prefetch — and stream depth and prefetch depth act on the same
resource, so the rule H59 earned names this a re-test. It flipped.

Clean interleaved smoke, nq=1 x86 ST, BLK=8 wins **all five rounds**:

| | BLK=4 | BLK=8 |
|---|---|---|
| median | 3.639 ms | **3.549 ms** |

Full cells, three interleaved rounds, medians — all four improve:

| x86 cell | BLK=4 | BLK=8 | |
|---|---|---|---|
| nq=1 ST | 3.824 ms | **3.499 ms** | **+9.3%** |
| nq=1 MT | 1.099 ms | **1.056 ms** | +4.1% |
| nq=100 ST | 75.364 ms | **74.530 ms** | +1.1% |
| nq=100 MT | 18.859 ms | 18.800 ms | +0.3% |

Bit-identical (`5939c346...`), recall 0.8030, 133/133 green.

**Measurement caveat, stated rather than buried.** This session's absolute
x86 numbers have drifted from the run that set HM x2.0414 — `nq1_mt` reads
1.099 for the *same* build that measured 0.999 earlier. So the honest claim
is the **within-run** result: BLK=8 beats BLK=4 on all four cells measured
against each other in the same rounds. Re-baselining the full 8-cell score
against `origin/main` needs a fresh paired run on both boxes, and the score
should not be quoted from mixed-session numbers.

**Third flip from the same rule** (H43 -> H59, then H56 -> H65; H35 and H36
held). The pattern is now specific enough to state as a search strategy
rather than a caution: *when a win changes which resource binds, the
parameters of every neighbouring mechanism are stale — not just the
hypotheses that were refuted, but the constants of the ones that landed.*

## Re-baseline after H65 — and why the headline number is not quoted

Paired run, both boxes, two rounds each, main and HEAD alternating:

| cell | main | now | speedup |
|---|---|---|---|
| arm nq100 MT | 41.430 ms | 14.324 ms | x2.892 |
| arm nq100 ST | 311.622 ms | 121.407 ms | x2.567 |
| **arm nq1 MT** | **0.847 ms** | 0.564 ms | **x1.504 — see below** |
| arm nq1 ST | 4.082 ms | 3.641 ms | x1.121 |
| x86 nq100 MT | 62.001 ms | 18.821 ms | **x3.294** |
| x86 nq100 ST | 243.063 ms | 77.067 ms | **x3.154** |
| x86 nq1 MT | 2.467 ms | 1.083 ms | x2.277 |
| x86 nq1 ST | 9.537 ms | 3.711 ms | **x2.570** |

Harmonic mean computes to **x2.1382** — and it should not be quoted.

**One cell's baseline is bad.** `arm nq1 MT` main is the median of two
samples that read **1.078 and 0.616 ms** — a 75% spread — on the cell this
log already identified as the noisiest on the board (H51: a 0.955 reading
against a 0.54 median for the same build). Every earlier clean run puts main
at ~0.60, which would make that cell **x1.064**, not x1.504.

Substituting the trustworthy baseline gives **HM x1.99**. So the honest
statement is a range: **the score is between x1.99 and x2.14, and one
two-sample cell decides which.** The cell harness takes best-of-three
sub-runs for nq=1 to reject a perturbed *process*, but two rounds is not
enough to reject a perturbed *round* — the fix is more rounds on that cell,
not more reps inside them.

What is solid regardless: **x86 nq=100 ST reached x3.154** (from x2.401
before H59), and **x86 nq=1 ST x2.570**. H59, H62 and H65 — all three from
the same re-test rule — moved x86 nq=100 ST by 1.31x between them.

*A score that moves 7% on one cell's baseline sample is not a score yet.*
Quoting x2.1382 would have been the same error as reading
`stalled-cycles-backend` as memory: a number that supports the story,
adopted without checking what produced it.

### Resolved: eight rounds on the suspect cell

| arm nq=1 MT | median |
|---|---|
| main | 0.589 ms |
| HEAD | 0.5515 ms |
| | **x1.068** |

The 1.078 ms baseline sample was a perturbed round, as suspected — the true
figure is x1.068, not x1.504. **Today's paired score is therefore
x1.9936**, with the resolved cell substituted.

### That is *lower* than the x2.0414 recorded earlier, and both are honest

x86's cells read x2.277 / x2.570 today where the earlier session measured
x2.462 / x2.646 — for a build that is strictly better, since H65 improved
all four x86 cells *within* its own run. The boxes drift between sessions by
more than a single hypothesis is worth.

The consequence is a methodological one this log should have adopted much
earlier: **cross-session scores are not comparable, only within-session
paired ones are.** Every `x1.985 -> x2.041` style claim in this log is sound
only because main and HEAD were measured in the same interleaved rounds; the
absolute figures attached to them are not stable across days, and quoting a
running best across sessions overstates precision.

Today's honest position, all eight cells measured against main in the same
paired run: **HM x1.9936**, with x86 nq=100 ST at **x3.154** and x86 nq=100
MT at **x3.294** the strongest cells, and arm nq=1 MT at x1.068 the weakest.

## H66 — nq=1 prefetch distance under BLK=8: null, H62's value stands

H62 fitted the nq=1 lookahead at 8 quads while `BLK` was 4. H65 doubled BLK
to 8, changing the nq=1 access pattern that distance was fitted to — so the
rule names it. Re-swept, nq=1 x86 ST, three rounds, medians:

| PF quads | 2 | 4 | **8** | 16 | 32 |
|---|---|---|---|---|---|
| ms | 3.693 | 3.532 | **3.590** | 3.719 | 4.023 |

4 and 8 are indistinguishable (ranges overlap; PF=8's best round is the best
single reading in the sweep). 16 and 32 are clearly worse, confirming the
shape H62 found: deep lookahead helps the 12.5-sweep nq=100 pattern and
hurts the single-sweep nq=1 one. **H62's value stands.**

**Fourth firing of the re-test rule, second null** (H43->H59 flipped,
H35 held, H36 held, H56->H65 flipped, now H62's nq=1 constant holds). Two
flips in four is a good enough rate to keep applying it, and the nulls cost
one sweep each.

Worth noting what *did* stay stale-proof: the nq=1 and nq=100 lookaheads are
already separated by query width, so H65's change to `BLK` — which only
applies at NQ=1 — could not have disturbed the nq=100 value. Parameters that
are already conditioned on the thing that changed do not go stale.

## H67 — deep prefetch on arm nq=100: **+8.3% ST**

The gap the rule kept pointing at from two directions. **H48 refuted arm
prefetch at nq=1 only**, and nq=100 was closed by P29's compute-bound
reading — which **P35 undercut** by attributing 18.4% of nq=100 cycles to
memory stalls. H62 separately showed a 32-unit lookahead is worth +25% on
x86's 12.5-sweep pattern, which arm nq=100 shares.

`prfm pldl1keep` 32 q8-units ahead in the batched vm8 kernel:

| arm nq=100 ST | base | H67 |
|---|---|---|
| | 120.94 / 119.69 / 122.09 / 120.93 | **114.06 / 113.53 / 114.31 / 113.65** |

x1.062, no overlap — H67's worst round beats base's best. Full cells,
three interleaved rounds, medians:

| arm cell | base | H67 | |
|---|---|---|---|
| nq=100 ST | 123.112 ms | **112.912 ms** | **+8.3%** |
| nq=100 MT | 14.361 ms | 14.245 ms | +0.8% |
| nq=1 ST | 3.659 ms | 3.667 ms | -0.2% |
| nq=1 MT | 0.546 ms | 0.571 ms | -4.6% |

**The nq=1 deltas are noise, and structurally must be**: nq=1 dispatches to
`score_block_vm8_single`, a different function this change does not touch.
nq=1 ST reads -0.2% and nq=1 MT — the noisiest cell on the board (H51, and
the two-sample baseline that nearly corrupted the last re-baseline) — reads
-4.6%. A code path that cannot be reached cannot regress.

127/127 green, bit-identical (`5939c346...`), `prfm` confirmed emitted
before timing.

**Two stale conclusions had to fall for this to be found**, and both were
mine: H48's refutation was scoped to nq=1 and I recorded it as closing
prefetch on arm; P29 declared nq=100 compute-bound from the roofline
misreading P33 later corrected. Neither error was in a measurement — both
were in the sentence I wrote about one.

## H68 — arm prefetch distance: null, 32 stands

H67 took 32 units by analogy with x86's H62, without sweeping — and arm's
unit is 256 bytes where x86's quad is 128, so the byte-distance differs 2x.
Swept, arm nq=100 ST, three rounds, medians:

| PF units | 8 | 16 | **32** | 64 | 96 |
|---|---|---|---|---|---|
| ms | 123.14 | 117.33 | **115.24** | 119.68 | 114.43 |

Short lookahead (8) is clearly worse, confirming H67's mechanism. Beyond
that the sweep is **non-monotonic** — 64 reads worse than both 32 and 96 —
which is the signature of noise rather than a second knee. 32 and 96 differ
by 0.7%, inside what this rig resolves. **32 stands.**

Worth contrasting with x86, where the same sweep was decisive: there PF=32
beat PF=8 by **x1.28** and the plateau from 32 to 128 was flat and clean. On
arm the whole span from 16 to 96 sits inside 3%, and the effect is mostly
"some lookahead versus almost none". That fits P35's attribution: arm nq=100
has 18.4% of cycles in memory stalls against x86's larger share, so there is
simply less for the distance to tune.

## H69 — arm tile floor 256 -> 512: **+3.3% nq=100 MT**

The most stale parameter on the board. `MIN_TILE_BLOCKS_NEON` was set in
H15 against a kernel that has since gained SMMLA (H33), the vm8 layout
(H41), the H45 restructure and prefetch (H67) — four confirmed wins, none of
which re-checked it.

Swept via the H20 env hooks, no rebuild needed. nq=100 MT:

| CAP | 64 | 128 | **256 (shipped)** | **512** |
|---|---|---|---|---|
| ms | 15.077 | 15.083 | 14.195 | **13.719** |

Focused soak, five interleaved rounds — CAP=512 wins **all five**, and its
worst round (14.120) beats shipped's best (14.205):

| | median |
|---|---|
| shipped | 14.304 ms |
| **CAP=512** | **13.648 ms** — x1.048 |

`MULT=32` looked promising in the smoke (13.968 vs 14.275) but adds nothing
on top of CAP=512 (13.839 vs 13.648), so only the floor moves.

Paired cells, three rounds: nq=100 MT **14.293 -> 13.839 (+3.3%)**, other
three cells neutral (ST uses one range regardless; nq=1 unaffected).
127/127 green, bit-identical.

**Fewer, larger ranges now beat a finer split.** H15's reasoning was that a
finer split amortizes the ragged final wave; that argument is unchanged, but
the other side of it moved — the per-range top-k duplication costs the same
as it always did while the scan itself got ~2.5x cheaper, so the duplication
is now a larger share. A kernel that streams also wants each worker on a
longer contiguous run.

### A test pinned the old constant, and that is worth distinguishing

`gate_tests::the_tile_target_binds_when_the_caps_do_not` failed:
`assert_eq!(MIN_TILE_BLOCKS_NEON, MIN_TILE_BLOCKS / 4)`. That assertion
**documents a tuned constant**, so updating it with the constant is correct.
The property assertion in the same test — that `n_block_ranges` returns the
per-worker target when both caps clear — was untouched and still passes.

*A failing test after a tuning change is a question, not an obstacle: does
it encode a property or a number?* Bending the first would have been the
error; this was the second.

## H70 — x86 tile floor 1024 -> 3072: **+3.7% nq=100 MT**

H69's twin. H37 found 7 block ranges optimal on x86 — measured against the
**pre-H34** kernel, since when H54, H59, H62 and H65 have all changed x86's
memory behaviour. Re-swept, nq=100 MT medians:

| floor | 256 | 512 | **1024 (shipped)** | 2048 | **3072** | 4096 |
|---|---|---|---|---|---|---|
| ms | 19.80 | 19.05 | 18.755 | 18.711 | **18.004** | 18.902 |

x1.042 with no overlap — 3072's worst round (18.019) beats 1024's best
(18.707) — and **4096 turning back up marks 3072 as a knee, not a trend**.
3 ranges at N=200k where the shipped floor gave 7.

Paired cells: nq=100 MT **18.732 -> 18.047 (+3.7%)**, other three neutral
(ST uses one range; nq=1 unaffected). 133/133 green, bit-identical.

### The constant could not simply move

`MIN_TILE_BLOCKS` is load-bearing in two other places: it is the
single-query pool gate (`SINGLE_QUERY_PARALLEL_MIN_BLOCKS >=
MIN_TILE_BLOCKS`, both 1024) and the base for `MIN_TILE_BLOCKS_NEON`.
Setting it to 3072 would have **broken the invariant and silently undone
H69** — the arm floor would have gone 512 -> 1536.

So x86 gets its own `MIN_TILE_BLOCKS_X86`, exactly as aarch64 already has
`MIN_TILE_BLOCKS_NEON`. *A shared constant that three call sites tuned
independently is not shared, it is three constants that happen to be
equal* — and the coupling only surfaced because the invariant was written
down where changing it would trip.

**Both arches now want coarser tiles than they did** (arm 256 -> 512, x86
1024 -> 3072), for the same reason: per-range top-k duplication costs what
it always did while the scan got 2.5-3x cheaper, and a streaming kernel
wants each worker on a longer contiguous run. H15 and H37 were right for
their kernels.

## H71 — arm tiles-per-thread under the new floor: null

H69's smoke suggested `MULT=32` beat the shipped 64 (13.968 vs 14.275) but
it added nothing once CAP=512 was applied. With H69 shipped, the interaction
has changed, so the constant is owed its own sweep. nq=100 MT, four rounds,
medians:

| MULT | 16 | 32 | **64 (shipped)** | 128 |
|---|---|---|---|---|
| ms | 14.018 | 13.796 | **13.763** | 13.764 |

32, 64 and 128 are indistinguishable; only 16 is clearly worse. **H15's
`TILES_PER_THREAD_NEON = TILES_PER_THREAD * 2` stands.**

The earlier hint was an artifact of the old floor: at CAP=256 the block cap
bound at 24 ranges and the target term mattered; at CAP=512 the cap binds at
12 and the target is slack for any MULT above 16. *A parameter that looked
promising in a smoke taken under a different neighbouring value is not
evidence — it is the same staleness the rule warns about, pointing the other
way.*

Six wins and four nulls now from the re-test rule (H59, H65, H67, H69, H70
against H60, H61, H66, H68, H71). The nulls have all been cheap sweeps; the
wins have been 3-8% each.

## H72 — x86 tiles-per-thread under the new floor: null, and structurally so

The last untested tiling constant. H37 set `TILES_PER_THREAD = 32` and H70
has since tripled the floor it interacts with. nq=100 MT, four rounds,
medians:

| TPT | 8 | 16 | **32 (shipped)** | 64 |
|---|---|---|---|---|
| ms | 18.008 | 17.980 | **18.029** | 18.000 |

All within 0.3% — no signal at any width, and **the arithmetic says there
cannot be one**. The range count is
`min(n_threads * TPT / n_quads, n_blocks / floor, k_cap)`. At N=200k, nq=100
the floor now gives `6250 / 3072 = 3`, while the target term gives
`8 * 32 / 13 = 20` even at TPT=8 (`8 * 8 / 13 = 5`). The floor binds for
every value swept, so TPT cannot move the result.

*H70 did not just retune a constant, it changed which term is binding —
and thereby made a neighbouring parameter irrelevant rather than stale.*
Worth distinguishing: H71's null was "the value still happens to be best",
this one is "the value can no longer matter". The second is stronger and
means TPT does not need re-testing after future x86 wins unless the floor
moves back down.

That closes every tiling constant on both arches: arm floor (H69, moved),
arm target (H71, holds), x86 floor (H70, moved), x86 target (H72,
irrelevant).

## H73 — deep prefetch on the arm single-query kernel: null

H48 refuted arm prefetch at nq=1 — **at a lookahead of one q8 unit**, before
H62 and H67 established that depth is what decides prefetch on both arches.
So the refutation was scoped to a distance, not to the technique, and P35
puts 13.9% of nq=1 cycles in memory stalls. Retested at H67's 32 units:

| arm nq=1 ST | base | H73 |
|---|---|---|
| median | 3.921 ms | 3.892 ms |

0.7%, ranges overlapping. Null. (`prfm` count 2 -> 3 confirms it was live.)

**H48's conclusion survives its scope correction**, which is worth noting
because the scope correction is usually where this log finds wins. The
difference from H67: the batched kernel sweeps the array 12.5 times and
every sweep re-reads a region the previous one evicted, so a deep lookahead
has something to hide. At nq=1 there is one sweep and the hardware
prefetcher already has the stride — P24's flat 18-20 GB/s across every cache
level, which is what a saturated prefetcher looks like.

*Not every scoped refutation is hiding a win.* Four of this log's re-tests
flipped and five did not; the mechanism has to have somewhere to work, and
here it does not.

## H74 — tile ordering, re-tested: block-major holds, and by more than before

H7/H9 chose block-range-major tile ordering over query-quad-major, worth
x1.019 on arm at the time — measured against a kernel that has since gained
SMMLA (H33), vm8 (H41), prefetch (H67) and a 2x coarser tile floor (H69, 12
ranges where there were 24). Flipped it back:

| arm nq=100 MT | block-major | quad-major |
|---|---|---|
| | 13.929 / 13.983 / 13.849 / 13.869 | 15.432 / 15.239 / 15.391 / 15.406 |

**x0.90** — quad-major is 10% worse, where it was 1.9% worse when H7/H9
measured it. The right choice, by a five-fold larger margin.

**A stale parameter can get *more* right, not just less.** Every previous
re-test in this log either flipped a value or found it unchanged; this is the
first where the original decision strengthened. The mechanism is
straightforward once stated: quad-major puts the tiles in flight at any
moment in *different* block ranges, so workers stream disjoint slices of the
code array. H69 made each range 2x larger, so those slices are now 2x
further apart and share even less cache — the same choice, amplified by a
later win.

That is worth carrying into the rule: **a win that changes a resource can
strengthen a neighbouring decision as easily as it can invalidate one**, and
the re-test tells you which without guessing. Five flips, six holds now.

## H75 — the k-cap cannot bind at the benchmark point (arithmetic, not measured)

`range_cap_for_k` is the last unexamined term in the range calculation. At
the benchmark shape it does not bind, and no sweep is needed to know:

    range_cap_for_k(200_000, 10) = 200_000 / (512 * 10) = 40

    arm: min(target 8*64/13 = 40, floor 6250/512 = 13, k-cap 40) = **13**
    x86: min(target 8*32/13 = 20, floor 6250/3072 = 3, k-cap 40) = **3**

The **floor** binds on both arches after H69 and H70, with the k-cap 3x
slack on arm and 13x on x86. Changing it cannot move any of the eight cells.

Recorded as decided-by-arithmetic rather than refuted, the same status as
H55. Two of this log's entries are now closed this way, and both were worth
writing down: the alternative is a measurement that returns "no change" for
a reason the numbers already gave.

**That exhausts the parallel-scheduling surface.** Every term in
`n_block_ranges` — target, floor, k-cap — plus tile ordering, batch width,
stream depth and prefetch distance has now been swept or shown structurally
inert, on both arches at both query widths. The wins came from the two
floors (H69, H70); everything else either held or could not matter.

## Paired re-baseline after H67/H69/H70 — **HM x2.0026**

Fresh three-round paired run on both boxes, main and HEAD alternating:

| cell | main | now | speedup |
|---|---|---|---|
| arm nq100 MT | 41.771 ms | 13.826 ms | **x3.021** |
| arm nq100 ST | 315.898 ms | 115.922 ms | **x2.725** |
| arm nq1 MT | 0.598 ms | 0.582 ms | x1.028 |
| arm nq1 ST | 4.102 ms | 3.786 ms | x1.084 |
| x86 nq100 MT | 61.907 ms | 18.040 ms | **x3.432** |
| x86 nq100 ST | 239.662 ms | 74.274 ms | **x3.227** |
| x86 nq1 MT | 2.439 ms | 1.050 ms | x2.322 |
| x86 nq1 ST | 9.411 ms | 3.473 ms | x2.710 |

**Harmonic mean x2.0026.** Six of eight cells are above x2.3; arm crossed
x3 on nq=100 MT for the first time and x86 reached x3.43.

**And the harmonic mean barely moved** — x1.9936 to x2.0026, +0.45% — while
three confirmed wins landed between the two runs (H67 +8.3% arm nq=100 ST,
H69 +3.3% arm nq=100 MT, H70 +3.7% x86 nq=100 MT). The reason is the
metric doing exactly what it was designed to do: a harmonic mean is
dominated by its worst terms, and **the two arm nq=1 cells (x1.028, x1.084)
contribute 1.95 of the 3.99 total reciprocal** — nearly half the denominator
from a quarter of the cells. Improving cells already at x3 is close to free
in this scoring.

That is the correct behaviour and worth stating plainly: *the last three
wins were real, measured, and reproducible, and they were worth 0.45% on the
goal's figure.* Any further gain on the six strong cells is similarly
capped. The score now moves only if arm nq=1 moves — and P35/P36 established
that cell is at 88% issue utilization with every mechanism refuted.

## H76 — a separate floor for arm nq=1 MT: null

The re-baseline showed the score now moves only via arm nq=1, and H69 tuned
the arm floor against **nq=100**. At nq=1 the range arithmetic differs: one
query quad means the block axis supplies all the parallelism, and 12 ranges
on 8 workers is a ragged 1.5 waves — exactly the shape `smooth_tile_count`
exists to avoid. So the two widths might want different floors.

arm nq=1 MT, three rounds, medians:

| CAP | 128 | 256 | **512 (shipped)** | 768 | 1024 |
|---|---|---|---|---|---|
| ms | 0.576 | 0.551 | **0.554** | 0.607 | 0.564 |

No signal — the whole span sits inside this cell's noise, and the ordering
is not even monotonic. **One floor serves both widths.** No split needed.

Worth recording why this cell resists tuning at all: at 0.55 ms across 8
threads it is ~0.07 ms of work per worker, where thread wake-up and the
per-range top-k merge are a material fraction of the total. H51 found the
same, the cell harness takes best-of-three sub-runs because of it, and the
last re-baseline was nearly corrupted by a single perturbed round on it.
*The cell that most needs improving is the one this rig can least resolve.*

## H77 — the tile floor is inert at nq=1 on both arches (structural)

H70 tuned x86's floor against nq=100, where 13 query quads supply the
parallelism. At nq=1 there is one quad, so `min(256, 6250/3072, 40) = 3`
ranges for 8 threads looked like a parallelism regression H70 might have
introduced. Swept x86 nq=1 MT across floors 512 / 1024 / 2048 / 3072:
medians **1.050 / 1.064 / 1.052 / 1.060 ms** — within 1.3%, no signal.

The reason is that **nq=1 never enters the batched tile path**. Both
single-query dispatches use their own splitter:

    block_range_stride(n_blocks, n_threads)
        = (6250 / 8).max(64).next_multiple_of(2) = 782

— one range per worker, exactly balanced, with no reference to
`MIN_TILE_BLOCKS`, `TILES_PER_THREAD` or the k-cap. So H69 and H70 could
not have affected nq=1, and H76's null on arm has the same cause.

Third entry closed by structure rather than measurement (H72, H75, now
H77), and the most useful kind: it says **the tile constants and the nq=1
cells are disjoint**, so neither needs re-testing when the other moves. That
retires a whole quadrant of the re-test rule's search space.

It also corrects a claim I made twice while chasing this: the x86 nq=1 MT
scaling figure (3.31x against main's 3.86x) is not a range-count effect and
cannot be recovered by tiling.

## H78 — nq=1 prefetch distance, ST and MT disagree: null

H62 and H66 swept the nq=1 lookahead in **ST only**. At nq=1 MT eight
workers share L2/L3, which is a different memory regime. Swept, x86 nq=1
MT, three rounds, medians:

| PF | 2 | 4 | **8 (shipped)** | 16 | 32 |
|---|---|---|---|---|---|
| ms | **1.045** | 1.072 | 1.093 | 1.162 | 1.228 |

**Perfectly monotonic — shorter is better, the opposite of nq=100.** Eight
workers make a deep lookahead pollution rather than prefetch. Clean signal:
the ordering holds in all three rounds.

PF=4 looked like the value that would beat the shipped 8 at both thread
counts (H66 measured ST at 3.532 for PF=4 against 3.590 for PF=8). Shipped
it and measured the cells:

| x86 cell | PF=8 | PF=4 | |
|---|---|---|---|
| nq=1 MT | 1.069 ms | **1.053 ms** | +1.5% |
| nq=1 ST | 3.535 ms | 3.608 ms | **-2.1%** |

**The ST half did not reproduce.** H66 measured PF=4 *better* than PF=8 in
ST (3.532 vs 3.590); this run measures it *worse* (3.608 vs 3.535). Two
sessions, opposite orderings, both within the ~2% drift this rig shows on
that cell — so the ST preference was never real, and H66's null ("4 and 8
indistinguishable") was the correct reading of its own data. Reverted; 8
stands.

*A monotonic sweep in one mode does not license a value change that a
non-monotonic sweep in another mode appeared to support.* The MT trend here
is solid and the ST one never was; combining them produced a change that
helped the cell with the signal and hurt the cell without it.

The real content: **nq=1 ST and nq=1 MT want different prefetch depths**
(4-8 versus 2), and no single constant serves both. Conditioning on thread
count is possible — `rayon::current_num_threads()` is available in the
dispatch — but the spread is 1.5-2% on two cells that contribute little to
the harmonic mean, against the cost of a runtime branch in the hot path.

## H79 — fusing the arm nq=1 epilogue: below the threshold by arithmetic

The last untouched thing in the single-query kernel: it writes 32 floats to
a `raw` buffer in the half loop, then reads them straight back to apply
`vec_scales`. Fusing the scale into the accumulator conversion would remove
8 stores and 8 loads per block.

**16 operations against ~5376 in the block's scan loop — 0.3%.** The
harmonic-mean arithmetic needs **~4% on an arm nq=1 cell** to move the score
1% (those two cells hold 1.895 of the total 3.994 reciprocal). A 0.3% kernel
saving is two orders of magnitude short and inside this cell's measurement
noise besides.

Closed by arithmetic, like H55 and H75. Not built.

### The arm nq=1 surface is now genuinely exhausted

Every mechanism, with its verdict and how it was reached:

| lever | verdict | how |
|---|---|---|
| instruction count | at the ISA floor | P25, matched against llama.cpp/KleidiAI |
| codebook (would delete 2 TBLs) | closed permanently | H50, Ryan's recall call |
| register width | structural | P30, 128-bit vs x86's 512 |
| chains | 16 is right | H44, H45, H57 |
| memory streams | refuted x0.57 | H64 |
| prefetch | refuted at 1 and 32 units | H48, H73 |
| load count / scheduling | null | H46, H47, H58 |
| ports (TBL -> TBX) | null | H63 |
| tile constants | structurally disjoint | H77 |
| epilogue fusion | 0.3%, below threshold | H79 |
| issue utilization | **88% of 4-wide** | P36, three independent routes |

Nothing here is closed by exhaustion of ideas; each has a measurement or a
piece of arithmetic behind it.

## P37 — re-modelling x86 after four wins, and what llvm-mca cannot see

P32 put x86 at 113% of its static model — measured before H59, H62, H65 and
H70. Re-modelled the shipped loop (GFNI high nibble, prefetch, NQ=8) with
llvm-18:

| | cycles per byte-group quad |
|---|---|
| llvm-18 `-mcpu=sapphirerapids` | 34.05 |
| measured (74.274 ms nq=100 ST) | **29.7** |

**115% of model**, essentially unchanged from P32. No compute headroom by
this measure, and the re-test rule is satisfied.

**The informative part is that the model did not move at all.** Adding the
`prefetcht0` — worth **+25%** in reality (H62) — leaves llvm-mca's cycle
count identical, because the tool assumes every load hits L1. Its number is
a *compute* ceiling and is structurally blind to the entire class of win
that H43/H59/H62/H67 turned out to be.

That reframes what "measuring above the model" means. It is not evidence of
beating the machine; it means the model over-charges something on the
compute side (each `vpdpbusd` with a `{1to16}` memory operand is cracked
into 2 µops) while under-charging memory by assuming it away. **A static
analyzer bounds the wrong half of this kernel** — which is why P32's "the
kernel-tuning phase is over" was premature: four wins followed it, all of
them memory-side.

Recorded so the next reader does not treat llvm-mca as a completeness check.
It answers "are we issue-bound?" and nothing else.

## P38 — x86 vPMU unavailable, and a competitive datapoint that changes the target

**x86 counters cannot be had on this rig.** The ARM trick (P34/P35 — create a
second box with `--performance-monitoring-unit`) does not transfer:

    c3-standard-8:  PerformanceMonitoringUnit is not supported (v1 and beta)
    n2-standard-8:  not supported
    c4-standard-8:  needs hyperdisk, not pd-balanced

So x86 stall attribution stays unavailable, and P37 established llvm-mca is
blind to exactly the memory effects that produced four of this branch's
wins. **x86 has no instrument for its remaining question.**

### The competitive picture, from Ryan

turbovec is **x0.70 and x0.42 against FAISS at nq=1 ST** — i.e. 1.4x to 2.4x
slower in the one cell that gates the goal's harmonic mean. Two independent
framings now point at the same cell.

**This cannot be a kernel deficit, and the log already has the evidence:**

- turbovec's nq=1 inner loop is **4.57 (vector,dim) pairs per instruction**
  against FAISS's LUT16 at ~2.7 (P26, counted from source).
- The arm loop runs at **88% of issue capacity** with 3% frontend idle
  (P36), confirmed by three independent routes.
- Every kernel mechanism is measured and closed (H79's table).

A kernel that is denser per instruction and near issue saturation cannot be
2.4x slower than one that is neither — **unless it is doing more work**.
That points squarely at the algorithm: FAISS at nq=1 is almost certainly not
scanning all N. Its fast-scan path is built for IVF, where a coarse
quantizer restricts the scan to a few percent of the database, and P26 found
FAISS's recent work is all in that direction (`Panorama.h` level-wise
pruning, `PdxLayout.h`, `AdSampling.cpp`).

turbovec scans every vector at nq=1. **The gap is structural, not
instructional**, and closing it means an index structure, not a faster loop
— which is the route this log has been pointing at since P32 and which no
amount of kernel work can reach.

## H80 — is the FAISS nq=1 gap flat-vs-flat? (the test P38's conclusion needs)

P38 concluded the x0.42 gap must be structural, because turbovec's kernel is
denser per instruction (4.57 vs ~2.7 pairs/instr) and at 88% issue
utilization, so it cannot be 2.4x slower unless it is doing more work.
Confirmed turbovec has **no non-exhaustive path** — no IVF, no coarse
quantizer, no `nprobe`; it scans every vector at every query width.

**But that conclusion has an untested premise**: it assumes FAISS is scanning
less. If the comparison is against FAISS *flat* — same N, same exhaustive
scan — then the density argument is wrong somewhere and there is a kernel
deficit this log has failed to find.

The two readings are distinguishable by one measurement, and it is worth
more than another parameter sweep:

- **Flat vs flat at nq=1 ST, same N and dim, on the bench box.** If FAISS
  flat is ~2x faster, P38 is wrong and the density analysis (P26's counts,
  P36's 88%) has an error in it — most likely that FAISS's nq=1 path is not
  the LUT16 kernel P26 counted, but `expanded_scanners.h`, whose entire
  stated purpose is devirtualizing the per-code path "because the speed
  difference matters for very small distance computations".
- If FAISS flat is comparable or slower, P38 stands and the gap is IVF.

*A conclusion built on an unmeasured premise about someone else's code is
exactly the shape of error this log has hit five times* (P23's flag, P31's
bound, P33's magnitude, P24/P29's roofline, H48's scope). The premise here
is "FAISS at nq=1 does not scan all N" and it has not been checked.

### Run on the vPMU box — **P38 is refuted, and by neither candidate**

The third box (P35) exists so measurements cannot touch the baselines, so
FAISS went there. nq=1 ST, N=200k, dim=768, single-threaded both sides:

| | nq=1 ST | bytes/vector |
|---|---|---|
| **turbovec 4-bit flat** | **3.459 ms** | 384 |
| faiss SQ4 flat | 74.308 ms | 384 |
| faiss SQ8 flat | 54.572 ms | 768 |
| **faiss PQ4 fastscan** | **2.414 ms** | **192** |

`2.414 / 3.459 = 0.698` — **exactly the x0.70 in Ryan's table.** So the
comparison *is* flat-vs-flat: both scan all N, no IVF, no pruning. P38's
premise was wrong.

**But it is not a kernel deficit either.** `PQ384x4fs` packs **two
dimensions per 4-bit code**; turbovec's scalar quantizer packs one. FAISS
reads 192 bytes per vector where turbovec reads 384. Per byte scanned:

| | ms per byte-per-vector |
|---|---|
| turbovec | **0.0090** |
| faiss PQ4fs | 0.0126 |

**turbovec's scan is 1.4x more efficient per byte.** FAISS wins the wall
clock by storing half the data, not by scanning it faster — which is
consistent with every kernel measurement in this log rather than contradicting
them.

Two other results worth keeping. **FAISS's own SQ4 — the like-for-like
quantizer — is 21x slower than turbovec** (74.3 ms vs 3.459), confirming
P26's finding that `Codec4bit` has no SIMD specialisation at all. And SQ8,
at twice the bytes, is *faster* than SQ4 (54.6 vs 74.3), which is only
possible because the 4-bit path is scalar.

**The real question this exposes is a product one, not a kernel one**: the
gap is a compression-ratio difference between scalar 4-bit and PQ-with-2-
dims-per-code, at an unmeasured recall difference. Closing it means a
different quantizer, which is the same family H50 closed on recall grounds —
but PQ is *not* what H50 refuted (that was uniform/affine scalar), and its
recall at this compression has never been measured here.

## H81 — what FAISS's 1.43x at nq=1 actually costs: **32.6 recall points**

H80 left the compression route open: FAISS wins nq=1 by storing 192 bytes
per vector against turbovec's 384, and this log had never measured what PQ's
two-dims-per-code buys. Measured, N=50k, dim=768, k=10, recall against exact
inner product:

| | bytes/vec | nq=1 ST | recall@10 |
|---|---|---|---|
| **turbovec SQ4** | 384 | 3.459 ms | **0.8385** |
| faiss PQ384x4fs | 192 | 2.414 ms | **0.5125** |
| faiss PQ192x4fs | 96 | — | 0.2240 |

**The x0.70 is not a deficit. It is a different point on the
speed/recall frontier, and turbovec's is the better one.** FAISS buys 1.43x
by giving up 32.6 recall points — against a project whose standing
constraint refused 0.85 points (H50).

Halving again (4 dims/code, 96 B) costs 61 points. The curve is brutal at
this compression because a 4-bit code must cover a 2- or 4-dimensional cell:
16 centroids in 2D is 4 per axis, against 16 per axis for scalar.

**Caveat, stated because it cuts against the conclusion**: this is
i.i.d. Gaussian data, which is the worst case for PQ — it has no
inter-dimensional structure for the sub-quantizers to exploit, while
turbovec's block-Hadamard rotation is designed for exactly this regime. On
real embeddings with correlated dimensions PQ closes some of the gap. **32.6
points is an upper bound on the penalty, not an estimate of it.** Confirming
the ordering on real vectors is the follow-up, and it is cheap.

Two conclusions for the competitive picture:

1. **The nq=1 comparison should be reported per recall level, not per index
   name.** turbovec at 0.8385 against PQ4fs at 0.5125 is not a like-for-like
   latency comparison, and a x0.70 headline invites reading it as one.
2. **Against FAISS's like-for-like quantizer, turbovec is 21x faster**
   (H80: SQ4 74.3 ms vs 3.459 ms). That is the comparison at matched recall,
   and it is the one that flatters turbovec rather than the one that does
   not.

## H82 — H81's caveat was right, and the equal-bytes answer inverts the story

H81 measured PQ at **-32.6 recall points** on Gaussian data and flagged that
as an upper bound, since i.i.d. data is PQ's worst case. Tested on real
OpenAI text-embedding-3 vectors (1536-d, N=50k, nq=200, recall@10 vs exact):

| | bytes/vec | recall@10 |
|---|---|---|
| turbovec SQ4 | 768 | **0.9685** |
| faiss PQ768x4fs (2 dims/code) | 384 | 0.8995 |
| faiss PQ384x4fs (4 dims/code) | 192 | 0.7985 |

**The penalty is 6.9 points on real data, not 32.6** — H81 overstated it by
4.7x, exactly as its caveat warned. Correlated dimensions are what PQ's
sub-quantizers exist to exploit, and Gaussian data has none.

### The comparison was never like-for-like, in either direction

Both H81 and the original x0.70 compare turbovec at **768 B/vec** against PQ
at **384 B/vec**. That is a footprint difference, not a latency one. turbovec
supports 2-bit, which at dim=1536 is exactly 384 B — the same footprint:

| 384 B/vec | recall@10 |
|---|---|
| **turbovec SQ2** | **0.8940** |
| **faiss PQ768x4fs** | **0.8995** |

**Statistically tied** — 0.55 points across 200 queries. At equal memory
turbovec matches PQ's recall, and H80 measured its scan at **1.4x more
efficient per byte**.

So the competitive position at nq=1, stated properly:

- **At equal recall** (0.90): turbovec SQ2 and PQ4fs are the same accuracy,
  and turbovec should win on time by its per-byte advantage. *Unmeasured* —
  SQ2's nq=1 latency was not taken, and that is the one number needed to
  close this. It is the obvious next measurement.
- **At equal bytes**: tied on recall.
- **At equal quantizer** (SQ4 vs SQ4): turbovec is **21x faster** (H80).
- **The x0.70 headline**: turbovec at 0.9685 recall against PQ at 0.8995 —
  more accurate and slower, which is a choice rather than a deficit.

*Three measurements were needed to see this, and the first two each looked
conclusive on their own.* H80 said "structural, FAISS stores less"; H81 said
"and it costs 32.6 points"; H82 says the penalty is a quarter of that and
the whole comparison was between different operating points. **A competitive
number is not interpretable until the operating points are matched** — and
matching them is a measurement, not an argument.

## H83 — the missing number: **turbovec is 1.35x faster at matched recall**

H82 left one inference unmeasured — that SQ2 should beat PQ4fs on time,
since they tie on recall at 384 B/vec and turbovec's scan is 1.4x more
efficient per byte. Taken rather than argued. OpenAI-1536, N=200k, nq=1 ST:

| config | B/vec | nq=1 ST | recall@10 |
|---|---|---|---|
| turbovec SQ4 | 768 | 7.178 ms | **0.9685** |
| **turbovec SQ2** | **384** | **3.833 ms** | **0.8940** |
| **faiss PQ768x4fs** | **384** | 5.160 ms | **0.8995** |
| faiss PQ384x4fs | 192 | 2.454 ms | 0.7985 |

**At equal bytes and statistically equal recall (0.8940 vs 0.8995),
turbovec is x1.35 faster.** The inference was right, and it is now a
measurement.

### The competitive picture at nq=1, complete

| comparison | result |
|---|---|
| equal quantizer (SQ4 vs SQ4) | turbovec **21x faster** (H80) |
| equal bytes + equal recall | turbovec **1.35x faster** |
| equal bytes, recall | **tied** (0.8940 vs 0.8995) |
| the x0.70 (H80, Gaussian dim=768) | turbovec 384 B/vec vs PQ384x4fs 192 B/vec |

**turbovec is not behind FAISS at nq=1 on any matched comparison.** The
x0.70 arises entirely from comparing a higher-accuracy configuration
against a lower-accuracy one, and disappears the moment either axis is
held fixed.

### Correction: two runs were quoted as one

The table above originally paired the **x0.70** with the recall figures
**0.9685 / 0.8995**. Those come from different measurements:

| | dataset | dim | turbovec | faiss |
|---|---|---|---|---|
| x0.70 (H80) | i.i.d. Gaussian | 768 | SQ4, 384 B/vec | `PQ384x4fs`, 192 B/vec |
| 0.9685 / 0.8995 (H82) | OpenAI-1536 | 1536 | SQ4, 768 B/vec | `PQ768x4fs`, 384 B/vec |

Different data, dimension and index config. The **pattern** holds in both —
turbovec carrying 2x the bytes at higher recall — but presenting them on one
row implies a single experiment and overstates what was measured together.

**And the identification itself is unconfirmed.** Ryan's x0.70 was matched
to H80's 0.698 by numeric coincidence; which `m` his `PQ{m}x4fs` uses is not
known, and `m` sets dims-per-code and therefore the byte ratio the whole
analysis turns on. At `PQ384x4fs` (2 dims/code) turbovec carries 2x the
bytes and the conclusions above apply. At a larger `m` the footprints are
closer and they do not.

*Matching a number is not identifying a configuration* — the same error as
reading `stalled-cycles-backend` as memory because the name fitted the
hypothesis (P35).

### Resolved by construction: Ryan's harness is `IndexPQFastScan(768, 384, 4)`

384 sub-quantizers over 768 dims = **2 dimensions per code**, 4 bits each =
**192 bytes/vector**, against turbovec 4-bit's **384**. That is exactly
H80's `PQ384x4fs`, so H80-H83 apply to that harness directly — now
identified from its construction rather than from a matching number.

**The harness calls this "the precision-matched configuration", and that is
the load-bearing error.** Both sides are 4 bits per *code*, but a PQ code
spans two dimensions where a scalar code spans one. It is bit-width-matched
at **half the memory** — not precision-matched by any measure that survives
being written down:

| | bits/code | dims/code | **bytes/vector** | recall@10 (OpenAI-1536) |
|---|---|---|---|---|
| turbovec SQ4 | 4 | 1 | 768 | 0.9685 |
| faiss PQ{d/2}x4fs | 4 | 2 | 384 | 0.8995 |
| **turbovec SQ2** | **2** | **1** | **384** | **0.8940** |

The genuinely matched row is the third: same bytes, tied recall, and
turbovec **1.35x faster** (H83). "4-bit vs 4-bit" reads as like-for-like and
is a 2x compression difference.

The harness's own stated limitations are sound — random data is fine for
speed and correctly excluded from recall claims, and H81 independently shows
why (i.i.d. data is PQ's worst case, so a recall number taken there would
flatter turbovec by ~4.7x). The gap is not in the methodology, it is in one
word in the configuration's description.

### What this cost to establish

Four measurements (H80, H81, H82, H83), and the first three each produced a
confident conclusion that the next one overturned or halved:

1. H80: "the gap is FAISS storing half the bytes" — true but not the point.
2. H81: "which costs 32.6 recall points" — 4.7x overstated, Gaussian data.
3. H82: "6.9 points on real data" — still comparing unmatched footprints.
4. H83: matched on both axes, turbovec wins on time at equal recall.

*Every step was a correctly-run measurement whose interpretation outran it.*
The discipline that eventually worked was mechanical: hold one axis fixed,
measure the other, and refuse to compare configurations that differ in two
things at once.

## Branch verification at `8e7e18d8` (129 commits ahead of main)

Full suites, both arches, working tree clean:

| | lib | integration |
|---|---|---|
| arm (c4a) | **127/127** | 1 failure |
| x86 (c3) | **133/133** | 1 failure |

The single failure is identical on both arches and is **pre-existing**:

    allocation_hot_paths::repack_allocation_count_does_not_scale_with_vector_count
    prepare allocations: 64 vectors = 0, 4096 vectors = 11

Verified at session start against `main`, on both arches, and with the
vector-major layout disabled — same 0-vs-11 signature every time. It is a
repack-path allocation regression that predates this branch and is
untouched by it. **Deliberately not fixed here**: bundling an unrelated fix
with a perf branch is exactly what the standing guidance says not to do, and
it would muddy the bisect if the perf work ever needs one.

Every kernel change on this branch is bit-identical to `main`'s output —
`score md5 5939c346f21ab325832ca46307495e46` and recall 0.8030, checked
after each of the seventeen confirmed improvements and re-checked through a
fresh write path whenever the layout or format touched (H41, H64).

## P39 — counters confirm H67's mechanism, and price what is left

P35 measured arm nq=100's stall profile **before** H67 added prefetch. Same
events, same shape, on the vPMU box after it:

| arm nq=100 ST | pre-H67 (P35) | post-H67 | |
|---|---|---|---|
| cycles | 4.637e9 | **4.466e9** | -3.7% |
| `stall_backend` | 43.2% | **37.2%** | -6.0 pts |
| `stall_backend_mem` | **18.4%** | **14.2%** | **-4.2 pts** |

**The mechanism is confirmed by attribution, not just by wall clock.** H67
was accepted on an A/B (+8.3% ST) with a *hypothesised* cause — that P29's
compute-bound reading was wrong and there were memory stalls to hide. The
counters now show memory-attributed stalls falling 4.2 points and total
cycles falling with them. That is the first mechanism claim in this log
verified by a direct measurement of the resource rather than by the outcome.

**And it prices the remainder**: 14.2% of cycles are still memory-attributed
at nq=100. H62/H67 tuned the depth and H68 showed the distance is at its
knee, so that residual is not reachable by more prefetch — it is the part
the hardware prefetcher and a 32-unit lookahead together cannot cover.

Worth contrasting with what the same counters said about nq=1 (P35: 13.9%
memory, 21.9% execution). The two widths now differ where they did not
before: prefetch moved nq=100's memory share down while nq=1's is untouched,
because H73 refuted prefetch there. The 35%-at-both-widths coincidence that
made P35's first reading ambiguous has been broken by a change that only
affected one of them.

*(The nq=1 leg of this run aborted early — 24M cycles against an expected
11e9 — so only the nq=100 figures are quoted. Same intermittent failure as
P35's first attempt; the harness builds a 200k index in-process and
occasionally dies before timing.)*

## H84 — arm batch width 8 -> 12: **+15.9% nq=100 ST, HM x2.003 -> x2.060**

**H40 refuted 12 and 16 — on the 4-group SMMLA kernel, before vm8.** That
kernel needed `NQ/2 * 4 = 24` accumulators at NQ=12 and spilled. vm8's
eighth-blocks (H41) need `NP * 2 = 12`. The register arithmetic that
justified the refutation stopped applying two hypotheses after it was
written, and nothing re-checked it for forty-three entries.

The motivation is memory, not registers: 100 queries in batches of 12 is
**9 sweeps** over the code array against 12.5 — **28% less traffic** —
against the 14.2% of cycles P39 still attributes to memory stalls.

arm nq=100 ST, three rounds, medians:

| QBS | **8 (shipped)** | **12** | 16 |
|---|---|---|---|
| ms | 114.84 | **98.92** | 101.26 |

**x1.161.** 16 is worse than 12 — the knee is real, not a trend. Parity
identical at all three widths.

Full cells, three interleaved rounds:

| arm cell | QBS=8 | QBS=12 | |
|---|---|---|---|
| nq=100 ST | 114.619 ms | **98.870 ms** | **+15.9%** |
| nq=100 MT | 13.841 ms | **12.563 ms** | **+10.2%** |
| nq=1 ST | 3.723 ms | 3.683 ms | +1.1% |
| nq=1 MT | 0.573 ms | 0.581 ms | -1.4% |

The nq=1 deltas are noise — that width dispatches to
`score_block_vm8_single`, which this does not touch (H77). 127/127 green,
bit-identical (`5939c346...`), recall 0.8030.

| cell | speedup |
|---|---|
| **arm nq100 MT** | **x3.325** |
| **arm nq100 ST** | **x3.195** |
| arm nq1 MT | x1.029 |
| arm nq1 ST | x1.114 |
| x86 nq100 MT | x3.432 |
| x86 nq100 ST | x3.227 |
| x86 nq1 MT | x2.322 |
| x86 nq1 ST | x2.710 |

**Harmonic mean x2.0596**, from x2.0026.

*The most productive rule in this log found a 16% win forty-three entries
after the refutation it overturned.* H40 was correctly measured and
correctly reasoned; H41 invalidated its premise two entries later and
neither entry noticed. The rule says re-test after a win moves the resource
— **H41 moved the register budget, which is exactly the resource H40's
refutation rested on**, and the connection was still missed. Re-testing
needs to be triggered by *which resource a refutation named*, not by
recency.

## H85 — arm prefetch distance after H84 cut traffic 28%: null, 32 holds

The sharpened rule's first application. H68 tuned the arm lookahead against
**12.5 sweeps** of the code array; H84 cut that to **9**, a 28% traffic
reduction — the exact resource the distance was fitted to. Re-swept, arm
nq=100 ST, three rounds, medians:

| PF units | 8 | 16 | **32 (shipped)** | 64 |
|---|---|---|---|---|
| ms | 104.55 | 100.48 | **98.48** | 101.23 |

32 is still the knee, with 8 and 64 both clearly worse. **H68 holds.**

Why it survived where H40 did not, which is the useful part: prefetch depth
is a property of the *stream* — how far ahead the scan reads within one
pass — and H84 changed how many passes there are, not what a pass looks
like. H40's refutation rested on a register count that H41 directly changed.
*The rule fires on the resource a refutation named; it only pays when that
resource actually moved, and "fewer sweeps" is not the same resource as
"prefetch depth within a sweep".*

Seven wins and seven nulls from the rule now (H59, H65, H67, H69, H70, H84
against H60, H61, H66, H68, H71, H73, H85). The nulls remain one sweep each.

## H86 — x86 batch width after the memory wins: null, and the asymmetry explains H84

H84 took arm from NQ=8 to 12 for **+15.9%**, on a memory argument — fewer
sweeps over the code array. x86's width was last re-tested in H60, before
H62 (prefetch), H65 (streams) and H70 (floor) all changed its memory
behaviour, so the same argument was owed a test there.

x86 nq=100 ST, three rounds, medians:

| NQ | **8 (shipped)** | 12 | 16 |
|---|---|---|---|
| ms | **74.48** | 105.55 | 106.42 |

**NQ=12 is 42% worse.** H35 and H60 hold, emphatically.

### Why the same change wins on arm and loses on x86

The accumulator arithmetic differs by a factor of two, and it is the whole
story:

| | accumulators at NQ | at NQ=8 | at NQ=12 |
|---|---|---|---|
| **arm** (vm8 eighth-blocks) | `NP * 2 = NQ` | 8 | **12** |
| **x86** (two 16-lane halves) | `NQ * 2` | 16 | **24** |

arm at NQ=12 uses 12 of 32 vector registers; x86 uses 24 of 32 zmm before
the level table, mask and the 12+ broadcast registers LLVM materializes.
arm had room for the memory win; x86 does not, and the spill costs several
times what the halved traffic saves.

*The same hypothesis, the same motivation, opposite verdicts — decided by a
factor of two in how each kernel lays out its accumulators.* This is the
sixth cross-arch transplant in this log to turn on a resource that differs
between the two (H36, H49, H53, H55, H64, now H86), and the rule that
predicts it every time is: **check the resource arithmetic in the target
before assuming the mechanism transfers.**

## H87 — NQ=10, the width H84 skipped: null, 12 is a true minimum

H84 sampled 8, 12, 16 and took 12. The register argument that explains the
arm/x86 split (H86) predicts pressure grows with NQ, so the optimum could
have sat between 8 and 12 with 12 already paying a spill cost. Sampled it:

| NQ | 8 | **10** | **12 (shipped)** | 16 |
|---|---|---|---|---|
| arm nq=100 ST, ms | 114.84 | **103.38** | **98.67** | 101.26 |

**12 is a genuine minimum**, not the best of three arbitrary samples. The
curve falls steeply from 8, bottoms at 12 and turns up by 16 — consistent
with memory traffic improving monotonically (12.5 -> 10 -> 9 -> 7 sweeps)
while register pressure begins to bite past 12.

That also bounds what a spill-relief change could win. The H84 build carries
~125 more whole-binary vector spills than the NQ=8 one, and freeing
registers by loading the 12 A-operands in halves was the obvious follow-up
— but if pressure were already costing at 12, NQ=10 (10 accumulators, 10 A
operands, 20 of 32) would have beaten it. It does not, by 4.8%. **The
pressure at 12 is not yet the binding cost**, so relieving it has nothing to
recover; not built.

Eight wins, nine nulls from the re-test rule.

## H88 — arm tile floor under NQ=12's tile count: null, 512 holds

H84 changed `n_quads` from 13 to 9 (100 queries in batches of 12), cutting
the tile count from 169 to **117** for 8 workers. H69 tuned the floor
against 169, so the resource it named moved. Re-swept, arm nq=100 MT:

**First sweep, 3 rounds** — CAP=1024 ahead by 0.6% (12.548 vs 12.618).
**Second sweep, 4 rounds:**

| CAP | **512 (shipped)** | 1024 | 2048 | 3072 |
|---|---|---|---|---|
| median | **12.448** | 12.618 | 12.835 | 13.605 |

**512 holds**, now ahead by 1.4%, and the curve is monotonically worse
toward coarser — the opposite ordering from the first sweep's marginal
reading.

*A 0.6% lead across three rounds reversed into a 1.4% deficit across four.*
This is the two-gate rule doing exactly its job: the smoke produced a
plausible, correctly-computed, wrong ordering, and the only thing that
distinguished it from H84's real 16% was running more rounds. Three of this
log's nulls now come from marginal signals that inverted under more
sampling (H51, H78, H88), against one win that survived it (H65 at 2.5%).

Nine nulls, eight wins from the re-test rule. Both floors on both arches are
now confirmed under the batch widths that followed them.

## Paired re-baseline after H84 — **HM x2.1001**

Three-round paired run on both boxes, main and HEAD alternating in the same
rounds. The x2.0596 quoted when H84 landed was computed against *older* main
baselines; this log's own rule says only within-session paired numbers are
comparable, so it is superseded by this:

| cell | main | now | speedup |
|---|---|---|---|
| **arm nq100 MT** | 42.022 ms | 12.529 ms | **x3.354** |
| **arm nq100 ST** | 315.781 ms | 98.044 ms | **x3.221** |
| arm nq1 MT | 0.631 ms | 0.579 ms | x1.090 |
| arm nq1 ST | 4.148 ms | 3.741 ms | x1.109 |
| **x86 nq100 MT** | 61.940 ms | 18.052 ms | **x3.431** |
| **x86 nq100 ST** | 240.125 ms | 74.026 ms | **x3.244** |
| x86 nq1 MT | 2.449 ms | 1.028 ms | x2.381 |
| **x86 nq1 ST** | 9.368 ms | 3.390 ms | **x2.764** |

**Harmonic mean x2.1001** (arithmetic x2.5742), from x2.0026 at the previous
paired baseline — **+4.9%**, and the first reading above x2.1.

All four nq=100 cells are now above **x3.2**, and the two arches have
converged there (arm x3.354/x3.221 against x86 x3.431/x3.244) after arm
trailed by 20-30% for most of this climb. The gap that remains is entirely
in arm's nq=1 pair.

The arithmetic-vs-harmonic spread is now **x2.57 vs x2.10** — the widest it
has been. Six cells above x2.38 pull the arithmetic mean up while the two
arm nq=1 cells at x1.09 hold the harmonic mean down, contributing **1.82 of
the 3.81 total reciprocal** from a quarter of the board. That ratio is the
single fact that should govern where any further work goes.

## H89 — the one arm nq=1 claim that was never measured: SDOT vs SMMLA

Auditing H79's exhaustion table against the standard the rest of this log
holds to, one row is weaker than it looks. Every other entry has a
measurement or an arithmetic identity behind it; **"SDOT and SMMLA are
equivalent at nq=1" is an analysis**, made twice (P19-era and again around
H64) and never tested.

The argument: a lone query rides SMMLA as a duplicated pair, so lanes 2/3
repeat 0/1 and half the MACs are discarded — 16 useful of 32. SDOT does 16,
all useful. Both are 1 µop at 4/cycle on all four V pipes, so the useful-MAC
rate is identical and the instruction counts match. Hence equal.

**That reasoning is sound only if the loop is issue-bound**, which P36's 88%
supports — but P35 also attributes 22% of cycles to execution-resource
stalls, and a kernel discarding half its MAC width is exactly the shape that
could be paying there. The two measurements do not settle it between them.

The test is contained: `score_block_vm8_single` swaps its two `smmla` for
two `sdot` against a broadcast A operand, and the epilogue gains a pairwise
fold (each vector's score arrives split across two lanes instead of one).
Registers and instruction count are unchanged, so a difference either way is
attributable to the MAC itself.

### Built and measured: **the analysis was right**

| arm nq=1 ST | SMMLA | SDOT |
|---|---|---|
| | 3.822 / 3.847 / 3.696 / 3.736 | 3.772 / 3.841 / 3.687 / 3.768 |
| median | **3.779 ms** | **3.770 ms** |

0.2%, fully overlapping. Output bit-identical (`5939c346...`), 127/127
green. **SDOT and SMMLA are equivalent at nq=1, as claimed.** Reverted —
there is no reason to change a kernel for parity, and SMMLA keeps the code
shared with the batched path.

**This is the first analysis-only claim in this log to survive measurement.**
The five before it — P23's premise, P31's bound, P33's magnitude, P38's
structural claim, H40's register count — were all wrong, and H40's cost
forty-three entries before H84 recovered a 16% win from it. That record was
the entire reason for testing this one.

Worth separating the two things that record actually shows. The failures
were all cases where an *unstated premise* rode along with a stated argument
(what FAISS scans, what a counter measures, which registers a kernel needs
after a later change). Here the premise was stated and checkable: both
instructions are 1 µop at 4/cycle on all four V pipes, and the loop is
issue-bound at 88%. **An analysis whose premises are all written down and
independently measured is a different object from one that smuggles a
premise** — and the log had been treating them as the same risk.

That also closes the arm nq=1 surface for real: every row in H79's table now
has a measurement or an arithmetic identity, with no analysis-only entries
remaining.

## H90 — H84 opened a cliff at intermediate nq, and this closes it

Checking H84 for the failure mode P27 found elsewhere: `tiles` chunks
queries by `qbs`, and any chunk whose size has no `pd_scan!` arm falls to
the **per-query tail**. H84 fixed `qbs` at 12, and the arms are 12, 8 and 4
— so nq=10 arrived as one unbatched chunk of ten.

| nq | QBS=8 (pre-H84) | H84 (fixed 12) | **H90 (stepped)** |
|---|---|---|---|
| 4 | 6.60 | 6.42 | 6.50 |
| 8 | 9.16 | 9.12 | 8.74 |
| **10** | **18.06** | **39.83** | **18.62** |
| 12 | 15.82 | 11.19 | 11.39 |
| 16 | 18.17 | 18.58 | 18.41 |
| 24 | 27.45 | 23.15 | 23.37 |

**39.83 ms is 10 x 3.74** — ten independent full scans, exactly the
single-query cost times ten. H84 turned a 2.2x regression loose on nq=8..11
and neither its cell run nor the paired re-baseline could see it, because
**both only sample nq=100 and nq=1**.

The fix steps `qbs` down to a width the dispatch has an arm for: 12 at
nq>=12, 8 at nq>=8, else 4. nq=10 returns to 18.62 ms while nq=12 keeps
H84's win. Goal cells untouched by construction — nq=100 still takes 12,
nq=1 uses a different path (H77). 127/127 green, bit-identical.

**A metric with two sample points cannot see a cliff between them.** This is
the second such hole this log has found (P27's small-N parallelism cliff was
the first, also invisible to all eight cells), and both were introduced or
missed by changes that the goal's own measurements certified as wins. The
eight cells are a good objective and a poor regression suite; anything that
changes a *dispatch boundary* — batch width, tile floor, a parallelism gate
— needs sweeping across the boundary, not sampling either side of it.

## P40 — arm's tail path vs x86's padding: a structural gap H90 only half closed

H90's standing check applied to both arches. **x86 has no cliff**: nq=10
costs 1.085 ms/query against nq=8's 0.753 — padding waste from an 8 + a
2-batch-padded-to-8, not a collapse. It pads every batch to `NQ_BATCH` with
`pad_qi` and dispatches one kernel width, so no nq can fall off.

**arm matches `batch_size` against `pd_scan!` arms and drops the remainder
to a per-query tail.** After H90's step-down, remainders still land there:

| nq | 2 | 3 | 5 | 6 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|---|---|
| ms/query | **3.656** | **3.630** | 2.184 | 2.534 | 2.833 | 1.502 | 2.038 | 1.178 |

At nq=2-3 that is **3.65 ms/query against the 3.74 ms single-query cost** —
no batching whatever, on a kernel that reaches 0.95 ms/query at nq=12. A
3.8x spread across query widths a caller may pick arbitrarily.

**The fix is to adopt x86's shape**: pad the `pds` array to `qbs` with the
last query, run one kernel width, fold only the real queries into heaps.
That removes the tail path entirely and makes every nq batched, at the cost
of scoring up to `qbs-1` padding lanes that are discarded — which is exactly
the trade x86 already makes and which its sweep shows costs ~30% at nq=10
rather than 280%.

Not implemented: it restructures the arm dispatch and the heap fold, and a
half-applied version silently mis-scores (H41, H64). **Recorded with the
measurement, the design, and the reason it matters** — H90 fixed the case
the goal's cells could reach and this is the rest of it.

*Two boundary sweeps have now each found a hole the eight cells cannot see*
(P27, H90/P40). The pattern is that dispatch boundaries are where regressions
hide, and neither the objective nor the paired baseline samples them.

## P41 — padding the arm dispatch: no version dominates, reverted

Built P40's fix two ways. ms/query, against H90's shipped arm-matching:

| nq | **H90 (arms + tail)** | P41 (pad to `qbs`) | P41b (pad to smallest fitting arm) |
|---|---|---|---|
| 2 | 3.656 | 3.157 | **3.000** |
| 3 | 3.630 | 2.195 | 2.252 |
| 5 | **2.184** | 2.633 | 2.605 |
| 7 | 2.833 | 1.877 | **1.877** |
| 9 | **1.502** | 1.947 | 1.787 |
| 11 | 2.038 | 1.616 | **1.470** |
| 13 | **1.178** | 1.693 | 1.425 |
| 16 | **1.151** | 1.400 | 1.149 |

Both padded versions are correct (127/127, `5939c346...`, recall 0.8030) and
**neither dominates**. P41b improves the worst case (3.656 -> 3.000) and most
mid widths while regressing 5, 9 and 13.

The arithmetic explains it exactly. For a remainder `r`, the tail costs
`r x 3.74 ms` and padding to arm `a` costs one `a`-wide batch — about
6.2 ms at a=4, 11.5 at a=12. So padding wins for `r >= 3` and the tail wins
at `r = 1`, which is why nq=13 (12+1) regresses under every padded variant
and nq=11 (8+3) improves under all of them.

**Reverted.** A change that trades nq=5 against nq=11 has no adjudicator:
the eight cells cannot see either, and "improve the worst case" is a
different objective from the one that is actually set. Recording the table
is worth more than picking a winner arbitrarily — the optimum is a per-chunk
choice (tail at r=1, smallest fitting arm above that), which is a small
addition to the dispatch and should be made deliberately rather than as a
side effect of perf work.

*Two versions built, measured, and neither shipped.* That is the correct
outcome when the objective does not cover the thing being changed, and it is
cheaper than shipping one and discovering the trade later.

## P42 — ARM nq=1 ST is at 95% of the streaming roofline (reframes H79)

The two ARM nq=1 cells contribute **1.819 of the 3.808** reciprocal sum — 48%
of the harmonic mean's denominator. A 10% win there is +4.5% HM; the same win
on ARM nq=100 is +0.8%. So the question of what limits them is the most
valuable question left.

H79 concluded "at the ceiling" from 88% issue utilisation. Measured the
premise instead:

| | time | effective |
|---|---|---|
| kernel, nq=1 ST | 3.828 ms | **20.06 GB/s** |
| single-core sequential sum, same footprint | 3.635 ms | 21.13 GB/s |

**95% of the single-core streaming roofline.** The kernel is not compute-bound
and never was. That is a strictly better explanation of H79's table than
"ceiling": every mechanism there was refuted because none of them changed the
number of bytes read, and bytes are the binding constraint. Issue utilisation
was a symptom of a core waiting on DRAM, not evidence of a full machine.

**Consequence.** No compute change can move ARM nq=1 ST. The only lever is
reading fewer bytes, and of the three ways to do that two are already closed
by standing constraints — lower bit width trades recall, a prefilter sidecar
costs RAM. The third is a dimension-prefix shortlist followed by a full
rescore: same bytes on disk, no extra RAM, half the bandwidth on pass one.
Its viability is a pure recall question, tested next.

*This is why premises get measured.* H79's refutations were all correct and
all uninformative; the family they appeared to close was never open.

## H91 — prefix shortlisting: refuted, and it closes ARM nq=1 ST

P42 left exactly one lever: read fewer bytes without more RAM. Score a prefix
of the rotated dimensions to shortlist, full-rescore the shortlist.

Fraction of the true top-10 surviving into a shortlist of size S:

*Uniform random data (the H81 trap):*

| prefix | S=500 | S=1000 | S=2000 |
|---|---|---|---|
| 384 (50%) | 0.1750 | 0.2313 | 0.3297 |
| 576 (75%) | 0.5266 | 0.6469 | 0.7719 |

*Anisotropic data, power-law spectrum, prefix taken in the rotated basis:*

| prefix | S=500 | S=1000 | S=2000 |
|---|---|---|---|
| 384 (50%) | 0.5656 | 0.6813 | 0.8016 |
| 576 (75%) | 0.9344 | 0.9719 | **0.9938** |

Realistic data is 3-4x kinder, exactly as H82 predicted it would be — running
only the uniform table would have overstated the penalty the same way H81 did.
It still refutes. The one configuration that nearly holds recall reads 75% of
the bytes, so it can win at most ~25%, and it *still* drops 0.6 recall points.

**Closed by constraint, not by arithmetic** — "don't trade recall. Recall is
more important imo."

**ARM nq=1 ST is finished.** Not "out of ideas": P42 puts it at 95% of the
streaming roofline, the only lever that survives that is fewer bytes, and all
three ways to get fewer bytes are ruled out by standing constraints (bit width
and prefix both trade recall; a sidecar costs RAM). Further hypotheses aimed
at this cell are a waste of the loop unless a constraint changes.

## H92 — ARM nq=1 MT roofline: probe invalid, inconclusive

Same question as P42 for the MT cell. Result:

| | time | effective |
|---|---|---|
| kernel, nq=1 MT | 0.531 ms | 144.65 GB/s |
| 8-thread sequential sum, same footprint | 0.601 ms | 127.68 GB/s |

The kernel measures at **113% of its own roofline**, which means the roofline
is wrong, not that the kernel is superhuman. Two reasons, both disqualifying:
at 0.5 ms a `ThreadPoolExecutor` dispatch costs a significant fraction of the
measurement, and the 76.8 MB footprint against a ~32 MB L3 gives the kernel
partial cache residency that a cold streaming reference does not model.

**Recorded as inconclusive.** P42's ST version was valid because a single
thread has no dispatch overhead and the footprint is 2.4x L3, so the stream
reference and the kernel see the same memory system. Neither holds at MT.

Answering this needs a reference that shares the kernel's own thread pool and
access pattern — a rayon-side scan that touches every byte and computes
nothing — rather than a numpy analogue. That is the next hypothesis, and until
it exists the MT cell's limiting resource is genuinely unknown: it is *not*
safe to assume it inherits ST's bandwidth verdict.

## H93 — ARM nq=1 MT is bandwidth-bound too; both ARM nq=1 cells are finished

H92 failed because any external reference at 0.5 ms is dominated by its own
dispatch cost. Avoided references entirely: vary the footprint, fit
`time = fixed + marginal x bytes`.

| N | footprint | time | GB/s |
|---|---|---|---|
| 50k | 19.2 MB | 0.170 ms | 112.7 |
| 100k | 38.4 MB | 0.283 ms | 135.6 |
| 200k | 76.8 MB | 0.579 ms | 132.7 |
| 400k | 153.6 MB | 1.153 ms | 133.2 |

**0.014 ms fixed + 135.3 GB/s marginal.**

Two findings, both clean:

1. **Fixed overhead is 2% of the cell.** Rayon fan-out, heap setup and query
   prep are already negligible. The obvious nq=1 hypothesis — "at 0.5 ms the
   cell must be dominated by per-search overhead" — is false, and no amount of
   dispatch tuning can pay.
2. **Marginal bandwidth is flat across an 8x footprint range** and exceeds
   H92's 127.68 GB/s stream reference. That is saturation.

So MT reaches the same verdict as ST by an independent method: bandwidth-bound,
only fewer bytes can win, and fewer bytes is closed by the recall constraint
(H91). **Both ARM nq=1 cells are finished.**

### What this bounds

The two cells contribute 1.819 to the 3.808 reciprocal sum and cannot move.
Even if the other six cells were made *infinitely* fast, the harmonic mean
cannot exceed `8 / 1.819 = 4.40x`. Current is 2.1001x. That is the real
ceiling of this goal as specified, and it is set entirely by DRAM bandwidth on
one machine plus one standing constraint — not by anything in the kernels.

Remaining live zones: ARM nq=100, and all four x86 cells.

## H94 — a 16-wide ARM batch is slower (register spill)

ARM nq=100 is the live ARM zone: ~99 ms ST for 76.8 MB is **0.78 GB/s**
against the 135 GB/s roofline H93 measured, so unlike nq=1 it is compute-bound
and has real headroom. Widening the batch amortizes the nibble unpack over
more queries, so added a `pd_scan!(16, 8)` arm and `qbs = 16` for `nq >= 16`.

Correct (127/127, `5939c346...`, recall 0.8030). Two paired rounds:

| cell | base | qbs=16 |
|---|---|---|
| nq100 MT | 12.396 / 12.620 | 13.463 / 13.281 — **6% worse** |
| nq100 ST | 98.271 / 99.536 | 100.925 / 101.745 — **2.3% worse** |
| nq1 MT | 0.5686 / 0.6157 | 0.5892 / 0.5918 — unchanged |
| nq1 ST | 3.695 / 3.691 | 3.723 / 3.684 — unchanged |

**Refuted.** vm8 needs `NP*2 = NQ` accumulators, so 16 wants 16 of 32 vector
registers before the TBL operands, the LUT halves and the addressing — the
inner loop spills, and the reload traffic costs more than the extra
amortization saves. nq=1 is untouched exactly as predicted, since `qbs` only
changes at `nq >= 16`; that the prediction held is what makes the nq=100
regression trustworthy rather than drift.

12 is therefore not an arbitrary tuned value but the widest batch that fits
the register file. Widening ARM further requires *freeing a register*, not
raising a constant — which is a different hypothesis, and a harder one.

## H95 — the one cell that loses to FAISS is the 2-bit unpack, and it is fixable

Ryan's corrected 16-cell grid (matched operating points, real OpenAI-1536,
N=200k) leaves exactly one loss: **x86 nq=1 ST at 384 B/vec, x0.86**. Note it
is the *2-bit* operating point; the 8-cell goal metric runs 4-bit, where the
same cell is x1.58. So this is outside the goal but is the last losing cell in
sixteen.

x86 nq=1 ST, per-byte throughput by footprint:

| footprint | 2-bit | 4-bit |
|---|---|---|
| 9.6 / 19.2 MB (cache-resident) | **20.0 GB/s** | **27.1 GB/s** |
| 19.2 / 38.4 MB | 21.6 | 27.5 |
| 38.4 / 76.8 MB (N=200k) | 22.1 | 22.9 |
| 76.8 / 153.6 MB | 12.8 | 18.0 |

**The 2-bit path moves bytes ~26% slower than the 4-bit path when
cache-resident.** That is the whole x0.86. 2-bit packs four codes per byte
against 4-bit's two, so halving the footprint doubles per-byte unpack work;
the memory win is real but the compute cost eats it, and at nq=1 there is no
batch to amortize across. FAISS runs its 4-bit LUT kernel at *both* operating
points and never pays this, so at 2-bit we hand it a cache-resident workload
and meet it with the one path this session never optimized.

**Headroom** is the gap between the columns: 2-bit reaching 4-bit's
27.1 GB/s moves the cell ~1.35x, turning x0.86 into ~x1.16.

> **CORRECTED by H96 — this headroom estimate is wrong.** It compares
> per-*byte* throughput across the two widths as though they did equal work
> per byte. `blocked_geometry` sets `codes_per_byte = 8/bits`, so a 2-bit
> byte-group carries **four** codes against 4-bit's two. Per code — the
> quantity the kernel actually processes — 2-bit runs 88.4 Gcodes/s against
> 4-bit's 45.7 at N=200k: **1.93x faster per code**, while being only 1.26x
> slower per byte despite extracting twice as many codes from each. That is
> the 2-bit path being efficient, not deficient. There is no 26% of waste to
> reclaim and the "~x1.16" inference does not follow. The measurements in the
> table above are sound; only this paragraph's reading of them was not.

**Caveat on method.** Both linear fits produced *negative* intercepts
(-0.738, -0.767 ms), so scaling is superlinear and the fitted marginal
bandwidths are not trustworthy — the 400k points fall off a cache cliff
(2-bit 22.1 -> 12.8 GB/s). H93's ARM fit was valid because its residuals were
flat across an 8x range. Read the per-footprint columns here; they are direct
measurements. Recording this because the fit *looked* like H93's and is not.

## H96 (open) — 2-bit shares the kernel; the gap is in the unpack, not the dispatch

First check on H95's target: does 2-bit even reach the optimized kernels? It
does. `search.rs` contains no `bit_width` at all — the width enters through
`pack::blocked_geometry(n_vectors, bit_width, dim)`, so the packed layout is
width-generic and 2-bit runs the same permute-dot kernels 4-bit does.

Reading `blocked_geometry` then dissolved the premise rather than localizing
it. `codes_per_byte = 8/bits` and `n_byte_groups = dim / codes_per_byte`, so
2-bit halves the byte-groups (192 vs 384) and puts four codes in each. H95's
"26% per-byte gap" therefore compares unequal work: per code, 2-bit runs
**88.4 Gcodes/s vs 4-bit's 45.7** at N=200k — 1.93x faster, i.e. nearly the
full 2x the halved footprint allows, while paying only 1.26x per byte for
doubling the codes per byte.

**So there is no 2-bit inefficiency to fix**, and H95's headroom estimate is
retracted (corrected in place above). The x0.86 against FAISS at 384 B/vec is
not turbovec leaving 26% on the table; it is FAISS's 4-bit LUT kernel doing
well on a cache-resident workload while turbovec's advantage at that footprint
is already spent. Establishing whether *any* headroom exists there requires
comparing against FAISS's per-code rate, not against our own other bit width.

*The error worth keeping:* I derived a headroom figure from a ratio between
two configurations without checking that the denominator meant the same thing
in both. The measurements were fine; the units were not.

## H97 — x86's batch width is also a wall, and widening it breaks 2-bit silently

H94 found ARM's `qbs = 12` is the widest batch the register file allows: 16
needs `NP*2 = NQ` = 16 of 32 vector registers before TBL operands, LUT halves
and addressing, and the resulting spill cost 6% at nq=100 MT.

x86's economics are different and untested. It accumulates `NQ*2` against 32
zmm registers, so its spill wall sits at a different width than ARM's, and
nothing in this log establishes where. The width in the code was inherited,
not swept — H94 only became informative because it was measured rather than
assumed, and the same assumption is currently unexamined on the other arch.

x86 nq=100 is x3.431 MT / x3.244 ST and, like ARM nq=100, is far from any
bandwidth roofline, so it is a live compute-bound zone. Both x86 nq=100 cells
are in the goal metric, unlike H95/H96's 2-bit territory.

Method: paired A/B on 136.64.63.204, both directions from the current width,
`load_parity.py` for correctness first, two rounds, all four x86 cells so a
regression at nq=1 cannot hide behind a win at nq=100.

### Wide direction: `NQ_BATCH` 8 -> 12, refuted twice over

| cell | 8 (baseline) | 12 | |
|---|---|---|---|
| nq=100 MT | 19.389 ms | **31.824 ms** | x0.61 |
| nq=100 ST | 98.807 ms | 105.625 ms | x0.94 |
| nq=1 MT | 1.160 ms | 1.047 ms | x1.11 |
| nq=1 ST | 5.261 ms | 3.577 ms | x1.47 |

The nq=1 cells move because `nq.div_ceil(nq_batch)` is 1 either way and the
padding arithmetic changes, not because the kernel got faster; the nq=100 MT
collapse is the answer to the question asked. Twelve queries need `NQ*2` = 24
of 32 zmm registers for accumulators alone, before the permuted code operand,
the level tables and addressing. **x86's spill wall is between 8 and 12, and
`NQ_BATCH = 8` sits under it** — the same shape as H94's ARM result at 12/16,
which is the substance of the hypothesis: on both architectures the batch
width in the code is a register-file boundary, not an inherited guess.

### It also exposed a latent silent-truncation bug

`cargo test --release -p turbovec --lib` failed
`x86_scalar_fallback_tests::scalar_fallback_matches_simd_topk` at bits=2: the
SIMD path returned correct top-k for queries 1-8 and `{0}` for 9-12.

`search.rs:1090` and `:1109` read `for qi in 0..nq.min(8)` against
`acc = [[_mm512_setzero_si512(); 2]; 8]` — the pre-permute-dot x86 kernel
(the one 2-bit still uses) **caps at 8 queries and drops the rest without an
error**. It has been correct only because `NQ_BATCH` has always been <= 8.
Nothing in the 8-cell goal reaches it today, so it is not a shipped bug, but
it is a tripwire under any future width change and the cap should be an
assertion rather than a `.min`. Recorded as a follow-up, not bundled here.

(An unrelated `io::tmp_protocol_tests::sweep_removes_only_stale_matching_temps`
failure appeared in the same run — timing-based, unconnected to search.)

### Narrow direction: `NQ_BATCH` 8 -> 4, also refuted

Paired A/B, alternating arms, two rounds, `--reps 12`. Identical id md5 and
recall@10 = 0.8030 from both widths, so this compares equal outputs.

| cell | 8 (round 1 / 2) | 4 (round 1 / 2) | median ratio |
|---|---|---|---|
| nq=100 MT | 18.114 / 18.067 | 22.702 / 22.737 | **x0.80** |
| nq=100 ST | 81.543 / 75.616 | 116.637 / 117.903 | **x0.67** |
| nq=1 MT | 1.103 / 1.071 | 1.070 / 1.072 | x1.00 |
| nq=1 ST | 3.632 / 3.624 | 3.698 / 3.647 | x0.99 |

nq=1 is unchanged to within noise in both arms, which is the control this
design was for: at nq=1 there is one batch at either width, so any difference
there would have been box drift rather than the variable. The nq=100 cells
move hard and in the same direction on both threading modes.

**Verdict: refuted, and `NQ_BATCH = 8` is confirmed as a peak rather than
merely a value under the wall.** 4 is 20-33% worse (too little amortization of
the shared nibble permute — the same effect H28 measured going 4 -> 8), 12 is
39% worse (register spill). The optimum is a single point between two
mechanisms that fail in opposite directions, which is why a sweep and not a
guess was the right instrument. Reverted to 8; no code change ships.

Taken with H94, both architectures' batch widths are now measured rather than
inherited: ARM 12 (16 spills), x86 8 (12 spills, 4 under-amortizes). Widening
either needs a register *freed*, not a constant raised.

## P43 — x86 nq=1 is at the memory system too, so all four nq=1 cells are closed

H93 fitted ARM nq=1 MT to `0.014 ms fixed + 135.3 GB/s marginal`, flat over an
8x footprint range: 2% fixed overhead, so both ARM nq=1 cells are finished and
the harmonic mean is capped at `8 / 1.819 = 4.40x` no matter what the other six
do. The same fit has never been run on x86, and x86 nq=1 is the pair of cells
with the next-largest reciprocal weight (0.420 MT + 0.362 ST of 3.808).

The probe is the reference-free one H93 settled on after H92's roofline
comparison measured the kernel at 113% of its own reference: sweep the index
footprint over ~8x, fit `time = fixed + bytes / rate`, and read the two
coefficients rather than compare against an external ceiling that may not mean
what it appears to.

Two outcomes, both worth having. A large fixed term says x86 nq=1 is dispatch-
or setup-bound and names a target. A 2% fixed term with a marginal rate near
x86's measured stream bandwidth closes those cells too, which would leave the
four nq=100 cells as the only live zone and put a number on how much of the
goal metric is still reachable at all.

### Result: the second outcome, on both threading modes

| footprint | nq=1 MT | | nq=1 ST | |
|---|---|---|---|---|
| 19.2 MB | 0.357 ms | 53.8 GB/s | 0.735 ms | 26.1 GB/s |
| 38.4 MB | 0.561 ms | 68.5 GB/s | 1.405 ms | 27.3 GB/s |
| 76.8 MB | 1.082 ms | 71.0 GB/s | 3.618 ms | 21.2 GB/s |
| 153.6 MB | 2.368 ms | 64.9 GB/s | 9.010 ms | 17.0 GB/s |

MT fits `0.000 ms fixed + 65.9 GB/s marginal`. The reference, from
`stream_bw` on the same box: 63.2 GB/s over 8 threads at 77 MB, 51.5 GB/s at
512 MB, 11.5 GB/s single-threaded. **The kernel's marginal rate is 104% of the
8-thread stream figure** — nominally over, because the sweep spans 19-154 MB
against a reference fixed at 77 MB and the small end is more L3-resident. The
conclusion does not depend on the overshoot: zero fixed cost and a marginal
rate at the memory system's measured limit means x86 nq=1 MT is saturated.

**ST does not admit the linear model, and that is reported rather than fitted
away.** The regression returns a *negative* 0.836 ms intercept, which is not a
physical quantity; the achieved rate falls monotonically with footprint
(27.3 -> 21.2 -> 17.0 GB/s) as the array outgrows the ~105 MiB L3. Two
coefficients cannot describe a curve that changes regime inside the sweep, so
the table is the result. The verdict is still available from it: 17.0 GB/s at
153.6 MB against 11.5 GB/s single-threaded DRAM says single-core is running
*above* the DRAM rate on cache residency and converging toward it as the
footprint grows. No fixed overhead exists to attack (the intercept is at worst
zero) and no compute slack is visible.

This repeats H95's lesson in a different shape. There the units were wrong;
here the *model* is wrong, and the tell is the same — a coefficient that
cannot be true (a negative intercept, a 113% ratio) is the fit reporting that
it was asked the wrong question.

### What it costs the goal

With H93/P42 on ARM and this on x86, **all four nq=1 cells are memory-bound
with no fixed overhead**, and the only lever left on them is fewer bytes —
which H91 priced and the recall constraint forbids.

The four nq=1 cells contribute `1/1.090 + 1/1.109 + 1/2.381 + 1/2.764 = 2.601`
of the reciprocal sum of 3.808. Frozen, they cap the harmonic mean at
`8 / 2.601 = 3.08x` even with the four nq=100 cells infinitely fast. The
earlier estimate of 4.40x came from freezing ARM nq=1 alone; the real ceiling
is lower. **Current 2.1001x is 68% of everything this architecture can reach
without spending recall.**

The four nq=100 cells are the entire remaining live zone, and they are
compute-bound rather than memory-bound (27% of read bandwidth on ARM, 59% on
x86), so they are a different problem from the one just closed.

## P44 — selection is not a target at k=10, on either box

Every cell this log has closed was closed by an argument about the *scan*.
The top-k epilogue is a separate component none of those arguments touch, and
its cost had never been separated out. Vary k, hold everything else fixed: the
scan does identical work at every k, so the movement is selection.

| | k=1 | k=10 | k=50 | k=100 |
|---|---|---|---|---|
| x86 nq=100 MT | 17.523 | 17.866 | 21.066 | 26.285 ms |
| x86 nq=100 ST | 74.673 | 74.921 | 79.896 | 89.844 ms |
| ARM nq=100 MT | 12.109 | 13.430 | 17.614 | 20.628 ms |
| ARM nq=100 ST | 99.486 | 99.086 | 105.768 | 113.200 ms |

Selection is real at large k — +50% on x86 MT and +70% on ARM MT going to
k=100 — and **almost absent at the k=10 the goal measures**: 1.9% x86 MT,
0.3% x86 ST, 0.4% below noise on ARM ST. Only ARM nq=100 MT shows anything
(9.8%), and that cell's ST twin moves *negatively* over the same step, which
is what a 1.3 ms difference at this scale looks like when it is noise.

Even taking the 9.8% at face value, halving it moves one cell x3.354 -> x3.53
and the harmonic mean x2.1001 -> x2.109: **+0.4%, under the 1% gate**, before
any of it is built. So the epilogue is priced out by arithmetic rather than by
a failed experiment, which is the cheaper way to close an idea.

**The four nq=100 cells are pure scan at k=10.** Combined with P43, the live
zone is not merely small, it is also structurally simple: one loop, no
epilogue worth attacking, and compute-bound rather than memory-bound.

## H98 — the x86 nq=100 scan is not dependency-limited; extra chains only cost

`search_multi_query_permute_dot<NQ, BLK>` carries two const parameters. The
sweeps in this log have all moved `NQ` — H23, H28, and H97 this session. `BLK`,
the number of 32-vector blocks handled per iteration, has never been varied at
nq=100: the batched path instantiates `<NQ_BATCH, 1>` while the nq=1 path uses
`<1, 8>`. The value 1 at nq=100 is inherited exactly the way `NQ_BATCH = 8`
was, and H97 is the argument for not trusting that.

The mechanism is different from `NQ`'s, which is why it is worth a separate
test rather than being folded into H97's result. Widening `NQ` amortizes the
shared nibble permute across more queries; raising `BLK` does nothing for
amortization and instead gives the out-of-order engine independent
`vpdpbusd` chains to interleave, which is the classic remedy when a kernel is
compute-bound but not issue-bound. x86 nq=100 sits at 59% of the box's read
bandwidth, so it is in exactly that regime.

The register arithmetic says this is tight and therefore informative either
way: accumulators scale as `NQ * 2 * BLK`, so the current `<8, 1>` holds 16 of
32 zmm and `<8, 2>` would need all 32 before any operand — H97 showed what
that costs. `<4, 2>` holds the same 16 by trading query width for chain depth,
which is the comparison that actually separates the two mechanisms, since
H97 already measured `<4, 1>` at x0.80.

Method: build `<8, 2>` and `<4, 2>`, A/B both against the shipped `<8, 1>` on
x86, `load_parity.py` first, all four cells. A win at `<4, 2>` over `<4, 1>`
isolates the ILP effect even if neither beats `<8, 1>`.

### Result: refuted, and the isolating comparison is the informative one

Identical id md5 and recall@10 = 0.8030 from all three builds, so this compares
equal outputs. Two alternating rounds, `--reps 12`.

| cell | `<8,1>` shipped | `<8,2>` | `<4,2>` |
|---|---|---|---|
| nq=100 MT | 18.035 / 17.988 | 24.954 / 24.857 (x0.72) | 28.622 / 29.004 (**x0.62**) |
| nq=100 ST | 74.868 / 74.374 | 96.809 / 96.975 (x0.77) | 126.684 / 127.962 (**x0.59**) |
| nq=1 MT | 1.065 / 1.063 | 1.078 / 1.053 | 1.081 / 1.051 |
| nq=1 ST | 3.505 / 3.488 | 3.376 / 3.402 | 3.460 / 3.399 |

`<8, 2>` losing 28% was predicted — 32 zmm of accumulators before any operand,
the same spill H97 measured. **The result that carries information is `<4, 2>`
against H97's `<4, 1>`: 28.8 ms against 22.7 ms at nq=100 MT, and 127.3 against
117.3 ST.** At identical register pressure and identical permute amortization,
doubling the number of independent `vpdpbusd` chains made the kernel *slower*
in both threading modes.

That is a mechanism result, not just another refutation. If the scan were
waiting on the accumulate dependency chain, extra chains at constant register
pressure is precisely the remedy, and it would have shown a gain here. It
showed a loss, so **the x86 nq=100 scan is issue-limited rather than
latency-limited** — the ports are full, and handing the out-of-order engine
more independent work only adds addressing and loop overhead to a machine that
has nothing spare to overlap it with.

This is the x86 counterpart of what P10 and H23 established on ARM ("lookups
are at maximum issue rate"), reached by a different route. Both nq=100
architectures are now issue-limited by measurement rather than by assumption.

Reverted to `<NQ_BATCH, 1>`; no code change ships.

**Where this leaves the live zone.** P43 closed the four nq=1 cells on the
memory system. P44 closed the top-k epilogue on arithmetic. H97 and H98 close
both const parameters of the one remaining loop, on spill and on issue rate.
The four nq=100 cells are compute-bound, at their issue ceiling, with no
epilogue and no scheduling slack. Anything further must reduce the *number of
operations* — not schedule them better and not move fewer bytes — which points
the next hypotheses at the codebook and level-table family rather than at the
loop, and that family has a recall price the standing constraint has already
refused twice (H50, H91).

## H99 (next) — AMX-INT8 raises the issue ceiling H98 just proved x86 is against

H98 established that x86 nq=100 is issue-limited: the ports are full and no
rescheduling helps. There are exactly two ways past an issue ceiling — fewer
operations, or operations that do more each. The op-count route runs into the
recall constraint (H50, H91). The second route has never been tried on x86,
and the hardware for it is on the bench box.

`lscpu` on the Xeon Platinum 8481C reports `amx_tile`, `amx_int8`, `amx_bf16`.
`TDPBSSD` multiplies a 16x64 INT8 tile by a 64x16 INT8 tile into a 16x16 INT32
accumulator — 16384 MACs per instruction against `vpdpbusd`'s 64. Even at a
several-cycle issue interval that is a large multiple of the per-slot work the
current kernel gets, which is the only quantity H98 says is binding.

The shape already exists in this codebase on the other architecture. ARM's vm8
layout was built so the TBL output *is* the SMMLA B operand — unpacked level
bytes fed straight into a matrix unit. x86's permute-dot already produces the
same permuted level bytes and then spends them on `vpdpbusd`. **The hypothesis
is that those bytes can be spent on AMX instead**, making this a change of
consumer rather than a new data layout, with the 16-query tile width sitting
naturally against a batch the register file currently caps at 8.

Three things to settle before any kernel work, in order:

1. **Reachability.** AMX intrinsics are unstable in `core::arch`
   (`x86_amx_intrinsics`), and H52 was already blocked on stable Rust. Inline
   `asm!` is stable, so the question is whether `LDTILECFG`/`TDPBSSD` are
   expressible that way without a nightly toolchain. If not, this is blocked
   and says so quickly and cheaply.
2. **Throughput, standalone.** A microbenchmark of `TDPBSSD` against
   `vpdpbusd` on the same box, on data already in registers/tiles, so the
   comparison is issue rate and nothing else. This is the <3 minute smoke.
3. **Tile-load cost.** AMX reads tiles from memory with a stride, and
   `LDTILECFG` plus the AMX/AVX transition are not free. A unit that is 8x
   faster per instruction and needs three loads per use may not clear.

Only if all three pass does the kernel get touched. Recorded now because the
reasoning is the deliverable whether or not step 1 survives: **it is the first
idea in this log that raises the ceiling rather than approaching it**, and the
four remaining cells have no other lever that does not cost recall.

### Step 1: reachable from stable Rust

rustc 1.95.0 stable, no nightly, no `core::arch` intrinsics. `LDTILECFG`,
`TILELOADD`, `TILEZERO`, `TDPBSSD` and `TILERELEASE` all assemble inside
`asm!`, and `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` — required
before the first tile instruction or it is SIGILL, not an error — returns 0
through a raw `syscall`. **H52's blocker does not apply here.**

One trap worth recording: the 64-byte tile config has `colsb[16]` at offset 16
and `rows[16]` at 48, and a struct that puts `rows` at 64 segfaults on
`ldtilecfg` with no diagnostic. That cost one run.

### Step 2: 6.36x the issue rate, and the denominator took two tries

| | Gmac/s | |
|---|---|---|
| AMX `TDPBSSD` | 2425.7 | median of 6, five within 0.3% |
| AVX-512 `vpdpbusd` | 381.3 | 110% of the 2.7 GHz base-clock ceiling (the box boosts) |
| | **x6.36** | |

**The first version of this probe answered 18.58x, and it was wrong.** The
VNNI loop used four accumulator chains against a ~5-cycle-latency instruction
that issues twice per cycle, so it measured `vpdpbusd`'s *latency* and called
it the issue rate — understating the denominator by 2.9x. Sixteen chains, the
number the real kernel holds at `NQ_BATCH = 8`, gives 381 Gmac/s, which is
110% of the arithmetic ceiling at base clock and therefore a credible issue
rate rather than a stalled loop.

This is H95's error caught before it was published rather than after: a ratio
is only as good as its denominator, and the check that caught it was computing
what the denominator *should* be (2 x 64 x 2.7 GHz) and noticing 129.9 was 38%
of it. Every ratio in this log now gets that check.

One run in six reported 1451 Gmac/s instead of 2425. That is the AMX frequency
ramp on a cold tile unit, not a distribution — the other five agree to 0.3% —
but it is a warning that any AMX result measured in a single short run is
untrustworthy.

### Step 3: the tile-load cost is real and almost entirely amortizable

Same process, same frequency state, three regimes, plus a fourth added after
the first three answered badly.

| regime | Gmac/s | of ceiling | vs `vpdpbusd` |
|---|---|---|---|
| resident (no loads) | 2425 | — | x6.36 |
| loaded, 1 `TILELOADD` : 1 `TDPBSSD` | 704 | 29% | x1.85 |
| mixed (+ AVX unpack interleaved) | 704 | 29% | x1.85 |
| **amortized, 1 : 3** | **2410** | **99%** | **x6.32** |

**Feeding a fresh B tile per multiply throws away 71% of the multiplier, and
reusing it three times gets all of it back.** The 1:1 loop was the worst case
and not the kernel: at nq=100 there are seven groups of 16 queries, so the
codes for a block are genuinely consumed seven times over. Three was chosen
only because the tile file holds eight and 3 A + 1 B + 3 C is seven of them.

The AVX-interleaved regime measuring identically to the plain loaded regime is
its own small result: the AMX/AVX transition penalty did not appear, so an
unpack-then-multiply kernel does not pay a unit-switch tax on this part.

**The bimodality from step 2 is real and machine-wide.** Runs land in one of
two states — resident 2425 with amortized 2410, or resident ~1250 with
amortized ~760 — and every regime moves together within a run. Three of six
runs in each. On a shared 8-vCPU VM this is a frequency licence or a co-tenant,
not the code, but it means **the honest range is x2.0 to x6.3 on the multiply,
not a single number**, and any AMX result measured in one run is worthless.

### Verdict: H99 passes all three gates and is worth building

The multiply gets 2.0-6.3x once tile loads are amortized 3:1, on the exact
quantity H98 proved binding, reachable from stable Rust today.

Two honest deductions before anyone expects that on the cells. The probe
charges nothing for the *staging* a real kernel needs — the unpacked level
bytes must be written somewhere `TILELOADD` can read them as 64 dims x 16
vectors, which is a layout the block format does not currently produce, and
those stores are new work this measurement does not contain. And scoring is
not all of the cell, so Amdahl applies twice over.

Even so the arithmetic is worth stating, because it is the first thing in this
log that could move the metric materially: if the two x86 nq=100 cells went
x3.431 -> x5.15 and x3.244 -> x4.87 (a 1.5x cell-level gain, well under the
multiply-level range), the reciprocal sum falls 3.808 -> 3.608 and the
harmonic mean goes **x2.1001 -> x2.217, +5.6%**. Against a ceiling of 3.08x
that is a fifth of the entire remaining distance.

Next: build the staging layout and an AMX scan for the x86 nq=100 path, keeping
`<NQ_BATCH, 1>` intact as the fallback for nq < 16 and for non-AMX hosts.

### Build result: a correct AMX kernel that reaches parity, not x6.36

The prototype scores 96 queries against a blocked index with AMX and matches a
scalar reference exactly (0 of 393216 mismatches), so this is a working kernel
being measured, not a failed port.

The layout needed no change at all, which was the encouraging part. The
existing kernel already loads 64 bytes as 16 dwords, each dword one database
vector holding 4 byte-groups — **that register is already a VNNI-packed B-tile
row**, and 16 of them stacked are a 16x64 B tile. `TDPBSSD` also takes signed
levels, so the `+128` bias `vpdpbusd` forced and its compensating `pd.zero`
seed both disappear.

| variant | ns/vec/query | Gmac/s |
|---|---|---|
| shipped `vpdpbusd` kernel, x86 nq=100 ST @ 200k | 3.74 | — |
| AMX, 3 query groups, A read in place at stride 768 | 4.05 | 190 |
| AMX, 3 groups, A repacked to contiguous 1 KiB tiles | 3.69 | 208 |
| AMX, 6 groups (6 C + 1 B + 1 A) | 3.69 | 208 |
| AMX, 6 groups, all six B tiles staged before any load | 3.69 | 208 |
| AMX, 6 groups, **A reload deleted** (wrong answers, timing only) | 2.53 | 304 |

**Parity.** And the prototype runs on a 1.5 MB L2-resident array while the
3.74 ms figure streams 76.8 MB, so like-for-like it is behind.

**The mechanism, and it invalidates step 3's optimism.** Two of the three
tuning attempts above did nothing — 6 groups instead of 3 bought 0%, and
batching the staging stores to give the store buffer one drain instead of six
bought 0%, which refutes the store-to-load-forwarding explanation cleanly. The
attribution probe found it instead: deleting the A tile reload is worth 41%.

`TDPBSSD` needs both operands in tiles, both operands change every 64 dims, and
the tile file has **no register renaming** — so `tileloadd tmm6` cannot begin
until the previous `tdpbssd` has finished reading tmm6. A 768-dim dot product
is 12 such reloads per C tile no matter how the loop is arranged, because
every arrangement that keeps an operand resident forces C into memory instead,
and C traffic is strictly worse.

**Step 3's x6.32 was measured on a loop that reused the same A tile forever.**
That is achievable in a microbenchmark and unreachable in a dot product over
768 dimensions. The number was real; it was an answer to a question no kernel
asks. This is the same failure as the x18.58 of step 2 and the units error of
H95 — the third time in this log that a favourable ratio came from a
denominator or a setup that did not match the thing being predicted, and the
first time it survived to the build stage before being caught.

**Verdict: refuted.** Even the unreachable no-A-reload bound is 2.53 ns/vec/
query, x1.48 on part of one cell pair, and the achievable figure is parity.
The 6.36x of raw issue rate is real and cannot be spent: reaching it requires
operands that stay in tiles, and this problem's operands cannot.

What would change the answer is a machine with tile renaming, or a dim count
small enough that all 12 dim-tiles of A fit resident (768 dims is 6x too many),
or an AMX generation whose `TILELOADD` pipelines against `TDPBSSD`. None of
those are this box.

The prototype and its scalar reference are kept — it is a correct AMX scan and
a working harness if any of those conditions change.

### Original framing of step 3, kept for the record

6.36x of headroom on the one quantity
H98 proved binding is a large prize, and the remaining unknown is whether
feeding the tiles costs more than the multiplier is worth: `TILELOADD` per
16x64 operand, the AMX/AVX transition penalty, and a B operand that must
arrive as 64 dims x 16 vectors when the code array is stored 32-vector blocks
by byte-group. That last item is the vm8 problem again, on the other arch.

## H100 — the rotation that makes 4-bit work forecloses exact prefix bounds

Every op-reduction idea this log has reached — H50's uniform codebooks, H91's
prefix shortlisting — was closed by the recall constraint, because each
changed which vectors win. A Cauchy-Schwarz bound does not: score a prefix of
P dims exactly, add `||q_rest|| * ||v_rest||` as the most the remainder could
contribute, and a vector whose total still falls below the running k-th best
provably cannot enter the top-k. **The returned top-k is bit-identical**, so
this is the one member of the op-reduction family that is free under the
standing constraint.

It is also the only idea that helps both zones at once: fewer ops for the
issue-limited nq=100 cells (H98), and fewer *bytes* for the memory-bound nq=1
cells, which P43 said was their only remaining lever. Cost is one f32 of
remainder norm per vector, 800 KB against a 76.8 MB code array.

### Smoke: the skip rate is a property of the data, so measure it on real data

| data | P=128 | P=256 | P=384 | P=512 | P=768 |
|---|---|---|---|---|---|
| **real OpenAI-1536, block-level** | 0.00% | 0.00% | 0.00% | 0.00% | **2.92%** |
| synthetic power-law, per-pair | — | ~100% | — | — | — |
| synthetic uniform Gaussian, per-pair | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

**Refuted.** Half the dimensions bought nothing at all, and even three
quarters of the way through the vector the bound proves 2.92% of blocks
unreachable. The measurement is deliberately generous — it charges block
granularity and a progressive threshold, but a 200k index and only 100
queries, and it still finds nothing.

**The cause is turbovec's own design, which is why this closes the family and
not just the idea.** A prefix bound pays only when energy is concentrated in
early dimensions, so that `||v_rest||` becomes small. turbovec applies a
block-Hadamard rotation before quantizing precisely to *spread* energy evenly
across dimensions — that is what makes 4 bits per dimension survivable at all.
Every dimension carries the same expected magnitude by construction, so
`||v_rest||` after P dims is `sqrt((D-P)/D)` of the whole and the bound stays
loose until almost the end. **The rotation that buys the recall forbids the
skip.** Any future prefix-, partial-distance- or early-termination idea in
this codebase meets the same wall and can be closed by citing this entry.

### And the synthetic generator was off by the whole answer

The power-law generator said ~100% prunable; real embeddings say 0%. That
generator is the same "kinder, more realistic than Gaussian" shape H91 used to
avoid closing on uniform data alone — and here it was not conservative, it was
wrong in the favourable direction by the entire result. **Uniform-random and
hand-shaped synthetic spectra are both unfit to decide this class of question;
only the real vectors are.** The box has had `openai-1536.npy` the whole time.

Cost: two probe runs, no kernel work, closed at the smoke gate.

## H101 — ARM nq=1 MT is at 71% of bandwidth, and P43 closed it too early

P43 and H93 closed the four nq=1 cells. Re-reading them: H93 measured ARM
nq=1 MT's marginal rate at 135.3 GB/s, and P42 measured 192.5 GB/s available
at that footprint. **That is 71%, not saturation.** What H93 actually
established is that fixed overhead is 2% — the cell is not overhead-bound —
and I generalised that into "closed", which is a different claim about a
different quantity. The cell matters more than any other: ARM nq=1 MT alone
carries 0.917 of the 3.808 reciprocal sum.

### The premise was wrong and the measurement is better than the premise

I expected the nq=1 pass to be *more* expensive than the batched pass, on the
arithmetic that nq=1 MT is 0.917 ms for one pass while nq=100 MT does nine
passes in 14.4 ms. Sweeping nq over the range where the pass count stays at
one says the opposite:

| ARM nq | total | ms/pass | GB/s | us/query |
|---|---|---|---|---|
| 1 | 0.560 | **0.560** | **137.3** | 559.5 |
| 2 | 1.046 | 1.046 | 73.4 | 523.2 |
| 4 | 0.943 | 0.943 | 81.5 | 235.6 |
| 8 | 1.235 | 1.235 | 62.2 | 154.4 |
| 12 | 1.618 | 1.618 | 47.5 | 134.8 |
| 100 | 13.324 | 1.480 | 51.9 | 133.2 |

**The nq=1 pass is the cheapest pass the kernel has**, by 1.7x over the next
one. That is the right result and it dissolves the premise: the batched
kernel does more arithmetic per byte, so its pass is slower, and P10's "the
batched scan is compute-bound" is visible here as a byte rate falling from
137 GB/s to 47 as queries are added.

So the headroom is real but it is not the one I went looking for. nq=1 moves
bytes at 137.3 GB/s where the machine offers 192.5 — **29% of the memory
system is unused by the cell with the largest weight in the metric.**

Two incidental findings from the same table. **nq=2 is slower than nq=4**
(1.046 against 0.943) despite padding to the same `qbs = 4` and doing
identical work — an 11% penalty for having fewer queries, outside the goal's
cells but real. And nq=1 is not merely nq=4 with padding: at 0.560 against
0.943 it is plainly a different, cheaper path.

### The mechanism to test

x86 hit this exact wall and H54 fixed it: at nq=1 only two accumulators are
live, so the kernel walks one sequential stream and outstanding misses are
capped at what one stream sustains. The fix was `BLK = 8` — interleaving eight
independent block streams — and x86's nq=1 path instantiates `<1, 8>` to this
day. The ARM nq=1 path has no equivalent; it walks `block_start..block_end`
one block at a time.

**Hypothesis: ARM nq=1 is short of memory-level parallelism for the same
reason x86 was, and the same fix applies.** If it reaches even 180 GB/s the
cell goes x1.09 -> ~x1.45, the reciprocal sum falls 3.808 -> 3.605, and the
harmonic mean goes x2.100 -> x2.219, **+5.7%** — the largest single move
available anywhere on the board.

### Result: refuted, and the first A/B was measuring the wrong thing

`scan_range_neon` had no prefetch at all, so a `prfm pldl1keep` lookahead went
in at depth 1 and depth 2 blocks, a line every 256 bytes. Identical id md5 and
recall in every arm.

| cell | pf0 (no prefetch) | depth 1 | depth 2 |
|---|---|---|---|
| nq=100 MT | 12.548 | 12.553 | 12.569 |
| nq=100 ST | 98.56 | 98.61 | 98.55 |
| nq=1 MT | 0.576 | 0.600 | 0.583 |
| nq=1 ST | 3.745 | 3.869 | 3.875 |

**No gain at nq=1 MT and a ~3% regression at nq=1 ST, at both depths.** The
hardware prefetcher already owns this stream — it is one long sequential walk,
the easiest possible case for it — and the extra instructions cost what they
cost. H62's finding on x86, that a deep lookahead helps the 12.5-sweep nq=100
scan and *hurts* the single-sweep nq=1 case, reproduces on aarch64 as a
regression at every depth.

The 29% bandwidth gap is real and remains unexplained. It is not prefetch.

### The first A/B showed a reproducible +6.5% that did not exist

Before this table, the same experiment run against `so_base.so` reported the
prefetch build 6.5% faster at nq=100 MT and 2.5% at nq=100 ST — consistent
across two independent builds and two alternating rounds, which is normally
exactly what a real effect looks like.

It was not real. `so_base.so` was the artifact already sitting on the box from
an earlier deploy, not a build of the current tree, so the comparison measured
some accumulated difference between that older build and HEAD and attributed
it to the prefetch.

**What exposed it was not a statistic — it was that the win appeared in a cell
the change cannot reach.** `scan_range_neon` runs only when `nq == 1`; a
prefetch inside it can no more affect nq=100 than it can affect x86. Rebuilt
with a same-tree control, the nq=100 cells agree to within 0.2% across all
three arms, which is the validity check the first run lacked.

The rule this earns: **a reproducible A/B difference in a cell the change
cannot causally reach is proof the control is wrong, not evidence of a win.**
Reproducibility and alternating arms defend against noise and drift; neither
defends against the wrong baseline. Every A/B in this log from here builds its
control from the same tree in the same session, and the cells a change cannot
touch are read as the control channel they are.

This one nearly shipped. A +6.5% on x86 nq=100 MT and +2.5% ST would have
been recorded as a confirmed improvement and committed.

## H102 — the 29% gap was a denominator error; the cell is at 91% scaling

H101 reopened ARM nq=1 MT on the grounds that it moves 137.3 GB/s against the
192.5 GB/s P42 measured available. Decomposing that gap into its two factors
dissolves most of it.

| threads | ms | GB/s | speedup | efficiency |
|---|---|---|---|---|
| 1 | 3.891 | 19.7 | 1.00x | 100% |
| 5 | 0.796 | 96.5 | 4.89x | 97.8% |
| 6 | 0.680 | 113.0 | 5.72x | 95.4% |
| 7 | 0.607 | 126.6 | 6.41x | 91.6% |
| 8 | 0.536 | 143.4 | 7.27x | **90.8%** |
| 9 | 0.906 | 84.7 | 4.29x | 47.7% |
| 10 | 0.842 | 91.2 | 4.62x | 46.2% |

The scaling is clean to the core count and falls off a cliff past it — 9
threads on 8 vCPUs costs 69%, which is worth knowing but is not a bug in
anything we control. An earlier run of this sweep read 76.4% at 8 threads;
this one reads 90.8% for the same build. **The 8-thread point is the noisiest
on the curve** — the whole search is 0.5 ms, so one descheduled worker puts an
entire range on the critical path — and a single reading of it should not be
trusted, which is how the first sweep misled me.

**The chain, correctly assembled.** Single-core runs 19.7-20.5 GB/s against
the 21.13 GB/s single-core streaming roofline P42 measured: 95%. Eight cores
scale that at 90.8%. 8 x 20 x 0.908 = 145 GB/s, which is what the cell
achieves. The 192.5 GB/s figure is `stream_bw`'s **pure read loop with eight
accumulators and no work per byte** — our scan does a TBL, an SMMLA and an
accumulate for every byte it reads, so it was never going to reach a number
measured by a kernel that does nothing.

**This is the fourth denominator mismatch in this log** — H95's units, step 2's
latency-bound VNNI loop, step 3's forever-resident A tile, and now a pure-read
bandwidth figure used as a ceiling for a compute-doing scan. The pattern is
consistent enough to name: *a ratio is only meaningful when the denominator was
produced by something facing the same constraints as the numerator.* Three of
the four were caught, one shipped as far as a build.

**P43's verdict stands, with a better justification than P43 gave.** It closed
the cell on 2% fixed overhead, which was the wrong argument for the right
conclusion. The correct one: single-core is at 95% of what one core can stream,
and this cell is 8 of those at 91% efficiency.

### What is actually left there

The recoverable slice is the 9% scaling inefficiency, not 29%. At perfect
scaling the cell would go 0.536 -> 0.487 ms, x1.09 -> x1.20, the reciprocal sum
3.808 -> 3.724, and the harmonic mean x2.100 -> x2.148: **+2.3%**. That clears
the 1% gate and is worth a hypothesis, but it means removing fan-out and tail
cost from a 0.5 ms operation whose fixed overhead H93 already measured at 2% —
so at most 7 of those 9 points are addressable, and the honest expectation is
under +2%.

## H103 — the 9% is not steal-starvation; more ranges is monotonically worse

H102 left one addressable item: ARM nq=1 MT scales at 90.8%, and the 8-thread
point swings between 76% and 91% run to run, which reads as a tail.
`block_range_stride(6250, 8)` returns 782, so the single-query path makes
**exactly 8 ranges for 8 threads** — rayon has nothing to steal, and a worker
that falls behind cannot be helped. Giving it 4 or 8 ranges per thread is a
two-line change and the ranges stay long (196 blocks, 2.4 MB).

| ARM | 1/thread (control) | 4/thread | 8/thread |
|---|---|---|---|
| nq=1 MT | **0.570** | 0.601 (x0.95) | 0.645 (**x0.88**) |
| nq=1 ST | **3.727** | 3.846 | 3.843 |
| nq=100 MT | 12.580 | 12.579 | 12.558 |
| nq=100 ST | 98.859 | 99.179 | 98.416 |

**Refuted, and monotonically**, which is the useful shape: more ranges is
worse in proportion to how many more, so this is a cost that scales with range
count and not a threshold effect. Each range allocates a heap `Vec`, `collect`s
its candidates into another, and shortens the sequential run the hardware
prefetcher is riding — and H101 just established that this stream is entirely
the prefetcher's to own. The ST regression confirms the reading: single-
threaded, 8 ranges per thread means 8 ranges rather than 1, with no balancing
benefit possible at all, and it costs 3%.

**The 9% scaling loss is not tail imbalance.** Whatever it is survives having
the work finely divided, so it is per-thread cost rather than per-thread
variance — memory-system contention between 8 cores on the same controller is
the remaining candidate, and that is not something the scheduler can fix.

`block_range_stride`'s one-range-per-thread choice is now measured rather than
inherited, which is the same audit H97 gave `NQ_BATCH` and H94 gave `qbs`.

### The control channel worked

Per H101's rule, the baseline was built from the same tree in the same session,
and the four nq=100 cells — which `search_single_query_block_parallel_neon`
cannot reach — agree to within 0.6% across all three arms. That is what a
sound A/B looks like, and it is exactly the check the phantom +6.5% failed.

## H104 — the ARM tile multiplier is at a flat optimum, and cache blocking already exists

Two layout questions, both closed cheaply.

**Loop order.** The idea was that a batched scan re-reads the code array once
per query batch — 9 times on ARM at `qbs = 12` — so tiling the vector axis to
a cache-resident chunk would turn eight of those into L2 hits. Reading the
code shows both architectures already do exactly this: the tile list is
block-range-major on ARM and x86 alike, so for a fixed block range every query
batch runs consecutively over bytes already in cache. H7/H9 established this
and measured it at x1.019 ARM. **Closed by reading, not by measuring.**

**Range size.** What H7/H9 did not fix is how *large* those ranges are, which
is chosen by `n_block_ranges` for thread balance rather than for cache
residency. `TV_NEON_MULT` exposes the tiles-per-thread multiplier, so this
sweeps with no rebuild at all — and therefore with no baseline to get wrong.

| `TV_NEON_MULT` | nq=100 MT | nq=100 ST |
|---|---|---|
| 8 | 12.662 | 99.195 |
| 16 | 12.717 | 98.990 |
| 32 | 12.670 | 98.099 |
| **64 (default)** | **12.485** | 98.656 |
| 128 | 12.512 | 99.613 |
| 256 | 12.702 | 99.165 |

**Flat.** A 32x range of the multiplier moves nq=100 MT by 1.9% with no
monotone trend, and nq=100 ST by 1.5% with its minimum at a different setting
— the signature of noise, not of a tuning curve. The default sits at the best
MT reading, but not by more than the spread. nq=1 is unmoved, as it must be:
that path does not use this multiplier, which is the control channel doing its
job again.

The reason the curve is flat is visible in the arithmetic: at the default the
ranges are already ~12 blocks, about 147 KB, comfortably inside L1/L2. The
cache blocking this hypothesis wanted to introduce is not merely present, it
is already finer than the level it was aiming at.

Cost: one sweep, no build, no code change.

## H105 — a lookup-free scan exists, costs H50's recall, and prices the constraint

If the 16 levels were uniformly spaced, `level[c] = a*c + b`, and the whole
table lookup disappears:

```
sum_d w[d] * level[code[d]]  =  a * (w . code)  +  b * sum_d w[d]
```

The second term is a per-query constant the existing bias machinery already
carries. The first takes the **raw nibble** as the multiply operand, so the
kernel never materialises a level byte at all — no `vpshufb`, no `TBL`. The
nibble split (`and`, `srli`) is still needed, because `vpdpbusd` and `SMMLA`
take bytes, but the lookup itself goes.

Op count per 64-byte load, x86 GFNI path at `NQ_BATCH = 8`:

| | shared unpack | per-query | total |
|---|---|---|---|
| now | `and`, `vpshufb`, `gf2p8affineqb`, `vpshufb` = 4 | 16 `vpdpbusd` | 20 |
| lookup-free | `and`, `gf2p8affineqb` = 2 | 16 `vpdpbusd` | 18 |

**~10% fewer instructions in a kernel H98 proved is issue-limited**, which is
the one regime where an instruction removed is time saved. ARM is the same
shape: the `TBL` that feeds the `SMMLA` B operand becomes the raw nibbles.

### It requires a uniform codebook, which is H50

`build_codebook` in `codebook.rs` runs Lloyd-Max against a Beta(a, a) prior —
centroids initialised uniformly, then iterated to conditional means until they
stop moving. The table is non-uniform **by construction and on purpose**: that
is what makes 4 bits carry the accuracy it does. There is no decomposition
that recovers exactness, since `level[c] = a*c + b + r[c]` leaves a residual
`r` that needs exactly the lookup being removed.

So this is H50 — uniform codebooks — reached from the opposite direction. H50
was priced at **-0.021 recall** and closed by Ryan's standing instruction that
recall is not traded. That decision stands and this does not reopen it.

### What is new is the price tag

H50 was recorded as an accuracy question. It is also an instruction-count
question, and this is the first entry to put a number on the other side of
that trade. If both x86 nq=100 cells gained the full 10%, x3.431 -> x3.774 and
x3.244 -> x3.568, the reciprocal sum falls 3.808 -> 3.754 and the harmonic
mean goes x2.1001 -> x2.1313. With ARM moving similarly the total is roughly
**+3% on the metric**.

**The recall constraint is costing about 3% of the goal figure.** That is a
fact worth having explicitly rather than implicitly, and it is Ryan's call to
make with the number in hand — the standing answer is no, and the work
continues under it.

Cost: no build, no measurement. Refuted by reading `codebook.rs`.

## H106 — the unpack is free on ARM and costs exactly its share on x86

Web research (inspiration was under the threshold) returned a mostly clean
negative: nothing published gets below one unpack plus one dot-accumulate per
code byte with a non-uniform table, nobody reports beating one dot-accumulate
per issue slot, and FastScan's inner loop has not changed in a way we have not
matched. Full source list in the research note appended to this entry.

Its one actionable claim was an *inference*, flagged as such: Neoverse V2 puts
`SMMLA` on pipes V0+V2, so if `TBL` issues there too, every unpack steals a
multiply slot. Same shape on x86, where `vpshufb` zmm is p5-only while
`vpdpbusd` is p0+p5. In an issue-limited kernel that would matter a lot. It is
measurable in twenty lines, so it got measured rather than believed.

### ARM: TBL is free alongside SMMLA

| loop (8 independent instrs) | Ginstr/s |
|---|---|
| `tbl` x8 | 11.96 / 10.81 |
| `smmla` x8 | 8.79 / 9.35 |
| **mixed 4 + 4** | **11.82 / 11.97** |
| `and` x8 (control, all four V pipes) | 11.89 / 11.96 |

Mixing runs at the rate of `TBL` alone and *faster* than `SMMLA` alone. If they
shared pipes the mix would be capped by the slower one; instead the TBLs fill
slots the SMMLAs were not using. **SMMLA is the bottleneck instruction and the
unpack rides along for free.** The inference is wrong for this core.

### x86: the shuffles cost exactly their instruction count, and no more

A 1:1 mix also read as disjoint, but the kernel's real ratio is 2 shuffles per
16 dot-accumulates, which is a different regime — that ratio would put 10 uops
on p5 against 8 on p0 if the port assignment were as documented. Measured at
the real ratio:

| | time |
|---|---|
| 16 `vpdpbusd` | 0.0538 s |
| 16 `vpdpbusd` + 2 `vpshufb` | 0.0608 s (**+12.8%**) |

Two instructions added to sixteen is +12.5% if they cost exactly their share.
The measurement is +12.8% twice. **No p5 theft, no free ride** — the unpack
costs its instruction count and nothing else.

### This corrects H105's price tag downward

H105 estimated the lookup-free formulation at ~10% on both architectures and
put the recall constraint's cost at roughly 3% of the metric. Half of that is
now measured away:

* **ARM: the win is ~0.** Deleting the `TBL` frees a slot `SMMLA` cannot use.
  The ARM kernel is limited by the matrix unit's own throughput, which is what
  P10 and H23 concluded by a different route and this confirms directly.
* **x86: the win is real at ~11%** of the dot-product portion, which is what
  the +12.8% is measuring.

So the lookup-free scan is worth roughly **+1.5% on the harmonic mean, not
+3%**, and only on the two x86 nq=100 cells. **The recall constraint is
cheaper than H105 said.** Recorded as a correction to that entry rather than
edited into it.

### Two items parked from the research, both with a reason

* **Arm SME2 `LUTI4`** does arbitrary non-uniform nibble→level in one
  instruction from a 64-byte `ZT0` table — precisely the instruction that
  would make a non-uniform table free. It needs SME2, and published
  microbenchmarks put int8 `SMOPA` on the only shipping consumer part at 2-3x
  *slower* than NEON DOTPROD. That is H99's failure mode exactly: a wider unit
  reached through a coprocessor whose feed cost exceeds its width. Park until
  a Neoverse core with in-core SME2 ships.
* **Elastic's OSQ** keeps a uniform grid and recovers accuracy with per-vector
  interval optimisation rather than a shared non-uniform codebook, reporting a
  26% recall gain from per-vector intervals. That is the only route found that
  reaches the lookup-free win without spending recall. It is a recall
  experiment on the quantizer, not a kernel change, and Elastic argue directly
  that data-dependent centering beats a data-oblivious rotation — a claim
  against the block-Hadamard front end this whole design rests on. Out of
  scope for a search-latency hill-climb; worth its own investigation.

## H107 — the recall constraint costs 0.84%, below this project's own gate

H106's ARM half was wrong, and the way it was wrong is worth more than the
conclusion. It compared *instruction* rates: a mixed TBL+SMMLA loop retired
11.9 Ginstr/s against 9.1 for SMMLA alone, and I read that as "TBL is free".
But the mixed loop was half TBL, so it retired ~2 SMMLA/cycle where the pure
loop retired ~3. **Comparing instructions per second hid a fall in multiplies
per second.** The pure-SMMLA loop was also latency-bound — 8 accumulator
chains against a 3-cycle instruction — which is the same defect H99 step 2
caught in its VNNI denominator, repeated four entries later.

### Corrected microbenchmark, at the kernel's verified ratio

The ARM inner loop is exactly 2 `vqtbl1q` and 12 `smmla` per iteration
(`score_block_permute_smmla_neon`, `NP = 6` at `qbs = 12`), so the probe uses
that ratio and 12 accumulators.

| | time | Gsmmla/s |
|---|---|---|
| 12 `smmla` | 0.0201 s | 11.94 |
| 12 `smmla` + 2 `tbl` | 0.0267 s | 8.98 (**+32.8%**) |

Modal over six runs, the pure loop stable to 4 digits. +16.7% would be "costs
exactly their instruction count", so the TBLs cost about **twice** their
share. That reverses H106 and predicts a large ARM win.

### The kernel says no, and the kernel is authoritative

Both probes were then run *in the real kernel* rather than in isolation. With
a uniform codebook the raw nibbles **are** the correct SMMLA/`vpdpbusd`
operand — `level[c] = a*c + b`, with `a` and `b` folding into the existing
scale and bias — so replacing the lookup with the raw nibble times the genuine
lookup-free kernel and only the scores come out wrong.

| cell | with lookup | lookup-free | |
|---|---|---|---|
| ARM nq=100 MT | 12.573 | 12.548 | **x1.00** |
| ARM nq=100 ST | 98.89 | 99.14 | **x1.00** |
| x86 nq=100 MT | 18.090 | 16.896 | x1.071 |
| x86 nq=100 ST | 77.82 | 75.04 | x1.037 |

**ARM gains nothing at all**, despite the microbenchmark promising 33%. The
kernel has loads, `vand`, `vshr` and two `vzip`s filling the same slots the
TBLs were accused of stealing; remove the TBLs and other work simply takes the
issue bandwidth. H106's conclusion was right for the wrong reason, and the
corrected microbenchmark was wrong for a good one.

**That is three consecutive entries where an isolated instruction-level
measurement predicted something the assembled kernel did not show** — H99's
tile-feed probe, H106's ARM mix, and now H107's corrected version. The rule to
carry: *a microbenchmark bounds what an instruction can cost; only the kernel
says what removing it is worth.*

### The number this was all for

x86 does gain, and less than its own +12.8% microbenchmark said. Taking the
measured cell speedups, x3.431 -> x3.676 MT and x3.244 -> x3.364 ST, the
reciprocal sum falls 3.808 -> 3.7775 and the harmonic mean moves
**x2.1001 -> x2.1178: +0.84%**.

**Below the 1% gate this project uses to call something an improvement.**

So the arc across three entries — H105 estimated the recall constraint at
+3%, H106 corrected it to +1.5%, and measuring it in the kernel puts it at
+0.84% — ends with the whole uniform-codebook question retired for this goal.
Trading 0.021 recall would not buy a change this log would be allowed to
record as a win. **Ryan's standing "recall is not traded" costs nothing
measurable here**, and that is now a measured statement rather than a
deference.

Both probes reverted; no code change ships.

## H108 — the headline figure was x2.0477, not x2.1001

The score had not been re-derived in a long time, and today gave two reasons
to distrust it: the x86 box reported nq=100 ST anywhere from 74.7 to 105.6 ms
within one session (the bimodal frequency state the AMX probes exposed), and
H101 produced a phantom +6.5% that reproduced across two builds purely because
its control was a stale artifact. A baseline captured in one machine state
against a head captured in another is wrong in a way nothing else catches.

So both arms were rebuilt from source in one session through the same deploy
path — `main` exported with `git archive` so no working-tree state leaks in,
head deployed normally — and alternated over three rounds on each box.

| cell | main | head | recorded | **re-derived** |
|---|---|---|---|---|
| arm nq=100 MT | 41.949 | 12.626 | x3.354 | **x3.322** |
| arm nq=100 ST | 317.811 | 98.852 | x3.221 | **x3.215** |
| arm nq=1 MT | 0.614 | 0.589 | x1.090 | **x1.041** |
| arm nq=1 ST | 4.126 | 3.766 | x1.109 | **x1.096** |
| x86 nq=100 MT | 62.091 | 18.083 | x3.431 | **x3.434** |
| x86 nq=100 ST | 243.585 | 77.353 | x3.244 | **x3.149** |
| x86 nq=1 MT | 2.460 | 1.063 | x2.381 | **x2.315** |
| x86 nq=1 ST | 9.519 | 3.627 | x2.764 | **x2.624** |

**Harmonic mean x2.0477.** The recorded x2.1001 was optimistic by 2.5%.

Nothing regressed — the code is identical to what those numbers were taken on.
Seven of eight cells came in at or below their recorded value, which is the
signature of an accumulated measurement bias rather than of noise: noise moves
cells in both directions. The recorded figure was assembled from readings taken
at different times rather than from one paired, alternating, same-session A/B,
and each such reading had a free chance to catch its arm in a favourable
machine state.

**x2.0477 is the defensible number and the log now uses it.** The three
consistently-worst offenders were `arm nq=1 MT` (-4.5%), `x86 nq=1 ST` (-5.1%)
and `x86 nq=100 ST` (-2.9%).

This does not change any verdict in this log: every hypothesis here was judged
on a paired A/B of its own, and a shifted baseline moves both arms of a pair
equally. What it changes is the headline, and the derived quantities that hang
off it — the ceiling from P43 is unmoved at x3.08, so the climb is at **66% of
what is reachable**, not 68%.

**Standing rule from here: the 8-cell figure is re-derived, never inherited.**
Any future statement of the score cites a run in which both arms were built and
measured in the same session.

## H109 — hugepages are already there; no TLB win exists

The code array is 76.8 MB, which is 18,750 pages at 4 KB, and every one of the
eight cells walks all of it. That is the classic shape of a TLB-bound scan and
nothing in this log had checked it.

`transparent_hugepage/enabled` is `[always]` on both boxes, but that alone
proves nothing: a 76.8 MB allocation only gets 2 MB pages where the mapping is
2 MB-aligned, and Rust's allocator makes no such promise. So the question is
what the process actually holds, measured from `smaps_rollup` after loading the
index and running a search.

| box | AnonHugePages | code array | covered |
|---|---|---|---|
| ARM | 73,728 kB | 76,800 kB | **96%** |
| x86 | 75,776 kB | 76,800 kB | **99%** |

**Already done, by the kernel, without anyone asking.** The 1-4% shortfall is
the unaligned head and tail of the mapping; aligning the allocation to 2 MB
would recover TLB coverage for about 3 MB of a 76.8 MB array, which is not a
measurable effect.

Recorded because "have you tried hugepages" is a perennial suggestion for
scans this size, and it is now answered with a measurement rather than a guess.

Cost: two commands, no build.

## H110 — 5% is sitting in x86 code compiled at the v2 baseline

`.cargo/config.toml` pins all x86 non-kernel code to `x86-64-v2` on purpose:
the dispatch prologue runs *before* `is_x86_feature_detected!` can choose a
kernel, so a higher baseline SIGILLs on old CPUs (#137). The SIMD kernels are
`#[target_feature]`-gated and unaffected. Nothing had measured what that
baseline costs the code around them.

Built `target-cpu=native` on both boxes, parity md5 identical in every arm:

| | HEAD | NATIVE | |
|---|---|---|---|
| x86 nq=100 MT | 18.003 | 17.185 | **x1.048** |
| x86 nq=100 ST | 76.49 | 70.57 | **x1.084** |
| x86 nq=1 ST | 3.669 | 3.499 | x1.049 |
| x86 nq=1 MT | 1.066 | 1.051 | x1.015 |
| ARM nq=100 MT | 12.723 | 13.305 | x0.956 |
| ARM nq=100 ST | 98.80 | 103.19 | x0.958 |
| ARM nq=1 ST | 3.728 | 3.814 | x0.977 |

**x86 gains up to 8.4%; ARM loses 4%.** The ARM half is a finding in itself —
the default aarch64 codegen target beats `native` on Neoverse V2, so nothing
should be changed there.

### The x86 win is AVX-512 reaching plain code, not tuning

`native` changes both the feature set and LLVM's scheduling model, and only one
of those has a portable fix. `-C tune-cpu` would separate them directly but is
nightly-only, so the levels were used instead:

| build | nq=100 MT | nq=100 ST |
|---|---|---|
| HEAD (v2) | 18.06 | 75.33 |
| v3 (AVX2) | 18.04 | 76.00 |
| **v4 (AVX-512)** | **17.26** | **71.31** |
| native | 17.17 | 70.70 |

**The entire effect is the v3 -> v4 step, and v4 matches native.** AVX2 buys
nothing; AVX-512 buys all of it; the scheduling model buys nothing measurable.
So this is a feature-availability effect on code *outside* the gated kernels,
which is exactly the class of thing `#[target_feature]` can fix portably —
unlike tuning, which cannot be expressed on stable at all.

### Why raising the baseline is not the answer, and what is

Shipping v4 is not available: it is the thing #137 forbids, and it would make
turbovec SIGILL on every pre-Skylake-X CPU before dispatch runs. The portable
route is to find which non-kernel code is hot enough to matter and give it an
AVX-512 variant behind the existing runtime dispatch.

The suspect is the per-block epilogue. `avx2_post_flush_heap_update` is gated
`avx2 + fma` and runs once per 32-vector block for every query in the batch —
it converts 32 int32 accumulators to f32, applies scale and bias, prunes
against the heap threshold and updates. P44 established that *selection* is
free at k=10, but selection is only the tail of that function; the convert,
scale and prune run over all 32 lanes unconditionally, and at 512 bits they
would run at half the instruction count.

**H111: give the post-flush epilogue an AVX-512 variant.** It is reached only
from the AVX-512 kernel, which already declares the features, so the dispatch
already exists and no new runtime check is needed. Expected value is a
fraction of the 5.3% measured at nq=100 ST — the first shippable candidate in
fourteen hypotheses.

## H111 — CONFIRMED +1.41%: the v2 baseline's cost was all in the epilogue

H110 measured 5.3% of x86 nq=100 ST sitting in code compiled for `x86-64-v2`,
and localised it to the v3 -> v4 step — feature availability, not scheduling.
The baseline cannot move (#137). A `#[target_feature]` variant can.

The per-block epilogue was the suspect: it runs once per block per query,
converts 32 int32 accumulators to f32, applies scale and bias, and prunes
against the heap threshold. At nq=100 over 200k vectors almost every block
beats nothing, so the path that matters is the early exit — four multiplies,
four compares and four movemasks at 256 bits, plus the caller's four converts,
four multiplies, four adds and four extracts.

At 512 bits that is two multiplies, two compares and one mask test, with the
caller handing over two `__m512` built by two converts and two FMAs.
Selection is untouched: P44 priced that at free for k=10, and this changes
only the arithmetic that runs over all 32 lanes regardless.

The non-full-block and heap-filling paths fall through to the AVX2 routine
rather than being duplicated — they run once per scan or once per index, so a
second copy would be all risk and no gain.

### Result

133 tests pass. Identical id md5 and recall@10 = 0.8030 in both arms. Soak of
four alternating rounds at `--reps 15`, control built from the same tree in
the same session:

| x86 cell | HEAD | H111 | |
|---|---|---|---|
| nq=100 MT | 18.056 | 17.134 | **x1.054** |
| nq=100 ST | 76.17 | 70.79 | **x1.076** |
| nq=1 MT | 1.061 | 1.052 | x1.008 |
| nq=1 ST | 3.519 | 3.554 | x0.990 |

nq=1 is neutral either way — one batch, so the epilogue runs 8x less often
relative to the scan — and the smoke read it at x1.042 where the soak reads
x0.990, which is the spread of that cell rather than an effect.

**This reaches v4's numbers (17.26 / 71.31) at the v2 baseline**, so the whole
5.3% H110 found was this one function. Nothing else in the non-kernel code was
costing anything measurable.

### Score

Against H108's re-derived baselines, x86 goes x3.434 -> x3.624 MT and
x3.149 -> x3.441 ST. The reciprocal sum falls 3.9069 -> 3.8526 and the
harmonic mean moves **x2.0477 -> x2.0765, +1.41%** — clear of the 1% gate.

ARM is untouched by construction: the change is inside `#[cfg(target_arch =
"x86_64")]` and reached only from the AVX-512 kernels.

**Ships.** The streak resets.

## H112 — naming the actual ARM core makes it slower, monotonically

H111's trick does not port: it moved the epilogue 256 -> 512 bits, and aarch64
has no wider unit to move to. NEON is 128-bit and SVE on Neoverse V2 is also
128-bit, so the width H111 exploited simply does not exist here. Closed by
architecture, not by measurement.

What did need measuring is H110's leftover anomaly. `native` resolves to
`neoverse-v2` on that box — the correct chip — and was **4% worse** than the
generic aarch64 default. A vendor model losing to generic is unusual enough
not to accept on one reading, so three were swept, with the default rebuilt
from the same tree in the same session as its own control.

| ARM | nq=100 MT | nq=100 ST | nq=1 MT | nq=1 ST |
|---|---|---|---|---|
| **default (generic)** | **12.564** | **98.69** | **0.584** | **3.725** |
| `neoverse-n2` | 12.905 | 101.46 | 0.633 | 3.799 |
| `neoverse-v1` | 13.294 | 102.33 | 0.616 | 3.889 |
| `native` (= `neoverse-v2`) | 13.333 | 103.59 | 0.601 | 3.927 |

Identical recall in every arm. **The default wins all four cells, and the
ordering is monotone in how closely the model matches the hardware** — the
scheduler that knows the most about this core produces the slowest code on
the two cells that matter most.

**That is the opposite of H110's x86 result and the explanation is the same
one.** On x86 the gap was *features*: AVX-512 became available to code that
had none, which is strictly more capability. Here there is no feature to gain
— every model targets the same 128-bit NEON — so all a vendor model can do is
reorder. And the inner loops it reorders are hand-scheduled intrinsics whose
instruction order and register allocation were tuned by measurement: H94 set
`qbs = 12` at the register-file boundary, H97 confirmed 8 on x86 by sweeping
both directions, H103 fixed the range count. **LLVM's core model reorders that
work and makes it worse, and the more confident the model, the more damage it
does.**

Two things follow. Nothing should ever add `target-cpu` to the aarch64 build,
and this is now a measured prohibition rather than an omission. And the
hand-tuning recorded across H94/H97/H98 is validated from an unexpected
direction: an automatic scheduler with full knowledge of the pipeline cannot
match it.

No change ships. The x86 build keeps its `x86-64-v2` baseline (#137) and gets
its width from H111's `#[target_feature]` variant instead.

## H113 — the ARM epilogue's memory round-trip is real and unremovable

H111 won by finding a shared helper that had been left behind, so the ARM
epilogue got the same audit. It is not structurally behind: `neon_block_topk_
update` already has the whole-block prune, a max-reduce over the 32 lanes with
an early return, which is the same trick the AVX2 path uses.

There is one real asymmetry. **x86 receives the block's scores as registers
and only stores to memory when a lane passes the threshold. ARM writes all 32
floats to `block_out` unconditionally, then loads all 8 vectors back to compute
the prune max.** In the common case — the heap is warm and the block beats
nothing — those 32 stores and 32 loads are pure waste, exactly the kind H111
removed.

**It cannot be fixed the same way.** The ARM scores are not available in
registers at one moment: `score_block_permute_smmla_neon` assembles them across
the `part` loop, four passes each writing 8 floats, and that loop exists
because `acc` is already `[[i32; 4]; 6]` — 24 of 32 registers. Holding a
block's 8 score vectors live for 12 queries needs registers H94 measured as
absent, and the `part` loop is precisely the structure that copes with their
absence. The round-trip is the price of the register blocking, not an
oversight.

The reachable residue is smaller: the kernel *does* hold each vector as it
stores it, so it could accumulate the block max there (7 `vmaxq` + 1 `vmaxvq`
over values already live) and hand the epilogue a single f32, letting the
pruned path skip its 8 loads entirely. That removes 8 loads per (block, query)
— about 5M loads per nq=100 search, near 1.7 ms against a 98 ms cell, so
**~1.7% of one cell and roughly +0.5% on the harmonic mean across both ARM
nq=100 cells.** Under the gate on its own, and it moves a max computation into
the hot loop to save loads in a helper, which is the kind of trade that has
measured worse than its arithmetic three times in this log already (H99, H106,
H107).

Recorded as reasoned-null rather than built: the mechanism is understood, the
size is below the threshold, and the honest expectation after H107 is that the
kernel would not show even that.

## H114 — the metric is noisier than the gate it is judged against

H111's +1.41% was *composed*: H108's `main` baselines combined with a fresh
x86 measurement. That is the inheritance H108's own standing rule forbids, one
entry after making it. Re-derived properly — `main` and head rebuilt from
source in-session on both boxes, three alternating rounds each:

| cell | main | now | speedup | H108 |
|---|---|---|---|---|
| arm nq=100 MT | 42.179 | 12.519 | x3.369 | x3.322 |
| arm nq=100 ST | 317.317 | 98.586 | x3.219 | x3.215 |
| **arm nq=1 MT** | 0.653 | 0.678 | **x0.963** | x1.041 |
| arm nq=1 ST | 4.150 | 3.713 | x1.118 | x1.096 |
| x86 nq=100 MT | 61.929 | 17.059 | **x3.630** | x3.434 |
| x86 nq=100 ST | 242.269 | 69.991 | **x3.461** | x3.149 |
| x86 nq=1 MT | 2.456 | 1.053 | x2.333 | x2.315 |
| x86 nq=1 ST | 9.477 | 3.479 | x2.724 | x2.624 |

**Harmonic mean x2.0509**, against x2.0477 before H111 — the +1.41% did not
appear.

**H111 is not in doubt.** Its two target cells moved exactly as measured:
x86 nq=100 MT x3.434 -> x3.630 and ST x3.149 -> x3.461, which is the 5-8% the
soak found, arrived at independently with a rebuilt baseline. What swallowed it
is `arm nq=1 MT`, which read x1.041 in H108 and x0.963 here — **on identical
code, since H111 is inside `#[cfg(target_arch = "x86_64")]` and cannot touch
ARM.** Both arms of that cell moved: main 0.614 -> 0.653, head 0.589 -> 0.678.

### The instrument, priced

H102 already found this cell is the noisiest on the board — 76.4% and 90.8%
scaling efficiency in consecutive sweeps of one build — because the whole
search is 0.5 ms across 8 threads and one descheduled worker puts an entire
range on the critical path. What H114 adds is what that costs the *metric*.

`arm nq=1 MT` carries the largest reciprocal weight of any cell. Holding the
other seven fixed and substituting its two observed values:

| `arm nq=1 MT` | harmonic mean |
|---|---|
| x0.963 (this run) | **x2.051** |
| x1.041 (H108) | **x2.093** |

**One cell's run-to-run noise moves the headline figure by 2%, against a gate
of 1%.** The metric cannot resolve the improvement it is asked to certify. That
is not a reason to distrust H111 — a paired same-session A/B on the cells a
change touches is a far sharper instrument than the 8-cell mean, and that is
what H111 passed. It is a reason to stop quoting the 8-cell figure to three
decimal places and to judge future changes primarily on their own cells.

**Standing correction to the method:** an improvement is certified by a soaked
paired A/B on the cells it can causally reach, with the untouched cells read as
a control channel (H101). The 8-cell mean is reported alongside, with its
uncertainty stated, and is not the arbiter for anything under ~2%.

The honest current figure is **x2.05 +/- 0.04**, and H111's contribution is
best stated where it is measurable: **+5.7% and +9.9% on the two x86 nq=100
cells.**

## H115 — the noisy cell is samplable; the harness was under-sampling it

H114 showed the metric cannot resolve its own 1% gate because `arm nq=1 MT`
swings ~8% run to run and carries the largest reciprocal weight. Before
down-weighting the cell or carrying its uncertainty forever, the question is
whether it is *samplable*: does a deeper floor converge, or is the machine
genuinely delivering different performance run to run?

`cells.py` already took min-of-3 rather than more reps, because H51 found extra
iterations do not help — the whole *process* runs slow, so the noise lives
between runs, not between iterations. It was the right shape and too shallow.

27 sub-runs, regrouped:

| sampling | n | min | max | spread |
|---|---|---|---|---|
| single sub-run | 27 | 0.549 | 0.641 | **16.8%** |
| min-of-3 (previous) | 9 | 0.549 | 0.591 | 7.8% |
| **min-of-9** | 3 | 0.549 | 0.563 | **2.7%** |

**It converges, and the floor is 0.549 ms at every depth.** That is the
signature of a real value being approached from above by a distribution with a
one-sided tail — descheduled runs can only make a 0.5 ms search slower, never
faster — rather than of a machine with two performance states. Min is the right
estimator here precisely because the contamination is one-sided.

Harness changed to min-of-9 for both nq=1 cells. Cost is a few seconds per
measurement; the return is that the cell's contribution to headline noise drops
about 3x, taking the metric's resolution from ~2% to under the 1% gate it is
supposed to enforce.

**This retires the caveat H114 had to attach.** Future 8-cell figures are
quotable at the gate's precision. It does not retrospectively fix H108 or H114
— those were measured with min-of-3 and keep their stated uncertainty — and the
method correction from H114 stands regardless: a change is certified on a
soaked paired A/B of the cells it can reach, with untouched cells read as a
control channel.

Not a speedup. The instrument was the binding constraint on being able to
certify one, which is the thing the previous four entries kept running into.

## H116 — the missing ARM single-query prune costs nothing, because that cell is memory-bound

A real gap in the code, found while auditing for H113: the batched ARM path
prunes a whole block with a max-reduce before touching lanes
(`neon_block_topk_update`), but **`scan_range_neon` — the nq=1 path — has no
prune at all** and falls straight into a 32-iteration scalar loop with an
unpredictable branch per lane, for every block including the overwhelming
majority that beat nothing. 6250 blocks x 32 lanes is 200k branchy iterations
per search, and the arithmetic said ~3% on both ARM nq=1 cells, which carry the
two largest reciprocal weights — about +1.4% on the metric.

Added the prune: eight loads, seven `vmaxq`, one `vmaxvq`, guarded on a full
heap and a full block. 127 tests pass, identical id md5 and recall.

| ARM cell | control | H116 | |
|---|---|---|---|
| nq=1 ST | 3.654 | 3.620 | x1.009 |
| nq=1 MT | 0.540 | 0.547 | x0.987 |
| nq=100 MT | 12.556 | 12.624 | x0.995 |
| nq=100 ST | 98.85 | 99.39 | x0.995 |

**Neutral.** The gap is real, the fix is correct, and it buys nothing.

The reason is the one P42 established and this log keeps relearning: ARM nq=1
runs at 95% of the single-core streaming roofline. The scalar lane loop
executes *inside* memory latency that is already being paid, so deleting its
work does not shorten the block. The 3% estimate counted instructions on a
cell whose instructions are free.

**Fourth time in this log** that removing work from a bound loop returned
nothing — H99 (tile feed), H107 (the ARM TBL, which predicted 33% and gave 0),
H113 (predicted and so not built), and now this. The pattern is sharp enough
to state as a rule: *on a cell measured at its roofline, an instruction-count
argument is not evidence.* Only cells with slack — which after H110 means the
x86 nq=100 pair — can convert removed work into time.

Reverted; the omission is now documented in the code as measured rather than
overlooked, so the next reader does not re-derive the same 3% and rebuild it.

## H117 — x2.1110 on the sharp instrument, and H111 confirmed three times over

The min-of-9 harness from H115, applied to a full re-derivation: `main` and
head rebuilt from source in-session on both boxes, three alternating rounds.

| cell | main | now | speedup |
|---|---|---|---|
| arm nq=100 MT | 42.112 | 12.730 | x3.308 |
| arm nq=100 ST | 317.562 | 98.888 | x3.211 |
| arm nq=1 MT | 0.608 | 0.563 | x1.081 |
| arm nq=1 ST | 4.056 | 3.646 | x1.112 |
| x86 nq=100 MT | 61.956 | 17.041 | x3.636 |
| x86 nq=100 ST | 240.794 | 70.504 | x3.415 |
| x86 nq=1 MT | 2.439 | 1.033 | x2.362 |
| x86 nq=1 ST | 9.366 | 3.379 | x2.772 |

**Harmonic mean x2.1110.**

### What is and is not comparable

H115 changed the estimator for the nq=1 cells only — nq=100 still takes a
median of reps, untouched. So the **nq=100 columns are directly comparable
across all three re-derivations**, and they are the ones that carry H111:

| cell | H108 (pre-H111) | H114 | H117 |
|---|---|---|---|
| x86 nq=100 MT | x3.434 | x3.630 | **x3.636** |
| x86 nq=100 ST | x3.149 | x3.461 | **x3.415** |
| arm nq=100 MT | x3.322 | x3.369 | x3.308 |
| arm nq=100 ST | x3.215 | x3.219 | x3.211 |

**H111 is confirmed by three independent re-derivations**: +5.9% MT and +7.7%
ST on the two x86 cells, against ARM flat to within 2% across the same runs —
which is the control channel behaving exactly as it must, since the change is
inside `#[cfg(target_arch = "x86_64")]`.

The nq=1 speedups all rose slightly under min-of-9 (arm MT x1.041 -> x1.081,
x86 ST x2.724 -> x2.772). Deeper sampling lowers both arms, so a *ratio* should
have been stable; that it drifted up says the faster arm gains marginally more
from extra chances at a clean run. Small, one-directional, and now part of the
instrument's definition rather than a mystery.

**x2.1110 is therefore not "x2.05 plus H111".** It is the same code measured
with a better estimator, and the two figures are on different instruments. The
comparable claim is the narrow one: H111 is worth +5.9% and +7.7% on the cells
it touches, and the current head reads x2.1110 on the sharpest measurement this
log has taken.

The log's headline is updated to **x2.1110**, with the note that anything
measured before H115 is on the blunt instrument and should not be differenced
against it.

## H118 — no build-level slack remains; H110's seam is fully mined

H110's win came from auditing the *build* rather than the kernel, so the rule
says look there again before looking anywhere else. The obvious remaining
knobs turn out to be already set:

```
[profile.release]
lto = true            # fat LTO, not thin
codegen-units = 1     # not the default 16
opt-level = 3
```

That is the configuration those flags have when someone has already thought
about them. Nothing to gain.

**And the seam is provably exhausted, not merely inspected.** H110 measured
the ceiling directly: `target-cpu=v4` gave 17.26 / 71.31 on the two x86
nq=100 cells, and H111 reaches 17.13 / 70.79 *at the v2 baseline*. The
portable build now matches what unrestricted codegen produces, so there is no
remaining gap between what the compiler is allowed to emit and what it would
emit given every instruction on the machine. H112 showed the same for ARM from
the other direction — every vendor scheduling model is worse than the default,
so the ARM build is at its optimum too.

Two knobs deliberately not tried. `panic = "abort"` would remove landing pads,
but the crate is a pyo3 extension and panics must be caught at the FFI
boundary — a correctness change dressed as a perf one. `prefer-256-bit` does
not apply: the kernels use explicit 512-bit intrinsics, which the flag does not
govern.

Binding overhead is also already accounted for. `cells.py` times through the
Python call, so pyo3 and numpy conversion sit inside P43's fitted intercept —
which came out at zero for x86 nq=1 and 2% for ARM. There is nothing hiding in
the wrapper.

**The build-level seam that produced this session's only win is closed.** What
remains is what the eight cells' measured constraints allow: nothing on nq=1
(memory-bound, four refusals), nothing on ARM nq=100 (issue-limited, unpack
free), and on x86 nq=100 only the lookup-free scan — worth ~10% of those cells
but gated behind the uniform codebook, which H107 priced at +0.84% overall and
which costs 0.021 recall that is not being traded.

## H119 (next) — free the register H94 said was needed, by re-blocking `part`

H94 swept the ARM query batch and found 16 spills, concluding that "widening
ARM needs a register *freed*". That sentence was left as a closing remark. It
is actually a design.

`score_block_permute_smmla_neon` blocks the 32 output lanes into
`for part in 0..4`, holding `acc = [[int32x4_t; 4]; NP]`. At `qbs = 12`,
`NP = 6`, so that is **24 of 32 vector registers**, plus `a[6]` for the query
operand, plus the level table and masks — which is exactly why H94's `qbs = 16`
spilled: `NP = 8` makes `acc` 32 registers on its own.

**But `part` and `qbs` have only ever been swept independently.** The `part`
loop is the lane-blocking factor, and it sets how wide `acc` is per pair:

| `part` | `acc` shape at `qbs=16` | acc registers | + `a[8]` | total |
|---|---|---|---|---|
| 4 (current) | `[[i32x4; 4]; 8]` | 32 | 8 | **40 — spills** |
| **8** | `[[i32x4; 2]; 8]` | **16** | 8 | **24 — fits** |

Doubling `part` halves the accumulator width per pair, which frees exactly the
registers H94 identified as the blocker. `qbs = 16` then becomes reachable, and
a wider batch amortizes the shared unpack — the same mechanism H28 measured
going 4 -> 8 on x86 and H97 confirmed by watching 4 under-amortize.

The cost is real and must be measured, not assumed: `part = 8` doubles the
A-operand reloads (6 per `(part, q4)` becomes the same 6 over twice as many
iterations) and doubles the outer-loop bookkeeping. So this is a trade of more
loads against fewer unpacks, and H107 is the warning — that entry predicted 33%
from an instruction count on this exact kernel and measured zero, because the
loads the loop already runs absorb the slack.

**Which is why it is worth building rather than reasoning about.** ARM nq=100
is issue-limited at ~80% of 4 instructions/cycle (H107's accounting), so it is
one of the two cells this log has shown *can* convert removed work into time.
The unpack at `qbs = 16` runs 16/12 = 1.33x fewer times per query.

Method: implement `part = 8` with the narrower `acc`, sweep `qbs` at 12 and 16
against a same-session control, all four ARM cells, `load_parity.py` first.
`qbs = 12` at `part = 8` is the control that isolates the re-blocking cost from
the widening benefit — if that alone regresses, the extra loads dominate and
the idea dies without needing the 16 arm at all.

### Result: already implemented, and it corrects H94's stated mechanism

The arithmetic above was computed against `score_block_permute_smmla_neon`.
**That is not the kernel that runs.** The shipped 4-bit ARM batched path uses
`score_block_smmla_vm8`, and it already reads:

```rust
for part in 0..8 {
    let mut acc = [[vdupq_n_s32(0); 2]; NP];
```

`part = 8` with the narrow two-wide accumulator — precisely the re-blocking
H119 proposed, present all along in the kernel the goal's cells exercise. The
`part = 4`, four-wide version is the fallback for indices not in the vm8
layout. So there is nothing to build.

**And that invalidates H94's explanation, though not its measurement.** H94
found `qbs = 16` 6% worse at nq=100 MT and attributed it to register spill:
`NP = 8` making `acc` 32 registers. In the vm8 kernel `acc` is
`[[int32x4_t; 2]; NP]`, so at `NP = 8` it is **16 registers, not 32**, plus
`a[8]` for 24 — comfortably inside the file. **`qbs = 16` does not spill the
accumulators, and whatever makes it 6% slower is something else**: more likely
the `a[]` operand reloads, the wider LUT working set, or the tail effect of
100 queries dividing badly by 16 (seven batches, the last holding four).

H97 recorded ARM's width as "a register-file boundary, not a tuned constant",
and this log has repeated that phrasing since. **That claim is now withdrawn.**
The boundary is real as a measurement and unexplained as a mechanism.

Two things follow. The obvious next experiment is `qbs = 16` re-measured with
the tail hypothesis controlled — nq = 96 and nq = 112 alongside nq = 100 — since
a batching-remainder effect and a microarchitectural one look identical at a
single query count. And the wider lesson: **H119 was formed by reading a
function that is not on the hot path**, which is the same class of error as
this session's four denominator mismatches — reasoning carefully about the
wrong object. The check that would have caught it costs one grep.

## Loop state

Streak 10 — H71, H72, H73, H75, H76, H77, H78, H79, H80 (null/open) and
H74 (refuted) since H70 landed. P37/P38 are probes, no streak effect. (+3.7% x86 nq=100 MT), after H69
(+3.3% arm nq=100 MT). Before it: H68 (null) and
H67 (+8.3% on arm nq=100 ST). Before it: H66 (null) and
H65 (BLK 4 -> 8 on x86) (BLK 4 -> 8 on x86, all four cells
improve within-run). Before it: H63 (null) and H64 (refuted, x0.57), and H62, which
took the 8-cell harmonic mean past x2 for the
first time (x1.985 -> x2.041). Before it: H60 (null), H61 (refuted), and
H59, which took the 8-cell harmonic mean from x1.935 to
x1.985 by re-testing the prefetch H43 had refuted. Before it: H56, H57,
H58 (null), H55 (blocked by the register file), and H54, which took the 8-cell harmonic mean from x1.851 to
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
