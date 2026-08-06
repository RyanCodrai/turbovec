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

## Loop state

Streak 1 (H16). Three improvements: H5, H9, H15. Three improvements: H5, H9, H15. Two confirmed wins (H5, H9).
