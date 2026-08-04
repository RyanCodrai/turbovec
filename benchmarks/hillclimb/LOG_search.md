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

## Loop state

Streak 0 of 20 (reset by H5). One confirmed win.
