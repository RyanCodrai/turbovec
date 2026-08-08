# 2-bit search hill-climb — results log

Goal in `GOAL_2bit.md`. Objective: HM of 8 cells, `{arm, x86} x {ST, MT} x
{nq=1, nq=100}` at `bit_width=2`, N=200k, dim=768, k=10.

Harness: `cells_2bit.py` (objective, `--bits 4` for the observation run),
`sweep_2bit.py` (nq and N gates), `parity_2bit.py` (digests), `whm_2bit.py`
(scorer and verdict).

**Baseline: not yet pinned.** No SSH to `turbovec-bench-search` /
`turbovec-bench-arm-search` — the instances carry no `ssh-keys` metadata and
OS Login is not enabled, so no measurement in this log is a rig measurement
yet. Everything below is code study and predictions made before measuring,
which is the point: they are refutable.

## S1 — the two arches do not agree on the 2-bit layout

`pack::vector_major_for` (pack.rs:1278):

```rust
let kernel_exists = cfg!(target_arch = "x86_64") || bits == 4;
kernel_exists && use_vector_major() && n_byte_groups % 4 == 0
```

At dim=768, `n_byte_groups` is a multiple of 4 at both widths, so the only
term that moves is `kernel_exists`:

| | layout | kernel |
|---|---|---|
| x86, 4-bit | vector-major | permute-dot (`vm && bits == 4`, search.rs:2721) |
| x86, **2-bit** | **vector-major** | classic (search.rs:2718) |
| arm, 4-bit | vector-major | permute-dot / vm8 |
| arm, **2-bit** | **sequential** | classic (`score_4bit_block_neon`, which despite its name is the both-widths TBL kernel; reached because `lut.pd` is `None`, search.rs:3084) |

So at 2 bits x86 keeps the vector-major layout while arm falls back to
`pack_blocked_sequential`. That asymmetry was never chosen for 2 bits — it
falls out of a condition written to gate the *4-bit* permute-dot kernel. One
of the two arches is on the wrong layout for its classic kernel, and which
one is an empirical question nobody has asked.

**This is the first thing to measure, not the first thing to fix.**

## Hypotheses queued (unmeasured — rig blocked)

### H1 — arm 2-bit on the vector-major layout

Mechanism: the classic NEON kernel reads code bytes through `vm_byte_index`
strides that x86's classic kernel already tolerates on the same layout. If
the layout is neutral-to-better for a classic kernel, arm is paying a
gratuitous packing difference. One-line change to `kernel_exists`.
Refuted by: no improvement at nq=100 arm, where layout effects are largest.
Label: shared-path (touches `vector_major_for`; reconciles as a bits split).

### H2 — x86 2-bit off the vector-major layout

The mirror of H1, and both cannot win. The vector-major layout exists to feed
the permute-dot kernel, which is not built at 2 bits, so x86 may be paying a
repack whose consumer is absent. Refuted by: no improvement at nq=100 x86.

### H3 — a 2-bit permute-dot — REFUTED by arithmetic (non-win 1/20)

Priced before building, as the hypothesis said it should be. The two kernels
scale differently in `bits`, and that alone settles it.

**Classic** (`score_4query_block_neon`, search.rs:1652). Per byte-group it
loads the codes once and splits nibbles once, then per *query* does
4 `TBL` + 2 `ADD` + 4 `VADDW` = 10 instructions. A byte-group is 32 vectors x
one byte, and a byte is 2 dimensions at 4 bits but **4 dimensions at 2 bits**.
So per query, cost per (vector.dimension) is `10/64` at 4 bits and `10/128` at
2 bits — the classic kernel gets **2x cheaper per unit work** purely from the
width change, before any optimisation.

**Permute-dot.** Its arithmetic is the dot product itself: one i8
multiply-accumulate lane per (dimension x query), which is *independent of
code width*. Narrowing 4 bits to 2 removes none of it. The unpack does change,
and against it: a byte carries four 2-bit fields instead of two nibbles, so
expanding to i8 levels needs 4 `TBL` + 4 `AND` + 3 `SHR` against 2 `TBL` +
1 `AND` + 1 `SHR` — about 2.75 ops/dim against 2. That unpack is shared across
queries, which is the family's whole advantage, but it is the smaller term.

So going 4 -> 2 bits, the classic kernel's per-query cost halves and
permute-dot's does not move. Permute-dot won at 4 bits by roughly x1.1-2.0
depending on cell; a 2x swing in the baseline it has to beat consumes that
margin entirely. The comment at search.rs:2511 reaches the right conclusion
by the wrong argument — the obstacle is not the unshared level map, it is
that the dot product does not get cheaper when the codes do.

**Corollary, and the reason this refutation is worth more than a non-win:**
the same scaling says the 4-bit climb's headline wins are *structurally*
unavailable at 2 bits. The 2-bit climb is not a re-run of #485 at a different
width, and hypotheses ported from it should be assumed dead until argued
otherwise. H1/H2 — layout, not kernel — remain the live pair.

Not refuted for `mask`/allowlist-heavy shapes, which are outside this goal's
cells, and not refuted at dim=1536 where the unpack amortizes differently.
Both are out of scope here; recorded so the boundary of this refutation is
explicit.

## Probes queued

### P1 — does 2-bit inherit 4-bit's memory-bound verdict?

P42 in `LOG_search.md` put arm nq=1 at 95% of the single-core streaming
roofline at 4 bits. At 2 bits the code array is 38.4 MB against 76.8 MB. If
the cell is still bandwidth-bound, time should track the byte ratio; the
interesting outcome is the one where it does **not**, because the leftover is
compute headroom that 4 bits does not have and every hypothesis above is
competing for it.

Measure: ns/(query·vector) at 2 and 4 bits, all four cells, both arches.
