# 2-bit search hill-climb — results log

Goal in `GOAL_2bit.md`. Objective: HM of 8 cells, `{arm, x86} x {ST, MT} x
{nq=1, nq=100}` at `bit_width=2`, N=200k, dim=768, k=10.

Harness: `cells_2bit.py` (objective, `--bits 4` for the observation run),
`sweep_2bit.py` (nq and N gates), `parity_2bit.py` (digests), `whm_2bit.py`
(scorer and verdict).

Rig: `turbovec-bench-arm-search` (c4a, Axion) and `turbovec-bench-search`
(c3, Sapphire Rapids). Reach them with `~/.ssh/gce_ed25519_tvbench` as user
`ryan` — gcloud's default `google_compute_engine` key is not registered on
them and fails with `Permission denied (publickey)`, which cost this climb
several hours of misdiagnosis. `~/.ssh/config` has `tvarm` / `tvx86` aliases.

## Resolved — the SSH blocker was the wrong key, not the project

Recorded because the wrong diagnosis was confident and detailed, and someone
will hit this again.

All four running instances (`turbovec-bench`, `turbovec-bench-arm-pmu`,
`turbovec-bench-search`, `turbovec-bench-arm-search`) refuse SSH identically
with `Permission denied (publickey)`. Established:

- OS Login is enforced on the instances (`google-oslogin-cache.service` is
  running); no `ssh-keys` metadata exists on any of them.
- The active account `ryan@docdojo.ai` holds `roles/owner` and
  `roles/compute.osAdminLogin`; its OS Login profile has posix username
  `ryan_docdojo_ai` and both keys registered (RSA `f1b4z…`, ed25519 `3M0BE…`).
- Failing combinations tried: `gcloud compute ssh` as default user, as
  `ryan_codrai_gmail_com`, as `ryan_docdojo_ai`; direct `ssh` with each
  registered key; `PubkeyAcceptedAlgorithms=+ssh-rsa`; IAP tunnel. Verbose ssh
  reports `Server accepts key` and *then* denies — authorization fails after
  the key matches.
- The serial console shows `google_guest_agent` failing with
  `IAM_PERMISSION_DENIED` on `logging.logEntries.create` for
  `475585223631-compute@developer.gserviceaccount.com`. A compute service
  account that has lost permissions would also break OS Login's
  `AuthorizedKeysCommand` lookup, which matches the project-wide symptom.

None of that was the cause. The boxes were reachable the whole time with
`~/.ssh/gce_ed25519_tvbench`, the dedicated bench key earlier sessions used.
The guest agent's IAM warning is real and unrelated; OS Login being enabled is
real and irrelevant once the right key is offered. **Check `~/.ssh/` for an
existing per-rig key before theorising about infrastructure.**

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

## Hypotheses

### H1 — arm 2-bit on the vector-major layout — REFUTED (non-win 2/20)

Not a one-liner in the end: the classic NEON kernel had to learn the layout.
`vm_byte_index` is `(g/4)*128 + (lane/16)*64 + (lane%16)*4 + (g%4)`, which is
a stride-4 interleave of four byte-groups — exactly what `LD4` undoes. Added
`vm_load_quad` (one `vld4q_u8` per 64-byte half, four registers out, register
`g%4` being that group's 16 lanes), made both NEON kernels generic over a
`const VM: bool`, and carried the flag on `QueryNeonLut` because `bits` is not
in scope in the scan helpers. `cargo test -p turbovec` green on aarch64 (194
tests), parity digests bit-identical to baseline on both widths — the LD4 path
is correct.

It is also slower, everywhere:

| cell | base | H1 | speedup |
|---|---|---|---|
| nq1_st | 1.933 | 2.904 | x0.666 |
| nq1_mt | 0.302 | 0.410 | x0.736 |
| nq100_st | 148.770 | 179.352 | x0.829 |
| nq100_mt | 18.441 | 22.885 | x0.806 |

**S1's premise is refuted, not confirmed.** The asymmetry looked like an
accident of a condition written for permute-dot; it is not. Each arch is on
the layout its *classic* kernel prefers. x86's reads four byte-groups per
`vpermb` and wants them interleaved (H2: x2.5 at nq=1). aarch64's reads one
group per pair of `vld1q_u8` and wants them contiguous — paying `LD4` to
rebuild that costs more than the locality returns.

Two refutations, opposite arches, same experiment: the layout question is
**closed**. What remains is the kernel question — P1's finding that 2 bits
loses to 4 bits at nq=100 — which is H3.

### H2 — x86 2-bit off the vector-major layout — REFUTED, decisively (non-win 1/20)

One line: `kernel_exists = bits == 4`, so 2-bit x86 falls back to the perm0
layout its classic kernel also reads. Parity digests identical to baseline on
both widths, as a pure layout change should be. Medians of three rounds, x86:

| cell | base | H2 | speedup |
|---|---|---|---|
| nq1_st | 1.669 | 4.581 | **x0.364** |
| nq1_mt | 0.502 | 1.261 | **x0.398** |
| nq100_st | 83.958 | 126.703 | **x0.663** |
| nq100_mt | 26.026 | 32.014 | **x0.813** |

Not marginal — the layout is worth **2.5x at nq=1** to the classic x86 kernel,
with no permute-dot anywhere in the picture. The premise was that
vector-major exists only to feed permute-dot; it is wrong. The layout is worth
having on its own, because it puts one vector's codes contiguous and the scan
is memory-bound at nq=1 (P1).

**This is a refutation that promotes its mirror.** aarch64 at 2 bits is
currently on exactly the layout this experiment just showed costs x86 2.5x.
H1 is no longer a symmetry question — it is the measured-good layout being
withheld from one arch by a condition written for a different purpose.

### H3 — a 2-bit permute-dot — REFUTATION OVERTURNED BY P1, RE-OPENED

**The verdict below is wrong.** P1 measured 4 bits beating 2 bits outright at
nq=100 — 12.65 ms against 18.44 on arm, 16.94 against 26.03 on x86 — with
twice the code bytes. The classic kernel is what runs at 2 bits and the
permute-dot family is what runs at 4, so the arithmetic that follows predicted
the opposite of what the rig shows.

Where it went wrong: it priced permute-dot's per-query cost as if the dot
product were one multiply-accumulate per (dimension x query). `SMMLA` is an
outer product — 2 queries x 2 vectors per instruction — so its per-query cost
falls as the batch grows, while the classic kernel's 10 instructions per query
per byte-group do not. At nq=1 the arithmetic holds and 2 bits is ~2x faster
than 4 (1.93 ms against 3.71 on arm); at nq=100 it inverts.

H3 is re-opened as the climb's largest target: the nq=100 cells are where 2
bits is losing to 4, and a dot-product kernel is what closes it. Kept in full
below as a record of a refutation that measurement killed.

### H3 (original, superseded) — refuted by arithmetic

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

## Baseline (climb HEAD = 262793f, three interleaved rounds per cell)

`turbovec-bench-arm-search` (c4a, Axion) and `turbovec-bench-search` (c3,
Sapphire Rapids), `rm -rf target`, `maturin develop --release`, arch libopenblas
LD_PRELOADed, one process per cell. Medians, ms:

| cell | arm | x86 |
|---|---|---|
| nq1_st | **1.995** | **1.727** |
| nq1_mt | **0.306** | **0.487** |
| nq100_st | **148.991** | **83.086** |
| nq100_mt | **18.425** | **25.491** |

Re-pinned after two harness corrections; spread across rounds is now under 2%
except arm nq1_st (5.9%).

**Correction 1 — x86 nq100_st is bimodal inside a single process.** Iterations
land at ~82 or ~98 ms on an unchanged build, so a median picks a mode by
chance: three consecutive processes measured 83.1, 96.8, 84.1. That is an 18%
band on an objective cell, wide enough to manufacture or hide any plausible
win. `cells_2bit.py` now takes the best of three sub-runs on **every** cell,
not just nq=1, which selects the unperturbed mode.

**Correction 2 — the first arm re-pin measured the H1 build.** The box was
never restored to baseline after H1, and the numbers (nq1_st 2.907 against
H1's 2.904) gave it away. Both boxes are now rebuilt from 262793f with no
patch before pinning. *Every candidate run must be followed by a rebuild, or
the next measurement silently inherits the last patch.*

Two structural facts fall out before any hypothesis:

- **arm ST is 1.77x slower than x86 ST at nq=100** (148.8 vs 84.0) while arm MT
  is 1.41x *faster* (18.4 vs 26.0). The arches are not close to each other at
  2 bits in either direction.
- **Thread scaling differs wildly**: arm 8.07x at nq=100, x86 3.23x. x86 has a
  parallel-efficiency problem at 2 bits; arm has a per-core one.

## P1 — 2 bits against 4 bits, same box, same build

| cell | 2-bit | 4-bit | 4bit/2bit |
|---|---|---|---|
| arm nq1_st | 1.933 | 3.712 | x1.920 |
| arm nq1_mt | 0.302 | 0.556 | x1.843 |
| arm nq100_st | 148.770 | 99.557 | **x0.669** |
| arm nq100_mt | 18.441 | 12.651 | **x0.686** |
| x86 nq1_st | 1.669 | 3.270 | x1.959 |
| x86 nq1_mt | 0.502 | 1.046 | x2.083 |
| x86 nq100_st | 83.958 | 65.750 | **x0.783** |
| x86 nq100_mt | 26.026 | 16.939 | **x0.651** |

**At nq=1, 2 bits is ~2x faster than 4 bits on both arches** — it tracks the
byte ratio almost exactly, which is the memory-bound signature P42 found at 4
bits, inherited intact.

**At nq=100, 2 bits is 1.3-1.5x *slower* than 4 bits** — with half the bytes.
That is the whole story of this climb. 4 bits runs the permute-dot / vm8
family there and 2 bits runs the classic per-query TBL kernel, whose cost
scales with NQ while `SMMLA`'s does not. Half the memory traffic is being
handed back, with interest, in instruction count.

The nq=100 cells are therefore the target and H3 is the instrument. The nq=1
cells are already at the bandwidth limit and should be defended, not attacked.

## P2 — x86's "parallel efficiency problem" is four cores, not a bug

The baseline's 3.23x thread scaling on x86 against arm's 8.07x looked like the
climb's biggest free win: x86 nq100_st is 83.96 ms, so perfect scaling would
put nq100_mt near 10.5 ms instead of 26.0.

There is nothing to win. `lscpu` on the c3-standard-8: **4 cores, 2 threads per
core**. The c4a-standard-8 has 8 physical cores. Scaling measured on the box
(nq=100, ms):

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| ms | 126.9 | 65.3 | 33.7 | 32.4 |

1->2 is x1.94, 2->4 is x1.93, **4->8 is x1.04**. The kernel scales essentially
perfectly across physical cores and gains nothing from SMT, which is what a
port-bound scan should do. arm's 8.07x is 8 real cores doing the same thing.

The two arches' MT numbers were never comparable, and no scheduling change can
close a gap that is a hardware core count. Probe, not a hypothesis — it removes
a target rather than testing one.

*(Measured on the H2 build still installed on the box — the ST figure is H2's
126.9 rather than baseline's 84.0. The ratios are what this probe is about and
they are unaffected; the box has since been rebuilt at baseline.)*
