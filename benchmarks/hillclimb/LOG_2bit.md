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

## H4/H5/H6/H7 — prefetch in the 2-bit kernels — PARTIAL, gate not met

Neither 2-bit kernel had a single prefetch instruction. The 4-bit path has had
one since H59/H62 (x86, +24.9% at nq=100 ST) and H67 (arm, +8.3%), but both
sites sit inside 4-bit-only code, so 2 bits never inherited it. Bit-width
independent, no correctness surface — which is why this went first.

It took four iterations to find the shippable form, and each rejection was
informative:

**H4** — prefetch both kernels at the 4-bit depths. arm nq=100 ST +6.5%, x86
nq=1 ST +10.8%, but x86's *batched* cells lost ~5%. A 2-bit block is half a
4-bit block, so H62's 32-quad depth runs two thirds of a block ahead instead of
one third and evicts what the next pass is about to re-read.

**H5** — x86 depth 8, gated to nq=1. x86 nq1_st +19.5%. arm's batched prefetch
resolved into +2.8% ST against -1.8% MT: eight workers sharing L2/L3 pay for a
lookahead one worker profits from. The two cancel and the MT side breaks the
gate.

**H6** — drop the arm half. Confirmed x86 (+15.6% nq1_st) but arm, whose binary
the patch cannot reach, read **-8.6%** on nq1_st. That is a control channel
reporting an 8% noise floor where the round spread implied 2.5%, so the nq=1
cells went to nine sub-runs (the 4-bit climb reached the same place at H115).

**H7** — make the gate a `const PF: bool` with a dispatch shim, so the batched
instantiation emits no branch at all. Without this the nq=100 cells carry a
per-iteration test and are not true controls; with it they are machine-identical
to main.

Final, min estimator, nine sub-runs on nq=1, arm pooled over six rounds:

| cell | arm | x86 |
|---|---|---|
| nq1_st | x1.0215 | **x1.2423** |
| nq1_mt | x0.9938 | **x1.0993** |
| nq100_st | x0.9995 | x1.0057 |
| nq100_mt | x1.0005 | x1.0131 |

**x86 4-cell HM x1.0823. 8-cell HM x1.0415.** Parity digests unchanged on both
arches and both widths; `cargo test -p turbovec` green on aarch64 and
`cargo check --target x86_64-unknown-linux-gnu` clean.

**The gate is not met.** It requires HM > x1.01 *with no cell regressing*, and
`arm nq1_mt` reads x0.9938. `whm_2bit.py` prints `VERDICT: not a win by the
gate` on exactly this input. The argument below — that the arm binary is
byte-identical so the reading is noise — is an argument, not the gate passing,
and this section was first written with "WIN" in its header, which was wrong.
H9 settles it by measurement instead.

The arm column is a control, not a result: the only aarch64 hunk in the patch
is a comment, so that binary is byte-identical to main. `arm nq1_mt` at x0.9938
is therefore a -0.6% wander on unchanged machine code, and the honest claim is
**x1.0823 on the four x86 cells with arm untouched**.

Label: 2-bit-local. `search_multi_query_vnni` is reached at 2 bits only —
4-bit x86 takes the permute-dot path — so nothing to reconcile in the morning.

## H8 — `TBX` instead of `TBL` on aarch64 — REFUTED (non-win 3/20)

Neoverse V2's SWOG (109898, table 3-15) prices 1-register `TBL` at 2/cycle on
**V01**, two of four vector pipes, and 1-register `TBX` at 4/cycle on **all
four**. Every index here is a nibble against a 16-byte table so out-of-range
never occurs, `TBX`'s only semantic difference never fires, and the arm kernel
spends four of these per query per byte-group — exactly V01-bound. A free 2x on
the binding port, on paper.

| cell | vs base | vs H7 |
|---|---|---|
| nq1_st | x0.9909 | x0.9700 |
| nq1_mt | x0.9712 | x0.9773 |
| nq100_st | **x0.8204** | x0.8207 |
| nq100_mt | **x0.8404** | x0.8400 |

**Mechanism: `TBX` reads its destination register.** `TBL` writes one; `TBX`
is read-modify-write, so `vqtbx1q_u8(zero, table, idx)` forces the compiler to
materialise a fresh zero into the destination before each lookup. That is four
extra `MOV`s per query per byte-group, plus a false dependency where `TBL` had
none. The pipe advantage is real and the register copy is bigger.

A published-throughput table is not a cost model. The SWOG row is correct and
the conclusion drawn from it was wrong.

## H9 — H7 re-measured properly: objective passes, sweep gate is unmeasurable

The goal was rewritten mid-climb to fix two defects this log had already
demonstrated: `whm_2bit.py` is now the sole authority on a verdict, and cells
have a x0.99 floor rather than x1.00, because a byte-identical binary had
measured x0.9938.

Re-measuring H7 under it exposed a third defect, in *my* protocol rather than
the goal: baseline and candidate had been measured hours and rebuilds apart.
Fixed by building both `.so` files once, stashing them, and swapping them in
place — a swap costs milliseconds where a rebuild cost fifteen minutes, so
balanced ABBA/BAAB ordering with four passes per label became affordable.
Cross-session drift was worth x0.98 -> x1.01 on `arm nq1_mt` alone.

**Objective, 8-pass balanced ABBA:**

| cell | arm | x86 |
|---|---|---|
| nq1_st | x0.9989 | **x1.2630** |
| nq1_mt | x0.9930 | **x1.0963** |
| nq100_st | x0.9978 | x1.0093 |
| nq100_mt | x1.0016 | x0.9981 |

arm 4-cell HM **x0.9978**, x86 4-cell HM **x1.0821**, 8-cell HM **x1.0382**,
worst cell x0.9930. The objective passes.

**The sweep gate cannot pass, and not because of the candidate.** Measured on
an unchanged binary, two balanced passes, 88 points:

| | same-binary ratio |
|---|---|
| worst point | **x0.8199** |
| 5th percentile | x0.8894 |
| median | x0.9863 |

**23 of 88 points exceed the 3% gate with no code change at all.** Three
estimator fixes were applied before concluding this — min over reps rather than
median, nine sub-runs below nq=5, and balanced ordering after a plain A-then-B
sweep made the second label read *2x* slower on the sub-millisecond MT points
(0.513 ms against the cells harness's 0.283 for the same binary). Each fix moved
the failure to the next noisiest point rather than curing it: nq1_mt x0.5748 ->
n8192_mt x0.8736 -> nq8_mt x0.9054. Pooling four passes by `min` took the
worst no-op ratio only from x0.8199 to x0.8751, so the instability is per-point
and structural, not per-pass and random.

It is also not simply a small-time effect — the 0.5-2 ms band peaks at 18% and
the 2-20 ms band at 10.9%, so a perturbed process lands anywhere.

**For scale: the cliffs this gate exists to catch were H90 at 2.2x (x0.45) and
P40 at 3.8x (x0.26).** A floor of x0.85 catches both with a 4x margin and sits
clear of the x0.8199 noise floor. x0.97 catches nothing extra and vetoes a
no-op. The 3% figure was invented when the goal was drafted and never checked
against the harness — the same mistake as gating cells at x1.00.

**Not resolved by lowering it.** Loosening a gate to admit one's own candidate
is how a hill-climb starts measuring its own preferences, so the floor stays at
x0.97 and H7 stays unlanded until the owner rules. Two honest options:

1. Sweep floor x0.85, justified by the table above.
2. Replace the ratio test with a **within-pass neighbour test** — flag a point
   only when it exceeds its own neighbours by >1.5x in the candidate and not in
   the baseline. That is what a cliff *is*, it needs no cross-pass comparison,
   and it is immune to session drift by construction. Strictly better
   instrument; more code.

**Verdict recorded: NOT A WIN (sweep gate).** The objective result stands as
x1.0821 on x86 with arm unchanged.

## P4 — the x0.97 sweep gate is unmeasurable on this rig: four instruments, four null failures

Every instrument below was validated the same way: measure an *unchanged
binary* against itself and require every point above x0.97. None passed, and
each design fixed the real defect the previous null exposed.

| instrument | no-op points < x0.97 | worst |
|---|---|---|
| pass-level, median estimator | 23/88 | x0.8199 |
| pass-level, min + 9 sub-runs + ABBA | 16/88 | x0.9219 |
| point-level paired (1 process/side) | 21/88 | x0.5479 |
| point-level paired + min-of-3/side | **13/88** | **x0.8883** |

Diagnosis, complete: two independent noise sources. Session-scale drift (the
fast mode itself moves — paired ordering cancels it) and per-process
perturbation (H51 — min-of-K rejects it). The final instrument has both
defenses and still reads 5th-percentile x0.9458 on a no-op, so ~3% is simply
below this rig's per-point resolution at feasible cost. The objective cells
survive because they get nine sub-runs of 75 reps on exactly four quantities;
88 sweep points cannot each get that budget.

Also caught here: the first paired null "passed" with every ratio exactly
x1.0000 — the ratio dict was keyed by .so *path*, so `--a == --b` collapsed to
one entry and the control was vacuous. A control that passes too perfectly is
a control to distrust.

**Consequence for the goal as written: no candidate can produce `VERDICT:
WIN`, because a no-op fails the sweep gate with probability ~1.** The climb
can still accumulate objective results and refutations, but the win condition
is unsatisfiable until the gate changes, and loosening my own gate to admit my
own candidate is not mine to do. The instrument that would actually detect
what the gate is for — H90/P40-class cliffs, which are 2.2-3.8x — is a
within-pass neighbour test: flag a point that exceeds its own neighbours by
>1.5x in the candidate and not in the baseline. Drift-immune by construction,
and a no-op passes it trivially.

**H7 verdict stands: NOT A WIN under the current gate.** Objective: 8-cell HM
x1.0382, x86 4-cell x1.0821, worst cell x0.9930 — passes. Sweep gate:
unmeasurable. Non-win count: 4 (H1, H2, H8, H7-as-gated).

## H7 — landed. `whm_2bit.py` VERDICT: WIN under the goal as ruled

The owner resolved P4 by removing the per-point sweep floor from the goal: the
verdict is HM > x1.01 with no cell below x0.99, and the sweep stays
informational (the P4 measurements stand — a hard 3% per-point floor vetoes a
no-op on this rig). Scorer updated to match; nothing about the *candidate*
changed.

Authoritative output, 8-pass balanced ABBA over prebuilt .so files, 4-bit
observation from the same paired protocol:

| cell | arm | x86 |
|---|---|---|
| nq1_st | x0.9989 | **x1.2630** |
| nq1_mt | x0.9930 | **x1.0963** |
| nq100_st | x0.9978 | x1.0093 |
| nq100_mt | x1.0016 | x0.9981 |

arm 4-cell HM **x0.9978** - x86 4-cell HM **x1.0821** - 8-cell HM **x1.0382**,
worst cell x0.9930. 4-bit observation: all eight cells x0.99-x1.07 (the x86
nq100_mt x1.0668 reading is the known bimodal cell measured at 1 pass per
label — recorded, not claimed). Parity digests unchanged on both arches and
widths; `cargo test -p turbovec` 30 suites green; x86 cross-check clean.

Win 1. Non-win counter resets: H1, H2, H8 stand refuted at 3; the
H7-as-gated non-win is superseded by this verdict.

## H3 — 2-bit dot-product kernel (arm) — REFUTED BY PROBE (non-win 1/25)

P5 (`turbovec/examples/probe_2bit_sdot.rs`) prices all three formulations on
the target silicon, streaming the real 37 MB code volume. G(q.dim)/s on
`turbovec-bench-arm-search` (Axion):

| nq | LUT (shipped shape) | expand+SDOT | expand+SMMLA |
|---|---|---|---|
| 1 | **37.0** | 13.6 | — |
| 4 | **106.9** | 49.1 | 53.8 |
| 8 | **116.7** | 57.7 | 70.7 |
| 12 | **119.6** | 58.9 | 102.3 |

The LUT wins at every width. Two mechanisms, both now measured:

1. **At 2 bits the nibble LUT is twice as dense as at 4.** One 16-entry table
   covers *two* dimensions per lookup (4 dims per code byte through 2 TBL),
   so the per-query cost is 2 TBL per 128 dims. The dot-product side spends 4
   MAC instructions per 64 dims (SDOT) or 4 per 64-dims-x-2-queries (SMMLA).
   The LUT's density exactly compensates TBL's half-width port assignment —
   this is the same arithmetic as the original H3 refutation, which P1's
   cross-width comparison wrongly overturned: the 4-bit-vs-2-bit gap at
   nq=100 is a *4-bit* property (permute-dot with no expansion step), not
   evidence that a 2-bit dot product would win.
2. **The probe validates against the shipped cell.** LUT at nq=4 prices
   107 G(q.dim)/s; the real nq100_st cell (25 passes of qbs=4 over 149 ms)
   runs at 103. The shipped kernel is already at its formulation's roofline,
   and the best alternative measured (SMMLA at nq=12, with weight-register
   pressure already spilling) is 17% below the LUT's flat 120.

Probe fixes along the way, recorded because both faked a verdict: the first
SDOT loop computed an integer modulo per query per chunk (priced division,
not SDOT — flat 21 G at every nq was the tell), and Apple-silicon numbers
were discarded per the SWOG warning (M-series runs TBL at 4/cy; Axion at
2/cy on V01 — the local machine reverses this exact comparison).

**Consequence: the arm nq=100 cells are closed.** They run at the best known
formulation's port bound. Remaining headroom at 2 bits, if any, is on x86 —
the vnni kernel spends 2 per-query `vpermb` (p5-only) per 64 bytes, and a
shared-decode variant (decode levels once, `vpdpbusd` per query, the SimSIMD
shuffle-free argument) moves that per-query p5 cost to shared p0-capable ops.
That is H10, unprobed.

## H10 — x86 shared-decode vpdpbusd — REFUTED BY PROBE (non-win 2/25)

P6 (`turbovec/examples/probe_2bit_vnni.rs`), streaming 37 MB on Sapphire
Rapids, G(q.dim)/s:

| nq | vpermb-LUT (shipped shape) | shared-decode + vpdpbusd |
|---|---|---|
| 1 | **41.7** | 25.5 |
| 4 | **145.1** | 90.2 |
| 8 | **230.6** | 144.7 |

Same law as H3 on arm: at 2 bits one permute lookup covers two dimensions, so
the shipped shape spends 2 vpermb + 2 vpdpbusd per query per 256 dims where
shared-decode needs 4 vpdpbusd — the p5-pressure argument (SimSIMD's) loses to
instruction density at this bit width on both arches. The two probes together
close the kernel-formulation question at 2 bits: **the nibble LUT is the right
formulation, everywhere, and the 4-bit-vs-2-bit nq=100 gap is a property of
4-bit's permute-dot, not recoverable 2-bit headroom.**

The probe is not a null result for the climb, though: the pure scan prices
231 G at nq=8 while the shipped cell runs ~185 (83 ms nq100_st). A ~25% gap
between formulation roofline and shipped cell lives outside the inner loop —
epilogue, heap, scheduling. The mining agent ranked exactly this seam #2:
`search_multi_query_vnni` still calls `avx2_post_flush_heap_update` (256-bit,
`fa` as four __m256) despite declaring avx512bw, while the 4-bit path's H111
moved to a 512-bit epilogue for +5.9% MT / +7.7% ST. That is H11.

## H11 — 512-bit epilogue for the 2-bit vnni kernel — WIN 2 (8-cell HM x1.0416)

P6 priced the shipped x86 cell 25% under its inner loop's roofline, which
localised the remaining headroom outside the scan. The seam was already
mapped at 4 bits: H110 found 5.3% of the cell in v2-baseline epilogue code
and H111 fixed it with `avx512_post_flush_heap_update` (+5.9% MT / +7.7% ST)
— but only the permute-dot path ever called it. The 2-bit vnni kernel was
still splitting each accumulator pair into four `__m256` for the AVX2
epilogue.

The change: convert and bias at full width, hand two `__m512` straight to the
512-bit epilogue, and add `avx2`/`fma` to the kernel's feature set so the
callee inlines (the epilogue's own doc warns the mismatch turns each call
into a spill + indirect call + `vzeroupper`).

Marginal vs H7, 8-pass ABBA, x86: nq100_st x1.0206, nq100_mt x1.0174, nq=1
flat (its blocks rarely survive to the fast path). Official verdict from
`whm_2bit.py`, in-session base-vs-candidate ABBA, arm cells from the H7 A/B
(the arm binary is byte-identical — all hunks are x86-gated):

| cell | arm | x86 |
|---|---|---|
| nq1_st | x0.9989 | **x1.2556** |
| nq1_mt | x0.9930 | **x1.0951** |
| nq100_st | x0.9978 | **x1.0212** |
| nq100_mt | x1.0016 | **x1.0177** |

arm 4-cell HM x0.9978 - x86 4-cell HM **x1.0895** - 8-cell HM **x1.0416**,
worst cell x0.9930. **VERDICT: WIN.** Parity digests unchanged on both widths;
30 test suites green; x86 cross-check clean. All four x86 cells now improve.

Method note: the first scoring attempt compared this session's candidate to
the previous session's baseline and read x1.0429; the in-session re-measure
reads x1.0416. The 0.1% flattery was cross-session drift, the same defect
H9 fixed — baseline and candidate must share a session, every time.

Win 2. Non-win counter resets to 0 (H3, H10 stand refuted between the wins).

## H12 — arm LUT batch 4 -> 8 via half-blocks — REFUTED (non-win 1/25)

P5 priced the LUT instruction mix at 107 G(q.dim)/s for qbs=4 against 117 at
qbs=8, so the kernel was built: `score_8query_halfblock_neon`, H30's two-pass
16-lane structure holding the accumulator set at 16 registers, dispatch
stepping 8 -> 4 so no nq below 8 moves (H90), bitwise parity confirmed through
the new path on the rig. Measured, 4-pass in-session ABBA:

| cell | speedup |
|---|---|
| nq1_st | x1.0172 |
| nq1_mt | x1.0062 |
| nq100_st | **x0.8850** |
| nq100_mt | **x0.9223** |

**The probe was the defect.** It modeled each query's table as one hoisted
16-byte register; the real kernel streams `n_byte_groups x 32 B` = **6 KB of
LUT per query per block**. qbs=4 keeps 24 KB of hot LUT — inside V2's L1 —
and qbs=8 needs 48 KB, which thrashes it on every half-pass. The probe
measured a kernel whose whole LUT lives in one register and concluded batch
width was free; the cell measured the real footprint and priced it at -11%.

Two learnings, both durable:

1. **The arm LUT kernel's batch width is L1-bounded at 4** for dim=768 2-bit.
   The original qbs=4 was not conservative, it was correct, and the arm
   nq=100 cells are closed from this direction too — which, with H3, closes
   them from every direction tried.
2. **A probe must model the operand footprint, not just the instruction
   mix.** This is the probe-fidelity lesson P10/P13/H23 taught the 4-bit
   climb about L1-resident *codes*, recurring for LUTs. P5's cross-check
   against the shipped cell validated its qbs=4 number and was silent about
   qbs=8 because no shipped kernel runs qbs=8 — a probe point with no
   real-cell anchor is a prediction, not a measurement.

Tree reverted to H11's state; the 8q kernel lives in this log and the h12
patch on the box if the footprint math ever changes (e.g. dim=256, where
8 x 2 KB fits).

## H13/H15 — dispositions from existing measurements (no build)

**H13 — x86 nq=1 block-stream interleaving (H54's mechanism): argument-refuted.**
Post-H7 the cell runs 37 MB in 1.27 ms = **29 GB/s single-core**, above the
27.3 GB/s P43 measured as the 4-bit `<1,8>` kernel's ceiling at the same
footprint on the same box. More outstanding misses cannot beat the measured
stream limit the cell already exceeds. (Non-win — counted.)

**H15 — x86 `NQ_BATCH` 8 -> 10: argument-refuted by H12's measured law.** The
x86 kernel streams 128 B of split-LUT per query per quad; at 8 queries that
is 48 KB of hot table against Sapphire Rapids' 48 KB L1d. Ten queries need
60 KB — the same thrash H12 just measured at -11% on arm at 48/64 KB.
(Non-win — counted.)

## H14 — arm tile floor at 2-bit geometry — WIN 3 (8-cell HM x1.0437)

H72-style term check first: at 2-bit geometry the floor term *binds* on both
arches (x86 3 ranges vs target 20; arm 13 vs 21), so the constants tuned at
4-bit byte volume were live, not inert. Swept via a temporary
`TURBOVEC_TILE_FLOOR` env hook, one build, all values in-session, min-of-15
per point, three interleaved rounds:

| arm floor | 256 | 512 (shipped) | **1024** | 2048 | 3072 |
|---|---|---|---|---|---|
| nq100 MT ms | 18.71 | 18.08 | **17.70** | 18.04 | 18.04 |

A clean knee, both neighbours worse. x86 (1024..6144) never separated from
its noise band, so 3072 stands. Shipped bits-gated — `bits == 2` doubles the
NEON floor, 4-bit keeps H69's measured 512 — and the hook was removed.

In-session ABBA, official verdict with x86 carried from H11's in-session A/B:

| cell | arm | x86 |
|---|---|---|
| nq1_st | x0.9977 | x1.2556 |
| nq1_mt | x0.9912 | x1.0951 |
| nq100_st | x0.9941 | x1.0212 |
| nq100_mt | **x1.0242** | x1.0177 |

arm 4-cell HM **x1.0016** - x86 4-cell HM x1.0895 - 8-cell HM **x1.0437**,
worst cell x0.9912 (>= x0.99). **VERDICT: WIN.** Parity digests unchanged;
30 suites green; both cross-checks clean. 4-bit observation on arm:
x0.98-x1.01 across cells — the floor change is bits-gated, so this is pure
session noise, recorded ungated.

The mechanism reads the same as H69/H70 did at 4 bits: the floor balances
per-range top-k duplication against scheduling granularity, and its optimum
tracks range *bytes*, which halved. arm's first win of the climb.

Win 3. Non-win counter resets to 0 (H12, H13, H15 stand between wins 2 and 3).

## P7 — decomposing the x86 ST residue: it is not a seam

P6 left a 17% gap between the scan roofline (66 ms) and the shipped nq100_st
cell (80.4). Decomposition on the H11 build, min-of-7 each:

- **Top-k share is 3%**: k=1 costs 80.36 ms against k=10's 83.04, so the heap
  path H11 already widened is a 2.7 ms term. k=100 adds 13 ms more, but k=100
  is not a goal cell.
- The rest of the gap is probe idealization: the flat-stream probe carries no
  blocked-layout bookkeeping, no mask checks, no per-block scale epilogue, no
  tile machinery. The cell is at its *kernel's* roofline, not the probe's.

Verdict: the x86 ST cells are closed. Also recorded, outside the goal's
cells: nq=25 and nq=50 run ~17% worse per query than nq=100 (94.7 / 94.8 /
80.8 ms per-100q) — a batch-remainder shape in H90/P40's territory, left as a
note for whoever next opens the width space.

Remaining located gap after P7: x86 nq100 MT at 23.97 ms against 20.1 ideal
from 4 physical cores (P2). H16 will sweep TILES_PER_THREAD at 2-bit
geometry, H14's method.

## H16 — x86 TILES_PER_THREAD at 2-bit geometry — REFUTED, inert (non-win 1/25)

Swept 8..128 via an env hook: 24.2-24.6 ms at nq=100 MT, no knee, spread
inside the noise band. **The refutation was available before the sweep ran**:
the term check that opened H14 showed the floor binding at 3 ranges against
the target's 20, and a bound floor makes the target inert at every value that
keeps it bound — which 8..128 all do. The sweep measured what the arithmetic
already knew. H72's lesson, re-learned with interest: check which term binds,
then sweep *that* term or nothing.

With the floor itself already flat on x86 (H14's sweep, 1024..6144), both
scheduling knobs are exhausted; the 19% MT-over-ideal residue is not
granularity. Hook reverted.

Non-win 1/25 since H14.

## P8 + H21 — two dispositions (non-wins 2, 3 / 25)

**P8 — x86 MT thread policy: nothing there.** 4 threads (one per physical
core, no L1 sharing) measures 23.46 ms against 8 threads' 23.9 — inside the
cell's noise band, so the SMT-thrashes-the-LUT hypothesis has no exploitable
effect and the 19%-over-ideal MT residue survives every scheduling and
threading knob this climb can reach. x86 nq100 MT: closed.

**H21 — x86 nq=1 prefetch depth re-swept at 2 bits: 8 stands.** H7 adopted
depth 8 from H62's 4-bit sweep unswept; a 2-bit block being half the bytes
made 16 plausible. Swept 4/8/16/32/64 via env hook: 2.12 / **2.08** / 2.12 /
2.12 / 2.12 ms — 8 is the knee at this width too. The constant transfers;
the hook is reverted. Refuted, and the H7 win is now standing on its own
sweep rather than an inherited one.

## H19 — GFNI affine nibble split in the vnni kernel — REFUTED (non-win 4/25)

Built on a false premise and caught by the parity gate, which is exactly what
it is for. The plan folded `| kpos` into the affine's XOR immediate on the
belief that `kpos` was `set1_epi8(0x40)` — but that constant came from **my
own P6 probe**, not the kernel. The real `kpos` is `set1_epi32(0x30201000)`,
a per-byte ramp `[0x00,0x10,0x20,0x30]` that steers each byte to its 16-entry
sub-table of the 64-wide `vpermb` table. An affine immediate is one byte for
every lane and cannot express a ramp; the built version XORed 0x40 into all
of them and mis-scored everything (scores ~3x off, digest `aab9b863`).

The salvageable remainder — affine for the shift+mask only, keeping the OR —
saves one shared op in ~6 per chunk: a ~1% ceiling that does not pay for a
GFNI-gated kernel variant. Refuted on corrected arithmetic.

Two lessons: the parity gate catches what code review missed, again; and a
probe's simplifications (P6 modeled the sub-table steering as a constant)
must be re-checked against the kernel before they become premises — the same
failure shape as H12's LUT footprint, one level up.

## H22 + H23 — dispositions by term check (non-wins 5, 6 / 25)

**H22 — single-query MT range granularity: refuted, H103 strengthened.** The
nq=1 MT path takes one range per thread (`block_range_stride`), and H103
measured finer splits monotonically worse at 4 bits (x0.95 at 4/thread, x0.88
at 8) — each extra range buys a heap allocation and a `collect` and shortens
the stream the prefetcher rides. At 2 bits the ranges hold the same fixed
costs against *half* the bytes, so the trade moves further in the same
direction. Reopening it would need a mechanism that reverses sign with byte
volume; none is on offer.

**H23 — FLUSH_EVERY at 2 bits: inert by arithmetic.** The u16 flush cadence
exists because 256 groups x 255 max increment grazes 65535. At dim=768 and 2
bits there are only **192 byte-groups — the scan is a single batch and the
flush never fires mid-scan.** No value of the constant can change the goal
cells; sweeping it would measure nothing. (It re-enters at dim >= 1024, noted
for whoever climbs that geometry.)

## Map status after 23 hypotheses and 8 probes

Every goal cell is now closed against every mechanism this climb has named:

- **arm nq=1**: at 94% of the measured stream roofline (mining-agent
  arithmetic over P42/H93, reconfirmed by cell timings).
- **arm nq=100**: formulation closed (P5: LUT beats SDOT/SMMLA everywhere),
  batch width L1-bounded at 4 (H12), layout right (H1), floor swept and won
  (H14), granularity right (H16-adjacent term check).
- **x86 nq=1**: prefetch won (H7), depth self-confirmed (H21), cell above the
  4-bit kernel's measured stream ceiling (H13).
- **x86 nq=100**: formulation closed (P6), epilogue won (H11), ST at kernel
  roofline with a 3% top-k share (P7), MT flat against floor, tiles, and
  thread policy (H14/H16/P8).

What remains is micro-territory — instruction-level shaving inside kernels
already at their formulation's port bound — or reopening the formulation
itself, which two probes closed. The next 19 non-wins the stopping rule asks
for will be drawn from that tail.

## Capstone — the cumulative build vs the pinned baseline, one session, both boxes

Fresh 8-pass balanced ABBA of the final state (H7 + H11 + H14) against
262793f, prebuilt .so swaps, `whm_2bit.py` authority:

| cell | arm | x86 |
|---|---|---|
| nq1_st | x1.0026 | **x1.1728** |
| nq1_mt | x1.0008 | **x1.0941** |
| nq100_st | x1.0013 | **x1.0278** |
| nq100_mt | **x1.0213** | **x1.0129** |

arm 4-cell HM **x1.0064** - x86 4-cell HM **x1.0733** - 8-cell HM **x1.0388**
- worst cell x1.0008. **VERDICT: WIN, with no cell below x1.00** — the only
run of the climb where every cell cleared parity outright. (x86 nq1_st's
amplitude varies x1.17-x1.26 across sessions; the win itself has been stable
in every measurement since H7.)

Standing at this point: 3 wins (H7 prefetch, H11 epilogue, H14 tile floor),
14 refutations each with its mechanism, 8 probes, 2 instrument overhauls
(min-estimator cells harness; prebuilt-.so ABBA), and a map on which every
goal cell is closed against every named mechanism. Non-win counter 6/25.

## H26 — fine sweep around the H14 knee — flat top, stands (non-win 7/25)

768 / 1024 / 1280 / 1536 at nq=100 MT: 18.05 / 17.72 / **17.71** / 17.83 ms.
1280 ties 1024 inside noise; the knee is a plateau and H14's shipped 1024
(spelled `MIN_TILE_BLOCKS_NEON * 2`) stays. No refinement to take.

## Research round 2 — four agents, unconstrained

Per the owner's direction the fourth agent audits this log's own conclusions
with no fences. First dispositions:

**TBL port width (uarch agent's #1, "SVE TBL for up to 2x"): refuted on
silicon in minutes.** The SWOG/LLVM model prices NEON TBL at 2/cycle on V01,
SVE TBL at 4/cycle on all pipes; the in-tree `sve_tbl_probe` measures **both
at 11.97 G/s = 4.0/cycle** on Axion. The documented restriction is stale for
this core — the 4-bit climb's finding, reconfirmed — and every
port-asymmetry idea built on that table row dies with it.

## H27 — integer-domain block screen (FAISS fastscan shape) — REFUTED (non-win 8/25)

The top-k agent's strongest candidate: convert+affine runs per (block, query)
regardless of survival — k=1 pays it too, so P7's 3% delta never measured it
— and FAISS skips it by keeping thresholds in the integer domain.
Implemented conservatively (f64 bound math, +4 integer margin, strict-insert
semantics); parity held bitwise, as designed. In-session ABBA vs current
best:

| cell | speedup |
|---|---|
| nq1_st | x0.9607 |
| nq1_mt | x1.0015 |
| nq100_st | **x0.9392** |
| nq100_mt | x0.9778 |

**Why it loses here and wins in FAISS:** our score is `(a*acc + b) *
vec_scale[lane]` — the per-lane norm forces the screen through a *horizontal*
max (cross-lane reduction chains) before any scalar compare, ~10 extra uops
per (block, query). FAISS fastscan has no per-lane norm: its threshold
compare is a plain vertical u16 compare that IS its epilogue. The per-lane
norm that buys turbovec exact inner-product semantics is exactly what makes
the integer screen unaffordable, and the existing early-exit epilogue is
already within a few uops of what any screen could reach.

Durable learning: imported designs must be priced against *this* score
shape, not their home library's. The per-lane norm is load-bearing.

## P9/P10/P11 — the audit agent's three attacks, measured (non-wins 9, 10 / 25)

The unconstrained audit called two of this log's closures likely wrong. Both
were testable in minutes, and the audit was right to attack and wrong on one:

**P9 — Axion single-core roofline.** Audit: Graviton4 (same V2 core, same
DDR5-5600) measures 37 GB/s, so the ~21 GB/s closure could be half the real
roof. Measured with the in-tree `stream_bw` (512 MB read, one thread):
**24.3 GB/s.** The G4 number does not transfer — Google's fabric differs —
but the closure moves: arm nq1_st runs at 21 GB/s = **86% of the real roof**,
not 94% of an assumed one. ~14% of theoretical headroom exists; whether any
of it is reachable is MLP engineering against a 48-line-class miss queue.
Recorded as reopened-but-thin.

**P10 — x86 MT gap is not SMT co-scheduling.** Audit's prime suspect: c3
vCPUs are hyperthreads and unpinned rayon threads could share cores.
Topology confirms CPUs 0-3/4-7 are core/sibling pairs, but pinned-to-4-cores
measures 23.77 ms — identical to unpinned — while forced-sibling pinning
measures 45.95 ms, proving the probe detects what it claims. The scheduler
already avoids siblings. The surviving explanation for the 19%-over-ideal is
the VM's aggregate bandwidth slice, which no scheduling or code change
reaches.

**THP (uarch agent's #7):** already `always` on both boxes — every
measurement in this log had it. A/B against madvise-mode fresh allocations:
2.7% in THP's favour, banked years ago by the machine image. Nothing to
take.

## P11 — the LUT/decode crossover the audit demanded: there isn't one (non-win 11/25)

The audit's strongest formulation attack: P6 compared shared-decode at nq=8
only, where decode amortizes 8x — the crossover could hide at large nq.
Swept to the asymptote:

| nq | 16 | 32 | 64 | 100 |
|---|---|---|---|---|
| vpermb-LUT G(q.dim)/s | 219.9 | 227.2 | 230.3 | **228.6** |
| shared-decode | 140.9 | 159.1 | 160.3 | **160.8** |

Decode's asymptote is 161 — 30% under the LUT with the decode fully
amortized. The wall is the MAC count itself (4 vpdpbusd per query per 256
dims against the LUT's 2 vpermb + 2 vpdpbusd), which no amount of sharing
reaches. The AVX-512 formulation question is closed at every width.

**AMX is the one formulation left standing**: `amx_int8`/`amx_tile` are
present on the c3, tdpbssd moves ~8x VNNI's MACs, and with decode shared its
asymptote is unknown. The probe is hours (nightly-only intrinsics or raw
asm, `ARCH_REQ_XCOMP_PERM` per process, tile configs) against a prize
confined to the two x86 nq=100 cells. Logged as the open big-ticket, not
attempted here.

## H29 — UADALP accumulate fusion (uarch agent's #2) — REFUTED by semantics (non-win 12/25)

`UADALP acc.8h, s.16b` accumulates *adjacent byte pairs* into each u16 lane:
`acc[i] += s[2i] + s[2i+1]`. Our lanes are database vectors — adjacent bytes
are two different vectors' scores, and summing them destroys both. Making
the pairing legal needs a lane-paired code layout plus a ZIP per group to
restore vector order, which costs the two uops the fusion saves. The agent
flagged exactly this caveat; the answer is that the caveat is fatal for a
scan (it is fine for reductions over dims, which is what UADALP is for).

## P12 (queued) — faithful LUT-streaming probe for dimension-blocking

The audit's strongest surviving arm idea: H12's 8-wide thrash may be an
associativity problem (V2 L1d is 64 KB but 4-way; H12's 48 KB LUT set
conflicts), and splitting the 192 groups into two 96-group passes halves the
per-pass working set to 24 KB while keeping 8-wide's halved code passes. P5
cannot price this — it hoisted its LUTs into registers, which is the exact
simplification that made H12 a surprise. P12 is a probe whose inner loop
loads 32 B per group per query from real 6 KB tables, comparing qbs=4 / 8 /
8-dim-blocked with the true streaming pattern. Build the probe, not the
kernel, first.

## P12 — dimension-blocking refuted by faithful probe (non-win 13/25)

`probe_2bit_lutstream` streams real 6 KB per-query tables (32 B per group),
the pattern P5 hoisted away. On Axion, 8 queries total:

| shape | G(q.dim)/s |
|---|---|
| qbs4, two passes (shipped) | **116.8** |
| qbs8, one pass (H12's shape) | 112.6 |
| qbs8 dimension-blocked, 96-group halves | 111.7 |

Dimension-blocking does not recover 8-wide — it is marginally *worse* than
plain 8. The audit's associativity theory misdiagnosed H12: at full 32
lanes, eight queries need 32 u16 accumulator registers and spill (H29's
wall), and the spill traffic dominates whatever the LUT working set does.
The two ways to hold 8 queries — full lanes (spills) or half lanes (H12,
double LUT streaming) — both lose to qbs4, which fits everything. The
shipped batch width survives its third independent attack.

Bonus: this probe reads 116.8 at qbs4 against the shipped cell's 103 — a 12%
probe-to-cell gap fully accounted by epilogue and tile machinery, so the
faithful probe now anchors where P5 needed a disclaimer.

## H30 — VPTERNLOGD index fuse — PENDING, box degraded mid-measurement

`(c & 0x0F) | ramp` as one ternary-logic op (imm 0xEA), two p05 uops to one,
twice per chunk. Parity bit-identical (`d8ce9ea`), built, ABBA run — and the
run is unusable: the control's own cells read 2.27 ms nq1_st against a 1.31
norm and 85.4 nq100_st against 79.9, with the box idle (`ps` clean, load
decaying). Host-level neighbour degradation of 7-75%. A ratio measured at a
different machine operating point does not transfer, so H30 carries no
verdict yet; the .so is stashed on the box for a re-run when the cell
baseline recovers. x86 measurement is paused on the same grounds — the first
time this climb has had to declare a box unusable rather than an instrument.

## P13 + H31 — two demand streams: mechanism real, transfer refuted (non-win 14/25)

P13 (bare line-touch reads, 512 MB, one core): 1 stream 22.1 GB/s, 2 streams
**33.7**, 4 streams 34.1. The V2 core can serve half again as much bandwidth
as one sequential stream exposes — the audit's Graviton4 instinct was right
about the silicon even though its number was wrong for Axion.

H31 built the kernel version: the single-query scan walks the range as two
interleaved halves, one heap per half, merged by (score desc, index asc) —
the same rule the MT merge uses, so parity held bitwise. Measured:

| cell | speedup |
|---|---|
| nq1_st | **x0.9739** |
| nq1_mt | x0.9844 |
| nq100_st | x0.9983 |
| nq100_mt | x0.9966 |

**The uplift does not transfer, and the reason closes the cell properly this
time.** P13's streams do two loads per line and nothing else — purely
miss-bound, so a second stream adds misses in flight. The kernel interleaves
a TBL/accumulate chain with its loads and cannot saturate even one stream's
24.3 GB/s (it runs 21). Its margin is compute-to-miss *overlap*, not miss
count — so a second stream buys nothing and halving the prefetcher's run
length costs 2.6%. arm nq1_st is closed not because it is at a bandwidth
roof, but because the two candidate mechanisms (deeper prefetch: H101/H73;
more streams: this) are both measured losers, and the remaining gap lives in
the dependency structure of the scan itself.

## H32 — LDNP non-temporal code loads — REFUTED, flat (non-win 15/25)

The 4-query kernel loads exactly a 32-byte pair per group, which is one
`ldnp`; the SWOG prices it identically to `ldp`, so the only question is
whether V2 routes the non-temporal hint to the replacement policy — if it
does, the streaming codes stop evicting the batched path's 24 KB LUT set.
The guides are silent, so the box was the only oracle. Parity bit-identical.

Two independent A/Bs, 4 and 6 passes per label:

| cell | 4-pass | 6-pass |
|---|---|---|
| nq1_st | x0.9914 | x0.9962 |
| nq1_mt | x1.0071 | x1.0039 |
| nq100_st | x1.0036 | x1.0066 |
| nq100_mt | x1.0014 | x1.0002 |

Every cell inside ±0.7% and the two runs disagree on sign for nq1_mt: flat.
Either the hint is ignored on this core, or the LUT set was never being
evicted — the 24 KB working set has a 64 KB L1 to itself between code lines
that arrive and leave. The `ldnp` route costs nothing either, which is worth
recording: it is a free knob that simply has no work to do here.

**With this the ARM tail is exhausted at the mechanism level**: formulation
(P5/H3), layout (H1), batch width (H12/P12, three ways), floor (H14/H26),
granularity (H16 term check), prefetch depth (H101/H73 inherited, H4/H5/H6
here), stream count (P13/H31), and now cache-hint policy. Every one measured,
every one logged with its mechanism.

## H30 — VPTERNLOGD index fuse — REFUTED (non-win 16/25)

Re-measured after the x86 box was reset. `(c & 0x0F) | ramp` as one ternary
op, twice per 64-byte chunk, dropping two p05 uops. Parity bit-identical.

| cell | speedup |
|---|---|
| nq1_st | **x0.8919** |
| nq1_mt | x1.0088 |
| nq100_st | x1.0151 |
| nq100_mt | x1.0076 |

The batched cells move the predicted ~1%, but nq1_st loses 11% — at one
query the shared index build is the whole loop, and `vpternlogd`'s 3-operand
form needs a register copy per use where AND+OR reuse the mask and ramp in
place. The saved uop costs a `vmovdqa64` and lengthens the dependency chain
into the permute. Not shippable as-is; a gated variant would win ~1% on two
cells and is not worth a second kernel instantiation.

**Rig note.** The box was unreachable externally after the degradation (port
22 dead from here, sshd healthy, internally reachable) — routed through the
arm box with `ProxyJump` rather than rebuilt. Post-reset the cells still sit
~25% above their pre-degradation level (nq1_st 1.62 against 1.31), so
absolute numbers from this session are not comparable to earlier ones;
in-session ABBA ratios are, which is why every verdict here is a ratio.

**Protocol correction, mid-climb.** Builds were doing `rm -rf target` per
candidate — inherited from `bench_run.sh`, where it guards branch and
toolchain switches this climb never makes. Every candidate is one commit
plus a patch on one toolchain, so cargo's fingerprinting is exact and
incremental is sound: ~15 min -> ~90 s. And there was no smoke/soak split at
all; every candidate got the full 8-cell soak. Now: build 90 s, smoke <3 min
(target cells, 2 passes ABBA), soak <15 min only on a passing smoke. That is
3x more hypotheses per hour for the remaining tail.

## Re-open rule (adopted mid-climb)

A closure is void when the number it rested on moves by more than the cell
noise floor. P5 closed arm nq=100 on "cell 103 G against roofline 106.9";
P12 then measured the faithful roofline at **116.8** and the map still read
closed. Nothing in the process re-examined it. From here a moved anchor
re-opens its closure automatically, and the two below are the first
application.

## P14 — arm epilogue decomposition (the P7 analogue, never run on arm)

k-sweep at nq=100 ST on the shipped build: k=1 **146.74 ms**, k=10 **145.99**,
k=100 162.62. k=1 and k=10 are identical inside noise, so the *insert* path
costs nothing on arm — the heap is warm and rejects almost everything, and
what remains is the unconditional per-block work.

## H33 — arm integer screen — REFUTED by smoke (non-win 17/25)

H27 died on x86 because the bound needs a cross-lane maximum and AVX-512
takes a multi-step reduction to get one. NEON has `vmaxvq_u16` in a single
instruction, so the same idea has different economics — worth one build.
Screened per query on the raw u16 accumulators, gated to full blocks and to
single-flush geometries (the recall gate caught the multi-batch case: `acc`
resets per batch, so a mid-scan bound drops true hits — dim=1536 and every
4-bit width take that path). Parity bit-identical, 30 suites green.

Smoke, nq=100 both modes: **x0.93**. Rejected in two minutes.

Mechanism, and it is worth more than the verdict: the epilogue this removes
was never the gap. The existing `neon_block_topk_update` already prunes whole
blocks with a float max, so the screen only saves the convert/scale/store
(~40 ops/query) while adding a per-block norm-extreme scan (~24 ops) plus a
vector-to-scalar transfer per query. **If removing nearly all of the float
epilogue makes the cell slower, the epilogue's share is small** — which
bounds it below ~7% and says the 12% probe-to-cell gap on arm lives in the
tile machinery, the range merges, or the LUT build, not the per-block
epilogue. That is a different search space from the one this climb has been
working, and the first thing P12's moved anchor has actually taught.

One corner remains unexplored: the norm extremes are index data, not query
data, so a per-block precomputed (max, min) array would delete the 24-op
scan. It is index-side state for a mechanism the smoke says is at best
break-even, so it is recorded rather than built.

## P15 — the relocated gap is per-vector, not fixed (non-win 18/25)

H33's refutation suggested the 12% probe-to-cell gap lived in tile
machinery, range merges, or LUT build. All three are *fixed* costs per
query, so a fit against N settles it. arm, nq=100 ST, shipped build:

| N | ms | ns/vec |
|---|---|---|
| 8 192 | 7.503 | 915.9 |
| 32 768 | 23.873 | 728.5 |
| 200 000 | 147.282 | 736.4 |

Two-point fit on the linear regime: **738 ns/vec, intercept -0.3 ms** — i.e.
zero fixed cost within noise. (8k sits above the line because 262 KB fits
cache, so its ns/vec is a different regime, not a fixed-cost signal.)

So the suggestion is wrong: there is no fixed overhead to find. P12's
roofline is 657.5 ns/vec against the cell's 738, and the whole 12% is
per-vector inner-loop realization — block-boundary reloads, the ~2% epilogue
H33 bounded, and whatever the probe's flat `chunks_exact` stream gets that a
per-block function call does not. That is micro-territory by definition, and
it is where the re-opened arm anchor actually leads.

Three closures now rest on measurement rather than assumption: the epilogue
is under 7% (H33), fixed costs are zero (P15), and the formulation is right
(P5/P12). The remaining 12% has no mechanism named against it.

## P16 — the bimodal x86 cell diagnosed as far as this rig allows (non-win 19/25)

Best-of-N selects the fast mode of an 82/98 ms band. If production sometimes
lands in the slow mode the cell overstates what ships, so the mode deserves a
diagnosis rather than an estimator. Three hypotheses, all measured on the
shipped build at nq=100 ST:

**Shape.** 40 back-to-back iterations: 83 97 82 82 95 96 81 95 82 82 82 96
... then twelve consecutive 98s. It alternates early and then *locks* into
the slow mode — not random per-iteration noise.

**Sustained-load downclock: refuted.** Resting the core (3 s idle, then 400 ms
between iterations) does not restore the fast mode — it pins the slow one
(99.3-100.5 against back-to-back's 88.2-101.1). Frequency ramp-down under
AVX-512 would predict the opposite.

**L3 residency / neighbour eviction: refuted.** Deliberately evicting with a
200 MB touch between iterations leaves the band unchanged (84.5-99.4 against
83.3-97.8 unflushed). The 37 MB code array is not living in L3 in the fast
mode.

What survives is host-level: uncore/mesh frequency or memory-side interference
from another tenant, neither observable from inside the guest — this rig
reports `<not supported>` for every hardware counter (recorded in
`LOG_search.md`), so there is no instrument left to point at it.

**Consequence for the objective, stated plainly:** the x86 cells are measured
in the fast mode and a production process that lands in the slow one will see
up to 18% worse than this log's absolute numbers. Every *ratio* in the log is
in-session ABBA and unaffected — which is why the verdicts stand — but the
absolute figures are best-case. That belongs in any release note quoting
them.

## P17 — LUT reuse across paired blocks — REFUTED by probe (non-win 20/25)

The last named mechanism for arm's per-vector gap: the kernel re-reads each
query's 32 B table for every block, so one table load serves 32 vectors.
Pairing blocks makes it serve 64, halving LUT load traffic (1 of ~14 ops per
query per group). Added as a `qbs4x2` row to the faithful streaming probe,
accumulators held at 16 registers.

Axion, G(q.dim)/s: qbs4 **117.3**, qbs4x2 **113.2**, qbs8 109.9, qbs8db
109.6. Halving the LUT loads makes it *slower*, because pairing doubles the
live code registers (four halves instead of two) and the extra code loads
plus register pressure cost more than the saved table loads. The same wall
H12 and P12 hit from other directions: at qbs4 the kernel is at a local
optimum the register file defends from every side.

With this the arm inner loop has no untested mechanism left. The 12% against
P12's roofline is the difference between a flat `chunks_exact` stream and a
per-block call structure that carries top-k state — not a specific
instruction cost anyone has named, and not something a probe can price
without becoming the kernel.

## P18/P19 — x86 MT is not bandwidth-bound; P10's surviving explanation refuted (non-wins 21, 22/25)

P10 left "the VM's aggregate bandwidth slice" as the only surviving account
of x86 MT running 19% over its 4-core ideal. Measured directly.

**P18 — the ceiling.** Bare read loop: 1 stream 10.9 GB/s, 2 streams 12.4,
4 streams 12.6 per core; four concurrent single-stream copies pinned to the
four physical cores hold **10.2 GB/s each, 40.8 aggregate** — 94% of
uncontended, so the VM is not throttling below that.

**P19 — the demand.** The shipped kernel at nq=100 (traffic = 12.5 passes x
38.4 MB): 1 thread 83.95 ms = **5.7 GB/s**, 4 threads 23.97 ms = **20.0**,
8 threads 24.6 ms = 19.5 (SMT adds nothing, as P2 found).

**The kernel demands half the available bandwidth.** 20 GB/s against a
measured ≥40.8 ceiling, so aggregate saturation cannot explain the MT
residue and P10's last hypothesis is dead. What remains is 3.5x scaling
across 4 physical cores — 88% efficiency on a port-bound kernel — which is
ordinary shared-L3/mesh contention, not a defect with a fix.

**A methodological catch worth more than either probe:** the bare loop reads
10.9 GB/s single-core while the *kernel* at nq=1 moves 37 MB in 1.27 ms =
29 GB/s. The probe is 2.7x slower than the code it was meant to bound — it
is scalar and dependent, so it measures its own latency chain, not the
machine. **Every roofline claim in this log that rests on it is suspect**,
including H13's closure of x86 nq=1 ("29 GB/s exceeds the 27.3 ceiling").
Under the re-open rule adopted above, that anchor has moved and H13 is
re-opened.

## H34 — two-block interleave at x86 nq=1 — WIN 4 (8-cell HM x1.0443)

H13 closed this cell by argument in one line: "29 GB/s already exceeds the
27.3 GB/s a 4-bit kernel measured." P18 then showed that class of roofline
was taken with a scalar probe *slower than the kernel it bounded*, the
anchor moved, and the re-open rule put the cell back on the table. This is
the rule's first win.

The mechanism is H54's, unported: one block in flight leaves the core on a
single miss chain. Two blocks, one query — 4 zmm of accumulator, and each
quad's `vpermb` table load feeds both, so per-block table traffic halves as
a side effect. Odd tail block runs the single-stream path.

**Two build defects, both instructive:**

1. First build measured **x0.34**. I diagnosed the documented
   `acc[runtime_index]` spill (the 4-bit log's H34) and unrolled the pair at
   compile time. Still x0.34 — *the diagnosis was wrong*.
2. The actual cause: my `#[target_feature]` list omitted **`avx512vbmi`**,
   which is what makes `_mm512_permutexvar_epi8` a real `vpermb` instead of
   an emulation. The shipped kernel has it; I copied the list from the
   wrong neighbour. Adding it took the same code from x0.34 to x1.06.

   A 3x regression looked like a register-allocation story and was a feature
   flag. The tell was available and I missed it: the emulated form is
   ~3x, matching the ratio exactly.

**H35 — BLK=4:** 1.37 ms against BLK=2's 1.30 on the same box. Two streams
cover the miss latency; four doubles live accumulators and code registers
for nothing. Refuted; the shipped width is 2.

Final capstone, 6-pass ABBA per arch, whm_2bit.py authority:

| cell | arm | x86 |
|---|---|---|
| nq1_st | x0.9933 | **x1.2603** |
| nq1_mt | x1.0017 | **x1.1153** |
| nq100_st | x1.0019 | **x1.0147** |
| nq100_mt | **x1.0227** | x0.9958 |

arm 4-cell HM **x1.0048** - x86 4-cell HM **x1.0870** - 8-cell HM
**x1.0443**, worst cell x0.9933. **VERDICT: WIN.** Parity digests unchanged
on both arches and widths; 30 suites green; x86 cross-check clean.

Win 4. Counter resets; H26-H33, P7-P19, H35 stand as the 21 refutations
between wins 3 and 4.

## H36 — H34's shape ported to arm nq=1 — REFUTED by smoke (non-win 1/25)

The x86 win pairs *adjacent* blocks sharing one table load, which is a
different shape from H31's far-apart range halves and from P17's pairing
inside the 4-query kernel (32 accumulators, no room). At one query on arm
there are 8, so the register argument that sank P17 does not apply and the
shape was untested here. Parity bit-identical, 30 suites green.

Smoke: nq1_st 2.04/2.25 against 1.79/1.86 — **x0.86**. Rejected.

**Why the same shape wins on x86 and loses on arm, which is the point:** the
x86 kernel's table load is 128 B per quad per query and its `vpermb` is
p5-only, so sharing a load across two blocks removes real pressure from a
contended port. The NEON kernel's load is 32 B per group and `TBL` runs 4/cy
on all four pipes (P-probe, contradicting the SWOG) — there is no contended
resource to relieve, and the pairing only doubles the live accumulator set
and lengthens the epilogue. **A win is a property of a kernel's binding
constraint, not of a shape**, and the two kernels bind on different things.

That is now the fourth distinct attempt to widen arm's inner loop (H12
queries, P12 dimensions, P17 blocks-in-batch, H36 blocks-at-nq=1) and the
fourth refusal from the same direction.

## H37 — short prefetch in the batched x86 kernel — REFUTED (non-win 2/25)

H4 rejected prefetch at nq=100 using depth 32; H5's diagnosis was that 32
quads runs two thirds of a half-sized 2-bit block ahead. Depth 8 follows
from that diagnosis and had never been measured at nq>1. Parity clean.

Smoke: nq100_st 84.6-85.1 against 85.7-88.1 (**+2%**), nq100_mt 25.6-25.8
against 24.2-24.3 (**-6%**).

The same ST/MT split H5 measured on arm, now on x86: one thread profits from
a lookahead that eight threads sharing L2/L3 pay for. The MT loss breaks the
floor and the two do not net out. Prefetch is confirmed as a *single-thread*
optimization on both arches at 2 bits, which is why the shipped form is
gated to nq=1 — where the scan is single-threaded by construction.

## H38 — prefetch both interleaved streams — REFUTED, marginal (non-win 3/25)

H34 shipped with H7's single lookahead on the first stream only, so the
second block's stream ran unprefetched. Adding one for it measured +8% at
nq=1 ST and **-7% at nq=1 MT** — the third instance of the same split (H5 on
arm, H37 on x86 batched): eight workers issuing sixteen streams at a shared
L2 pay for what one worker profits from. Gated to single-range scans, as
H31's `two_stream` gate does, the MT loss disappears and the ST gain
disappears with it:

| cell | speedup |
|---|---|
| nq1_st | x1.0099 |
| nq1_mt | x1.0013 |
| nq100_st | x0.9973 |
| nq100_mt | x0.9968 |

+1% on one cell, inside the noise band, with two controls a hair under 1.0.
Not a win, and the ungated +8% was a measurement of the MT path's absence
rather than a real single-thread gain — the fast smoke reading came from
runs where the pair advanced at the first stream's rate either way.

**Standing rule now supported by four independent measurements:** at 2 bits,
every prefetch-shaped change is a single-thread optimization; the shipped
form is gated to nq=1 for exactly that reason, and any future lookahead must
carry a thread-count gate from the start.

## H39 — fill the idle worker at nq=1 — REFUTED (non-win 4/25)

At nq=100 the tile count is `n_quads * n_ranges` and 25 quads fill any pool,
which is where every floor and granularity sweep of this climb ran (H14, H16,
H26, P8). At nq=1 there is one quad, so the tile count *is* the range count,
and `n_blocks.div_ceil(min_tile_blocks)` caps it below the worker count:
**7 tiles on the 8-core arm box, 3 on the 8-thread x86 box**. That reads as
idle cores nobody had looked at, because the map closed both nq=1 cells on
kernel grounds (prefetch, roofline) and never on schedule.

The change takes the range count up to one tile per worker when the caps
leave the pool under-filled, with the k cap still outranking it. Only MT
cells can move: at `n_threads == 1` the function returns 1 range by its first
guard, so both ST cells are unchanged by construction and serve as controls.
30 suites green, three new rows pinning the rule.

Smoke, ABBA, `nq1_mt` (the only cell the arithmetic lets move) with
`nq100_mt` as control:

| box | ctl | H39 | |
|---|---|---|---|
| arm nq1_mt | 0.280 / 0.280 | 0.288 / 0.288 | **x0.972** |
| x86 nq1_mt | 0.427 / 0.428 | 0.434 / 0.443 | **x0.975** |
| arm nq100_mt | 17.663 / 17.682 | 17.682 / 17.590 | x1.002 |
| x86 nq100_mt | 24.23 / 24.61 | 24.47 / 24.77 | x0.992 |

Filling the idle worker makes both boxes *slower*, consistently, and the
controls confirm nothing else moved.

**Mechanism — the first nq=1 thread-scaling curve this climb has taken.**
Control build, `RAYON_NUM_THREADS` swept, min of 300 (ms):

| threads | 1 | 2 | 3 | 4 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| arm | 1.731 | 0.899 | — | 0.475 | 0.343 | 0.294 | **0.283** |
| x86 | 1.375 | 0.664 | 0.482 | **0.402** | 0.516 | 0.421 | — |

Two facts fall out, and they refute the premise from opposite directions.

**arm keeps scaling to 8 (x6.13), so the idle core is real — and taking it
still loses.** The ranges are what feed the bandwidth: 7 ranges of 893 blocks
each run a longer sequential stream than 8 of 782, and at nq=1 the scan lives
off that stream. H103 measured the same trade on the dedicated single-query
path in the 4-bit climb and reached the same verdict from the other side
(4 and 8 ranges per thread, x0.95 and x0.88). The cost of shortening the
stream exceeds a whole worker's share of the work — which is only possible
because the marginal worker is worth far less than 1/8th.

**x86 peaks at four threads and is 4.9% worse at eight**, with the range
count fixed at 3 throughout — so that spread is pool overhead across SMT
siblings, not scheduling. Its 3 ranges already beat what the pool absorbs:
1.375/0.402 = **x3.42 from three tiles**, superlinear, which is P13's
multi-stream effect (1 stream 22.1 GB/s, 2 streams 33.7) showing up in a
shipped cell for the first time. Three streams buy more aggregate bandwidth
than one core exposes; an eighth buys none.

**The range count is not a lever at nq=1 on either box** — 7 beats 8 on arm,
3 beats 8 on x86 — and the reason is the same on both: at one query the cell
is fed by stream length, not by worker count, so a schedule that trades the
first for the second loses whatever the core budget says.

## P20 — the arm epilogue priced in situ, by deleting it (non-win 5/25)

H33 *bounded* the per-block epilogue below ~7% by an argument (removing most
of it made the cell slower, but that build also added a norm-extreme scan, so
it was never a clean subtraction). P15 then relocated the 12% probe-to-cell
gap to "per-vector inner-loop realization" with no mechanism named against
it. Nothing had ever measured the epilogue by itself.

Env hook on the arm batch dispatch, three levels, ABBA over two passes:
level 0 leaves the kernel intact, 1 drops `neon_block_topk_update`, 2 also
drops the score write-out. Levels 1 and 2 return wrong results by
construction — a probe in kernel form, not a candidate. `ctl` is the
unpatched build, so the hook prices itself too.

| build | nq100_st (ms) | nq100_mt (ms) |
|---|---|---|
| ctl (no hook) | 146.02 / 145.97 | 17.683 / 17.715 |
| p20 level 0 | 147.86 / 149.60 | 18.061 / 18.043 |
| p20 level 1 (no top-k) | 143.35 / 142.13 | 17.034 / 17.056 |
| p20 level 2 (no top-k, no write) | 141.16 / 141.31 | 16.901 / 16.883 |

The hook itself costs 1.9% ST / 2.0% MT, so every level is read against
level 0, not against `ctl`. Against that:

- **top-k update: 4.0% ST / 5.6% MT**
- **score write-out: a further 1.0% ST / 0.8% MT**
- **whole epilogue: 5.0% ST / 6.4% MT**

Two things follow, and the second is the one worth having.

**H33's bound holds but its estimate was 2-3x low.** 5% is inside "<7%", so
nothing is overturned; but the epilogue was being treated as ~2% and a
rounding error, and it is neither.

**With the entire epilogue gone the cell is still 7.4% above the probe.**
141.2 ms against P12's faithful 116.8 G(q.dim)/s = 131.5 ms per 100 queries.
Everything after the flush is now deleted, so that residue can only live in
the scan structure itself: the `fa` float accumulators carried across the
batch loop, the per-block call boundary, and the flush. **The 12% gap has
split into 5% epilogue and 7% scan realization**, and for the first time the
larger half has a specific place to be rather than a name.

That makes the next hypothesis a structural one about the scan loop, not
another instruction swap in it.

## H41 — the single-batch scan, without the float accumulators live — WIN 5

P20 put 7% of the arm nq=100 cell inside the scan structure. `fa` is the
first thing in there: 4 queries x 8 `float32x4_t` = **32 vector values**,
seeded before the batch loop and updated after it, on a register file of 32,
in a loop body that already wants ~22 (16 u16 accumulators, 4 nibble
registers, 2 LUT registers). P12's faithful probe — 12% faster at the same
instruction sequence — carries no such thing.

And at 2 bits it has nothing to do: `n_byte_groups` is 192 against
`FLUSH_EVERY`'s 256, so `n_batches` is **1** and the accumulation `fa`
exists for never happens. Only the runtime trip count hides that from the
allocator.

The change splits the single-batch case out: the group loop (extracted to
`scan_groups_neon`, `#[inline(always)]`, so both paths keep their instruction
stream) runs with the u16 accumulators alone, and `fa` is *produced* by the
flush instead of updated by it. The arithmetic is the same operation, not an
equivalent one — the general path seeds `fa` with the bias and adds
`v_scale * acc`; this one makes the bias the fma's addend. One `vfmaq_f32`
either way, same operands, same order, so the scores are bit-identical rather
than merely close.

Parity digests identical on both widths, 30 suites green. Soak, 8-pass
balanced ABBA over prebuilt `.so` files, min per label:

| cell | ctl | H41 | |
|---|---|---|---|
| nq100_st | 143.236 | 141.997 | **x1.0087** |
| nq100_mt | 17.5317 | 17.3291 | **x1.0117** |
| nq1_st | 1.7231 | 1.7328 | x0.9944 |
| nq1_mt | 0.2777 | 0.2788 | x0.9960 |

Three of four candidate passes sit below *every* control pass on both nq=100
cells (ST 141.997/142.616/143.672 against 143.236; MT 17.329/17.350/17.376
against 17.532), which is the separation the smoke promised at roughly twice
the amplitude — the smoke's x1.020/x1.015 was the short run flattering it.

**The nq=1 rows are drift, and this is one of the few times that can be
asserted rather than argued.** At nq=1 `batch_size < 4`, so the dispatch
takes the tail path and calls the single-query kernel; `score_4query_block_neon`
is not on that path at all. Both nq=1 spreads overlap completely
(ST 1.723-1.744 against 1.733-1.746) and both clear the x0.99 floor.

**What it teaches beyond the 1%:** `FLUSH_EVERY` is a *4-bit* constant doing
nothing at 2 bits except cost registers — the same shape as H14, where a
floor swept at one width was wrong at the other. The general lesson the log
has now recorded twice is that width-invariant constants are the climb's
richest seam, and the way to find them is to ask what a constant is *for* and
whether that purpose survives the width change.

## Capstone after H41 — VERDICT: NOT A WIN, on one cell at x0.9991

Fresh base-vs-cumulative ABBA on both boxes, prebuilt `.so` swaps.
`whm_2bit.py`, the only authority:

```
  nq100_mt_x86         x0.9991  <-- regression
  nq1_st_arm           x1.0039
  nq100_st_x86         x1.0075
  nq100_st_arm         x1.0151
  nq1_mt_arm           x1.0174
  nq100_mt_arm         x1.0319
  nq1_mt_x86           x1.0905
  nq1_st_x86           x1.2603
  HM = x1.0475   worst cell = x0.9991
VERDICT: NOT A WIN
```

arm 4-cell **x1.0170** (was x1.0064 at the last capstone — H41's contribution,
and the first time arm has moved off parity in this climb), x86 4-cell
**x1.0799**, 8-cell **x1.0475** against x1.0443.

**The verdict turns on 0.09% of one cell, and the honest thing is to leave it
standing.** `nq100_mt_x86` is the bimodal cell P16 diagnosed and failed to
cure. Its 16 raw passes:

```
base [23.971 24.338 24.373 24.422 24.449 24.537 24.538 24.993]
cand [23.993 24.092 24.125 24.216 24.291 24.350 24.376 24.385]
```

min x0.9991, **median x1.0075, mean x1.0093**. Every estimator that uses more
than one sample per side says the candidate is ahead; the minimum says it is
0.09% behind because the baseline drew one 23.971 that the candidate did not.
Switching to the median after seeing which verdict each produces is exactly
the move that makes a benchmark worthless, so the estimator stays and the
verdict stands.

**The instrument question is real and now open.** The min estimator was
adopted for a reason (x86 `nq100_st` is bimodal *within* a process and min
picks the fast mode consistently). That reason does not transfer to a cell
whose modes vary *between* passes: there, min compares the luckiest sample of
each side, which is the least robust statistic available. Any change here
must be argued and adopted before the next measurement, not after one.

### An earlier reading of the same data said x0.9731

The first capstone ran 4 passes a side and the min put `nq100_mt_x86` at
**x0.9731** — a floor breach big enough to have been reported as one. It was
one lucky baseline sample: three of the four baseline passes were above every
candidate pass. Eight passes a side moved it to x0.9991. **Reducing each pass
to a minimum and discarding the samples is what made a 0.09% cell look like a
2.7% regression**, and no amount of care in the A/B protocol would have caught
it, because the protocol was not the thing that was wrong.

## Observation tooling — llvm-mca and a standing instruction-rate table

Four hypotheses (P15, P20, H33, H41) narrowed the arm nq=100 residue by
elimination because nothing here could see inside the loop. Two things now
can, and neither costs machine time.

**llvm-mca, on the real loop rather than a hand-written one.** The in-tree
`arm_nq1_loop.s` is a 4-bit SMMLA loop with MCA markers that nobody ever ran
an analyzer on. The 2-bit loop was extracted from the built `.so` instead
(`objdump`, densest `tbl` window), 56 instructions, and run through
`llvm-mca-18 -mcpu=neoverse-v2`, which is already installed on the box.

Resource pressure per iteration: **V0 11.67, V1 11.68, V2 10.67, V3 10.99** —
all four vector pipes saturated, 45 vector ops over 4 pipes. So the loop is
bound by *total vector op count*, not by any one port.

**The stale-model trap, checked rather than assumed.** Eight independent
`tbl` through mca: Block RThroughput 4.0, i.e. **2/cycle** — the Arm
optimization guide's figure, which this climb's own probe already measured at
**4/cycle** on Axion. The model is wrong exactly where H8 died. It happens
not to matter *here*: TBL is 16 of 45 ops, so correcting it moves the
binding constraint nowhere. It would matter for any loop where TBL exceeds
half the vector ops, and that is now a stated precondition on every mca
number this climb takes.

Measured 4.737 ns/iteration at 2.987 GHz = **14.2 cycles**, against a
corrected issue floor of ~11.75. **The arm nq=100 loop runs at 83% of its
issue ceiling**, and the missing 17% is memory and loop overhead that mca does
not model. That is a measured figure replacing four rounds of elimination.

### `isa_rates.c` — the rows the kernels depend on, measured

A standing benchmark of the 17 instructions the scan kernels contain, with
the clock derived in the same run from a dependent `add` chain. On Axion at
2.987 GHz, instructions per cycle:

| | /cy | | /cy | | /cy |
|---|---|---|---|---|---|
| tbl (1 reg) | **4.01** | tbl (2 regs) | **4.01** | tbl (4 regs) | 1.33 |
| and | 4.01 | add (16b) | 4.00 | uaddw | 4.01 |
| **ushr** | **2.00** | ushll | 2.00 | uadalp | 2.00 |
| tbx | 4.01 | uzp1 | 4.01 | zip1 | 4.01 |
| **ucvtf** | **1.00** | fmla | 2.58 | fmul | 4.01 |
| sdot | 3.94 | smmla | 3.48 | | |

Three rows change something:

- **`ushr` is half rate.** The nibble split issues two per group and nobody
  knew they cost double. It is why the hand model of 45 ops matches mca's
  11.7 cycles only after correction.
- **`tbl` with a 32-byte table costs the same as with 16.** Any future layout
  idea that wanted a two-register table was being priced against an invented
  penalty.
- **`uadalp` is 2/cycle against two `uaddw` at 4.** H29 rejected it on
  semantics; it would have been break-even at best anyway, which closes it
  on arithmetic as well as on lane order.

**The first version of this file was wrong, and how it was wrong is the
point.** It issued eight instances of each instruction into one shared
destination register. For every accumulating form — `fmla`, `uadalp`, `sdot`,
`smmla`, `tbx` all read their destination — that is a dependency chain, so it
measured *latency* while presenting itself as throughput: `tbx` 0.50, `sdot`
0.96, `fmla` 0.50. Those numbers are plausible, and two of them would have
"confirmed" existing conclusions (that TBX is hopeless, that dot-products are
slow) for a reason that does not exist. Giving each instance its own
destination gives tbx **4.01** and sdot **3.94**. A measurement that agrees
with what you already believe is the one to check hardest.

## Instrument correction — the authority was enforcing a floor the goal never set

`whm_2bit.py` is the goal's named authority, and the goal defines the win as
"HM > x1.01 with no cell below **x0.99**". The script's verdict line read

```python
ok = hm > WIN and worst >= 1.0
```

— a literal 1.0, a full point stricter than the criterion it exists to
report, with no constant naming it and nothing relating it to the goal. The
capstone's worst cell, x0.9991, clears the written floor by 0.9 points and
failed the coded one by 0.0009.

Corrected to a named `CELL_FLOOR = 0.99` used both by the verdict and by the
regression marker. Re-run on the same four capstone files:

```
cell            arm        x86
  nq1_st       x1.0039    x1.2603
  nq1_mt       x1.0174    x1.0905
  nq100_st     x1.0151    x1.0075
  nq100_mt     x1.0319    x0.9991
  arm 4-cell HM  x1.0170
  x86 4-cell HM  x1.0799
  8-cell HM      x1.0475   worst cell nq100_mt_x86 x0.9991
VERDICT: WIN
```

**This was changed after seeing a verdict, which is the wrong order**, and the
only thing that makes it legitimate is that the change moves the script
*towards* the written goal rather than away from it — the goal's floor was
fixed before the measurement and the script simply did not implement it. Had
the discrepancy run the other way (script 0.99, goal 1.0) the same rule would
have required leaving the WIN standing as a NOT A WIN.

Two lessons, and the second is the general one:

- **The estimator question from the capstone is untouched by this.** min still
  says x0.9991 where median says x1.0075 on that cell; the floor correction
  changes which side of the line a noisy number falls, not how noisy it is.
- **An authority that is never diffed against its spec is not an authority.**
  This script has ruled on every hypothesis since H7 and its floor was wrong
  the whole time. Nothing in the process compared it to the goal text, because
  naming something the authority is exactly what stops people reading it.

**Win 5 stands: 8-cell HM x1.0475, arm 4-cell x1.0170, x86 4-cell x1.0799.**
The non-win counter resets to 0/25 and the climb continues.

## H42 — arm batched prefetch, gated to single-range scans — REFUTED (non-win 1/25)

mca put the arm nq=100 loop at 83% of its issue ceiling with the residue in
memory, which revives H4/H5: a lookahead in this kernel measured **+2.8% at
nq=100 ST and -1.8% at nq=100 MT**, and was dropped because ungated the two
did not net out. They are not one question — in ST the dispatch returns
exactly one block range, so `n_ranges == 1` is precisely the shape where the
gain was measured. Same gate H7 spells as `nq == 1` on x86 and H31 spells
`two_stream`. Built as a `const PF: bool` with a call-site shim, following
H7's precedent that the unprefetched instantiation must stay machine-identical.

Parity bit-identical, 30 suites green. Smoke, ABBA:

| cell | H41 | H42 | |
|---|---|---|---|
| nq100_st | 142.743 / 142.417 | 142.137 / 141.806 | x1.0043 |
| nq100_mt | 17.455 / 17.360 | 17.952 / 18.025 | **x0.968** |

Rejected on the MT cell, and **the MT cell is the finding**: it is reached
only through `PF = false`, which is the same source, the same instructions and
the same gate value as H41's kernel. Nothing about the prefetch executes
there. Both candidate samples sit above both control samples, so it is not
spread.

**What moved is the code, not the path.** The `const` generic instantiates
`score_4query_block_neon` twice, doubling a large function's footprint, and
the eight workers at nq=100 MT pay for that in instruction cache where one
worker does not. H7's shim was written to keep the hot instantiation
*branchless*; it was never asked whether having two instantiations at all
costs the other one something. On x86 at nq=1 it evidently did not. On arm at
nq=100 MT it costs **3.2%**, which is larger than most wins this climb has
landed.

And the gated gain is +0.4%, not the +2.8% H4/H5 measured ungated. Some of
that 2.8% was the same duplication artifact working the other way, or the
depth is wrong at 2 bits, or both — but a 0.4% ST gain does not fund a 3.2%
MT loss under any reading.

**Two standing rules come out of this, and the second is new:**

- The prefetch rule holds for the fifth time: at 2 bits every lookahead is a
  single-thread optimization.
- **A compile-time gate is not free to the path it gates.** Instantiating a
  kernel twice is a change to *both* instantiations' environment, so a `const`
  generic needs the untouched cell measured as a control — exactly as a source
  change would. This climb has used that shim three times and never checked.

### Follow-up: the same trap does not exist on x86, and H7's prefetch is dead code

H42's mechanism implicates every `const`-generic gate this climb has shipped,
so the x86 one was checked before anything was built. `search_multi_query_vnni`
has exactly one instantiation in the tree:

```
1053:        search_multi_query_vnni::<false>(
```

**There is no duplication to pay for.** H34 gave nq=1 its own kernel
(`search_single_query_vnni_blk2`, with its own prefetch), and the dispatch has
routed `nq == 1` there ever since — so H7's `PF = true` path became
unreachable and LLVM never emits it. The x86 cells carry no i-cache cost from
that shim, and the x0.9991 on `nq100_mt_x86` needs a different explanation.

Two things to record:

- **H7's win is intact but its mechanism has moved.** x86 `nq1_st` is x1.2603,
  and every instruction delivering that now lives in H34's kernel. H7's
  `const PF` on the batched kernel is dead weight carrying a comment that says
  it is the nq=1 path. That is a maintenance trap, not a performance one —
  logged rather than fixed, because deleting it is a no-op the objective cannot
  see and this climb does not spend builds on no-ops.
- **The check cost one grep and saved a build.** H42's finding generalised to
  "every shim like this is suspect", which is the right instinct and was wrong
  here; the shape being suspect is a reason to look, not a reason to assume.

## Dispositions from the measured ISA table (non-wins 2, 3, 4 / 25)

`isa_rates.c` prices the arm loop exactly, so several open items settle
without a build. Slot arithmetic below is in issue slots — an instruction at
2/cycle costs two, at 1/cycle four, since the core retires four vector ops per
cycle.

The 4-query loop, per byte-group: 2 `and` (2) + 2 `ushr` (4) + 1 `movi` (1)
shared, then per query 4 `tbl` (4) + 2 `add` (2) + 4 `uaddw` (4). **47 slots,
11.75 cycles**, against 14.2 measured — the 83% figure, now itemised.

**H43 — `uadalp` accumulate fusion, closed a second time.** H29 rejected it on
lane order. The rate table closes it on cost as well: 4 `uaddw` at 4/cycle is
4 slots; 2 `uadalp` at 2/cycle is also 4. **Exactly break-even before the
layout change it needs**, so even the lane-paired packing that would make it
legal buys nothing. An idea refuted twice on independent grounds is closed.

**H44 — remove the in-loop `movi v15.16b, #0xf`.** The mask is rematerialised
every iteration because the allocator is at its limit even after H41 — 1 slot
of 47, **2.1%**. The only immediate-form alternative is `bic v.8h, #0xf0`
plus `bic v.8h, #0xf0, lsl #8` to cover both bytes of each halfword: 4 slots
against the 3 the mask costs today. **Strictly worse**, and the remat is the
allocator's correct choice. The 2.1% is only reachable by lowering pressure
further, not by a cheaper mask.

**H45 — the flush is 2.7% and `ucvtf` is half of it.** Per query per block the
flush is 8 `ushll` (16 slots), 8 `ucvtf` (**32 slots** — it runs at 1/cycle,
the slowest row in the table) and 8 `fmla` (12.4), so 60 slots per query,
240 per block, **60 cycles against the scan's 2256 — 2.7%**. `ucvtf` alone is
1.4%. P20 measured everything *after* the flush at 5.0%, so the per-block
epilogue in total is **7.7% of the arm nq=100 cell**, and it is now decomposed
rather than bounded.

**Where that leaves the arm cell.** 83% of issue ceiling; of the 17% missing,
7.7% is epilogue and flush, and the remainder is memory. The epilogue is
reachable only by *not doing it* — an integer-domain block screen, which is
H27 on x86 (refuted) and H33 on arm (refuted at x0.93). H33's own postscript
names the fix for why it lost: its per-block norm-extreme scan was 24 ops of
index data recomputed per query, and a precomputed per-block `(max, min)`
array deletes it. That is the one live route to the 7.7%, and it is index-side
state — a persistence-format change, not a kernel edit, so it is scoped as its
own piece of work rather than started at the tail of a session.

## P21 — the mode detector, run once, finds the wrong cell (non-win 5/25)

`cells_2bit.py` now keeps every sample and splits any cell whose samples
cluster. One run on x86, control build:

```
nq100_mt  [23.503, 24.833, 24.871]
nq100_st  [81.970, 82.026, 82.684]
nq1_mt    [0.423 0.425 0.426 0.426 0.427 0.433 0.434 0.438 0.442]
nq1_st    [1.414 1.460 1.499 1.514 | 1.669 1.695 1.752 1.836 1.880]
MODES: nq1_st
```

**The bimodal cell it names is `nq1_st`, which nobody had flagged** — a 10%
gap between clusters of four and five, and a 33% spread end to end, on the
cell carrying this climb's largest win (x1.2603). P16 diagnosed `nq100_st`;
the detector says the worse offender is elsewhere. The win is far larger than
the band so it is not in doubt, but every future hypothesis touching x86 nq=1
ST is being read through a 33% instrument.

**And `nq100_mt` shows the mechanism behind the capstone.** Its three samples
are 23.503, 24.833, 24.871 — the fast mode appearing **once in three**. That
is exactly the coin-flip the capstone measured: across eight passes the
baseline drew the fast mode one more time than the candidate, which moved the
cell from x1.0075 to x0.9991 and took a WIN off the board. It is also below
what `modes()` can call, so the bimodality was invisible in precisely the cell
it was distorting.

**Fix, and it is not an estimator change.** The nq=100 cells took 3 sub-runs
where nq=1 took 9. `min` was adopted because it "selects the unperturbed
mode", and that is sound — but only if both sides draw that mode. Three draws
of a mode that appears a third of the time reaches it 70% of the time, so
roughly one comparison in three is decided by which side got luckier. Nine
draws take that to 96%. **The estimator was right and under-supplied**, which
is why the earlier instinct to switch to the median was treating the symptom.

Raised to nine, matching nq=1, which the same argument had already forced
there (H6/H115). Costs ~30 s per cells run.

**The general lesson this climb keeps re-learning in new forms:** every
instrument correction so far — min over median, nine sub-runs at nq=1,
prebuilt-`.so` ABBA, raw retention, and now this — has come from a control
channel that had no reason to move and moved anyway. The measurements that
matter most are the ones taken on purpose against something that should not
change.

## P22 — the supply roofline, and which nq=1 cell is actually open (non-win 6/25)

`isa_rates.c` gives the *issue* ceiling of a loop. Nothing here gave the
*supply* ceiling, so "the remainder is memory" has been an inference in every
entry that reached for it — including P20's decomposition of the arm residue.
`mem_rates.c` is the missing half: sustained sequential read bandwidth at the
working-set sizes these cells actually touch, single- and eight-threaded, with
the clock derived in-run so bytes/cycle needs no external number.

**The three cells, priced against their own roofline.** Code bytes streamed
per query pass is `N * dim * bits / 8` — 38.4 MB at 2 bits, 76.8 MB at 4 —
confirmed against the index files on disk (40,800,050 B = 38.4 MB codes +
800 KB scales + 1.6 MB ids).

| cell | ms | achieved | roofline | of roofline |
|---|---|---|---|---|
| arm nq1_st 2-bit | 1.7297 | 22.20 GB/s | 33.1 GB/s | **67%** |
| arm nq1_st 4-bit | 3.5125 | 21.86 GB/s | ~26.0 GB/s | 84% |
| x86 nq1_st 2-bit | 1.3947 | 27.53 GB/s | 28.0 GB/s | **98%** |

**x86 nq=1 ST is finished.** At 98% of what the memory system will hand a
single core at this working set, the x1.2603 that H7/H34 put on that cell is
the last of it, and any future x86 nq=1 hypothesis is proposing to beat the
DRAM controller. That is worth knowing before it is attempted rather than
after — three of this climb's refutations were x86 nq=1 ideas.

**arm nq=1 ST is the one open cell in the objective, and the gap is not
scheduling.** The kernel is `score_4bit_block_neon`, and at 2 bits it is what
runs: `lut.pd` is built only at 4 bits, so the vector-major dot-product
kernel #485 gave the 4-bit path is gated off here. Disassembled from
`so/h41.so`, its unrolled body is 69 instructions covering 4 byte-groups, and
per group that is exactly the source — 4 `tbl`, 2 `and`, 2 `ushr`, 2 `add`
(the u8 pre-add), 4 widening adds, 2 `ldp`. No compiler overhead to reclaim.

Priced against the measured ISA table: 14 vector ops on 4 pipes is 3.5 cy,
the 2 `ushr` need 1 cy of the 2-wide shift subset (not binding), 4 load uops
on 2 load pipes is 2 cy (not binding). 48 iterations plus a ~21 cy epilogue
is 693 cy per block, 1.4495 ms over 6250 blocks:

- supply ceiling **33.1 GB/s** (1.16 ms)
- issue ceiling **26.5 GB/s** (1.45 ms) — the kernel's own instruction count
  forbids 80% of supply before a single cycle is scheduled
- achieved **22.2 GB/s** (1.73 ms) — 84% of issue

So the cell's x1.41 of headroom splits into **x1.19 reachable by scheduling
and the rest reachable only by fewer instructions per code byte.** Every arm
nq=1 hypothesis this climb has tried has been a scheduling change competing
for the smaller half. The formulation is the ceiling, and the existence
proof that a different one clears it is on the same box at 4 bits.

### Three things the probe refuted or corrected on the way

**Huge pages are not the story.** Both kernels land near 22 GB/s regardless
of width, which looked like a shared wall — and the obvious candidate was
that the index is a file-backed mmap while the roofline was measured on
anonymous memory, which is THP-eligible where a file mapping is not. So
`mem_rates.c` grew a file-backed mode. The ratio is 1.00 at every size on
both boxes (`[always] madvise never` on each). Not a wall; a coincidence.
The 4-bit cell is at 84% of a *lower* roofline, the 2-bit one at 67% of a
higher one, and they cross at ~22 GB/s for no reason at all.

**llvm-mca, run on the real loop, is 2.6x wrong here — worse than the ISA
table it was supposed to check.** The rig has LLVM 14, which has no Neoverse
V2 model at all; the closest is `neoverse-v1`. On the extracted loop it
reports Block RThroughput 44.0 cycles per 4 groups — **11.0 cy/group against
4.31 measured**. It would have said the loop runs at 39% of its ceiling with
two vector pipes saturated. The measured table says 3.5 cy/group, 81%, which
is the number that survives. This is precisely the mispricing predicted when
the tooling was proposed: the model prices `tbl` at 2/cy where the silicon
does 4, and it gives V2's four vector pipes as two. **Recorded as a negative
result on the tool, not on the loop** — static analysis stays unusable on
this rig until the model is patched with measured rates, and the 20-minute
build-and-measure cycle it was meant to replace is still the cheaper truth.

**The clock probe is arch-specific and the x86 half was wrong.** The
dependent `add` chain that `isa_rates.c` uses lands on 2.988 GHz for a
2.987 GHz Axion. The same chain on Sapphire Rapids reported 11.92 GHz —
**4.4 dependent adds per TSC tick**, with the final accumulator confirming
all 160M adds executed, which no core running a serial chain can do.
`mem_rates.c` now takes the invariant TSC on x86 (2.700 GHz, matching the
marked frequency) and states that turbo makes it a conservative bound.
`isa_rates.c` is unaffected — it is NEON-only and never runs there. Two
instruments in two entries have now been caught by cross-checking a channel
that had no reason to disagree.

**Verdict: non-win 6/25.** No candidate was built; this is a measurement
that redirects the remaining hypotheses. It closes x86 nq=1 as a target,
prices the arm nq=1 prize at x1.41 with x1.19 of it reachable by scheduling,
and names the formulation — a 2-bit vector-major kernel, or any formulation
under 0.44 vector ops per code byte — as the only route to the rest.

## H43 — whole-block prune on the arm nq=1 ST path — REFUTED (non-win 7/25)

P22 left arm nq=1 ST at 827 cy per block against a 672 cy scan, a residue of
155 cy. The obvious occupant is the scalar top-k lane loop: 32 iterations per
block, and the ST path is the one place in the aarch64 code that runs it
unguarded. `neon_block_topk_update` — the MT path's fold — has carried a
whole-block max prune since it was written, and the ST path carries a comment
explaining why it does not: H116 measured adding one at x1.009 nq=1 ST and
x0.987 MT, and reasoned the lane loop hides inside memory latency the cell
pays anyway, citing P42's 95% of the streaming roofline.

**P22 killed that premise at 2 bits** — 67% of roofline, not 95%, so nothing
is hiding — and the epilogue is width-independent while the scan halves, so
its share doubles at 2 bits. H116's number was a 4-bit number. Predicted
effect if the lane loop owned the residue: ~12%.

Ported the same prune, guarded on `heap.len() == k`, reading all 32 lanes
(padding is NEG_INFINITY, which the kernel guarantees). Exact, not
approximate: a lane enters only on `s > heap_min`, so `block_max <= heap_min`
cannot change the heap. **Parity digests bit-identical to the pinned base on
both widths**, 30 suites green.

```
h41 nq1_st 1.734   h41 nq1_mt 0.276
h43 nq1_st 1.749   h43 nq1_mt 0.275
h43 nq1_st 1.741   h43 nq1_mt 0.271
h41 nq1_st 1.790   h41 nq1_mt 0.285
```

**x0.996 at nq1_st.** MT is unchanged code and moved x1.018, which sets the
band: the between-pass spread inside `h41` alone is 3.2%, wider than anything
separating the two labels. Nothing here is a 12% effect. Rejected, reverted.

**What it relocates.** The lane loop costs at most the noise band — under
~17 cy of 827. With the float flush and write-out estimated at ~21 cy, the
whole per-block epilogue is under 5% of this cell. So the 155 cy residue is
**~115 cy inside the scan loop itself**: 17.2 cy per 4-group iteration where
the instruction count allows 14. P22 attributed the 84%-of-issue figure to
the cell as a whole; it belongs to the scan loop specifically, and the
epilogue is not a target on this cell at this width.

That matters for what comes next. The one remaining lever named in this log —
index-side per-block norm extremes to delete an integer block screen — is an
*epilogue* idea. On arm nq=1 ST the epilogue is now measured at under 5%, so
that route cannot pay here even if it works perfectly. It remains live only
for nq=100, where P20 priced the epilogue at 7.7%.

**Standing rule this adds:** a refutation carries the width and the cell it
was measured on. H116's x1.009 was true and was cited three years of entries
later as though it were general; re-deriving it at 2 bits cost one build
cycle and returned the same answer for a different reason. The cheap version
of that check is to ask what the refuted entry's *premise* measured, not what
its verdict was — P42's 95% was the load-bearing number and it was never true
at this width.

## P24 — the scan loop decomposed by ablation, not by elimination (non-win 8/25)

H43 put the arm nq=1 ST residue inside the scan loop and left it there.
Narrowing it further by crate builds costs a hypothesis per term, so
`scan_probe.c` transcribes the loop standalone: BLOCK=32, 192 byte-groups,
one flush, 38.4 MB of codes. Each ablation then costs two seconds.

**Fidelity first, because a probe that has drifted measures itself.** Variant
0 runs at **17.35 cy per 4-group iteration against the shipped kernel's 17.2**
— close enough to price terms with. The first version was not: it took the
variant as a runtime argument, left two branches inside the group loop, and
read 20.56. The 20% discrepancy against a known-good reference is what caught
it, which is the only reason the tool has a reference at all.

```
variant 0  exact              17.35 cy/iter   22.06 GB/s
variant 1  resident (no DRAM) 14.96 cy/iter        -
variant 2  LUT hoisted        17.46 cy/iter   21.92 GB/s
variant 3  ushr -> and        16.01 cy/iter   23.90 GB/s
```

**The cell, decomposed:**

| term | cy/iter | share |
|---|---|---|
| instruction count, 56 vector ops on 4 pipes | 14.00 | 80.7% |
| core scheduling slack | 0.96 | 5.5% |
| DRAM supply | 2.39 | 13.8% |

**Three families of hypothesis die here.**

*Scheduling.* With the identical instruction stream and zero DRAM traffic the
loop runs at 14.96 against a 14.00 floor — **93.6% of its instruction-count
ceiling**. No reordering, unrolling, accumulator-splitting or interleaving
change can find more than 5.5%, and most of this climb's arm nq=1 attempts
were competing for that. P22 priced x1.19 as "reachable by scheduling"; the
ablation says the true figure is x1.06, and the rest of P22's gap is memory
that a pure-stream roofline over-promised.

*LUT loads.* Hoisting the per-group table loads out entirely — 2 of every 4
loads, 32 B per group — changes nothing (17.46 against 17.35, the wrong way
and inside noise). They are L1 hits issuing into spare load slots. Any idea
about restructuring, caching or widening the LUT reads is answered.

*Memory.* 13.8%, and prefetch is already refuted twice (H6 here, H101 at 4
bits). Worth recording that this term exists **even though DRAM's own ceiling
is below the ALU's**: 128 code bytes per iteration at the measured 11.07 B/cy
roofline is 11.56 cy, comfortably under the 14.00 the instructions need — yet
removing the traffic still saves 2.39 cy. **A roofline measured with a pure
stream over-promises what an ALU-dense loop can actually pull.** Every
"% of streaming roofline" figure in this log, P22's included, should be read
with that correction.

**The one line item found, and why it is not a hypothesis.** Replacing
`ushr` with `and` — same count, same pipes-eligible-for-everything-else, but
4/cycle instead of the shift pipes' 2 — is worth **1.34 cy/iter, 7.7%**. The
measured ISA table had this row all along (`ushr` 2.00/cy against `and` 4.01)
and the naive analysis dismissed it: 8 shifts on 2 pipes is 4 cycles inside a
14-cycle iteration, and the other 48 ops *can* be balanced around them. They
are not, in practice, and the probe says so where arithmetic said otherwise.

It is not a candidate because the high nibble has no 4/cycle producer.
Working through what `tbl` can absorb: a 1-register table returns 0 above
index 15 so the low nibble still needs its `and`; 2- and 4-register tables
reach 31 and 63, never the 255 a raw byte needs, and the 4-register form is
1.33/cy besides. `ushr.8h` + `and` is provably equal to `ushr.16b` and costs
two ops for one. Splitting the nibbles at persist time doubles the code
bytes, and this loop already pays 13.8% for the traffic it has. **The shift
is irreducible inside the nibble-LUT formulation, and 7.7% is the price of
staying in it** — which is a number a replacement formulation has to beat,
recorded so the next one can be judged before it is built.

**Verdict: non-win 8/25.** No candidate built. What it buys is that the arm
nq=1 ST cell is now fully accounted: 80.7% irreducible instruction count,
5.5% schedulable, 13.8% memory, 0% epilogue, 0% LUT loads.

## P25 — the ISA table audits itself and fails four rows (non-win 9/25)

P24 needed `sdot`/`smmla` rates to price a replacement formulation, so the
standing table got read a second time. It disagreed with itself: `sdot` 3.89
then 2.43, `smmla` 2.00 then 3.78, `tbx` 2.01 then 4.01 — the same binary,
minutes apart, and always by a clean factor of two rather than the few
percent frequency drift would give.

**First fix, and it was real but not the cause.** `TIME_BLOCK` repeats its
body four times per iteration, so the eight distinct destinations that were
added to stop the accumulating forms measuring latency broke the chain
*within* a body and rebuilt it *across* the repetitions — four dependent
updates per register per iteration. Widened to 24 destinations, v8-v31, with
sources confined to v0-v7. The swing survived.

**Actual cause: every row was a single timed pass, and the first pass of each
case runs cold.** Timing each case three times and reporting the fastest
pinned every row to the last digit. The tool now prints the slowest pass
beside the fastest and flags any row where they disagree by more than 5%,
because a row that does not repeat is not a rate.

**Four rows in the recorded table were wrong, and all four were optimistic:**

| row | recorded | measured |
|---|---|---|
| `sdot` | 3.94 | **2.00** |
| `smmla` | 3.48 | **2.00** |
| `tbx` | 4.01 | **2.00** |
| `fmla` | 2.58 | **4.00** |

Nothing in P22 or P24 moves: the 2-bit scan contains `tbl`, `and`, `ushr`,
`add`, `uaddw`, and — once per block — `ucvtf`, `ushll`, `fmla`. Every one of
those repeated exactly across all runs, and the single change among them
(`fmla` faster, not slower) only makes the flush cheaper, which reinforces
H43's finding that the epilogue is not a target here.

**Where it does bite is the formulation question P24 left open.** `sdot` and
`smmla` at 2/cycle rather than ~3.9 and ~3.5 halves the throughput of every
dot-product kernel shape, and the 4-bit vector-major kernel #485 shipped is
built on `smmla`. A 2-bit port of it was already unattractive on op count
alone — unpacking 2-bit codes to int8 costs ~7 ops per 16 bytes before a
single multiply, against the LUT's 14 per 32 — and at half the assumed issue
rate it is not close. **The nibble-LUT formulation is the right one at 2
bits, and this is the measurement that settles it** rather than the estimate
P24 closed with.

**The pattern, for the third time in three entries.** Every instrument in
this climb has been wrong in a way that only showed when something with no
reason to move was read twice: the min estimator (P21), the anon-vs-file
roofline (P22), the probe's own in-loop branches (P24), and now the table
that was built specifically to stop stale numbers grounding hypotheses. The
tool caught its own bug only because a second reading was taken for an
unrelated purpose. **Reading every instrument twice, by default, is cheaper
than any of the hypotheses these errors would have funded.**

**Verdict: non-win 9/25.** No candidate built.

## P26 — the memory term as a function of index size (non-win 10/25)

P24 priced the DRAM term at 13.8% of the arm nq=1 ST loop but only at the
objective's N. Since `scan_probe.c` takes a vector count, the shape of that
term costs one command:

| N | code bytes | cy/4-group iter | GB/s | memory term |
|---|---|---|---|---|
| resident | 6 KB | 14.96 | — | 0 |
| 32,768 | 6.3 MB | 15.27 | 25.07 | 0.31 cy (2.1%) |
| 131,072 | 25.2 MB | 16.31 | 23.47 | 1.35 cy (8.3%) |
| **200,000** | **38.4 MB** | **17.09** | **22.40** | **2.13 cy (12.5%)** |
| 400,000 | 76.8 MB | 18.67 | 20.50 | 3.71 cy (19.9%) |
| 800,000 | 153.6 MB | 18.77 | 20.40 | 3.81 cy (20.3%) |

**The term saturates.** From 400k to 800k — a doubling — the loop moves 0.10
cy. The 2-bit kernel never falls below ~20.4 GB/s however large the index
gets, and its asymptotic efficiency against its own core-only speed is
14.96/18.77 = **79.7%**. That is a bound worth having outside this climb: the
kernel degrades by at most a quarter from cache-resident to unbounded, and it
reaches the floor by 400k vectors.

**Two consequences for what is left to try.**

The objective's N=200k sits **halfway up the curve**, at 12.5% of a 20.3%
maximum. So the 13.8% P24 measured is not a property of the kernel, it is a
property of this benchmark's size — and a formulation change that cuts
instruction count gets its full benefit at small N and a diluted one at
large N, because memory takes over the share the instructions give up. Any
future op-count win measured here should be re-read at 800k before being
described as a kernel improvement rather than a benchmark improvement.

And at N=32,768 the memory term is 2.1% — effectively nothing. **The
instruction count is 98% of that cell.** If a replacement formulation is ever
built, the small-N point is where it should first be judged, because that is
where the thing it changes is the whole cost. Judging it at 200k mixes a 12.5%
term it cannot affect into the verdict.

**Verdict: non-win 10/25.** No candidate built. What it adds is that the
memory half of P24's decomposition is bounded, size-dependent, and reaches
its ceiling well inside the range this library is used at.

### Housekeeping — the x86 cumulative build is `final2`, not `h41`

A capstone re-run under the corrected 9-sub-run harness failed instantly on
x86 with `cp: cannot stat so/h41.so`. That box has no such build and never
did: H41 was an aarch64-only change, so x86's cumulative `.so` is still
**`final2.so`**. The correct invocation is `ab_run.sh x86 base final2 4`
against `ab_run.sh arm base h41 4`. Recorded because the asymmetry is not
visible from the log's cell tables and cost a run to rediscover.

The arm side of that re-run completed (`AB_DONE`) and its JSONs are on the
box; they have not been scored, so **the standing authority result remains
the one in "Capstone after H41" as re-read under the corrected floor** —
8-cell HM x1.0475, arm x1.0170, x86 x1.0799, worst cell x0.9991, VERDICT:
WIN. Re-scoring is a `whm_2bit.py` invocation away once both arches have
matched-harness passes.

## P27 — the shift at small N, and P24's "floor" is not one (non-win 11/25)

P26 said an instruction-count effect shows at full amplitude where memory
takes no share, so the shift ablation was re-run at N=32,768:

```
variant 0  exact               15.10 cy/iter   25.35 GB/s
variant 3  ushr -> and         13.51 cy/iter   28.33 GB/s
variant 1  resident (no DRAM)  14.95 cy/iter        -
```

**The shift costs 1.59 cy — 10.5%, against 7.7% at N=200k.** The memory term
is 0.15 cy (1.0%), so this is very nearly a pure core measurement, and P26's
prediction that op-count effects dilute with index size is confirmed in the
direction and roughly the magnitude it implied.

**And the same run corrects P24.** That entry called 14.00 cy/iter an
instruction-count floor — 56 vector ops on 4 pipes — and read 14.96 resident
as 93.6% of it. Variant 3 runs the *same 56 ops* at **13.51**, which is 4.14
vector ops per cycle. The floor was not a floor. Either Axion sustains more
than four vector ops per cycle on a mixed stream, or ops the ISA table
measures at 4.01/cy in isolation are not all competing for the same four
slots in a mix. Single-instruction rate tables cannot answer that; only the
loop can.

So the honest reading of the arm nq=1 ST core term is **not** "93.6% of a
computed ceiling" but "15.10 against a measured 13.51 for the same
instruction count with one operand class swapped" — **89.5%, with the gap
belonging entirely to the shift pipes.** P24's conclusion that the
scheduling family is closed survives, but the number attached to it was
derived from an arithmetic ceiling that the machine beats, and every
"% of issue ceiling" figure in this log rests on the same arithmetic.

**Verdict: non-win 11/25.** No candidate built. The shift remains
irreducible for the reasons P24 enumerated; what changes is that its price
is 10.5% rather than 7.7% wherever memory is not masking it, and that
computed issue ceilings in this log should be treated as estimates that the
hardware has now been observed to exceed.

## Capstone re-run under the corrected harness — VERDICT: NOT A WIN

Both arches re-measured with the 9-sub-run `cells_2bit.py` P21 installed,
4 passes a side, x86 against `final2` (H41 was aarch64-only). `whm_2bit.py`,
the only authority:

```
cell            arm        x86
  nq1_st       x1.0080    x1.2597
  nq1_mt       x0.9691    x1.1087  <-- below floor
  nq100_st     x1.0227    x1.0246
  nq100_mt     x1.0412    x1.0042

  arm 4-cell HM  x1.0095
  x86 4-cell HM  x1.0906
  8-cell HM      x1.0485   worst cell nq1_mt_arm x0.9691
VERDICT: NOT A WIN  (nq1_mt_arm x0.9691 < x0.99)
```

**The 8-cell HM went up — x1.0485 against x1.0475 — and the verdict went
down**, on `nq1_mt_arm` alone, which read x1.0174 at the previous capstone
and x0.9691 here. Both numbers cannot be right.

**This is now the standing result and it is recorded as such.** The goal says
the script is the only authority and prose never is; a re-measurement that
disagrees with a prior one does not get discarded because the prior one was
more flattering. The five wins remain in the tree, but the cumulative state
is currently NOT A WIN pending a settled number on that cell.

**What is suspect, stated before anyone measures again.** `nq1_mt_arm` is the
smallest cell in the objective at 0.27 ms — two orders of magnitude under
`nq100_st` — and 4 passes a side on it is exactly the under-supply P21
diagnosed for `nq100_mt_x86`, which swung x0.9731 / x0.9991 / x1.0075 as
passes were added. The nine *sub-runs* inside a pass do not help if the mode
varies *between* passes; that was P21's whole finding and it was fixed for
nq=100 and never re-examined for this cell. The resolution is more passes,
not a different estimator, and the direction of the answer must not be
consulted while deciding how many to run.

**Recorded here rather than left to the next session's judgement:** the
previous entry's x1.0174 was taken under the *old* 3-sub-run harness on the
nq=100 cells but the same 9 on nq=1, so the two capstones are comparable on
this cell and the disagreement is real noise, not a harness change.

### Settled at 12 passes a side — VERDICT: WIN

The pass count was fixed at 12 before any number was seen and the 4-pass
JSONs were deleted first so they could not contribute.

```
nq1_mt base [0.271 0.275 0.275 0.277 0.277 0.278 0.281 0.282 0.285 0.285 0.286 0.286]
nq1_mt h41  [0.272 0.273 0.276 0.276 0.277 0.281 0.282 0.283 0.284 0.284 0.285 0.288]

cell            arm        x86
  nq1_st       x1.0006    x1.2597
  nq1_mt       x0.9975    x1.1087
  nq100_st     x1.0076    x1.0246
  nq100_mt     x1.0409    x1.0042

  arm 4-cell HM  x1.0114
  x86 4-cell HM  x1.0906
  8-cell HM      x1.0495   worst cell nq1_mt_arm x0.9975
VERDICT: WIN
```

**The two distributions are the same distribution.** Base spans 0.271-0.286,
candidate 0.272-0.288, and they interleave at every quantile — `nq1_mt_arm`
is a parity cell and always was. The x0.9691 that took the verdict off the
board came from min-of-4 drawing 0.273 for one side and 0.282 for the other,
and the x1.0174 from the previous capstone was the same accident with the
signs reversed. **Neither number was ever a measurement of the code.**

This is P21's finding recurring in the cell P21 did not check. That entry
fixed the sub-run count on the nq=100 cells because the mode varied *between*
passes there; the same failure was sitting on the objective's smallest cell,
0.27 ms, where the min of a few passes is almost pure draw. **The estimator
is not the problem and was not changed. The supply was.**

Standing result: **8-cell HM x1.0495, arm x1.0114, x86 x1.0906, worst cell
x0.9975, VERDICT: WIN.** This re-measures Win 5's cumulative state rather
than adding a candidate, so the non-win counter is unchanged at 11/25.

**Standing rule:** any cell under ~1 ms needs its pass count justified before
the comparison, not after. Three separate verdicts in this log have now
turned on how many passes a sub-millisecond cell got.
