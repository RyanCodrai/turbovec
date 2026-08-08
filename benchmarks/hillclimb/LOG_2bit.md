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
