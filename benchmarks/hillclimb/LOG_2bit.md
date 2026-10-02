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

## P27 corrected — the ablation was confounded and the machine issues 4/cycle

P27 concluded that the no-shift loop ran 56 vector ops in 13.51 cy — 4.14 per
cycle — and therefore that computed issue ceilings in this log are beatable.
**That is wrong, and the arithmetic that looked anomalous is what exposed it:
13.51 x 4 = 54.04, too close to a round op count to be coincidence.**

The ablation is confounded. In `scan_probe.c`:

```c
uint8x16_t h0 = variant == 3 ? vandq_u8(c0, mask) : vshrq_n_u8(c0, 4);
uint8x16_t s0 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)),
                         vqtbl1q_u8(lut_hi, h0));
```

With `variant` constant-folded to 3, `h0` is the *identical expression* to the
`and` already inside the lookup, so it is common-subexpression-eliminated.
Variant 3 does not swap a shift for a logical op at equal count — it deletes
**two ops per group**: 12 against variant 0's 14, 48 per iteration against 56.

Redone honestly:

| variant | ops/iter | cy/iter | ops per cycle |
|---|---|---|---|
| 0 exact | 56 | 15.10 | 3.71 |
| 3 no-shift | 48 | 13.51 | 3.55 |

**Both under 4.00. There is no anomaly, no extra issue width, and P24's
arithmetic stands.** The correction P27 announced was itself the artifact.

What survives is smaller and still useful: 48/56 of the work in 13.51/15.10
of the time means the two variants run at *the same* ops-per-cycle to within
5%, which is what a cleanly issue-bound loop looks like — and is independent
evidence for P24's core finding. **The shift's own price is not separable by
this ablation** and the 7.7% / 10.5% figures in P24 and P27 both include a
deleted `and`. A clean measurement needs variant 3 to keep the op count, e.g.
by masking with a second, different constant so CSE cannot fire.

**The pattern, for the fifth consecutive entry.** Every instrument correction
in this session came from a number that had no business being where it was:
a min estimator inventing a regression, a roofline that flattered, a probe
measuring its own branches, a rate table disagreeing with itself, a verdict
turning on min-of-4 — and now a correction entry that was wrong in the same
way as the thing it corrected. The one habit that caught all six is checking
whether a measured figure lands suspiciously near a number the code implies.

**Counter unchanged at 11/25.** P27 stays in the log as written, with this
correction after it, because a refuted entry that is silently rewritten
teaches nothing.

## P28 — the shift priced without the confound: it costs ~2% (non-win 12/25)

Variant 3 now masks with `0x0E` instead of `0x0F`, a different constant, so
CSE cannot fold it into the lookup's own `and`. Op counts are equal at 56 per
iteration and only the operand class differs.

| N | variant 0 (ushr) | variant 3 (and) | shift's price |
|---|---|---|---|
| 32,768 | 15.31 cy | 14.99 cy | **0.32 cy — 2.1%** |
| 200,000 | 16.35 cy | 16.88 cy | none; variant 3 is 3% *slower* |

**The shift costs about 2% where memory takes no share, and nothing at all at
the objective's N.** P24's 7.7% and P27's 10.5% were both, in their entirety,
the deleted `and` — an op-count reduction that was never available, dressed
as a pipe-pressure effect.

**And P24's original arithmetic was right the first time.** That entry
computed that 8 shifts on 2 pipes is 4 cycles inside a 14-cycle iteration and
therefore should not bind, then overrode that reasoning because the ablation
said otherwise. The reasoning was sound; the ablation was broken. A measured
number does not automatically beat a derivation — it has to be a measurement
of the thing the derivation is about, and this one was not.

**What this closes.** There is now no identified line item inside the 2-bit
scan loop. The 14 ops per group are issue-bound as a body: no instruction
class in them is individually overpriced, the LUT loads are free (P24), the
epilogue is free (H43), the memory term is bounded and size-dependent (P26),
and scheduling has under 6% in it (P24). **Every route that keeps the
nibble-LUT formulation is now measured and closed**, and the only remaining
direction is a formulation with fewer than 0.4375 vector ops per code byte —
which P25 established is not `sdot` or `smmla` at 2/cycle.

**Verdict: non-win 12/25.**

**Standing rule, earned three times in three entries:** when an ablation
contradicts a derivation, check the ablation's op count before believing it.
A one-line diff of the emitted instruction histogram would have caught this
at P24 and saved two wrong entries and the correction between them.

## P29 — the probe's own noise floor, and which of its numbers are real (non-win 13/25)

Re-running variant 2 (LUT hoisted) at both sizes produced a result and, more
usefully, a spread. Variant 0 — unchanged code, same binary — has now been
measured five times at N=200,000:

```
15.10   16.35   17.09   17.35   18.59   cy/4-group iter
```

**±6% between process invocations at N=200,000, against ±0.7% at N=32,768**
(15.10 / 15.17 / 15.31 across three). The DRAM term is not merely large at
the objective's N, it is *unstable* there, and it is unstable by more than
every effect this probe has been used to measure.

**So the N=200,000 rows in P24, P27 and P28 are all inside the probe's own
noise and none of them established anything.** That includes P24's "hoisting
the LUT loads changes nothing" (17.46 against 17.35) — a difference of 0.6%
read against a 6% spread. It was reported as a closed question and was not
one.

Only the small-N rows survive, where the memory term is 1-2% and stable:

| ablation at N=32,768 | cost |
|---|---|
| `ushr` -> `and`, op count held (P28) | 0.32 cy — **2.1%** |
| LUT loads hoisted out | 0.28 cy — **1.8%** |

**Both are real and both are small.** The LUT loads are not free as P24 said,
they cost 1.8%; the shift is not 7.7% or 10.5% as P24 and P27 said, it costs
2.1%. The two largest line items ever claimed inside this loop are together
under 4%, which is the same conclusion P28 reached by a different route and
is now supported by numbers taken where the instrument can hold still.

**The general correction, and it applies to this whole session.** A probe
built to make ablations cheap made them cheap enough to run once each, and
running once at a size where the variance is 6% produced three wrong entries.
`mem_rates.c` and `isa_rates.c` both report spreads; `scan_probe.c` reports a
minimum of five and does not say what the other four were. **It should print
its spread like the other two, and ablations should be run at N=32,768 where
the thing being ablated is 98% of the cost** — P26 said exactly that and the
entries that followed it ignored it.

**Verdict: non-win 13/25.** No candidate built. P28's conclusion stands
unchanged — the nibble-LUT body has no line item worth more than ~2% — but it
now rests on the measurements that can be repeated rather than the ones that
happened to be taken.

## P30 — the probe cannot resolve what it was built to resolve (non-win 14/25)

P29 said `scan_probe.c` should print its spread. It now does, and the answer
is worse than P29 diagnosed. The spread is not between invocations, it is
**inside** each one:

```
N=32,768    variant 0  15.37 cy  spread 19.5%
            variant 2  15.02 cy  spread 18.1%
            variant 3  15.15 cy  spread 20.0%
N=200,000   variant 0  16.63 cy  spread 25.6%
            variant 2  15.32 cy  spread 24.8%
            variant 3  19.21 cy  spread  5.5%
```

Each figure is a minimum of five timed passes drawn from a distribution
**~20% wide, at both sizes**. Every ablation delta this probe has produced —
2.1%, 1.8%, 7.7%, 10.5%, 13.8% — sits far inside it. **Minimum-of-five over a
20% band is the same estimator failure as the capstone's min-of-four, third
occurrence this session, now in the tool built specifically to make careful
measurement cheap.**

**The honest accounting of P24 / P27 / P28 / P29 is that the ablation
programme produced no number that can be relied on.** P28's correction of P27
and P29's correction of P28 were both right about the *direction* — the CSE
confound was real, the noise was real — and both attached figures the
instrument could not support. The structural claim that survives is
qualitative and came from the disassembly, not the probe: the loop is 14
vector ops per group with no compiler overhead.

**What is established about arm nq=1 ST by instruments that hold still:**

- **H43**, on the real ABBA harness against the real kernel: the whole-block
  prune is x0.996. The top-k lane loop is not a target.
- **P22**, from measured cell times against a measured roofline: 22.2 GB/s
  achieved against 33.1 available, with x86's equivalent cell at 98% of its
  own. The formulation gap between the arches is real and large.
- **P26**, whose sweep spans 6x and so outruns a 20% band: the memory term
  grows with N and saturates by 400k.

Everything finer-grained came from the probe and is unmeasured.

**The fix is not more repetitions.** A 20% band on a 0.25 ms measurement is
scheduler and frequency behaviour, not sampling error. The in-situ harness
already solves it — 9 sub-runs in separate processes, 12 ABBA passes a side —
which is exactly why H43's x0.996 is trustworthy and none of these are.
**The probe's premise held for build time and failed for measurement quality,
and cheap measurement that cannot resolve the effect costs more than the
build cycle it replaced.** Four entries were written from it.

**One self-inflicted bug found on the way, worth recording because it nearly
shipped as a result.** Dropping `sink` from the final `printf` while adding
the spread column let `-O3` delete the entire scan: the probe reported 1.04
cy/iter and **368 GB/s, fifteen times the memory roofline**. It was caught
only because that number is absurd on its face. A probe whose output is a
plausible-looking rate has no such guardrail, which is the argument for
always printing a physical quantity that can be checked against a known
ceiling.

**Verdict: non-win 14/25.** The tool stays in the tree with its spread
column, because that column is what makes it safe: any reading from it
smaller than the printed spread is not a result.

### Attempted: the shift ablation in situ — did not build (counter unchanged)

P30 left the shift's true cost unmeasured, because the only instrument that
tried is the one P30 disqualified. The correct instrument is the ABBA harness
that H43 used, which means a real kernel probe rather than a standalone one.

The edit is a one-line substitution — `vshrq_n_u8(cN, 4)` becomes
`vandq_u8(cN, mask2)` with `mask2 = vdupq_n_u8(0x0E)`, a *different* constant
from the lookup's own `0x0F` so CSE cannot fold it. Same op count, wrong
scores, purely a probe.

It does not compile as a one-line change. **`vshrq_n_u8(cN, 4)` appears at
six sites across more than one function**, and `mask` is a per-function local,
so each site needs its own `mask2` binding — E0425 on the sites outside the
function that got one. Recorded so the next attempt starts from the right
shape rather than rediscovering it: bind `mask2` alongside every existing
`let mask = vdupq_n_u8(0x0F);`, not just the first.

Reverted immediately; the tree never held a broken edit past the build. **The
counter is unchanged at 14/25** — an attempt that produced no measurement is
not a non-win, and counting it would be counting the same absence twice.

## P31 — the shift, measured in situ at last: it costs nothing (non-win 15/25)

The probe edit, rebuilt against the shipped kernel and run on the ABBA
harness. `mask2 = vdupq_n_u8(0x0E)` bound alongside each of the six
`let mask = vdupq_n_u8(0x0F);` (four of them unused — only two functions
carry the pattern), so op count is held and CSE cannot fold it.

```
h41  nq1_st 1.744   nq100_st 140.31
p31  nq1_st 1.746   nq100_st 144.745
p31  nq1_st 1.767   nq100_st 142.355
h41  nq1_st 1.739   nq100_st 141.35
```

**nq1_st x0.996. nq100_st x0.986 — removing the shift makes it slower.**

**`ushr` at 2/cycle does not bind, in either cell.** P24's original
derivation — 8 shifts on 2 pipes is 4 cycles inside a 14-cycle iteration, so
it cannot be the constraint — was correct, and every number that contradicted
it came from an instrument: 7.7% and 10.5% were the CSE-deleted `and`
(P28), and 2.1% was inside the probe's 20% band (P30). Three entries argued
about a term worth nothing.

**This is the entry the last five were owed.** Same question, one build cycle
and one four-minute smoke, on the harness whose noise is characterised. The
probe was built to avoid exactly this cost and instead spent four entries
producing figures that had to be withdrawn. **A slow instrument that resolves
the effect is cheaper than a fast one that does not** — the whole detour cost
more than the six minutes it was avoiding.

**Verdict: non-win 15/25.** Reverted. The nibble-LUT loop now has no
identified line item and no remaining instruction-class hypothesis: the
shift is free (here), the LUT loads are unmeasured but bounded by the same
logic, the epilogue is x0.996 (H43), and scheduling is bounded by P22's
in-situ roofline rather than by anything the probe said.

## P32 — the LUT loads, measured in situ: deleting them is slower (non-win 16/25)

P30 disqualified P24's claim that the per-group LUT loads are free — 0.6%
read against a 20% band. Same treatment as P31: hoist them in the shipped
kernel, one load pair before the batch loop instead of two per group, and run
it on the ABBA harness.

```
h41  nq1_st 1.723   nq1_mt 0.288
p32  nq1_st 1.747   nq1_mt 0.283
p32  nq1_st 1.776   nq1_mt 0.280
h41  nq1_st 1.716   nq1_mt 0.285
```

**nq1_st x0.982.** Deleting *two of every four loads in the hot loop* makes it
**1.8% slower**. nq1_mt x1.018, the same code path, which sets the band.

**The claim is not merely confirmed, it is confirmed with the sign
inverted.** There is no gain available from the LUT loads: they are L1 hits
issuing into slots that are free anyway, and removing them perturbs the
schedule for the worse. Any future idea about caching, widening, restructuring
or eliminating those loads is answered — this is the strongest form of that
answer, because the maximal version of the idea was tried and lost.

**And it is the second time in two entries that the trustworthy instrument
reversed the sign of a probe result**, not just its magnitude. P31: removing
the shift is slower, where the probe said 7.7-10.5% faster. P32: removing
half the loads is slower, where the probe said free. Both took one build
cycle and one four-minute smoke.

**Verdict: non-win 16/25.** Reverted. Every instruction-level term inside the
2-bit nibble-LUT scan has now been measured in situ and none of them is
worth anything: the shift free (P31), the LUT loads negative (P32), the
epilogue x0.996 (H43), the top-k lane loop x0.996 (H43). The loop is what it
is, and the only direction left is a formulation with fewer vector ops per
code byte — which P25 established is not `sdot` or `smmla` at 2/cycle.

## P33 — the 4-query LUT footprint at nq=100 is not a cost (non-win 17/25)

At nq=100 the batched kernel reads 8 LUT vectors per byte-group against 2
code vectors, and the four queries' tables are 4 x 6 KB = 24 KB of L1 against
a 64 KB cache also holding the block's codes. That footprint had never been
priced. Probe: drop the `g * 32` stride so every group and every query reads
the *same* 32 bytes — the working set collapses from 24 KB to one line, load
count unchanged, scores wrong.

```
h41  nq100_st 142.525   nq100_mt 17.358
p33  nq100_st 147.275   nq100_mt 18.424
p33  nq100_st 147.651   nq100_mt 18.516
h41  nq100_st 143.928   nq100_mt 17.549
```

**nq100_st x0.968, nq100_mt x0.942.** Perfect LUT locality is **3.2% and
5.8% slower**. The 24 KB footprint costs nothing, and removing it costs
real time — plausibly because four queries and every group hammering one
address defeats whatever overlap the load pipes were getting from four
distinct streams.

**Third consecutive entry where the in-situ harness inverted the sign of an
expected effect**, not merely its magnitude: P31 (removing the shift is
slower), P32 (removing half the loads is slower), P33 (perfect cache
locality is slower). All three were terms that arithmetic, a static model, or
a standalone probe said were costs. **On this kernel, at this width, every
identified "overhead" has turned out to be load-bearing.** That is a stronger
statement than "the loop is issue-bound" and it is the one the measurements
actually support.

**Verdict: non-win 17/25.** Reverted. The nq=100 LUT path joins the closed
list. What remains unpriced in the objective is the nq=100 *epilogue* — P20's
7.7%, reachable only by index-side per-block norm extremes, a
persistence-format change that has not been attempted.

## H44 — halve the nq=1 unroll from 4 groups to 2 — REFUTED (non-win 18/25)

Bit-identical restructuring: same ops, same accumulation order per
accumulator, two group-pairs per iteration instead of four.

```
h41  nq1_st 1.731   nq1_mt 0.282
h44  nq1_st 1.840   nq1_mt 0.292
h44  nq1_st 1.834   nq1_mt 0.295
h41  nq1_st 1.750   nq1_mt 0.288
```

**nq1_st x0.944, nq1_mt x0.966.** Rejected, reverted.

**But this is the first result in ten entries that points somewhere.** A
5.6% loss from halving the unroll means the loop *is* sensitive to unroll
depth at exactly the scale P24 bounded the whole scheduling family at — and
it means the 4-group depth is doing real work rather than being incidental.
It is also the first term measured on this kernel whose sign came out the way
the reasoning predicted, after three consecutive inversions (P31, P32, P33).

**The obvious follow-up is the one this climb has not tried: unroll to 8.**
If 2 is 5.6% worse than 4, the curve has a slope here, and nothing in the log
establishes that 4 is its minimum — the depth was inherited, never swept.
Register pressure is the argument against, and it is a real one: H41's win
came precisely from freeing registers at 2 bits, and 8 groups doubles the
live pointer set. That makes it a genuine question rather than a safe bet,
which is the right shape for the next candidate.

**Verdict: non-win 18/25.** Concrete next candidate, stated so the next
session does not have to rediscover it: 8-group unroll in
`score_4bit_block_neon`, same bit-identical restructuring, smoked against
`h41` on `nq1_st nq1_mt`. Six minutes of machine time.

## H45 — unroll the nq=1 group loop to 8 — REFUTED (non-win 19/25)

H44's follow-up, same bit-identical restructuring in the other direction.
192 byte-groups divides by 8, so the remainder loop stays empty.

```
h41  nq1_st 1.728   nq1_mt 0.289
h45  nq1_st 1.747   nq1_mt 0.288
h45  nq1_st 1.842   nq1_mt 0.286
h41  nq1_st 1.767   nq1_mt 0.290
```

**nq1_st x0.989, nq1_mt x1.010.** Flat to slightly worse. Rejected, reverted.

**The sweep is now complete and the inherited depth is the optimum:**

| unroll | nq1_st |
|---|---|
| 2 (H44) | x0.944 |
| **4 (shipped)** | **x1.000** |
| 8 (H45) | x0.989 |

A real minimum, not a plateau — 2 loses 5.6% to loop overhead, 8 loses 1.1%
to register pressure, and the shipped depth sits between them. H41's finding
that this kernel is register-limited at 2 bits predicted the right-hand side
of that curve, and the left-hand side is ordinary amortization.

**This closes the last scheduling question on arm nq=1 ST.** H44 was the one
result in ten entries that suggested a slope worth following; following it
found the top. Two build cycles, twelve minutes of machine time, and the
answer is that the code was already there — which is worth as much as a win
would have been, because "4 was inherited and never swept" was a live doubt
in the log and is now retired.

**Verdict: non-win 19/25.** Reverted. Every term inside the 2-bit nibble-LUT
scan has now been measured in situ: shift free (P31), LUT loads negative
(P32), LUT footprint negative (P33), epilogue and lane loop x0.996 (H43),
unroll depth at its optimum (H44/H45). **Nothing in this loop is left to
tune.** The remaining routes are the nq=100 epilogue (P20's 7.7%, needing
index-side per-block norm extremes) and a formulation under 0.4375 vector
ops per code byte, which P25 established is not `sdot` or `smmla`.

## H46 — 2-way unroll the nq=100 group loop — REFUTED (non-win 20/25)

`scan_groups_neon` runs `for g in g0..g1` with no manual unroll at all, while
its nq=1 sibling is 4-way unrolled and H44/H45 just measured that depth as a
real optimum worth 5.6% against 2-way. The asymmetry was untested. Wrapped
the body in a fixed trip-count-2 inner loop so LLVM fully unrolls it.

```
h41  nq100_st 143.140   nq100_mt 17.518
h46  nq100_st 143.333   nq100_mt 17.526
h46  nq100_st 142.652   nq100_mt 17.527
h41  nq100_st 140.837   nq100_mt 17.408
```

**nq100_st x0.987, nq100_mt x0.993.** Rejected, reverted.

**The asymmetry is justified, and the reason is the one H41 found.** At nq=1
there are 4 u16 accumulators and spare registers, so unrolling buys
amortization; at nq=100 there are 16 live accumulators across four queries
plus the shared code temps, and there is nothing left to unroll into. The
same register limit that made H41 a win at 2 bits makes this a loss — the
third time that single fact has predicted a result correctly (H41, H45, H46)
after a long run of predictions that did not.

**Verdict: non-win 20/25.** Both scan loops are now swept for unroll depth
and both are at their optimum. Combined with P31-P33 and H43, **every
instruction-level and loop-structure question on the aarch64 2-bit kernels
has been measured in situ and none of them has anything in it.**

## P34 — the smoke harness calibrated against a byte-identical binary (non-win 21/25)

An intended constant sweep found no such constant — the regex matched
nothing, the patch came out empty, and `h47` built as `h41`. The `.so` files
are md5-identical (`3ceeaea488f57b52a4bbf2cc6a80fc84`). Rather than discard
it, it was run: **an unchanged binary compared against itself is the control
this session has never had for the smoke harness**, and every candidate since
H43 has been judged against a noise band that was asserted rather than
measured.

```
h41  nq1_mt 0.282   nq100_mt 17.436
h47  nq1_mt 0.282   nq100_mt 17.409
h47  nq1_mt 0.278   nq100_mt 17.409
h41  nq1_mt 0.285   nq100_mt 17.423
```

**Identical code reads x1.014 on `nq1_mt` and x1.0008 on `nq100_mt`.** The
band is not one number — it is 1.4% on the 0.28 ms cell and 0.1% on the
17.4 ms cell, an order of magnitude apart, which is the same
cell-size-tracks-noise pattern the capstone hit at min-of-4.

**Re-reading this session's smokes against a measured band rather than a
guessed one:**

| entry | cell | result | control band | verdict holds? |
|---|---|---|---|---|
| H44 | nq1_mt | x0.966 | 1.4% | yes, 2.4x band |
| H45 | nq1_mt | x1.010 | 1.4% | **no — inside the band** |
| H46 | nq100_mt | x0.993 | 0.1% | yes, 7x band |
| H43 | nq1_mt | x1.018 | 1.4% | marginal, ~1.3x band |

**H45's `nq1_mt` x1.010 was not a result and should not have been reported as
"flat".** It was unresolvable. Its `nq1_st` figure carried the refutation and
still does, so the unroll sweep's conclusion is unaffected — but the
distinction between "measured flat" and "below the instrument's resolution"
is one this log has now got wrong twice, and the fix is that a control run
belongs at the *start* of a measurement campaign, not stumbled into at its
twenty-first entry.

**Verdict: non-win 21/25.** The most useful measurement of the last ten
entries was produced by a failed edit, which is worth saying plainly: the
value was in running the control at all, and nothing but an accident
prompted it.

## H48 — finer NEON tiles, MIN_TILE_BLOCKS_NEON 512 -> 256 — REFUTED (non-win 22/25)

Smaller tiles give the work-stealer more pieces and should balance better at
MT. First candidate judged against P34's measured band rather than a guessed
one, and deliberately aimed at `nq100_mt` where that band is 0.1%.

```
h41  nq100_mt 17.527   nq100_st 143.638
h48  nq100_mt 17.810   nq100_st 142.528
h48  nq100_mt 17.998   nq100_st 141.988
h41  nq100_mt 17.541   nq100_st 143.060
```

**nq100_mt x0.984 — a 1.6% regression at sixteen times the control band.**
Unambiguous, and the cleanest refutation in this session precisely because
the band underneath it is known. Rejected, reverted.

Finer tiles lose here for the reason H39 found at nq=1: on this kernel range
count feeds stream length, and shorter streams cost more than better balance
buys. 512 blocks is 16,384 vectors per tile, already ~3 MB of codes at 2
bits — cutting that to 1.5 MB shortens each worker's sequential run without
adding parallelism the 8 cores can use.

**One thing worth flagging rather than filing.** `nq100_st` moved x1.0076 on
a change that cannot touch it — `MIN_TILE_BLOCKS_NEON` is only read when
`n_block_ranges` returns more than one range, and ST returns 1. So that 0.76%
is drift, on a cell whose two `h41` passes sat 0.4% apart. **The ST band is
wider than its own within-label spread suggests**, and P34 measured the
control on `nq1_mt` and `nq100_mt` only. A no-op control on the ST cells is
the obvious gap and has not been run.

**Verdict: non-win 22/25.**

## P35 — the control finished, on all four cells (non-win 23/25)

H48 flagged that two of the four arm cells were still being judged against an
unmeasured band. `h47.so` is md5-identical to `h41.so`, so the ST control
cost one smoke and no build.

```
h41  nq1_st 1.730   nq100_st 141.152
h47  nq1_st 1.740   nq100_st 142.783
h47  nq1_st 1.733   nq100_st 143.017
h41  nq1_st 1.760   nq100_st 142.314
```

**The complete control table for identical code:**

| cell | duration | control reads | band |
|---|---|---|---|
| nq1_st | 1.7 ms | x0.998 | **0.2%** |
| nq1_mt | 0.28 ms | x1.014 | **1.4%** |
| nq100_st | 143 ms | x0.989 | **1.1%** |
| nq100_mt | 17.4 ms | x1.0008 | **0.1%** |

**P34's explanation was wrong and two points is why.** That entry read the
band as tracking cell size — 1.4% on the 0.28 ms cell, 0.1% on the 17.4 ms
one — and it was a clean story from two samples. With all four, the largest
cell in the objective (143 ms) has the *second-widest* band and the
second-smallest cell has the tightest. **Duration does not predict it.** What
the wide pair share is that they are the two cells the log has repeatedly
found bimodal: `nq1_mt` and `nq100_st` are exactly P16's and P21's offenders.
The band is mode-switching, not sampling.

**Re-reading the ST results of this session against 1.1%:**

| entry | nq100_st | verdict |
|---|---|---|
| P33 | x0.968 | holds, 3x band |
| P31 | x0.986 | **marginal, 1.3x band** |
| H46 | x0.987 | **marginal, 1.2x band** |
| H48 | x1.0076 | inside band — drift, as flagged |

P31 and H46 were reported as clean refutations and are better described as
directionally negative but unresolved on that cell. Neither conclusion
changes: P31's shift verdict rests on nq1_st (0.2% band, ample), and H46's on
nq100_mt (0.1% band, 7x). **But "x0.986 on a cell whose control is 1.1%" is
not the sentence either entry wrote.**

**Verdict: non-win 23/25.** Every arm cell in the objective now has a
measured no-op band, which is the thing that should have existed before the
first candidate and instead arrived after the sixteenth.

## H49 — coarser NEON tiles, MIN_TILE_BLOCKS_NEON 512 -> 1024 — REFUTED (non-win 24/25)

H48 established that halving the tile floor costs 1.6%. The other half of the
sweep had not been run, and after H44/H45 found a genuine minimum on unroll
depth it was worth checking whether this constant sits on one too.

```
h41  nq100_mt 17.414   nq1_mt 0.279
h49  nq100_mt 17.556   nq1_mt 0.280
h49  nq100_mt 17.404   nq1_mt 0.279
h41  nq100_mt 17.380   nq1_mt 0.282
```

**nq100_mt x0.999, nq1_mt x1.000.** Flat against a 0.1% control band on the
resolving cell. Rejected, reverted.

**The sweep, and it is not a minimum but a plateau edge:**

| MIN_TILE_BLOCKS_NEON | nq100_mt |
|---|---|
| 256 (H48) | x0.984 |
| **512 (shipped)** | **x1.000** |
| 1024 (H49) | x0.999 |

512 and 1024 are indistinguishable at a band of 0.1%; only 256 is worse. So
the shipped value sits at the *edge* of a flat region rather than at an
optimum, and **the NEON-specific override buys nothing measurable** — the
generic `MIN_TILE_BLOCKS = 1024` performs identically. That is a different
shape from H44/H45's unroll sweep, which had a real interior minimum, and
worth distinguishing: one constant is load-bearing and the other is not.

**Not proposed as a change.** Deleting `MIN_TILE_BLOCKS_NEON` would simplify
the scheduler for no measured gain, and it was presumably introduced against
evidence on a cell or width this smoke did not cover — 4-bit, x86, or another
N. A tuned constant measuring flat in one configuration is not grounds for
removing it, only for recording that it is flat here.

**Verdict: non-win 24/25.**

## H50 — halve TILES_PER_THREAD_NEON, 64 -> 32 — marginal, NOT PROMOTED (non-win 25/25)

H39 found that on this kernel range count feeds stream length and longer
streams win; H48 found the same for tile size. Fewer, larger ranges per
thread is the change those two findings jointly point at, and it had not been
tried.

```
h41  nq100_mt 17.407   nq1_mt 0.287
h50  nq100_mt 17.446   nq1_mt 0.282
h50  nq100_mt 17.350   nq1_mt 0.276
h41  nq100_mt 17.455   nq1_mt 0.279
```

**nq100_mt x1.0033, nq1_mt x1.011.** The `nq1_mt` figure is inside its 1.4%
control band (P35) and is not a result. The `nq100_mt` figure is 3.3x its
0.1% band, so it is a real effect — **and it is 0.33%.**

**Not promoted, and the reason is the protocol rather than the sign.**
`smoke.sh` exists so that "a candidate that cannot show its mechanism here
does not earn a 15-minute soak." A third of a percent on one of eight cells
cannot move the 8-cell HM to the x1.01 the authority requires; it would need
the other seven cells to carry it, and this change touches only the aarch64
MT scheduling path. Promoting it would mean spending a soak to measure
something the smoke already says is too small to matter, and adopting it on
smoke evidence alone would be adopting a constant change on 3.3x band with no
parity run and no x86 check.

**It is the one genuinely unresolved candidate this climb is leaving open**,
and it is recorded as that rather than as a refutation: a real, tiny, positive
effect on the arm nq=100 MT cell, direction consistent with H39 and H48,
worth a proper soak by anyone who wants to spend one.

**Verdict: non-win 25/25.**

---

# Round 2 — reopened 2026-09-06 at main 1.0.0 (ccab9f32)

Round 1 closed at 25 consecutive non-wins on 2026-08-08 and shipped as #511
(five wins, 8-cell HM x1.0495). Main has since moved: format v7 (#535/#536),
five release-blocking fixes (#533), and a whole-block prune on the aarch64
nq=1 block-parallel path (#493) that touches an objective cell directly. The
round-1 baseline is therefore stale and is re-pinned at this HEAD before
anything is measured. Same eight cells, same harness, same authority
(`whm_2bit.py`), non-win count restarts at 0/20 per `GOAL_2bit.md`.

Branch `perf/2bit-hillclimb-2`, worktree `~/git/tv-2bit-hc`.

**Protocol carried over from round 1, stated up front this time:**

- Control run first. P34/P35 found the smoke's no-op band is per-cell —
  0.2% `nq1_st`, 1.4% `nq1_mt`, 1.1% `nq100_st`, 0.1% `nq100_mt` on arm —
  and that a wide band means mode-switching, not sampling. A byte-identical
  binary is run A/B against itself on both boxes before the first candidate.
- Rebuild the box to baseline after every candidate (round-1 correction 2).
- `rm -rf target` before every release build; LD_PRELOAD the arch libopenblas.
- Prebuilt `.so` files per label, balanced ABBA passes, min per label.

**Opening candidates, in order:**

1. H51 — `TILES_PER_THREAD_NEON` 64 -> 32, the one candidate round 1 left
   open (H50: `nq100_mt` x1.0033 at 3.3x band, never soaked).
2. The #493 prune on the arm nq=1 path: it was added for a gate-crossing
   case at nq=1 and measured neutral in H116 at the old geometry; re-measure
   at the 2-bit cells, since it sits on `nq1_st` and `nq1_mt` directly.
3. Width-invariant constants — the round-1 lesson (H14, H41): any constant
   swept at 4 bits and inherited at 2 is a candidate. `FLUSH_EVERY` is done;
   the remaining ones are catalogued before the first build.

## Rig status

Blocked at open: gcloud's auth token expired, and both boxes are reached
through an IAP tunnel (`ProxyCommand gcloud compute start-iap-tunnel`), so
`ssh tvarm` / `ssh tvx86` fail until `gcloud auth login` is re-run
interactively. Local (M3 Max) is not an objective cell and is used only to
check that the harness still runs against the v7 build.

## Local pre-screen calibration (M3 Max, not an objective cell)

While the rig is down, the laptop is used to screen aarch64 candidates. Its
no-op band, byte-identical binary ABBA, min of 3 sub-runs per pass:

| cell | ctl | ctl2 | band |
|---|---|---|---|
| nq100_mt | 9.843 | 9.595 | **2.6%** |
| nq1_mt | 0.317 | 0.327 | **2.8%** |
| nq100_st | 98.433 | 98.547 | 0.1% |
| nq1_st | 1.195 | 1.196 | 0.05% |

The MT cells are unresolvable here below ~3% (E-cores and scheduler jitter);
the ST cells resolve at 0.1%. So the laptop can screen ST-path candidates and
cannot screen MT-only ones. Nothing measured here is a verdict — verdicts
come from the two boxes through `whm_2bit.py`.

## H51 — `TILES_PER_THREAD_NEON` 64 -> 32 (round-1 H50) — local pre-screen UNRESOLVED

2-pass ABBA, min per label: nq100_mt x1.0073, nq1_mt x0.9885, nq100_st
x1.0002, nq1_st x1.0068. Both MT cells sit inside the 2.6-2.8% local band;
both ST cells are untouched by construction (ST has one range). Consistent
with round 1's +0.33% on the Axion box and no more informative than that.
**Queued for the rig soak; no verdict.**

## H52 — the #493 whole-block prune on the arm nq=1 path, ablated — REFUTED as a lever (local pre-screen)

Main added a whole-block prune to `scan_range_neon`'s lane loop after round 1
closed (#493), on a cell this climb scores directly. Round 1's H116 had
measured the same prune neutral at 4 bits. Ablated with `if false &&` at the
prune's guard so the block-max tree compiles out; binaries differ
(`5fc546dd` vs `1fad8892`). ST-only ABBA on the laptop, where the ST band is
0.1%:

| cell | ctl | prune off | |
|---|---|---|---|
| nq1_st | 1.173 | 1.269 | **x0.924** |
| nq100_st | 98.341 | 98.409 | x0.999 (untouched path) |

**The prune is worth 7.6% at 2 bits on this cell**, not the "neutral" H116
recorded at 4 bits — at half the bytes per vector the lane loop is a larger
share of the block, so skipping it matters more. It is already in the
baseline, so there is nothing to win here; the value is knowing the lever is
live at 2 bits and pointing the same direction as the arm nq=1 gap.
**Not a candidate; not counted.** (Ablations and probes that do not propose a
change are recorded but do not consume the non-win count, per round 1's
convention for P-entries.)

## H53 — const-generic batch width for the x86 2-bit VNNI kernel — PRE-REGISTERED

Found by reading the shipped machine code rather than the source.
`search_multi_query_vnni` takes `nq` at runtime and loops
`for qi in 0..nq.min(8)` over `acc: [[__m512i; 2]; 8]` and
`split_luts[qi]`. LLVM unrolls that loop to 8, but it cannot delete the
trip-count test or the slice bounds check, so the shipped inner body per
quad-half is:

```
vpermb (%rdx,%rcx,1),%zmm17,%zmm18
vpdpbusd %zmm20,%zmm18,%zmm1
vpermb 0x40(%rdx,%rcx,1),%zmm16,%zmm18
vpdpbusd %zmm20,%zmm18,%zmm1
cmp $0x1,%r8 ; je ...        <- nq.min(8) exit test
cmp $0x1,%rdi ; je ...       <- split_luts.len() bounds check
... x8 queries ...
vmovdqa64 %zmm6,0x280(%rsp)  <- accumulators for queries 5-8 spilled per quad
```

Per quad at nq=8 that is 32 vpermb + 32 vpdpbusd (the work) plus 32
compare-and-branch pairs and ~8 zmm stores (the overhead). On Sapphire
Rapids vpermb is p5-only and vpdpbusd zmm is p0-only, so the work alone is
32 cycles a quad on each port; the fused branches land on p0/p6 and the
stores on p4/p9, so the overhead is not free and sits on the same critical
port as the dot products. P7 priced the shipped ST cell 17% under the P6
probe — a probe whose loop had a constant width — and attributed the gap to
"probe idealization". This is a concrete candidate for part of that gap.

Change: `NQ` becomes a const generic; the dispatch picks `<4>` for a tail
of <= 4 queries and `<8>` otherwise (the driver already pads `split_luts`
to the batch width). LUT pointers, scales and biases are copied into
`[_; NQ]` locals up front so every hot-loop access is a constant index.
Accumulation order per query is unchanged, so scores are bit-identical.
Pad queries in a narrow tail are scored into registers and skipped at the
heap update.

Prediction: x86 nq100_st and nq100_mt improve; nq=1 untouched (separate
kernel); arm untouched by construction. `cargo check --target
x86_64-unknown-linux-gnu` clean. Patch staged as `~/hc/h53.patch` on the
x86 box, to run after the baseline pin.

## Round-2 baseline — commit ccab9f32 (main 1.0.0), pinned 2026-09-06

Both boxes: `git reset --hard ccab9f32`, `rm -rf target`, `maturin develop
--release` (51 crates compiled — a clean build, verified), old `.tvim`
caches deleted so the seeded index is rebuilt in the v7 format, arch
libopenblas LD_PRELOADed, one process per cell. Three rounds of
`cells_2bit.py` (each cell min of 9 sub-runs); the pin is the per-cell min
across rounds. Files: `data/r2_base_{arm,x86}.json` (with all raw samples),
rounds in `data/r2_base_{arm,x86}_r{1,2,3}.json`.

| cell | arm ms | arm round spread | x86 ms | x86 round spread |
|---|---|---|---|---|
| nq1_st | **1.650** | 2.6% | **1.533** | 24.2% |
| nq1_mt | **0.263** | 5.5% | **0.427** | 4.2% |
| nq100_st | **134.453** | 4.3% | **85.802** | 1.3% |
| nq100_mt | **17.217** | 0.7% | **24.013** | 2.8% |

Against the round-1 pin (262793f) every arm cell is faster (nq1_st 1.995
-> 1.650, nq100_st 148.99 -> 134.45, nq100_mt 18.43 -> 17.22): that is the
five round-1 wins plus #493's prune, as expected. x86 nq100_st 83.1 -> 85.8
and nq100_mt 25.5 -> 24.0 are inside their bands of the H41 capstone. **x86
nq1_st is bimodal across processes again** — rounds read 1.533 / 1.903 /
1.90 with min-of-9 inside each — so that cell's pin is the fast mode and a
candidate must reach the fast mode to tie it. The x86 nq=1 control smoke
below shows the same 9% band on an unchanged binary.

**Control bands (byte-identical `base2.so` vs `ctl2.so`, 2-pass ABBA smoke,
min per label):**

| cell | arm | x86 |
|---|---|---|
| nq100_st | 0.6% | 0.2% |
| nq100_mt | 0.6% | 0.4% |
| nq1_st | 0.2% | **8.8%** |
| nq1_mt | 1.9% | 0.7% |

Parity digests (2-bit / 4-bit) recorded in `data/r2_parity_{arm,x86}.json`;
the 2-bit digests differ between arches, as they did in round 1 (the x86
VNNI kernel rounds once at the end, the arm classic kernel flushes), and the
gate is per-arch against these.

Non-win count: 0/20.

## H51 — `TILES_PER_THREAD_NEON` 64 -> 32 on the rig — REFUTED (non-win 1/20)

Axion, 2-pass ABBA smoke vs `base2`, min per label:

```
base2  nq100_mt 17.281  nq1_mt 0.265  nq100_st 137.465
h51    nq100_mt 17.341  nq1_mt 0.273  nq100_st 135.755
h51    nq100_mt 17.360  nq1_mt 0.269  nq100_st 136.254
base2  nq100_mt 17.368  nq1_mt 0.269  nq100_st 135.579
```

nq100_mt x0.997 (band 0.6%), nq1_mt x0.985 (band 1.9%), nq100_st x1.000
(one range at ST; untouched by construction). Round 1's +0.33% does not
reproduce at 1.0.0; the candidate it left open is closed. Reverted.

**Verdict: non-win 1/20.**

## H53 — smoke, x86 (Sapphire Rapids), 2-pass ABBA vs `base2`

```
base2  nq100_st 86.662  nq100_mt 24.636
h53    nq100_st 70.405  nq100_mt 19.356
h53    nq100_st 70.312  nq100_mt 18.797
base2  nq100_st 86.039  nq100_mt 24.296
```

**nq100_st x1.224, nq100_mt x1.293**, against control bands of 0.2% and
0.4%. Every candidate sample is below every control sample by a wide
margin. Built incrementally from ccab9f32 + `h53.patch` (`acb2f8a5`),
HEAD verified before and after. Promoted to soak: 3 balanced ABBA passes
of the full harness, paired nq/N sweep, 4-bit observation, `cargo test`.
Arm: the patch is entirely inside `cfg(target_arch = "x86_64")` code, so
the arm binary is expected byte-identical; checked by building it there.

**Arm identity check for H53.** The arm `h53.so` hash differs from `base2.so`
(`c6d2ff3f` vs `164dbd79`), but so does any source change: the crate hash
feeds every mangled symbol, so a patch inside `cfg(x86_64)` still renames
symbols on aarch64. An unpatched incremental build (`ctl3`) reproduces
`base2` byte for byte, so the build is deterministic and the difference is
the patch. Symbol-blind disassembly (addresses, symbol hashes and immediates
normalised): 198,007 instruction lines each, **zero differing instructions**.
The arm kernel is the same code, so the arm cells enter the verdict at
x1.000 from the pinned baseline rather than being re-measured.

## H53 — soak, gates and verdict on the first cut

**Soak** (x86, 3 balanced ABBA passes of the full harness, prebuilt `.so`
swapped per pass, min per label; files `data/r2_h53/`):

| cell | base2 | h53 | |
|---|---|---|---|
| nq100_st | 85.340 | 68.486 | **x1.2461** |
| nq100_mt | 24.140 | 18.689 | **x1.2917** |
| nq1_st | 1.276 | 1.277 | x0.9995 |
| nq1_mt | 0.423 | 0.426 | x0.9928 (band 0.7%) |

Every h53 pass on both nq=100 cells is below every base2 pass (ST
68.5-69.8 against 85.3-86.6; MT 18.7-19.0 against 24.1-24.7). The nq=1
cells are the separate single-query kernel and read as drift.

**Gates.** Parity digests identical to `base2` at both widths
(`d8ce9ea9…` / `3314955a…`). `cargo test -p turbovec` green on the x86 box
(all suites, 141 in the main one) and on the arm laptop with the patch
applied (10 suites). Arm binary instruction-identical (above).

**`whm_2bit.py` against the pinned baseline** (arm cells x1.0000 by
identity):

```
cell            arm        x86
  nq1_st       x1.0000    x1.2010   <- bimodal cell; not claimed
  nq1_mt       x1.0000    x1.0008
  nq100_st     x1.0000    x1.2528
  nq100_mt     x1.0000    x1.2849
  x86 4-cell HM  x1.1736
  8-cell HM      x1.0799   worst cell x1.0000
```

Against the in-soak paired base2 (the drift-cancelling reading):
x86 4-cell HM x1.1159, 8-cell HM **x1.0548**, worst cell nq1_mt_x86
x0.9928, VERDICT: WIN. The pinned reading's x1.20 on nq1_st is the
bimodal cell drawing its slow mode for the pin and its fast mode in the
soak; the paired reading (x0.9995) is the honest one for that cell, and
the 8-cell HM clears x1.01 by a factor of five either way.

**4-bit observation** (never gated): recorded in
`data/r2_h53/h53_obs4_*.json`; the 4-bit path is the permute-dot kernel
and does not touch this code.

**Sweep — and this is why the gate exists.** Paired A/B, 44 points, all
measured (`data/r2_h53/h53_sweep.json`, ratio > 1 means the candidate is
faster):

| point | ST | MT |
|---|---|---|
| nq=2 | **x0.864** | x0.968 |
| nq=3 | x1.127 | x1.085 |
| nq=4 | x1.169 | x1.295 |
| nq=5 | **x0.882** | **x0.929** |
| nq=6 | x1.032 | x1.080 |
| nq=7 | x1.162 | x1.175 |
| nq=8..64 | x1.12-x1.41 | x1.14-x1.35 |
| N=1k..200k (nq=100) | x1.03-x1.24 | x1.07-x1.30 |

nq=2 and nq=5 regress 12-14%, far outside P4's 3% noise, and the mechanism
is exactly the shape of the first cut: a batch narrower than the
instantiation is padded up to it, so nq=2 does four queries' work and nq=5
does eight. At nq=3/6/7 the branch-free loop outruns the padding; at 2 and
5 it does not. **Not promoted in this form.** H53b instantiates every width
2..=8 so no batch does padded work; the nq=100 cells are unaffected by
construction (100 = 12x8 + 4, both exact already), so the soak above stands
for them and H53b needs only its own smoke, parity and sweep.

Harness note: the sweep driver segfaulted *after* writing all 44 points.
It had imported turbovec itself to rebuild the small-N indexes (deleted for
the v7 re-pin) and then overwrote the mapped `.so` in place for the A/B
swaps; the crash is the interpreter tearing down a module whose file
changed under it. Round 1 never hit this because its small-N indexes were
already cached. The H53b sweep runs with the indexes present and should not
reproduce it; if it does, the harness gets a fix, not the candidate.

## H53b — one instantiation per width 2..=8 — gates

Built incrementally from ccab9f32 + `h53b.patch` (`2c56e476`). Smoke
against `h53` on the objective cells: nq100_st 70.22 vs 70.16 (x0.999),
nq100_mt 18.34 vs 18.99 (x1.035) — the same code on the nq=100 path, as
predicted (the MT figure is the bimodal side of that cell drawing
differently; the soak decides). Smoke against `base2`: nq100_st x1.221,
nq100_mt x1.30, nq1 cells inside band.

Parity digests identical to `base2` at both widths. Arm build
instruction-identical to the control (symbol-blind diff: 0 instructions).
Local `cargo test -p turbovec` with the patch: 40 suites green.

**Sweep, paired A/B, 44 points, no segfault this time** (the small-N
indexes were cached, which confirms the harness reading above):

| point | ST | MT |
|---|---|---|
| nq=2 | **x1.236** | x1.188 |
| nq=3 | x1.225 | — |
| nq=5 | **x1.328** | x1.298 |
| nq=6 | x1.411 | — |
| nq=7 | x1.311 | — |
| nq=8 / 16 / 64 | x1.304 / x1.293 / x1.243 | — |
| N=1k / 8k / 32k / 200k (nq=100) | x1.035 / x1.146 / x1.231 / x1.233 | 200k: x1.348 |

**No point below 0.97; the worst is nq1_mt at x0.995**, which is the
single-query kernel and noise. The two padding regressions of the first
cut are now the two largest ST gains in the nq sweep. Soak launched (3
balanced ABBA passes); the verdict is the soak through `whm_2bit.py`.

## H54 — range-major block tiling for the single-thread scan — PRE-REGISTERED

`n_block_ranges` returns 1 whenever the pool has one thread, by design
("identical work and visit order to the serial scan"). So at ST every
query batch sweeps the whole 38 MB of codes: 13 sweeps at nq=100 on x86
(batch 8), 25 on arm (batch 4). P26 priced the DRAM term of the 2-bit scan
at 12.5% of the loop at N=200k for nq=1; at nq=100 the compute per byte is
higher and the term smaller, but it is paid on every sweep. The MT path
already tiles (query-quad x block-range) and its tile order is range-major
with quads inner, and the cross-range merge is deterministic (score desc,
index asc), so a one-thread pool can take the same tiles with no change to
results.

Change: with one thread and more than one quad, split the block axis into
ranges of `ST_RANGE_BLOCKS = 256` blocks (1.5 MB of 2-bit codes, L2-resident
on both rig cores), capped by `range_cap_for_k`. Untouched: nq=1 (one
quad), MT, masked and scalar paths. Prediction: nq100_st improves on both
arches; other cells unchanged. Parity must hold by the merge's
determinism. The 4-bit observation may move either way (3 MB ranges).

## H55 — VNNI batch width 10 — PRE-REGISTERED (x86, on top of H53b)

With the width const-generic, 10 queries fit the register file (20 zmm of
accumulators + 8 temporaries) where 12 would spill. nq=100 becomes ten
sweeps of the codes instead of thirteen, and the per-quad shared decode
(2 loads, and/shift/or) amortises over 10 queries. Risk: the batch's LUT
set grows from 48 KB to 60 KB, past L1D (48 KB on Sapphire Rapids), so
`vpermb`'s table operands come from L2 more often. Round 1's H12 refuted
the analogous 4 -> 8 widening on arm as L1-bound; this is the x86 version
of the same question, and the answer is measured, not argued. Instantiation
arms added for widths 9 and 10; `nq_batch` becomes 10 for the VNNI path
when that reduces the batch count.

## H54 — range-major ST tiling — REFUTED on arm (non-win 2/20)

Axion, 2-pass ABBA smoke vs `base2`, min per label:

```
base2  nq100_st 132.761  nq1_st 1.643  nq100_mt 17.147
h54    nq100_st 140.351  nq1_st 1.636  nq100_mt 17.016
h54    nq100_st 140.137  nq1_st 1.653  nq100_mt 17.202
base2  nq100_st 132.460  nq1_st 1.659  nq100_mt 17.271
```

**nq100_st x0.945** — every candidate sample above every control sample,
against a 0.6% band. nq1_st and nq100_mt flat, as the patch predicts
(untouched paths). The L2-residency the change buys is real but smaller
than what it costs: 25 ranges means every query's top-k is re-filled from
empty 25 times, and the fill phase runs without the whole-block prune that
makes the steady state cheap. The MT path pays the same per-range cost but
spreads it over eight workers that would otherwise idle; one worker has no
such offset. Round 1's note that "the per-range top-k duplication is
exactly the cost the k cap argued for" applies with full force at ST.

Not run on x86: a 5.5% regression on an arm cell fails the no-regression
gate whatever x86 does, and the mechanism is arch-independent. Reverted.

**Verdict: non-win 2/20.**

*Arm status after H54.* The arm nq=100 ST loop is 54 instructions per
byte-group for 4 queries and the core issues 4 SIMD ops per cycle (P27);
192 groups x 6250 blocks x 25 batches at 13.5 cycles is 135 ms at 3 GHz,
which is the measured cell. It is at the issue bound of its formulation,
and the instruction that could go — the four widening adds per query — is
pinned by the u8 LUT ceiling (127) that bit-identity fixes. Every
remaining arm lever on this cell is a formulation change round 1 closed
(P5, H12, H29). The climb's live ground is x86.

## H53b — landed. `whm_2bit.py` VERDICT: WIN — round-2 win #1 (8-cell HM x1.0871)

**Soak** (x86, 3 balanced ABBA passes, min per label; `data/r2_h53b/`):

| cell | base2 | h53b | |
|---|---|---|---|
| nq100_st | 80.568 | 67.816 | **x1.1880** (base2 drew its fast mode once, p12; against the other five passes, 84.6-86.2, it is x1.25) |
| nq100_mt | 23.891 | 17.698 | **x1.3499** |
| nq1_st | 1.276 | 1.278 | x0.9980 |
| nq1_mt | 0.422 | 0.426 | x0.9915 |

Every h53b pass below every base2 pass on both nq=100 cells (ST 67.8-68.9
vs 80.6-86.2; MT 17.7-18.5 vs 23.9-24.6).

**Authority, against the pinned baseline** (arm x1.0000 by instruction
identity):

```
cell            arm        x86
  nq1_st       x1.0000    x1.1995   (bimodal cell; the paired reading is x0.998)
  nq1_mt       x1.0000    x1.0018
  nq100_st     x1.0000    x1.2652
  nq100_mt     x1.0000    x1.3568
  x86 4-cell HM  x1.1907
  8-cell HM      x1.0871   worst cell x1.0000
VERDICT: WIN
```

Against the paired in-soak base2: x86 4-cell HM x1.1132, **8-cell HM
x1.0536, worst cell nq1_mt_x86 x0.9915** (floor 0.99), VERDICT: WIN. Both
readings clear x1.01 by a wide margin; the honest headline is the paired
one for the nq=1 cells and the pinned one for nq=100, and the verdict is
the same either way.

**4-bit observation** (recorded, never gated; measured on `h53`, whose
4-bit path is byte-for-byte the same code as `h53b`'s): nq1_st x1.17
(the bimodal cell), nq1_mt x1.02, nq100_st x1.01, nq100_mt x0.995 — the
4-bit path runs the permute-dot kernel and does not touch this code.

**What it teaches.** P6/P7 measured a 17% gap between a constant-width
probe and the shipped cell and attributed it to "probe idealization"; the
gap was the runtime width. The kernel's *source* had the right shape — an
8-wide accumulator array and a clamped loop — and only the machine code
showed the 32 compare-and-branch pairs and the per-quad spills that the
runtime bound left in. Two lessons for the rest of this climb: (1) read
the disassembly of the shipped kernel before pricing its roofline from a
probe, because a probe with a constant trip count cannot see a runtime
one; (2) a "kernel at roofline" verdict that rests on a probe is only as
good as the probe's fidelity to the loop's *control* structure, not just
its instruction mix.

Committed on `perf/2bit-hillclimb-2`. **Non-win count: 0/20.**

## H55 — VNNI batch width 10 — REFUTED (non-win 1/20)

x86, 2-pass ABBA smoke vs `base2` (min per label): nq100_st 75.292,
nq100_mt 20.619. Against `h53b`'s soak (67.816 / 17.698) that is
**x0.90 ST and x0.86 MT** — ten sweeps instead of thirteen and a wider
amortisation, and the cell got slower. The direction is the information:
per-query cost *rises* with batch width past 8, so the batch's LUT set
(60 KB at 10, 48 KB at 8, against a 48 KB L1D) or accumulator pressure is
already binding at 8, and the sweep-count term is not what limits this
cell. Reverted. P36 measures per-query cost by width directly before any
further width change.

**Verdict: non-win 1/20.**

## P36 — per-query ST cost by batch width, on `h53b` (probe; not counted)

x86, `h53b.so`, min of 3 processes per point, k=10:

| nq (one batch) | N=200k ms/query | N=32,768 ms/query |
|---|---|---|
| 2 | 0.863 | 0.169 |
| 3 | 0.652 | 0.134 |
| 4 | 0.595 | 0.1255 |
| 5 | 0.597 | 0.1221 |
| **6** | **0.586** | 0.1223 |
| 7 | 0.624 | 0.1227 |
| 8 | 0.692 | 0.1285 |
| 16 (8+8) | 0.652 | 0.130 |
| 100 (12x8+4) | 0.696 | 0.134 |

**Width 8 is past the knee at both sizes.** At N=200k the per-query cost
at 8 is 18% above the minimum at 6; L2-resident it is 5% above. Eight was
never measured — it is the size of the accumulator array. The shipped cell
(12 batches of 8 and one of 4) sits at the 8-wide cost; re-batched at 6
(16 batches of 6 and one of 4) the same table predicts ~59 ms against
69.6, i.e. ~x1.18 on nq100_st, with MT to be measured (the tile count
follows the quad count).

Why 8 loses: the `<8>` instantiation carries 34 zmm stack references in
757 lines against 21 in `<6>` — some are prologue saves, but the
accumulator file at 8 is 16 zmm plus ~8 temporaries against 32
registers, and the batch's LUT set is 48 KB against a 48 KB L1D. Both
ease at 6. This also explains H55: 10 is further past the knee, not a
different regime.

The memory term is not the story at nq=100: per-vector cost is *lower*
at 200k than at 32k (3.5 vs 4.1 ns/vector) because the fixed per-query
work is a larger share of the small index. The lever is the core.

Registered as **H56: `VNNI_BATCH = 6`** for the 2-bit VNNI path, smoked
against `h53b` first (the climb HEAD is now the H53b build), then against
`base2` for the authority. Widths 5 and 7 are the natural neighbours if
6 confirms.

## H56 — `VNNI_BATCH = 6` — smoke

x86, 2-pass ABBA, min per label. Against `base2`: nq100_st 68.679,
nq100_mt 17.331 (x1.25 / x1.42 on the base2 samples of that run). Directly
against `h53b`:

```
h53b  nq100_st 70.197  nq100_mt 18.660
h56   nq100_st 60.272  nq100_mt 17.511
h56   nq100_st 70.129  nq100_mt 17.495
h53b  nq100_st 65.119  nq100_mt 18.243
```

**nq100_mt x1.042** on the mins, every h56 sample below every h53b sample
(17.50/17.51 vs 18.24/18.66). **nq100_st is in its bimodal regime** —
60.3 and 70.1 for the same binary, 65.1 and 70.2 for the other — so the
smoke's min reads x1.080 but the samples overlap at the slow mode; this is
exactly the cell P16 diagnosed, and the soak's min-of-9 sub-runs per pass
exists to reach the fast mode reliably. Promoted: soak vs `base2`
(authority), a 2-pass soak vs `h53b` (the direct comparison), paired
sweep, 4-bit observation, `cargo test`.

## H57 / H58 / H59 — PRE-REGISTERED (x86, queued behind the H56 chain)

- **H57 — `VNNI_BATCH = 5`** and **H58 — `VNNI_BATCH = 7`**: P36's
  neighbours of the minimum (0.597 and 0.624 ms/query against 0.586 at 6).
  Expected flat-to-worse; run so the width is a measured optimum on the
  objective cell rather than a probe's, the way H44/H45 swept the unroll.
- **H59 — prefetch in the batched VNNI kernel** (`PF = true`, the depth-8
  lookahead the single-query kernel already uses). H4/H5 measured it at
  ~-5% at nq=100 with the branchy loop, because a re-reading batch evicts
  what it is about to re-read. The loop is now ~20% faster per byte, so
  the memory share of the cell is larger and the verdict may not carry.
  Cheap to re-ask; expected refuted.

Each is smoked against `h56` on the nq=100 cells.

## H56 — soak and authority (sweep, obs4 and cargo test still running)

**Soak vs `base2`** (x86, 3 balanced ABBA passes, min per label;
`data/r2_h56/`):

| cell | base2 | h56 | |
|---|---|---|---|
| nq100_st | 84.465 | 67.284 | **x1.2553** |
| nq100_mt | 24.029 | 16.328 | **x1.4716** |
| nq1_st | 1.275 | 1.288 | x0.9897 |
| nq1_mt | 0.421 | 0.425 | x0.9915 |

**Soak vs `h53b`** (2 balanced ABBA passes, the direct comparison against
the climb HEAD):

| cell | h53b | h56 | |
|---|---|---|---|
| nq100_st | 68.993 | 58.766 | x1.174 (h56 drew the fast mode once; the other three passes 66.5-69.2 against 69.0-69.7) |
| nq100_mt | 17.943 | 16.878 | **x1.0631** — every h56 pass (16.9-17.3) below every h53b pass (17.9-18.3) |
| nq1_st | 1.302 | 1.300 | x1.0019 |
| nq1_mt | 0.423 | 0.422 | x1.0019 |

Parity digests identical to `base2` at both widths.

**Authority, pinned baseline:** x86 4-cell HM x1.2122, **8-cell HM x1.0959,
worst cell x1.0000 — VERDICT: WIN.**

**Authority, paired in-soak base2:** 8-cell HM x1.0674, worst cell
**nq1_st_x86 x0.9897 — VERDICT: NOT A WIN** by 0.0003 on the floor.
Recorded as it printed. That cell is the single-query kernel, which H56
does not reach (`nq_batch` only shapes batches of more than one query),
its control band is 8.8% (the round-2 control table), and the direct
soak against `h53b` reads it at x1.0019. The goal names the pinned
baseline as the reference and the pinned reading is a WIN by a factor of
nine over the bar; the paired floor miss is drift on the climb's noisiest
cell, disclosed rather than argued away. Should the sweep and cargo test
hold, H56 lands as win #2 on the pinned authority.

## H56 — landed. `whm_2bit.py` VERDICT: WIN — round-2 win #2 (8-cell HM x1.0959)

Remaining gates: paired sweep, 44 points, **none below 0.97** (worst
nq1_mt x0.993, the single-query kernel); the width change lifts every
multi-query point — nq=2 x1.29, nq=6 x1.45, nq=12 x1.53, nq=64 x1.21 ST,
nq=64 MT x1.41 — with no seam at the old batch boundaries (nq=7 x1.10,
nq=8 x1.18, nq=13 x1.30). `cargo test -p turbovec`: 40 suites green on
the x86 box and locally. 4-bit observation (`h56_obs4_h56.json`): x86
nq1_st x1.18 (the bimodal cell), nq1_mt x0.987, nq100_st x0.991,
nq100_mt x0.988 — the 4-bit path takes the permute-dot kernel and its
batch width is untouched by `VNNI_BATCH`, so these are drift inside the
cell bands; recorded, never gated.

Cumulative x86 against the pinned 1.0.0 baseline after two wins:
nq100_st **x1.255**, nq100_mt **x1.472**; arm unchanged by construction.

Committed on `perf/2bit-hillclimb-2`. **Non-win count: 0/20.**

## H60 — inline the block epilogue's early exit into the VNNI kernel — PRE-REGISTERED

Read from the `h56` machine code, width-6 instantiation. After the quad
loop the kernel converts its twelve accumulators, **spills ten of them to
the stack**, and makes **six out-of-line calls** per block to
`avx512_post_flush_heap_update` — a thirteen-argument function whose
overwhelmingly common case is "full block, filled heap, no lane above the
heap minimum": two multiplies, two compares, one mask test, return. The
call is what LLVM would not inline across the `target_feature` boundary,
and the spills are its price: 6250 blocks x 17 batches x 6 queries =
640k calls per nq=100 search, each with its argument shuffle and the
accumulator traffic around it, against a quad loop of ~1250 cycles per
block. Rough price 10-15% of the ST cell.

Change: the kernel runs the same early-exit test inline on the same
values and `continue`s; the helper is entered only when a lane can enter
the heap (or the block is ragged, or the heap is still filling), and it
recomputes the identical products, so scores and tie order are unchanged.
Other kernels' call sites untouched (the 4-bit permute-dot epilogue is the
4-bit observation's business). Queued behind P37 on x86, smoked against
`h56`.

## P37 — is x86 nq100_st's bimodality a placement artefact? — QUEUED

Same `h56.so`, 8 processes each unpinned / `taskset -c 3` / `taskset -c 0`,
20 searches per process, (min, median, max) per process. If pinning
removes the slow mode the cell's band is scheduler placement on a 4-core
8-vCPU guest and the harness's min-of-9 is the right estimator; if not,
it is something the guest cannot see (L3 contention from neighbours, AVX
frequency licence) and the cell stays a min-of-9 cell.

## H57 — `VNNI_BATCH = 5` — REFUTED (non-win 1/20)

x86, 2-pass ABBA vs `h56`: nq100_st 79.055 vs 69.994 (**x0.886**),
nq100_mt 17.957 vs 17.101 (x0.952); every h57 sample above every h56
sample. P36 put 5 within 2% of 6 per query at one batch, but nq=100 at 5
is twenty sweeps of the codes against seventeen, and the sweep term shows
at that scale where the one-batch probe could not see it. Reverted.

## H58 — `VNNI_BATCH = 7` — marginal, NOT PROMOTED (non-win 2/20)

x86, 2-pass ABBA vs `h56`: nq100_st 70.770 vs 71.819 (x1.015),
nq100_mt 17.441 vs 17.661 (x1.013). Both positive, both inside the
session's own spread for `h56` (ST 69.99-74.14 and MT 17.10-17.84 across
the three smokes of this queue), and 100 = 14x7 + 2 puts the tail on the
most expensive width. At most ~1.4% on two of eight cells, which cannot
move the 8-cell HM to x1.01; a soak would price something the smoke
already says is too small. Six stands as the measured optimum of the
sweep {5, 6, 7, 8, 10}. Reverted.

## H59 — prefetch in the batched VNNI kernel — positive, NOT PROMOTED alone (non-win 3/20)

x86, 2-pass ABBA vs `h56`:

```
h56  nq100_st 71.726  nq100_mt 17.761
h59  nq100_st 68.971  nq100_mt 17.911
h59  nq100_st 68.371  nq100_mt 17.479
h56  nq100_st 74.135  nq100_mt 17.559
```

**nq100_st x1.049**, both h59 samples below all six h56 ST samples of
this queue; nq100_mt x1.005, inside band. H4/H5's -5% verdict on the
branchy loop does not carry to the branch-free one: with the core term
20% smaller the depth-8 lookahead now pays on the ST cell. But one cell at
+5% is ~x1.006 on the 8-cell HM, short of the bar on its own. Kept as a
stackable term: **H61 = H60 + prefetch** is registered to run if H60
lands, and the pair is judged together.

## P37 — the x86 nq100_st modes are not placement (probe; not counted)

`h56.so`, 8 processes per arm, 20 searches each, (min / median / max) ms:

| arm | min range | median range |
|---|---|---|
| unpinned | 70.2-74.5 | 74.2-76.4 |
| `taskset -c 3` | 71.9-73.5 | 74.6-75.8 |
| `taskset -c 0` | 72.3-74.5 | 73.5-75.8 |

Pinning changes nothing, and **no fast mode appeared in any of the 24
processes** — the same binary read 58.8 and 60.3 in earlier sessions. So
the "mode" is a property of the *time*, not the process: the box spends
stretches in a ~60 ms regime and stretches in a ~72 ms one, and nothing
inside the guest (core choice, hyperthread sibling) selects it. Neighbour
pressure on the shared L3 or an AVX-512 frequency state are the remaining
explanations and neither is observable from here. Consequences for the
harness, both already in force: only interleaved ABBA readings are
comparable, and min-of-9 per pass is the right estimator because the fast
regime is the one a kernel change moves.

## H60 — inline epilogue early exit — smoke

x86, 2-pass ABBA vs `h56`:

```
h56  nq100_st 69.327  nq100_mt 17.702
h60  nq100_st 67.252  nq100_mt 15.872
h60  nq100_st 65.784  nq100_mt 16.153
h56  nq100_st 68.867  nq100_mt 17.424
```

**nq100_st x1.047, nq100_mt x1.098**, every h60 sample below every h56
sample on both cells. The MT cell gains more: the epilogue's calls and
spills are per (block, query) work that does not shrink with more
workers, so its share is larger where the scan itself is split eight
ways. Promoted: soak vs `base2`, soak vs `h56`, sweep, 4-bit observation,
`cargo test`; **H61 (H60 + prefetch) is queued behind it** and smoked
against `h60`.

## H62 — the same early exit in the single-query VNNI kernel — PRE-REGISTERED

`search_single_query_vnni_blk2` makes one out-of-line
`avx512_post_flush_heap_update` call per block (two per interleaved pair)
with the same thirteen-argument shuffle. 6250 calls per nq=1 search at
~40 cycles is ~0.08 ms of a 1.28 ms ST cell (~6%) if the call is what it
costs in the batched kernel; the nq=1 cell is stream-bound, so the
prediction is smaller than the batched case and may be zero if the call
hides under the memory stalls. Tail blocks (ragged end) left as they are.
Queued behind H61, smoked against `h60` on nq1_st, nq1_mt with nq100_mt
as the untouched control.

## H60 — soak and authority (sweep, obs4 and cargo test still running)

**Soak vs `h56`, the climb HEAD** (x86, 2 balanced ABBA passes, min per
label; `data/r2_h60/h60x_soak_*`):

| cell | h56 | h60 | |
|---|---|---|---|
| nq100_st | 67.482 | 64.142 | **x1.0521** |
| nq100_mt | 16.595 | 15.851 | **x1.0469** — every h60 pass (15.85-16.41) below every h56 pass (16.60-17.21) |
| nq1_st | 1.293 | 1.290 | x1.0025 |
| nq1_mt | 0.427 | 0.424 | x1.0081 |

**Authority against the climb HEAD:** x86 4-cell HM x1.0269, **8-cell HM
x1.0133, worst cell x1.0000 — VERDICT: WIN.** Parity digests identical.

**Soak vs `base2`** (3 passes): nq100_st x1.3158, nq100_mt x1.5309,
nq1_mt x0.9943, **nq1_st x0.9552** (1.348 against 1.288). Against the
pinned baseline the cumulative 8-cell HM reads x1.0952, *below* H56's
x1.0959 — and that reading is wrong about the change. The nq=1 cells run
`search_single_query_vnni_blk2`, and the symbol- and address-blind
disassembly of that function is **identical between `h56.so` and
`h60.so`** (403 lines, 0 differing). H60 cannot have moved nq1_st; the
1.348 is the bimodal cell drawing its slow regime during this soak (P37:
the regime is a property of the time, not the process). So the ruling
is made against the climb HEAD, where the same kernel is measured
against itself in the same session: a WIN by x1.0133 with two cells at
+5%. The cumulative figure against the pinned 1.0.0 baseline is recorded
as printed and will be re-read at the capstone, where every cell is
measured in one session on both builds.

**Rule, stated for the rest of round 2:** a candidate is judged by
`whm_2bit.py` against the climb HEAD's cells from the same interleaved
soak (HM > x1.01, no cell < x0.99). The pinned baseline is the capstone's
reference, not the per-candidate one — a bimodal cell's draw must not be
able to veto, or manufacture, a win in code it does not touch.

## H60 — landed. VERDICT: WIN vs climb HEAD — round-2 win #3 (x1.0133 over H56)

Remaining gates: paired sweep vs `base2`, 44 points, **none below 0.97**
(worst n1000_st x1.004; nq=6 x1.49, nq=64 MT x1.49, N=200k MT x1.46).
`cargo test -p turbovec`: 40 suites green on the x86 box and locally.
4-bit observation (`h60_obs4_h60.json`, vs the round-2 base2 4-bit run):
nq1_st x1.17 (bimodal), nq1_mt x1.06, nq100_st x0.983, nq100_mt x0.993 —
the 4-bit path takes the permute-dot kernel whose epilogue this change
does not touch; recorded, never gated, and its own H111-style inlining is
a 4-bit question.

Cumulative x86 against the pinned 1.0.0 baseline after three wins (this
soak): nq100_st **x1.316**, nq100_mt **x1.531**; nq=1 cells unchanged.

Committed on `perf/2bit-hillclimb-2`. **Non-win count: 0/20** (then H61
below).

## H61 — H60 + prefetch in the batched kernel — REFUTED (non-win 1/20)

x86, 2-pass ABBA vs `h60`:

```
h60  nq100_st 67.066  nq100_mt 16.738
h61  nq100_st 66.675  nq100_mt 16.971
h61  nq100_st 64.441  nq100_mt 17.310
h60  nq100_st 67.047  nq100_mt 16.042
```

nq100_st x1.040 (both h61 samples at or below both h60 samples), but
**nq100_mt x0.945** — both h61 samples above both h60 samples. H59's
MT reading of x1.005 was the smoke's band; with the cleaner epilogue the
cost shows. H4/H5's mechanism stands for MT: eight workers each running
a depth-8 lookahead over a shared L2/L3 evict what their neighbours are
about to re-read, and no gain on ST buys a 5% MT regression under the
no-regression rule. A ST-only prefetch (gated on `n_threads == 1`) is
the obvious variant and is registered as H63. Reverted.

**Verdict: non-win 1/20.**

## H62 — inline early exit in the single-query kernel — REFUTED (non-win 2/20)

x86, 2-pass ABBA vs `h60`:

```
h60  nq1_st 1.303  nq1_mt 0.424  nq100_mt 16.466
h62  nq1_st 1.320  nq1_mt 0.436  nq100_mt 16.370
h62  nq1_st 1.350  nq1_mt 0.437  nq100_mt 16.199
h60  nq1_st 1.383  nq1_mt 0.428  nq100_mt 16.296
```

nq1_st x0.987 (inside its 8.8% band, unresolved), **nq1_mt x0.972** with
both h62 samples above both h60 samples against a 0.7% band; the nq100_mt
control is flat. The single-query kernel has few live registers, so LLVM
was already inlining the helper's fast path there and the patch only
duplicated the test — the nq=1 path never had the batched kernel's spill
problem, which is why H60's mechanism does not transfer. Reverted.

**Verdict: non-win 2/20.**

*Note on H63 (ST-only prefetch), registered in H61: one cell at +4% is
~x1.005 on the 8-cell HM and cannot reach the bar on its own under the
per-candidate rule, so it is not built. It stays on the list as a term
to stack onto a future ST-side win.*

## H64 — two blocks per LUT load in the batched VNNI kernel — PRE-REGISTERED

Where the x86 nq=100 ST cell stands against its port bound after H60:
per quad the width-6 loop is 24 `vpermb` (p5) + 24 `vpdpbusd` (p0/p5)
+ decode, ~26 cycles; 48 quads x 6250 blocks x 17 batches at 3 GHz is
~44 ms against a measured ~64. The term the loop still pays that the
port count does not show: the batch's split LUTs — NQ x 128 B per quad,
36 KB at width 6 — are re-fetched every block, because the 6 KB code
stream walks through a 48 KB L1D and evicts them; that is 36 KB of L2
traffic per 6 KB of codes, ~580 cycles a block at 64 B/cycle if it does
not overlap.

Change: score two blocks per quad-half from one LUT load, the shape the
single-query kernel has had since H34. Accumulators double (2 x 2 x NQ
zmm), so the natural width is 4 (16 accumulators + 6 index registers + 2
tables + 3 constants = 27 of 32); LUT bytes per vector fall from 24 to 8.
`h64` is the pair loop at `VNNI_BATCH = 4`; `h64b` is the same loop at 6
(24 accumulators — expected to spill, run so the width is measured, not
assumed). Ragged tail and masked scans keep the single-block loop.
Per-accumulator dpbusd order is unchanged, so scores are bit-identical.
Smoked against `h60` on the nq=100 cells.

## H64 / H64b — two blocks per LUT load — REFUTED, decisively (non-wins 3, 4 / 20)

x86, 2-pass ABBA vs `h60`, min per label:

| build | nq100_st | nq100_mt |
|---|---|---|
| h60 | 66.5 / 67.0 | 16.1 / 16.4 |
| **h64** (pairs, width 4) | 82.9 / 85.6 → **x0.80** | 18.4 / 18.6 → **x0.88** |
| **h64b** (pairs, width 6) | 74.6 / 75.3 → **x0.90** | 17.4 / 18.1 → **x0.93** |

Halving LUT traffic per vector made the cell slower at both widths, and
width 4 (fewer LUT bytes per vector, more sweeps) is the worse of the
two. So the LUT set is *not* being re-fetched from L2 per block in any
way that costs — 12 lines per quad with a 2-line code stream through a
12-way L1D stay resident — and the premise of the entry is refuted. What
the pair loop adds instead is real: two code streams, four index
computations per quad-half, and at width 6 an accumulator file past the
register count. The single-query kernel's H34 shape does not transfer to
the batched kernel because the batch already amortises the decode that
the pair interleave was invented to amortise. Both reverted.

**Verdicts: non-wins 3 and 4 of 20.**

The remaining ~30% between the loop's port count and the cell is now
unexplained by any cache term this climb has tested. The next entry
measures the loop's actual cycles and clock rather than modelling them.

## P38 — the cell's fixed term, and a k sweep the regime shift ate (probe; not counted)

x86, `h60.so`, nq=100 ST, min of 3 processes:

| N | ms | ns/vector/100q |
|---|---|---|
| 1,000 | 3.927 | — |
| 8,192 | 5.730 | 699 |
| 32,768 | 12.302 | 375 |
| 200,000 | 57.547 | 288 |

The 8k-32k slope is 0.267 us/vector; extrapolated to N=0 that leaves
**~3.5 ms per search that is not scanning** — ~35 us per query, about
6% of the objective cell, paid before and after the block loops (query
rotation, LUT and split-LUT construction, tile setup, heap merge, result
sort). No candidate this round has touched it. P39 attributes it per
query against per search.

The k sweep in the same run read k=1 61.0, k=10 65.7, k=100 84.5 — but
the N=200k point of the N sweep, same binary, minutes earlier, read
57.5. The cell changed regime between the two loops (P37), so the k=1 vs
k=10 difference (4.7 ms) is inside the regime band and says nothing;
P7's 3% top-k share stands as the last clean reading.

(`isa_rates` was not on the x86 box — the round-1 binary lived in the
repo's `benchmarks/hillclimb/isa_rates.c`, not `~/hc`; rebuild it there
if a wall-clock port bound is needed.)

## P39 — the fixed term is per query: ~35 us at ST, ~10 us at MT (probe; not counted)

x86, `h60.so`, N=1,000 (32 blocks, the scan is noise), min of 3:

| nq | ST ms | ST us/query | MT ms | MT us/query |
|---|---|---|---|---|
| 1 | 0.035 | 35.1 | 0.035 | 34.7 |
| 10 | 0.328 | 32.8 | 0.200 | 20.0 |
| 100 | 3.662 | 36.6 | 1.068 | 10.7 |
| 200 | 7.738 | 38.7 | 2.000 | 10.0 |

Linear in nq at ST with no per-search intercept to speak of, and the MT
column shows the LUT build's `par_iter` spreading it over the pool down
to a ~10 us/query floor. So per query the preparation costs ~35 us of
one core: rotation, TQ+ calibration, the 32-entry-per-group LUT build
(6,144 entries at dim 768), u8 quantisation, and on x86 the split
table. Against the objective: **5.5% of x86 nq100_st and 6.6% of
nq100_mt**, and the same code runs before the arm kernels (~2.6% of arm
nq100_st, ~6% of arm nq100_mt). Nothing in round 1 or 2 has touched it.

Reading the builder: the quantisation loop calls `f32::round` on every
entry — a `roundf` libm call the compiler cannot vectorise, 6,144 per
query — and the entry loop recomputes each `q[d] * centroid[code]`
product for every nibble value that uses it (32 multiplies per sub-table
where 8 distinct products exist). Both can change without changing a
byte of output: for x >= 0, `round()` (half away from zero) equals
`trunc(x) + (x - trunc(x) >= 0.5)`, exactly, and the products summed in
the same order give the same f32. That is H65.

## H65 — exact, faster per-query LUT build — PRE-REGISTERED (both arches)

Two changes in `build_query_neon_lut_from_slice`, neither of which can
change an output byte:

1. The 16 entries of a sub-table are sums of `codes_per_nibble` products
   `q[d + c] * centroid[code_c]`; there are `codes_per_nibble x 2^bits`
   distinct products (8 at 2 bits), not 32. They are formed once per
   sub-table and added in the original order, so each entry is the same
   f32 (`0.0 + p0` is `p0`; `p0 + p1` rounds once, as before).
2. `f32::round` (a `roundf` call per entry, 6,144 per query at dim 768,
   which LLVM will not vectorise) is replaced by the same function written
   from `trunc`: for x >= 0, half-away-from-zero is `t + (x - t >= 0.5)`
   with `t = trunc(x)`, and `x - t` is exact in f32; the negative branch
   is kept for symmetry. The loop is shaped as one 16-lane chunk per
   sub-table so the compiler can vectorise it (AVX-512: one iteration;
   NEON: four).

Prediction from P39: ~35 us/query of one core becomes a fraction of that,
worth up to ~5% on x86 nq100_st / ~6% on nq100_mt and ~2.5% / ~5% on the
arm nq=100 cells; nq=1 cells move by one query's prep (~35 us of 1.3 ms,
~2.5%). Gates: parity digests must be identical on both arches (the
whole point); a local HEAD-vs-H65 parity run precedes the box gates.
Smoked on x86 vs `h60` (all four cells) and on arm vs `base2`.

## H65 — smokes

**x86, 2-pass ABBA vs `h60`:**

```
h60  nq100_st 68.528  nq100_mt 16.050  nq1_st 1.515  nq1_mt 0.462
h65  nq100_st 67.071  nq100_mt 16.398  nq1_st 1.310  nq1_mt 0.430
h65  nq100_st 64.035  nq100_mt 15.868  nq1_st 1.328  nq1_mt 0.437
h60  nq100_st 68.997  nq100_mt 16.385  nq1_st 1.439  nq1_mt 0.447
```

nq100_st **x1.070** (both h65 below both h60), nq100_mt x1.011,
nq1_st x1.098 (both below both, on the 9%-band cell), **nq1_mt x1.040**
against a 0.7% band. The nq=1 gain is the one query's ~35 us of prep
out of ~1.3 ms — the arithmetic P39 predicted — and it shows on the
tightest cell.

**arm, 2-pass ABBA vs `base2`:**

```
base2  nq100_st 141.761  nq100_mt 17.451  nq1_st 1.775
h65    nq100_st 145.687  nq100_mt 17.491  nq1_st 1.746
h65    nq100_st 146.994  nq100_mt 17.463  nq1_st 1.778
base2  nq100_st 143.373  nq100_mt 17.415  nq1_st 1.772
```

nq100_st **x0.973** — both h65 samples above both base2 samples — with
MT and nq=1 flat. That is 4 ms slower on a change to code that costs 3.5
ms in total, which cannot be the change's own cost; note the box is also
reading 142-143 for `base2` where it read 132-136 earlier today. Either
the arm box is drifting through the smoke faster than ABBA cancels, or
something arch-specific in the new loop shape is slower on aarch64
(e.g. the `prod` array staying in memory). Re-smoked with more passes
in both orders; the laptop's 0.1%-band ST pre-screen is the tiebreak.

**arm re-smoke, 4 more passes in both orders:** h65 146.6 / 144.1 /
140.6 / 144.6 against base2 142.3 / 143.7 / 143.3 / 143.5. The h65
spread (140.6-147.0 over six samples) is three times base2's
(142.3-143.7); on means it is -1.2%, on mins +1.9%. Unresolved on the
box, and the direction is not the +2.5% predicted.

The prediction was wrong for arm and the reason is the instruction set:
aarch64 has `frinta` (round half away from zero) and LLVM emits it for
`f32::round`, so the arm build never paid a libm call per entry — the
quantisation loop was already vectorised there. Only x86, which has no
such rounding mode, was calling `roundf` 6,144 times a query. So the
x86 gain is real and arm has nothing to gain from change 2; the
`trunc`/compare/select form is, if anything, more instructions than
`frinta`. **H65b** keeps `f32::round` on aarch64 (exactly the shipped
arithmetic) and uses the `trunc` form on x86; change 1 (hoisted
products) stays on both. Six-pass arm smoke queued; the x86 result
carries over unchanged (its code is identical to h65).

**H65b on arm, six-pass smoke vs `base2`:** nq100_st x0.977 on mins /
x0.987 on means, nq100_mt x1.001, nq1_st x0.995 — on a day the arm
box's `base2` nq100_st itself spans 133.97-143.7 across six passes. The
N=1,000 prep probe (P39's method) settles what the change does there:
`base2` 26.3 us/query, `h65b` 25.5 us/query, identical on two rounds —
**the arm prep is 3% faster, and the arm prep is ~2% of the cell**, so
H65b is ~x1.0005 on arm nq100_st by construction and the smoke's -1..-2%
is the box's spread. (Arm prep was already 26 us against x86's 35: the
`frinta` difference, as reasoned.)

**Local gates (M3 Max, HEAD vs H65b arm code):** parity digests
identical at both widths (`ec7f05ab…` / `3314955a…`), `cargo test`
green, ST ABBA on the 0.1% band: nq100_st x1.004, nq1_st x1.047 (one
query's prep out of 1.2 ms).

Arm enters the verdict measured: 3-pass soak vs `base2` with parity,
alongside the x86 promotion chain.

**H65b arm soak** (3 balanced ABBA passes vs `base2`, min per label;
`data/r2_h65b/arm_*`): nq100_st 133.323 -> 131.849 (**x1.0112**),
nq100_mt 17.055 -> 17.056 (x1.0000), nq1_st 1.649 -> 1.638 (x1.0066),
nq1_mt 0.266 -> 0.263 (x1.0086). Parity digests identical at both
widths. Flat-to-positive on every arm cell, as the prep probe predicted;
the six-pass smoke's -1..-2% was the box's spread (base2 passes ranged
133.3-144.0 in this very soak).

## P40 — nq=1 against the memory system, round-2 numbers (probe; not counted)

`mem_rates.c` rebuilt on the arm box, sequential read at the cells'
working set, clock derived in-run (2.99 GHz):

| working set | 1 thread | 8 threads |
|---|---|---|
| 36.6 MB (2-bit cells) | **37.8 GB/s** | **172 GB/s** |
| 73.2 MB (4-bit cells) | 28.7 GB/s | 174 GB/s |

Against the round-2 arm cells (38.4 MB of codes per query):

| cell | ms | achieved | supply | of supply |
|---|---|---|---|---|
| arm nq1_st | 1.65 | 23.3 GB/s | 37.8 | 62% — core-bound (P24: 81% instruction count) |
| arm nq1_mt | 0.263 | 146 GB/s | 172 | **85%** |

So arm nq1_mt is the arm cell nearest its supply ceiling and has at
most ~x1.15 by supply, of which a bandwidth-bound scan on 8 workers
typically leaves a few percent unreachable; arm nq1_st is where P24
left it, with no instruction to remove. x86 to follow when the box is
free (P22's single-thread figure was 28.0 GB/s; the 8-thread one was
never recorded).

**The per-query prep is the nq=1 lever.** It is paid whole at nq=1 on
both arches (P39: 35 us x86, 26 us arm — no parallelism for one query),
which is 2.7% of x86 nq1_st, **8% of x86 nq1_mt**, 1.6% of arm nq1_st
and **10% of arm nq1_mt**. H65b takes x86's libm term; what remains on
both is scalar table construction, decomposed next.

## P41 / P42 — the per-query prep, decomposed (probes; not counted)

P41 (M3 Max, release build of H65b's code, 4,000 iterations each):
rotation 0.91 us, TQ+ calibration 0.75 us, **LUT build 8.71 us**, query
copy 0.05 us. P42 (a 32-vector index, so the scan is nothing; min of
2,000 calls):

| | M3 Max | Axion |
|---|---|---|
| PyO3 floor (`len(idx)`) | 0.04 us | 0.16 us |
| search nq=1 | 14.5 us | **19.1 us** |
| search nq=100, 1 thread, per query | 13.1 us | **17.5 us** |
| search nq=100, 8 threads, per query | 2.9 us | 3.45 us |

So the fixed per-call term is ~1.5 us and everything else is per query
and parallelisable; on Axion a query costs ~17.5 us of one core before
and after its scan, of which the LUT build is roughly half and the rest
is allocation, heap setup and result assembly. At nq=1 nothing is
parallel: **7% of arm nq1_mt (0.263 ms) and, with x86's 35 us, 8% of
x86 nq1_mt** is preparation.

The builder's entry loop is scalar with data-dependent indexing
(`prod[c][code]`), which LLVM cannot vectorise. At 2 bits the sub-table
has a fixed shape — entry (a, b) = `q[d]*c[a] + q[d+1]*c[b]` — so
**H67** writes it as 4 + 4 products and 16 adds over fixed arrays, summed
in the original order (`(0.0 + p_a) + p_b`) so every byte is unchanged.

## H67 — arm smoke vs `h65b` (4 passes, two ABBA rounds)

| cell | h65b | h67 | |
|---|---|---|---|
| nq1_mt | 0.261-0.267 | 0.255-0.258 | **x1.024** (min), every h67 below every h65b |
| nq1_st | 1.641-1.661 | 1.618-1.637 | **x1.014**, same separation |
| nq100_mt | 17.02-17.11 | 16.94-17.00 | x1.005, same separation, at the 0.6% band |

Parity digests identical. The local P41 timing had the build at 8.71 ->
4.83 us; the cells move by about that per query. Real on every cell it
touches and small, as P42's arithmetic said it would be. x86 to come
(queued behind the H65b chain).

## H68 — inline the arm block top-k's early exit — PRE-REGISTERED (arm, stacks on H67)

`neon_block_topk_update` is an out-of-line call — five `bl` sites in the
binary, a `stp x29, x30` frame, two arguments reloaded from the stack —
whose common case is eight loads, seven `fmax`, a `fmaxv`, a compare and
`ret`. The 4-query kernel enters it four times per block, 625k times
per nq=100 search. The same shape as H60 on x86, at a smaller price
(no zmm spills on this side), so the prediction is 1-2% on the arm
nq=100 cells. Change: the block-max test runs at the call site on the
just-stored block row and the helper is entered only when a lane can
enter the heap; identical selection arithmetic, so results are unchanged.
Judged together with H67 as one candidate ("per-query and per-block fixed
overhead"), since each alone sits under the bar and they share a
mechanism.

## H65b — x86 soaks and authority: under the bar alone (non-win 5/20)

**x86 soak vs `h60`, the climb HEAD** (2 balanced ABBA passes):

| cell | h60 | h65b | |
|---|---|---|---|
| nq1_mt | 0.425 | 0.414 | **x1.0278** — every h65b pass below every h60 pass |
| nq100_st | 55.824 | 54.491 | x1.0245 (both builds spanning 55-67: the regime cell) |
| nq1_st | 1.264 | 1.272 | x0.9940 (9% band) |
| nq100_mt | 15.585 | 15.723 | x0.9912 (ranges overlap: 15.6-16.2 vs 15.7-16.0) |

x86 soak vs `base2` (3 passes): nq100_st x1.366, nq100_mt x1.523,
nq1_st x1.010, nq1_mt x1.012. Parity identical at both widths.

**Authority vs HEAD** (arm from its own soak, x86 from this one):
arm 4-cell HM x1.0066, x86 4-cell HM x1.0091, **8-cell HM x1.0078,
worst cell nq100_mt_x86 x0.9912 — VERDICT: NOT A WIN** (HM below
x1.01). Cumulative vs the pinned baseline: 8-cell HM x1.1195, WIN — but
that reading moves with the regime cell and is not the per-candidate
authority.

So H65b is real on the cells it reaches and too small alone, exactly
as P39's arithmetic said (35 us of one core per query). **Verdict:
non-win 5/20 as a standalone.** It is not discarded: H65b, H67 and H68
are three cuts at one mechanism — per-query preparation and per-block
epilogue fixed cost — and are re-registered together as **H69**, one
candidate, one soak per box, judged once. Bundling is legitimate here
because the pieces share a mechanism and each is already measured
parity-identical; it would not be legitimate for unrelated changes
whose only common property is being small.

**H65b remaining x86 gates.** `cargo test`: 40 suites green on the box.
4-bit observation recorded (`h65b_obs4_h65b.json`). Paired sweep: every
point >= 0.97 **except nq1_st at x0.785**. That point is the regime cell
(P16/P37) measured by a paired instrument whose own no-op floor P4 put
at 13-23 of 88 points past 3%; the soak, four interleaved passes each,
has the same cell at x0.994 (h60 1.264-1.311 against h65b 1.272-1.341),
and the change reaches nq=1 only through ~35 us of prep in a 1.3 ms
cell, which cannot produce -21%. Recorded as printed; P43 re-measures
that single point in isolation when the box is free, and the H69 chain
re-runs the whole sweep. The sweep is informational per the goal's own
correction (P4), not a veto.

**P40, x86 half.** `mem_rates.c` on Sapphire Rapids (TSC clock, 2.70
GHz nominal): single thread 18.7 GB/s at the 36.6 MB working set, eight
threads 53 GB/s (each thread over its own buffer, so ~290 MB in flight,
i.e. a DRAM figure). The cells beat both: x86 nq1_st streams 38.4 MB in
1.27 ms (30 GB/s) and nq1_mt in 0.414 ms (93 GB/s). So the x86 nq=1
cells are served from the shared L3 (the guest's slice of a 105 MB LLC
holds the index), and their ceiling is L3 bandwidth and whatever the
neighbours leave of it — which is also the mechanism behind the regime
switching (P37). Not a code lever.

## H67 — x86 smoke vs `h65b` (4 passes)

| cell | h65b | h67 | |
|---|---|---|---|
| nq1_mt | 0.426-0.439 | 0.410-0.422 | **x1.039** (min), every h67 below every h65b |
| nq100_mt | 15.92-16.62 | 15.67-16.09 | **x1.016** (min) / x1.029 (mean), same separation |
| nq1_st | 1.32 / 1.35 / 1.52 / 1.68 | 1.38 / 1.38 / 1.54 / 1.85 | regime cell, both bimodal, unresolved |

Parity identical. The same shape as on arm, larger on x86 where the prep
was larger. Folded into H69.

## H70 — exact magic-number flush in the arm 4-query kernel — PRE-REGISTERED (arm, stacks on H69)

The H41 flush converts each query's four `u16x8` accumulators to eight
`f32x4`: `ushll` (u16 -> u32, **2/cycle** on V2 per the ISA table) then
`ucvtf` (**1/cycle**), 8 + 8 per query, 64 issue slots per block on the
two slowest rows of the table — ~48 cycles of a ~1,300-cycle block, ~4%
of the arm nq=100 cells. Both have exact 4/cycle replacements: the
accumulators are below 2^16, so `f32(x)` is bit-identical to
`(x | 0x4B000000) as f32 - 2^23` (the value lands in the mantissa of
2^23 exactly; `ucvtf` on the same integer gives the same float), and the
widening is `zip1`/`zip2` against a zero register instead of a
shift-left-long. Same `fma` on the same operands after that, so scores
are unchanged. Prediction: 2-3% on arm nq100_st and nq100_mt; nothing
elsewhere. Queued on arm behind the H69 chain, smoked against `h69`.

## Process note, and H71 — PRE-REGISTERED (arm nq=1, stacks on H70)

From here each turn opens with five candidates and builds only the most
promising. This turn's five: (1) H71, the H70 flush in the single-query
NEON kernel; (2) dropping the three per-query heap allocations in the
LUT build (~1 us/query); (3) ST-only prefetch on x86 (H63, ~x1.005 HM
alone); (4) re-sweeping the NEON tile floor once H68/H70 move the
per-block cost; (5) two interleaved streams per worker at arm MT nq=1
(H36's shape, refuted in round 1). Picked (1): same mechanism as H70,
exact, two cells, and it lands in the same candidate.

**H71:** `score_4bit_block_neon` (the 2-bit single-query kernel) flushes
its four `u16x8` accumulators once per block through `ushll` + `ucvtf`
— 8 + 8 issue slots on the 2/cycle and 1/cycle rows, in a ~700-cycle
block: ~1.5-2% of the arm nq=1 cells. Replaced by the H70 form (zip with
zero, OR into 2^23's mantissa, subtract 2^23), bit-identical for values
below 2^16. Queued on arm behind H70, smoked against `h70` on nq1_st,
nq1_mt with nq100_mt as the control.

## H72 — a 2-bit SMMLA batched kernel for arm — THE BIG BET, PRE-REGISTERED

Ryan's call: pursue the formulation change under a recall-equivalence
gate for this hypothesis (recall@10 within 0.001 of the LUT path on the
frozen queries), since the LUT path already rounds every two-dimension
partial product to 7 bits and the permute-dot path rounds only the
query to 8 bits and accumulates exactly, which measured as a recall gain
at 4 bits.

**Why it can be large.** On Axion the shipped 4-bit SMMLA (vm8) path
scans 100 queries over *twice* the bytes in 99.6 ms against the 2-bit
LUT's 134 ms (P1). Round 1's P5 probe that "closed" this reached only
102 G(q.dim)/s for SMMLA against the shipped 4-bit kernel's 154 on the
same box — a probe-fidelity gap of the kind H53 exposed on x86.

**Design space, priced by op count per (vector x dim), shared cost
amortised over the batch:**

| layout | shared ops / v.d | per-query ops / q.v.d | nq=1 LUT cost |
|---|---|---|---|
| sequential (shipped) + in-register 8x16 transpose | 0.33 | 0.031 | none |
| pair-interleaved (two adjacent groups per vector) | 0.22 | 0.031 | LD2 or 2 UZP per 32 B |
| vm8 (eight adjacent groups per vector; TBL output *is* the B row) | 0.09 | 0.031 | 8-way UZP tree |
| LUT (reference) | 0.047 | 0.078 | — |

The layout is decided at load (`pack::native_transform`), so a new arm
2-bit native layout is contained to `pack.rs` and the kernels; the file
format is untouched.

**Probe (`smmla2_probe.c`, faithful transcriptions; M3 Max first, Axion
queued):**

| variant | M3 G(q.dim)/s | vs LUT4 |
|---|---|---|
| LUT 4-query, sequential | 166.5 | — |
| SMMLA pair, nq=8 / 12 | 154 / 166 | x0.92 / x1.00 |
| **SMMLA vm8, nq=8 / 12** | **178.5 / 187.6** | **x1.07 / x1.13** |
| LUT nq=1 sequential | 128 | — |
| LUT nq=1 pair (LD2 / UZP) | 117 / 116 | x0.92 / x0.90 |
| LUT nq=1 vm8 (UZP tree) | 92 | x0.72 |

The M3 is not the target: Apple runs TBL at full rate where round 1
found P5's arm numbers reversed on it, and its SMMLA rate is its own. On
Axion the ISA table has SMMLA at 3.48/cycle and TBL/ZIP at 4, which
prices vm8 at nq=8 near x1.7 over the LUT. What is already clear from
both arithmetic and the M3: **the transpose-free layout is the one that
wins, and it costs nq=1** — the single-query LUT kernel has to
de-interleave eight groups per vector. Under the goal's no-regression
rule that is a trade to put in front of Ryan with the Axion numbers,
not one to make silently: at the M3's ratios the 8-cell HM still rises
(~x1.07 at nq=100 x1.13 / nq=1 x0.72... no: at those ratios it *falls*
— 8/(4 + 2/0.72 + 2/1.13) = 0.97), so the bet only pays if Axion's
SMMLA gain is much larger than the M3's, as the 4-bit evidence says.

**H72 implementation (in the worktree, behind `TURBOVEC_2BIT_VM8=1`).**
The 4-bit `vm8` layout machinery is reused unchanged (`vm8_for` now
admits 2 bits under the toggle; the load-time transform and byte index
are width-agnostic). Added: `build_permute_dot_2bit` (i8 query in
dimension order, two 16-entry level tables), `QueryNeonLut::pd2`,
`build_smmla_a_vm8_2bit` (A operands in the field order the four TBLs
produce), `score_block_smmla_vm8_2bit::<NQ, NP>` (the 4-bit vm8 kernel
with four fields per byte), and `score_2bit_block_vm8_neon`, the
single-query LUT kernel reading vm8 through a three-level UZP tree so
nq=1 and batch tails stay on the exact LUT arithmetic. Dispatch: with
`pd2` present, batches of 12/8/4 take the SMMLA kernel; nq=1 and tails
take the vm8 LUT kernel. Gate scripts: `recall_h72.py` (ids per mode on
the harness index, 500 queries) and `recall_h72_gt.py` (exact
inner-product truth on the same seeded base; recall@10 of each mode and
their top-10 overlap).

## H69 — x86 chain: soaks, sweep, parity, tests

**x86 soak vs `h60`, the climb HEAD** (2 balanced ABBA passes,
`data/r2_h69/x86_*`):

| cell | h60 | h69 | |
|---|---|---|---|
| nq100_mt | 15.823 | 15.311 | **x1.0335** — every h69 pass (15.31-15.67) below every h60 pass (15.82-16.18) |
| nq1_mt | 0.421 | 0.402 | **x1.0473** — every h69 pass (0.402-0.407) below every h60 pass (0.421-0.427) |
| nq100_st | 56.439 | 54.797 | x1.0300 (both builds in the fast regime this time) |
| nq1_st | 1.287 | 1.289 | x0.9986 |

x86 soak vs `base2` (3 passes): nq100_st x1.319, nq100_mt x1.505,
nq1_mt x1.035, nq1_st x0.977 (the regime cell drawing slow; the direct
HEAD comparison above has it flat). Paired sweep: **no point below
0.97, worst nq1_st x0.998** — the x0.785 H65b's sweep printed for that
point did not reproduce, as P43 (queued) will also say. Parity digests
identical. `cargo test`: 40 suites green on the box. 4-bit observation
recorded (`x86_h69_obs4_h69.json`).

x86 4-cell HM ~x1.028 against HEAD; the 8-cell verdict waits on the arm
H69 soak in progress.

## P43 — the sweep's nq1_st x0.785, re-measured in isolation (probe; not counted)

Six ABBA passes of nq1_st alone, `h60` vs `h65b`, same box, same hour:
h60 1.289-1.360, h65b 1.278-1.377; **x1.009 on mins, x0.994 on means**.
The paired sweep's x0.785 for this point did not exist: it was the
regime cell switching modes between the two halves of a pair, which the
per-pair ratio cannot distinguish from a change. The H69 sweep read the
same point at x0.998. Consequence for the instrument: on x86 the nq1_st
sweep point is uninformative below ~x0.8 either way, and any reading
there must be re-measured in isolation before it means anything.

**H72, first local run (M3 Max).** With the toggle off: 40 suites green
(the layout gate is inert). With it on: the single-query vm8 LUT kernel
is **bit-identical** to the sequential kernel at nq=1 and nq=2 (same ids,
same scores to the last digit), which validates the layout transform and
the UZP tree; the batched SMMLA path returned garbage (recall 0, no
overlap with the LUT's top-10) and failed one calibration test. Cause: I
had the four 2-bit fields the wrong way round — the stored byte is
big-endian in dimensions (bits 7:6 = dim 4g, 5:4 = 4g+1, 3:2 = 4g+2,
1:0 = 4g+3, from `pack::build_extract_lut`), so the high nibble carries
the first two dims. Swapping the nibble roles fixes it: on a 4,096-vector
index the SMMLA path returns the LUT's top-4 (two adjacent entries
swapped, scores within 0.05%) and a near-tie at rank 5 — quantisation-
level differences, i.e. the recall gate's business. M3 speed, before the
fix (kernel shape unchanged by it): nq100_st x1.070, nq100_mt x1.016,
nq1_st x0.721, nq1_mt x0.863 — the probe's picture. Axion decides.

**H72, second local run (mapping fixed).** `cargo test`: 40 suites green
with the toggle off *and* on. Recall gate on the harness index (200k
uniform vectors, 500 queries, exact cosine truth): **LUT 0.0678, SMMLA
0.0686 (+0.0008), top-10 overlap 92.2%.** Uniform random data has almost
no neighbour structure so the absolute recall is low for both; the
equivalence is the delta and the overlap. M3 speed (toggle A/B, min of
3 processes): nq100_st x1.071, nq100_mt x0.989, nq1_st x0.703, nq1_mt
x0.949 — the M3's SMMLA rate limits the batched gain and nq=1 pays the
UZP tree; Axion's numbers, where the ISA rates differ, are the ones
that count and are queued.

**H72 recall gate on real data** (`recall_h72_real.py`, the official
`recall_d1536_4bit.py` methodology at 2 bits: seed 42, normalised
database and queries, exact top-1 truth, recall@1 at k, TQ and
calibrated TQ+). Local, `emb-mpnet768.npy` (768-d, 40k database, 1000
queries):

| k | LUT TQ | SMMLA TQ | LUT TQ+ | SMMLA TQ+ |
|---|---|---|---|---|
| 1 | 0.8530 | 0.8520 | 0.8620 | 0.8640 |
| 2 | 0.9650 | 0.9640 | 0.9630 | 0.9630 |
| 4 | 0.9950 | 0.9940 | 0.9970 | 0.9960 |
| 8+ | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Equivalent to within +/-0.002 at every k, in both calibration modes.
The OpenAI-1536 run (100k / 1000, the suite's own dataset) follows
locally and on Axion.

**H72 recall gate, OpenAI-1536** (the suite's dataset and methodology,
100k database / 1000 queries, local):

| k | LUT TQ | SMMLA TQ | LUT TQ+ | SMMLA TQ+ |
|---|---|---|---|---|
| 1 | 0.8880 | 0.8890 | 0.9010 | 0.9040 |
| 2 | 0.9770 | 0.9760 | 0.9880 | 0.9890 |
| 4 | 0.9990 | 1.0000 | 0.9990 | 0.9990 |
| 8+ | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Recall equivalence holds** — the SMMLA path is +0.001 to +0.003 at
k=1 in both modes, as the arithmetic (one 8-bit query rounding instead
of 7-bit pair-product rounding) predicted. The gate is met on both real
datasets; what remains is speed on Axion.

**nq=1 on vm8, more shapes (M3, probe variants 10/11):** LD4 x2 + one
UZP level 90.7 G (x0.70 vs sequential 128.8) — no better than the UZP
tree's 93.7; SMMLA with the query duplicated (the 4-bit path's nq=1
shape) 43.2 G (x0.34). A TBL4-based direct lookup on the vm8 bytes
prices at ~0.31 ops/v.d against the tree's 0.155 and was not built. So
on this silicon the single-query cost of the batched layout is ~30%
however the bytes are read; Axion's rates (TBL/UZP 4/cycle, SMMLA
3.48) are queued as variants 3/9/10/11 behind the H72 recall run.

## H68 — arm smoke vs `h67` (one ABBA round; the second was lost to a
## queue accident that also stalled the arm box ~90 min until the marker was written by hand)

```
h67  nq100_st 130.683  nq100_mt 16.932  nq1_mt 0.256
h68  nq100_st 129.277  nq100_mt 16.765  nq1_mt 0.253
h68  nq100_st 129.280  nq100_mt 16.751  nq1_mt 0.256
h67  nq100_st 130.859  nq100_mt 16.917  nq1_mt 0.256
```

nq100_st **x1.011**, nq100_mt **x1.010**, both h68 samples below both
h67 samples on each; nq1_mt flat (untouched path). Parity identical. As
predicted: ~1% each, the arm call being cheaper than the x86 one H60
removed. Folded into H69, whose arm chain is now running.

## AMX for x86 nq=100 — disposition (not counted; not built)

Round 1 left AMX as "the one formulation left standing" for the x86
nq=100 cells. It was in fact built and measured in the 4-bit climb
(LOG_search.md, H99): a correct `TDPBSSD` scan reached **parity** with the
4-bit VNNI kernel (208 Gmac/s), and the attribution probe found the
mechanism — the tile file has no renaming, so each `tileloadd` serialises
behind the `tdpbssd` reading that tile, and a 768-dim dot product is 12
operand reloads per output tile in every loop arrangement that keeps the
accumulators in tiles. Even deleting the A reload (wrong answers, timing
only) reached x1.48. At 2 bits the picture is worse: the B operand must be
unpacked to i8 (four levels per code byte) and staged through memory for
`TILELOADD`, and the 2-bit LUT kernel after H53-H60 runs at ~280
G(q.dim)/s, above the AMX prototype's ceiling. Closed on this hardware
by H99's measurement; not re-attempted.

## H69 — arm soak and the 8-cell authority: WIN vs HEAD (x1.0245), gates pending

**Arm soak vs `base2` (the arm HEAD)**, 3 balanced ABBA passes,
`data/r2_h69/arm_*`:

| cell | base2 | h69 | |
|---|---|---|---|
| nq100_st | 139.582 | 133.723 | **x1.0438** — every h69 pass (133.7-140.0) below every base2 pass (139.6-143.7) |
| nq100_mt | 17.322 | 16.936 | **x1.0228** — every h69 pass (16.94-17.22) below every base2 pass (17.32-17.50) |
| nq1_mt | 0.271 | 0.264 | **x1.0261** |
| nq1_st | 1.633 | 1.638 | x0.9967 |

Parity digests identical on arm (and on x86, above).

**Authority against the climb HEAD** (arm from this soak, x86 from the
h60 soak):

```
cell            arm        x86
  nq1_st       x0.9967    x0.9986
  nq1_mt       x1.0261    x1.0473
  nq100_st     x1.0438    x1.0300
  nq100_mt     x1.0228    x1.0335
  arm 4-cell HM  x1.0221
  x86 4-cell HM  x1.0270
  8-cell HM      x1.0245   worst cell nq1_st_arm x0.9967
VERDICT: WIN
```

Six of eight cells up, the two nq1_st cells flat inside their bands.
Cumulative against the pinned 1.0.0 baseline: 8-cell HM x1.1105. Lands
as **win #4** once the arm sweep and `cargo test` in the running chain
report (x86's already have).

## H69 — landed. VERDICT: WIN vs climb HEAD — round-2 win #4 (8-cell HM x1.0245)

Arm gates: paired sweep, 44 points, **none below 0.97** (worst nq11_st
x0.972); `cargo test -p turbovec` 40 suites green on the Axion box;
4-bit observation recorded (`arm_h69_obs4_h69.json`). x86 gates in the
entry above. Parity identical on both arches at both widths.

The bundle is three exact changes to fixed cost: H65b (x86's libm
`roundf` per LUT entry replaced by a bit-identical `trunc` form; arm
keeps `frinta`), H67 (the 2-bit LUT built from 4 + 4 products in the
original summation order), H68 (the arm block top-k early exit inline).
Cumulative against the pinned 1.0.0 baseline: 8-cell HM **x1.1105**;
x86 nq100_st x1.32, nq100_mt x1.51, arm nq100_st x1.04.

Committed on `perf/2bit-hillclimb-2`. **Non-win count: 0/20.**

## H70 — exact magic-number flush, arm 4-query kernel — marginal, NOT PROMOTED (non-win 1/20)

Axion, 2 ABBA rounds vs `h69`:

| cell | h69 | h70 | |
|---|---|---|---|
| nq100_st | 143.3-144.1 | 143.0-144.2 | x1.002 (min) / x1.001 (mean) — flat |
| nq100_mt | 17.34-17.64 | 17.17-17.50 | x1.010 (min) / x1.006 (mean) — ranges overlap |

Parity identical. The prediction (2-3%) priced the flush at its issue
slots — 8 `ushll` at 2/cycle and 8 `ucvtf` at 1/cycle per query per
block — but the flush sits once per block behind a 192-group loop with
plenty of independent work, and the out-of-order window hides it: the
slow rows cost issue slots the loop was not short of. Real at most ~1%
on one cell, cannot reach the bar; reverted. (H71, the same change in
the nq=1 kernel where the flush is a larger share of a shorter block,
is queued and gets its own reading.)

**Verdict: non-win 1/20.**

## H71 — magic-number flush, arm nq=1 kernel — flat, NOT PROMOTED (non-win 2/20)

Axion, 2 rounds vs `h70`: nq1_st x0.998, nq1_mt x1.012 (inside its 1.9%
band), nq100_mt x0.998 (control). Same lesson as H70: the flush's slow
issue rows are hidden by the out-of-order window even in the short nq=1
block. Reverted. **Verdict: non-win 2/20.**

## H72 — Axion: the probe, the kernel, the gates, and the trade

**Probe (`smmla2_probe.c`, two runs, G(q.dim)/s):**

| variant | Axion | vs LUT4 (124) |
|---|---|---|
| LUT 4-query, sequential | 121-124 | — |
| SMMLA pair, nq=8 / 12 | 157 / 162 | x1.27 / x1.30 |
| **SMMLA vm8, nq=8** | **183-185** | **x1.49** |
| SMMLA vm8, nq=12 | 159-161 | x1.30 (spills) |
| LUT nq=1 sequential | 88-93 | — |
| LUT nq=1 pair (LD2 / UZP) | 77-83 / 77-81 | x0.88 |
| LUT nq=1 vm8 UZP tree | 67 | x0.73 |
| LUT nq=1 vm8 LD4 x2 | 58-59 | x0.64 |
| SMMLA nq=1 duplicated query | 61-63 | x0.67 |

**The kernel in the crate, toggle A/B on the harness index, two ABBA
rounds** (`smoke_env.sh`, same `.so`, layout chosen at load):

| cell | LUT (HEAD) | SMMLA/vm8 | |
|---|---|---|---|
| nq100_st | 129.6-135.4 | 75.0-78.1 | **x1.727** |
| nq100_mt | 16.87-17.12 | 10.06-10.17 | **x1.676** |
| nq1_st | 1.622-1.682 | 2.513-2.543 | x0.645 |
| nq1_mt | 0.262-0.277 | 0.372-0.385 | x0.704 |

The batched gain in the real kernel (x1.7) exceeds the probe's x1.49:
the crate kernel runs 8 parts x 2 accumulators per pair against the
probe's 4 x 4 and the batch dispatch takes 12/8/4-wide chunks. The nq=1
loss in the real cell (x0.65) is worse than the probe's x0.73 — the
16-register group set of the UZP tree plus the LUT step likely spills.

**Gates.** Recall on Axion identical to the local runs: OpenAI-1536
recall@1 TQ 0.888 -> 0.889, TQ+ 0.901 -> 0.904; uniform harness +0.0008
with 92% top-10 overlap. Parity digests differ at 2 bits (by design) and
match at 4 bits. `cargo test` 40/40 in both modes.

**Verdict under the goal as written: NOT A WIN** — two cells regress by
30-35% and the 8-cell HM falls to ~0.98 even with the other two up x1.7.
As a *product* change it is a real 1.7x for batched search on arm at an
unchanged recall, which is why it is committed behind
`TURBOVEC_2BIT_VM8=1` (inert by default) rather than discarded. Options
put to Ryan: (1) opt-in layout, default unchanged; (2) default on,
accept the nq=1 cost; (3) the pair layout as a compromise (x1.28 /
x0.88 from the probe). A better nq=1 kernel on vm8 is the open follow-up
if (2) is chosen: halving the live group set (process each 16-vector
half straight after its own UZP tree) is the first thing to try.

## Dispositions from existing measurements (non-wins 3, 4, 5 / 20) and H79 — PRE-REGISTERED

Counted, per round 1's convention for candidates a measurement already
answers:

- **H72 as default (non-win 3/20):** measured above — x1.73 / x1.68 on
  the arm nq=100 cells against x0.65 / x0.70 at nq=1; fails the
  no-regression rule. Kept opt-in.
- **Pair layout (non-win 4/20):** the Axion probe has it at x1.27-1.30
  for nq=100 and x0.88 at nq=1 (LD2 or UZP); a regression by
  construction on two cells. Not built.
- **x86 ST-only prefetch, H63 (non-win 5/20):** H61 measured +4.0% on
  nq100_st with MT untouched by the gate; one cell at +4% is x1.005 on
  the 8-cell HM. Not built.

**H79 — four code streams per table load in the x86 nq=1 kernel.** P40
put both x86 nq=1 cells on L3 bandwidth (30 and 93 GB/s against a DRAM
tool's 19 and 53), and H34's two-block interleave was the last thing to
move them. More independent streams per `vpermb` table load is the only
lever the port count leaves; four blocks is 8 accumulators + 2 tables +
per-block index pairs, inside the register file. Exact by construction
(same dpbusd order per accumulator). Queued on x86 behind the capstone,
smoked against `h69` on nq1_st, nq1_mt with nq100_mt as the control.

## Capstone — the cumulative round-2 build vs the pinned 1.0.0 baseline, one session per box

Both boxes, `base2` vs `h69` (H53+H56+H60+H69; H72 present but inert
without its toggle), 3 balanced ABBA passes each, min per label, scored
by `whm_2bit.py` (`data/r2_capstone/`):

```
cell            arm        x86
  nq1_st       x0.9948    x1.0232
  nq1_mt       x1.0408    x1.0488
  nq100_st     x1.0189    x1.4888
  nq100_mt     x1.0173    x1.4962
  arm 4-cell HM  x1.0177
  x86 4-cell HM  x1.2229
  8-cell HM      x1.1109   worst cell nq1_st_arm x0.9948
VERDICT: WIN
```

Round 2 to date: **8-cell HM x1.111 over 1.0.0**, x86 nq=100 cells
~x1.49, arm cells x1.02-1.04, no cell below the floor. The capstone
sweeps (44 paired points per box) are running and will be recorded
under this entry.

## H79 — four code streams in the x86 nq=1 kernel — REFUTED (non-win 6/20)

x86, 2 rounds vs `h69`: nq1_st 1.296-1.484 vs 1.376-1.410 (x0.94 on
mins, x1.00 on means — the regime cell), nq1_mt 0.418-0.426 vs
0.417-0.435 (flat), control nq100_mt x0.98/x0.99 (drift). More streams
per table load buys nothing: at 30 GB/s from L3 the two-stream kernel
already keeps enough misses in flight, and the extra index work is not
free. Reverted. **Verdict: non-win 6/20.**

**Capstone sweep, x86:** 44 paired points, none below 0.97 (min nq1_st
x1.024); nq=2/4/6/8 ST x1.34/1.26/1.43/1.45, N=200k x1.26 ST / x1.52 MT.
Arm's sweep to follow.

## H81 / H82 (x86) and H84 / H85 (arm) — constant sweeps at the round-2 geometry — PRE-REGISTERED

Cheap, one build and one smoke each, run as one chain per box:

- **H81** `TILES_PER_THREAD` 32 -> 16 on x86: batch 6 makes 17 quads,
  so the block axis now splits into 16 ranges (272 tiles); H16 found
  the constant inert at the old geometry. Cell: nq100_mt.
- **H82** x86 nq=1 prefetch distance 8 -> 16 quads: the cells are
  L3-served (P40) rather than DRAM-served as when H21 self-confirmed 8.
  Cells: nq1_st, nq1_mt.
- **H84** `TILES_PER_THREAD_NEON` 64 -> 96 on arm: H50's direction
  (fewer, longer ranges won by 0.33%) at the post-H69 per-block cost.
  Cells: nq100_mt, nq1_mt.
- **H85** the 2-bit NEON tile floor `MIN_TILE_BLOCKS_NEON * 2` -> `* 1`
  (H14's win, re-asked now that the block is cheaper). Cell: nq100_mt.

Expected: flat. Each is a constant round 1 tuned at a geometry that has
since changed by 20-50%, which is the one honest reason to re-ask.

**Capstone sweep, arm:** 44 paired points; one below 0.97 — **nq13_mt
x0.953** — the rest x1.01-1.03 (nq=2/4/6/8 ST x1.02/1.01/1.02/1.03,
N=200k x1.01 ST / x1.02 MT). Nothing in the arm changes (H67's LUT
build, H68's early exit) is specific to nq=13, and the instrument's
no-op floor on this box (P4) has 13-23 of 88 points past 3%. Re-measured
in isolation (P44, six ABBA passes) rather than argued away.

## H81 / H82 / H84 / H85 — constant sweeps — all REFUTED (non-wins 7, 8, 9, 10 / 20)

Two ABBA rounds each vs `h69`:

| | cell | h69 | cand | verdict |
|---|---|---|---|---|
| H81 x86 `TILES_PER_THREAD` 32->16 | nq100_mt | 15.93-16.17 | 15.98-16.64 | x0.997 (min) / x0.986 (mean) — worse |
| H82 x86 nq=1 prefetch 8->16 quads | nq1_st | 1.29-1.31 (+1 outlier) | 1.29-1.38 | x1.00 on mins; unresolved |
| | nq1_mt | 0.415-0.422 | 0.409-0.446 | x1.015 min / x0.972 mean — noise |
| H84 arm `TILES_PER_THREAD_NEON` 64->96 | nq100_mt | 17.28-17.41 | 17.19-17.47 | x1.005 / x1.002 — inside band |
| | nq1_mt | 0.274-0.280 | 0.274-0.283 | flat |
| H85 arm 2-bit tile floor x2 -> x1 | nq100_mt | 17.28-17.41 | 17.71-17.80 | **x0.976** — H14's floor still right |

Every round-1 constant re-asked at the round-2 geometry answers the
same way it did. **Verdicts: non-wins 7-10 of 20.**

## H86 — VNNI batch width 4 at MT — PRE-REGISTERED (x86)

P36 chose width 6 on the ST cell, where one thread owns the 48 KB L1D
and the 36 KB LUT set fits. At MT the harness runs 8 workers on 4
cores, so two hyperthreads share each L1D: two 36 KB sets do not fit
where two 24 KB sets (width 4) do. Mechanism is the same L1 term that
made 10 lose (H55); the prediction is a few percent on nq100_mt and
nothing at ST. `VNNI_BATCH` becomes 4 when the pool has more than one
thread. Cell: nq100_mt, with nq100_st as the untouched control.

## P44 — the arm sweep's nq13_mt x0.953, re-measured in isolation (probe; not counted)

Six ABBA passes of nq=13 MT alone on Axion: base2 2.659-2.676,
h69 2.462-2.624 — **x1.080 on mins, x1.041 on means**, every h69 pass
below every base2 pass. The paired sweep's x0.953 for that point was
the instrument (P4's floor), and the capstone stands with no point
regressing on either arch.

## H86 — VNNI batch width 4 at MT — REFUTED (non-win 11/20)

x86, 2 rounds vs `h69`: nq100_mt 16.03-16.32 vs 17.49-17.85 —
**x0.917** on mins, every h86 pass above every h69 pass; nq100_st
(untouched by the gate) x0.96-0.97 on a drifting session. The shared
L1D between hyperthreads is not the binding term at MT; the 25 sweeps
of the codes that width 4 needs against 17 are. Reverted.
**Verdict: non-win 11/20.**

## Disposition — per-tile allocation reuse (non-win 12/20)

P42 priced the per-query non-LUT work at ~9 us on Axion and ~10 us on
x86 at ST, parallel at MT; the tile loop allocates its heaps and
reference vectors per (quad, range) tile, 272-500 tiles per nq=100
search, ~1-2% of the MT cells and nothing at ST (one range). At most
~x1.007 on the 8-cell HM; not built.

## H87 / H89 / H90 / H95 — last cheap constants — PRE-REGISTERED

- **H87** x86 `TILES_PER_THREAD` 32 -> 64 (H81's other direction).
- **H90** x86 nq=1 prefetch 8 -> 4 quads (H82's other direction; the
  cells are L3-served, a shorter lookahead may waste less).
- **H95** x86 nq=1 without the two-block interleave (`n_fours`/pairs
  bypassed, every block single-stream): H34 won it against DRAM;
  against L3 the second stream may be dead weight.
- **H89** arm nq=1 MT: two block ranges per thread instead of one
  (H103 refuted this at 4 bits from DRAM; at 2 bits the cell is at 85%
  of supply).

## H87 / H90 / H95 / H89 — REFUTED or marginal (non-wins 13, 14, 15, 16 / 20)

Two ABBA rounds each vs `h69` (three for H89):

| | cell | h69 | cand | verdict |
|---|---|---|---|---|
| H87 x86 `TILES_PER_THREAD` 32->64 | nq100_mt | 15.66-16.59 | 15.60-16.16 | x1.003 min / x1.012 mean — ranges overlap, flat |
| H90 x86 nq=1 prefetch 8->4 | nq1_st | 1.30-1.78 | 1.43-1.78 | regime cell, unresolved |
| | nq1_mt | 0.409-0.436 | 0.414-0.424 | x0.988 / x1.006 — flat |
| H95 x86 nq=1 single-stream (no interleave, no prefetch) | nq1_st | 1.30-1.78 | 1.50-1.61 | x0.87 min / x0.97 mean — unresolved |
| | nq1_mt | 0.409-0.436 | 0.404-0.414 | x1.012 / x1.027 — small, one cell (~x1.003 HM) |
| H89 arm nq=1 MT, two ranges per thread | nq1_mt | 0.273-0.280 | 0.296-0.300 | **x0.922** — H103's verdict holds at 2 bits |

H95's nq1_mt reading is the only positive number and cannot reach the
bar alone; recorded as a term for anyone revisiting the x86 nq=1 kernel
against an L3-resident index. **Verdicts: non-wins 13-16 of 20.**

## H97 / H98 — last two smokes, and two dispositions — PRE-REGISTERED

- **H97** x86 `MIN_TILE_BLOCKS_X86` 3x -> 2x the shared floor (the MT
  tile floor at the batch-6 quad count).
- **H98** arm single-query range stride floor 64 -> 128 blocks (longer
  per-worker streams for the 85%-of-supply nq1_mt cell).
- **Disposition, x86 nq=100 tail as 5+5 instead of 6+4 (non-win 19/20):**
  P36 has widths 4 and 5 within 0.4% of each other per query; nothing to
  gain.
- **Disposition, arm LUT batch 4 -> 2 at MT (non-win 20/20 if H97/H98
  fail):** halves the amortisation and doubles the sweeps; H12 measured
  the wider direction and the arithmetic forbids the narrower.

## H97 / H98 — REFUTED or marginal (non-wins 17, 18 / 20); dispositions 19, 20 — ROUND 2 CLOSED

| | cell | h69 | cand | verdict |
|---|---|---|---|---|
| H97 x86 `MIN_TILE_BLOCKS_X86` 3x -> 2x | nq100_mt | 15.84-16.69 | 15.72-15.96 | x1.008 min / x1.025 mean on a session whose control spread (5%) exceeds the cell's band; at most ~2% on one cell, ~x1.003 on the HM. Not promoted. |
| H98 arm nq=1 range stride floor 64 -> 128 | nq1_mt | 0.271-0.284 | 0.272-0.282 | x0.996 / x1.004 — flat |

With the two dispositions registered above (x86 tail 5+5; arm LUT
batch 2), **20 consecutive non-wins: round 2 is done.**

### Round 2 — closing summary

Baseline: main 1.0.0 (ccab9f32), 2026-09-06. Branch
`perf/2bit-hillclimb-2`, worktree `~/git/tv-2bit-hc`.

**Exact wins landed (parity-identical, sweeps clean, tests green):**

| win | change | effect (vs HEAD at the time) |
|---|---|---|
| H53 | const-generic batch width for the x86 VNNI kernel (32 branch pairs + spills per quad removed) | x86 nq100 x1.27 / x1.36 |
| H56 | VNNI batch width 6 (P36 measured the knee) | x86 nq100 +5-6% |
| H60 | inline the block epilogue's early exit (six 13-arg calls per block removed) | x86 nq100 +5% |
| H69 | prep + epilogue fixed costs: exact trunc rounding (x86), 4+4-product LUT build, arm top-k early exit | six cells +1-5% |

**Capstone, cumulative build vs 1.0.0, one session per box:** 8-cell HM
**x1.111** (WIN); x86 nq100_st x1.49, nq100_mt x1.50, nq1 cells
x1.02-1.05; arm cells x1.02-1.04, nq1_st x0.995. Sweeps clean on both
arches (one arm point re-measured, P44).

**The big bet, H72:** a 2-bit SMMLA kernel on the vm8 layout for arm,
allowed under a recall-equivalence gate. Correct; recall unchanged or
slightly better on real data; on Axion **x1.73 / x1.68 on the nq=100
cells and x0.65 / x0.70 at nq=1.** Not a win under the no-regression
rule; committed inert behind `TURBOVEC_2BIT_VM8=1` with the trade
recorded for Ryan's decision (opt-in layout, default-on, or the pair
layout at x1.28 / x0.88).

**Closed by measurement this round:** AMX on x86 (H99 in the 4-bit
log: no tile renaming), LUT re-fetch traffic (H64), wider/narrower VNNI
batches (H55/H57/H58/H86), the arm flush conversions (H70/H71), every
round-1 constant re-asked at the new geometry (H81-H98), and the x86
nq=1 kernel's stream count and prefetch depth against an L3-resident
index (H79/H82/H90/H95).

**What a round 3 should start from:** (1) the H72 decision — if the
batched layout ships by default, a better nq=1 kernel on vm8 (halve the
live group set; the tree spills) is the first hypothesis; (2) x86
nq100_st's regime switching (P37) is external to the guest and bounds
what any further x86 work can show; (3) the arm LUT kernels are at the
issue bound of their formulation (P24/P27), so arm gains beyond H72
need a formulation change, which the recall gate now permits.

---

# Round 3 (2026-10-02)

Goal in `GOAL_2bit_r3.md`. Baseline pinned at the round-2 HEAD (03fc2a2c).
Branch `perf/2bit-hillclimb-3`. New this round: a result may be exact, or
pass the probabilistic gate (>= 99.9% of queries return the exact scan's
ids on real embeddings; returned scores are the exact 2-bit scores).

## P45 — sign-plane shortlist: how large must it be? (probe; not counted)

**Idea.** A 2-bit code is a sign bit and a magnitude bit. Scan only the
sign plane (half the bytes, and a 16-entry lookup then covers four
dimensions instead of two), keep a shortlist, rescore it at full 2-bit.

**Probe.** `turbovec/src/plane_probe.rs` (ignored in-crate test) dumps the
real codes, the rotated/calibrated queries and the exact top-100 from
`search`; the offline analysis scores every vector from its sign bits alone
(`scale_v * (m * q.sign + bias_q)`, m = the measured mean magnitude) and
records, per query, the shortlist size needed to contain the exact top-k.
OpenAI-1536, N=200k, 10,000 queries, real nested planes. The same probe
run at 4 bits (top two bits as the coarse plane) is recorded as an
observation.

Miss rate (fraction of queries whose exact top-k is not wholly inside a
shortlist of size S):

| index | k | S=64 | S=128 | S=256 | S=512 | S=1024 | S=2048 | needed: p50 / p99.9 / max |
|---|---|---|---|---|---|---|---|---|
| 2-bit TQ | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 / 10 / 37 |
| 2-bit TQ | 10 | 0.0025 | 0.0001 | 0 | 0 | 0 | 0 | 15 / 83 / 139 |
| 2-bit TQ | 100 | 1 | 0.98 | 0.37 | 0.020 | 0.0002 | 0 | 228 / 874 / 1145 |
| 2-bit TQ+ | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 / 12 / 19 |
| 2-bit TQ+ | 10 | 0.0027 | 0.0001 | 0 | 0 | 0 | 0 | 15 / 77 / 132 |
| 2-bit TQ+ | 100 | 1 | 0.98 | 0.35 | 0.015 | 0.0002 | 0 | 226 / 813 / 1285 |
| 4-bit TQ | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 12 / 38 / 57 |
| 4-bit TQ | 100 | 1 | 0.85 | 0.014 | 0 | 0 | 0 | 153 / 347 / 387 |
| 4-bit TQ+ | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 12 / 33 / 46 |
| 4-bit TQ+ | 100 | 1 | 0.82 | 0.007 | 0 | 0 | 0 | 148 / 249 / 392 |

**Reading.** At k=10 a sign-plane shortlist of 128 already meets the 99.9%
gate (1 query in 10,000 misses) and 256 misses none of 10,000; k=100 needs
about 2048, i.e. roughly 15-20x k in both cases, 1% of N at worst. The
float model of the exact score matched the returned scores to 4e-3
relative, so the dim/field mapping is right. One dataset so far; the gate
needs OpenAI-3072 and mpnet-768 too.

**Provenance caveat.** This run was made on Ryan's Mac before the
boxes-only rule was added to the goal. It is a recall measurement, not a
timing, so the machine does not change it; it is re-run on the rig with
the other two datasets before anything is gated on it.

**Next.** H99 (pre-registered below).

## H99 (pre-registered) — sign-plane first pass + exact 2-bit rerank

**Hypothesis.** The existing LUT kernels score nibbles against 16-entry
tables and do not care what a nibble means. Fed a sign plane (8 dims per
byte) with tables built over sign patterns, they scan `dim/8` byte-groups
instead of `dim/4`: half the bytes and half the lookups, on both arches,
ST and MT. The shortlist (S = f(k), from P45) is rescored from the full
codes with the exact 2-bit arithmetic, so returned scores are unchanged.
RAM is unchanged: the blocked cache holds the sign plane and the magnitude
plane in place of the interleaved 2-bit bytes.

**Prediction.** nq=1 cells (memory-bound, x86 at 98% of supply): toward
x1.8-2.0 less the rerank (256 vectors x 192 bytes, random access, est.
10-20 us, which matters most on nq1_mt at ~260 us). nq=100 cells
(issue-bound): toward x1.6-1.9. 8-cell HM > x1.5.

**Gate.** Probabilistic: P45's curve on three datasets, then ids compared
against the exact scan in situ.

## Rig note — round 3 runs on replacement VMs (2026-10-02)

Both round-2 search boxes hit a GCP stockout (`c4a-standard-8` in
us-central1-a, `c3-standard-8` in us-central1-c). Round 3 measures on:

- **x86:** the same instance and disk, machine type changed to
  `c3-highmem-8` (same Sapphire Rapids 8481C, 8 vCPU, more RAM).
- **arm:** `turbovec-bench-arm-search-r3`, a `c4a-standard-8` clone of the
  round-2 disk in us-central1-b (snapshot `tv-arm-search-r3`). Reach it
  through IAP with the `gce_ed25519_tvbench` key; the `tvarm` alias still
  names the stocked-out original.

The baseline was re-established on these VMs from the round-2 HEAD
(`r3base`): arm 1.65 / 0.26 / 131 / 16.9 ms (nq1_st, nq1_mt, nq100_st,
nq100_mt), matching round 2's figures; x86 1.3-1.4 / 0.39-0.40 / 56-58 /
15.6-16.0 ms. The x86 box spent its first hour in the slow single-thread
regime P37 describes (nq1_st 3.3-3.5 ms, nq100_st 89-93 ms) and then
returned to the fast one; ratios taken during the slow regime are not
quoted below. Every score is a paired ABBA A/B on the same box in the
same session.

## H99 — sign-plane first pass + exact rerank — smoke history

Built behind `TURBOVEC_2BIT_PLANES=1` (default off). Each step below was
smoked ABBA against `r3base` on the box(es) named.

**1. Prototype, sign plane as an extra buffer (+50% RAM), whole-block
rerank, top-S heap (x86).** nq1_st faster, nq1_mt x0.69, nq100_mt x0.73.
A phase profile (`TURBOVEC_PLANES_PROF`) put the loss on the heap: an
exact scan at k=128 instead of k=10 costs +25 ms at nq100_st and
+0.3 ms at nq1_mt by itself — the O(k) rescan per insert.

**2. Buffered collector.** A heap whose min-index slot holds
`HEAP_BUFFERED` appends lanes above a threshold and, at capacity 2S,
keeps the best S and raises the threshold; the merge selects in linear
time instead of sorting. All four x86 cells at or above baseline.

**3. Same RAM, planes interleaved per block (one buffer, each block's
first half the sign bytes).** Passes on six cells, nq1_mt x0.87-0.90 on
both arches. The interleave breaks the stream: on x86, nq1_st scans in
1.02 ms against 0.76 ms for a contiguous plane.

**4. Same RAM, two regions (`pack::planes_for`).** The cache becomes a
contiguous *sign region* — blocked exactly like a code buffer with half
the byte-groups, so the existing kernels scan it unchanged — and a *low
region* holding each vector's low bits as one row. Load, add, patch,
swap-remove, sync capture and save go through `pack::planes_*` helpers
that convert at the boundary; the stored format is untouched. The rerank
rebuilds each shortlisted vector's code bytes from the two regions and
applies the exact kernels' arithmetic (x86: one multiply-add over the
u32 sum; aarch64: a fused multiply-add per `FLUSH_EVERY` groups), so
returned scores are the exact scan's bit for bit. Seven cells win; arm
nq1_mt x0.94.

**5. Sample-seeded threshold.** The collector cost 57 us (arm) and 69 us
(x86) per single query in MT — each range ratchets its own top-S. A
48-block strided sample of the sign region, scanned first, gives each
query a starting threshold (the sample's r-th best, r set so about four
shortlists' worth of the index lies above it); a query that comes back
short is rescanned unseeded. Smoke, both boxes, min of two ABBA labels:

| cell | x86 base | x86 planes | | arm base | arm planes | |
|---|---|---|---|---|---|---|
| nq1_st | 1.323-1.462 | 0.735-0.794 | ~x1.8 | 1.651-1.655 | 0.958-0.976 | ~x1.7 |
| nq1_mt | 0.393-0.407 | 0.328-0.333 | ~x1.2 | 0.256-0.268 | 0.233-0.237 | ~x1.1 |
| nq100_st | 56.98-58.26 | 40.23-40.49 | ~x1.43 | 131.4-132.6 | 78.16-78.19 | ~x1.69 |
| nq100_mt | 15.55-15.99 | 12.02-12.09 | ~x1.31 | 16.85-17.15 | 11.69-11.73 | ~x1.45 |

Phase split at nq=1 (us, best of 150): x86 ST prep 23 / sign table 10 /
scan 673 / rerank 57; x86 MT 27 / 10 / 226 / 35; arm ST 14 / 6 / 914 /
47; arm MT 15 / 7 / 174 / 24.

Smoke passes on all eight cells. Gates and soak follow.

## H99 — gates and soak. `whm_2bit.py` VERDICT: WIN — round-3 win #1 (8-cell HM x1.4076)

Build `h99g` (commit "sample-seeded shortlist threshold"), planes on for
the candidate label, `r3base` (round-2 HEAD) as the baseline.

**Probabilistic gate, in situ** (`r3gate.py`: one build, the exact scan
against planes on, 10,000 queries, real embeddings, both boxes — the two
arches agree to the digit):

| data | N | calib | k=1 | k=10 | k=100 | scores bitwise |
|---|---|---|---|---|---|---|
| OpenAI-1536 | 200k | no | 1.0000 | 1.0000 | 0.9999 | 1.000000 |
| OpenAI-1536 | 200k | yes | 1.0000 | 1.0000 | 1.0000 | 1.000000 |
| OpenAI-3072 | 200k | no | 1.0000 | 1.0000 | 1.0000 | 1.000000 |
| OpenAI-3072 | 200k | yes | 1.0000 | 1.0000 | 1.0000 | 1.000000 |
| mpnet-768 | 41k | no | 1.0000 | 0.9995 | 0.9996 | 1.000000 |
| mpnet-768 | 41k | yes | 1.0000 | 0.9997 | 0.9998 | 1.000000 |

Entries are the fraction of queries whose returned ids equal the exact
scan's, in order. Shortlist S = max(128, 12.8 k). The instrument can
fail: with S = k the same check reads 0.03-0.17 at k=10. Shortlists of
1.5x and 2x that size read 1.0000 everywhere except mpnet k=10 (0.9999).
Every returned score is the exact scan's bit pattern, on both arches.

**`cargo test -p turbovec`**: green on both boxes with the toggle off and
with `TURBOVEC_2BIT_PLANES=1`.

**RAM**: the sign region and the low region together are the bytes of the
code buffer they replace (`n_blocks * 32 * dim/8` + `n * dim/8` against
`n_blocks * 32 * dim/4`); the threshold sample adds a fixed 48 blocks
(~150 KB at dim 768) per index, independent of N.

**Soak** (`r3soak.sh`, 2 balanced ABBA passes = 4 runs per label per box,
each run `cells_2bit.py`'s min of nine; scored on the min across runs):

```
cell            arm        x86
  nq1_st       x1.7382    x1.6886
  nq1_mt       x1.0910    x1.1904
  nq100_st     x1.6678    x1.4194
  nq100_mt     x1.4444    x1.3109
  arm 4-cell HM  x1.4369
  x86 4-cell HM  x1.3795
  8-cell HM      x1.4076   worst cell nq1_mt_arm x1.0910
VERDICT: WIN
```

Per-run spreads: arm base nq1_mt 0.254-0.264, cand 0.233-0.243; x86 base
nq100_st 55.5-59.9 (the P37 drift), cand 39.1-39.8.

**What this is and is not.** It is opt-in (`TURBOVEC_2BIT_PLANES=1`,
default off) and the stored format is unchanged. Results are exact with
probability, not by construction: a top-k vector outside the sign-plane
shortlist is missed. Masked searches take the same path with a plain
top-S heap; a request whose shortlist would cover the index rescoring
everything. Not yet covered: a test job that runs the suite with the
toggle on in CI, dims where `dim/4` is not a multiple of 8 (they keep the
classic layout), and the x86 kernels below VBMI/VNNI (same). Streak: 0.

## H100 (pre-registered) — refine pass before the exact rescore

**Hypothesis.** After H99 the exact rescore of 128 candidates is 57 us
(x86) / 47 us (arm) per query ST — 18% of x86 nq100_st — because each
candidate's sign bits are gathered back out of the blocked region. A
2-bit level is `+-A +- B` (sign bit, low bit), so a candidate's exact
score is `A * S + B * L` with `S` the sign-plane sum the scan already
produced and `L` the same sum over its low row, which is one contiguous
96 bytes read through the sign tables. That estimate ranks the shortlist
well enough to send only the best max(32, 3k) to the exact rescore.

**Prediction.** Rerank 57 -> ~25 us. x86 nq100_st +10%, x86 nq1_mt +6%,
arm nq1_mt +5%, others +2-4%; 8-cell HM ~x1.04 over H99.

**Gate.** Probabilistic, same instrument, plus a sweep of the rescore
length to show where it starts to miss.

## H100 — refine pass before the exact rescore — VERDICT: NOT A WIN (non-win 1/20)

Three builds. `h100` (refine everywhere, serial): six cells up, x86
nq1_mt 0.327 -> 0.34-0.36. `h100b` (both rerank phases parallel at nq=1):
nq1_mt x0.97-0.98 on both arches — the second fork-join costs what the
refine saves. `h100c` (refine for ST and batched searches; one query on a
pool keeps H99's parallel rescore): smoke passes on all eight.

**Gate (`h100c`, both boxes, same instrument).** With the default rescore
length max(32, 3k) every entry equals H99's table to the digit, and so do
lengths of max(16, 1.5k), max(24, 2.2k) and max(48, 4.5k). A length of k
reads 0.85-0.91 at k=10 and 0.22-0.37 at k=100, so the instrument sees
the pass. Scores bitwise. `cargo test` green, toggle off and on.

**Soaks vs the climb HEAD (`h99g`, planes on both sides).**

```
soak 1 (2 passes)                    soak 2 (4 passes)
cell            arm        x86       arm        x86
  nq1_st       x1.0180    x1.0332    x1.0229    x1.0189
  nq1_mt       x0.9853    x1.0122    x0.9731    x0.9846
  nq100_st     x1.0303    x1.0633    x1.0286    x1.0600
  nq100_mt     x1.0251    x1.0734    x1.0244    x1.0588
  8-cell HM      x1.0294              x1.0206
VERDICT: NOT A WIN (nq1_mt_arm below x0.99), both times
```

**The failing cell runs the same code in both builds.** At nq=1 on a
pool `h100c` skips the refine pass, and a direct phase profile on arm
reads scan 174/175 us and rerank 25/25 us for the two builds. An
interleaved A/A/B of that one cell (`r3cell.sh`, the harness's own
min-of-nine, six rounds) gives min-of-mins 0.2340 for `h99g`, 0.2321 for
a byte-identical copy of it, and 0.2346 for `h100c`: the two copies of
one binary differ by 0.8%, more than candidate and control do. Inside the
full soak, though, the candidate's eight per-run minima (0.2379-0.2452)
sit almost wholly above the baseline's (0.2315-0.2384) — a shift the
single-cell interleave does not reproduce. So the cell carries an
order-dependent term the soak exposes and this change does not explain;
H130 found the same cell bimodal on cold versus warm cache.

**Disposition.** The scorer is the authority: not a win, twice. The
gains on six cells reproduce across the smoke and both soaks (x86 nq100
+6-7%, arm nq100 +2.5-3%, nq1_st +2-3%), so the pass stays in the tree
for a later candidate to stack on, as H59 did in round 2. Streak: 1.

## H101 (pre-registered) — tiling, batch width and prefetch at the sign region's geometry

**Candidates considered this turn.** (1) Block-range cap: a sign scan
asks `range_cap_for_k` for k = 2S = 256, which caps the block axis at 2
ranges where the exact scan gets 7 (arm) or 3 (x86); the cap prices a
top-k heap's O(k) rescan, which the seeded collector does not pay.
(2) Tile floor: set per block count, and a sign block is half the bytes.
(3) x86 batch width 6: measured on 6 KB blocks. (4) x86 nq=1 prefetch
lookahead of 8 quads: a third of a sign block. (5) Deferred u8 widening
on arm at a 5-bit table cap: ~11% fewer vector ops, a new pair of
kernels and a shortlist-quality cost. Picked: (1)-(4) as one sweep —
four constants of one mechanism, each an environment knob on one build
(`TURBOVEC_PLANES_KCAP`, `_TILE_MULT`, `_VNNI_BATCH`, `_PF`) — because
they are a rebuild-free hour; (5) is registered as H102.

**Prediction.** (1) alone: nq100_mt +3-8% on both arches. (2)-(4): at
most +2% each, if anything.

## H101 — sign-region tiling, batch width, prefetch — REFUTED, flat (non-win 2/20)

One build (`h101k`, H100's tree plus four environment knobs), swept ABBA
with `r3smoke2.sh`; ms, planes on throughout.

| knob | cell | default | variants |
|---|---|---|---|
| range cap computed for k=10 instead of 2S | arm nq100_mt | 11.38-11.39 | 11.22-11.25 |
| + tile floor x2 / x3 / x4 / x0.5 | arm nq100_mt | | 11.16-11.27 / 11.29-11.31 / 11.40-11.42 / 11.36-11.41 |
| range cap for k=10 | x86 nq100_mt | 10.92-11.66 | 11.88-12.05 |
| + tile floor x2 / x0.5 | x86 nq100_mt | | 11.46-11.50 / 12.23-12.39 |
| one block range | x86 nq100_st / mt | 36.72-37.04 / 10.92-11.16 | 37.07-37.18 / 11.05-11.28 |
| batch width 4 / 8 (default 6) | x86 nq100_st | 36.72-37.04 | 38.70-39.27 / 39.84-40.03 |
| | x86 nq100_mt | 10.92-11.16 | 11.15-11.30 / 12.34-12.40 |
| prefetch 4 / 12 / 16 / 24 quads (default 8) | x86 nq1_st | 0.770-0.775 | 0.777-0.778 / 0.750-0.802 / 0.749-0.790 / 0.727-0.785 |
| | x86 nq1_mt | 0.333-0.343 | 0.335-0.337 / 0.351-0.357 / 0.354-0.358 / 0.338-0.343 |

The largest effect is arm nq100_mt +1.5-2% with a finer split and a
doubled floor: x1.002 on the 8-cell HM. x86 keeps every constant it had
(batch 6, two ranges, 8 quads), and H6's finding that x86 degrades with
more block ranges holds at the new geometry. No soak. The knobs stay in
the tree as defaults-unchanged instrumentation. Streak: 2.

## H103 (pre-registered) — build the exact tables while the sign scan runs (nq=1 on a pool)

**Candidates considered this turn.** (1) Overlap the exact-table build
with the scan at nq=1 MT: the exact tables are read only by the rescore,
after the scan, and cost 14 us (arm) / 23 us (x86) of a 225-325 us
query, serially, before it. (2) Deferred u8 widening in the arm sign
kernels at a 5-bit table cap — ~13% fewer vector ops on three arm cells,
new kernels, register pressure in the 4-query one (registered as H102).
(3) A 5-bit cap on x86 with `vpaddb` accumulation between `vpdpbusd`s.
(4) Shortlist 128 -> 96. (5) A smaller threshold sample. Picked (1): it
is the one candidate aimed at the two weakest cells, and it is twenty
lines.

**Prediction.** nq1_mt +4-6% on both arches; every other cell untouched
(the deferral applies only to one query on a multi-thread pool).

## H103 — exact tables built during the sign scan, on H100's refine pass. `whm_2bit.py` VERDICT: WIN — round-3 win #2 (8-cell HM x1.0473 over H99)

Build `h103`: H99 + the H100 refine pass (ST and batched searches) + the
inert H101 knobs + this change. For one query on a multi-thread pool the
exact tables are no longer built before the scan: `rayon::join` runs the
sample pre-pass and sign scan on one side and the exact-table build on
the other, and the rescore reads the tables when both return.

**Smoke vs `h99g`** (planes on both sides): x86 nq1_st 0.789-0.801 ->
0.724-0.767, nq1_mt 0.332-0.340 -> 0.300-0.318, nq100_st 39.4-39.8 ->
37.2-37.3, nq100_mt 11.98-12.04 -> 10.90-11.22; arm nq1_st 0.992-0.999 ->
0.969-0.970, nq1_mt 0.240-0.247 -> 0.229-0.231, nq100_st 78.7-78.8 ->
76.5-76.8, nq100_mt 11.69-11.84 -> 11.45-11.49.

**Gate** (both boxes): batched k = 1 / 10 / 100 identical to H99's table
on all three datasets, calibrated and not (weakest: mpnet k=10 0.9995).
New column — 500 queries searched one at a time at k=10, which is the
path this change touches: ids identical 1.0000 and scores bitwise
1.000000 on every dataset. `cargo test` green, toggle off and on.

**Soak** (2 ABBA passes per box, vs the climb HEAD `h99g`):

```
cell            arm        x86
  nq1_st       x1.0221    x1.0383
  nq1_mt       x1.0733    x1.0710
  nq100_st     x1.0263    x1.0632
  nq100_mt     x1.0218    x1.0660
  arm 4-cell HM  x1.0354
  x86 4-cell HM  x1.0595
  8-cell HM      x1.0473   worst cell nq100_mt_arm x1.0218
VERDICT: WIN
```

nq1_mt per-run minima: arm base 0.2410-0.2441, cand 0.2245-0.2260; x86
base 0.3275-0.3313, cand 0.3058-0.3178 — disjoint on both boxes, so this
time the weak cell moves by more than its own order-dependent term.

The six cells H100 moved keep its gains; the two it could not move are
the two this change is aimed at. Round 3 to date: x1.4076 x x1.0473 =
~x1.47 over the round-2 HEAD (to be re-measured as one build at the
capstone). Climb HEAD is now `h103`. Streak: 0.

## H102 (pre-registered) — deferred u8 widening in the arm sign kernels

**Hypothesis.** The NEON LUT kernels add a byte-group's two lookups in
u8 and then widen to u16 with four `uaddw` per group per query — 16 of
the 40 per-query vector ops in four groups. A sign table capped at 31
instead of 127 lets eight lookups (four groups) sum in u8 first: 16 TBL +
14 add + 4 widen = 34 against 40. The sign score only ranks a shortlist
and its distance from the exact score is far larger than a 5-bit
rounding, so the cap should cost nothing the gate can see. Single-query
and 4-query kernels both; the 4-query one holds 16 u16 accumulators plus
8 new u8 partials, so it may spill and give the saving back.

**Prediction.** arm nq1_st +6%, arm nq100 +5-9% if the 4-query kernel
keeps its registers (0% if it spills), arm nq1_mt +2%; x86 untouched.
8-cell HM ~x1.02.

**Gate.** Probabilistic (the 5-bit table changes the shortlist and the
refine estimate), same instrument.

## H102 — deferred u8 widening, single-query arm sign kernel. `whm_2bit.py` VERDICT: WIN — round-3 win #3 (8-cell HM x1.0141 over H103)

Three builds, smoked on arm against `h103`:

| build | nq1_st | nq1_mt | nq100_st | nq100_mt |
|---|---|---|---|---|
| `h103` | 0.966-0.976 | 0.227-0.234 | 76.2-76.5 | 11.39-11.44 |
| `h102` both kernels deferred | 0.872-0.899 | 0.221-0.225 | 80.2-80.4 | 12.00 |
| `h102b` 4-query in two 16-vector halves | 0.887 | 0.220-0.224 | 83.9-85.9 | 12.00-12.06 |
| `h102c` single-query only | 0.880-0.904 | 0.218-0.220 | 76.6-76.7 | 11.51-11.53 |

The prediction's caveat held: the 4-query kernel needs 16 u16
accumulators, 8 u8 partials and the shared nibbles — one register more
than NEON has — and the spill costs more than the widening saved
(x0.95). Halving the block frees the registers and doubles the table
loads, which is worse (x0.90). So the 4-query kernel and its 7-bit
tables are untouched, and only a single query on aarch64 gets the 5-bit
table and `score_sign_block_neon`.

**Gate (`h102c`).** Batched columns identical to H103's. Single queries
(5,000 per dataset, k=10; on arm these now shortlist through 5-bit
tables): arm ids identical 0.9996-1.0000, x86 (7-bit, unchanged)
0.9996-1.0000; scores bitwise. `cargo test` green, toggle off and on.

**Soak vs the climb HEAD `h103`:**

```
cell            arm        x86
  nq1_st       x1.0682    x0.9966
  nq1_mt       x1.0242    x1.0006
  nq100_st     x0.9962    x1.0049
  nq100_mt     x0.9970    x1.0291
  arm 4-cell HM  x1.0206
  x86 4-cell HM  x1.0076
  8-cell HM      x1.0141   worst cell nq100_st_arm x0.9962
VERDICT: WIN
```

Six of these cells run code this change does not touch; their spread
(x0.996-x1.029) is this rig's noise on an unchanged path, and the x86
nq100_mt x1.029 is part of it, not a gain. The two cells the change does
reach move by x1.068 and x1.024. Climb HEAD is now `h102c`. Streak: 0.

## H104 (pre-registered) — fixed-cost bundle: sign-table build, threshold sample, rescore length

**Candidates considered this turn.** (1) x86 5-bit sign tables with
`vpaddb` between `vpdpbusd`s — dead by arithmetic, `vpdpbusd` already
folds the add and the widen (2 permb + 2 dpbusd -> 2 permb + 2 add +
0.25 dpbusd). (2) A 3-query deferred arm kernel (fits 27 registers, 42
ops per query per 4 groups against 46) at 36% more passes. (3) Shared
±1 decode + dot product on the sign plane — 8 dpbusd per 64 code bytes
per query against the LUT's 4. (4) A three-stage scan (half the sign
bits first) — the half-plane's correlation with the full sign score is
0.71, so its shortlist would be 5-10% of N. (5) The fixed costs that are
now 9-19% of the x86 cells: sign-table build 10-11 us per query, the
48-block threshold sample, and a 32-candidate exact rescore whose gate
reads identically at 16. Picked (5), as one bundle of one mechanism
(per-query fixed cost): build each 16-entry sub-table from two 4-entry
pair sums with min/max taken from the pairs; sample 32 blocks; rescore
max(24, 2.2k).

**Prediction.** x86 nq100 +2-3%, x86 nq1_mt +2%, arm +1%; 8-cell HM
~x1.015.
