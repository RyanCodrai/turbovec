# 2-bit search hill-climb — results log

Goal in `GOAL_2bit.md`. Objective: HM of 8 cells, `{arm, x86} x {ST, MT} x
{nq=1, nq=100}` at `bit_width=2`, N=200k, dim=768, k=10.

Harness: `cells_2bit.py` (objective, `--bits 4` for the observation run),
`sweep_2bit.py` (nq and N gates), `parity_2bit.py` (digests), `whm_2bit.py`
(scorer and verdict).

**Baseline: not yet pinned.** Everything below is code study and predictions
made before measuring, which is the point: they are refutable.

## Blocker — no rig access (project-wide)

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

Candidate fixes, owner-only: restore that service account's bindings; or set
`enable-oslogin=FALSE` per instance so metadata keys apply; or reset the
instances in case the guest agent is merely wedged.

**Next three actions once SSH works:** pin the baseline (three interleaved
rounds per cell, both boxes), run P1, then the H1/H2 A/B.

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
| nq1_st | **1.933** | **1.669** |
| nq1_mt | **0.302** | **0.502** |
| nq100_st | **148.770** | **83.958** |
| nq100_mt | **18.441** | **26.026** |

Spread across rounds is under 1% except x86 nq100_mt (~5%).

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

### P1 (as originally queued) — does 2-bit inherit 4-bit's memory-bound verdict?

P42 in `LOG_search.md` put arm nq=1 at 95% of the single-core streaming
roofline at 4 bits. At 2 bits the code array is 38.4 MB against 76.8 MB. If
the cell is still bandwidth-bound, time should track the byte ratio; the
interesting outcome is the one where it does **not**, because the leftover is
compute headroom that 4 bits does not have and every hypothesis above is
competing for it.

Measure: ns/(query·vector) at 2 and 4 bits, all four cells, both arches.
