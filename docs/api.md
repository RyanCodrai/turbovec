# API Reference

turbovec exposes two index types and one serialization format per type.

- [`TurboQuantIndex`](#turboquantindex) — positional index, O(1) `swap_remove` delete.
- [`IdMapIndex`](#idmapindex) — stable external `u64` ids on top of `TurboQuantIndex`.
- [TQ+ calibration](#tq-calibration) — the per-coordinate calibration lifecycle.
- [File formats](#file-formats) — `.tv` and `.tvim`, plus [incremental saves](#incremental-saves--sync).

All examples below are Python. The Rust API mirrors it closely (exceptions noted below) — see each type's rustdoc for the exact signatures.

---

## `TurboQuantIndex`

Positional index. Each vector is identified by its insertion slot (`0..n`). Fast and small, but external references to slots are invalidated by `swap_remove`. If you need stable ids, use [`IdMapIndex`](#idmapindex).

```python
from turbovec import TurboQuantIndex

idx = TurboQuantIndex(dim=1536, bit_width=4)
idx.add(vectors)                        # np.ndarray of shape (n, dim), float32
scores, indices = idx.search(queries, k=10)

idx.swap_remove(5)                      # O(1); the previously-last vector moves into slot 5

idx.write("index.tv")                   # .tv format
loaded = TurboQuantIndex.load("index.tv")
```

`dim` is optional. Omit it to let the index pick up the dimensionality from the first batch of vectors:

```python
idx = TurboQuantIndex(bit_width=4)      # dim inferred on first add
idx.add(vectors)                         # locks dim to vectors.shape[1]
```

Before the first add, `idx.dim` is `None`, `len(idx)` is `0`, and `search()` returns empty results. Adding a zero-row batch is a no-op: `dim` is still checked against the batch, but a lazy index stays lazy and its serialized bytes are unchanged. (On the Rust API, `dim_opt()` is the equivalent of `idx.dim` and returns `Option<usize>`; `dim()` is deprecated — it returns `usize` with `0` for a lazy index, which is unsafe to do arithmetic with, so use `dim_opt()` on any path that can see one.)

### Methods

| Method | Notes |
|---|---|
| `TurboQuantIndex(dim=None, bit_width=4)` | `bit_width ∈ {2, 3, 4}`. `dim` must be a positive multiple of 8 and `≤ 16384` (`MAX_DIM`). `dim` is optional; when omitted it is inferred from the first `add` call. |
| `add(vectors)` | `vectors` is a contiguous float32 array of shape `(n, dim)`. On a lazy index the first call locks `dim`; subsequent calls must match. Raises `ValueError` on dim mismatch, a zero-width (0-column) batch, or any coordinate that is non-finite (NaN/Inf) or `\|value\| ≥ 1e16`. A vector whose L2 norm is at or below `1e-10` has no representable direction and is stored with scale 0, scoring 0 against every query. On the Rust API, a lazy index's first add must use `add_2d(vectors, dim)` — the flat `add(&[f32])` requires an already-committed dim and panics otherwise. (Python arrays carry their shape, so this applies to Rust only.) |
| `search(queries, k, *, mask=None)` | Returns `(scores, indices)`, both shape `(nq, effective_k)`. Indices are `int64` slot positions. `mask` is an optional `bool` array of length `len(idx)`; when given, only slots with `mask[i] == True` contribute. `effective_k = min(k, mask.sum())`. Raises `ValueError` on a non-finite or `\|value\| ≥ 1e16` query coordinate. The returned ids are invariant to multiplying a query by any positive constant, for as long as the scaled coordinates stay within float32's normal range (smallest normal `1.18e-38`). Scale a query far enough that its coordinates go subnormal and they lose relative precision before scoring, so the ranking can change — measured at query magnitude `~1e-36` for `dim=256` and `~1e-35` for `dim=768`, well below any realistic embedding. The scores are inner products, so they scale with the query. |
| `swap_remove(idx)` | O(1). Moves the last vector into `idx`; returns the previous position of that moved vector (so external refs can be updated if needed). |
| `prepare()` | Optional. Eagerly builds the rotation matrix, Lloyd-Max centroids and SIMD-blocked layout so the first `search` call doesn't pay the one-time cost. No-op on a lazy index that hasn't seen its first add. |
| `sync(path)` | Incremental save: writes only what changed since the last sync to the same path. See [Incremental saves](#incremental-saves--sync). `load(path)` reads both formats. |
| `write(path, *, durable=True)` / `load(path)` | `.tv` format. `durable=False` skips the fsync before the atomic rename — faster, but a power loss can lose the file. A `durable=True` save whose post-rename directory fsync fails still succeeds (the file is committed and visible) and raises a `RuntimeWarning` saying the rename may not survive power loss. On the Rust API this is not a flag: use `write_with_durability(path, io::Durability::Fast \| Durable)`. |
| `to_bytes()` / `from_bytes(data)` | In-memory `.tv` serialization — see [In-memory serialization](#in-memory-serialization). |
| `pickle` / `copy.copy` / `copy.deepcopy` | Supported on both index types via `__reduce__`; see [In-memory serialization](#in-memory-serialization). Indexes are also weakly referenceable, so one can be cached in a `weakref.WeakValueDictionary`. |
| `len(idx)` / `idx.dim` / `idx.bit_width` | Introspection. `idx.dim` returns `int` once committed, or `None` on a lazy index that hasn't seen its first add. |
| `idx.calibration_state` | TQ+ calibration state: `"uncalibrated"` or `"calibrated"` — see [TQ+ calibration](#tq-calibration). |

### `swap_remove` semantics

`swap_remove(i)` is named to match Rust's [`Vec::swap_remove`](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.swap_remove): the last element moves into slot `i`, and the vector is truncated by one. It is **not** a shift — the slots after `i` do not move down by one. Order is not preserved; slot indices of vectors you didn't delete may now point at different vectors than before.

Use [`IdMapIndex`](#idmapindex) if external references have to stay stable across deletes. Search masks are external references too — see [A mask is invalidated by any mutation](#a-mask-is-invalidated-by-any-mutation-not-just-a-length-changing-one).

### Low-level construction from raw parts (Rust)

Rust embedders that hold an index payload already in memory — e.g. read out of a database page instead of a `.tv` file — can construct an index directly from its decoded fields with `TurboQuantIndex::from_parts`, skipping the file round-trip:

```rust
let index = TurboQuantIndex::from_parts(
    dim_opt,        // Option<usize>: Some(dim) committed, or None for lazy
    bit_width,      // 2, 3, or 4
    n_vectors,
    packed_codes,   // Vec<u8>
    scales,         // Vec<f32>
    tqplus_shift,   // Vec<f32> (length dim, or empty = identity)
    tqplus_scale,   // Vec<f32> (length dim, or empty = identity)
)?;
```

It is the single validated entry point for raw-part construction: every structural invariant is checked once and any violation returns a named `FromPartsError` (bit_width range, dim a positive multiple of 8 and `≤ 16384`, `packed_codes` / `scales` / TQ+ lengths with overflow-checked size math, the lazy-state constraints, and the same value-level validation as the file loader — finite non-negative per-vector scales, finite TQ+ shifts, finite positive TQ+ scales) rather than panicking or reading out of bounds. An index accepted by `from_parts` therefore always survives its own `write` → `load` round-trip. The paired accessors `packed_codes()`, `scales()`, `tqplus_shift()`, `tqplus_scale()`, `bit_width()`, `dim_opt()` and `len()` return the fields it consumes, so an index round-trips through your own storage format. The per-coordinate `encode` / `pack` / `search` / `codebook` kernels are crate-internal — `from_parts` is the supported low-level API. (Rust only; the Python binding uses `write` / `load`.)

---

## `IdMapIndex`

Stable-id wrapper around `TurboQuantIndex`: a hash-table-backed `u64 id ↔ slot` mapping, with O(1) `remove(id)`. Slot indices still move when a vector is removed, but ids do not.

```python
import numpy as np
from turbovec import IdMapIndex

idx = IdMapIndex(dim=1536, bit_width=4)
idx.add_with_ids(vectors, np.array([1001, 1002, 1003], dtype=np.uint64))

scores, ids = idx.search(queries, k=10)   # ids are uint64 external ids

idx.remove(1002)                           # O(1) by id
assert 1003 in idx                         # __contains__ sugar

idx.write("index.tvim")                    # .tvim format
loaded = IdMapIndex.load("index.tvim")
```

As with [`TurboQuantIndex`](#turboquantindex), `dim` is optional and gets inferred from the first `add_with_ids` call:

```python
idx = IdMapIndex(bit_width=4)            # dim inferred on first add
idx.add_with_ids(vectors, ids)           # locks dim to vectors.shape[1]
```

### Methods

| Method | Notes |
|---|---|
| `IdMapIndex(dim=None, bit_width=4)` | `bit_width ∈ {2, 3, 4}`; `dim` must be a positive multiple of 8 and `≤ 16384`. `dim` is optional; when omitted it is inferred from the first `add_with_ids` call. |
| `add_with_ids(vectors, ids)` | `ids` is a `uint64` array with length `vectors.shape[0]`. On a lazy index the first call locks `dim`. Raises `ValueError` on dim mismatch, duplicate ids, `len(ids) != vectors.shape[0]`, a zero-width batch, or a non-finite / `\|value\| ≥ 1e16` coordinate. On the Rust API, a lazy index's first add must use `add_with_ids_2d(vectors, dim, ids)` — the flat `add_with_ids` requires an already-committed dim and panics otherwise. (Rust only; Python arrays carry their shape.) |
| `remove(id) -> bool` | `True` if the id was present and removed, `False` otherwise. O(1). |
| `search(queries, k, *, allowlist=None)` | Returns `(scores, ids)` — `ids` are `uint64` external ids. `allowlist` is an optional `uint64` array of ids; when given, results are restricted to those ids and `effective_k = min(k, number of unique ids in allowlist)` (the allowlist is deduplicated; repeated ids don't widen the result). Raises `ValueError` on an empty allowlist or a non-finite / `\|value\| ≥ 1e16` query coordinate, and `KeyError` on unknown ids. On the Rust API `search_with_allowlist` returns `Result<(Vec<f32>, Vec<u64>), SearchError>` and reports every one of those conditions as `Err`: `AllowlistEmpty`, `UnknownId(id)`, and the query-shape pair `QueryBufferNotMultipleOfDim` / `InvalidQueryValue`. The allowlist-free `search` returns the tuple directly and is the panicking form — it re-panics with the same message on the two query-shape conditions. Rust callers who want the row count and the *effective* `k` rather than a bare tuple use `try_search` / `try_search_with_allowlist`, which return `Result<IdSearchResults, SearchError>` — the id-space counterpart of `SearchResults`, with `scores`, `ids`, `nq`, `k` and `scores_for_query` / `ids_for_query` row accessors. |
| `contains(id)` / `id in idx` | Membership. |
| `sync(path)` | Incremental save: writes only what changed since the last sync to the same path. See [Incremental saves](#incremental-saves--sync). `load(path)` reads both formats. |
| `write(path, *, durable=True)` / `load(path)` | `.tvim` format. `durable=False` skips the fsync before the atomic rename — faster, but a power loss can lose the file. A `durable=True` save whose post-rename directory fsync fails still succeeds (the file is committed and visible) and raises a `RuntimeWarning` saying the rename may not survive power loss. On the Rust API this is not a flag: use `write_with_durability(path, io::Durability::Fast \| Durable)`. |
| `to_bytes()` / `from_bytes(data)` | In-memory `.tvim` serialization — see [In-memory serialization](#in-memory-serialization). |
| `pickle` / `copy.copy` / `copy.deepcopy` | Same as `TurboQuantIndex`. |
| `len(idx)` / `idx.dim` / `idx.bit_width` / `idx.calibration_state` | Same as `TurboQuantIndex`. |
| `prepare()` | As `TurboQuantIndex.prepare()`, and additionally warms the lazy `id -> slot` map, so the first `search(..., allowlist=)`, `contains()` or `remove()` after a load doesn't pay the one-time O(n) build either. |

### When to use which

- `TurboQuantIndex` — you never delete, or you're fine with positional ids.
- `IdMapIndex` — you need stable external ids (e.g. string-id → vector mapping maintained by the caller).

All the framework integrations (LangChain, LlamaIndex, Haystack) use `IdMapIndex` internally for exactly this reason.

---

## TQ+ calibration

TQ+ fits a per-coordinate `(shift, scale)` pair from the empirical quantiles of a sample of your vectors, and every stored vector is encoded in that one calibrated coordinate system. It is worth roughly +2.5 points of R@10 on average, and up to ~8.7 on the most anisotropic data measured.

The calibration comes from exactly one place: an explicit `idx.calibrate(sample)` call (`calibrate` / `calibrate_2d` in Rust). The index never fits one on its own — an index that is never calibrated is plain TurboQuant, with no fitted state anywhere, and its encoded bytes are independent of how adds were batched or ordered. `idx.calibration_state` reports which of the two states an index is in:

| state | meaning |
|---|---|
| `"uncalibrated"` | No calibration committed. Fully functional, just without the TQ+ recall gain. |
| `"calibrated"` | A calibration is committed, and every stored row is encoded under it — including rows added *before* the `calibrate` call, which that call re-encoded. |

**The sample is your responsibility.** `calibrate` uses every row you give it. Around 1024 rows gets within half a point of R@10 of a fit on the entire corpus on most corpora measured, and 2048 does so everywhere — but it must be a *representative, random* sample of the vectors the index will hold. A sorted or clustered prefix of the same size fits quantiles that are shifted and far too narrow, and actively destroys recall. Passing the whole corpus is always safe.

`calibrate` may be called at any time and repeatedly. On a populated index it re-encodes every stored row from its stored codes — no original vectors needed. Know what that re-encode can and cannot do:

- Refitting with the same or a nearby pair is free: the codes reach an exact fixed point.
- Calibrating **after** a large uncalibrated ingest costs several points of recall versus calibrating first (the re-encode is a second quantization). Calibrate before adding when you can.
- A **badly biased** earlier calibration cannot be repaired by refitting: its too-narrow fit clipped coordinates to the outer centroids at encode time, and no later pair recovers what clipping destroyed. Rebuild from the source vectors.

The calibration round-trips exactly through `write`/`load`, `to_bytes`/`from_bytes`, pickling and copying, on both index types. Draining an index to zero vectors keeps its committed calibration.

---

## Filtering

Both index types support restricting the returned top-`k` to a caller-supplied subset of vectors. Unlike post-filtering (search then drop), the kernel never inserts disallowed vectors into the per-query heap, so you always get up to `k` results from the allowed set rather than fewer.

```python
# IdMapIndex — allowlist of external ids (typical use)
allowed = np.array([1003, 1010, 1042], dtype=np.uint64)
scores, ids = idx.search(queries, k=10, allowlist=allowed)
# scores.shape == (nq, min(k, n_allowed)) == (nq, 3)   # 3 unique allowed ids

# TurboQuantIndex — bool mask over slots
mask = np.ones(len(idx), dtype=bool)
mask[disabled_slots] = False
scores, slots = idx.search(queries, k=10, mask=mask)
```

The output shape is `(nq, min(k, n_allowed))`, where `n_allowed` is the number of *distinct* allowed vectors — unique ids in the allowlist, or `mask.sum()` for a mask — the same shrinking behaviour you already see when `k > len(idx)`. No `-1` / `NaN` padding; pad on the caller side if you need a fixed-width batch.

### A mask is invalidated by any mutation, not just a length-changing one

A mask names slots, and [`swap_remove`](#swap_remove-semantics) renumbers slots — so **any** mutation invalidates a mask, including one that leaves `len(idx)` unchanged. Rebuild the mask after every mutation.

The length check is not what protects you. It catches only a size difference (`ValueError: mask length 100 does not match index size 99`); a `swap_remove(i)` + `add(...)` pair restores the original length while leaving a *different* vector in slot `i`, so a mask built before that pair passes validation and then silently selects a different set of vectors than you intended. Nothing outside the index leaks and no error is raised — the selected set is simply the wrong one.

Allowlists on `IdMapIndex` do not have this failure mode, because they name external ids and the index never renumbers an id. An allowlist entry for an id that has since been removed raises `KeyError` (Rust: `SearchError::UnknownId`) instead of quietly resolving to some other vector. The one way an allowlist entry can come to name a different vector is if you re-add that same id yourself with different data.

Common use cases:

- Hybrid retrieval where a SQL/BM25 stage produces a candidate id set.
- Access control or multi-tenant queries (only return ids the caller can see).
- Time-windowed search (e.g. only documents from the last 7 days).

---

## Two-stage 2-bit search

A 2-bit index of 32,768 vectors or more searches in two stages. Nothing changes on disk: the file is the same whichever way it is searched. `TURBOVEC_2BIT_PLANES=0` in the environment before the process first searches keeps the whole-index exact scan instead; it is read once per process.

A 2-bit code is a sign bit and a magnitude bit per coordinate. The in-memory search cache holds the two bits apart — the same bytes per vector, arranged differently — and a search

1. scans the sign bits alone (half the bytes of a full scan) for a shortlist of `max(384, 16 × k)` candidates — on a calibrated index the scan ranks against the query pulled halfway toward the coordinates' centre, which is what keeps the shortlist honest on corpora whose vectors share a strong common direction (see the MedCPT row below), and each candidate carries its plain score on,
2. ranks the shortlist with an estimate that adds the magnitude bits (a bit-count against the query rounded to 6 bits per coordinate), and
3. rescores the best `max(64, 2 × k)` with the exact scan's own arithmetic.

**Scores are exact; the candidate set is approximate.** Every returned score is bit-identical to what the whole-index scan returns for that id. What can differ is *which* ids are returned: a vector whose sign bits alone rank it outside the shortlist is not seen. How often that happens depends on the data:

| data | queries returning exactly the whole-index scan's ids |
|---|---|
| OpenAI `text-embedding-3` d=1536 and d=3072, N=200K, k = 1, 10, 100 | 99.99–100% of 10,000 |
| `all-mpnet-base-v2` d=768, N=41K, k = 1, 10, 100 | 99.95–100% of 10,000 |
| MedCPT (PubMed) d=768, N=100K, k = 1 / 10 / 100 | 99.9% / 99.0% / 88.7% of 3,000 — recall k@k 0.737 / 0.778 / 0.797, the whole-index scan's 0.737 / 0.779 / 0.798 |
| isotropic random unit vectors, d=64–1536, N=50K, k=10 | 4–7% (75% of ids shared) |

Recall against float ground truth on the OpenAI corpora is unchanged at every k the benchmark suite reports. On random vectors — where a query has no real neighbours and the top-k is decided by noise-sized margins — the true nearest neighbour is in the top 10 for 74% of queries at d=768, against 86% with the whole-index scan. Check agreement on your own data if it is unlike the corpora above, and set `TURBOVEC_2BIT_PLANES=0` if the whole-index scan's id set is what you need.

**When it is faster.** The first stage is cheaper than a full scan, and the later stages cost more as `k` grows, so the gain is largest for small `k`. Milliseconds per query on 100K OpenAI d=1536 vectors, 1,000 queries, whole-index scan → two-stage:

| | k=10 | k=64 | k=100 |
|---|---|---|---|
| ARM (c4a), 1 thread, batch | 1.43 → 0.76 | 1.51 → 0.95 | 1.55 → 1.07 |
| ARM (c4a), 1 thread, one query per call | 1.74 → 1.00 | 1.89 → 1.18 | 2.03 → 1.27 |
| ARM (c4a), 8 threads, batch | 0.167 → 0.100 | 0.190 → 0.130 | 0.195 → 0.143 |
| ARM (c4a), 8 threads, one query per call | 0.259 → 0.205 | 0.305 → 0.312 | 0.351 → 0.366 |
| x86 (c3), 1 thread, batch | 0.55 → 0.39 | 0.62 → 0.57 | 0.68 → 0.68 |
| x86 (c3), 1 thread, one query per call | 1.29 → 0.77 | 1.35 → 0.93 | 1.42 → 1.04 |
| x86 (c3), 8 threads, batch | 0.137 → 0.096 | 0.161 → 0.137 | 0.183 → 0.174 |
| x86 (c3), 8 threads, one query per call | 0.397 → 0.286 | 0.450 → 0.439 | 0.538 → 0.507 |

Through `k=100` every cell is faster or within 4% of the whole-index scan; the cells at parity are multi-threaded single queries and x86 batches at the largest `k`.

**What scans the whole index instead.** Indexes below 32,768 vectors (an index that grows past the threshold switches then, and keeps the layout if it later shrinks); dimensions that are not a multiple of 32; x86 CPUs without AVX-512 VBMI and VNNI. Filtered searches use the two-stage path with a plain top-shortlist heap.

---

## Staged 4-bit search

A 4-bit index of 32,768 vectors or more searches in stages. Nothing changes on disk: the file is the same whichever way it is searched. `TURBOVEC_4BIT_PLANES=0` in the environment before the process first searches keeps the whole-index exact scan instead; it is read once per process.

A 4-bit code is a sign bit and three lower bits per coordinate. The in-memory search cache holds the four bit planes apart — the same bytes per vector, arranged differently — and a search

1. scans the sign bits alone (a quarter of the bytes of a full scan) for a shortlist of `max(256, 20 × k)` candidates (`16 × k` from `k = 64`) — on a calibrated index the scan ranks against the query pulled halfway toward the coordinates' centre, which is what keeps the shortlist honest on corpora whose vectors share a strong common direction (see the MedCPT row below), and each candidate carries its plain score on,
2. ranks the shortlist with an estimate that adds the next bit plane and keeps the best `max(256, 6 × k)`,
3. ranks those with all three lower planes, and
4. rescores the best `max(64, 1.5 × k)` with the exact scan's own arithmetic.

**Scores are exact; the candidate set is approximate.** Every returned score is bit-identical to what the whole-index scan returns for that id. What can differ is *which* ids are returned: a vector whose sign bits alone rank it outside the shortlist is not seen. On the corpora below the first stage keeps a wide margin:

| data | queries returning exactly the whole-index scan's ids |
|---|---|
| OpenAI `text-embedding-3` d=1536 and d=3072, N=200K, k = 1, 10, 100 | 99.98–100% of 10,000 |
| `all-mpnet-base-v2` d=768, N=41K, k = 1, 10, 100 | 99.92–100% of 10,000 |
| MedCPT (PubMed) d=768, N=100K, k = 1 / 10 / 100 | 100% / 99.1% / 84.4% of 3,000 — recall k@k 0.936 / 0.943 / 0.956, the whole-index scan's 0.936 / 0.944 / 0.957 |

The MedCPT corpus (#562) is the bunched case: mean pairwise cosine 0.65, a shared direction of norm 0.80, and a median gap between a query's first and second neighbour four times smaller than OpenAI's. Its ids differ from the whole-index scan's more often, but almost every difference is a swap at the list's edge: recall is within 0.1% of the scan's at every k and at both bit widths, and the true nearest neighbour is in the 4-bit top 10 for every query measured. Recall against float ground truth on the OpenAI corpora is unchanged at every k the benchmark suite reports. Structureless random vectors are a different matter: with no real neighbours, a shortlist of sign bits misses the exact top-k for most queries. Check agreement on your own data if it is unlike the corpora above, and set `TURBOVEC_4BIT_PLANES=0` if the whole-index scan's id set is what you need.

**How much faster.** Milliseconds per query on 100K OpenAI d=1536 vectors, whole-index scan → staged; batches are 1,000 queries in one call:

| | k=10 | k=32 | k=64 | k=100 |
|---|---|---|---|---|
| ARM (c4a), 1 thread, batch | 0.985 → 0.741 | 1.012 → 0.800 | 1.052 → 0.872 | 1.105 → 0.956 |
| ARM (c4a), 1 thread, one query per call | 3.58 → 0.95 | 3.62 → 1.01 | 3.70 → 1.09 | 3.86 → 1.19 |
| ARM (c4a), 8 threads, batch | 0.111 → 0.094 | 0.126 → 0.101 | 0.135 → 0.112 | 0.133 → 0.123 |
| ARM (c4a), 8 threads, one query per call | 0.506 → 0.179 | 0.514 → 0.214 | 0.553 → 0.235 | 0.606 → 0.270 |
| x86 (c3), 1 thread, batch | 0.639 → 0.348 | 0.659 → 0.419 | 0.701 → 0.509 | 0.765 → 0.609 |
| x86 (c3), 1 thread, one query per call | 3.34 → 0.73 | 3.35 → 0.82 | 3.45 → 0.92 | 3.47 → 0.96 |
| x86 (c3), 8 threads, batch | 0.155 → 0.087 | 0.162 → 0.099 | 0.175 → 0.117 | 0.199 → 0.142 |
| x86 (c3), 8 threads, one query per call | 0.999 → 0.303 | 1.010 → 0.327 | 1.066 → 0.401 | 1.137 → 0.470 |

Every cell is faster: one query per call 2.25x–4.57x, batches 1.09x–1.84x (harmonic mean over the 32 cells 1.87x). A single query is where the full 4-bit scan is dearest, so that is where the staged search gains most; a batch already shares each block's bytes across queries, so its gain is the quarter-width first stage alone, and it shrinks as `k` grows. On 100K OpenAI d=3072 the same cells read 1.10x–5.65x (harmonic mean 1.83x on ARM, 2.27x on x86).

**Small indexes.** The shortlist is sized by `k`, not by the index, so on an index of 40K vectors a `k=100` shortlist is 4% of it and the ranking stages cost a batch more than the shorter scan saves: measured on 40K vectors (OpenAI d=1536 cut down, and all-mpnet-base-v2 d=768 at 41K), batches at `k=100` run 0.80x–1.04x and at `k=64` 0.88x–1.15x of the whole-index scan, while every other cell is faster (one query per call 1.3x–3.7x; harmonic mean 1.3x–1.5x over the 16 cells). From about 64K vectors every cell is ahead.

**What scans the whole index instead.** Indexes below 32,768 vectors (an index that grows past the threshold switches then, and keeps the layout if it later shrinks); dimensions that are not a multiple of 32; x86 CPUs without AVX-512 VBMI and VNNI. Filtered searches use the staged path with a plain top-shortlist heap.

---

## File formats

### `.tv` — `TurboQuantIndex`

```
┌───────────────────────────────────────────┐
│ magic    "TVPI"  (4 bytes)                │
│ version  u8    = 6                        │
├───────────────────────────────────────────┤
│ core header                               │
│   bit_width    (u8)                       │
│   dim          (u32 LE)                   │
│   n_vectors    (u64 LE)                   │
├───────────────────────────────────────────┤
│ Lloyd-Max codebook                        │
│   boundaries  ((2^bit_width − 1) × f32 LE)│
│   centroids   (2^bit_width × f32 LE)      │
├───────────────────────────────────────────┤
│ codes — sequential blocked layout         │
│   n_byte_groups = dim / (8 / bit_width)   │
│   ceil(n_vectors / 32)                    │
│     × n_byte_groups × 32 bytes            │
├───────────────────────────────────────────┤
│ scales  (n_vectors × f32 LE)              │
│   per-vector length-renormalization       │
├───────────────────────────────────────────┤
│ TQ+ trailer                               │
│   n_calib  (u32 LE)  — 0 or dim           │
│   shift    (n_calib × f32 LE)             │
│   scale    (n_calib × f32 LE)             │
└───────────────────────────────────────────┘
```

The code payload is grouped into 32-vector blocks and padded up to a
whole block, so it is `ceil(n_vectors / 32) * 32` vectors wide on disk.
At `bit_width = 3` a byte holds only two codes rather than 8/3, making
the payload ~33% larger than `dim * bit_width / 8` per vector would
suggest.

### `.tvim` — `IdMapIndex`

```
┌───────────────────────────────────────────┐
│ magic    "TVIM"  (4 bytes)                │
│ version  u8    = 6                        │
├───────────────────────────────────────────┤
│ core payload (same as .tv: header +       │
│   codebook + codes + scales + TQ+)        │
├───────────────────────────────────────────┤
│ slot_to_id  (n_vectors × u64 LE)          │
└───────────────────────────────────────────┘
```

On load, the reverse `id → slot` map is rebuilt in memory. Duplicate ids in the `slot_to_id` table are rejected as corrupt.

### The v8 container

What `write()`, `to_bytes()` and `sync()` produce today is the v8 container (magic `TV8\0`), the v7 container with one more superblock byte:

```
┌───────────────────────────────────────────┐
│ superblock                                │
│   magic     "TV8\0"                       │
│   revision  u8 = 1                        │
│   bit_width u8 · kind u8 (0 .tv, 1 .tvim) │
│   layout    u8 — 0 sequential, 1 planes   │
│   dim u32 · nonce u64 · ops capacity u32  │
│   codebook, TQ+ calibration, CRC          │
├───────────────────────────────────────────┤
│ header slot A │ header slot B  (alternate │
│   commits: generation, n, the partial     │
│   tail block's rows, pending redo ops,    │
│   delta digest, CRC)                      │
├───────────────────────────────────────────┤
│ block units — one per whole 32-row block  │
│   codes in `layout`, 32 scales, 32 ids    │
│   (.tvim)                                 │
└───────────────────────────────────────────┘
```

A unit's codes are the block's sequential-blocked rows (`layout` 0, as every v7 unit) or its bit planes (`layout` 1): the sign plane first, one byte per group of eight coordinates for each of the 32 rows, group-major, then the 32 rows' lower planes, least significant first. The two forms are the same number of bytes, so every offset in the file is the same whichever a unit holds. An index of 32,768 vectors or more keeps its search cache as bit planes (see the two-stage and staged searches above), and writes `layout` 1; smaller indexes, 3-bit indexes and dimensions that are not a multiple of 32 write `layout` 0. Either loads anywhere: a planes file on a host that scans the whole index (x86 without AVX-512, or under `TURBOVEC_*BIT_PLANES=0`) converts on load, as a sequential file converts to planes on a host that uses them. The planes form is arch-neutral; x86 permutes the sign bytes in place at load.

A `sync()` into a v7 file rewrites it as v8 once, as it would for a calibration change, and continues incrementally from there.

### In-memory serialization

Both index types (de)serialize their wire format in memory, without a filesystem round-trip:

```python
payload = idx.to_bytes()                  # bytes, byte-identical to write(path)'s file
restored = IdMapIndex.from_bytes(payload) # same validation as load(path) on a write() file
```

`to_bytes()` returns exactly the bytes `write(path)` would put in the file (`.tv` for `TurboQuantIndex`, `.tvim` for `IdMapIndex`). `from_bytes(data)` accepts `bytes` or `bytearray` and applies exactly the same validation `load` applies to a `write()` file — version handling, structural and value-level checks, the embedded codebook check (a v6 file carries the Lloyd-Max codebook, and a file whose codebook is not a valid one for its `(bit_width, dim)` is rejected — the relevant case for anyone hand-writing files through the raw `io::*` writers), and the `.tvim` duplicate-id check — raising `ValueError` on a corrupt payload (there is no file to blame, so it is not an `OSError`). Both release the GIL. This is the path to use for caches and database columns, and it is what both index types' own `pickle` / `copy` support and the integration stores' are built on.

`pickle.dumps(idx)`, `copy.copy(idx)` and `copy.deepcopy(idx)` work on both index types — they reduce to `from_bytes(to_bytes())`, so an index can cross a `multiprocessing` `spawn` boundary (the default start method on macOS and Windows) and a container holding one can be deep-copied. The copy is fully independent of the original. Everything true of `to_bytes` is therefore true of a pickle — in particular the calibration state round-trips exactly.

Equality and hashing stay identity-based, so `idx == pickle.loads(pickle.dumps(idx))` is `False` even though the two hold the same vectors. Compare `to_bytes()` payloads to check that a saved and a loaded index agree.

An index defines no `__bool__`, so truthiness falls back to `__len__`: an index holding no vectors is **falsy**, and `idx = idx or build_index()` therefore discards a perfectly good empty index. Test for an index with `idx is None`, and for its contents with `len(idx)`.

An index also takes no user attributes (`idx.tag = "x"` raises `AttributeError`) and is not subclassable (`class Sub(IdMapIndex): ...` raises `TypeError`). Both are deliberate. A per-instance `__dict__` is not traversed by the garbage collector, so a reference cycle running through an attribute would leak the whole index rather than being collected, and the attributes would be silently dropped by `pickle` / `copy`, which reduce through `from_bytes(to_bytes())` and carry the payload only. A subclass instance would likewise pickle and copy back to the base class, silently changing type. Attach per-index state by holding the index in an object of your own instead. Re-invoking `idx.__init__(dim=...)` on a built index does nothing at all — it neither resets nor re-shapes it — so build a new index instead of trying to reconfigure one in place.

On the Rust API the same pair exists as `to_bytes()` / `from_bytes(&[u8])`, alongside generic-sink forms `write_to_writer<W: Write>` / `load_from_reader<R: Read>` on both types. All four move the same v7 image the file writers produce: `to_bytes()` is byte-identical to the file `write()` leaves on disk, and `write_to_writer` streams it unit by unit rather than building a second copy in memory. The raw module-level `io::write*` / `io::load*` entry points were removed when v7 became the only format — build an index and use its own methods instead.

On the Rust API `TurboQuantIndex::serialized_len()` returns the exact byte count `to_bytes()` will return and `write(path)` will put in the file, computed from the index's geometry without serializing anything — for sizing a buffer, a database column or a quota check ahead of time. It is exact rather than an upper bound, and `to_bytes()` uses it to allocate its buffer once.

### Incremental saves — `sync()`

`write(path)` rewrites the whole file. `sync(path)` writes only what changed since the last sync to the same path, in a second container format (`.tv` / `.tvim` magic `TV7\0`) built for repeated small commits:

```python
idx.sync("index.tv")     # first sync to a fresh path: writes the whole container
idx.add(more_vectors)
idx.swap_remove(3)
idx.sync("index.tv")     # writes the delta and commits
reloaded = TurboQuantIndex.load("index.tv")   # load() recognises both formats
```

A loaded index stays bound to the path it came from, so it keeps syncing forward incrementally rather than rewriting.

**What it costs.** An append writes the new 32-row blocks plus a commit header. A removal writes no block at all — it rides the commit header as a redo op, and a later sync folds it into its block. The whole-file events are an explicit `calibrate()` (a refit re-encodes every stored code) and enough accumulated removals to exceed the header's op capacity; both compact the file by rewriting it whole, through the same temp-file-and-rename path `write()` uses.

**Durability.** Every sync is durable when it returns — there is no fast mode, and the fsync is `sync_all`, not a data-only variant. A crash at any byte leaves the previous commit intact: the file carries two alternating commit headers, so a torn header fails its checksum and the other one is adopted, and each header names the blocks its own sync wrote along with their digest, so a commit that reached disk ahead of its data is detected rather than served. See [Versioning and limits](#versioning-and-limits) for what this does *not* cover — damage arriving after the write.

**One writer per path.** Each full write stamps the file with a random nonce, so if another process replaces the file underneath a bound index, the next `sync` reports it rather than writing over their commits. Two processes syncing one path concurrently is not supported; the check makes the unsupported case loud, not safe.

**`sync()` files are path-only.** `from_bytes` and `load_from_reader` read the `write()` format. A v7 container needs random access — two header slots, fixed-stride block units, redo ops — which a byte stream cannot serve, and `to_bytes()` only ever emits the `write()` format. Handing v7 bytes to `from_bytes` raises an error saying so and pointing at `load(path)`.

### Load performance

The file stores the codes in the layout the search kernels consume — the arch-neutral sequential blocked rows, or, for an index of 32,768 vectors or more, its bit planes — plus the Lloyd-Max codebook, so a load seeds the search caches directly: there is no O(n·dim) repack and no codebook solve on first search. Non-x86 uses the stored layout as-is; x86 applies one cheap in-block permutation at load (a threaded pass, a few milliseconds for a 77 MB index). A v7 file, which holds the sequential rows only, converts to the planes on load (block-parallel; ~20 ms for 100K × 1536 at 4 bits with the pool, ~80 ms on one thread) and is rewritten as v8 by its next `sync()`. The rotation is deterministic and rebuilt from `dim` in well under a millisecond. A stored index survives cross-platform load → re-save byte-identically; the format itself adds no platform dependence.

### Versioning and limits

Both `.tv` and `.tvim` loads validate the header **before allocating**: `bit_width` must be 2/3/4, `dim` a positive multiple of 8 and `≤ 16384` (`MAX_DIM` — the same cap enforced at construction, so any index this build can create it can also load back), and every payload size is computed with checked arithmetic and read through a length-capped reader. A malformed or untrusted file therefore raises a clean error rather than panicking, dividing by zero, or driving an oversized allocation. Codebook, scale, and calibration values are additionally validated at the value level (finite, in-support), so a structurally valid file carrying out-of-range values in those fields is rejected rather than loaded.

What that validation does **not** give you is integrity checking of the payload. Neither format checksums its stored codes, and the value-level checks only reject values that leave the valid range — a flipped mantissa bit in a scale is still a finite positive float, and a flipped code byte is indistinguishable from a legitimate one. Damage arriving from outside the writer (a failing disk, a truncated copy, a bad transfer) therefore loads clean and changes search results silently.

Measured by flipping every one of the 32,912 bits of a 4114-byte `.tv` file in turn and loading each result: **1460 flips (4%) are rejected and 31,452 (96%) load and return a different index.** By section, rejections are 144 of 144 header bits, 988 of 992 codebook bits, 169 of 3072 scale bits, 159 of 4128 calibration-trailer bits, and **0 of 24,576 code bits**. The scale and trailer figures are the value-level checks doing exactly what they claim and no more: a flip that drives a float non-finite, negative, or out of support is caught, and every mantissa flip is not. The codes carry no validation of any kind, which is why that column is zero and why it is also the largest section.

This is a deliberate scope choice, not an oversight. A save is atomic and a crash mid-write leaves the previous file intact, so the writer cannot leave a torn index behind; what is out of scope is damage that arrives afterwards. If you need to detect that, checksum the file yourself or store it on a filesystem that does.

`n_calib = 0` in the TQ+ trailer means an uncalibrated index; otherwise it equals `dim`. v7 and v8 are read: a v5 or v6 file is refused with an error naming its version, and `turbovec::convert` moves a file between v5, v6, v7 and v8 in any direction (`cargo run --example convert -- <in> <out> v8`); converting a v8 planes file down re-containers the same codes, never re-quantizes. Versions 1 through 4 predate the v5 rotation change and cannot be decoded at all — their codes were encoded under a rotation this build cannot reproduce — so they must be rebuilt from the source vectors.

`dim = 0` in the core header signals a lazy uncommitted index. It is only valid alongside `n_vectors = 0`; on load it produces an index whose `dim` is `None` until the first `add` / `add_with_ids` call.

Both formats carry a magic + version byte and are stable across minor versions. Breaking changes bump the version byte. `write()`, `to_bytes()` and `sync()` all produce v8: `write()` and `to_bytes()` an *unclaimed* snapshot, `sync()` a container it claims and then updates incrementally (see [Incremental saves](#incremental-saves--sync)). v8 files are not readable by earlier turbovec releases, whose loaders reject the magic rather than misparse it; `turbovec::convert` writes a v7, v6 or v5 file for one.
