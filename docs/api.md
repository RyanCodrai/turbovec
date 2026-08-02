# API Reference

turbovec exposes two index types and one serialization format per type.

- [`TurboQuantIndex`](#turboquantindex) — positional index, O(1) `swap_remove` delete.
- [`IdMapIndex`](#idmapindex) — stable external `u64` ids on top of `TurboQuantIndex`.
- [TQ+ calibration](#tq-calibration) — the per-coordinate calibration lifecycle.
- [File formats](#file-formats) — `.tv` and `.tvim`.

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

### In-memory serialization

Both index types (de)serialize their wire format in memory, without a filesystem round-trip:

```python
payload = idx.to_bytes()                  # bytes, byte-identical to write(path)'s file
restored = IdMapIndex.from_bytes(payload) # same validation as load(path)
```

`to_bytes()` returns exactly the bytes `write(path)` would put in the file (`.tv` for `TurboQuantIndex`, `.tvim` for `IdMapIndex`). `from_bytes(data)` accepts `bytes` or `bytearray` and applies exactly the same validation as `load` — version handling, structural and value-level checks, the embedded codebook check (a v6 file carries the Lloyd-Max codebook, and a file whose codebook is not a valid one for its `(bit_width, dim)` is rejected — the relevant case for anyone hand-writing files through the raw `io::*` writers), and the `.tvim` duplicate-id check — raising `ValueError` on a corrupt payload (there is no file to blame, so it is not an `OSError`). Both release the GIL. This is the path to use for caches and database columns, and it is what both index types' own `pickle` / `copy` support and the integration stores' are built on.

`pickle.dumps(idx)`, `copy.copy(idx)` and `copy.deepcopy(idx)` work on both index types — they reduce to `from_bytes(to_bytes())`, so an index can cross a `multiprocessing` `spawn` boundary (the default start method on macOS and Windows) and a container holding one can be deep-copied. The copy is fully independent of the original. Everything true of `to_bytes` is therefore true of a pickle — in particular the calibration state round-trips exactly.

Equality and hashing stay identity-based, so `idx == pickle.loads(pickle.dumps(idx))` is `False` even though the two hold the same vectors. Compare `to_bytes()` payloads to check that a saved and a loaded index agree.

An index defines no `__bool__`, so truthiness falls back to `__len__`: an index holding no vectors is **falsy**, and `idx = idx or build_index()` therefore discards a perfectly good empty index. Test for an index with `idx is None`, and for its contents with `len(idx)`.

An index also takes no user attributes (`idx.tag = "x"` raises `AttributeError`) and is not subclassable (`class Sub(IdMapIndex): ...` raises `TypeError`). Both are deliberate. A per-instance `__dict__` is not traversed by the garbage collector, so a reference cycle running through an attribute would leak the whole index rather than being collected, and the attributes would be silently dropped by `pickle` / `copy`, which reduce through `from_bytes(to_bytes())` and carry the payload only. A subclass instance would likewise pickle and copy back to the base class, silently changing type. Attach per-index state by holding the index in an object of your own instead. Re-invoking `idx.__init__(dim=...)` on a built index does nothing at all — it neither resets nor re-shapes it — so build a new index instead of trying to reconfigure one in place.

On the Rust API the same pair exists as `to_bytes()` / `from_bytes(&[u8])`, alongside generic-sink forms `write_to_writer<W: Write>` / `load_from_reader<R: Read>` on both types and the raw module-level entry points `io::write_to`, `io::load_from`, `io::write_id_map_to`, `io::load_id_map_from` (whose code-payload parameter is the v6 sequential blocked layout plus the codebook arrays — see `codes_blocked_seq()` / `codebook_for_write()`). Every `io::write*` entry point validates the lengths of the buffers it is handed against the header it writes them under — one scale per vector, and a code buffer of exactly the blocked-layout size `(bit_width, dim, n_vectors)` implies, alongside the codebook, TQ+-calibration and `slot_to_id` length invariants. A mismatch panics before anything is written, so a hand-built buffer cannot produce a file that loads clean and silently mis-scores; the code checks the buffers, so its `# Panics` section is the authority on the exact conditions.

On the Rust API `TurboQuantIndex::serialized_len()` returns the exact byte count `to_bytes()` will return and `write(path)` will put in the file, computed from the index's geometry without serializing anything — for sizing a buffer, a database column or a quota check ahead of time. It is exact rather than an upper bound, and `to_bytes()` uses it to allocate its buffer once.

### Load performance

The file stores the codes in the arch-neutral *sequential blocked* layout the search kernels consume, plus the Lloyd-Max codebook, so a load seeds the search caches directly: there is no O(n·dim) repack and no codebook solve on first search. Non-x86 uses the stored layout as-is; x86 applies one cheap in-block nibble interleave at load (a threaded SIMD pass, ~2 ms for a 77 MB index). The rotation is deterministic and rebuilt from `dim` in well under a millisecond. A stored index survives cross-platform load → re-save byte-identically; the format itself adds no platform dependence.

### Versioning and limits

Both `.tv` and `.tvim` loads validate the header **before allocating**: `bit_width` must be 2/3/4, `dim` a positive multiple of 8 and `≤ 16384` (`MAX_DIM` — the same cap enforced at construction, so any index this build can create it can also load back), and every payload size is computed with checked arithmetic and read through a length-capped reader. A malformed or untrusted file therefore raises a clean error rather than panicking, dividing by zero, or driving an oversized allocation. Codebook, scale, and calibration values are additionally validated at the value level (finite, in-support), so a corrupt file fails loudly instead of silently poisoning search results.

`n_calib = 0` in the TQ+ trailer means an uncalibrated index; otherwise it equals `dim`. Loading a version-5 file (packed bit-plane payload, same rotation) is supported transparently and converts on load; versions 1 through 4 predate the v5 rotation break and are rejected with a rebuild hint.

`dim = 0` in the core header signals a lazy uncommitted index. It is only valid alongside `n_vectors = 0`; on load it produces an index whose `dim` is `None` until the first `add` / `add_with_ids` call.

Both formats carry a magic + version byte and are stable across minor versions. Breaking changes bump the version byte: the writer emits version 6 only, and version-6 files are not readable by earlier turbovec releases (their loaders reject the version byte with an "unsupported format version" error — no silent misparse).
