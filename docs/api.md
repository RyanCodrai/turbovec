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

Before the first add, `idx.dim` is `None`, `len(idx)` is `0`, and `search()` returns empty results.

### Methods

| Method | Notes |
|---|---|
| `TurboQuantIndex(dim=None, bit_width=4)` | `bit_width ∈ {2, 3, 4}`. `dim` must be a positive multiple of 8 and `≤ 16384` (`MAX_DIM`). `dim` is optional; when omitted it is inferred from the first `add` call. |
| `add(vectors)` | `vectors` is a contiguous float32 array of shape `(n, dim)`. On a lazy index the first call locks `dim`; subsequent calls must match. Raises `ValueError` on dim mismatch, a zero-width (0-column) batch, or any coordinate that is non-finite (NaN/Inf) or `\|value\| ≥ 1e16`. On the Rust API, a lazy index's first add must use `add_2d(vectors, dim)` — the flat `add(&[f32])` requires an already-committed dim and panics otherwise. (Python arrays carry their shape, so this applies to Rust only.) |
| `search(queries, k, *, mask=None)` | Returns `(scores, indices)`, both shape `(nq, effective_k)`. Indices are `int64` slot positions. `mask` is an optional `bool` array of length `len(idx)`; when given, only slots with `mask[i] == True` contribute. `effective_k = min(k, mask.sum())`. Raises `ValueError` on a non-finite or `\|value\| ≥ 1e16` query coordinate. |
| `swap_remove(idx)` | O(1). Moves the last vector into `idx`; returns the previous position of that moved vector (so external refs can be updated if needed). |
| `prepare()` | Optional. Eagerly builds the rotation matrix, Lloyd-Max centroids and SIMD-blocked layout so the first `search` call doesn't pay the one-time cost. No-op on a lazy index that hasn't seen its first add. |
| `write(path)` / `load(path)` | `.tv` format. |
| `to_bytes()` / `from_bytes(data)` | In-memory `.tv` serialization — see [In-memory serialization](#in-memory-serialization). |
| `len(idx)` / `idx.dim` / `idx.bit_width` | Introspection. `idx.dim` returns `int` once committed, or `None` on a lazy index that hasn't seen its first add. |
| `idx.calibration_state` | TQ+ calibration state: `"warming_up"`, `"fitted"` or `"identity"` — see [TQ+ calibration](#tq-calibration). |

### `swap_remove` semantics

`swap_remove(i)` is named to match Rust's [`Vec::swap_remove`](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.swap_remove): the last element moves into slot `i`, and the vector is truncated by one. It is **not** a shift (FAISS's `IndexPQ::remove_ids` behaviour). Order is not preserved; slot indices of vectors you didn't delete may now point at different vectors than before.

Use [`IdMapIndex`](#idmapindex) if external references have to stay stable across deletes.

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

Stable-id wrapper around `TurboQuantIndex`. Roughly equivalent to FAISS's `IndexIDMap2` — hash-table backed, O(1) `remove(id)`.

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
| `search(queries, k, *, allowlist=None)` | Returns `(scores, ids)` — `ids` are `uint64` external ids. `allowlist` is an optional `uint64` array of ids; when given, results are restricted to those ids and `effective_k = min(k, number of unique ids in allowlist)` (the allowlist is deduplicated; repeated ids don't widen the result). Raises `ValueError` on an empty allowlist or a non-finite / `\|value\| ≥ 1e16` query coordinate, and `KeyError` on unknown ids. |
| `contains(id)` / `id in idx` | Membership. |
| `write(path)` / `load(path)` | `.tvim` format. |
| `to_bytes()` / `from_bytes(data)` | In-memory `.tvim` serialization — see [In-memory serialization](#in-memory-serialization). |
| `len(idx)` / `idx.dim` / `idx.bit_width` / `idx.calibration_state` / `prepare()` | Same as `TurboQuantIndex`. |

### When to use which

- `TurboQuantIndex` — you never delete, or you're fine with positional ids.
- `IdMapIndex` — you need stable external ids (e.g. string-id → vector mapping maintained by the caller).

All the framework integrations (LangChain, LlamaIndex, Haystack) use `IdMapIndex` internally for exactly this reason.

---

## TQ+ calibration

TQ+ fits a per-coordinate `(shift, scale)` pair from the empirical quantiles of the vectors in the index, and every stored vector is encoded in that one calibrated coordinate system. The fit needs at least 1000 vectors to be stable, so an index passes through three states, reported by `idx.calibration_state` (`TurboQuantIndex::calibration_state()` in Rust, returning a `CalibrationState`):

| State | Meaning |
|---|---|
| `"warming_up"` | Fewer than 1000 vectors added so far. The rows are searchable, encoded under identity calibration, and their raw float32 values are also buffered (at most 1000 rows, `< 1000 × dim × 4` bytes). |
| `"fitted"` | A calibration fitted from at least 1000 vectors is locked in. Every stored row is encoded in it and every later add reuses it. |
| `"identity"` | The index is committed to identity calibration for good: no TQ+ recall gain, now or later. |

The add that takes the total to 1000 or more fits the calibration and re-encodes the buffered rows with it, in place and in slot order — so ingesting 3000 vectors as six calls of 500 ends up as well calibrated as one bulk `add`, and external ids and slot positions are unaffected.

A saved index carries no warm-up buffer, so **saving before the index holds 1000 vectors freezes it at `"identity"`**: the loaded copy declares the identity calibration its codes were actually encoded with, and adding more vectors later cannot change that (recovering the TQ+ gain means rebuilding from the original float32 vectors). Python emits a one-shot `RuntimeWarning` on such a save. The same applies to an index reconstructed through `from_parts` from rows encoded under identity.

Adds into a warming-up index are never chunked by the interruptibility wrapper (see `chunk_size`), because the calibrating add must see its whole batch.

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

Common use cases:

- Hybrid retrieval where a SQL/BM25 stage produces a candidate id set.
- Access control or multi-tenant queries (only return ids the caller can see).
- Time-windowed search (e.g. only documents from the last 7 days).

---

## File formats

### `.tv` — `TurboQuantIndex`

```
┌──────────────────────────────────────┐
│ magic    "TVPI"  (4 bytes)            │
│ version  u8    = 4                     │
├──────────────────────────────────────┤
│ core header                           │
│   bit_width    (u8)                   │
│   dim          (u32 LE)               │
│   n_vectors    (u64 LE)               │
│   rotation fingerprint                │
│     hash    (u64 LE, FNV-1a)          │
│     probes  (64 × f32 LE)             │
├──────────────────────────────────────┤
│ packed codes                          │
│   (dim / 8) * bit_width * n_vectors   │
├──────────────────────────────────────┤
│ scales  (n_vectors × f32 LE)          │
│   per-vector length-renormalization   │
├──────────────────────────────────────┤
│ TQ+ trailer                           │
│   n_calib  (u32 LE)  — 0 or dim       │
│   shift    (n_calib × f32 LE)         │
│   scale    (n_calib × f32 LE)         │
└──────────────────────────────────────┘
```

### `.tvim` — `IdMapIndex`

```
┌──────────────────────────────────────┐
│ magic    "TVIM"  (4 bytes)            │
│ version  u8    = 4                     │
├──────────────────────────────────────┤
│ core payload (same as .tv:            │
│   header + codes + scales + TQ+)      │
├──────────────────────────────────────┤
│ slot_to_id  (n_vectors × u64 LE)      │
└──────────────────────────────────────┘
```

On load, the reverse `id → slot` map is rebuilt in memory. Duplicate ids in the `slot_to_id` table are rejected as corrupt.

### In-memory serialization

Both index types (de)serialize their wire format in memory, without a filesystem round-trip:

```python
payload = idx.to_bytes()                  # bytes, byte-identical to write(path)'s file
restored = IdMapIndex.from_bytes(payload) # same validation as load(path)
```

`to_bytes()` returns exactly the bytes `write(path)` would put in the file (`.tv` for `TurboQuantIndex`, `.tvim` for `IdMapIndex`). `from_bytes(data)` accepts `bytes` or `bytearray` and applies exactly the same validation as `load` — version handling, structural and value-level checks, and the `.tvim` duplicate-id check — raising `ValueError` on a corrupt payload (there is no file to blame, so it is not an `OSError`). Both release the GIL. This is the path to use for caches, database columns, and pickling; the integration stores' pickle support is built on it.

On the Rust API the same pair exists as `to_bytes()` / `from_bytes(&[u8])`, alongside generic-sink forms `write_to_writer<W: Write>` / `load_from_reader<R: Read>` on both types and the raw module-level entry points `io::write_to`, `io::load_from`, `io::write_id_map_to`, `io::load_id_map_from` (whose code-payload parameter is the v6 sequential blocked layout plus the codebook arrays — see `codes_blocked_seq()` / `codebook_for_write()`).

### Load performance

The file stores the codes in the arch-neutral *sequential blocked* layout the search kernels consume, plus the Lloyd-Max codebook, so a load seeds the search caches directly: there is no O(n·dim) repack and no codebook solve on first search. Non-x86 uses the stored layout as-is; x86 applies one cheap in-block nibble interleave at load (a threaded SIMD pass, ~2 ms for a 77 MB index). The rotation is deterministic and rebuilt from `dim` in well under a millisecond. A stored index survives cross-platform load → re-save byte-identically; the format itself adds no platform dependence.

### Versioning and limits

Both `.tv` and `.tvim` loads validate the header **before allocating**: `bit_width` must be 2/3/4, `dim` a positive multiple of 8 and `≤ 16384` (`MAX_DIM` — the same cap enforced at construction, so any index this build can create it can also load back), and every payload size is computed with checked arithmetic and read through a length-capped reader. A malformed or untrusted file therefore raises a clean error rather than panicking, dividing by zero, or driving an oversized allocation. Codebook, scale, and calibration values are additionally validated at the value level (finite, in-support), so a corrupt file fails loudly instead of silently poisoning search results.

`n_calib = 0` in the TQ+ trailer means identity calibration (a lazy index with no `add` yet, or a pre-TQ+ index that was re-saved); otherwise it equals `dim`. Loading a version-5 file (packed bit-plane payload, same rotation) is supported transparently and converts on load; versions 1 through 4 predate the v5 rotation break and are rejected with a rebuild hint.

`dim = 0` in the core header signals a lazy uncommitted index. It is only valid alongside `n_vectors = 0`; on load it produces an index whose `dim` is `None` until the first `add` / `add_with_ids` call.

Both formats carry a magic + version byte and are stable across minor versions. Breaking changes bump the version byte: the writer emits version 6 only, and version-6 files are not readable by earlier turbovec releases (their loaders reject the version byte with an "unsupported format version" error — no silent misparse).
