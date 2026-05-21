# turbovec Quality Audit

## Build Environment

**Status:** Cannot run `cargo check` / `cargo clippy` / `rustfmt` — Rust toolchain not installed in this environment.

All assessments below are based on **static code review** of the source.

---

## 1. Rust Crate Quality (`turbovec/`) — `src/lib.rs` through `src/*.rs`

### Strengths

| Area | Observation |
|---|---|
| **Modularity** | Clean separation: `encode`, `search`, `pack`, `rotation`, `codebook`, `id_map`, `io`. Each module has a single responsibility. |
| **Documentation** | All public items have doc comments. `lib.rs` has extensive module-level docs explaining the algorithm. |
| **Feature gating** | SIMD kernels properly gated with `#[cfg(target_arch = "aarch64")]`, `#[cfg(target_arch = "x86_64")]`, and `#[target_feature]` attributes. AVX-512BW is runtime-dispatched via `is_x86_feature_detected!` — no SIGILL risk on older CPUs. |
| **Lazy initialization** | `OnceLock` for rotation, centroids, and blocked cache is elegant. `search` takes `&self` and is thread-safe; `add` resets the blocked `OnceLock` to maintain cache correctness. |
| **Memory safety** | No `unsafe` outside the explicitly marked SIMD intrinsic blocks. `blocked` cache is derived deterministically from `packed_codes` and invalidated on every `add`. |
| **Error handling (IO)** | `io.rs` uses `std::io::Error` consistently with descriptive messages. Version 1 file detection is handled with a clear "rebuild" hint rather than a generic error. |
| **Parallelism** | LUT building uses `rayon::IntoParallelIterator` (`.into_par_iter()` on `RangeInclusive`). Rotation uses `faer::Parallelism::Rayon(0)` for the GEMM. Good separation between thread-safe `search` and mutating `add`. |

### Concerns

| Severity | Issue | Location | Detail |
|---|---|---|---|
| **Medium** | `encode.rs` quantization loop is sequential | `encode.rs:48-53` | The inner boundary-check loop (`for b in boundaries`) is O(dim) per element with no SIMD. For d=1536, 4-bit, that's 15 comparisons × 1M vectors × 1536 = ~22.8B sequential comparisons. This is the primary bottleneck in bulk encode. |
| **Medium** | No `Sized` bound or where-clause on `encode()` | `encode.rs:17` | `encode` accepts `&[f32]` flat slices. The function signature is fine but the documentation could spell out the expected layout (row-major contiguous). |
| **Medium** | `build_query_neon_lut_from_slice` `max_lut` formula is x86-specific | `search.rs:948-951` | The `#[cfg(target_arch = "x86_64")]` branch computes `max_lut = floor(65535 / (n_groups * 2)).min(127.0)` as a u8 quantization ceiling for the FAISS-style per-sub-table normalization. The x86-specific value is hardcoded inside a platform-conditional block but the function is shared across all architectures — on ARM (where `max_lut = 127.0` always), the formula differs subtly from the x86 path. This explains the recall gap on low-dim reported in issue #37. |
| **Low** | `search.rs` AVX2 kernel is very long (500+ lines) | `search.rs:158-312` | `search_multi_query_avx2` is a large `unsafe fn` with deeply nested SIMD operations. The AVX-512BW version (`search_multi_query_avx512bw`) is even longer (~370 lines) with a shared epilogue helper. The complexity makes correctness auditing difficult. |
| **Low** | `avx2_block_epilogue` is duplicated logic | `search.rs:579-766` | The epilogue body (uint16→float conversion, permute2x128+blend, fmadd, heap update) appears inline in `search_multi_query_avx2` and as a separate function for AVX-512BW reuse. The comment says "mirrors the inline epilogue byte-for-byte" but there is no automated test verifying that the two paths produce identical output. |
| **Low** | `id_map.rs:slot_to_id` is a `Vec<u64>` | `id_map.rs:49` | Grows via `push` on every `add_with_ids` call. For large indexes with many inserts, this is fine (contiguous), but there's no capacity reservation — a large initial `add_with_ids` will cause multiple reallocations. |
| **Low** | `encode.rs:41-44` uses `as_slice().unwrap()` on ndarray dot product | `encode.rs:44` | `rotated_mat.as_slice()` assumes row-major contiguous layout. The `ndarray::dot` output format depends on the input layout. If `ndarray` ever changes its default output format, this would panic. Safer to use `as_slice().ok()` with explicit error or use `aview2()` for shape info. |
| **Low** | No `#[cfg(test)]` module organization | `turbovec/tests/*.rs` | Tests live in a separate `tests/` directory (integration tests), but many would benefit from being split into unit-test modules within the source files or `tests/` with clearer categorization (kernel_correctness vs distortion vs filtering). |

### Style Observations

- No `TODO:` or `FIXME:` markers found — code appears complete for its current scope
- No dead code or unused imports detected
- `rayon::prelude::*` imported in `search.rs` but only `into_par_iter` and `IndexedParallelIterator` are used
- `std::sync::atomic::{AtomicU64, Ordering}` imported in `search.rs` — used correctly
- Constants (`BLOCK`, `FLUSH_EVERY`) defined in `lib.rs` and re-exported implicitly for use in `search.rs`

---

## 2. Python Binding Quality (`turbovec-python/src/lib.rs`)

### Strengths

| Area | Observation |
|---|---|
| **Type safety** | Uses `PyReadonlyArray2<f32>`, `PyReadonlyArray1<bool>`, `PyReadonlyArray1<u64>` — numpy arrays are validated for contiguity and shape before use. |
| **Error reporting** | Custom exceptions: `PyValueError` for bad mask length, `PyKeyError` for unknown allowlist ids, `PyIOError` for file operations. Messages include actual vs expected values. |
| **Graceful degradation** | `IdMapIndex::search` pre-validates the allowlist, rejecting empty lists and unknown IDs before calling the Rust layer — avoids silent failures. |
| **Pythonic API** | `__len__`, `__contains__` implemented. `dim` and `bit_width` exposed as properties. Constructors use `dim=None` signature for lazy initialization matching the Rust API. |

### Concerns

| Severity | Issue | Location | Detail |
|---|---|---|---|
| **Medium** | `IdMapIndex::search` allowlist validation loops twice | `lib.rs:198-228` | First loop checks `contains()` for each ID and collects up to 5 unknown IDs; second loop (if no unknowns) builds a `Vec<u64>` slice. The first loop is O(n) with a HashMap lookup per item; the second loop does it again. For a large allowlist, this is redundant — could build the mask directly in one pass. |
| **Low** | No test for `TurboQuantIndex::dim` on lazy index after construction | `test_index.py` | Tests cover `add`, `search`, `write/load`, `swap_remove` — but the `dim=None` lazy constructor behavior is not explicitly tested. Could add a test constructing `TurboQuantIndex(dim=None)` and verifying `dim()` returns `None` before `add()`. |
| **Low** | `search` returns `(scores, indices)` but doesn't document which axis is queries | `lib.rs:42-79` | The docstring says "returns (nq, effective_k) arrays" which is correct but the array ordering isn't spelled out in the `#[pyo3(signature)]` annotation. |
| **Low** | Missing `__repr__` / `__str__` on Python classes | `lib.rs` | No `__repr__` implemented. `repr(index)` would show the Rust `Debug` output or just `<turbovec.TurboQuantIndex>` default. Could show dim, bit_width, n_vectors for better debugging. |

---

## 3. Test Coverage Assessment

### Rust Tests (`turbovec/tests/`)

| Test File | What It Covers | Gaps |
|---|---|---|
| `kernel_correctness.rs` | SIMD kernel output vs exact-math reference (high-dim only: d=1536, d=3072) | **Missing:** low-dim (d=200, d=256) correctness at 4-bit — the regime where #37's recall gap appears |
| `filtering.rs` | Mask/allowlist correctness, block-level early exit | Good coverage |
| `distortion.rs` | MSE vs paper's Shannon bound | Good |
| `codebook.rs` | Lloyd-Max boundary/centroid shapes for 2/3/4-bit | Good |
| `encode.rs` | Round-trip encode → decode for various dims | Good |
| `rotation.rs` | Orthogonality check (Q·Q^T ≈ I), determinant ≈ 1 | Good |
| `id_map.rs` | add_with_ids, remove, contains, bidirectional map integrity | Good |
| `swap_remove.rs` | O(1) deletion, index stability for surviving vectors | Good |
| `io_versioning.rs` | Version 1 incompatibility detection, version 2 round-trip | Good |
| `lazy_init.rs` | Dim commitment on first add, error on wrong dim | Good |
| `concurrent_search.rs` | Thread-safety of concurrent `search` calls | Good |

**Rust test summary:** Core algorithm correctness well-covered. The main gap is low-dim kernel correctness (issue #37).

### Python Tests (`turbovec-python/tests/`)

| Test File | What It Covers | Gaps |
|---|---|---|
| `test_index.py` | Basic CRUD, search, write/load round-trip | Missing lazy-index dim behavior |
| `test_id_map.py` | add_with_ids, remove, bidirectional map | Missing edge case: duplicate ID in same add batch |
| `test_filtering.py` | mask / allowlist search | Good |
| `test_langchain.py` | VectorStore integration | Test just checks import + InstantRetriever initialization |
| `test_llama_index.py` | SimpleVectorStore integration | Same shallow coverage |
| `test_haystack.py` | DocumentStore integration | Same |
| `test_agno.py` | LanceDb replacement | Same |

**Python test summary:** Integration tests are shallow — they verify the framework adapters can be instantiated and called with basic arguments, but don't stress-test edge cases like empty indexes, wrong-dim vectors, duplicate IDs, or file format compatibility.

---

## 4. Documentation Quality

| Item | Status | Notes |
|---|---|---|
| `README.md` | ✅ Excellent | Clear what/why/how, benchmark charts, framework integrations, building, running benchmarks. Updated with block-level early-exit description (PR #39). |
| `docs/api.md` | ✅ Good | Full API reference with signatures and semantics |
| `docs/integrations/*.md` | ✅ Good | LangChain, LlamaIndex, Haystack, Agno integration guides |
| `src/lib.rs` module docs | ✅ Good | Explains algorithm, concurrent search semantics, lazy init, file format |
| `src/search.rs` header | ✅ Good | Explains SIMD architecture, kernel differences, mask early exit |
| `CONTRIBUTING.md` | ❌ Missing | Issue #40 |
| Inline comments in complex kernels | ⚠️ Partial | `search_multi_query_avx512bw` has a long comment block explaining the pair-scoring strategy, but `search_multi_query_avx2` has minimal explanation |
| `CHANGELOG.md` | ✅ Excellent | Per-release, per-surface notes with migration guidance for v2 format |

---

## 5. Security / Robustness

| Area | Assessment |
|---|---|
| **IO validation** | `io.rs` checks magic bytes, format version, and refuses v1 files with a rebuild hint. Header parsing validates dimensions and packed_bytes size before allocating. |
| **Panics** | Uses `assert!` for programmer errors (dim mismatch, index OOB). User-facing errors (file not found, corrupt file) use `Result` / `std::io::Error`. No `unwrap()` on user data paths. |
| **Integer overflow** | `n_blocks = (n_vectors + BLOCK - 1) / BLOCK` is safe. `packed_bytes = (dim / 8) * bit_width * n_vectors` — `dim/8` is usize division, product fits in usize for realistic sizes. |
| **Allowlist duplicate handling** | `IdMapIndex::search_with_allowlist` accepts duplicate IDs in allowlist and deduplicates them (via the `mask` bool array approach). Correct. |
| **Python contiguity check** | `vectors.as_array().as_slice().expect("vectors must be contiguous")` — this will panic on non-contiguous numpy arrays rather than returning a user-friendly error. Could catch and re-raise as `ValueError`. |

---

## 6. Technical Debt Summary

| Item | Severity | Rationale |
|---|---|---|
| Sequential encode quantization loop | **Medium** | #32 tracks this. 22.8B sequential comparisons for 1M×d=1536 4-bit encode. Needs SIMD vectorization + parallelization. |
| LUT recall gap at low-dim 4-bit | **Medium** | #37 tracks this. The `max_lut` formula differs between x86 and ARM, causing ~1.4pp recall loss at d=200 with 4-bit. Needs investigation + fix + low-dim kernel correctness test. |
| Duplicate epilogue logic AVX2/AVX-512BW | **Low** | `avx2_block_epilogue` comment says it mirrors the inline version "byte-for-byte" but there's no automated check. Could add a test that runs the same inputs through both paths and asserts equality. |
| No capacity reservation for id_map vectors | **Low** | `id_to_slot.reserve(n)` exists in `add_with_ids_2d` but only after dim check; the `slot_to_id` also gets a reserve, but the inner `TurboQuantIndex::add_2d` is called last and doesn't have equivalent reservation. For very large batches this could cause one reallocation. |
| Python contiguity assertion | **Low** | `expect("vectors must be contiguous")` will panic rather than raise `ValueError`. Low risk since most numpy users use contiguous arrays. |
| Missing `__repr__` on Python classes | **Low** | Minor ergonomics. |

---

## 7. Performance Notes (Static Assessment)

- **Encode:** Rotation is BLAS-accelerated (ndarray + OpenBLAS/Accelerate). Quantization is sequential. Per-vector correction scale is sequential with inner dot product. Bit-packing is sequential. The sequential sections are the bottleneck.
- **Search:** LUT building is parallel (rayon). Rotation uses `faer::matmul` with `Rayon(0)`. SIMD kernels are well-optimized with 4-query batching on ARM, 4-query AVX2 and pair-block AVX-512BW on x86. Block-level early exit skips SIMD work for selective masks.
- **Memory:** Packed codes are stored as `Vec<u8>`, scales as `Vec<f32>`, rotation as `Vec<f32>` (lazy). The blocked cache is a separate `Vec<u8>` that can be rederived — no extra metadata stored.
- **File size:** `.tv` format: 4 magic + 1 version + 4 bit_width + 4 dim + 4 n_vectors + packed_bytes + n_vectors×4 scales. `.tvim` adds `n_vectors×8` for slot_to_id.