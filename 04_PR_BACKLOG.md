# turbovec PR Backlog

Prioritized list of PR candidates for external contributors.

---

## Tier 1: Quick Wins (1–2 days, low risk)

### PR-1: Add `CONTRIBUTING.md` *(addresses issue #40)*

**What:** Write a contributing guide at the repo root.

**Content checklist:**
- Branch strategy: fork → feature branch from `upstream/main`; PRs target `upstream/main`
- Rust build prerequisites: Rust 1.70+, OpenBLAS dev libs (macOS: Accelerate is automatic; Linux: `apt-get install libopenblas-dev`); for Python: maturin, Python 3.9+
- `cargo test` for Rust unit/integration tests; `pytest` for Python tests
- Benchmark expectation: if PR changes performance, note benchmark delta in the PR description
- Versioning convention: which version (Rust crate vs Python package) gets bumped for each change type; Python is at 0.5.1, Rust is at 0.4.1
- Contacts: mention opening an issue before starting large PRs

**Template sources:** `ripgrep`, `serde`, or `rayon` CONTRIBUTING.md files are well-regarded in the Rust ecosystem.

**Effort:** ~2 hours. **Risk:** None. **Value:** Immediate — unblocks external contributions.

---

### PR-2: Add `__repr__` / `__str__` to Python classes

**What:** Implement `__repr__` on `TurboQuantIndex` and `IdMapIndex` Python classes.

**Implementation:**
```python
def __repr__(self) -> str:
    return f"turbovec.TurboQuantIndex(dim={self.dim()}, bit_width={self.bit_width()}, n_vectors={len(self)})"
```

**Benefit:** Makes debugging in Python REPL significantly easier. Good first PR for someone learning the codebase.

**Effort:** ~30 minutes. **Risk:** None.

---

### PR-3: Add Python test for lazy-index dim behavior

**What:** Add a test case to `test_index.py` covering:
- Construct `TurboQuantIndex(dim=None, bit_width=4)`
- Verify `index.dim()` returns `None` before any `add()`
- After `add()`, verify `dim()` returns the committed dimension

Also verify the error case: `add` with a vector of a different dim raises `ValueError` with an appropriate message.

**Effort:** ~1 hour. **Risk:** None.

---

### PR-4: Add Python test for duplicate ID rejection in `IdMapIndex`

**What:** In `test_id_map.py`, add a test:
```python
def test_duplicate_id_in_batch_raises(self):
    idx = IdMapIndex(dim=128, bit_width=4)
    vectors = np.random.rand(3, 128).astype(np.float32)
    ids = np.array([100, 200, 100], dtype=np.uint64)  # 100 duplicated
    with pytest.raises(Exception):
        idx.add_with_ids(vectors, ids)
```

**Effort:** ~30 minutes. **Risk:** None.

---

## Tier 2: Medium Effort (2–5 days, moderate risk)

### PR-5: Fix IdMapIndex allowlist validation (double-pass)

**What:** Currently `search_with_allowlist` in the Python binding does two passes over the allowlist (first to check `contains()`, second to build the mask). Combine into one pass:

```python
# Current: two passes
for &id in slice { if !self.inner.contains(id) { unknown.push(id); } }
if unknown.is_empty() { /* build mask */ }

# Fix: one pass, build mask while checking
# - If contains: set mask[slot] = true
# - If not contains: immediately return KeyError with first unknown id
```

Also remove the `unknown` vec and the 5-preview logic — just fail on first unknown id.

**Effort:** ~2 hours. **Risk:** Low — reduces work, maintains same semantics.

---

### PR-6: Add capacity reservation to `IdMapIndex::add_with_ids_2d`

**What:** In `id_map.rs:98-128`, add `self.inner.add_2d()` pre-warming — reserve capacity on the inner index before adding:

Currently the code reserves on `id_to_slot` and `slot_to_id`, then calls `inner.add_2d` last. The inner index's `packed_codes` and `scales` vectors may need reallocation. Since the inner `add_2d` appends, we can pre-reserve:

```rust
// Before calling self.inner.add_2d, hint capacity
self.inner.reserve(n); // Requires adding a reserve() method to TurboQuantIndex
```

This requires adding `TurboQuantIndex::reserve(&mut self, n: usize)` which calls `packed_codes.reserve(n * bytes_per_vec)` and `scales.reserve(n)`. Simple and safe.

**Effort:** ~2–3 hours. **Risk:** Low — additive, no behavior change.

---

### PR-7: Add `as_slice().ok()` safety in `encode.rs`

**What:** In `encode.rs:44`, replace `as_slice().unwrap()` with explicit error handling:

```rust
let rotated: &[f32] = rotated_mat.as_slice().ok_or_else(|| 
    "encode: rotation result is not contiguous row-major slice"
)?;
```

**Effort:** ~1 hour. **Risk:** Very low.

---

## Tier 3: High Value / Higher Effort (3–7 days, high complexity)

### PR-8: Vectorize encode quantization loop *(addresses issue #32)*

**What:** Replace the sequential boundary-check loop in `encode.rs:48-53` with SIMD comparisons.

**Technical approach:**
- For each 4-bit code: 15 boundary checks → 4 `vcmpgtq` (NEON) / 4 `_mm256_cmpgt_ps` (AVX2) comparisons, then a SIMD pack to collect the 4-bit results
- Unroll across `BLOCK=32` vectors per iteration using the blocked layout
- Use `std::intrinsics` or explicit `std::arch` imports
- Keep the 2-bit path as-is (3 boundaries = trivially SIMD-able but not the bottleneck)

**Key constraint:** The quantization loop operates on `rotated` (f32), producing `codes` (u8). The boundary list is 15 entries for 4-bit. A SIMD approach:
1. Load 32 rotated values into a SIMD register
2. Compare against each of the 15 boundaries in 4 parallel comparisons (4 groups of ~4 boundaries each)
3. Pack the comparison results into 32 4-bit codes

**Verification:** Add a test at d=256 and d=384 comparing SIMD encode output against the sequential reference. Run distortion test (already exists in `tests/distortion.rs`) to confirm paper bounds still hold.

**Effort:** 3–5 days. **Risk:** Medium — correctness of SIMD quantization is non-trivial. Consider making it a behind-a-flag experimental path initially.

---

### PR-9: Parallelize encode per-vector operations

**What:** Wrap the per-vector sections of `encode()` in rayon parallel iterators.

The normalize step (loop over `i in 0..n`) and the scale computation step (loop over `i in 0..n` with inner `j in 0..dim` dot product) are both independent per-vector operations. Use `rayon::prelude::*`:

```rust
// Normalize: parallel over vectors
let norms: Vec<f32> = (0..n).into_par_iter().map(|i| {
    let row = &vectors[i * dim..(i + 1) * dim];
    row.iter().map(|x| x * x).sum::<f32>().sqrt()
}).collect();
```

For the scale computation, the inner dot product (rotated[i] · centroids[codes[i]]) is the heavy part — parallelizing just the outer vector loop gives a 4-8× speedup on 8-core machines.

**Effort:** 1–2 days. **Risk:** Low — purely additive parallelism, no SIMD complexity.

---

### PR-10: Fix LUT recall gap at low-dim 4-bit *(addresses issue #37)*

**What:** Investigate and fix the recall gap in the LUT scoring kernel at low dimensions.

**Step 1 — Reproduce:** Add a `kernel_correctness_lowdim` test in `turbovec/tests/` that runs the same kernel comparison at d=200 and d=256, k=1, bit_width=4. This gives us a baseline measurement.

**Step 2 — Investigate:** The `max_lut` formula in `search.rs:949-951` is the primary suspect. The x86 path uses `max_lut = floor(65535 / (n_byte_groups * 2)).min(127.0)` while the ARM path uses `max_lut = 127.0`. At low dims (small `n_byte_groups`), the x86 formula produces a larger `max_lut` → coarser u8 quantization → more rounding error → lower recall.

**Step 3 — Fix options:**
- Option A: Lower the `max_lut` ceiling specifically for dims where `n_byte_groups` is small
- Option B: Switch to a per-group adaptive `max_lut` instead of a global ceiling
- Option C: Use `f32` LUT values directly instead of u8 quantization (at cost of 4× LUT memory, but d=200 is only 200×32×4 = 25KB per query — negligible)

Option C is cleanest and avoids the quantization error entirely for low dims. At high dims the 4× memory cost matters more, so it could be gated on dim.

**Effort:** 3–5 days. **Risk:** Medium — requires careful benchmark comparison at multiple dim/bit_width configs.

---

### PR-11: Add automated AVX2 vs AVX-512BW epilogue equivalence test

**What:** Add a test in `tests/kernel_correctness.rs` that runs the same search inputs through both the AVX2 path and the AVX-512BW path (on hardware that supports both), asserting bit-identical scores.

**Why:** The `avx2_block_epilogue` function comment says it "mirrors the inline epilogue byte-for-byte" but there's no automated check. This is a correctness risk if someone modifies one path but forgets the other.

**Effort:** ~1 day. **Risk:** Very low — test-only, no production code change.

---

## Tier 4: Nice to Have (low priority, unbounded)

### PR-12: `ndarray::ArrayView2::from_shape` error handling improvement

Currently `encode.rs:41` uses `ArrayView2::from_shape(...).unwrap()`. The shape is guaranteed valid (n, dim from the vectors input), but a panic here would be confusing in production. Use `ok()` and return a `Result`-based error instead, or at minimum add an explicit comment explaining why the shape is always valid.

---

### PR-13: Python `ValueError` for non-contiguous numpy arrays

Currently `as_slice().expect()` will panic on non-contiguous numpy arrays. Replace with:
```python
let slice = arr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err(
    "vectors must be a contiguous numpy array"
))?;
```

---

### PR-14: CLI tool or Python script for index inspection

A `turbovec-info` CLI (or Python function) that reads a `.tv` or `.tvim` file and prints: format version, bit_width, dim, n_vectors, file size, estimated memory usage. Useful for debugging user issues with corrupted files.

---

## PR Sequencing Recommendation

```
Week 1: PR-1 (CONTRIBUTING.md) + PR-2 (__repr__) → merged
Week 2: PR-3 (lazy-index test) + PR-4 (duplicate ID test) → merged
Week 3: PR-5 (allowlist validation fix) + PR-6 (reserve capacity) → merged
Week 4: PR-7 (as_slice safety) + PR-9 (parallelize encode) → merged
Week 5+: PR-10 (LUT recall gap) — the big one, requires benchmarks
Week 6+: PR-8 (SIMD encode quantization) — most complex, consider starting after PR-10
```

---

## What Not to Touch

| Item | Reason |
|---|---|
| `gpu-support-investigation` branch | Exploratory draft, not ready for contribution |
| `search.rs` SIMD intrinsics without a benchmark setup | Risk of regressions; need hardware + existing baseline |
| Version bumps without corresponding changelog | Maintain consistency |
| The file format v1→v2 migration logic | Works correctly; changing it risks index corruption |
| The Lloyd-Max quantizer initialization | 200 iterations of Beta CDF + adaptive Simpson integration is deterministic and correct per the paper |