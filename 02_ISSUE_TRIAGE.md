# turbovec Issue Triage

## Open Issues Summary

| # | Title | Labels | Priority | Age |
|---|---|---|---|---|
| **40** | Add CONTRIBUTING.md with contribution guidelines | `enhancement` | **Medium** | 2026-05-19 |
| **37** | LUT scoring kernel loses up to 1.4pp recall@1 vs exact-math on low-dim 4-bit | `enhancement` | **High** | 2026-05-18 |
| **32** | Optimise insertion: vectorise + parallelise the encode kernels | `enhancement` | **High** | 2026-05-19 |

**3 open issues** · 0 bugs reported · 1 draft PR (GPU support investigation)

---

## Issue Deep Dives

### #40 — Add CONTRIBUTING.md *(medium, documentation)*

**What it's asking for:** A `CONTRIBUTING.md` file with contribution guidelines at the repo root.

**Why it matters:** Without it, external contributors have no guidance on branch conventions, PR conventions, test expectations, or build requirements. Given the repo is actively developed (10 releases in ~6 weeks), this is a basic hygiene item.

**How to address it:**
1. Copy the template from a well-maintained Rust project (e.g. `rayon`, `ripgrep`)
2. Cover: branch strategy (main + feature branches from upstream), PR checklist (tests pass, benchmark delta if performance-relevant), build prerequisites (Rust 1.70+, OpenBLAS dev libs, maturin for Python), and CI expectations
3. Mention `cargo test` for Rust tests and `pip install -e .` + pytest for Python tests
4. The Rust crate version is ahead of the Python package version (0.4.1 vs 0.5.1) — clarify which gets bumped on which type of change

**Effort:** ~1–2 hours. Low technical risk. Good first PR for a new contributor.

---

### #37 — LUT scoring kernel loses up to 1.4pp recall@1 vs exact-math on low-dim 4-bit *(high, correctness/performance)*

**What it's about:** The LUT-based SIMD scoring path produces slightly lower recall at low dimensions (e.g. GloVe d=200) compared to exact floating-point math on the same algorithm. The gap is attributed to the FAISS-style per-sub-table u8 quantization in `build_query_neon_lut_from_slice` — specifically the `max_lut = floor(65535 / (n_byte_groups * 2)).min(127.0)` clamp (line 949 in `search.rs`).

**Root cause hypothesis:** At low dimensions, `n_byte_groups` is small (e.g. d=200 with 4-bit → 50 groups, so `max_lut = floor(65535/100) = 127`), so the u8 quantization range is wide enough that rounding errors are small. At even lower dims or with 4-bit, the sub-table spans may be large enough that the `round().clamp(0, max_lut)` step introduces meaningful quantization error on the inner-product lookup values.

**Why it's complex:** The existing kernel correctness tests (`tests/kernel_correctness.rs`) verify exact-math equivalence on high-dim (d=1536, 3072) configs where the gap is negligible. To catch this, you'd need a new test at d=200 or d=256 with k=1 at 4-bit — which is exactly the regime where the paper's asymptotic assumptions are weakest.

**How to address it:**
1. Add a test at low dim (d=200, d=256) comparing LUT-kernel scores against exact-math reference — this documents the known gap
2. Investigate whether reducing `max_lut` (or changing the quantization strategy per sub-table) narrows the gap at low dims without regressing high-dim throughput
3. If the fix is architecture-specific (e.g., different `max_lut` for dims ≤ 256), gate it behind a dim check in `build_query_neon_lut_from_slice`

**Effort:** 1–3 days. Requires understanding the SIMD kernel and LUT quantization interaction.

---

### #32 — Optimise insertion: vectorise + parallelise the encode kernels *(high, performance)*

**What it's about:** Currently `encode.rs` processes vectors sequentially (iterating `n` in a for loop). The encode pipeline is: normalize → rotate (BLAS matmul via ndarray) → quantize (bit-plane accumulation) → bit-pack → compute per-vector correction scale. Most of these are O(n·d) with good data parallelism potential.

**Current encode perf:** Not benchmarked in the suite, but the README mentions "Encoding cost: one extra d-dimensional dot product per vector" for the correction scale. On 1M vectors at d=1536 that's sub-second additional encode time, but at lower dims or larger batches the sequential loop could dominate.

**Why it's interesting:** `encode::encode()` does:
1. Sequential norm extraction + per-vector loop (line 30-38) — trivially parallelizable
2. `ndarray::dot` for rotation — already BLAS-parallelized
3. Sequential quantization with inner loop over dim (line 48-53) — the heaviest loop
4. Sequential per-vector scale computation with inner loop over dim (line 60-75) — another hot loop
5. Bit-packing loop (line 84-103) — sequential

The quantization loop (step 3) is the heaviest: for each vector × each coordinate it does one comparison per boundary. For 4-bit that's 15 boundary checks per coord. At d=1536 and 1M vectors that's 22.8 billion comparisons — this absolutely needs SIMD.

**How to address it:**
1. **SIMD quantize:** Replace the sequential `codes[idx] += (rotated[idx] > boundary)` with SIMD comparisons (NEON `vcgtq` / AVX2 `_mm256_cmpgt_ps`) applied across the rotation output. The boundary check itself requires one SIMD comparison per codebook level — 15 for 4-bit, 3 for 2-bit.
2. **Parallelize the outer loop:** Wrap the per-vector ops in rayon `par_bridge()` or `par_chunks()` since they're independent
3. **Fuse normalize + rotate:** Since normalize just computes norms and unit vectors, the rotation step could be done in-place on the already-allocated unit vectors (no need to copy first)
4. **SIMD bit-pack:** The bit-plane packing is memory-bandwidth bound but could benefit from SWAR bit manipulation

**Effort:** 3–5 days. High technical complexity in the SIMD intrinsics. Involves modifying the critical path of the encode pipeline.

---

## PR Status

| PR | Title | State | Note |
|---|---|---|---|
| #39 | README: hybrid-retrieval filtering as block-level early exit | **MERGED** | Just merged |
| #38 | Block-level early-exit for selective mask searches | **MERGED** | Recent, major perf work |
| #36 | Haystack: clamp cosine scores | **MERGED** | |
| #35 | Length-renormalized scoring (POC) | **MERGED** | |
| #33 | CI: Windows x64 wheel on release | **MERGED** | |
| #29 | Add Agno framework integration | **MERGED** | |
| #28 | Add distortion (MSE) benchmark | **CLOSED** | (contributor PR, closed in favor of internal impl) |
| #20 | Apple GPU (MLX) backend — phase 1 scaffolding | **DRAFT** | Long-lived investigation branch |

---

## Recommendations for Contributors

1. **Start with #40** (CONTRIBUTING.md) — no risk, clear deliverable, immediate value
2. **Then #37** (LUT recall gap) — needs careful benchmarking at low dims; good for someone who wants to learn the SIMD kernel
3. **Then #32** (encode optimization) — highest payoff but requires Rust SIMD experience

**Avoid:** The GPU support investigation branch (#20) is a draft exploratory branch — not ready for contribution.