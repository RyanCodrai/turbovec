# turbovec Repository Map

## What is turbovec?

**turbovec** implements Google's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm — a data-oblivious vector quantizer that compresses high-dimensional vectors to 2–4 bits/coordinate with near-optimal distortion, requiring zero training and zero data passes.

- **Core claim:** Fits a 10M-document corpus (31 GB float32) into 4 GB RAM, searching faster than FAISS.
- **Languages:** Rust core + Python bindings (via PyO3/maturin).
- **License:** MIT.

## Repository Structure

```
turbovec/                         # Rust crate (crates.io: turbovec)
├── src/
│   ├── lib.rs                    # Public API: TurboQuantIndex, SearchResults, BLOCK, FLUSH_EVERY
│   ├── encode.rs                 # Normalize → rotate → quantize → bit-pack → per-vector scale
│   ├── search.rs                 # SIMD scoring kernels (NEON/AVX2/AVX-512BW), LUT build, heap top-k
│   ├── pack.rs                   # Bit-plane → SIMD-blocked layout repacking (x86 perm0 / ARM seq)
│   ├── rotation.rs               # Deterministic orthogonal matrix via QR(SeededGaussian)
│   ├── codebook.rs              # Lloyd-Max quantizer for Beta(d/2, d/2) distribution
│   ├── id_map.rs                # IdMapIndex wrapper (stable external u64 IDs)
│   └── io.rs                    # .tv / .tvim file serialization (format version 2)
├── tests/                        # Rust unit/integration tests
│   ├── kernel_correctness.rs    # Verifies SIMD kernel output vs exact-math reference
│   ├── filtering.rs             # Mask/allowlist search correctness
│   ├── distortion.rs           # MSE benchmark verifying paper distortion bounds
│   ├── codebook.rs / encode.rs / rotation.rs
│   ├── id_map.rs / swap_remove.rs / io_versioning.rs
│   ├── lazy_init.rs / concurrent_search.rs
└── examples/
    └── dump_state.rs

turbovec-python/                  # Python package (PyPI: turbovec)
├── src/lib.rs                    # PyO3 bindings: TurboQuantIndex + IdMapIndex Python classes
├── python/turbovec/__init__.py   # Public Python API (pip install turbovec)
└── tests/                        # Python integration tests
    ├── test_index.py / test_id_map.py / test_filtering.py
    ├── test_langchain.py / test_llama_index.py / test_haystack.py / test_agno.py

benchmarks/
├── suite/                        # Individual benchmark scripts (speed / recall / compression)
│   └── *.py                      # e.g. speed_d1536_4bit_arm_mt.py, recall_glove_2bit.py
├── results/                     # JSON results: recall_d1536_2bit.json, etc.
├── create_diagrams.py           # Regenerate SVG charts from results
└── download_data.py             # Download GloVe, OpenAI DBpedia datasets

docs/
├── api.md                        # Full API reference
├── integrations/                 # Framework integration guides
│   ├── langchain.md / llama_index.md / haystack.md / agno.md
└── *.svg                         # Charts: recall_*, arm_speed_*, x86_speed_*, compression.svg

.github/workflows/
├── release-crates.yml           # `cargo publish` on version tag (v*)
└── release-pypi.yml             # Build+upload Python wheels on version tag (py-v*)

Cargo.toml (workspace root)       # Workspace: turbovec + turbovec-python
Cargo.toml (turbovec/)           # [package] version 0.4.1, edition 2021, rust-version 1.70
turbovec-python/pyproject.toml   # [project] version 0.5.1, requires-python >=3.9
```

## Key Architecture Decisions

| Aspect | Detail |
|---|---|
| **Bit widths** | 2, 3, or 4 bits per coordinate (asserted at construction) |
| **DIM constraint** | Must be multiple of 8 |
| **Lazy index** | `TurboQuantIndex::new_lazy(bit_width)` defers dim commitment to first `add_2d` |
| **Rotation** | Deterministic (seed=42) via QR(Gaussian matrix) — not data-dependent |
| **Scoring** | Length-renormalized: stored scalar = `‖v‖ / ⟨u_rot, x̂⟩` (RaBitQ-style) |
| **File format** | Version 2 (0.4.4+): `.tv` = "TVPI" magic + header + codes + scales; `.tvim` = "TVIM" + slot_to_id table |
| **SIMD kernels** | `target_arch = "aarch64"`: NEON 4-query fused kernel; x86_64: AVX2 4-query then AVX-512BW 8-query via runtime dispatch |
| **Blocking** | BLOCK=32 vectors per SIMD block; FLUSH_EVERY=256 groups per batch |
| **Concurrent search** | `search` takes `&self`; caches initialized lazily via `OnceLock`; `add` resets blocked cache |
| **x86 baseline** | `-C target-cpu=x86-64-v3` (AVX2 baseline); AVX-512 is runtime-gated |

## Dependencies (Rust)

- `ndarray` + BLAS (Accelerate on macOS, OpenBLAS on Linux)
- `rayon` for parallel LUT building
- `faer` for matrix operations (rotation, GEMM)
- `statrs` for Beta distribution (Lloyd-Max quantizer)
- `rand_chacha`, `rand_distr` for seeded Gaussian + StandardNormal

## CI/CD

- **crates.io:** Publish on `v*` tag → `cargo publish -p turbovec`
- **PyPI:** Publish on `py-v*` tag → maturin wheels for x86_64 + aarch64 (manylinux_2_28)
- Python 3.11 on GitHub Actions; no cross-platform test matrix for ARM on x86 CI

## Remote Repositories

```
origin   https://github.com/okwn/turbovec.git   (your fork — has gpu-support-investigation branch)
upstream https://github.com/RyanCodrai/turbovec.git  (canonical)
```