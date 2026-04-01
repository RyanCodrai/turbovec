# turbovec

Rust implementation of TurboQuant for vector search, with Python bindings via PyO3.

Compresses high-dimensional vectors to 2-4 bits per coordinate with near-optimal distortion. Data-oblivious (no training), zero indexing time.

Unofficial implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

## Usage

### Python

```python
from turbovec import TurboQuantIndex

index = TurboQuantIndex(dim=1536, bit_width=4)
index.add(vectors)
index.add(more_vectors)

scores, indices = index.search(query, k=10)

index.write("my_index.tq")
loaded = TurboQuantIndex.load("my_index.tq")
```

### Rust

```rust
use turbovec::TurboQuantIndex;

let mut index = TurboQuantIndex::new(1536, 4);
index.add(&vectors);
let results = index.search(&queries, 10);
index.write("index.tv").unwrap();
let loaded = TurboQuantIndex::load("index.tv").unwrap();
```

## Performance vs FAISS

TurboQuant vs FAISS IndexPQFastScan on OpenAI DBpedia d=1536 (100K vectors, 1K queries, k=64). FAISS PQ configurations sized to match TurboQuant compression ratios.

**TurboQuant requires zero training.** FAISS PQ needs a training step (4-10 seconds). TurboQuant index build is 3-4x faster.

### ARM (Apple Silicon M3 Max)

| | TQ speed | FAISS speed | Ratio | TQ recall@1 | FAISS recall@1 |
|---|---|---|---|---|---|
| **2-bit MT** | 0.125ms/q | 0.128ms/q | **0.97x** | 0.870 | 0.882 |
| **2-bit ST** | 1.272ms/q | 1.247ms/q | 1.02x | 0.870 | 0.882 |
| **4-bit MT** | 0.232ms/q | 0.246ms/q | **0.94x** | **0.955** | 0.930 |
| **4-bit ST** | 2.474ms/q | 2.485ms/q | **1.00x** | **0.955** | 0.930 |

On ARM, TurboQuant **matches or beats FAISS** on speed while requiring no training step. At 4-bit, TurboQuant recall is **higher than FAISS** (0.955 vs 0.930).

### x86 (Intel Sapphire Rapids, 4 vCPUs)

| | TQ speed | FAISS speed | Ratio | TQ recall@1 | FAISS recall@1 |
|---|---|---|---|---|---|
| **2-bit MT** | 0.733ms/q | 0.590ms/q | 1.24x | 0.870 | 0.882 |
| **2-bit ST** | 1.443ms/q | 1.208ms/q | 1.19x | 0.870 | 0.882 |
| **4-bit MT** | 1.391ms/q | 1.181ms/q | 1.18x | **0.955** | 0.930 |
| **4-bit ST** | 2.998ms/q | 2.477ms/q | 1.21x | **0.955** | 0.930 |

On x86, TurboQuant is within 18-25% of FAISS on speed. At 4-bit, TurboQuant recall is **higher than FAISS** (0.955 vs 0.930). The speed gap is primarily from TurboQuant's rotation step (~5% of total time) and differences in AVX2 code generation vs FAISS's template-instantiated C++ kernels.

### Compression

| Bit width | Index size (100K x 1536) | Compression vs FP32 |
|:----------|:------------------------|:--------------------|
| 2-bit     | 37.0 MB                  | 15.8x               |
| 4-bit     | 73.6 MB                  | 8.0x                |

## Benchmark results

Reproducing Section 4.4 of the paper. recall@1@k = probability that the true nearest neighbor appears in the top-k results. Benchmarked on Apple M3 Max.

### GloVe d=200 (100K database vectors, 10K queries)

**Recall:**

| k    | 2-bit | 4-bit |
|:-----|:------|:------|
| 1    | 0.505 | 0.812 |
| 2    | 0.658 | 0.932 |
| 4    | 0.789 | 0.986 |
| 8    | 0.879 | 0.998 |
| 16   | 0.943 | 1.000 |
| 32   | 0.976 | 1.000 |
| 64   | 0.992 | 1.000 |

**Search latency:**

| | 2-bit | 4-bit |
|---|---|---|
| MT | 0.029ms/q | 0.040ms/q |
| ST | 0.275ms/q | 0.429ms/q |

**Compression:**

| Bit width | Index size | Compression vs FP32 |
|:----------|:-----------|:--------------------|
| 2-bit     | 5.1 MB     | 14.8x               |
| 4-bit     | 9.9 MB     | 7.7x                |

### OpenAI DBpedia d=1536 (100K database vectors, 1K queries)

**Recall:**

| k    | 2-bit | 4-bit |
|:-----|:------|:------|
| 1    | 0.870 | 0.955 |
| 2    | 0.961 | 0.996 |
| 4    | 0.998 | 1.000 |
| 8    | 1.000 | 1.000 |
| 16   | 1.000 | 1.000 |
| 32   | 1.000 | 1.000 |
| 64   | 1.000 | 1.000 |

**Search latency:**

| | 2-bit | 4-bit |
|---|---|---|
| MT | 0.138ms/q | 0.256ms/q |
| ST | 1.448ms/q | 2.784ms/q |

**Compression:**

| Bit width | Index size | Compression vs FP32 |
|:----------|:-----------|:--------------------|
| 2-bit     | 37.0 MB    | 15.8x               |
| 4-bit     | 73.6 MB    | 8.0x                |

### OpenAI DBpedia d=3072 (100K database vectors, 1K queries)

**Recall:**

| k    | 2-bit | 4-bit |
|:-----|:------|:------|
| 1    | 0.912 | 0.967 |
| 2    | 0.986 | 0.997 |
| 4    | 1.000 | 1.000 |
| 8    | 1.000 | 1.000 |
| 16   | 1.000 | 1.000 |
| 32   | 1.000 | 1.000 |
| 64   | 1.000 | 1.000 |

**Search latency:**

| | 2-bit | 4-bit |
|---|---|---|
| MT | 0.395ms/q | 0.604ms/q |
| ST | 3.240ms/q | 5.711ms/q |

**Compression:**

| Bit width | Index size | Compression vs FP32 |
|:----------|:-----------|:--------------------|
| 2-bit     | 73.6 MB    | 15.9x               |
| 4-bit     | 146.9 MB   | 8.0x                |

## How it works

Each vector is a direction on a high-dimensional hypersphere. TurboQuant compresses these directions using a simple insight: after applying a random rotation, every coordinate follows a known distribution -- regardless of the input data.

**1. Normalize.** Strip the length (norm) from each vector and store it as a single float. Now every vector is a unit direction on the hypersphere.

**2. Random rotation.** Multiply all vectors by the same random orthogonal matrix. After rotation, each coordinate independently follows a Beta distribution that converges to Gaussian N(0, 1/d) in high dimensions. This holds for any input data -- the rotation makes the coordinate distribution predictable.

**3. Lloyd-Max scalar quantization.** Since the distribution is known, we can precompute the optimal way to bucket each coordinate. For 2-bit, that's 4 buckets; for 4-bit, 16 buckets. The [Lloyd-Max algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) finds bucket boundaries and centroids that minimize mean squared error. These are computed once from the math, not from the data.

**4. Bit-pack.** Each coordinate is now a small integer (0-3 for 2-bit, 0-15 for 4-bit). Pack these tightly into bytes. A 1536-dim vector goes from 6,144 bytes (FP32) to 384 bytes (2-bit). That's 16x compression.

**Search.** Instead of decompressing every database vector, we rotate the query once into the same domain and score directly against the codebook values. The scoring kernel uses SIMD intrinsics (NEON on ARM, AVX2 on x86) with nibble-split lookup tables for maximum throughput.

The paper proves this achieves distortion within a factor of 2.7x of the information-theoretic lower bound (Shannon's distortion-rate limit). You cannot do much better for a given number of bits.

**Online by design.** Because the codebook and rotation are derived from math (not from the data), new vectors can be added at any time without rebuilding the index. Traditional methods like Product Quantization require expensive offline codebook training that must be re-run when data changes.

## Architecture

The project is a Cargo workspace with two crates:

- **turbovec** -- pure Rust crate with zero Python dependency. Contains all SIMD kernels, encoding, and search logic.
- **turbovec-python** -- thin PyO3 wrapper exposing `TurboQuantIndex` to Python.

### Search pipeline

- **NEON kernel (ARM):** Sequential code layout. `vqtbl1q_u8` shuffle-based LUT scoring with `vaddw_u8` widening accumulation. Per-block heap with QBS=4 query batching.
- **AVX2 kernel (x86):** FAISS-style perm0-interleaved code layout to work around AVX2's cross-lane shuffle constraint. `vpshufb` with uint16 reinterpret trick and `combine2x2` flush. Multi-query NQ=4 scoring with fused heap and SIMD early-rejection.
- **Rotation:** BLAS-accelerated matrix multiplication via ndarray. Uses Accelerate (macOS) or OpenBLAS (Linux).

## Building

### Python (via maturin)

```bash
pip install maturin
cd turbovec-python
RUSTFLAGS="-C target-cpu=native" maturin build --release
pip install target/wheels/*.whl
```

### Rust

```bash
cargo build --release
```

## Running benchmarks

Download datasets:
```
python3 benchmarks/benchmark.py download glove openai-1536 openai-3072
```

Run benchmarks:
```
python3 benchmarks/benchmark.py glove openai-1536 openai-3072
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026) -- the paper this implements
- [FAISS Fast accumulation of PQ and AQ codes](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) -- the FAISS FastScan approach our x86 kernel is based on
