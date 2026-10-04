<p align="center">
  <img src="https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/header.png" alt="turbovec — Google's TurboQuant for vector search" width="100%">
</p>

<p align="center">
  <a href="https://github.com/RyanCodrai/turbovec/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/turbovec" alt="License"></a>
  <a href="https://pypi.org/project/turbovec/"><img src="https://img.shields.io/pypi/v/turbovec?label=pypi&color=blue" alt="PyPI version"></a>
  <a href="https://crates.io/crates/turbovec"><img src="https://img.shields.io/crates/v/turbovec?label=crates.io&color=blue" alt="crates.io version"></a>
  <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="TurboQuant paper"></a>
</p>

<p align="center">
  <a href="#python">Python</a> ·
  <a href="#rust">Rust</a> ·
  <a href="#how-it-compares">How it compares</a> ·
  <a href="#when-to-use-something-else">When to use something else</a> ·
  <a href="https://github.com/RyanCodrai/turbovec/blob/main/docs/api.md">API reference</a> ·
  <a href="https://github.com/RyanCodrai/turbovec/blob/main/docs/benchmarks.md">Benchmarks</a>
</p>

---

**Ten million 768-dimensional embeddings take 31 GB of RAM as float32. turbovec holds them in 4 GB and searches them 4.4× faster than FAISS, at matching recall.**

turbovec is an in-process vector index for Python and Rust. It stores each vector at 2 or 4 bits per dimension using Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) and searches that compressed form directly with hand-written SIMD kernels. There is no training step: add vectors and they are searchable.

- **No training, no rebuilds.** The codebook comes from the math, not your data. Add, remove and search in any order; the index never needs to be retrained as the corpus grows.
- **Small.** 7.8× smaller than float32 at 4-bit, 15.5× at 2-bit, with recall@1 ahead of FAISS's product quantizer at the same bit rate on five of six measured cells and 0.7 points behind on the sixth.
- **Fast.** 4.4× FAISS FastScan at 4-bit and 2.2× at 2-bit, on ARM (NEON) and x86 (AVX-512 VNNI, with AVX2 and scalar fallbacks): a staged search scans one bit plane first and rescores the shortlist exactly.
- **Filtered search.** Pass an allowlist of ids (or a bitmask) and the kernel skips blocks the filter excludes. You get up to `k` results from the allowed set, with no over-fetching.
- **Incremental, crash-safe saves.** `sync(path)` writes only what changed since the last call: one fsync, crash-safe at any byte, milliseconds for a small change however large the index.
- **Local.** A library, not a service. Nothing leaves your process; pair it with any open-source embedding model for a fully offline stack.

## Python

```bash
pip install turbovec
```

```python
import numpy as np
from turbovec import TurboQuantIndex

vectors = np.load("embeddings.npy").astype(np.float32)   # shape (n, 1536)

index = TurboQuantIndex(dim=1536, bit_width=4)
index.add(vectors)

scores, indices = index.search(vectors[:1], k=5)
print(indices)         # [[    0 86544  3385 84492 24967]]   int64 slot positions
print(scores.shape)    # (1, 5)                             float32 inner products

index.write("index.tv")                 # whole-file snapshot
index = TurboQuantIndex.load("index.tv")
index.add(more_vectors)
index.sync("index.tv")                  # writes just the change, durably
```

Inputs are 2-D `float32` arrays of shape `(n, dim)`; other dtypes raise rather than silently convert. Scores are inner products, so normalise your vectors if you want cosine similarity.

Need ids that survive deletes? `IdMapIndex` keys everything by your own `uint64` ids:

```python
from turbovec import IdMapIndex

index = IdMapIndex(dim=1536, bit_width=4)
index.add_with_ids(vectors, ids)               # ids: uint64 array, one per row

scores, ids = index.search(query, k=10)        # your ids, not slot positions
index.remove(1002)                             # O(1)

# Filter: only ids another system approved — a SQL predicate, an ACL, a time window.
allowed = np.array([1001, 1003, 1042], dtype=np.uint64)
scores, ids = index.search(query, k=10, allowlist=allowed)   # shape (1, min(k, 3))
```

Framework integrations are drop-in replacements for each framework's in-memory store — same public surface, same persistence, same retriever wiring:
[LangChain](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/langchain.md) (`pip install turbovec[langchain]`) ·
[LlamaIndex](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/llama_index.md) (`turbovec[llama-index]`) ·
[Haystack](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/haystack.md) (`turbovec[haystack]`) ·
[Agno](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/agno.md) (`turbovec[agno]`).

## Rust

```bash
cargo add turbovec
```

```rust
use turbovec::{IdMapIndex, TurboQuantIndex};

let mut index = TurboQuantIndex::new(1536, 4)?;
index.add(&vectors);                          // &[f32], row-major, n × 1536
let results = index.search(&queries, 10);     // scores and slot indices, nq × 10
index.write("index.tv")?;
let index = TurboQuantIndex::load("index.tv")?;

let mut ids = IdMapIndex::new(1536, 4)?;
ids.add_with_ids(&vectors, &[1001, 1002, 1003])?;
let (scores, found) = ids.search(&queries, 10);
ids.remove(1002);
```

The Rust API mirrors the Python one; [`docs/api.md`](https://github.com/RyanCodrai/turbovec/blob/main/docs/api.md) covers both, including the file formats, `sync()`, calibration and the staged search.

## How it compares

All figures: 100K OpenAI embeddings (d=1536 and d=3072) and GloVe (d=200), k=64, 1K queries, median of 5 runs, on a GCP c4a-standard-8 (ARM) and c3-standard-8 (x86), against FAISS at a matched bit rate. Scripts, raw JSON and every chart are in [`docs/benchmarks.md`](https://github.com/RyanCodrai/turbovec/blob/main/docs/benchmarks.md).

![Search speed, ARM, multi-threaded: turbovec vs FAISS IndexPQFastScan](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/arm_speed_mt.svg)

| | turbovec | FAISS | |
|---|---|---|---|
| Search, 4-bit | 3.8–5.1× faster across the eight cells (ARM 4.0×, x86 4.9×) | `IndexPQFastScan` | every cell |
| Search, 2-bit | 1.8–2.5× faster (ARM 2.05×, x86 2.35×) | `IndexPQFastScan` | every cell |
| Recall@1, OpenAI | 0.959 / 0.981 at 4-bit, 0.901 / 0.931 at 2-bit (d=1536 / d=3072) | 0.966 / 0.971, 0.876 / 0.911 | `IndexPQ`, m matched to bit rate |
| Recall@1, GloVe d=200 | 0.860 at 4-bit, 0.572 at 2-bit | 0.842, 0.565 | FAISS keeps a slim edge at 2-bit from k≈8 |
| Memory, d=1536 | 75 MB at 4-bit, 38 MB at 2-bit | 586 MB float32 | 7.8× / 15.5× |
| Single `add()` | 7.9–21.7 µs | 6.3–12.6× slower | into a trained, populated index |
| `remove(id)` | 1.4–5.8 µs | 0.19–1.03 **s** | FAISS repacks all codes per call |
| Whole-file save | 1.0–1.4× slower than FAISS | `write_index` | turbovec fsyncs and renames atomically |
| Load → first search | 1.0–1.5× slower than FAISS | `read_index` | 40 vs 31 ms at d=1536 4-bit on x86 |

Both recall columns reach 1.0 by k=8 on the OpenAI corpora. Where FAISS is ahead — snapshot save and load, GloVe at 2-bit beyond the first few results — the table says so; one benchmark on one machine is never the whole story, and the suite is yours to re-run.

## When to use something else

- **You need a graph index.** turbovec scans every vector (a staged scan, but a scan). At 100K vectors a query costs 0.1 ms (batched, 8 threads) to 1.2 ms (one query at a time, one thread), and the cost grows linearly with the corpus. For hundreds of millions of vectors behind strict latency, an HNSW or IVF index in front of it is the right shape, and turbovec is not one.
- **You need exact float results.** Quantization is lossy. On the OpenAI corpora the top result matches the float ground truth 96–98% of the time at 4-bit (90–93% at 2-bit) and the top-8 set is complete; at d=200 (GloVe) it is 86% and 57%, and on structureless data the gap is wider still. Check recall on a sample of your own data.
- **You need a server.** turbovec is a library with a file format. There is no network API, replication or multi-tenant service; the integrations above are how it slots into a stack that has those.
- **Your vectors are not multiples of 8 wide, or wider than 16,384.** Those are the dimension limits. Bit widths are 2, 3 and 4.

## How it works

Normalise each vector and store its length. Rotate every vector by one shared random orthogonal matrix, after which each coordinate follows a known distribution whatever the input. Quantise each coordinate with a Lloyd-Max codebook computed for that distribution — 4 levels at 2-bit, 16 at 4-bit — and bit-pack. At search time the query is rotated once and scored against the codebook with SIMD lookup tables; a per-vector scalar fixed at encode time removes the inner-product bias quantization introduces. An optional one-shot `calibrate(sample)` fits two scalars per coordinate for data that drifts from the asymptotic distribution (low-dimensional and word-vector embeddings gain most).

The full walk-through, with the maths and the references, is in [`docs/how-it-works.md`](https://github.com/RyanCodrai/turbovec/blob/main/docs/how-it-works.md).

## Building from source

<details>
<summary>Python wheel and Rust crate</summary>

```bash
pip install maturin
cd turbovec-python && maturin build --release && pip install target/wheels/*.whl
```

```bash
cargo build --release
```

x86_64 builds target `x86-64-v2` (SSE4.2, Nehalem 2008+) via `.cargo/config.toml`; the AVX-512 and AVX2 kernels are `#[target_feature]`-gated and chosen at runtime with `is_x86_feature_detected!`, so one binary runs everywhere and uses what the CPU has. Running the benchmark suite is described in [`docs/benchmarks.md`](https://github.com/RyanCodrai/turbovec/blob/main/docs/benchmarks.md#running-the-suite-yourself).

</details>

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026) — the paper this implements
- [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497) (SIGMOD 2024) — the per-vector length-renormalization correction
- [FAISS Fast accumulation of PQ and AQ codes](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) — turbovec's x86 kernel adapts FastScan's pack layout, nibble-LUT scoring and u16 accumulator strategy
