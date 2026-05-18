<p align="center">
  <img src="docs/header.png" alt="turbovec — Google's TurboQuant for vector search" width="100%">
</p>

<p align="center">
  <a href="https://github.com/RyanCodrai/turbovec/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/turbovec" alt="License"></a>
  <a href="https://pypi.org/project/turbovec/"><img src="https://img.shields.io/pypi/v/turbovec?label=pypi&color=blue" alt="PyPI version"></a>
  <a href="https://crates.io/crates/turbovec"><img src="https://img.shields.io/crates/v/turbovec?label=crates.io&color=blue" alt="crates.io version"></a>
  <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="TurboQuant paper"></a>
</p>

---

**A 10 million document corpus takes 31 GB of RAM as float32. turbovec fits it in 4 GB - and searches it faster than FAISS.**

turbovec is a Rust vector index with Python bindings, built on Google Research's [**TurboQuant**](https://arxiv.org/abs/2504.19874) algorithm - a data-oblivious quantizer that matches the Shannon lower bound on distortion with zero training and zero data passes.

- **No codebook training.** Add vectors, they're indexed. No data-dependent calibration, no rebuilds as the corpus grows.
- **Faster than FAISS.** Hand-written NEON (ARM) and AVX-512BW (x86) kernels beat FAISS IndexPQFastScan by 12–20% on ARM and match-or-beat it on x86.
- **Filter at search time.** Pass an id allowlist (or a slot bitmask) to `search()` and the kernel honours it directly. You always get up to `k` results from the allowed set — no over-fetching, no recall hit on selective filters.
- **Pure local.** No managed service, no data leaving your machine or VPC. Pair with any open-source embedding model for a fully air-gapped RAG stack.

Building RAG where privacy, memory, or latency matters? **You're in the right place.**

## Python

```bash
pip install turbovec
```

```python
from turbovec import TurboQuantIndex

index = TurboQuantIndex(dim=1536, bit_width=4)
index.add(vectors)
index.add(more_vectors)

scores, indices = index.search(query, k=10)

index.write("my_index.tq")
loaded = TurboQuantIndex.load("my_index.tq")
```

Need stable ids that survive deletes? Use `IdMapIndex`:

```python
import numpy as np
from turbovec import IdMapIndex

index = IdMapIndex(dim=1536, bit_width=4)
index.add_with_ids(vectors, np.array([1001, 1002, 1003], dtype=np.uint64))

scores, ids = index.search(query, k=10)   # ids are your uint64 external ids
index.remove(1002)                         # O(1) by id

index.write("my_index.tvim")
loaded = IdMapIndex.load("my_index.tvim")
```

### Hybrid retrieval (filtered search)

Restrict results to a candidate set produced by another system (SQL, BM25, ACL, time window, …):

```python
import numpy as np
from turbovec import IdMapIndex

idx = IdMapIndex(dim=1536, bit_width=4)
idx.add_with_ids(vectors, ids)

# Stage 1: external system narrows to candidate ids.
allowed = np.array(db.execute("SELECT id FROM docs WHERE tenant=?", (t,)).fetchall(),
                   dtype=np.uint64)

# Stage 2: dense rerank within the candidate set.
scores, ids = idx.search(query, k=10, allowlist=allowed)
```

Filtering happens inside the SIMD kernel at 32-vector block granularity: blocks with no allowed slots are short-circuited before any LUT lookup or scoring work, and individual non-allowed slots inside scored blocks are dropped at heap-insert. Selective allowlists (small fraction of the index allowed) therefore avoid most of the SIMD cost rather than paying it and discarding the result afterwards.

The output length is `min(k, len(allowed))` — when the allowlist is smaller than `k` you get exactly `len(allowed)` results rather than padded fallbacks.

See [`docs/api.md`](docs/api.md) for the full reference.

### Framework integrations

Drop-in replacements for the in-tree reference vector / document stores in each framework. Same public surface, same persistence semantics, same retriever and pipeline wiring — swap the import and keep your pipeline.

- [LangChain](docs/integrations/langchain.md) — `pip install turbovec[langchain]` · replaces `langchain_core.vectorstores.InMemoryVectorStore`
- [LlamaIndex](docs/integrations/llama_index.md) — `pip install turbovec[llama-index]` · replaces `llama_index.core.vector_stores.SimpleVectorStore`
- [Haystack](docs/integrations/haystack.md) — `pip install turbovec[haystack]` · replaces `haystack.document_stores.in_memory.InMemoryDocumentStore`
- [Agno](docs/integrations/agno.md) — `pip install turbovec[agno]` · replaces `agno.vectordb.lancedb.LanceDb`

## Rust

```bash
cargo add turbovec
```

```rust
use turbovec::TurboQuantIndex;

let mut index = TurboQuantIndex::new(1536, 4);
index.add(&vectors);
let results = index.search(&queries, 10);
index.write("index.tv").unwrap();
let loaded = TurboQuantIndex::load("index.tv").unwrap();
```

For stable external ids that survive deletes:

```rust
use turbovec::IdMapIndex;

let mut index = IdMapIndex::new(1536, 4);
index.add_with_ids(&vectors, &[1001, 1002, 1003]);
let (scores, ids) = index.search(&queries, 10);
index.remove(1002);
index.write("index.tvim").unwrap();
let loaded = IdMapIndex::load("index.tvim").unwrap();
```

## Recall

TurboQuant vs FAISS `IndexPQ` (LUT256, nbits=8) — the paper's Section 4.4 baseline. 100K vectors, k=64. FAISS PQ sub-quantizer counts sized to match TurboQuant's bit rate (m=d/4 at 2-bit, m=d/2 at 4-bit).

![Recall GloVe d=200](docs/recall_glove.svg)

![Recall d=1536](docs/recall_d1536.svg)

![Recall d=3072](docs/recall_d3072.svg)

Across OpenAI d=1536 and d=3072, TurboQuant beats FAISS by 0.4–3.4 points at R@1 across 2-bit and 4-bit, and both converge to 1.0 by k=4. GloVe d=200 is the harder regime — at low dim the asymptotic Beta assumption is looser. TurboQuant beats FAISS by 0.3 points at 4-bit and trails by 1.2 points at 2-bit at R@1, both closing to FAISS by k≈16.

**A note on baselines.** We compare against FAISS `IndexPQ` (LUT256, nbits=8, float32 LUT) because it's the default production-grade PQ most users would reach for. This is a stronger baseline than the custom u8-LUT PQ in the [TurboQuant paper](https://arxiv.org/abs/2504.19874) — FAISS uses a higher-precision LUT at scoring time and k-means++ for codebook training. We reproduce the paper's TurboQuant numbers on OpenAI d=1536 / d=3072 and hit similar numbers to other community reference implementations on low-dim embeddings (see [`turboquant-py`](https://pypi.org/project/turboquant-py/) at d=384). The visible gap on GloVe reflects FAISS being a strong baseline, not a TurboQuant implementation issue.

Full results: [d=1536 2-bit](benchmarks/results/recall_d1536_2bit.json), [d=1536 4-bit](benchmarks/results/recall_d1536_4bit.json), [d=3072 2-bit](benchmarks/results/recall_d3072_2bit.json), [d=3072 4-bit](benchmarks/results/recall_d3072_4bit.json), [GloVe 2-bit](benchmarks/results/recall_glove_2bit.json), [GloVe 4-bit](benchmarks/results/recall_glove_4bit.json).

## Compression

![Compression](docs/compression.svg)

## Search Speed

All benchmarks: 100K vectors, 1K queries, k=64, median of 5 runs.

### ARM (Apple M3 Max)

![ARM Speed — Single-threaded](docs/arm_speed_st.svg)

![ARM Speed — Multi-threaded](docs/arm_speed_mt.svg)

On ARM, TurboQuant beats FAISS FastScan by 12–20% across every config.

### x86 (Intel Xeon Platinum 8481C / Sapphire Rapids, 8 vCPUs)

![x86 Speed — Single-threaded](docs/x86_speed_st.svg)

![x86 Speed — Multi-threaded](docs/x86_speed_mt.svg)

On x86, TurboQuant wins every 4-bit config by 1–6% and runs within ~1% of FAISS on 2-bit ST. The 2-bit MT rows (d=1536 and d=3072) are the only configs sitting slightly behind FAISS (2–4%), where the inner accumulate loop is too short for unrolling amortization to match FAISS's AVX-512 VBMI path.

## How it works

Each vector is a direction on a high-dimensional hypersphere. TurboQuant compresses these directions using a simple insight: after applying a random rotation, every coordinate follows a known distribution -- regardless of the input data.

**1. Normalize.** Strip the length (norm) from each vector and store it as a single float. Now every vector is a unit direction on the hypersphere.

**2. Random rotation.** Multiply all vectors by the same random orthogonal matrix. After rotation, each coordinate independently follows a Beta distribution that converges to Gaussian N(0, 1/d) in high dimensions. This holds for any input data -- the rotation makes the coordinate distribution predictable.

**3. Lloyd-Max scalar quantization.** Since the distribution is known, we can precompute the optimal way to bucket each coordinate. For 2-bit, that's 4 buckets; for 4-bit, 16 buckets. The [Lloyd-Max algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) finds bucket boundaries and centroids that minimize mean squared error. These are computed once from the math, not from the data.

**4. Bit-pack.** Each coordinate is now a small integer (0-3 for 2-bit, 0-15 for 4-bit). Pack these tightly into bytes. A 1536-dim vector goes from 6,144 bytes (FP32) to 384 bytes (2-bit). That's 16x compression.

**5. Length-renormalized scoring.** Scalar quantization systematically underestimates inner products — the reconstructed unit direction is a little shorter than the original. We compute one scalar per vector at encode time — the inner product of the rotated unit vector with its own centroid reconstruction — and store `||v|| / ⟨u, x̂⟩` alongside each compressed vector. The search kernel multiplies the per-candidate score by this scalar before heap insertion, turning the inner-product estimator from downward-biased into unbiased at zero search-time cost and zero extra storage. The recall gain shows up most at low bit widths, where the quantization shrinkage is largest.

Encoding cost: one extra `d`-dimensional dot product per vector to compute `⟨u, x̂⟩`. On 1M vectors at d=1536 this is sub-second of additional encode time — a one-shot price paid at ingest, not at query.

**Search.** Instead of decompressing every database vector, we rotate the query once into the same domain and score directly against the codebook values. The scoring kernel uses SIMD intrinsics (NEON on ARM, AVX-512BW on modern x86 with an AVX2 fallback) with nibble-split lookup tables for maximum throughput.

The Lloyd-Max codebook achieves distortion within a factor of 2.7x of the information-theoretic lower bound (Shannon's distortion-rate limit); the length-renormalization step removes the residual bias the Lloyd-Max codebook introduces on the inner-product estimator itself.

## Building

### Python (via maturin)

```bash
pip install maturin
cd turbovec-python
maturin build --release
pip install target/wheels/*.whl
```

### Rust

```bash
cargo build --release
```

All x86_64 builds target `x86-64-v3` (AVX2 baseline, Haswell 2013+) via `.cargo/config.toml`. Any CPU that can run the AVX2 fallback kernel can run the whole crate — the AVX-512 kernel is gated at runtime via `is_x86_feature_detected!` and only kicks in on hardware that supports it.

## Running benchmarks

Download datasets:
```bash
python3 benchmarks/download_data.py all            # all datasets
python3 benchmarks/download_data.py glove          # GloVe d=200
python3 benchmarks/download_data.py openai-1536    # OpenAI DBpedia d=1536
python3 benchmarks/download_data.py openai-3072    # OpenAI DBpedia d=3072
```

Each benchmark is a self-contained script in `benchmarks/suite/`. Run any one individually:
```bash
python3 benchmarks/suite/speed_d1536_2bit_arm_mt.py
python3 benchmarks/suite/recall_d1536_2bit.py
python3 benchmarks/suite/compression.py
```

Run all benchmarks for a category:
```bash
for f in benchmarks/suite/speed_*arm*.py; do python3 "$f"; done    # all ARM speed
for f in benchmarks/suite/speed_*x86*.py; do python3 "$f"; done    # all x86 speed
for f in benchmarks/suite/recall_*.py; do python3 "$f"; done       # all recall
python3 benchmarks/suite/compression.py                            # compression
```

Results are saved as JSON to `benchmarks/results/`. Regenerate charts:
```bash
python3 benchmarks/create_diagrams.py
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026) -- the paper this implements
- [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497) (SIGMOD 2024) -- the source of the per-vector length-renormalization correction adapted in step 5
- [FAISS Fast accumulation of PQ and AQ codes](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) -- turbovec's x86 SIMD kernel adapts FastScan's pack layout, nibble-LUT scoring, and u16 accumulator strategy
