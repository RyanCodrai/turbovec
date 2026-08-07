<p align="center">
  <img src="https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/header.png" alt="turbovec — Google's TurboQuant for vector search" width="100%">
</p>

<p align="center">
  <a href="https://github.com/RyanCodrai/turbovec/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/turbovec" alt="License"></a>
  <a href="https://pypi.org/project/turbovec/"><img src="https://img.shields.io/pypi/v/turbovec?label=pypi&color=blue" alt="PyPI version"></a>
  <a href="https://crates.io/crates/turbovec"><img src="https://img.shields.io/crates/v/turbovec?label=crates.io&color=blue" alt="crates.io version"></a>
  <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="TurboQuant paper"></a>
</p>

---

**A 10 million document corpus takes 31 GB of RAM as float32. turbovec fits it in 4 GB - and searches it faster than FAISS.**

turbovec is a Rust vector index with Python bindings, built on Google Research's [**TurboQuant**](https://arxiv.org/abs/2504.19874) algorithm — a data-oblivious quantizer with near-optimal distortion and no separate training phase.

- **Online ingest.** Add vectors, they're indexed — no train step, no parameter tuning, no rebuilds as the corpus grows.
- **Fast SIMD search.** Hand-written kernels — NEON SDOT/SMMLA on ARM, AVX-512 VNNI and `vpermb` on x86, with AVX2 and scalar fallbacks — beat FAISS IndexPQFastScan in every measured config: 3.2–3.7× at 4-bit and 5–27% at 2-bit, on both architectures.
- **Incremental saves.** `sync(path)` persists just what changed since the last sync — one fsync per call, crash-safe at any byte, and a removal or a small append costs milliseconds however large the index. `write`/`load` stay for whole-file snapshots.
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

index.write("my_index.tv")
loaded = TurboQuantIndex.load("my_index.tv")

index.sync("my_index.tv")   # after more changes: durable incremental save
```

`vectors` and `query` are 2-D `float32` arrays of shape `(n, dim)` — other dtypes are rejected rather than silently converted, so cast with `np.asarray(x, dtype=np.float32)` first if needed.

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

index.sync("my_index.tvim")   # durable incremental save, ids included
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

The output length is `min(k, n_allowed)`, where `n_allowed` counts *distinct* allowed vectors — when fewer vectors are allowed than `k` you get exactly that many results rather than padded fallbacks.

See [`docs/api.md`](https://github.com/RyanCodrai/turbovec/blob/main/docs/api.md) for the full reference.

### Framework integrations

Drop-in replacements for the in-tree reference vector / document stores in each framework. Same public surface, same persistence semantics, same retriever and pipeline wiring — swap the import and keep your pipeline.

- [LangChain](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/langchain.md) — `pip install turbovec[langchain]` · replaces `langchain_core.vectorstores.InMemoryVectorStore`
- [LlamaIndex](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/llama_index.md) — `pip install turbovec[llama-index]` · replaces `llama_index.core.vector_stores.SimpleVectorStore`
- [Haystack](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/haystack.md) — `pip install turbovec[haystack]` · replaces `haystack.document_stores.in_memory.InMemoryDocumentStore`
- [Agno](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/agno.md) — `pip install turbovec[agno]` · replaces `agno.vectordb.lancedb.LanceDb`

## Rust

```bash
cargo add turbovec
```

```rust
use turbovec::TurboQuantIndex;

let mut index = TurboQuantIndex::new(1536, 4).unwrap();
index.add(&vectors);
let results = index.search(&queries, 10);
index.write("index.tv").unwrap();
let loaded = TurboQuantIndex::load("index.tv").unwrap();
```

For stable external ids that survive deletes:

```rust
use turbovec::IdMapIndex;

let mut index = IdMapIndex::new(1536, 4).unwrap();
index.add_with_ids(&vectors, &[1001, 1002, 1003]).unwrap();
let (scores, ids) = index.search(&queries, 10);
index.remove(1002);
index.write("index.tvim").unwrap();
let loaded = IdMapIndex::load("index.tvim").unwrap();
```

## Recall

TurboQuant vs FAISS `IndexPQ` (LUT256, nbits=8) — the paper's Section 4.4 baseline. 100K vectors, k=64. FAISS PQ sub-quantizer counts sized to match TurboQuant's bit rate (m=d/4 at 2-bit, m=d/2 at 4-bit).

![Recall GloVe d=200](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/recall_glove.svg)

![Recall d=1536](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/recall_d1536.svg)

![Recall d=3072](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/recall_d3072.svg)

Across OpenAI d=1536 and d=3072, TurboQuant edges FAISS by 0.1–1.2 points at R@1 across 2-bit and 4-bit, and both reach 1.0 by k=8 (≥0.997 already at k≤4). GloVe d=200 is the harder regime — at low dim the asymptotic Beta assumption is looser. TurboQuant beats FAISS by 1.6 points at 4-bit but trails by 1.4 points at 2-bit at R@1, both tracking FAISS closely by k≈16.

**A note on baselines.** We compare against FAISS `IndexPQ` (LUT256, nbits=8, float32 LUT) because it's the default production-grade PQ most users would reach for. This is a stronger baseline than the custom u8-LUT PQ in the [TurboQuant paper](https://arxiv.org/abs/2504.19874) — FAISS uses a higher-precision LUT at scoring time and k-means++ for codebook training. We reproduce the paper's TurboQuant numbers on OpenAI d=1536 / d=3072 and hit similar numbers to other community reference implementations on low-dim embeddings (see [`turboquant-py`](https://pypi.org/project/turboquant-py/) at d=384). On GloVe (d=200) — the low-dim regime where the asymptotic Beta assumption is loosest — TurboQuant lands ahead of FAISS at 4-bit but trails it at 2-bit; TQ+ calibration closes most, not all, of the low-dim gap the base algorithm leaves.

Full results: [d=1536 2-bit](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/recall_d1536_2bit.json), [d=1536 4-bit](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/recall_d1536_4bit.json), [d=3072 2-bit](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/recall_d3072_2bit.json), [d=3072 4-bit](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/recall_d3072_4bit.json), [GloVe 2-bit](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/recall_glove_2bit.json), [GloVe 4-bit](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/recall_glove_4bit.json).

## Compression

![Compression](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/compression.svg)

## Search Speed

All benchmarks: 100K vectors, 1K queries, k=64, median of 5 runs.

### ARM (GCP c4a-standard-8, Google Axion, 8 vCPUs)

![ARM Speed — Single-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/arm_speed_st.svg)

![ARM Speed — Multi-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/arm_speed_mt.svg)

On ARM, TurboQuant beats FAISS FastScan in every config: 3.4–3.7× at 4-bit (the SDOT/SMMLA dot-product kernels score the vector-major layout directly) and 17–26% at 2-bit.

### x86 (Intel Xeon Platinum 8481C / Sapphire Rapids, 8 vCPUs)

![x86 Speed — Single-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/x86_speed_st.svg)

![x86 Speed — Multi-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/x86_speed_mt.svg)

On x86, TurboQuant now wins every config: 3.2–3.5× at 4-bit (the AVX-512 VNNI dot-product kernel on the vector-major layout) and 5–27% at 2-bit, where the `vpermb` LUT scan closes the gap FAISS's VBMI path used to hold on the short 2-bit accumulate loop.

## Insertion & Removal Speed

Same corpus as the search cells: 100K OpenAI vectors, median of 5 runs, fresh index per timed run. Insertion measures bulk `add()` into an empty index (one-time rotation/codebook init included) against FAISS `IndexPQFastScan` bulk add (training untimed). A warm 10K append matches the bulk rate within measurement noise — within ±2% on the single-threaded cells and ±4% on the noisier multi-threaded ones, scattering in both directions — because calibration is explicit and the steady-state encode path is the bulk path. It is therefore not charted separately (the `tq_warm_insert_vecs_per_sec` field remains in the JSONs). Removal measures per-op latency of `IdMapIndex.remove(id)` against raw `TurboQuantIndex.swap_remove` — both O(1) swap-and-pop; the gap is the id-map bookkeeping. Single-threaded cells pin `RAYON_NUM_THREADS=1`. Scripts: [`benchmarks/suite/`](https://github.com/RyanCodrai/turbovec/tree/main/benchmarks/suite/). On ARM the FAISS baseline is the generic aarch64 `faiss-cpu` wheel, so its insert figure partly reflects that wheel's PQ-training path on Axion (the previous ARM environment linked Apple's Accelerate BLAS instead), which is a build/BLAS difference rather than a like-for-like algorithmic gap.

### ARM (GCP c4a-standard-8, Google Axion, 8 vCPUs)

![ARM Insertion — Single-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/arm_insert_st.svg)

![ARM Insertion — Multi-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/arm_insert_mt.svg)

![ARM Removal — Single-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/arm_remove_st.svg)

Full results: [d=1536 2-bit insert ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d1536_2bit_arm_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d1536_2bit_arm_mt.json), [d=1536 4-bit insert ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d1536_4bit_arm_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d1536_4bit_arm_mt.json), [d=3072 2-bit insert ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d3072_2bit_arm_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d3072_2bit_arm_mt.json), [d=3072 4-bit insert ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d3072_4bit_arm_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d3072_4bit_arm_mt.json), and the matching [`speed_remove_*`](https://github.com/RyanCodrai/turbovec/tree/main/benchmarks/results/) files.

### x86 (Intel Xeon Platinum 8481C / Sapphire Rapids, 8 vCPUs)

![x86 Insertion — Single-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/x86_insert_st.svg)

![x86 Insertion — Multi-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/x86_insert_mt.svg)

![x86 Removal — Single-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/x86_remove_st.svg)

Full results: [d=1536 2-bit insert ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d1536_2bit_x86_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d1536_2bit_x86_mt.json), [d=1536 4-bit insert ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d1536_4bit_x86_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d1536_4bit_x86_mt.json), [d=3072 2-bit insert ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d3072_2bit_x86_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d3072_2bit_x86_mt.json), [d=3072 4-bit insert ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d3072_4bit_x86_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_insert_d3072_4bit_x86_mt.json), and the matching [`speed_remove_*`](https://github.com/RyanCodrai/turbovec/tree/main/benchmarks/results/) files.

## Save & Load

Same corpus as the search cells: 100K OpenAI vectors, median of 5 runs. TurboQuant serializes to a single `.tv` file with an fsync + atomic rename; FAISS is `write_index` / `read_index` on the precision-matched `IndexPQFastScan` (sub-quantizer count matched to TurboQuant's bit rate, as in the search cells). **Save (warm)** is a write after a search has run, so the blocked layout cache is populated. **Load → first search** opens a fresh index and times the first query — separating bare deserialization (the page cache is warm throughout, so this is layout work, not cold-storage I/O) from the first-query cost. **Round-trip** chains the checkpoint/resume cycle an embedding store actually pays — mutate 1K vectors → save → reopen → serve the first query; FAISS has no measured equivalent for this path, so it is shown for TurboQuant only. On the smaller payloads the round-trip can come in *below* the isolated post-mutation ("dirty") write: the two are timed in separate suite steps, and at small file sizes the standalone `fsync` in the dirty-write step dominates and inflates it — a measurement artifact of the harness, not a repack win in the combined path. Single-threaded cells pin `RAYON_NUM_THREADS=1`. Scripts: [`benchmarks/suite/`](https://github.com/RyanCodrai/turbovec/tree/main/benchmarks/suite/).

### ARM (GCP c4a-standard-8, Google Axion, 8 vCPUs)

![ARM Save/Load — Single-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/arm_persist_st.svg)

![ARM Save/Load — Multi-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/arm_persist_mt.svg)

Full results: [d=1536 2-bit persist ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d1536_2bit_arm_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d1536_2bit_arm_mt.json), [d=1536 4-bit persist ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d1536_4bit_arm_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d1536_4bit_arm_mt.json), [d=3072 2-bit persist ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d3072_2bit_arm_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d3072_2bit_arm_mt.json), [d=3072 4-bit persist ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d3072_4bit_arm_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d3072_4bit_arm_mt.json).

### x86 (Intel Xeon Platinum 8481C / Sapphire Rapids, 8 vCPUs)

![x86 Save/Load — Single-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/x86_persist_st.svg)

![x86 Save/Load — Multi-threaded](https://raw.githubusercontent.com/RyanCodrai/turbovec/main/docs/x86_persist_mt.svg)

Full results: [d=1536 2-bit persist ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d1536_2bit_x86_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d1536_2bit_x86_mt.json), [d=1536 4-bit persist ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d1536_4bit_x86_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d1536_4bit_x86_mt.json), [d=3072 2-bit persist ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d3072_2bit_x86_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d3072_2bit_x86_mt.json), [d=3072 4-bit persist ST](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d3072_4bit_x86_st.json), [MT](https://github.com/RyanCodrai/turbovec/blob/main/benchmarks/results/speed_persist_d3072_4bit_x86_mt.json).

## How it works

Each vector is a direction on a high-dimensional hypersphere. TurboQuant compresses these directions using a simple insight: after applying a random rotation, every coordinate follows a known distribution -- regardless of the input data.

**1. Normalize.** Strip the length (norm) from each vector and store it as a single float. Now every vector is a unit direction on the hypersphere.

**2. Random rotation.** Multiply all vectors by the same random orthogonal matrix. After rotation, each coordinate independently follows a Beta distribution that converges to Gaussian N(0, 1/d) in high dimensions. This holds for any input data -- the rotation makes the coordinate distribution predictable.

**3. Per-coordinate calibration (TQ+).** The Beta distribution from step 2 is asymptotic — at finite dimensions, individual coordinates drift from the canonical shape (especially low-bit and word-vector-style embeddings). TQ+ fits two scalars per coordinate — a shift and a scale — mapping each coordinate's empirical quantiles onto the codebook's outermost centroids. The probability level comes from the codebook, so it tracks the bit width (~0.933 at 2-bit, ~0.996 at 4-bit) rather than being fixed. The Lloyd-Max codebook then quantizes against the *target* distribution it was designed for. The fit is explicit: call `index.calibrate(sample)` once with a random, representative sample of your vectors (~1024 rows is enough — a draw of that size matches fitting on the whole corpus) before adding; afterwards the calibration is committed and reused by every add — no retraining, no rebuilds, no separate train phase. An index you never calibrate is plain TurboQuant. `index.calibration_state` reports `"uncalibrated"` or `"calibrated"`. Recall gain: up to +1.4pp at @1 on the cells that drift most (e.g. GloVe at 2-bit).

**4. Lloyd-Max scalar quantization.** Since the distribution is known, we can precompute the optimal way to bucket each coordinate. For 2-bit, that's 4 buckets; for 4-bit, 16 buckets. The [Lloyd-Max algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) finds bucket boundaries and centroids that minimize mean squared error. These are computed once from the math, not from the data.

**5. Bit-pack.** Each coordinate is now a small integer (0-3 for 2-bit, 0-15 for 4-bit). Pack these tightly into bytes. A 1536-dim vector goes from 6,144 bytes (FP32) to 384 bytes (2-bit). That's 16x compression.

**6. Length-renormalized scoring.** Scalar quantization systematically underestimates inner products — the reconstructed unit direction is a little shorter than the original. We compute one scalar per vector at encode time — the inner product of the rotated unit vector with its own centroid reconstruction — and store `||v|| / ⟨u, x̂⟩` alongside each compressed vector. The search kernel multiplies the per-candidate score by this scalar before heap insertion, turning the inner-product estimator from downward-biased into unbiased at zero search-time cost and zero extra storage. The recall gain shows up most at low bit widths, where the quantization shrinkage is largest.

Encoding cost: one extra `d`-dimensional dot product per vector to compute `⟨u, x̂⟩`. On 1M vectors at d=1536 this is sub-second of additional encode time — a one-shot price paid at ingest, not at query.

**Search.** Instead of decompressing every database vector, we rotate the query once into the same domain and score directly against the codebook values. The scoring kernel uses SIMD intrinsics (NEON on ARM; AVX-512BW on modern x86, falling back to AVX2, then to a scalar path on pre-AVX2 CPUs) with nibble-split lookup tables for maximum throughput.

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

All x86_64 builds target `x86-64-v2` (SSE4.2 baseline, Nehalem 2008+) via `.cargo/config.toml`, so any x86-64-v2 CPU can run the whole crate. The AVX-512 and AVX2 kernels are `#[target_feature]`-gated and selected at runtime via `is_x86_feature_detected!`, so they kick in on hardware that supports them regardless of the compile baseline; CPUs with neither run the scalar fallback.

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

### Quick harness for optimization work

The suite above is the source of every published number — real embeddings,
FAISS comparator, fixed shapes, run on the two official environments. For the
inner loop of an optimization pass there's also a Rust harness that reproduces
the four mutation metrics (cold bulk add, warm append, single add, remove) on
deterministic synthetic vectors, so a hypothesis can be measured in seconds on
any machine with no dataset and no FAISS:

```bash
cargo run --release --example insert_bench -- --dim 1536 --bits 2
RAYON_NUM_THREADS=1 cargo run --release --example insert_bench
```

It is a screening tool, not a source of published numbers.

`examples/encode_hash` prints a per-stage hash of the encode pipeline for a
fixed input; CI runs it on every OS in the matrix and fails if they disagree,
which is how cross-platform byte identity of the encode is checked.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026) -- the paper this implements
- [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497) (SIGMOD 2024) -- the source of the per-vector length-renormalization correction adapted in step 5
- [FAISS Fast accumulation of PQ and AQ codes](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) -- turbovec's x86 SIMD kernel adapts FastScan's pack layout, nibble-LUT scoring, and u16 accumulator strategy
