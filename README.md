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

Fast vector index in Rust with Python bindings. Compresses vectors to 2-4 bits per dimension using [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) with near-optimal distortion.

Unlike trained methods like FAISS PQ, TurboQuant is **data-oblivious** — no training step, no codebook retraining when data changes, and new vectors can be added at any time. This means faster index creation, simpler infrastructure, and comparable or higher recall.

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

## Recall

TurboQuant vs FAISS IndexPQFastScan (100K vectors, k=64). FAISS PQ configurations sized to match TurboQuant compression ratios.

![Recall d=1536](docs/recall_d1536.svg)

![Recall d=3072](docs/recall_d3072.svg)

Both libraries converge to 1.0 by k=4–8. At 4-bit, TurboQuant and FAISS both score 0.955+ at top-1 across every dataset and are within 0.001 of each other. The small differences at 2-bit top-1 (TurboQuant 0.870 vs FAISS 0.882 at d=1536; TurboQuant 0.912 vs FAISS 0.903 at d=3072) reflect how each method behaves at the most aggressive end of the compression curve — they disappear once k ≥ 2. Full results: [d=1536 2-bit](benchmarks/results/recall_d1536_2bit.json), [d=1536 4-bit](benchmarks/results/recall_d1536_4bit.json), [d=3072 2-bit](benchmarks/results/recall_d3072_2bit.json), [d=3072 4-bit](benchmarks/results/recall_d3072_4bit.json), [GloVe 2-bit](benchmarks/results/recall_glove_2bit.json), [GloVe 4-bit](benchmarks/results/recall_glove_4bit.json).

No FAISS FastScan comparison for GloVe d=200 (dimension not compatible with FastScan's m%32 requirement).

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

**Search.** Instead of decompressing every database vector, we rotate the query once into the same domain and score directly against the codebook values. The scoring kernel uses SIMD intrinsics (NEON on ARM, AVX-512BW on modern x86 with an AVX2 fallback) with nibble-split lookup tables for maximum throughput.

The paper proves this achieves distortion within a factor of 2.7x of the information-theoretic lower bound (Shannon's distortion-rate limit). You cannot do much better for a given number of bits.

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
- [FAISS Fast accumulation of PQ and AQ codes](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) -- turbovec's x86 SIMD kernel adapts FastScan's pack layout, nibble-LUT scoring, and u16 accumulator strategy
