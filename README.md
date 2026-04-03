# turbovec

Fast vector quantization in Rust with Python bindings. Compresses vectors to 2-4 bits per dimension with near-optimal distortion. Implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

Unlike trained methods like FAISS PQ, TurboQuant is **data-oblivious** — no training step, no codebook retraining when data changes, and new vectors can be added at any time. This means faster index creation, simpler infrastructure, and comparable or higher recall.

## Python

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

![Recall](docs/recall.png)

Both converge to 1.0 by k=4-8. At d=3072 2-bit, TurboQuant recall exceeds FAISS (0.912 vs 0.903). At d=1536 2-bit, FAISS is slightly ahead (0.882 vs 0.870). The recall discrepancy between TQ and FAISS varies by dimension and bit width — this requires further investigation. Full results: [d=1536 2-bit](benchmarks/results/recall_d1536_2bit.json), [d=1536 4-bit](benchmarks/results/recall_d1536_4bit.json), [d=3072 2-bit](benchmarks/results/recall_d3072_2bit.json), [d=3072 4-bit](benchmarks/results/recall_d3072_4bit.json), [GloVe 2-bit](benchmarks/results/recall_glove_2bit.json), [GloVe 4-bit](benchmarks/results/recall_glove_4bit.json).

No FAISS FastScan comparison for GloVe d=200 (dimension not compatible with FastScan's m%32 requirement).

## Compression

![Compression](docs/compression.png)

## Search Speed

All benchmarks: 100K vectors, 1K queries, k=64, median of 5 runs.

### ARM (Apple M3 Max)

![ARM Speed](docs/arm_speed.png)

On ARM, TurboQuant is within 2-25% of FAISS. Optimization is ongoing.

### x86 (Intel Sapphire Rapids, 4 vCPUs)

![x86 Speed](docs/x86_speed.png)

On x86, TurboQuant is 1.4-3.7x behind FAISS. Optimization is ongoing.

## How it works

Each vector is a direction on a high-dimensional hypersphere. TurboQuant compresses these directions using a simple insight: after applying a random rotation, every coordinate follows a known distribution -- regardless of the input data.

**1. Normalize.** Strip the length (norm) from each vector and store it as a single float. Now every vector is a unit direction on the hypersphere.

**2. Random rotation.** Multiply all vectors by the same random orthogonal matrix. After rotation, each coordinate independently follows a Beta distribution that converges to Gaussian N(0, 1/d) in high dimensions. This holds for any input data -- the rotation makes the coordinate distribution predictable.

**3. Lloyd-Max scalar quantization.** Since the distribution is known, we can precompute the optimal way to bucket each coordinate. For 2-bit, that's 4 buckets; for 4-bit, 16 buckets. The [Lloyd-Max algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) finds bucket boundaries and centroids that minimize mean squared error. These are computed once from the math, not from the data.

**4. Bit-pack.** Each coordinate is now a small integer (0-3 for 2-bit, 0-15 for 4-bit). Pack these tightly into bytes. A 1536-dim vector goes from 6,144 bytes (FP32) to 384 bytes (2-bit). That's 16x compression.

**Search.** Instead of decompressing every database vector, we rotate the query once into the same domain and score directly against the codebook values. The scoring kernel uses SIMD intrinsics (NEON on ARM, AVX2 on x86) with nibble-split lookup tables for maximum throughput.

The paper proves this achieves distortion within a factor of 2.7x of the information-theoretic lower bound (Shannon's distortion-rate limit). You cannot do much better for a given number of bits.

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RyanCodrai/turbovec&type=Date)](https://star-history.com/#RyanCodrai/turbovec&Date)

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026) -- the paper this implements
- [FAISS Fast accumulation of PQ and AQ codes](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) -- the FAISS FastScan approach our x86 kernel is based on
