# py-turboquant

Python implementation of TurboQuant for vector search.

Compresses high-dimensional vectors to 2-4 bits per coordinate with near-optimal distortion. Data-oblivious (no training), zero indexing time.

Unofficial implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

## Usage

```python
from turboquant import TurboQuantIndex

index = TurboQuantIndex.from_vectors(vectors, bit_width=3)
index.add_vectors(more_vectors)

scores, indices = index.search(query, k=10)

index.save("my_index.tq")
loaded = TurboQuantIndex.from_bin("my_index.tq")
```

## How it works

Each vector is a direction on a high-dimensional hypersphere. TurboQuant compresses these directions using a simple insight: after applying a random rotation, every coordinate follows a known distribution — regardless of the input data.

**1. Normalize.** Strip the length (norm) from each vector and store it as a single float. Now every vector is a unit direction on the hypersphere.

**2. Random rotation.** Multiply all vectors by the same random orthogonal matrix. After rotation, each coordinate independently follows a Beta distribution that converges to Gaussian N(0, 1/d) in high dimensions. This holds for any input data — the rotation makes the coordinate distribution predictable.

**3. Lloyd-Max scalar quantization.** Since the distribution is known, we can precompute the optimal way to bucket each coordinate. For 2-bit, that's 4 buckets; for 4-bit, 16 buckets. The [Lloyd-Max algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) finds bucket boundaries and centroids that minimize mean squared error. These are computed once from the math, not from the data.

**4. Bit-pack.** Each coordinate is now a small integer (0-3 for 2-bit, 0-15 for 4-bit). Pack these tightly into bytes. A 1536-dim vector goes from 6,144 bytes (FP32) to 384 bytes (2-bit). That's 16x compression.

**Search.** Instead of decompressing every database vector, we rotate the query once into the same domain and score directly against the codebook values. Since the rotation is orthogonal: `q . R_inverse(centroids[codes]) = R(q) . centroids[codes]`. One rotation of the query replaces 100K inverse rotations of database vectors.

The paper proves this achieves distortion within a factor of 2.7x of the information-theoretic lower bound (Shannon's distortion-rate limit). You cannot do much better for a given number of bits.

**Online by design.** Because the codebook and rotation are derived from math (not from the data), new vectors can be added at any time without rebuilding the index. Each vector is encoded independently in ~4ms at d=1536. Traditional methods like Product Quantization require expensive offline codebook training that must be re-run when data changes.

## Benchmark results

Reproducing Section 4.4 of the paper. recall@1@k = probability that the true nearest neighbor appears in the top-k results. Benchmarked on Apple M3 Max.

### GloVe d=200 (100K database vectors, 10K queries)

| k    | 2-bit recall@1@k | 4-bit recall@1@k |
|:-----|:-----------------|:-----------------|
| 1    | 0.511            | 0.826            |
| 2    | 0.666            | 0.943            |
| 4    | 0.791            | 0.988            |
| 8    | 0.887            | 0.998            |
| 16   | 0.947            | 1.000            |
| 32   | 0.977            | 1.000            |
| 64   | 0.991            | 1.000            |

| Bit width | Index size | Compression vs FP32 | Index time | Search latency |
|:----------|:-----------|:--------------------|:-----------|:---------------|
| 2-bit     | 5.1 MB     | 14.8x               | 454ms      | 0.9ms/query    |
| 4-bit     | 9.9 MB     | 7.7x                | 1,141ms    | 0.9ms/query    |

### OpenAI DBpedia d=1536 (100K database vectors, 1K queries)

| k    | 2-bit recall@1@k | 4-bit recall@1@k |
|:-----|:-----------------|:-----------------|
| 1    | 0.862            | 0.967            |
| 2    | 0.967            | 0.995            |
| 4    | 0.995            | 1.000            |
| 8    | 0.999            | 1.000            |
| 16   | 1.000            | 1.000            |
| 32   | 1.000            | 1.000            |
| 64   | 1.000            | 1.000            |

| Bit width | Index size | Compression vs FP32 | Index time | Search latency |
|:----------|:-----------|:--------------------|:-----------|:---------------|
| 2-bit     | 37.0 MB    | 15.8x               | 3,249ms    | 2.2ms/query    |
| 4-bit     | 73.6 MB    | 8.0x                | 4,986ms    | 2.5ms/query    |

### OpenAI DBpedia d=3072 (100K database vectors, 1K queries)

| k    | 2-bit recall@1@k | 4-bit recall@1@k |
|:-----|:-----------------|:-----------------|
| 1    | 0.907            | 0.970            |
| 2    | 0.980            | 0.999            |
| 4    | 0.998            | 1.000            |
| 8    | 1.000            | 1.000            |
| 16   | 1.000            | 1.000            |
| 32   | 1.000            | 1.000            |
| 64   | 1.000            | 1.000            |

| Bit width | Index size | Compression vs FP32 | Index time | Search latency |
|:----------|:-----------|:--------------------|:-----------|:---------------|
| 2-bit     | 73.6 MB    | 15.9x               | 8,350ms    | 5.5ms/query    |
| 4-bit     | 146.9 MB   | 8.0x                | 12,260ms   | 5.1ms/query    |

## Running benchmarks

Download datasets:
```
python3 benchmark.py download glove
python3 benchmark.py download openai-1536
python3 benchmark.py download openai-3072
```

Run benchmarks:
```
python3 benchmark.py glove
python3 benchmark.py openai-1536
python3 benchmark.py openai-3072
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026) — the paper this implements
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482) — the 1-bit residual correction technique
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) — related approach using polar coordinates
- [turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) — PyTorch implementation focused on KV cache compression
