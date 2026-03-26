# py-turboquant

Python implementation of TurboQuant for vector search.

Compresses high-dimensional vectors to 2-4 bits per coordinate with near-optimal distortion. Data-oblivious (no training), zero indexing time.

Based on: Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## Usage

```python
from turboquant import TurboQuantIndex

index = TurboQuantIndex.from_vectors(vectors, bit_width=3)
scores, indices = index.search(query, k=10)

index.save("my_index.tq")
loaded = TurboQuantIndex.from_bin("my_index.tq")
```

## How it works

1. Strip the norm from each vector (stored as 1 float per vector)
2. Rotate with a random orthogonal matrix so each coordinate follows a known distribution
3. Quantize each coordinate to a small integer using a precomputed Lloyd-Max codebook
4. Bit-pack the integers for storage

Search rotates the query into the same domain and scores directly against the packed codes. No decompression needed.

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

| Bit width | Index size | Compression vs FP32 | Search latency |
|:----------|:-----------|:--------------------|:---------------|
| 2-bit     | 5.1 MB     | 14.8x               | 0.9ms/query    |
| 4-bit     | 9.9 MB     | 7.7x                | 0.9ms/query    |

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

| Bit width | Index size | Compression vs FP32 | Search latency |
|:----------|:-----------|:--------------------|:---------------|
| 2-bit     | 37.0 MB    | 15.8x               | 2.5ms/query    |
| 4-bit     | 73.6 MB    | 8.0x                | 2.5ms/query    |

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

## Dependencies

- numpy
- scipy (for codebook generation only)
- h5py (for GloVe benchmark)
- datasets (for OpenAI benchmark download)

## Citation

```bibtex
@inproceedings{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://arxiv.org/abs/2504.19874}
}
```
