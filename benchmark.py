"""
Benchmark py-turboquant against the paper's datasets.

Reproduces Section 4.4 of TurboQuant (arXiv:2504.19874):
  - GloVe d=200: 100K vectors, 10K queries
  - OpenAI DBpedia d=1536: 100K vectors, 1K queries
  - OpenAI DBpedia d=3072: 100K vectors, 1K queries

Usage:
  python3 benchmark.py glove
  python3 benchmark.py openai-1536
  python3 benchmark.py openai-3072
  python3 benchmark.py glove openai-1536 openai-3072
"""

import os
import sys
import time

import h5py
import numpy as np

from turboquant import TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")

GLOVE_PATH = os.path.join(DATA_DIR, "glove-200-angular.hdf5")
GLOVE_URL = "http://ann-benchmarks.com/glove-200-angular.hdf5"


def download_glove():
    if os.path.exists(GLOVE_PATH):
        print(f"  Already downloaded: {GLOVE_PATH}")
        return GLOVE_PATH

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"  Downloading {GLOVE_URL}...")
    import subprocess
    subprocess.run(["curl", "-L", "-o", GLOVE_PATH, GLOVE_URL], check=True)
    print(f"  Saved: {GLOVE_PATH} ({os.path.getsize(GLOVE_PATH) / 1024 / 1024:.0f} MB)")
    return GLOVE_PATH


def load_glove(seed=42):
    if not os.path.exists(GLOVE_PATH):
        download_glove()
    f = h5py.File(GLOVE_PATH, "r")
    all_train = f["train"][:].astype(np.float32)
    queries = f["test"][:].astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_train), 100_000, replace=False)
    database = all_train[idx]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database, queries


def download_openai(dim=1536):
    from datasets import load_dataset

    path = os.path.join(DATA_DIR, f"openai-{dim}.npy")
    if os.path.exists(path):
        print(f"  Already downloaded: {path}")
        return path

    os.makedirs(DATA_DIR, exist_ok=True)
    name = f"Qdrant/dbpedia-entities-openai3-text-embedding-3-large-{dim}-1M"
    col = f"text-embedding-3-large-{dim}-embedding"
    print(f"  Downloading {name}...")
    ds = load_dataset(name, split="train")
    ds.set_format("numpy")
    vecs = ds[col].astype(np.float32)
    np.save(path, vecs)
    print(f"  Saved: {path} ({os.path.getsize(path) / 1024 / 1024:.0f} MB)")
    return path


def load_openai(dim=1536, seed=42):
    path = os.path.join(DATA_DIR, f"openai-{dim}.npy")
    if not os.path.exists(path):
        download_openai(dim)

    all_vecs = np.load(path)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]]
    queries = all_vecs[idx[100_000:101_000]]
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database, queries


def recall_at_1_at_k(true_top1, predicted_indices, k):
    return np.mean([true_top1[i] in predicted_indices[i, :k] for i in range(len(true_top1))])


def run_benchmark(database, queries, bit_widths, label=""):
    n, dim = database.shape
    n_queries = len(queries)

    # Ground truth
    true_top1 = np.argmax(queries @ database.T, axis=1)

    # Encode + search each bit width
    results = {}
    for bw in bit_widths:
        t0 = time.time()
        index = TurboQuantIndex.from_vectors(database, bit_width=bw)
        encode_time = time.time() - t0

        index.save("/tmp/bench.tq")
        file_size = os.path.getsize("/tmp/bench.tq")
        loaded = TurboQuantIndex.from_bin("/tmp/bench.tq")

        t0 = time.time()
        _, all_indices = loaded.search(queries, k=64)
        search_time = time.time() - t0

        recalls = {}
        for k in [1, 2, 4, 8, 16, 32, 64]:
            recalls[k] = recall_at_1_at_k(true_top1, all_indices, k)

        results[bw] = {
            "encode_time": encode_time,
            "search_time": search_time,
            "file_size": file_size,
            "recalls": recalls,
        }

    # Print summary table
    original_mb = n * dim * 4 / 1024 / 1024
    bw_labels = [f"{bw}-bit" for bw in bit_widths]
    header = f"  {'k':>4}  " + "  ".join(f"{l:>10}" for l in bw_labels)
    print(header)
    print(f"  {'─' * 4}  " + "  ".join("─" * 10 for _ in bit_widths))
    for k in [1, 2, 4, 8, 16, 32, 64]:
        row = f"  {k:>4}  " + "  ".join(f"{results[bw]['recalls'][k]:>10.4f}" for bw in bit_widths)
        print(row)

    print()
    for bw in bit_widths:
        r = results[bw]
        mb = r["file_size"] / 1024 / 1024
        print(f"  {bw}-bit: {mb:.1f} MB ({original_mb / mb:.1f}x compression), "
              f"encode {r['encode_time']:.2f}s, search {r['search_time']:.1f}s "
              f"({r['search_time'] / n_queries * 1000:.1f}ms/query)")


DATASETS = {
    "glove": ("GloVe d=200 (100K vectors, 10K queries)", load_glove),
    "openai-1536": ("OpenAI DBpedia d=1536 (100K vectors, 1K queries)", lambda: load_openai(1536)),
    "openai-3072": ("OpenAI DBpedia d=3072 (100K vectors, 1K queries)", lambda: load_openai(3072)),
}

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["glove"]

    # Handle download command
    if args[0] == "download":
        targets = args[1:] if len(args) > 1 else ["glove", "openai-1536", "openai-3072"]
        for t in targets:
            if t == "glove":
                download_glove()
            elif t == "openai-1536":
                download_openai(1536)
            elif t == "openai-3072":
                download_openai(3072)
            else:
                print(f"Unknown download target: {t}")
        sys.exit(0)

    print("py-turboquant benchmark")
    print("=" * 60)

    for name in args:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            continue

        label, loader = DATASETS[name]
        print(f"\nDataset: {label}")
        database, queries = loader()
        n, dim = database.shape
        print(f"Database: {n:,} x {dim}, Queries: {len(queries):,}")
        print(f"Original size: {n * dim * 4 / 1024 / 1024:.1f} MB (FP32)")

        run_benchmark(database, queries, [2, 4], name)

    print("\nDone.")
