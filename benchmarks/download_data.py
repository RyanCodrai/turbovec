"""
Download benchmark datasets.

Usage:
  python3 download_data.py glove openai-1536 openai-3072
  python3 download_data.py all
"""

import os
import subprocess
import sys

import numpy as np

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
GLOVE_PATH = os.path.join(DATA_DIR, "glove-200-angular.hdf5")
GLOVE_URL = "http://ann-benchmarks.com/glove-200-angular.hdf5"


def download_glove():
    if os.path.exists(GLOVE_PATH):
        print(f"Already downloaded: {GLOVE_PATH}")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {GLOVE_URL}...")
    # Download to a temp path, then rename into place on success — an
    # interrupted run leaves only a .tmp the existence guard ignores,
    # never a partial file at the final path.
    tmp = GLOVE_PATH + ".tmp"
    subprocess.run(["curl", "-L", "-o", tmp, GLOVE_URL], check=True)
    os.replace(tmp, GLOVE_PATH)
    print(f"Saved: {GLOVE_PATH} ({os.path.getsize(GLOVE_PATH) / 1024 / 1024:.0f} MB)")


def download_openai(dim):
    from datasets import load_dataset

    path = os.path.join(DATA_DIR, f"openai-{dim}.npy")
    if os.path.exists(path):
        print(f"Already downloaded: {path}")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    name = f"Qdrant/dbpedia-entities-openai3-text-embedding-3-large-{dim}-1M"
    col = f"text-embedding-3-large-{dim}-embedding"
    print(f"Downloading {name}...")
    ds = load_dataset(name, split="train")
    ds.set_format("numpy")
    vecs = ds[col].astype(np.float32)
    # Save to a temp path, then rename into place on success (same
    # rationale as download_glove). Pass an open file object so np.save
    # can't append a `.npy` suffix to the temp name.
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, vecs)
    os.replace(tmp, path)
    print(f"Saved: {path} ({os.path.getsize(path) / 1024 / 1024:.0f} MB)")


TARGETS = {
    "glove": download_glove,
    "openai-1536": lambda: download_openai(1536),
    "openai-3072": lambda: download_openai(3072),
}

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in args:
        args = list(TARGETS.keys())

    for name in args:
        if name not in TARGETS:
            print(f"Unknown dataset: {name}")
            print(f"Available: {', '.join(TARGETS.keys())}")
            continue
        TARGETS[name]()
