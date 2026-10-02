#!/bin/bash
exec > ~/hc/prfinal/kx.log 2>&1
while ! grep -q PR6_DONE ~/hc/prfinal/pr6.log; do sleep 20; done
grep -c "test result: ok" ~/hc/prcheck.log; grep -E "FAILED|panicked|clippy rc|^error" ~/hc/prcheck.log
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
cp ~/hc/so/h118.so ~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so; cd ~/hc
for th in 1 8; do RAYON_NUM_THREADS=$th python kx.py openai-1536.npy 200000; done
RAYON_NUM_THREADS=8 python kx.py emb-mpnet768.npy 41000
RAYON_NUM_THREADS=8 python kx.py openai-3072.npy 200000
echo "== phase profile k=64"
python - <<'PY' 2>&1 | grep -E "PLANES_PROF" | cut -c1-200 | tail -4
import os, numpy as np
os.environ["TURBOVEC_2BIT_PLANES"] = "1"; os.environ["TURBOVEC_PLANES_PROF"] = "1"
from turbovec import TurboQuantIndex
v = np.load(os.path.expanduser("~/data/py-turboquant/openai-1536.npy"), mmap_mode="r")
db = np.ascontiguousarray(v[:200000]).astype(np.float32); q = np.ascontiguousarray(v[-1000:]).astype(np.float32)
ix = TurboQuantIndex(1536, bit_width=2); ix.add(db)
for _ in range(2): ix.search(q, k=64)
for _ in range(2): ix.search(q[:1], k=64)
PY
echo KX_DONE
