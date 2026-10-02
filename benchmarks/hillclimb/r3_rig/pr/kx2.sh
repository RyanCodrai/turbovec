#!/bin/bash
exec > ~/hc/prfinal/kx2.log 2>&1
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
cp ~/hc/so/h118.so ~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so; cd ~/hc
sed -i 's/^for mult, tmult in .*/for mult, tmult in ((128, 30), (128, 20), (128, 15), (128, 10)):/' kx.py
for f in "emb-mpnet768.npy 41000" "openai-1536.npy 200000" "openai-3072.npy 200000"; do RAYON_NUM_THREADS=8 python kx.py $f; done
echo "== phase profile, k=64 and k=100, by rescore size"
for tm in 30 20 15 10; do for k in 64 100; do
TM=$tm K=$k python - <<'PY' 2>&1 | grep -E "PLANES_PROF" | sed -E 's/ ranges=.*//; s/prep_all.*sign_lut/sign_lut/' | tail -3 | sed "s/^/T=$tm k=$k /"
import os, numpy as np
os.environ["TURBOVEC_2BIT_PLANES"] = "1"; os.environ["TURBOVEC_PLANES_PROF"] = "1"; os.environ["TURBOVEC_PLANES_T_MULT"] = os.environ["TM"]
from turbovec import TurboQuantIndex
k = int(os.environ["K"])
v = np.load(os.path.expanduser("~/data/py-turboquant/openai-1536.npy"), mmap_mode="r")
db = np.ascontiguousarray(v[:200000]).astype(np.float32); q = np.ascontiguousarray(v[-1000:]).astype(np.float32)
ix = TurboQuantIndex(1536, bit_width=2); ix.add(db)
for _ in range(3): ix.search(q, k=k)
for _ in range(3): ix.search(q[:1], k=k)
PY
done; done
echo KX2_DONE
