#!/bin/bash
# P46: per-range (start, duration) and phase markers of the nq=1 sign scan; two fastest searches and the median one.
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
cp ~/hc/so/$1.so ~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
IDX=$HOME/.cache/turbovec-hillclimb/cells_200000_2bit.tvim
TURBOVEC_2BIT_PLANES=1 TURBOVEC_PLANES_PROF=1 RAYON_NUM_THREADS=${2:-8} python -c "
import numpy as np
from turbovec import IdMapIndex
idx=IdMapIndex.load('$IDX')
q=np.random.default_rng(7).random((1,768),dtype=np.float32)
for _ in range(300): idx.search(q,k=10)
" 2>&1 | grep PLANES_PROF | python3 ~/hc/r3ranges.py
