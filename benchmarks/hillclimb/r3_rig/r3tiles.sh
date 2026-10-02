#!/bin/bash
# P47: per-tile timing of the nq=100 MT sign scan.
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
cp ~/hc/so/$1.so ~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
IDX=$HOME/.cache/turbovec-hillclimb/cells_200000_2bit.tvim
env ${3:-} TURBOVEC_2BIT_PLANES=1 TURBOVEC_PLANES_PROF=1 RAYON_NUM_THREADS=${2:-8} python -c "
import numpy as np
from turbovec import IdMapIndex
idx=IdMapIndex.load('$IDX')
q=np.random.default_rng(7).random((100,768),dtype=np.float32)
for _ in range(30): idx.search(q,k=10)
" 2>&1 | grep PLANES_PROF | python3 ~/hc/r3tiles.py
