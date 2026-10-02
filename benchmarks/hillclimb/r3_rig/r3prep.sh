#!/bin/bash
# P48: phase lines for nq=1 ST and nq=100 ST (fastest search of each).
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
cp ~/hc/so/$1.so ~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
IDX=$HOME/.cache/turbovec-hillclimb/cells_200000_2bit.tvim
for cfg in "1 1 200" "100 1 20" "100 8 30" "1 8 200"; do set -- $cfg $1; 
TURBOVEC_2BIT_PLANES=1 TURBOVEC_PLANES_PROF=1 RAYON_NUM_THREADS=$2 python -c "
import numpy as np
from turbovec import IdMapIndex
idx=IdMapIndex.load('$IDX')
q=np.random.default_rng(7).random(($1,768),dtype=np.float32)
for _ in range($3): idx.search(q,k=10)
" 2>&1 | grep PLANES_PROF | sed 's/ ranges=.*//' | sort -t= -k4 | awk -v c="nq=$1 th=$2" 'NR>3{a[NR]=$0} END{}{l[NR]=$0} END{print c": "l[int(NR/2)]}'
done
