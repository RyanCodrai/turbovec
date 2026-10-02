#!/bin/bash
# Phase profile, min over many searches. usage: r3prof2.sh TAG "S list" "cells(nq:th ...)"
source ~/venv/bin/activate
export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
cp ~/hc/so/$1.so ~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
IDX=$HOME/.cache/turbovec-hillclimb/cells_200000_2bit.tvim
for cell in ${3:-1:1 1:8 100:1 100:8}; do nq=${cell%%:*}; th=${cell##*:}
 for S in ${2:-128}; do
 TURBOVEC_PLANES_MIN=$S TURBOVEC_PLANES_MULT=0 TURBOVEC_2BIT_PLANES=1 TURBOVEC_PLANES_PROF=1 RAYON_NUM_THREADS=$th python -c "
import numpy as np
from turbovec import IdMapIndex
idx=IdMapIndex.load('$IDX')
q=np.random.default_rng(7).random(($nq,768),dtype=np.float32)
for _ in range(${REPS:-150} if $nq==1 else 25): idx.search(q,k=10)
" 2>&1 | grep PLANES_PROF | python3 -c "
import sys,re
def us(s):
    v=float(re.match(r'[0-9.]+',s).group()); return v*(1e3 if s.endswith('ms') and not s.endswith('µs') else 1) if not s.endswith('ns') else v/1e3
rows=[dict(kv.split('=') for kv in l.split()[1:]) for l in sys.stdin]
f=lambda k: min(us(r[k]) for r in rows)
tot=min(sum(us(r[k]) for k in ('prep_all','sign_lut','scan','rerank')) for r in rows)
print(f'$1 nq=$nq th=$th S=$S min-us: prep={f(\"prep_all\"):.0f} sign_lut={f(\"sign_lut\"):.0f} scan={f(\"scan\"):.0f} rerank={f(\"rerank\"):.0f} best-total={tot:.0f}')"
 done
done
