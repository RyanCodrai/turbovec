#!/bin/bash
# Round 3 smoke: labels are so-tag[:planes]; ABBA over "$A $B $B $A". <3 min.
set -uo pipefail
source ~/venv/bin/activate
ARCH=$1; A=$2; B=$3; CELLS=${4:-"nq1_st nq1_mt nq100_st nq100_mt"}
if [ "$ARCH" = x86 ]; then export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libopenblas.so.0; else export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libopenblas.so.0; fi
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
IDX=$HOME/.cache/turbovec-hillclimb/cells_200000_2bit.tvim
cd ~/hc
for label in $A $B $B $A; do
  tag=${label%%:*}; planes=0; [ "$label" != "$tag" ] && planes=1
  cp ~/hc/so/${tag}.so $DEST
  for cell in $CELLS; do
    case $cell in
      nq1_st)   nq=1;   th=1;;
      nq1_mt)   nq=1;   th=$(nproc);;
      nq100_st) nq=100; th=1;;
      *)        nq=100; th=$(nproc);;
    esac
    ms=$(TURBOVEC_2BIT_PLANES=$planes RAYON_NUM_THREADS=$th python -c "
import time,numpy as np
from turbovec import IdMapIndex
idx=IdMapIndex.load('$IDX')
q=np.random.default_rng(7).random(($nq,768),dtype=np.float32)
idx.search(q,k=10); best=1e9
for _ in range(30 if $nq==100 else 200):
    t0=time.perf_counter(); idx.search(q,k=10); best=min(best,time.perf_counter()-t0)
print(round(best*1e3,3))" 2>&1 | tail -1)
    echo "$label $cell $ms"
  done
done
echo SMOKE_DONE
