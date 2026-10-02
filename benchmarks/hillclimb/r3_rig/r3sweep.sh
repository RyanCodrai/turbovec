#!/bin/bash
# Round 3 sweep: baseline vs planes over nq and N, ST and MT (min of reps, one process per point). Informational.
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
A=$1; B=$2; DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
run() { # tag planes n nq th
  cp ~/hc/so/$1.so $DEST
  TURBOVEC_2BIT_PLANES=$2 RAYON_NUM_THREADS=$5 python -c "
import time,numpy as np
from turbovec import IdMapIndex
idx=IdMapIndex.load('$HOME/.cache/turbovec-hillclimb/cells_$3_2bit.tvim')
q=np.random.default_rng(7).random(($4,768),dtype=np.float32)
idx.search(q,k=10); best=1e9
for _ in range(60):
    t0=time.perf_counter(); idx.search(q,k=10); best=min(best,time.perf_counter()-t0)
print(best*1e3)"
}
for th in 1 $(nproc); do
  for pt in "200000 2" "200000 3" "200000 5" "200000 8" "200000 13" "200000 16" "200000 32" "200000 64" "1000 1" "1000 100" "8192 1" "8192 100" "32768 1" "32768 100"; do
    set -- $pt; n=$1; nq=$2
    a=$(run $A 0 $n $nq $th); b=$(run $B 1 $n $nq $th)
    python3 -c "print(f'N=$n nq=$nq th=$th base={$a:.4f} planes={$b:.4f} x{$a/$b:.3f}')"
  done
done
echo SWEEP_DONE
