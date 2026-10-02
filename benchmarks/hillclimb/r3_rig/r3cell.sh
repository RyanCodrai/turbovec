#!/bin/bash
# One objective cell, measured exactly as cells_2bit.py does (min of 9 sub-processes, each min of reps), per label; labels cycle ROUNDS times.
# usage: r3cell.sh CELL ROUNDS label... ; label = so-tag[+ENV=VAL...]
source ~/venv/bin/activate
export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
CELL=$1; ROUNDS=$2; shift 2
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
cd ~/hc
for r in $(seq 1 $ROUNDS); do
  for label in "$@"; do
    tag=${label%%+*}; envs=""; [ "$label" != "$tag" ] && envs=$(echo "${label#*+}" | tr '+' ' ')
    cp ~/hc/so/${tag}.so $DEST
    env $envs python - "$CELL" "$label" <<'PY'
import sys, cells_2bit as c
cell, label = sys.argv[1], sys.argv[2]
nq = 1 if cell.startswith("nq1_") else 100; st = cell.endswith("_st")
path = c.index_path(2)
xs = [c.search_cell(path, nq, st, 75 if nq == 1 else 15) for _ in range(9)]
print(f"{label} {cell} min={min(xs):.4f} med={sorted(xs)[4]:.4f}", flush=True)
PY
  done
done
