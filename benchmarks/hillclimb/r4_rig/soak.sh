#!/bin/bash
# 4-bit round 2 soak (< 15 min): PASSES balanced ABBA passes of the 16 cells at 5 reps.
# usage: soak.sh ARCH BASE CAND [PASSES=2] [NAME]   -> ~/hc/r4/NAME_soak.log, scored
ARCH=$1; A=$2; B=$3; PASSES=${4:-2}; NAME=${5:-${B%%:*}}; O=~/hc/r4; mkdir -p $O
exec > $O/${NAME}_soak.log 2>&1
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so; cd ~/hc; rm -f $O/${NAME}_soak_*.json; n=0
for p in $(seq 1 $PASSES); do for pos in a b b a; do n=$((n+1)); label=$A; side=base; [ $pos = b ] && { label=$B; side=cand; }
  tag=${label%%:*}; sw=0; [ "$label" != "$tag" ] && sw=1; cp so/$tag.so $DEST
  TURBOVEC_4BIT_PLANES=$sw python cells_real.py --bits 4 --out $O/${NAME}_soak_${side}_$n.json >/dev/null 2>$O/${NAME}_soak_$n.err || echo "RUN $n $label FAILED"
done; done
python score.py $ARCH=$(ls $O/${NAME}_soak_base_*.json | paste -sd, -):$(ls $O/${NAME}_soak_cand_*.json | paste -sd, -)
echo SOAK_DONE
