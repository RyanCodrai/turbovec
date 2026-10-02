#!/bin/bash
# 4-bit round 2 smoke (< 3 min): one balanced pass (base, cand, cand, base) of the 16
# real-embedding cells at 3 reps. A screen, not a verdict: identical builds read within
# about +-4% a cell here, so it passes a candidate to the soak or stops an obvious loss.
# usage: smoke.sh ARCH BASE CAND   (a label is TAG or TAG:planes; TAG.so lives in ~/hc/so)
ARCH=$1; A=$2; B=$3; O=~/hc/r4; mkdir -p $O
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so; cd ~/hc
n=0; rm -f $O/smoke_*.json
for pos in a b b a; do n=$((n+1)); label=$A; [ $pos = b ] && label=$B; tag=${label%%:*}; sw=0; [ "$label" != "$tag" ] && sw=1
  cp so/$tag.so $DEST
  CELLS_REPS=3 TURBOVEC_4BIT_PLANES=$sw python cells_real.py --bits 4 --out $O/smoke_${pos}_$n.json >/dev/null 2>$O/smoke_$n.err || echo "SMOKE $label FAILED"
done
python score.py $ARCH=$(ls $O/smoke_a_*.json | paste -sd, -):$(ls $O/smoke_b_*.json | paste -sd, -)
