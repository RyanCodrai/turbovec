#!/bin/bash
# usage: h1.sh ARCH TAG   (patch ~/hc/TAG.patch against ccab9f32) -> ~/hc/r4/TAG.log
ARCH=$1; TAG=$2; O=~/hc/r4; mkdir -p $O; exec > $O/$TAG.log 2>&1
cd ~/hc; bash build_so2.sh $TAG ~/hc/$TAG.patch; tail -1 ~/hc_build_so2.log
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
for f in "openai-1536.npy 200000" "emb-mpnet768.npy 41000" "openai-3072.npy 200000"; do set -- $f; [ -f ~/data/py-turboquant/$1 ] || { echo "missing $1"; continue; }
  python h1gate.py $1 $2 2>&1 | grep -E "bits=|Error|error" ; done
n=0; for p in 1 2; do for sw in 0 1 1 0; do n=$((n+1)); TURBOVEC_4BIT_PLANES=$sw python cells_real.py --bits 4 --out $O/${TAG}_sw${sw}_$n.json >/dev/null 2>$O/${TAG}_$n.err || echo "RUN $n FAILED"; done; done
python score.py $ARCH=$(ls $O/${TAG}_sw0_*.json | paste -sd, -):$(ls $O/${TAG}_sw1_*.json | paste -sd, -)
echo ${TAG}_DONE
