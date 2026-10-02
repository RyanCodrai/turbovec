#!/bin/bash
# 4-bit round 2 gate, for a candidate that has won its soak: ids / scores against the exact scan
# on the three corpora. usage: gate.sh TAG  -> ~/hc/r4/TAG_gate.log
TAG=$1; O=~/hc/r4; mkdir -p $O; exec > $O/${TAG}_gate.log 2>&1
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
cp ~/hc/so/$TAG.so ~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so; cd ~/hc
for f in "emb-mpnet768.npy 41000" "openai-1536.npy 200000" "openai-3072.npy 200000"; do set -- $f
  python h1gate.py $1 $2 2>&1 | grep -E "bits=|rror"; done
for d in 1536 3072; do for sw in 0 1; do echo "recall $(TURBOVEC_4BIT_PLANES=$sw python recall4.py $d 2>&1 | tail -1)"; done; done
echo GATE_DONE
