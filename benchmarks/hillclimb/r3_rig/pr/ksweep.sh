#!/bin/bash
exec > ~/hc/prfinal/ksweep.log 2>&1
while ! grep -q FINAL_DONE ~/hc/prfinal/run.log; do sleep 20; done
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
cp ~/hc/so/pr.so ~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so; cd ~/hc
for d in 1536 3072; do for th in 1 8; do for p in 0 1 1 0; do RAYON_NUM_THREADS=$th TURBOVEC_2BIT_PLANES=$p python ksweep.py $d 2>&1 | tail -1; done; done; done
echo KSWEEP_DONE
