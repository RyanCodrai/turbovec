#!/bin/bash
exec > ~/hc/prfinal/post.log 2>&1
while ! grep -q KSWEEP_DONE ~/hc/prfinal/ksweep.log; do sleep 20; done
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so; cd ~/hc
for tag in main pr; do cp so/$tag.so $DEST; for p in 0 1; do [ $tag = main ] && [ $p = 1 ] && continue; echo "rss $tag $(TURBOVEC_2BIT_PLANES=$p python prrss.py 2>&1 | tail -1)"; done; done
cp so/h118.so $DEST
for th in 1 8; do for m in 128 192 256 384; do echo "floor=$m $(RAYON_NUM_THREADS=$th TURBOVEC_2BIT_PLANES=1 TURBOVEC_PLANES_MIN=$m python ksweep.py 1536 2>&1 | tail -1)"; done; done
cp so/pr.so $DEST
echo POST_DONE
