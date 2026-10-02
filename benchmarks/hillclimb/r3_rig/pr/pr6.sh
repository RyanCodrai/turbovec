#!/bin/bash
ARCH=$1; O=~/hc/prfinal; exec > $O/pr6.log 2>&1
cd ~/hc; bash build_so2.sh pr6 ~/hc/pr6.patch; tail -1 ~/hc_build_so2.log
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
for f in openai-1536.npy openai-3072.npy emb-mpnet768.npy; do nn=200000; [ $f = emb-mpnet768.npy ] && nn=41000
  NSINGLE=5000 GATE_MODES=exact,p128 python r3gate.py $f $nn 10000 2>&1 | grep -E 'p128|Error'; done
for d in 1536 3072; do for th in 1 8; do echo "pr6 $(RAYON_NUM_THREADS=$th TURBOVEC_2BIT_PLANES=1 python ksweep.py $d 2>&1 | tail -1)"; done; done
for rep in 1 2; do for s in d1536_2bit d3072_2bit; do for th in st mt; do echo "speed6 $s $th $rep $(cd ~/turbovec/benchmarks/suite && TURBOVEC_2BIT_PLANES=1 python speed_${s}_${ARCH}_${th}.py 2>/dev/null | tr -d '\n ')"; done; done; done
git -C ~/turbovec checkout -q -- benchmarks/results
for p in 0 1; do TURBOVEC_2BIT_PLANES=$p python rnd.py | sed "s/^/random /"; TURBOVEC_2BIT_PLANES=$p python rnd2.py; done
( unset LD_PRELOAD; bash ~/hc/prcheck.sh ~/hc/pr6.patch; grep -c "test result: ok" ~/hc/prcheck.log; grep -E "FAILED|panicked|clippy rc|^error" ~/hc/prcheck.log )
exec >> $O/pr6.log 2>&1
echo PR6_DONE
