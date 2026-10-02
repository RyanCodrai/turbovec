#!/bin/bash
# Final PR measurements on this box. usage: prfinal.sh ARCH PATCH
ARCH=$1; PATCHF=$2
O=~/hc/prfinal; mkdir -p $O; exec > $O/run.log 2>&1
cd ~/hc
bash build_so2.sh main none; bash build_so2.sh pr $PATCHF; tail -1 ~/hc_build_so2.log
source ~/venv/bin/activate
export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
use() { tag=${1%%:*}; cp ~/hc/so/$tag.so $DEST; export TURBOVEC_2BIT_PLANES=0; [ "$1" != "$tag" ] && export TURBOVEC_2BIT_PLANES=1; }
echo "== cells 2-bit $(date -Is)"; n=0
for p in 1 2 3; do for label in main pr pr:planes pr:planes pr main; do n=$((n+1)); use $label
  python cells_2bit.py --bits 2 --out $O/c2_${label/:/_}_$n.json >/dev/null 2>$O/c2_$n.err || echo "RUN $n FAILED"; echo "c2 $n $label $(date -Is)"; done; done
echo "== cells 4-bit"; n=0
for label in main pr pr main; do n=$((n+1)); use $label
  python cells_2bit.py --bits 4 --out $O/c4_${label}_$n.json >/dev/null 2>$O/c4_$n.err || echo "RUN4 $n FAILED"; echo "c4 $n $label $(date -Is)"; done
echo "== official speed"
cd ~/turbovec/benchmarks/suite
for rep in 1 2; do for s in d1536_2bit d3072_2bit; do for th in st mt; do for label in main pr pr:planes; do use $label
  python speed_${s}_${ARCH}_${th}.py 2>$O/speed.err | tr -d '\n ' > $O/speed_${s}_${th}_${label/:/_}_$rep.json; echo "speed $s $th $label $rep $(cat $O/speed_${s}_${th}_${label/:/_}_$rep.json)"; done; done; done; done
for s in d1536_4bit; do for th in st mt; do for label in main pr; do use $label
  python speed_${s}_${ARCH}_${th}.py 2>$O/speed.err | tr -d '\n ' > $O/speed_${s}_${th}_${label}_1.json; echo "speed $s $th $label $(cat $O/speed_${s}_${th}_${label}_1.json)"; done; done; done
git -C ~/turbovec checkout -q -- benchmarks/results
cd ~/hc
echo "== recall"
for d in 1536 3072; do for label in main pr pr:planes; do use $label; echo "recall $label $(python prrecall.py $d 2>&1 | tail -1)"; done; done
echo "== rss"
for label in main pr pr:planes; do use $label; echo "rss $label $(python prrss.py 2>&1 | tail -1)"; done
echo "== gate"
use pr; unset TURBOVEC_2BIT_PLANES
for f in openai-1536.npy openai-3072.npy emb-mpnet768.npy; do [ -f ~/data/py-turboquant/$f ] || continue
  nn=200000; [ $f = emb-mpnet768.npy ] && nn=41000
  NSINGLE=5000 GATE_MODES=exact,p128 python r3gate.py $f $nn 10000 2>&1 | grep -E 'p128|GATE_DONE|Error'; done
echo "== parity"; use pr; unset TURBOVEC_2BIT_PLANES; python parity_2bit.py 2>&1 | tail -2; use main; unset TURBOVEC_2BIT_PLANES; python parity_2bit.py 2>&1 | tail -2
echo "FINAL_DONE $(date -Is)"
