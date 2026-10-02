#!/bin/bash
# usage: measure_build.sh ARCH TAG PREV  -> builds TAG from ~/hc/TAG.patch; gate, k sweep, cells vs PREV (planes on), official speed, full check
ARCH=$1; TAG=$2; PREV=$3; O=~/hc/prfinal; exec > $O/$TAG.log 2>&1
cd ~/hc; bash build_so2.sh $TAG ~/hc/$TAG.patch; tail -1 ~/hc_build_so2.log
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
for f in openai-1536.npy openai-3072.npy emb-mpnet768.npy; do nn=200000; [ $f = emb-mpnet768.npy ] && nn=41000
  NSINGLE=5000 GATE_MODES=exact,p128 python r3gate.py $f $nn 10000 2>&1 | grep -E 'p128|Error'; done
for d in 1536 3072; do for th in 1 8; do for p in 1 0; do echo "$TAG $(RAYON_NUM_THREADS=$th TURBOVEC_2BIT_PLANES=$p python ksweep.py $d 2>&1 | tail -1)"; done; done; done
n=0; for p in 1 2; do for tag in $PREV $TAG $TAG $PREV; do n=$((n+1)); cp so/$tag.so $DEST
  TURBOVEC_2BIT_PLANES=1 python cells_2bit.py --bits 2 --out $O/e_${TAG}_${tag}_$n.json >/dev/null 2>$O/e_$n.err || echo "RUN $n FAILED"; done; done
python - <<PY
import json, glob
for tag in ("$PREV", "$TAG"):
    runs = [json.load(open(f))["cells"] for f in sorted(glob.glob("$O/e_${TAG}_%s_[0-9]*.json" % tag))]
    print("cells planes-on", tag, {c: round(min(r[c] for r in runs), 3) for c in runs[0]})
PY
cp so/$TAG.so $DEST
for rep in 1 2; do for s in d1536_2bit d3072_2bit; do for th in st mt; do echo "speed $s $th $rep $(cd ~/turbovec/benchmarks/suite && TURBOVEC_2BIT_PLANES=1 python speed_${s}_${ARCH}_${th}.py 2>/dev/null | tr -d '\n ')"; done; done; done
git -C ~/turbovec checkout -q -- benchmarks/results
for d in 1536 3072; do echo "recall $(TURBOVEC_2BIT_PLANES=1 python prrecall.py $d 2>&1 | tail -1)"; done
( unset LD_PRELOAD; bash ~/hc/prcheck.sh ~/hc/$TAG.patch )
exec >> $O/$TAG.log 2>&1
echo "checks ok=$(grep -c 'test result: ok' ~/hc/prcheck.log) $(grep -E 'FAILED|panicked|clippy rc|^error' ~/hc/prcheck.log | tr '\n' ' ')"
echo ${TAG}_DONE
