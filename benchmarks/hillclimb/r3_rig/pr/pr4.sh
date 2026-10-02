#!/bin/bash
ARCH=$1; O=~/hc/prfinal; exec > $O/pr4.log 2>&1
cd ~/hc; bash build_so2.sh pr4 ~/hc/pr4.patch; tail -1 ~/hc_build_so2.log
source ~/.cargo/env
( cd ~/turbovec && cargo test -p turbovec --release --lib planes_tests 2>&1 | grep -E "test result|FAILED|panicked" )
source ~/venv/bin/activate; export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
cp so/pr4.so $DEST
for d in 1536 3072; do for th in 1 8; do echo "pr4 $(RAYON_NUM_THREADS=$th TURBOVEC_2BIT_PLANES=1 python ksweep.py $d 2>&1 | tail -1)"; done; done
n=0; for p in 1 2; do for tag in pr pr4 pr4 pr; do n=$((n+1)); cp so/$tag.so $DEST
  TURBOVEC_2BIT_PLANES=1 python cells_2bit.py --bits 2 --out $O/d2_${tag}_$n.json >/dev/null 2>$O/d2_$n.err || echo "RUN $n FAILED"; done; done
python - <<PY
import json, glob
for tag in ("pr", "pr4"):
    runs = [json.load(open(f))["cells"] for f in sorted(glob.glob("$O/d2_%s_[0-9]*.json" % tag))]
    print("cells planes-on", tag, {c: round(min(r[c] for r in runs), 3) for c in runs[0]})
PY
cp so/pr4.so $DEST
for rep in 1 2; do for s in d1536_2bit d3072_2bit; do for th in st mt; do echo "speed4 $s $th $rep $(cd ~/turbovec/benchmarks/suite && TURBOVEC_2BIT_PLANES=1 python speed_${s}_${ARCH}_${th}.py 2>/dev/null | tr -d '\n ')"; done; done; done
git -C ~/turbovec checkout -q -- benchmarks/results
for f in openai-1536.npy openai-3072.npy emb-mpnet768.npy; do nn=200000; [ $f = emb-mpnet768.npy ] && nn=41000
  NSINGLE=5000 GATE_MODES=exact,p128 python r3gate.py $f $nn 10000 2>&1 | grep -E 'p128|Error'; done
for d in 1536 3072; do echo "recall4 $(TURBOVEC_2BIT_PLANES=1 python prrecall.py $d 2>&1 | tail -1)"; done
echo PR4_DONE
