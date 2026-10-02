#!/bin/bash
# Round-3 gates for one build, detached: real-data id/score gate, then cargo test with the planes toggle off and on.
TAG=$1; ARCH=$2
source ~/venv/bin/activate; source ~/.cargo/env
export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
cp ~/hc/so/$TAG.so ~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
cd ~/hc
for f in openai-1536.npy openai-3072.npy emb-mpnet768.npy; do
  [ -f ~/data/py-turboquant/$f ] || continue
  n=200000; [ $f = emb-mpnet768.npy ] && n=41000
  python r3gate.py $f $n 10000 > ~/hc/${TAG}_gate_${f%.npy}.log 2>&1
done
cd ~/turbovec
( unset LD_PRELOAD; cargo test --release -p turbovec > ~/hc/${TAG}_test_off.log 2>&1; echo "exit=$?" >> ~/hc/${TAG}_test_off.log
  TURBOVEC_2BIT_PLANES=1 TURBOVEC_PLANES_MIN_N=0 cargo test --release -p turbovec > ~/hc/${TAG}_test_on.log 2>&1; echo "exit=$?" >> ~/hc/${TAG}_test_on.log
  TURBOVEC_2BIT_PLANES=1 cargo test --release -p turbovec > ~/hc/${TAG}_test_on_gated.log 2>&1; echo "exit=$?" >> ~/hc/${TAG}_test_on_gated.log )
[ -n "${SOAK_BASE:-}" ] && bash ~/hc/r3soak.sh $ARCH $SOAK_BASE $TAG:planes 2 $TAG
echo GATES_DONE > ~/hc/${TAG}_gates.done
