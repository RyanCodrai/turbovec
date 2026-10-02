#!/bin/bash
# usage: cand.sh ARCH TAG  -> build TAG from ~/hc/TAG.patch, layout tests, smoke vs base4 (planes on)
ARCH=$1; TAG=$2; O=~/hc/r4; exec > $O/$TAG.log 2>&1
while pgrep -f "soak.sh|smoke.sh|cells_real" >/dev/null; do sleep 5; done
cd ~/hc; bash build_so2.sh $TAG ~/hc/$TAG.patch; tail -1 ~/hc_build_so2.log
( source ~/.cargo/env; cd ~/turbovec && cargo test -p turbovec --release --lib planes_tests 2>&1 | grep -E "test result|FAILED|panicked|^error" )
bash smoke.sh $ARCH base4 $TAG:planes | tail -17
echo ${TAG}_DONE
