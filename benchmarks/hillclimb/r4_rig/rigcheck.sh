#!/bin/bash
# Establish the rig: build the pinned baseline, then time and noise-check the smoke and soak with A/A runs.
ARCH=$1; O=~/hc/r4; exec > $O/rigcheck.log 2>&1
pkill -f "h1.sh" ; pkill -f cells_real.py; pkill -f h1gate.py; sleep 2
cd ~/hc; bash build_so2.sh base4 ~/hc/base4.patch; tail -1 ~/hc_build_so2.log
t0=$(date +%s); echo "== A/A smoke"; bash smoke.sh $ARCH base4 base4 | tail -18; echo "smoke seconds: $(( $(date +%s) - t0 ))"
t0=$(date +%s); bash soak.sh $ARCH base4 base4 2 aa; echo "== A/A soak"; tail -19 $O/aa_soak.log; echo "soak seconds: $(( $(date +%s) - t0 ))"
t0=$(date +%s); echo "== smoke base4 vs h1b:planes"; bash smoke.sh $ARCH base4 h1b:planes | tail -18; echo "smoke seconds: $(( $(date +%s) - t0 ))"
echo RIG_DONE
