#!/bin/bash
# Round 3 soak: PASSES balanced ABBA passes of the full 4-cell harness; a label "tag:planes" runs tag.so with TURBOVEC_2BIT_PLANES=1.
# Leaves ~/hc/<NAME>_soak_{base,cand}.json (min per cell over that label's runs, plus every raw sample).
set -uo pipefail
ARCH=$1; A=$2; B=$3; PASSES=${4:-2}; NAME=${5:-${B%%:*}}
exec > ~/hc/${NAME}_soak.log 2>&1
source ~/venv/bin/activate
export LD_PRELOAD=$(ls /usr/lib/*-linux-gnu/libopenblas.so.0)
DEST=~/turbovec/turbovec-python/python/turbovec/_turbovec.abi3.so
cd ~/hc; rm -f ${NAME}_soak_run_*.json; n=0
for p in $(seq 1 $PASSES); do
  for label in $A $B $B $A; do
    n=$((n+1)); tag=${label%%:*}; planes=0; [ "$label" != "$tag" ] && planes=1
    cp ~/hc/so/${tag}.so $DEST
    side=base; [ "$label" = "$B" ] && side=cand
    TURBOVEC_2BIT_PLANES=$planes python cells_2bit.py --bits 2 --out ~/hc/${NAME}_soak_run_${side}_${n}.json >/dev/null 2>~/hc/${NAME}_soak_run_${n}.err || echo "RUN $n $label FAILED"
    echo "run $n $label done $(date -Is)"
  done
done
python - <<PY
import json, glob
for side in ("base", "cand"):
    runs = [json.load(open(f)) for f in sorted(glob.glob("$HOME/hc/${NAME}_soak_run_%s_*.json" % side))]
    cells = {c: min(r["cells"][c] for r in runs) for c in runs[0]["cells"]}
    json.dump({"cells": cells, "runs": [r["cells"] for r in runs], "raw": [r.get("raw") for r in runs]}, open("$HOME/hc/${NAME}_soak_%s.json" % side, "w"))
    print(side, {c: round(v, 4) for c, v in cells.items()}, "per-run", [{c: round(v, 3) for c, v in r["cells"].items()} for r in runs])
PY
cp ~/hc/so/${A%%:*}.so $DEST
echo SOAK_DONE
