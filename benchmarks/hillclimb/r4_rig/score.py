"""Score a candidate against the baseline on the 32 real-embedding cells (16 per arch).

usage: score.py ARCH=BASE.json,BASE2.json:CAND.json,CAND2.json [ARCH=...]
A cell's time is the minimum over that side's runs. WIN = harmonic mean > 1.01 and every cell >= 0.99.
"""
import json, statistics, sys

sp = {}
for spec in sys.argv[1:]:
    arch, rest = spec.split("=")
    base, cand = (side.split(",") for side in rest.split(":"))
    def cells(files):
        runs = [json.load(open(f))["cells"] for f in files]
        return {c: min(r[c] for r in runs) for c in runs[0]}
    b, c = cells(base), cells(cand)
    for name in sorted(b):
        sp[f"{arch} {name}"] = (b[name], c[name], b[name] / c[name])
for name, (b, c, s) in sp.items():
    print(f"{name:28s} {b:9.4f} {c:9.4f}  x{s:.4f}{'   <-- below floor' if s < 0.99 else ''}")
hm = statistics.harmonic_mean(s for _, _, s in sp.values())
floor = min(s for _, _, s in sp.values())
print(f"cells={len(sp)} HM=x{hm:.4f} floor=x{floor:.4f} -> {'WIN' if hm > 1.01 and floor >= 0.99 else 'NOT A WIN'}")
