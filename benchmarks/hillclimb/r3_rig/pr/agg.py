import json, glob, os, statistics as st
O = os.path.expanduser("~/hc/prfinal")
for bits, labels in ((2, ("main", "pr", "pr_planes")), (4, ("main", "pr"))):
    res = {}
    for l in labels:
        runs = [json.load(open(f))["cells"] for f in sorted(glob.glob(f"{O}/c{bits}_{l}_[0-9]*.json"))]
        res[l] = ({c: min(r[c] for r in runs) for c in runs[0]}, {c: st.median(r[c] for r in runs) for c in runs[0]}, len(runs))
    for c in sorted(res["main"][0]):
        print(f"bits={bits} {c:9s} " + " ".join(f"{l}: min {res[l][0][c]:.3f} med {res[l][1][c]:.3f} (n={res[l][2]})" for l in labels)
              + " | x vs main (min): " + " ".join(f"{l} {res['main'][0][c] / res[l][0][c]:.3f}" for l in labels[1:]))
