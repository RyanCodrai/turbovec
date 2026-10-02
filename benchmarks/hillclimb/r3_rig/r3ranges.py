import sys, re
MK = {900000: "region-collected", 900001: "sample-done", 900002: "scan-returned"}
rows = []
for l in sys.stdin:
    m = re.search(r"scan=([0-9.]+)(µs|ms).*ranges=(\[.*\])", l)
    if not m: continue
    scan = float(m.group(1)) * (1000 if m.group(2) == "ms" else 1)
    rows.append((scan, eval(m.group(3)), l.split("ranges=")[0].strip()))
rows.sort(key=lambda r: r[0])
for scan, rg, head in rows[:2] + [rows[len(rows) // 2]]:
    print(head)
    print("   markers(us):", [(MK[a], b) for a, b in rg if a >= 900000])
    print("   ranges (start, dur):", sorted(r for r in rg if r[0] < 900000))
