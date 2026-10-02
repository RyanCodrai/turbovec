import sys, re
rows = []
for l in sys.stdin:
    m = re.search(r"scan=([0-9.]+)(µs|ms).*ranges=(\[.*\])", l)
    if not m: continue
    scan = float(m.group(1)) * (1000 if m.group(2) == "ms" else 1)
    rows.append((scan, eval(m.group(3)), l.split("ranges=")[0].strip()))
rows.sort(key=lambda r: r[0])
for scan, rg, head in rows[:2]:
    marks = [b for a, b in rg if a == 900000]
    tiles = sorted(r for r in rg if r[0] < 900000)
    # the last region is the main scan (the first is the sample pre-pass)
    main = [t for t in tiles]
    durs = sorted(t[1] for t in main)
    busy = sum(durs)
    print(head)
    print(f"   regions collected at {marks} us; tiles={len(main)} dur min/med/max={durs[0]}/{durs[len(durs)//2]}/{durs[-1]} us; sum of tile time={busy} us; last tile ends at {max(a+b for a,b in main)} us; first starts {sorted(a for a,b in main)[:10]}")
