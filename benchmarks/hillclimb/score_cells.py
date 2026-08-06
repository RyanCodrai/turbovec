"""Score a build against origin/main on the 8-cell goal metric.

Cells: {arm, x86} x {st, mt} x {nq=100, nq=1}, weighted equally. Per-cell
speedup is main_ms / now_ms, and the score is the HARMONIC mean of the
eight — harmonic, not arithmetic, because a regressing cell contributes a
large 1/s and must drag the figure down rather than be averaged away by
seven wins. Speedups rather than raw times, because the cells span 0.6 ms
to 320 ms and a mean over raw times would be decided by the smallest one.

Reads `<arch> <arm> <json>` lines on stdin, as emitted by the runner.
"""
import json
import statistics
import sys

rows = {}
for line in sys.stdin:
    parts = line.split(None, 2)
    if len(parts) != 3 or not parts[2].lstrip().startswith("{"):
        continue
    arch, arm, blob = parts
    try:
        cells = json.loads(blob)
    except json.JSONDecodeError:
        continue
    for cell, ms in cells.items():
        rows.setdefault((arch, cell), {}).setdefault(arm, []).append(ms)

speedups = {}
print(f"{'cell':<18} {'main':>10} {'now':>10} {'speedup':>9}")
for (arch, cell), arms in sorted(rows.items()):
    if "main" not in arms or "now" not in arms:
        continue
    m, n = statistics.median(arms["main"]), statistics.median(arms["now"])
    s = m / n
    speedups[f"{arch}_{cell}"] = s
    flag = "  <-- REGRESSION" if s < 0.99 else ""
    print(f"{arch + '_' + cell:<18} {m:>10.3f} {n:>10.3f} {s:>8.3f}x{flag}")

if speedups:
    hm = len(speedups) / sum(1.0 / s for s in speedups.values())
    am = sum(speedups.values()) / len(speedups)
    print(f"\n{len(speedups)} cells   harmonic mean {hm:.4f}x   (arithmetic {am:.4f}x)")
    worst = min(speedups.items(), key=lambda kv: kv[1])
    print(f"worst cell: {worst[0]} at {worst[1]:.3f}x")
