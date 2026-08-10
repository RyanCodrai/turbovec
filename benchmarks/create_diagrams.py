"""Generate benchmark charts as SVG in the turboquant-wasm aesthetic.

Reads JSON files from ./results/ and writes:
  ../docs/arm_speed_st.svg, ../docs/arm_speed_mt.svg
  ../docs/x86_speed_st.svg, ../docs/x86_speed_mt.svg
  ../docs/arm_insert_online_st.svg, ../docs/x86_insert_online_st.svg
  ../docs/arm_remove_online_st.svg, ../docs/x86_remove_online_st.svg
  ../docs/arm_persist_st.svg, ../docs/arm_persist_mt.svg
  ../docs/x86_persist_st.svg, ../docs/x86_persist_mt.svg
  ../docs/recall_d1536.svg, ../docs/recall_d3072.svg, ../docs/recall_glove.svg
  ../docs/compression.svg
"""

import json
import math
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
C = {
    "title": "#0f172a",
    "subtitle": "#475569",
    "label": "#0f172a",
    "secondary": "#475569",
    "tick": "#64748b",
    "axis": "#334155",
    "grid": "#e5e7eb",
    "baseline": "#94a3b8",
    "tq": "#635bff",
    "tq_stroke": "#4338ca",
    "tq_text": "#4338ca",
    "faiss": "#9aa7b6",
    "fp32": "#9aa7b6",
    "four_bit": "#1d4ed8",
    "two_bit": "#635bff",
    "tq_2": "#635bff",
    "tq_4": "#0f766e",
    "faiss_2": "#9aa7b6",
    "faiss_4": "#64748b",
}


def xe(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def nice_ceil(value):
    if value <= 1:
        return 1
    exponent = math.floor(math.log10(value))
    fraction = value / 10 ** exponent
    if fraction <= 1:
        nf = 1
    elif fraction <= 1.5:
        nf = 1.5
    elif fraction <= 2:
        nf = 2
    elif fraction <= 5:
        nf = 5
    else:
        nf = 10
    return nf * 10 ** exponent


def style_block():
    return (
        f'<style>\n'
        f'  .title {{ font: 700 20px {FONT}; fill: {C["title"]}; }}\n'
        f'  .subtitle {{ font: 400 12px {FONT}; fill: {C["subtitle"]}; }}\n'
        f'  .panel {{ font: 700 14px {FONT}; fill: {C["title"]}; }}\n'
        f'  .label {{ font: 600 12px {FONT}; fill: {C["label"]}; }}\n'
        f'  .secondary {{ font: 400 11px {FONT}; fill: {C["secondary"]}; }}\n'
        f'  .tick {{ font: 400 11px {FONT}; fill: {C["tick"]}; }}\n'
        f'  .value {{ font: 700 11px {FONT}; fill: {C["label"]}; }}\n'
        f'  .value-accent {{ font: 700 11px {FONT}; fill: {C["tq_text"]}; }}\n'
        f'  .axis {{ font: 600 12px {FONT}; fill: {C["axis"]}; }}\n'
        f'  .legend {{ font: 600 12px {FONT}; fill: {C["label"]}; }}\n'
        f'</style>'
    )


def grid_lines(px, py, pw, ph, y_lo, y_hi, fmt, step_count=5):
    parts = []
    for i in range(step_count + 1):
        v = y_lo + (y_hi - y_lo) * i / step_count
        y = py + ph - (v - y_lo) / (y_hi - y_lo) * ph
        parts.append(
            f'<line x1="{px}" y1="{y:.1f}" x2="{px + pw}" y2="{y:.1f}" stroke="{C["grid"]}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{px - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{xe(fmt(v))}</text>'
        )
    parts.append(
        f'<line x1="{px}" y1="{py + ph:.1f}" x2="{px + pw}" y2="{py + ph:.1f}" stroke="{C["baseline"]}" stroke-width="1.5" />'
    )
    return "\n".join(parts)


def paired_panel(px, py, pw, ph, panel_title, groups, tick_fmt, value_fmt, y_max):
    parts = [grid_lines(px, py, pw, ph, 0, y_max, tick_fmt)]
    parts.append(f'<text x="{px}" y="{py - 14}" class="panel">{xe(panel_title)}</text>')
    n = len(groups)
    band = pw / n
    bar_w = min(44, band * 0.32)
    gap = 6
    for i, g in enumerate(groups):
        cx = px + band * i + band / 2
        tq_x = cx - bar_w - gap / 2
        faiss_x = cx + gap / 2
        tq_h = (g["tq"] / y_max) * ph
        faiss_h = (g["faiss"] / y_max) * ph
        tq_y = py + ph - tq_h
        faiss_y = py + ph - faiss_h
        label_y = py + ph + 22
        parts.append(
            f'<rect x="{tq_x:.1f}" y="{tq_y:.1f}" width="{bar_w}" height="{tq_h:.1f}" rx="6" '
            f'fill="{C["tq"]}" stroke="{C["tq_stroke"]}" stroke-width="1.5" />'
        )
        parts.append(
            f'<rect x="{faiss_x:.1f}" y="{faiss_y:.1f}" width="{bar_w}" height="{faiss_h:.1f}" rx="6" fill="{C["faiss"]}" />'
        )
        parts.append(
            f'<text x="{tq_x + bar_w/2:.1f}" y="{tq_y - 6:.1f}" text-anchor="middle" class="value-accent">{xe(value_fmt(g["tq"]))}</text>'
        )
        parts.append(
            f'<text x="{faiss_x + bar_w/2:.1f}" y="{faiss_y - 6:.1f}" text-anchor="middle" class="value">{xe(value_fmt(g["faiss"]))}</text>'
        )
        primary, _, secondary = g["label"].partition("|")
        parts.append(f'<text x="{cx:.1f}" y="{label_y}" text-anchor="middle" class="label">{xe(primary)}</text>')
        if secondary:
            parts.append(f'<text x="{cx:.1f}" y="{label_y + 15}" text-anchor="middle" class="secondary">{xe(secondary)}</text>')
    return "\n".join(parts)


def legend_tq_faiss(x, y):
    parts = [
        f'<rect x="{x}" y="{y - 10}" width="14" height="14" rx="3" fill="{C["tq"]}" stroke="{C["tq_stroke"]}" stroke-width="1.5" />',
        f'<text x="{x + 22}" y="{y + 1}" class="legend" style="fill: {C["tq_text"]};">TurboQuant</text>',
        f'<rect x="{x + 140}" y="{y - 10}" width="14" height="14" rx="3" fill="{C["faiss"]}" />',
        f'<text x="{x + 162}" y="{y + 1}" class="legend">FAISS</text>',
    ]
    return "\n".join(parts)


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def speed_panels(arch):
    panels = {"st": [], "mt": []}
    for dim in (1536, 3072):
        for bw in (2, 4):
            for th in ("st", "mt"):
                entry = load_json(f"speed_d{dim}_{bw}bit_{arch}_{th}.json")
                panels[th].append(
                    {
                        "label": f"d={dim}|{bw}-bit",
                        "tq": entry["tq_ms_per_query"],
                        "faiss": entry["faiss_ms_per_query"],
                    }
                )
    return panels


def write_speed_panel(arch, hw_label, thread_key, thread_label, tick_fmt, value_fmt, filename):
    panels = speed_panels(arch)
    width, height = 900, 460
    margin = {"top": 82, "right": 32, "bottom": 108, "left": 84}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]
    px = margin["left"]
    py = margin["top"]

    y_max = nice_ceil(max(max(g["tq"], g["faiss"]) for g in panels[thread_key]) * 1.22)

    parts = [
        paired_panel(
            px, py, pw, ph, thread_label, panels[thread_key],
            tick_fmt=tick_fmt,
            value_fmt=value_fmt,
            y_max=y_max,
        ),
        f'<text x="26" y="{py + ph/2}" transform="rotate(-90, 26, {py + ph/2})" class="axis">ms / query</text>',
        legend_tq_faiss(margin["left"], height - 26),
    ]
    body = "\n".join(parts)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Search Latency — {xe(hw_label)} — {xe(thread_label)}">
  {style_block()}
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin["left"]}" y="32" class="title">Search Latency — {xe(hw_label)} — {xe(thread_label)}</text>
  <text x="{margin["left"]}" y="52" class="subtitle">100K vectors, 1K queries, k=64, median of 5 runs</text>
  {body}
</svg>
"""
    out = os.path.join(DOCS_DIR, filename)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


def write_online_insert_panel(arch, hw_label, filename):
    # Two panels: per-vector add() latency at n=1 and n=100, TurboQuant vs
    # FAISS on the trained, populated index. Single-threaded cells: single
    # add is serial, so the ST figures are the per-call cost any caller pays.
    single, batch = [], []
    for dim in (1536, 3072):
        for bw in (2, 4):
            e = load_json(f"speed_insert_d{dim}_{bw}bit_{arch}_st.json")
            lbl = f"d={dim}|{bw}-bit"
            single.append({"label": lbl, "tq": e["tq_single_add_us"], "faiss": e["faiss_single_add_us"]})
            batch.append({"label": lbl, "tq": e["tq_batch100_add_us"] / 100, "faiss": e["faiss_batch100_add_us"] / 100})

    width, height = 1100, 470
    margin = {"top": 92, "right": 24, "bottom": 100, "left": 84}
    ph = height - margin["top"] - margin["bottom"]
    py = margin["top"]
    inner = width - margin["left"] - margin["right"]
    panel_gap = 64
    panel_w = (inner - panel_gap) / 2

    def us_val(v):
        return f"{v:.0f}" if v >= 20 else f"{v:.1f}"

    parts = []
    for i, (title, groups) in enumerate([("Single add (n=1)", single), ("Batched add (n=100), per vector", batch)]):
        px = margin["left"] + i * (panel_w + panel_gap)
        y_max = nice_ceil(max(max(g["tq"], g["faiss"]) for g in groups) * 1.22)
        parts.append(
            paired_panel(px, py, panel_w, ph, title, groups,
                         tick_fmt=lambda v: f"{v:.0f}", value_fmt=us_val, y_max=y_max)
        )

    parts.append(
        f'<text x="26" y="{py + ph/2}" transform="rotate(-90, 26, {py + ph/2})" class="axis">\u00b5s / vector</text>'
    )
    parts.append(legend_tq_faiss(margin["left"], height - 26))

    body = "\n".join(parts)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Online Insert Latency \u2014 {xe(hw_label)}">
  {style_block()}
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin["left"]}" y="32" class="title">Online Insert Latency \u2014 {xe(hw_label)}</text>
  <text x="{margin["left"]}" y="52" class="subtitle">Per-vector add() latency on a warm 100K-vector index, single-threaded, median of 5 runs. FAISS adds into the trained, populated IndexPQFastScan.</text>
  {body}
</svg>
"""
    out = os.path.join(DOCS_DIR, filename)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


def log_paired_panel(px, py, pw, ph, panel_title, groups, y_lo, y_hi, value_fmt):
    # Log10 y-axis variant of paired_panel for quantities spanning orders
    # of magnitude. Gridlines at decades; bars rise from y_lo.
    import math as _m
    lo, hi = _m.log10(y_lo), _m.log10(y_hi)

    def ypx(v):
        return py + ph - (_m.log10(v) - lo) / (hi - lo) * ph

    parts = [f'<text x="{px}" y="{py - 14}" class="panel">{xe(panel_title)}</text>']
    d = int(round(lo))
    while d <= hi + 1e-9:
        v = 10 ** d
        y = ypx(v)
        lbl = f"{v:g}" if v < 1000 else f"{v/1000:g}k"
        parts.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px + pw}" y2="{y:.1f}" stroke="{C["grid"]}" stroke-width="1" />')
        parts.append(f'<text x="{px - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{lbl}</text>')
        d += 1
    parts.append(f'<line x1="{px}" y1="{py + ph:.1f}" x2="{px + pw}" y2="{py + ph:.1f}" stroke="{C["baseline"]}" stroke-width="1.5" />')

    n = len(groups)
    band = pw / n
    bar_w = min(44, band * 0.32)
    gap = 6
    for i, g in enumerate(groups):
        cx = px + band * i + band / 2
        for xbar, key, accent in ((cx - bar_w - gap / 2, "tq", True), (cx + gap / 2, "faiss", False)):
            v = g[key]
            y = ypx(v)
            h = py + ph - y
            stroke = f' stroke="{C["tq_stroke"]}" stroke-width="1.5"' if accent else ""
            fill = C["tq"] if accent else C["faiss"]
            cls = "value-accent" if accent else "value"
            parts.append(f'<rect x="{xbar:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="6" fill="{fill}"{stroke} />')
            parts.append(f'<text x="{xbar + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="{cls}">{xe(value_fmt(v))}</text>')
        label_y = py + ph + 22
        primary, _, secondary = g["label"].partition("|")
        parts.append(f'<text x="{cx:.1f}" y="{label_y}" text-anchor="middle" class="label">{xe(primary)}</text>')
        if secondary:
            parts.append(f'<text x="{cx:.1f}" y="{label_y + 15}" text-anchor="middle" class="secondary">{xe(secondary)}</text>')
    return "\n".join(parts)


def write_online_remove_panel(arch, hw_label, filename):
    # Two panels: per-op remove-by-id latency at n=1 (steady per-op rate)
    # and n=100 (first 100 removes on a fresh index, per op), TurboQuant
    # IdMapIndex.remove vs FAISS IndexIDMap(IndexPQFastScan).remove_ids.
    # The y-axis is log10: TurboQuant's swap-and-pop sits near a
    # microsecond while FAISS's per-call compaction sits at milliseconds.
    import math as _m
    single, first100 = [], []
    for dim in (1536, 3072):
        for bw in (2, 4):
            e = load_json(f"speed_remove_d{dim}_{bw}bit_{arch}_st.json")
            lbl = f"d={dim}|{bw}-bit"
            single.append({"label": lbl, "tq": e["tq_remove_1_us"], "faiss": e["faiss_remove_1_us"]})
            first100.append({"label": lbl, "tq": e["tq_remove_100_us"] / 100, "faiss": e["faiss_remove_100_us"] / 100})

    width, height = 1100, 470
    margin = {"top": 92, "right": 24, "bottom": 100, "left": 84}
    ph = height - margin["top"] - margin["bottom"]
    py = margin["top"]
    inner = width - margin["left"] - margin["right"]
    panel_gap = 64
    panel_w = (inner - panel_gap) / 2

    def us_val(v):
        if v >= 1000:
            return f"{v/1000:.1f}ms"
        return f"{v:.0f}" if v >= 20 else f"{v:.2f}"

    all_vals = [g[k] for gs in (single, first100) for g in gs for k in ("tq", "faiss")]
    y_lo = 0.1
    y_hi = 10 ** _m.ceil(_m.log10(max(all_vals) * 1.2))

    parts = []
    for i, (title, groups) in enumerate([("Single remove (n=1)", single), ("First 100 removes, per op (n=100)", first100)]):
        px = margin["left"] + i * (panel_w + panel_gap)
        parts.append(log_paired_panel(px, py, panel_w, ph, title, groups, y_lo, y_hi, us_val))

    parts.append(
        f'<text x="26" y="{py + ph/2}" transform="rotate(-90, 26, {py + ph/2})" class="axis">\u00b5s / op (log scale)</text>'
    )
    legend_y = height - 26
    lx = margin["left"]
    parts.append(f'<rect x="{lx}" y="{legend_y - 10}" width="14" height="14" rx="3" fill="{C["tq"]}" stroke="{C["tq_stroke"]}" stroke-width="1.5" />')
    parts.append(f'<text x="{lx + 22}" y="{legend_y + 1}" class="legend" style="fill: {C["tq_text"]};">IdMapIndex.remove</text>')
    parts.append(f'<rect x="{lx + 190}" y="{legend_y - 10}" width="14" height="14" rx="3" fill="{C["faiss"]}" />')
    parts.append(f'<text x="{lx + 212}" y="{legend_y + 1}" class="legend">FAISS IndexIDMap(FastScan).remove_ids</text>')

    body = "\n".join(parts)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Online Remove Latency \u2014 {xe(hw_label)}">
  {style_block()}
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin["left"]}" y="32" class="title">Online Remove Latency \u2014 {xe(hw_label)}</text>
  <text x="{margin["left"]}" y="52" class="subtitle">Per-op remove-by-id latency on a fresh 100K-vector index, single-threaded, median of 5 runs, log-scale axis. FAISS remove_ids compacts the stored codes on every call, so its per-op cost scales with index size.</text>
  {body}
</svg>
"""
    out = os.path.join(DOCS_DIR, filename)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


def persist_panel(px, py, pw, ph, panel_title, groups, tick_fmt, value_fmt, y_max, paired=True):
    parts = [grid_lines(px, py, pw, ph, 0, y_max, tick_fmt)]
    parts.append(f'<text x="{px}" y="{py - 14}" class="panel">{xe(panel_title)}</text>')
    n = len(groups)
    band = pw / n
    bar_w = min(28, band * (0.30 if paired else 0.34))
    gap = 5
    for i, g in enumerate(groups):
        cx = px + band * i + band / 2
        tq_x = (cx - bar_w - gap / 2) if paired else (cx - bar_w / 2)
        tq_h = (g["tq"] / y_max) * ph
        tq_y = py + ph - tq_h
        parts.append(
            f'<rect x="{tq_x:.1f}" y="{tq_y:.1f}" width="{bar_w}" height="{tq_h:.1f}" rx="6" '
            f'fill="{C["tq"]}" stroke="{C["tq_stroke"]}" stroke-width="1.5" />'
        )
        parts.append(
            f'<text x="{tq_x + bar_w/2:.1f}" y="{tq_y - 6:.1f}" text-anchor="middle" class="value-accent">{xe(value_fmt(g["tq"]))}</text>'
        )
        if paired:
            faiss_x = cx + gap / 2
            faiss_h = (g["faiss"] / y_max) * ph
            faiss_y = py + ph - faiss_h
            parts.append(
                f'<rect x="{faiss_x:.1f}" y="{faiss_y:.1f}" width="{bar_w}" height="{faiss_h:.1f}" rx="6" fill="{C["faiss"]}" />'
            )
            parts.append(
                f'<text x="{faiss_x + bar_w/2:.1f}" y="{faiss_y - 6:.1f}" text-anchor="middle" class="value">{xe(value_fmt(g["faiss"]))}</text>'
            )
        label_y = py + ph + 22
        primary, _, secondary = g["label"].partition("|")
        parts.append(f'<text x="{cx:.1f}" y="{label_y}" text-anchor="middle" class="label">{xe(primary)}</text>')
        if secondary:
            parts.append(f'<text x="{cx:.1f}" y="{label_y + 15}" text-anchor="middle" class="secondary">{xe(secondary)}</text>')
    return "\n".join(parts)


def write_persist_panel(arch, hw_label, thread_key, thread_label, filename):
    # Three panels: Save (warm), Load -> first search (cold start), Round-trip.
    # The first two are precision-matched TurboQuant-vs-FAISS pairs; the round-trip
    # (mutate -> save -> load -> search) exercises TurboQuant's durability path and
    # has no measured FAISS equivalent, so that panel is TurboQuant-only.
    save, cold, rtrip = [], [], []
    for dim in (1536, 3072):
        for bw in (2, 4):
            e = load_json(f"speed_persist_d{dim}_{bw}bit_{arch}_{thread_key}.json")
            lbl = f"d={dim}|{bw}-bit"
            save.append({"label": lbl, "tq": e["tq_write_warm_ms"], "faiss": e["faiss_write_ms"]})
            cold.append({"label": lbl, "tq": e["tq_load_first_search_ms"], "faiss": e["faiss_read_first_search_ms"]})
            rtrip.append({"label": lbl, "tq": e["tq_mutate_save_load_search_ms"]})

    width, height = 1440, 480
    margin = {"top": 92, "right": 24, "bottom": 96, "left": 70}
    ph = height - margin["top"] - margin["bottom"]
    py = margin["top"]
    inner = width - margin["left"] - margin["right"]
    panel_gap = 56
    panel_w = (inner - 2 * panel_gap) / 3

    def ms_val(v):
        return f"{v:.0f}" if v >= 20 else f"{v:.1f}"

    def ms_tick(v):
        return f"{v:.0f}" if v >= 10 else f"{v:.1f}"

    specs = [
        ("Save (warm)", save, True),
        ("Load → first search", cold, True),
        ("Round-trip (mutate → save → load → search)", rtrip, False),
    ]
    parts = []
    for i, (title, groups, paired) in enumerate(specs):
        px = margin["left"] + i * (panel_w + panel_gap)
        vals = [g["tq"] for g in groups] + ([g["faiss"] for g in groups] if paired else [])
        y_max = nice_ceil(max(vals) * 1.22)
        parts.append(persist_panel(px, py, panel_w, ph, title, groups, ms_tick, ms_val, y_max, paired=paired))

    parts.append(
        f'<text x="24" y="{py + ph/2}" transform="rotate(-90, 24, {py + ph/2})" class="axis">milliseconds</text>'
    )

    legend_y = height - 24
    lx = margin["left"]
    parts.append(
        f'<rect x="{lx}" y="{legend_y - 10}" width="14" height="14" rx="3" fill="{C["tq"]}" stroke="{C["tq_stroke"]}" stroke-width="1.5" />'
    )
    parts.append(f'<text x="{lx + 22}" y="{legend_y + 1}" class="legend" style="fill: {C["tq_text"]};">TurboQuant</text>')
    parts.append(f'<rect x="{lx + 140}" y="{legend_y - 10}" width="14" height="14" rx="3" fill="{C["faiss"]}" />')
    parts.append(f'<text x="{lx + 162}" y="{legend_y + 1}" class="legend">FAISS</text>')
    parts.append(
        f'<text x="{lx + 240}" y="{legend_y + 1}" class="secondary">Round-trip is TurboQuant-only — FAISS has no measured mutate→save→load→search equivalent.</text>'
    )

    body = "\n".join(parts)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Persist — {xe(hw_label)} — {xe(thread_label)}">
  {style_block()}
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin["left"]}" y="32" class="title">Save / Load — {xe(hw_label)} — {xe(thread_label)}</text>
  <text x="{margin["left"]}" y="52" class="subtitle">100K vectors. Save: full index → .tv file. Load → first search: open a cold .tv and run the first query. Round-trip: mutate → save → load → search. Median of 5 runs.</text>
  {body}
</svg>
"""
    out = os.path.join(DOCS_DIR, filename)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


def line_panel(px, py, pw, ph, panel_title, series, x_values, x_labels, y_lo, y_hi):
    parts = [
        grid_lines(px, py, pw, ph, y_lo, y_hi, lambda v: f"{v:.2f}"),
        f'<text x="{px}" y="{py - 14}" class="panel">{xe(panel_title)}</text>',
    ]
    x_min = math.log2(x_values[0])
    x_max = math.log2(x_values[-1])

    def xpx(v):
        return px + (math.log2(v) - x_min) / (x_max - x_min) * pw

    def ypx(v):
        return py + ph - (v - y_lo) / (y_hi - y_lo) * ph

    for v, lbl in zip(x_values, x_labels):
        parts.append(
            f'<text x="{xpx(v):.1f}" y="{py + ph + 20}" text-anchor="middle" class="label">{xe(lbl)}</text>'
        )

    for s in series:
        color = s["color"]
        pattern = s.get("dash") or ("6 4" if s.get("dashed") else None)
        dash = f' stroke-dasharray="{pattern}"' if pattern else ""
        points = [(xpx(x), ypx(y)) for x, y in zip(x_values, s["values"])]
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.25"{dash} />'
        )
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" />')

    return "\n".join(parts)


def write_recall_panel(dim_key, dim_label, filename, y_lo=0.85):
    width, height = 900, 460
    margin = {"top": 82, "right": 32, "bottom": 108, "left": 84}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]
    px = margin["left"]
    py = margin["top"]

    x_values = [1, 2, 4, 8, 16, 32, 64]
    x_labels = ["1", "2", "4", "8", "16", "32", "64"]

    # Draw FAISS lines first (background), then TQ+ on top — emphasises the
    # TQ+ series when lines overlap or cross at high-K.
    faiss_series = []
    tqp_series = []
    for bw_key, bw_label in [("2bit", "2-bit"), ("4bit", "4-bit")]:
        data = load_json(f"recall_{dim_key}_{bw_key}.json")
        tqp_vals = [float(data["tqplus_recalls"][str(k)]) for k in x_values]
        faiss_vals = [float(data["faiss_recalls"][str(k)]) for k in x_values]
        tqp_color = C["tq_2"] if bw_key == "2bit" else C["tq_4"]
        faiss_color = C["faiss_2"] if bw_key == "2bit" else C["faiss_4"]
        tqp_series.append({"label": f"TQ+ {bw_label}", "values": tqp_vals, "color": tqp_color})
        faiss_series.append({"label": f"FAISS {bw_label}", "values": faiss_vals, "color": faiss_color, "dashed": True})
    series = faiss_series + tqp_series

    parts = [
        line_panel(px, py, pw, ph, dim_label, series, x_values, x_labels, y_lo, 1.005),
        f'<text x="{px - 62}" y="{py + ph/2}" transform="rotate(-90, {px - 62}, {py + ph/2})" class="axis">recall@1@k</text>',
        f'<text x="{px + pw/2}" y="{py + ph + 48}" text-anchor="middle" class="axis">k</text>',
    ]

    legend_y = height - 26
    lx = margin["left"]
    items = [
        ("TQ+ 2-bit", C["tq_2"], None),
        ("TQ+ 4-bit", C["tq_4"], None),
        ("FAISS 2-bit", C["faiss_2"], "6 4"),
        ("FAISS 4-bit", C["faiss_4"], "6 4"),
    ]
    for i, (lbl, col, dash) in enumerate(items):
        cx = lx + i * 140
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{cx}" y1="{legend_y - 2}" x2="{cx + 24}" y2="{legend_y - 2}" stroke="{col}" stroke-width="2.25"{dash_attr} />'
        )
        parts.append(f'<circle cx="{cx + 12}" cy="{legend_y - 2}" r="3.5" fill="{col}" />')
        parts.append(f'<text x="{cx + 32}" y="{legend_y + 1}" class="legend">{xe(lbl)}</text>')

    body = "\n".join(parts)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Recall — {xe(dim_label)}">
  {style_block()}
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin["left"]}" y="32" class="title">Recall — {xe(dim_label)}</text>
  <text x="{margin["left"]}" y="52" class="subtitle">100K vectors, k=64 search. recall@1@k measures how often the true top-1 result appears in the top-k returned.</text>
  {body}
</svg>
"""
    out = os.path.join(DOCS_DIR, filename)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


def write_compression_chart(filename):
    # Read from results/compression.json rather than restating it. The
    # hardcoded copy happened to agree, but only by hand — a rerun that
    # moved the JSON left the shipped figure silently stale (#492).
    raw = load_json("compression.json")

    def cell(key):
        d = raw[key]
        return d["fp32_mb"], d["index_mb"]

    datasets = []
    for label, stem in (
        ("GloVe|d=200", "glove_d200"),
        ("OpenAI|d=1536", "openai_d1536"),
        ("OpenAI|d=3072", "openai_d3072"),
    ):
        fp32, four_bit = cell(f"{stem}_4bit")
        fp32_two, two_bit = cell(f"{stem}_2bit")
        assert fp32 == fp32_two, (
            f"{stem}: the 2-bit and 4-bit rows disagree on fp32_mb "
            f"({fp32_two} vs {fp32}) — one of them is stale"
        )
        datasets.append((label, fp32, four_bit, two_bit))
    width, height = 900, 460
    margin = {"top": 82, "right": 32, "bottom": 108, "left": 84}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]
    px = margin["left"]
    py = margin["top"]

    y_max = nice_ceil(max(d[1] for d in datasets) * 1.15)

    parts = [grid_lines(px, py, pw, ph, 0, y_max, lambda v: f"{v:.0f}")]

    n = len(datasets)
    band = pw / n
    bar_w = min(56, band * 0.22)
    gap = 10

    for i, (label, fp32, four, two) in enumerate(datasets):
        cx = px + band * i + band / 2
        x_fp = cx - 1.5 * bar_w - gap
        x_4 = cx - 0.5 * bar_w
        x_2 = cx + 0.5 * bar_w + gap

        def draw(xbar, val, color, accent=False):
            h = (val / y_max) * ph
            y = py + ph - h
            stroke = (
                f' stroke="{C["tq_stroke"]}" stroke-width="1.5"' if accent else ""
            )
            value_cls = "value-accent" if accent else "value"
            return "\n".join(
                [
                    f'<rect x="{xbar:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="6" fill="{color}"{stroke} />',
                    f'<text x="{xbar + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="{value_cls}">{xe(f"{val:.0f}")}</text>',
                ]
            )

        parts.append(draw(x_fp, fp32, C["fp32"]))
        parts.append(draw(x_4, four, C["four_bit"]))
        parts.append(draw(x_2, two, C["two_bit"], accent=True))

        label_y = py + ph + 22
        primary, _, secondary = label.partition("|")
        parts.append(f'<text x="{cx:.1f}" y="{label_y}" text-anchor="middle" class="label">{xe(primary)}</text>')
        if secondary:
            parts.append(f'<text x="{cx:.1f}" y="{label_y + 15}" text-anchor="middle" class="secondary">{xe(secondary)}</text>')

    parts.append(
        f'<text x="26" y="{py + ph/2}" transform="rotate(-90, 26, {py + ph/2})" class="axis">Index size (MB)</text>'
    )

    legend_y = height - 26
    lx = margin["left"]
    items = [
        ("FP32", C["fp32"], False),
        ("4-bit", C["four_bit"], False),
        ("2-bit", C["two_bit"], True),
    ]
    for i, (lbl, col, accent) in enumerate(items):
        lcx = lx + i * 120
        stroke = f' stroke="{C["tq_stroke"]}" stroke-width="1.5"' if accent else ""
        parts.append(f'<rect x="{lcx}" y="{legend_y - 10}" width="14" height="14" rx="3" fill="{col}"{stroke} />')
        parts.append(f'<text x="{lcx + 22}" y="{legend_y + 1}" class="legend">{xe(lbl)}</text>')

    body = "\n".join(parts)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Index Size — TurboQuant">
  {style_block()}
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin["left"]}" y="32" class="title">Index Size — 100K vectors</text>
  <text x="{margin["left"]}" y="52" class="subtitle">TurboQuant packs vectors ~16× smaller than FP32 at 2-bit with comparable recall</text>
  {body}
</svg>
"""
    out = os.path.join(DOCS_DIR, filename)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    write_speed_panel("arm", "ARM (GCP c4a-standard-8, Google Axion, 8 vCPUs)", "st", "Single-threaded",
                      tick_fmt=lambda v: f"{v:.1f}", value_fmt=lambda v: f"{v:.2f}",
                      filename="arm_speed_st.svg")
    write_speed_panel("arm", "ARM (GCP c4a-standard-8, Google Axion, 8 vCPUs)", "mt", "Multi-threaded",
                      tick_fmt=lambda v: f"{v:.2f}", value_fmt=lambda v: f"{v:.3f}",
                      filename="arm_speed_mt.svg")
    write_speed_panel("x86", "x86 (Intel Sapphire Rapids, 8 vCPUs)", "st", "Single-threaded",
                      tick_fmt=lambda v: f"{v:.1f}", value_fmt=lambda v: f"{v:.2f}",
                      filename="x86_speed_st.svg")
    write_speed_panel("x86", "x86 (Intel Sapphire Rapids, 8 vCPUs)", "mt", "Multi-threaded",
                      tick_fmt=lambda v: f"{v:.2f}", value_fmt=lambda v: f"{v:.3f}",
                      filename="x86_speed_mt.svg")
    # Insert/remove online-latency figures, per architecture.
    write_online_insert_panel("arm", "ARM (GCP c4a-standard-8, Google Axion, 8 vCPUs)",
                              filename="arm_insert_online_st.svg")
    write_online_insert_panel("x86", "x86 (Intel Sapphire Rapids, 8 vCPUs)",
                              filename="x86_insert_online_st.svg")
    write_online_remove_panel("arm", "ARM (GCP c4a-standard-8, Google Axion, 8 vCPUs)",
                              filename="arm_remove_online_st.svg")
    write_online_remove_panel("x86", "x86 (Intel Sapphire Rapids, 8 vCPUs)",
                              filename="x86_remove_online_st.svg")
    # Save/load (persist) figures, per architecture.
    write_persist_panel("arm", "ARM (GCP c4a-standard-8, Google Axion, 8 vCPUs)", "st", "Single-threaded",
                        filename="arm_persist_st.svg")
    write_persist_panel("arm", "ARM (GCP c4a-standard-8, Google Axion, 8 vCPUs)", "mt", "Multi-threaded",
                        filename="arm_persist_mt.svg")
    write_persist_panel("x86", "x86 (Intel Sapphire Rapids, 8 vCPUs)", "st", "Single-threaded",
                        filename="x86_persist_st.svg")
    write_persist_panel("x86", "x86 (Intel Sapphire Rapids, 8 vCPUs)", "mt", "Multi-threaded",
                        filename="x86_persist_mt.svg")
    write_recall_panel("d1536", "d=1536", "recall_d1536.svg")
    write_recall_panel("d3072", "d=3072", "recall_d3072.svg")
    write_recall_panel("glove", "GloVe d=200", "recall_glove.svg", y_lo=0.4)
    write_compression_chart("compression.svg")
