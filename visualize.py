"""
Gateway Analysis — Charts
Reads sessions.csv and turns.csv produced by analyze.py.
Outputs 8 PNG charts for the Medium article.

Usage:
    python visualize.py [analysis_dir]
"""

import csv
import sys
from pathlib import Path

COLORS = [
    "#534AB7", "#D85A30", "#1D9E75", "#639922", "#BA7517",
    "#185FA5", "#993556", "#5F5E5A", "#A32D2D", "#0F6E56"
]

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def f(val, default=0.0):
    try: return float(val)
    except: return default

def i(val, default=0):
    try: return int(val)
    except: return default

def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
    })
    return plt

# ── Chart 1: Request size per turn, one line per session ─────────────────────
def chart_request_growth(turns, out, plt):
    from collections import defaultdict
    import matplotlib.cm as cm
    import numpy as np

    by_session = defaultdict(list)
    for r in turns:
        by_session[r["session_id"]].append(r)

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = cm.tab10(np.linspace(0, 1, len(by_session)))

    for (sid, rows), color in zip(sorted(by_session.items()), palette):
        rows = sorted(rows, key=lambda r: i(r["turn"]))
        x = [i(r["turn"]) for r in rows]
        y = [f(r["request_kb"]) for r in rows]
        label_str = rows[0]["label"] if rows else sid
        ax.plot(x, y, marker="o", markersize=3.5, label=f"{sid} {label_str}",
                color=color, linewidth=1.8)

    ax.set_xlabel("LLM call (turn number within session)")
    ax.set_ylabel("Request size (KB)")
    ax.set_title("Request payload grows with every turn — history accumulates in context")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "01_request_growth_per_turn.png")
    plt.close(fig)
    print("  Saved: 01_request_growth_per_turn.png")

# ── Chart 2: Baseline vs peak request size per session ───────────────────────
def chart_baseline_vs_peak(sessions, out, plt):
    import numpy as np
    sids = [r["session_id"] for r in sessions]
    base = [f(r["baseline_req_kb"]) for r in sessions]
    peak = [f(r["peak_req_kb"]) for r in sessions]
    x = np.arange(len(sids))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, base, w, label="First request (baseline)", color="#1D9E75", alpha=0.85)
    ax.bar(x + w/2, peak, w, label="Peak request (last turn)", color="#534AB7", alpha=0.85)

    for xi, (b, p) in enumerate(zip(base, peak)):
        ax.text(xi, p + 1, f"+{p-b:.0f}K", ha="center", va="bottom", fontsize=8, color="#534AB7")

    ax.set_xticks(x)
    ax.set_xticklabels(sids)
    ax.set_ylabel("Request size (KB)")
    ax.set_title("Baseline vs peak request size — the cost of conversation history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_baseline_vs_peak.png")
    plt.close(fig)
    print("  Saved: 02_baseline_vs_peak.png")

# ── Chart 3: Total request vs response size per session ──────────────────────
def chart_req_vs_resp(sessions, out, plt):
    import numpy as np
    sids = [r["session_id"] for r in sessions]
    req = [f(r["total_req_kb"]) for r in sessions]
    resp = [f(r["total_resp_kb"]) for r in sessions]
    ratios = [f(r["req_resp_ratio"]) for r in sessions]
    x = np.arange(len(sids))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, req,  w, label="Total sent (KB)",     color="#534AB7", alpha=0.85)
    ax.bar(x + w/2, resp, w, label="Total received (KB)", color="#1D9E75", alpha=0.85)

    for xi, ratio in enumerate(ratios):
        ax.text(xi, max(req[xi], resp[xi]) + 30, f"{ratio:.0f}:1",
                ha="center", va="bottom", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(sids)
    ax.set_ylabel("Size (KB)")
    ax.set_title("Total data sent vs received per session  (label = req/resp ratio)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_request_vs_response.png")
    plt.close(fig)
    print("  Saved: 03_request_vs_response.png")

# ── Chart 4: History inflation % per session ─────────────────────────────────
def chart_growth_pct(sessions, out, plt):
    sids   = [r["session_id"] for r in sessions]
    labels = [r["label"] for r in sessions]
    growth = [f(r["req_growth_pct"]) for r in sessions]
    calls  = [i(r["llm_calls"]) for r in sessions]

    bar_colors = ["#639922" if g < 50 else "#BA7517" if g < 100 else "#D85A30" for g in growth]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(sids, growth, color=bar_colors, alpha=0.85)

    for bar, g, c, lbl in zip(bars, growth, calls, labels):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{g:.0f}%  ({c} calls)", va="center", fontsize=9)

    ax.set_xlabel("Request size growth from first to last call (%)")
    ax.set_title("Context inflation per session\n(how much bigger is the last payload vs the first?)")
    ax.set_xlim(0, max(growth) * 1.3)
    fig.tight_layout()
    fig.savefig(out / "04_history_inflation.png")
    plt.close(fig)
    print("  Saved: 04_history_inflation.png")

# ── Chart 5: Latency box plot per session ────────────────────────────────────
def chart_latency_boxplot(turns, out, plt):
    from collections import defaultdict
    by_sess = defaultdict(list)
    for r in turns:
        if r["latency_ms"] not in (None, "", "None"):
            by_sess[r["session_id"]].append(f(r["latency_ms"]) / 1000)

    sids = sorted(by_sess.keys())
    data = [by_sess[s] for s in sids]

    fig, ax = plt.subplots(figsize=(13, 6))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    ax.set_xticks(range(1, len(sids) + 1))
    ax.set_xticklabels(sids)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Latency (seconds)")
    ax.set_xlabel("Session")
    ax.set_title("Response latency distribution per session")
    fig.tight_layout()
    fig.savefig(out / "05_latency_boxplot.png")
    plt.close(fig)
    print("  Saved: 05_latency_boxplot.png")

# ── Chart 6: Avg + P95 latency bars ──────────────────────────────────────────
def chart_latency_bars(sessions, out, plt):
    import numpy as np
    sids    = [r["session_id"] for r in sessions]
    avg_lat = [f(r["avg_latency_ms"]) / 1000 for r in sessions]
    p95_lat = [f(r["p95_latency_ms"]) / 1000 for r in sessions]
    x = np.arange(len(sids))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, avg_lat, w, label="Avg latency (s)",  color="#185FA5", alpha=0.85)
    ax.bar(x + w/2, p95_lat, w, label="P95 latency (s)",  color="#D85A30", alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(sids)
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Average and P95 response latency per session")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "06_latency_bars.png")
    plt.close(fig)
    print("  Saved: 06_latency_bars.png")

# ── Chart 7: Req/resp ratio per session ──────────────────────────────────────
def chart_req_resp_ratio(sessions, out, plt):
    sids   = [r["session_id"] for r in sessions]
    ratios = [f(r["req_resp_ratio"]) for r in sessions]
    bar_colors = ["#534AB7" if r > 7 else "#1D9E75" if r > 4 else "#639922" for r in ratios]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(sids, ratios, color=bar_colors, alpha=0.85)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{ratio:.0f}x", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=5, color="#D85A30", linestyle="--", alpha=0.5, linewidth=1.2, label="5x reference")
    ax.set_ylabel("Total request KB / total response KB")
    ax.set_title("How much more data does Claude receive than it sends back?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "07_req_resp_ratio.png")
    plt.close(fig)
    print("  Saved: 07_req_resp_ratio.png")

# ── Chart 8: Complexity scatter (calls × duration, bubble = total req) ───────
def chart_complexity_scatter(sessions, out, plt):
    calls    = [i(r["llm_calls"]) for r in sessions]
    duration = [f(r["duration_s"]) for r in sessions]
    total_req = [f(r["total_req_kb"]) for r in sessions]
    sids     = [r["session_id"] for r in sessions]

    max_req = max(total_req) if total_req else 1
    sizes = [max(60, (r / max_req) * 1400) for r in total_req]

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(calls, duration, s=sizes, alpha=0.65,
                    c=total_req, cmap="YlOrRd", edgecolors="#555", linewidth=0.5)
    for xi, sid in enumerate(sids):
        ax.annotate(sid, (calls[xi], duration[xi]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)

    plt.colorbar(sc, ax=ax, label="Total request KB")
    ax.set_xlabel("LLM calls in session")
    ax.set_ylabel("Session duration (seconds)")
    ax.set_title("Session complexity\n(bubble size = total data sent)")
    fig.tight_layout()
    fig.savefig(out / "08_complexity_scatter.png")
    plt.close(fig)
    print("  Saved: 08_complexity_scatter.png")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    analysis_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("analysis")

    for fname in ["sessions.csv", "turns.csv"]:
        if not (analysis_dir / fname).exists():
            print(f"ERROR: {analysis_dir / fname} not found. Run analyze.py first.")
            sys.exit(1)

    sessions = load_csv(analysis_dir / "sessions.csv")
    turns    = load_csv(analysis_dir / "turns.csv")
    print(f"Loaded {len(sessions)} sessions, {len(turns)} turns")

    plt = setup()
    print("Generating charts...")
    chart_request_growth(turns, analysis_dir, plt)
    chart_baseline_vs_peak(sessions, analysis_dir, plt)
    chart_req_vs_resp(sessions, analysis_dir, plt)
    chart_growth_pct(sessions, analysis_dir, plt)
    chart_latency_boxplot(turns, analysis_dir, plt)
    chart_latency_bars(sessions, analysis_dir, plt)
    chart_req_resp_ratio(sessions, analysis_dir, plt)
    chart_complexity_scatter(sessions, analysis_dir, plt)
    print(f"\nAll charts saved to {analysis_dir}/")

if __name__ == "__main__":
    main()
