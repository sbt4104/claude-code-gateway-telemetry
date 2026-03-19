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

# ── Chart 9: Stacked bar — payload breakdown % per session ───────────────────
def chart_cache_payload_breakdown(sessions, out, plt):
    """
    Stacked bar showing what % of each session's payload falls into each category:
    system_cached / system_uncached / tools / messages_cached / messages_uncached
    """
    # Skip sessions that don't have cache columns (old CSVs)
    if "system_prompt_chars" not in sessions[0]:
        print("  Skipped: 09_cache_payload_breakdown.png (no cache columns — run new analyze.py)")
        return

    import numpy as np
    sids = [r["session_id"] for r in sessions]
    tc   = [max(f(r["cache_total_chars"]), 1) for r in sessions]

    sys_p  = [f(r["system_prompt_chars"])   / tc[i] * 100 for i, r in enumerate(sessions)]
    tools  = [f(r["tools_chars"])           / tc[i] * 100 for i, r in enumerate(sessions)]
    prior  = [f(r["prior_history_chars"])   / tc[i] * 100 for i, r in enumerate(sessions)]
    bndry  = [f(r["new_input_chars"])  / tc[i] * 100 for i, r in enumerate(sessions)]
    bhead  = [f(r["billing_header_chars"])  / tc[i] * 100 for i, r in enumerate(sessions)]

    x = np.arange(len(sids))
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x, sys_p, label="System prompt (cache read)",   color="#1D9E75", alpha=0.9)
    ax.bar(x, tools, bottom=sys_p,
           label="Tools (cache read)",                      color="#534AB7", alpha=0.75)
    ax.bar(x, prior, bottom=[a+b for a,b in zip(sys_p, tools)],
           label="Prior history (cache read)",              color="#185FA5", alpha=0.75)
    ax.bar(x, bndry, bottom=[a+b+c for a,b,c in zip(sys_p, tools, prior)],
           label="New input / boundary block (cache write)",color="#BA7517", alpha=0.9)
    ax.bar(x, bhead, bottom=[a+b+c+d for a,b,c,d in zip(sys_p, tools, prior, bndry)],
           label="Billing header (uncached)",               color="#D85A30", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(sids)
    ax.set_ylabel("% of total payload (chars)")
    ax.set_ylim(0, 105)
    ax.set_title("What is every payload actually made of?\n(% breakdown per session — summed across all turns)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "09_cache_payload_breakdown.png")
    plt.close(fig)
    print("  Saved: 09_cache_payload_breakdown.png")


# ── Chart 10: Cache read % per session ───────────────────────────────────────
def chart_cache_v1_vs_v2(sessions, out, plt):
    """
    Bar chart showing cache_read % per session.
    """
    if "cache_read_pct" not in sessions[0]:
        print("  Skipped: 10_cache_read_pct.png (no cache columns — run new analyze.py)")
        return

    import numpy as np
    sids      = [r["session_id"] for r in sessions]
    read_pct  = [f(r["cache_read_pct"])  for r in sessions]
    write_pct = [f(r["cache_write_pct"]) for r in sessions]
    unc_pct   = [f(r["billing_header_pct"])    for r in sessions]
    x = np.arange(len(sids))
    w = 0.55

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, read_pct,  w, label="Cache read (system + tools + prior history)", color="#1D9E75", alpha=0.85)
    ax.bar(x, write_pct, w, bottom=read_pct,
           label="Cache write (new user input)", color="#BA7517", alpha=0.85)
    ax.bar(x, unc_pct,   w, bottom=[a+b for a,b in zip(read_pct, write_pct)],
           label="Uncached (billing header)", color="#D85A30", alpha=0.75)

    for xi, (r, w2, u) in enumerate(zip(read_pct, write_pct, unc_pct)):
        ax.text(xi, r + 0.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=8, color="#1D9E75")

    ax.set_xticks(x)
    ax.set_xticklabels(sids)
    ax.set_ylabel("% of payload")
    ax.set_ylim(0, 110)
    ax.set_title("Cache read vs cache write vs uncached per session\n"
                 "cache_read = system + tools + prior history  |  cache_write = new user input")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "10_cache_read_pct.png")
    plt.close(fig)
    print("  Saved: 10_cache_read_pct.png")


# ── Chart 11: Cache read % over turns ────────────────────────────────────────
def chart_cache_pct_over_turns(turns, out, plt):
    """
    Line chart: cache_read % per turn per session.
    Shows how the fraction of already-cached content changes as sessions grow.
    Also saves 11b: cache_write % per turn (how much is new each call).
    """
    if "cache_read_pct" not in turns[0]:
        print("  Skipped: 11_cache_pct_over_turns.png (no cache columns — run new analyze.py)")
        return

    from collections import defaultdict
    import matplotlib.cm as cm
    import numpy as np

    by_session = defaultdict(list)
    for r in turns:
        by_session[r["session_id"]].append(r)

    palette = cm.tab10(np.linspace(0, 1, len(by_session)))

    # Chart 11: all lines start from a single 0 point at turn 1,
    # then connect to their turn 2+ values on a tight Y axis.
    # S10 turn 21 restart (also 0%) is annotated as a special case.
    vals_t2 = [f(r["cache_read_pct"]) for r in turns
               if r.get("cache_parse_ok") in (True, "True")
               and i(r["turn"]) > 1 and f(r["cache_read_pct"]) > 0]
    y_min = min(vals_t2) - 2 if vals_t2 else 70
    y_max = max(vals_t2) + 1 if vals_t2 else 100

    fig = plt.figure(figsize=(13, 7))

    # Two subplots sharing x: top = tight Y axis (turns 2+), bottom = turn 1 dot at 0
    ax_main = fig.add_axes([0.08, 0.25, 0.88, 0.65])  # main chart
    ax_zero = fig.add_axes([0.08, 0.08, 0.88, 0.12])  # small strip at bottom for 0%

    # Share x axis
    ax_zero.sharex(ax_main)

    for (sid, rows), color in zip(sorted(by_session.items()), palette):
        rows = sorted(rows, key=lambda r: i(r["turn"]))
        label_str = rows[0]["label"] if rows else sid

        # Turn 1 dot in bottom strip
        ax_zero.plot(1, 0, marker="o", markersize=5, color=color, zorder=3)

        # Turns 2+ in main chart — skip zeros (restarts) except annotate S10
        x_main, y_main = [], []
        for r in rows:
            t = i(r["turn"])
            v = f(r["cache_read_pct"])
            if t == 1:
                continue
            if v == 0:
                # Annotate S10 restart
                ax_main.annotate("S10\nrestart", xy=(t, y_min + 1),
                                 xytext=(t + 0.8, y_min + 3),
                                 fontsize=7, color=color,
                                 arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
                continue
            x_main.append(t)
            y_main.append(v)

        if x_main:
            ax_main.plot(x_main, y_main, marker="o", markersize=3.5,
                         label=f"{sid} {label_str}", color=color, linewidth=1.8)

        # Connecting line from turn 1 (bottom strip) to turn 2 (main chart)
        if x_main:
            # Draw a dashed line bridging the gap between the two axes
            con = plt.matplotlib.patches.ConnectionPatch(
                xyA=(1, 0), xyB=(x_main[0], y_main[0]),
                coordsA="data", coordsB="data",
                axesA=ax_zero, axesB=ax_main,
                color=color, linestyle="dashed", linewidth=0.8, alpha=0.5
            )
            fig.add_artist(con)

    # Style main axis
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_ylabel("Cache read % of payload")
    ax_main.set_title("Cache read % per turn — all sessions start at 0% (turn 1)\n"
                      "Axis break shows jump to turns 2+ range", pad=10)
    ax_main.legend(loc="upper right", fontsize=8, ncol=2)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.tick_params(bottom=False, labelbottom=False)

    # Style zero strip
    ax_zero.set_ylim(-0.5, 0.8)
    ax_zero.set_yticks([0])
    ax_zero.set_yticklabels(["0%"])
    ax_zero.set_xlabel("LLM call (turn number within session)")
    ax_zero.spines['top'].set_visible(False)
    ax_zero.set_xlim(ax_main.get_xlim())

    # Broken axis markers between the two panels
    d = 0.015
    kwargs = dict(transform=ax_main.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax_main.plot((-d, +d), (-d, +d), **kwargs)
    ax_main.plot((-d, +d), (-2.5*d, -1.5*d), **kwargs)
    kwargs2 = dict(transform=ax_zero.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax_zero.plot((-d, +d), (1-d, 1+d), **kwargs2)
    ax_zero.plot((-d, +d), (1-2.5*d, 1-1.5*d), **kwargs2)

    fig.savefig(out / "11_cache_pct_over_turns.png")
    plt.close(fig)
    print("  Saved: 11_cache_pct_over_turns.png")

    # Chart 11b: cache_write % — single axis
    # Turn 1 and session restarts (~100%) shown as dummy dots at DUMMY_100
    # All other turns plotted at real values with tight Y axis
    vals_t2_write = [f(r["cache_write_pct"]) for r in turns
                     if r.get("cache_parse_ok") in (True, "True") and i(r["turn"]) > 1
                     and f(r["cache_write_pct"]) < 50]  # exclude restarts from range calc
    y_max_w  = max(vals_t2_write) + 3 if vals_t2_write else 25
    DUMMY_100 = y_max_w + 4   # visual position for the "100%" anchor dots

    fig, ax = plt.subplots(figsize=(13, 6))

    for (sid, rows), color in zip(sorted(by_session.items()), palette):
        rows = sorted(rows, key=lambda r: i(r["turn"]))
        label_str = rows[0]["label"] if rows else sid
        plotted_label = False
        prev_x, prev_y_real = None, None  # track last real (non-reset) point

        for idx, r in enumerate(rows):
            t = i(r["turn"])
            v = f(r["cache_write_pct"])
            is_reset = v > 50  # turn 1 or session restart

            lbl = f"{sid} {label_str}" if not plotted_label else ""

            if is_reset:
                # Dummy dot at DUMMY_100
                ax.plot(t, DUMMY_100, marker="o", markersize=5, color=color,
                        zorder=3, label=lbl)
                plotted_label = True

                # Annotate restart (not turn 1)
                if t > 1:
                    ax.annotate("restart\n(~100%)", xy=(t, DUMMY_100),
                                xytext=(t + 0.8, DUMMY_100 - 1),
                                fontsize=7, color=color,
                                arrowprops=dict(arrowstyle="->", color=color, lw=0.7))

                # Dashed line from previous real point up to this dummy
                if prev_x is not None and prev_y_real is not None:
                    ax.plot([prev_x, t], [prev_y_real, DUMMY_100],
                            color=color, linestyle="dashed", linewidth=0.8, alpha=0.6)

            else:
                ax.plot(t, v, marker="o", markersize=3.5, color=color,
                        zorder=3, label=lbl)
                plotted_label = True

                # Find previous point — either a reset dummy or last real point
                prev_dummy_turn = None
                for rr in reversed(rows[:idx]):
                    if f(rr["cache_write_pct"]) > 50:
                        prev_dummy_turn = i(rr["turn"])
                        break

                if prev_dummy_turn is not None and (prev_x is None or prev_dummy_turn > prev_x):
                    # Dashed from dummy down to this real point
                    ax.plot([prev_dummy_turn, t], [DUMMY_100, v],
                            color=color, linestyle="dashed", linewidth=0.8, alpha=0.6)
                elif prev_x is not None and prev_y_real is not None:
                    # Solid line from previous real point
                    ax.plot([prev_x, t], [prev_y_real, v],
                            color=color, linewidth=1.8)

                prev_x, prev_y_real = t, v

    # Y axis: real range at bottom, dummy 100% at top
    ax.set_ylim(-1, DUMMY_100 + 3)

    real_ticks = [t for t in range(0, int(y_max_w) + 1, 5) if t <= y_max_w]
    ax.set_yticks(real_ticks + [DUMMY_100])
    ax.set_yticklabels([f"{int(t)}%" for t in real_ticks] + ["~100%\n(reset)"])

    ax.axhline(y=DUMMY_100, color="#aaa", linestyle=":", linewidth=1.0, zorder=0)

    # Broken axis markers
    gap_y = (DUMMY_100 - y_max_w) / 2 + y_max_w
    gap_frac = (gap_y - (-1)) / (DUMMY_100 + 3 - (-1))
    d = 0.015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax.plot((-d, +d), (gap_frac - d, gap_frac + d), **kwargs)
    ax.plot((-d, +d), (gap_frac - 2.5*d, gap_frac - 1.5*d), **kwargs)

    ax.set_xlabel("LLM call (turn number within session)")
    ax.set_ylabel("Cache write % of payload (new input this turn)")
    ax.set_title("Cache write % per turn — turn 1 and session restarts shown at ~100% marker\n"
                 "Spikes = turns where large tool results are the new input")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "11b_cache_write_over_turns.png")
    plt.close(fig)
    print("  Saved: 11b_cache_write_over_turns.png")


# ── Chart 12: Absolute cached vs uncached KB per session — V2 (primary) ──────
def chart_cache_absolute_kb(sessions, out, plt):
    """
    Stacked bars showing actual KB of cached vs uncached per session.
    Primary (12): V2 — tools counted as cached.
    Secondary (12b): V1 — tools counted as uncached.
    """
    if "cache_read_chars" not in sessions[0]:
        print("  Skipped: 12_cache_absolute_kb.png (no cache columns — run new analyze.py)")
        return

    import numpy as np

    sids = [r["session_id"] for r in sessions]
    x    = np.arange(len(sids))
    w    = 0.55

    # V2 chart (primary)
    cached_kb = [f(r["cache_read_chars"]) / 1024 for r in sessions]
    write_kb  = [f(r["cache_write_chars"]) / 1024 for r in sessions]
    uncach_kb = [f(r["billing_header_chars"]) / 1024 for r in sessions]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x, cached_kb, w, label="Cache read (system + tools + prior history)", color="#1D9E75", alpha=0.85)
    ax.bar(x, write_kb,  w, bottom=cached_kb,
           label="Cache write (new user input)", color="#BA7517", alpha=0.85)
    ax.bar(x, uncach_kb, w, bottom=[a+b for a,b in zip(cached_kb, write_kb)],
           label="Uncached (billing header)", color="#D85A30", alpha=0.75)
    for xi, (c, w2, u) in enumerate(zip(cached_kb, write_kb, uncach_kb)):
        ax.text(xi, c + w2 + u + 10, f"{c+w2+u:.0f}K", ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(sids)
    ax.set_ylabel("Total chars across all turns (KB)")
    ax.set_title("Cache read / write / uncached bytes per session — absolute scale\n"
                 "stacked = total payload sent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "12_cache_absolute_kb.png")
    plt.close(fig)
    print("  Saved: 12_cache_absolute_kb.png")

    # V1 chart (strict — tools uncached)
    # 12b removed — single cache definition now


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
    # cache charts — require new analyze.py output (cache_* and v1_*/v2_* columns)
    chart_cache_payload_breakdown(sessions, analysis_dir, plt)
    chart_cache_v1_vs_v2(sessions, analysis_dir, plt)
    chart_cache_pct_over_turns(turns, analysis_dir, plt)
    chart_cache_absolute_kb(sessions, analysis_dir, plt)
    print(f"\nAll charts saved to {analysis_dir}/")

if __name__ == "__main__":
    main()
