"""
Claude Code Gateway Analysis
Real data from events.json — 10 sessions of captured gateway traffic.

Session detection: project_key_id — each Claude Code session gets a unique
API key in the gateway, making session boundaries exact and unambiguous.

Usage:
    python analyze.py events.json [output_dir]
"""

import json
import csv
import sys
from datetime import datetime
from pathlib import Path

# Exact session mapping verified against ground truth call counts
SESSIONS = [
    ('S01', 'key_live_c217ac302a058b956fb32dfc',  7,  'small_talk'),
    ('S02', 'key_live_c33d0b292a30832c01b57eb1',  8,  'factual_qa'),
    ('S03', 'key_live_aae7483f827ef1f268fc3d89',  7,  'single_file_read'),
    ('S04', 'key_live_70a54abaa79bbc331a505c4b', 29,  'bug_hunt'),
    ('S05', 'key_live_f5d3dc2ce387b5f4594425b5', 23,  'feature_add'),
    ('S06', 'key_live_f85bc879de8ff600b6a9ea4d', 14,  'large_input_analysis'),
    ('S07', 'key_live_efce4c7db503b86691b31c97', 34,  'refactor'),
    ('S08', 'key_live_73181a1a427c7f0288706665', 23,  'git_workflow'),
    ('S09', 'key_live_c33a1910a4ec13d98ae36217', 39,  'mixed_social_code'),
    ('S10', 'key_live_ae04405ca36f3f8692dd4947', 39,  'open_source_prep'),
]

def load_events(path):
    with open(path) as f:
        return json.load(f)

def get_user_message(preview):
    try:
        payload = json.loads(preview)
        for msg in reversed(payload.get("messages", [])):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    txt = block.get("text", "") if isinstance(block, dict) else ""
                    if txt and "system-reminder" not in txt and len(txt.strip()) > 3:
                        return txt.strip()
            elif isinstance(content, str):
                return content.strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return ""

def get_message_count(preview):
    try:
        return len(json.loads(preview).get("messages", []))
    except (json.JSONDecodeError, KeyError, TypeError):
        return 0

def build_session_events(data):
    """Return dict of project_key_id -> sorted list of STARTED events."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for e in data:
        if e["event_type"] == "MODEL_CALL_STARTED" and e["request_size_bytes"] > 10_000:
            by_key[e["project_key_id"]].append(e)
    for key in by_key:
        by_key[key].sort(key=lambda e: e["timestamp"])
    return by_key

def build_session_metrics(data, completed_by_rid, errors_by_rid):
    by_key = build_session_events(data)
    rows = []

    for sid, key, expected, label in SESSIONS:
        sess = by_key.get(key, [])
        assert len(sess) == expected, f"{sid}: expected {expected} calls, got {len(sess)}"

        req_sizes = [e["request_size_bytes"] for e in sess]
        start_ts = datetime.fromisoformat(sess[0]["timestamp"])
        end_ts = datetime.fromisoformat(sess[-1]["timestamp"])

        resp_sizes, latencies, err_count = [], [], 0
        for e in sess:
            rid = e["request_id"]
            if rid in completed_by_rid:
                resp_sizes.append(completed_by_rid[rid]["response_size_bytes"])
                latencies.append(completed_by_rid[rid]["latency_ms"])
            elif rid in errors_by_rid:
                err_count += 1

        n = len(sess)
        n_ok = len(latencies)
        baseline = req_sizes[0]
        peak = max(req_sizes)

        rows.append({
            "session_id": sid,
            "label": label,
            "project_key_id": key,
            "date": sess[0]["timestamp"][:10],
            "start_time": sess[0]["timestamp"][11:19],
            "end_time": sess[-1]["timestamp"][11:19],
            "llm_calls": n,
            "calls_completed": n_ok,
            "calls_errored": err_count,
            "error_rate_pct": round(err_count / n * 100, 1),
            "duration_s": round((end_ts - start_ts).total_seconds(), 1),
            "baseline_req_kb": round(baseline / 1024, 1),
            "peak_req_kb": round(peak / 1024, 1),
            "avg_req_kb": round(sum(req_sizes) / n / 1024, 1),
            "total_req_kb": round(sum(req_sizes) / 1024, 1),
            "total_resp_kb": round(sum(resp_sizes) / 1024, 1),
            "req_growth_kb": round((peak - baseline) / 1024, 1),
            "req_growth_pct": round((peak - baseline) / baseline * 100, 1),
            "avg_resp_kb": round(sum(resp_sizes) / n_ok / 1024, 1) if n_ok else 0,
            "req_resp_ratio": round(sum(req_sizes) / max(sum(resp_sizes), 1), 1),
            "avg_latency_ms": round(sum(latencies) / n_ok) if n_ok else 0,
            "p95_latency_ms": sorted(latencies)[int(n_ok * 0.95)] if n_ok else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "first_user_msg": get_user_message(sess[0].get("input_preview", ""))[:120],
        })
    return rows

def build_turn_metrics(data, completed_by_rid, errors_by_rid):
    by_key = build_session_events(data)
    rows = []

    for sid, key, expected, label in SESSIONS:
        sess = by_key.get(key, [])
        baseline = sess[0]["request_size_bytes"]

        for turn_idx, e in enumerate(sess):
            rid = e["request_id"]
            completed = completed_by_rid.get(rid)
            is_error = rid in errors_by_rid
            req_bytes = e["request_size_bytes"]

            rows.append({
                "session_id": sid,
                "label": label,
                "turn": turn_idx + 1,
                "timestamp": e["timestamp"][11:22],
                "request_kb": round(req_bytes / 1024, 1),
                "response_kb": round(completed["response_size_bytes"] / 1024, 1) if completed else 0,
                "growth_from_base_kb": round((req_bytes - baseline) / 1024, 1),
                "growth_from_base_pct": round((req_bytes - baseline) / baseline * 100, 1),
                "latency_ms": completed["latency_ms"] if completed else None,
                "status": "completed" if completed else ("error" if is_error else "no_response"),
                "msg_count_in_payload": get_message_count(e.get("input_preview", "")),
                "user_message": get_user_message(e.get("input_preview", ""))[:80],
            })
    return rows

def compute_summary(session_rows, turn_rows):
    total_calls = sum(r["llm_calls"] for r in session_rows)
    total_errors = sum(r["calls_errored"] for r in session_rows)
    total_req_kb = sum(r["total_req_kb"] for r in session_rows)
    total_resp_kb = sum(r["total_resp_kb"] for r in session_rows)
    latencies = sorted(r["latency_ms"] for r in turn_rows if r["latency_ms"] is not None)

    return {
        "sessions": len(session_rows),
        "total_llm_calls": total_calls,
        "total_errors": total_errors,
        "error_rate_pct": round(total_errors / total_calls * 100, 1),
        "total_req_mb": round(total_req_kb / 1024, 2),
        "total_resp_mb": round(total_resp_kb / 1024, 2),
        "overall_req_resp_ratio": round(total_req_kb / max(total_resp_kb, 1), 1),
        "avg_req_kb_per_call": round(total_req_kb / total_calls, 1),
        "median_latency_ms": latencies[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "avg_req_growth_pct": round(sum(r["req_growth_pct"] for r in session_rows) / len(session_rows), 1),
        "max_req_growth_pct": round(max(r["req_growth_pct"] for r in session_rows), 1),
        "max_req_growth_session": max(session_rows, key=lambda r: r["req_growth_pct"])["session_id"],
        "most_calls_session": max(session_rows, key=lambda r: r["llm_calls"])["session_id"],
        "highest_avg_latency_session": max(session_rows, key=lambda r: r["avg_latency_ms"])["session_id"],
    }

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")

def main():
    events_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("events.json")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("analysis")
    out_dir.mkdir(exist_ok=True)

    print(f"Loading {events_path} ...")
    events = load_events(events_path)
    print(f"  {len(events)} total events")

    completed_by_rid = {e["request_id"]: e for e in events if e["event_type"] == "MODEL_CALL_COMPLETED"}
    errors_by_rid = {e["request_id"]: e for e in events if e["event_type"] == "MODEL_CALL_ERROR"}

    print("Building session metrics (project_key_id detection) ...")
    session_rows = build_session_metrics(events, completed_by_rid, errors_by_rid)
    turn_rows = build_turn_metrics(events, completed_by_rid, errors_by_rid)

    write_csv(session_rows, out_dir / "sessions.csv")
    write_csv(turn_rows, out_dir / "turns.csv")

    summary = compute_summary(session_rows, turn_rows)
    with open(out_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Written: {out_dir / 'summary_stats.json'}")

    print("\n=== HEADLINE NUMBERS ===")
    for k, v in summary.items():
        print(f"  {k:<38} {v}")

    print("\n=== PER SESSION ===")
    print(f"{'ID':<5} {'Label':<25} {'Calls':>5} {'Err':>4} {'Base KB':>8} {'Peak KB':>8} {'Growth':>7} {'Ratio':>6} {'AvgLat':>8} {'Dur':>6}")
    print("-" * 100)
    for r in session_rows:
        print(
            f"{r['session_id']:<5} {r['label']:<25} {r['llm_calls']:>5} "
            f"{r['calls_errored']:>4} {r['baseline_req_kb']:>7.1f}K "
            f"{r['peak_req_kb']:>7.1f}K {r['req_growth_pct']:>6.0f}% "
            f"{r['req_resp_ratio']:>5.1f}x {r['avg_latency_ms']:>7.0f}ms {r['duration_s']:>5.0f}s"
        )

if __name__ == "__main__":
    main()
