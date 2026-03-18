"""
Claude Code Gateway Analysis
Real data from events.json — 10 sessions of captured gateway traffic.

Session detection: project_key_id — each Claude Code session gets a unique
API key in the gateway, making session boundaries exact and unambiguous.

Cache analysis:
  Two views of cached vs uncached content per turn:
    VIEW 1 — strict:        tools counted as UNCACHED (they have no cache_control)
    VIEW 2 — tools_cached:  tools counted as CACHED (constant across all calls)

  Categories tracked:
    system_cached      — system blocks with cache_control: ephemeral
    system_uncached    — system blocks without cache_control (billing header)
    tools              — all 25 tool schemas (always 67,425 chars, no cache_control)
    messages_cached    — message blocks with cache_control (latest boundary marker)
    messages_uncached  — message blocks without cache_control (history, reminders)

Usage:
    python analyze.py events.json [output_dir]
"""

import json
import csv
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Exact session mapping verified against ground truth call counts
SESSIONS = [
    ('S01', 'key_005',  7,  'small_talk'),
    ('S02', 'key_007',  8,  'factual_qa'),
    ('S03', 'key_003',  7,  'single_file_read'),
    ('S04', 'key_001', 29,  'bug_hunt'),
    ('S05', 'key_009', 23,  'feature_add'),
    ('S06', 'key_010', 14,  'large_input_analysis'),
    ('S07', 'key_008', 34,  'refactor'),
    ('S08', 'key_002', 23,  'git_workflow'),
    ('S09', 'key_006', 39,  'mixed_social_code'),
    ('S10', 'key_004', 39,  'open_source_prep'),
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

def parse_cache_breakdown(preview):
    """
    Parse a payload's input_preview and return char counts by cache category.

    Returns dict with keys:
      system_cached_chars      — system blocks marked cache_control: ephemeral
      system_uncached_chars    — system blocks without cache_control
      tools_chars              — all tool schemas (no cache_control, but constant)
      messages_cached_chars    — message blocks with cache_control
      messages_uncached_chars  — message blocks without cache_control
      total_preview_chars      — total length of the preview string
      parse_ok                 — bool, False if preview could not be parsed
    """
    empty = {
        'system_cached_chars': 0,
        'system_uncached_chars': 0,
        'tools_chars': 0,
        'messages_cached_chars': 0,
        'messages_uncached_chars': 0,
        'total_preview_chars': len(preview) if preview else 0,
        'parse_ok': False,
    }
    if not preview:
        return empty

    try:
        payload = json.loads(preview)
    except Exception:
        return empty

    r = dict(empty)
    r['parse_ok'] = True

    # System blocks
    for block in payload.get('system', []):
        chars = len(block.get('text', ''))
        if block.get('cache_control'):
            r['system_cached_chars'] += chars
        else:
            r['system_uncached_chars'] += chars

    # Tool schemas — always 25 tools, always 67,425 chars, never have cache_control
    for tool in payload.get('tools', []):
        r['tools_chars'] += len(json.dumps(tool))

    # Messages — walk every block in every turn
    for msg in payload.get('messages', []):
        content = msg.get('content', [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get('type', '')
            if btype == 'text':
                chars = len(block.get('text', ''))
            elif btype == 'thinking':
                chars = len(block.get('thinking', ''))
            elif btype == 'tool_use':
                chars = len(json.dumps(block.get('input', {})))
            elif btype == 'tool_result':
                c = block.get('content', '')
                chars = len(c) if isinstance(c, str) else len(json.dumps(c))
            else:
                chars = len(json.dumps(block))

            if block.get('cache_control'):
                r['messages_cached_chars'] += chars
            else:
                r['messages_uncached_chars'] += chars

    return r

def build_session_events(data):
    """Return dict of project_key_id -> sorted list of STARTED events."""
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
        if not sess:
            print(f"  WARNING: {sid} ({label}) not found — skipping")
            continue
        if len(sess) != expected:
            print(f"  WARNING: {sid} expected {expected} calls, got {len(sess)} — continuing")

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

        # Aggregate cache breakdown across all turns in session
        agg = defaultdict(int)
        parse_ok_count = 0
        for e in sess:
            cb = parse_cache_breakdown(e.get("input_preview", ""))
            if cb['parse_ok']:
                parse_ok_count += 1
                for k in ('system_cached_chars', 'system_uncached_chars', 'tools_chars',
                          'messages_cached_chars', 'messages_uncached_chars', 'total_preview_chars'):
                    agg[k] += cb[k]

        total_chars = agg['total_preview_chars']

        # VIEW 1: tools = uncached
        v1_cached   = agg['system_cached_chars'] + agg['messages_cached_chars']
        v1_uncached = agg['system_uncached_chars'] + agg['tools_chars'] + agg['messages_uncached_chars']

        # VIEW 2: tools = cached (constant content)
        v2_cached   = agg['system_cached_chars'] + agg['tools_chars'] + agg['messages_cached_chars']
        v2_uncached = agg['system_uncached_chars'] + agg['messages_uncached_chars']

        n = len(sess)
        n_ok = len(latencies)
        baseline = req_sizes[0]
        peak = max(req_sizes)

        rows.append({
            "session_id":                   sid,
            "label":                        label,
            "project_key_id":               key,
            "date":                         sess[0]["timestamp"][:10],
            "start_time":                   sess[0]["timestamp"][11:19],
            "end_time":                     sess[-1]["timestamp"][11:19],
            "llm_calls":                    n,
            "calls_completed":              n_ok,
            "calls_errored":                err_count,
            "error_rate_pct":               round(err_count / n * 100, 1),
            "duration_s":                   round((end_ts - start_ts).total_seconds(), 1),
            "baseline_req_kb":              round(baseline / 1024, 1),
            "peak_req_kb":                  round(peak / 1024, 1),
            "avg_req_kb":                   round(sum(req_sizes) / n / 1024, 1),
            "total_req_kb":                 round(sum(req_sizes) / 1024, 1),
            "total_resp_kb":                round(sum(resp_sizes) / 1024, 1),
            "req_growth_kb":                round((peak - baseline) / 1024, 1),
            "req_growth_pct":               round((peak - baseline) / baseline * 100, 1),
            "avg_resp_kb":                  round(sum(resp_sizes) / n_ok / 1024, 1) if n_ok else 0,
            "req_resp_ratio":               round(sum(req_sizes) / max(sum(resp_sizes), 1), 1),
            "avg_latency_ms":               round(sum(latencies) / n_ok) if n_ok else 0,
            "p95_latency_ms":               sorted(latencies)[int(n_ok * 0.95)] if n_ok else 0,
            "max_latency_ms":               max(latencies) if latencies else 0,
            "first_user_msg":               get_user_message(sess[0].get("input_preview", ""))[:120],
            # cache breakdown — totals across all turns (chars)
            "cache_system_cached_chars":    agg['system_cached_chars'],
            "cache_system_uncached_chars":  agg['system_uncached_chars'],
            "cache_tools_chars":            agg['tools_chars'],
            "cache_messages_cached_chars":  agg['messages_cached_chars'],
            "cache_messages_uncached_chars":agg['messages_uncached_chars'],
            "cache_total_chars":            total_chars,
            "cache_turns_parsed":           parse_ok_count,
            # view 1: tools uncached
            "v1_cached_chars":              v1_cached,
            "v1_uncached_chars":            v1_uncached,
            "v1_cached_pct":               round(v1_cached / max(total_chars, 1) * 100, 1),
            "v1_uncached_pct":             round(v1_uncached / max(total_chars, 1) * 100, 1),
            # view 2: tools cached
            "v2_cached_chars":              v2_cached,
            "v2_uncached_chars":            v2_uncached,
            "v2_cached_pct":               round(v2_cached / max(total_chars, 1) * 100, 1),
            "v2_uncached_pct":             round(v2_uncached / max(total_chars, 1) * 100, 1),
        })
    return rows

def build_turn_metrics(data, completed_by_rid, errors_by_rid):
    by_key = build_session_events(data)
    rows = []

    for sid, key, expected, label in SESSIONS:
        sess = by_key.get(key, [])
        if not sess:
            continue
        baseline = sess[0]["request_size_bytes"]

        for turn_idx, e in enumerate(sess):
            rid = e["request_id"]
            completed = completed_by_rid.get(rid)
            is_error = rid in errors_by_rid
            req_bytes = e["request_size_bytes"]

            cb = parse_cache_breakdown(e.get("input_preview", ""))
            total_chars = cb['total_preview_chars']

            # view 1: tools uncached
            v1_cached   = cb['system_cached_chars'] + cb['messages_cached_chars']
            v1_uncached = cb['system_uncached_chars'] + cb['tools_chars'] + cb['messages_uncached_chars']

            # view 2: tools cached
            v2_cached   = cb['system_cached_chars'] + cb['tools_chars'] + cb['messages_cached_chars']
            v2_uncached = cb['system_uncached_chars'] + cb['messages_uncached_chars']

            rows.append({
                "session_id":                   sid,
                "label":                        label,
                "turn":                         turn_idx + 1,
                "timestamp":                    e["timestamp"][11:22],
                "request_kb":                   round(req_bytes / 1024, 1),
                "response_kb":                  round(completed["response_size_bytes"] / 1024, 1) if completed else 0,
                "growth_from_base_kb":          round((req_bytes - baseline) / 1024, 1),
                "growth_from_base_pct":         round((req_bytes - baseline) / baseline * 100, 1),
                "latency_ms":                   completed["latency_ms"] if completed else None,
                "status":                       "completed" if completed else ("error" if is_error else "no_response"),
                "msg_count_in_payload":         get_message_count(e.get("input_preview", "")),
                "user_message":                 get_user_message(e.get("input_preview", ""))[:80],
                # cache breakdown — this turn only (chars)
                "cache_system_cached_chars":    cb['system_cached_chars'],
                "cache_system_uncached_chars":  cb['system_uncached_chars'],
                "cache_tools_chars":            cb['tools_chars'],
                "cache_messages_cached_chars":  cb['messages_cached_chars'],
                "cache_messages_uncached_chars":cb['messages_uncached_chars'],
                "cache_total_chars":            total_chars,
                "cache_parse_ok":               cb['parse_ok'],
                # view 1: tools uncached
                "v1_cached_chars":              v1_cached,
                "v1_uncached_chars":            v1_uncached,
                "v1_cached_pct":               round(v1_cached / max(total_chars, 1) * 100, 1),
                "v1_uncached_pct":             round(v1_uncached / max(total_chars, 1) * 100, 1),
                # view 2: tools cached
                "v2_cached_chars":              v2_cached,
                "v2_uncached_chars":            v2_uncached,
                "v2_cached_pct":               round(v2_cached / max(total_chars, 1) * 100, 1),
                "v2_uncached_pct":             round(v2_uncached / max(total_chars, 1) * 100, 1),
            })
    return rows

def compute_summary(session_rows, turn_rows):
    total_calls  = sum(r["llm_calls"] for r in session_rows)
    total_errors = sum(r["calls_errored"] for r in session_rows)
    total_req_kb = sum(r["total_req_kb"] for r in session_rows)
    total_resp_kb = sum(r["total_resp_kb"] for r in session_rows)
    latencies = sorted(r["latency_ms"] for r in turn_rows if r["latency_ms"] is not None)

    parsed = [r for r in turn_rows if r["cache_parse_ok"]]
    total_chars  = sum(r["cache_total_chars"] for r in parsed)
    sys_cached   = sum(r["cache_system_cached_chars"] for r in parsed)
    sys_uncached = sum(r["cache_system_uncached_chars"] for r in parsed)
    tools        = sum(r["cache_tools_chars"] for r in parsed)
    msg_cached   = sum(r["cache_messages_cached_chars"] for r in parsed)
    msg_uncached = sum(r["cache_messages_uncached_chars"] for r in parsed)

    v1_cached   = sys_cached + msg_cached
    v1_uncached = sys_uncached + tools + msg_uncached
    v2_cached   = sys_cached + tools + msg_cached
    v2_uncached = sys_uncached + msg_uncached

    return {
        "sessions":                         len(session_rows),
        "total_llm_calls":                  total_calls,
        "total_errors":                     total_errors,
        "error_rate_pct":                   round(total_errors / total_calls * 100, 1),
        "total_req_mb":                     round(total_req_kb / 1024, 2),
        "total_resp_mb":                    round(total_resp_kb / 1024, 2),
        "overall_req_resp_ratio":           round(total_req_kb / max(total_resp_kb, 1), 1),
        "avg_req_kb_per_call":              round(total_req_kb / total_calls, 1),
        "median_latency_ms":                latencies[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms":                   latencies[int(len(latencies) * 0.95)] if latencies else 0,
        "max_latency_ms":                   max(latencies) if latencies else 0,
        "avg_req_growth_pct":               round(sum(r["req_growth_pct"] for r in session_rows) / len(session_rows), 1),
        "max_req_growth_pct":               round(max(r["req_growth_pct"] for r in session_rows), 1),
        "max_req_growth_session":           max(session_rows, key=lambda r: r["req_growth_pct"])["session_id"],
        "most_calls_session":               max(session_rows, key=lambda r: r["llm_calls"])["session_id"],
        "highest_avg_latency_session":      max(session_rows, key=lambda r: r["avg_latency_ms"])["session_id"],
        "cache_system_cached_chars":        sys_cached,
        "cache_system_uncached_chars":      sys_uncached,
        "cache_tools_chars":                tools,
        "cache_messages_cached_chars":      msg_cached,
        "cache_messages_uncached_chars":    msg_uncached,
        "cache_total_chars":                total_chars,
        "v1_cached_chars":                  v1_cached,
        "v1_uncached_chars":                v1_uncached,
        "v1_cached_pct":                   round(v1_cached / max(total_chars, 1) * 100, 1),
        "v1_uncached_pct":                 round(v1_uncached / max(total_chars, 1) * 100, 1),
        "v2_cached_chars":                  v2_cached,
        "v2_uncached_chars":                v2_uncached,
        "v2_cached_pct":                   round(v2_cached / max(total_chars, 1) * 100, 1),
        "v2_uncached_pct":                 round(v2_uncached / max(total_chars, 1) * 100, 1),
    }

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")

def print_cache_block(sys_cached, sys_uncached, tools, msg_cached, msg_uncached, total):
    tc = max(total, 1)
    print(f"    {'Category':<30} {'Chars':>12} {'%':>7}")
    print(f"    {'-'*51}")
    print(f"    {'system  (cached)':30} {sys_cached:>12,}  {sys_cached/tc*100:>6.1f}%")
    print(f"    {'system  (uncached)':30} {sys_uncached:>12,}  {sys_uncached/tc*100:>6.1f}%")
    print(f"    {'tools   (no cache_control)':30} {tools:>12,}  {tools/tc*100:>6.1f}%")
    print(f"    {'messages (cached)':30} {msg_cached:>12,}  {msg_cached/tc*100:>6.1f}%")
    print(f"    {'messages (uncached)':30} {msg_uncached:>12,}  {msg_uncached/tc*100:>6.1f}%")
    print(f"    {'-'*51}")
    v1c = sys_cached + msg_cached
    v1u = sys_uncached + tools + msg_uncached
    v2c = sys_cached + tools + msg_cached
    v2u = sys_uncached + msg_uncached
    print(f"    VIEW 1 — tools as UNCACHED (strict):")
    print(f"      cached:   {v1c:>12,}  {v1c/tc*100:>6.1f}%")
    print(f"      uncached: {v1u:>12,}  {v1u/tc*100:>6.1f}%")
    print(f"    VIEW 2 — tools as CACHED (constant content):")
    print(f"      cached:   {v2c:>12,}  {v2c/tc*100:>6.1f}%")
    print(f"      uncached: {v2u:>12,}  {v2u/tc*100:>6.1f}%")

def main():
    events_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("events.json")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("analysis")
    out_dir.mkdir(exist_ok=True)

    print(f"Loading {events_path} ...")
    events = load_events(events_path)
    print(f"  {len(events)} total events")

    completed_by_rid = {e["request_id"]: e for e in events if e["event_type"] == "MODEL_CALL_COMPLETED"}
    errors_by_rid = {e["request_id"]: e for e in events if e["event_type"] == "MODEL_CALL_ERROR"}

    print("Building session metrics ...")
    session_rows = build_session_metrics(events, completed_by_rid, errors_by_rid)
    turn_rows = build_turn_metrics(events, completed_by_rid, errors_by_rid)

    write_csv(session_rows, out_dir / "sessions.csv")
    write_csv(turn_rows, out_dir / "turns.csv")

    summary = compute_summary(session_rows, turn_rows)
    with open(out_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Written: {out_dir / 'summary_stats.json'}")

    print("\n=== HEADLINE NUMBERS ===")
    headline_keys = [
        "sessions", "total_llm_calls", "total_errors", "error_rate_pct",
        "total_req_mb", "total_resp_mb", "overall_req_resp_ratio",
        "avg_req_kb_per_call", "median_latency_ms", "p95_latency_ms", "max_latency_ms",
        "avg_req_growth_pct", "max_req_growth_pct", "max_req_growth_session",
        "most_calls_session", "highest_avg_latency_session",
    ]
    for k in headline_keys:
        print(f"  {k:<38} {summary[k]}")

    print("\n=== CACHE ANALYSIS — AGGREGATE (all sessions, all turns) ===")
    print_cache_block(
        summary["cache_system_cached_chars"],
        summary["cache_system_uncached_chars"],
        summary["cache_tools_chars"],
        summary["cache_messages_cached_chars"],
        summary["cache_messages_uncached_chars"],
        summary["cache_total_chars"],
    )

    print("\n=== CACHE ANALYSIS — PER SESSION ===")
    print(f"{'ID':<5} {'Label':<22} {'Turns':>5}  {'SysCach%':>8} {'Tools%':>7} {'MsgCach%':>9} {'MsgUnc%':>8}  {'V1cach%':>7} {'V2cach%':>7}")
    print("-" * 85)
    for r in session_rows:
        tc = max(r["cache_total_chars"], 1)
        print(
            f"{r['session_id']:<5} {r['label']:<22} {r['llm_calls']:>5}  "
            f"{r['cache_system_cached_chars']/tc*100:>7.1f}%"
            f"{r['cache_tools_chars']/tc*100:>8.1f}%"
            f"{r['cache_messages_cached_chars']/tc*100:>9.1f}%"
            f"{r['cache_messages_uncached_chars']/tc*100:>9.1f}%"
            f"  {r['v1_cached_pct']:>7.1f}%"
            f"  {r['v2_cached_pct']:>7.1f}%"
        )

    print("\n=== PER SESSION (size metrics) ===")
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
