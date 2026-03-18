# claude-code-gateway-telemetry

Real API gateway telemetry captured across 10 Claude Code sessions — built to answer one question: **what does the model actually receive when you type a message?**

The short answer: a lot more than you typed.

---

## What's in this repo

```
events.json              Sanitized gateway logs — 452 events, 10 sessions
analyze.py               Parses events.json → sessions.csv, turns.csv, summary_stats.json
visualize.py             Reads CSVs → 8 PNG charts
prepare_repo.py          How the raw data was sanitized (for transparency)
README.md
analysis/
  sessions.csv           Per-session summary metrics
  turns.csv              Per-LLM-call metrics (223 rows)
  summary_stats.json     Aggregate headline numbers
  01_request_growth_per_turn.png
  02_baseline_vs_peak.png
  03_request_vs_response.png
  04_history_inflation.png
  05_latency_boxplot.png
  06_latency_bars.png
  07_req_resp_ratio.png
  08_complexity_scatter.png
```

---

## Quick start

```bash
pip install matplotlib
python analyze.py events.json analysis/
python visualize.py analysis/
```

---

## Sessions

| ID | Task type | LLM calls | Baseline | Peak | Growth | Req/resp |
|----|-----------|-----------|----------|------|--------|----------|
| S01 | Small talk | 7 | 82 KB | 90 KB | +10% | 5x |
| S02 | Factual Q&A | 8 | 82 KB | 105 KB | +27% | 2x |
| S03 | Single file read | 7 | 82 KB | 106 KB | +28% | 3x |
| S04 | Bug hunt | 29 | 82 KB | 149 KB | +81% | 7x |
| S05 | Feature add | 23 | 82 KB | 132 KB | +60% | 6x |
| S06 | Large input analysis | 14 | 82 KB | 108 KB | +30% | 4x |
| S07 | Refactor | 34 | 82 KB | 197 KB | +138% | 6x |
| S08 | Git workflow | 23 | 82 KB | 105 KB | +27% | 7x |
| S09 | Mixed social + code | 39 | 82 KB | 159 KB | +91% | 6x |
| S10 | Open source prep | 39 | 82 KB | 173 KB | +109% | 8x |

> **Note on S10:** Contains a visible dip in request size at turn 21 (from ~140 KB back to ~83 KB). This is a real session restart — Claude Code was restarted mid-task due to an issue. Because both runs were launched under the same gateway project, they share the same `project_key_id` and appear together in the telemetry. This is not a bug: the gateway uses `project_key_id` to identify projects, not individual `claude` invocations. Turns 1–20 and turns 21–39 are two completely independent `claude` runs with separate context windows. The dip marks the second invocation resetting to the 82 KB baseline.

---

## Key numbers

- **223 total LLM calls** across 10 sessions
- **25.95 MB sent**, 4.46 MB received — **5.8:1 ratio**
- **82.6 KB baseline** before the user types anything
  - ~80% tool schemas (67,425 chars, ~16,856 tokens)
  - ~18% system prompt (15,245 chars, ~3,811 tokens)
- **Session 1 Turn 1**: user typed 74 characters (~18 tokens) out of 84,357 bytes total (**0.088%**)
- Median latency: **6.3s** · P95: **20.3s** · Max: **31.1s**
- Average request growth per session: **60%** from first to last call
- Max growth: **138%** (S07 refactor, 34 calls)

---

## Event schema

```json
{
  "event_type": "MODEL_CALL_STARTED | MODEL_CALL_COMPLETED | MODEL_CALL_ERROR",
  "timestamp": "2026-03-17T19:05:45.382921-07:00",
  "trace_id": "trace_0042",
  "project_id": "proj_007",
  "project_key_id": "key_007",
  "request_id": "req_0089",
  "route": "/v1/messages",
  "model": "claude-sonnet-4-5",
  "stream": true,
  "request_size_bytes": 84357,
  "debug_stored": false,
  "input_preview": "...(truncated or full JSON payload)...",
  "response_size_bytes": 28518,
  "latency_ms": 12510,
  "output_preview": "The key difference is when they execute..."
}
```

`MODEL_CALL_STARTED` events have `request_size_bytes` and `input_preview`.  
`MODEL_CALL_COMPLETED` events add `response_size_bytes`, `latency_ms`, and `output_preview`.  
Events are linked by `request_id`.

---

## Session detection

Sessions are identified by `project_key_id`. Each unique key corresponds to one Claude Code project. A single project key can contain multiple `claude` invocations if the user restarts Claude Code in the same project directory — individual invocations are detectable by a drop in `request_size_bytes` back to the ~82 KB baseline.

---

## Sanitization

The raw logs were sanitized before publishing:

| What | Replaced with |
|---|---|
| Personal email address | `researcher@example.com` |
| Local filesystem paths | `/Users/researcher/` |
| Username in shell output | `researcher` |
| Internal gateway hostname | `[internal-gateway-host]` |
| Internal IP addresses | `[internal-ip]` |
| `project_id` values | `proj_001`, `proj_002` … |
| `project_key_id` values | `key_001`, `key_002` … |
| `trace_id` values | `trace_0001`, `trace_0002` … |
| `request_id` values | `req_0001`, `req_0002` … |

No API keys, passwords, or Anthropic credentials were present in the original logs. `input_preview` and `output_preview` content is preserved except for path and username substitutions.

---

## Token count methodology

Token counts use a **characters ÷ 4** approximation — standard for Claude-class models. The exact Claude tokenizer is not publicly available so real counts will differ slightly. All byte sizes (`request_size_bytes`, `response_size_bytes`) are exact — captured directly from HTTP response headers at the gateway layer.

---

## Related article

*"Your 'Hello' Costs 20,000 Tokens: What's Really Inside Every Claude Code Request"*  
[github.com/sbt4104/LLM_GATEWAY_ANALYSIS](https://github.com/sbt4104/LLM_GATEWAY_ANALYSIS)
