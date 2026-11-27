#!/usr/bin/env python3
"""Trace log summarizer over a configurable time window (Part D3).

Reports:
  - Top-1 retrieval similarity distribution (min, p25, p50, p75, p95, max, mean, histogram)
  - Abstention rate
  - Cache hit rate
  - p50 & p95 latency per stage (cache_lookup, retrieval, guardrail, generation, total)
  - Number of prompt injection attempts flagged

CLI:
    python scripts/summarize_traces.py
    python scripts/summarize_traces.py --trace-file traces/traces.jsonl --last-hours 24
    python scripts/summarize_traces.py --since 2026-11-27T00:00:00Z --until 2026-11-28T00:00:00Z
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_trace_file(explicit_path: str | None) -> Path:
    """Resolve the traces JSONL path."""
    if explicit_path:
        return Path(explicit_path)
    env_dir = os.environ.get("TRACE_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir) / "traces.jsonl")
    candidates.extend([
        REPO_ROOT / "traces" / "traces.jsonl",
        Path("/data/traces/traces.jsonl"),
    ])
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[0]


def parse_ts(ts_str: str) -> datetime:
    """Parse ISO-8601 timestamp string (supporting trailing Z) into UTC datetime."""
    cleaned = ts_str.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_traces_in_window(
    trace_file: Path,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[dict[str, Any]]:
    """Load trace records filtered by optional [since, until] UTC window."""
    if not trace_file.is_file():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    records: list[dict[str, Any]] = []
    for raw_line in trace_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "trace_id" not in obj:
            continue

        ts_raw = obj.get("timestamp")
        if (since or until) and ts_raw:
            ts = parse_ts(str(ts_raw))
            if since and ts < since:
                continue
            if until and ts > until:
                continue
        records.append(obj)
    return records


def summarize_traces(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute D3 operational metrics from a list of trace records."""
    total = len(records)
    if total == 0:
        return {
            "total_queries": 0,
            "window_start": None,
            "window_end": None,
            "top1_similarity_distribution": {},
            "abstention_rate": 0.0,
            "abstained_count": 0,
            "cache_hit_rate": 0.0,
            "cache_hit_count": 0,
            "latency_ms": {},
            "injection_attempts_flagged": 0,
            "injection_flagged_queries": 0,
            "flagged_trace_ids": [],
        }

    timestamps = [str(r["timestamp"]) for r in records if r.get("timestamp")]
    window_start = min(timestamps) if timestamps else None
    window_end = max(timestamps) if timestamps else None

    top1_scores: list[float] = []
    abstained_count = 0
    cache_hit_count = 0
    injection_events = 0
    injection_queries = 0
    flagged_trace_ids: list[str] = []

    stage_names = ("cache_lookup_ms", "retrieval_ms", "guardrail_ms", "generation_ms", "total_ms")
    stage_latencies: dict[str, list[float]] = {s: [] for s in stage_names}

    for rec in records:
        chunks = rec.get("retrieved_chunks") or []
        if chunks:
            scores = [float(c.get("score", 0.0)) for c in chunks]
            top1_scores.append(max(scores))

        resp = rec.get("response") or {}
        if not resp.get("answered_from_context", True):
            abstained_count += 1

        if rec.get("cache_hit"):
            cache_hit_count += 1

        lat = rec.get("latency_per_stage") or {}
        for s in stage_names:
            if s in lat and lat[s] is not None:
                stage_latencies[s].append(float(lat[s]))

        guard_actions = rec.get("guardrail_actions") or []
        rec_injections = sum(1 for g in guard_actions if g.get("injection_detected"))
        if rec_injections > 0:
            injection_events += rec_injections
            injection_queries += 1
            flagged_trace_ids.append(str(rec.get("trace_id", "")))

    # Top-1 retrieval similarity distribution & buckets
    if top1_scores:
        arr = np.array(top1_scores, dtype=float)
        buckets = {
            "<0.30": int(np.sum(arr < 0.30)),
            "0.30-0.50": int(np.sum((arr >= 0.30) & (arr < 0.50))),
            "0.50-0.70": int(np.sum((arr >= 0.50) & (arr < 0.70))),
            "0.70-0.85": int(np.sum((arr >= 0.70) & (arr < 0.85))),
            ">=0.85": int(np.sum(arr >= 0.85)),
        }
        top1_dist = {
            "count": len(top1_scores),
            "min": round(float(np.min(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "max": round(float(np.max(arr)), 4),
            "mean": round(float(np.mean(arr)), 4),
            "histogram": buckets,
        }
    else:
        top1_dist = {"count": 0, "histogram": {}}

    latency_summary: dict[str, dict[str, float]] = {}
    for s, vals in stage_latencies.items():
        if vals:
            arr = np.array(vals, dtype=float)
            latency_summary[s] = {
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "max_ms": round(float(np.max(arr)), 3),
            }
        else:
            latency_summary[s] = {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

    return {
        "total_queries": total,
        "window_start": window_start,
        "window_end": window_end,
        "top1_similarity_distribution": top1_dist,
        "abstention_rate": round(abstained_count / total, 4),
        "abstained_count": abstained_count,
        "cache_hit_rate": round(cache_hit_count / total, 4),
        "cache_hit_count": cache_hit_count,
        "latency_ms": latency_summary,
        "injection_attempts_flagged": injection_events,
        "injection_flagged_queries": injection_queries,
        "flagged_trace_ids": flagged_trace_ids,
    }


def print_summary(summary: dict[str, Any], trace_file: Path) -> None:
    """Print human-readable D3 operational metrics table."""
    print("=" * 80)
    print("  RAHBAR OBSERVABILITY & TRACE SUMMARY (Part D3)")
    print(f"  Source Log: {trace_file}")
    print(f"  Window:     {summary['window_start']}  ->  {summary['window_end']}")
    print("=" * 80)
    print(f"  Total Queries:              {summary['total_queries']}")
    print(
        f"  Abstention Rate:            {summary['abstention_rate'] * 100:.1f}% "
        f"({summary['abstained_count']} / {summary['total_queries']})"
    )
    print(
        f"  Cache Hit Rate:             {summary['cache_hit_rate'] * 100:.1f}% "
        f"({summary['cache_hit_count']} / {summary['total_queries']})"
    )
    print(
        f"  Injection Attempts Flagged: {summary['injection_attempts_flagged']} event(s) "
        f"across {summary['injection_flagged_queries']} query/queries"
    )
    if summary["flagged_trace_ids"]:
        print(f"  Flagged Trace IDs:          {', '.join(summary['flagged_trace_ids'])}")

    print("\n" + "-" * 80)
    print("TOP-1 RETRIEVAL SIMILARITY DISTRIBUTION")
    print("-" * 80)
    dist = summary["top1_similarity_distribution"]
    if dist.get("count", 0) == 0:
        print("  (No retrieved chunks recorded in window)")
    else:
        print(
            f"  n={dist['count']} | min={dist['min']:.4f} | p25={dist['p25']:.4f} | "
            f"p50={dist['p50']:.4f} | p75={dist['p75']:.4f} | p95={dist['p95']:.4f} | "
            f"max={dist['max']:.4f} | mean={dist['mean']:.4f}"
        )
        print(f"  Buckets: {json.dumps(dist['histogram'])}")

    print("\n" + "-" * 80)
    print("LATENCY PER STAGE (ms)")
    print("-" * 80)
    print(f"  {'STAGE':<20} | {'p50 (ms)':>12} | {'p95 (ms)':>12} | {'max (ms)':>12}")
    print(f"  {'-' * 20}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")
    for stage, vals in summary["latency_ms"].items():
        print(
            f"  {stage:<20} | {vals['p50_ms']:>12.3f} | {vals['p95_ms']:>12.3f} | {vals['max_ms']:>12.3f}"
        )
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize RAHBAR query traces over a time window.")
    parser.add_argument("--trace-file", default=None, help="Path to traces.jsonl")
    parser.add_argument("--since", default=None, help="Start ISO-8601 UTC timestamp")
    parser.add_argument("--until", default=None, help="End ISO-8601 UTC timestamp")
    parser.add_argument("--last-minutes", type=float, default=None, help="Filter traces from last N minutes")
    parser.add_argument("--last-hours", type=float, default=None, help="Filter traces from last N hours")
    parser.add_argument("--json", action="store_true", help="Output summary as JSON")
    parser.add_argument("--output", default=None, help="Optional file path to save JSON summary")
    args = parser.parse_args()

    since_dt: datetime | None = parse_ts(args.since) if args.since else None
    until_dt: datetime | None = parse_ts(args.until) if args.until else None

    now_utc = datetime.now(timezone.utc)
    if args.last_minutes is not None:
        since_dt = now_utc - timedelta(minutes=args.last_minutes)
    elif args.last_hours is not None:
        since_dt = now_utc - timedelta(hours=args.last_hours)

    trace_file = resolve_trace_file(args.trace_file)
    try:
        records = load_traces_in_window(trace_file, since=since_dt, until=until_dt)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary = summarize_traces(records)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary, trace_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
