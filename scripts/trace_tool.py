#!/usr/bin/env python3
"""Trace inspection CLI tool (Part D2).

Takes a trace_id, loads the corresponding JSONL trace record, prints the exact
retrieved context, rebuilds the exact prompt from the query and retrieved chunks,
verifies its SHA-256 prompt_hash, and prints the answer and guardrail actions.

CLI:
    python scripts/trace_tool.py <trace_id>
    python scripts/trace_tool.py --list
    python scripts/trace_tool.py <trace_id> --trace-file traces/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
API_DIR = REPO_ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.schemas import Passage  # noqa: E402
from app.services.tracing import build_prompt, compute_prompt_hash  # noqa: E402


def resolve_trace_file(explicit_path: str | None) -> Path:
    """Locate the trace JSONL file from explicit flag, env, repo traces/, or /data/traces/."""
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


def load_traces(trace_file: Path) -> list[dict[str, Any]]:
    """Parse JSONL trace file into a list of trace dicts."""
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
        if isinstance(obj, dict) and "trace_id" in obj:
            records.append(obj)
    return records


def find_trace_by_id(records: list[dict[str, Any]], trace_id: str) -> dict[str, Any] | None:
    """Find trace by exact match or unique prefix match."""
    for rec in records:
        if rec.get("trace_id") == trace_id:
            return rec
    prefix_matches = [rec for rec in records if str(rec.get("trace_id", "")).startswith(trace_id)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def rebuild_context_and_prompt(trace: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct Passage objects, rebuild prompt deterministically, and verify hash."""
    raw_chunks = trace.get("retrieved_chunks") or []
    passages: list[Passage] = [
        Passage(
            chunk_id=str(ch.get("chunk_id", "")),
            text=str(ch.get("text", "")),
            source=str(ch.get("source", "unknown")),
            section=str(ch.get("section", "General")),
            page=ch.get("page"),
            score=float(ch.get("score", 0.0)),
            doc_id=ch.get("doc_id"),
            doc_version=ch.get("doc_version"),
        )
        for ch in raw_chunks
    ]

    user_query = str(trace.get("user_query", ""))
    provider_name = str((trace.get("model_parameters") or {}).get("provider", "groq"))
    rebuilt_system, rebuilt_user = build_prompt(user_query, passages, provider_name=provider_name)
    rebuilt_hash = compute_prompt_hash(rebuilt_system, rebuilt_user)
    stored_hash = str(trace.get("prompt_hash", ""))

    return {
        "passages": passages,
        "system_prompt": rebuilt_system,
        "user_prompt": rebuilt_user,
        "rebuilt_hash": rebuilt_hash,
        "stored_hash": stored_hash,
        "hash_verified": bool(stored_hash and rebuilt_hash == stored_hash),
    }


def print_trace_list(records: list[dict[str, Any]]) -> None:
    """Print a compact table of recorded traces."""
    print("=" * 108)
    print(f"{'TRACE ID':<34} | {'TIMESTAMP':<24} | {'CACHE':<6} | {'ABSTAIN':<7} | {'GUARD':<8} | {'QUERY'}")
    print("-" * 108)
    for r in records:
        tid = str(r.get("trace_id", ""))
        ts = str(r.get("timestamp", ""))
        cache = "HIT" if r.get("cache_hit") else "MISS"
        abstained = str(not (r.get("response") or {}).get("answered_from_context", True))
        flagged = any(
            g.get("injection_detected") for g in (r.get("guardrail_actions") or [])
        )
        guard_str = "FLAGGED" if flagged else "PASS"
        q = str(r.get("user_query", ""))
        if len(q) > 36:
            q = q[:33] + "..."
        print(f"{tid:<34} | {ts:<24} | {cache:<6} | {abstained:<7} | {guard_str:<8} | {q}")
    print("=" * 108)


def print_reconstructed_trace(trace: dict[str, Any], rebuilt: dict[str, Any]) -> None:
    """Print formatted trace inspection including context and rebuilt prompt."""
    print("=" * 88)
    print(f"  RAHBAR TRACE INSPECTOR | trace_id: {trace.get('trace_id')}")
    print("=" * 88)
    print(f"Timestamp:          {trace.get('timestamp')}")
    print(f"User Query:         {trace.get('user_query')}")
    print(f"Rewritten Query:    {trace.get('rewritten_query')}")
    print(f"Embedding Model:    {trace.get('embedding_model')} (v{trace.get('embedding_model_version')})")
    print(f"Collection:         {trace.get('collection_name')} ({trace.get('collection_version')})")
    print(f"Model & Params:     {trace.get('model')} | {json.dumps(trace.get('model_parameters', {}))}")
    print(f"Cache Status:       hit={trace.get('cache_hit')} ({trace.get('cache_status')})")
    print(f"Filters Applied:    {json.dumps(trace.get('filters_applied', {}))}")
    print(f"Reranked Order:     {trace.get('reranked_order', [])}")
    print(f"Latency Per Stage:  {json.dumps(trace.get('latency_per_stage', {}))}")
    print(f"Guardrail Actions:  {json.dumps(trace.get('guardrail_actions', []))}")

    print("\n" + "-" * 88)
    print("RETRIEVED CONTEXT CHUNKS")
    print("-" * 88)
    chunks = trace.get("retrieved_chunks") or []
    if not chunks:
        print("(No chunks retrieved for this query — cache hit, blocked, or abstained)")
    else:
        for idx, ch in enumerate(chunks, start=1):
            print(
                f"[{idx}] chunk_id={ch.get('chunk_id')} | doc_id={ch.get('doc_id')} "
                f"| version={ch.get('doc_version')} | score={ch.get('score')} "
                f"| source={ch.get('source')} | section={ch.get('section')} | page={ch.get('page')}"
            )
            print(f"    {ch.get('text')}\n")

    print("-" * 88)
    print("REBUILT PROMPT")
    print("-" * 88)
    print(f"[SYSTEM]\n{rebuilt['system_prompt']}\n")
    print(f"[USER]\n{rebuilt['user_prompt']}")
    status_label = "VERIFIED MATCH" if rebuilt["hash_verified"] else "MISMATCH"
    print(f"Prompt Hash Verification: {status_label}")
    print(f"  Stored Hash:  {rebuilt['stored_hash']}")
    print(f"  Rebuilt Hash: {rebuilt['rebuilt_hash']}")

    print("\n" + "-" * 88)
    print("RESPONSE & CITATIONS")
    print("-" * 88)
    resp = trace.get("response") or {}
    print(f"Answered From Context: {resp.get('answered_from_context')} (confidence={resp.get('confidence')})")
    print(f"Answer:    {resp.get('answer')}")
    print(f"Citations: {json.dumps(trace.get('citations', []))}")
    print("=" * 88)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a RAHBAR query trace and rebuild its prompt.")
    parser.add_argument("trace_id", nargs="?", help="Trace ID (or unique prefix) to inspect")
    parser.add_argument("--trace-file", default=None, help="Path to traces.jsonl")
    parser.add_argument("--list", action="store_true", help="List available traces in the log")
    parser.add_argument("--json", action="store_true", help="Print the full raw JSON trace record")
    args = parser.parse_args()

    trace_file = resolve_trace_file(args.trace_file)
    try:
        records = load_traces(trace_file)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.list:
        print_trace_list(records)
        return 0

    if not args.trace_id:
        parser.error("trace_id is required unless --list is specified")

    trace = find_trace_by_id(records, args.trace_id)
    if trace is None:
        print(f"Error: trace_id '{args.trace_id}' not found in {trace_file}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(trace, indent=2, ensure_ascii=False))
        return 0

    rebuilt = rebuild_context_and_prompt(trace)
    print_reconstructed_trace(trace, rebuilt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
