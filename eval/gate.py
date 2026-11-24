#!/usr/bin/env python3
"""Release Gate for RAHBAR RAG pipeline candidates.

CLI:
    python eval/gate.py --baseline <json> --candidate <json> --thresholds eval/thresholds.yaml

Evaluates candidate metrics against baseline and hard thresholds:
  1. candidate recall@5 drop vs baseline <= 2.0 points
  2. candidate faithfulness >= 0.75
  3. candidate p95 latency <= 3000 ms
  4. candidate abstention correctness >= 1.0 (and >= baseline)

Prints PASS/FAIL per check and overall result.
Exits 0 on overall PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def extract_metrics(path: Path | str) -> dict[str, float]:
    """Extract standard metrics from eval results JSON."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Result file not found: {path}")

    data = json.loads(p.read_text(encoding="utf-8"))

    # Support nested run1 or flat dictionary
    r = data.get("run1", data)
    ret = r.get("retrieval", {})
    ret_base = ret.get("baseline", ret)
    e2e = r.get("e2e", {})

    # 1. Recall@5
    recall_5 = float(
        ret_base.get("recall@5")
        or r.get("recall@5")
        or data.get("recall@5", 0.0)
    )

    # 2. Faithfulness
    faithfulness = float(
        e2e.get("faithfulness_rate")
        or r.get("faithfulness_rate")
        or r.get("faithfulness")
        or data.get("faithfulness_rate", 0.0)
    )

    # 3. p95 Latency (normalize to milliseconds)
    raw_p95 = (
        e2e.get("latency_p95_sec")
        or e2e.get("latency_p95_ms")
        or r.get("latency_p95_sec")
        or data.get("latency_p95_sec")
        or 0.0
    )
    raw_p95 = float(raw_p95)
    # If in seconds (<= 50.0), convert to milliseconds
    p95_ms = raw_p95 * 1000.0 if raw_p95 <= 50.0 else raw_p95

    # 4. Abstention Correctness
    abstention = float(
        e2e.get("abstention_rate")
        or r.get("abstention_correctness")
        or r.get("abstention_rate")
        or data.get("abstention_rate", 0.0)
    )

    return {
        "recall@5": recall_5,
        "faithfulness": faithfulness,
        "p95_latency_ms": p95_ms,
        "abstention_correctness": abstention,
    }


def run_gate(baseline_path: Path, candidate_path: Path, thresholds_path: Path) -> int:
    base = extract_metrics(baseline_path)
    cand = extract_metrics(candidate_path)

    thresholds: dict[str, Any] = {}
    if thresholds_path and thresholds_path.is_file():
        thresholds = yaml.safe_load(thresholds_path.read_text(encoding="utf-8")) or {}

    max_recall_drop_pts = float(os.getenv("RECALL_AT_5_MIN_DROP", thresholds.get("recall_at_5_min_drop", 2.0)))
    min_faithfulness = float(os.getenv("FAITHFULNESS_MIN", thresholds.get("faithfulness_min", 0.75)))
    max_p95_ms = float(os.getenv("P95_LATENCY_MAX_MS", thresholds.get("p95_latency_max_ms", 3000.0)))
    min_abstention = float(os.getenv("ABSTENTION_CORRECTNESS_MIN", thresholds.get("abstention_correctness_min", 1.0)))

    # Normalize recall scales: if 0.0-1.0 scale, 2.0 points is 0.02
    b_recall = base["recall@5"]
    c_recall = cand["recall@5"]
    scale_factor = 100.0 if b_recall <= 1.0 else 1.0
    recall_drop_pts = (b_recall - c_recall) * scale_factor

    checks = []

    # Check 1: Recall@5 drop vs baseline
    # candidate recall@5 must not drop more than 2 points vs baseline
    recall_pass = recall_drop_pts <= max_recall_drop_pts
    b_rec_pct = b_recall * (100.0 if b_recall <= 1.0 else 1.0)
    c_rec_pct = c_recall * (100.0 if c_recall <= 1.0 else 1.0)
    checks.append({
        "name": "Recall@5 Drop",
        "baseline": f"{b_rec_pct:.1f}%",
        "candidate": f"{c_rec_pct:.1f}%",
        "observed": f"{recall_drop_pts:+.2f} pts drop",
        "criterion": f"<= {max_recall_drop_pts:.1f} pts drop",
        "passed": recall_pass,
    })

    # Check 2: Faithfulness minimum
    c_faith = cand["faithfulness"]
    faith_pass = c_faith >= min_faithfulness
    checks.append({
        "name": "Faithfulness",
        "baseline": f"{base['faithfulness']:.2f}",
        "candidate": f"{c_faith:.2f}",
        "observed": f"{c_faith:.2f}",
        "criterion": f">= {min_faithfulness:.2f}",
        "passed": faith_pass,
    })

    # Check 3: p95 Latency ceiling
    c_p95 = cand["p95_latency_ms"]
    p95_pass = c_p95 <= max_p95_ms
    checks.append({
        "name": "p95 Latency",
        "baseline": f"{base['p95_latency_ms']:.0f} ms",
        "candidate": f"{c_p95:.0f} ms",
        "observed": f"{c_p95:.0f} ms",
        "criterion": f"<= {max_p95_ms:.0f} ms",
        "passed": p95_pass,
    })

    # Check 4: Abstention correctness
    c_abstain = cand["abstention_correctness"]
    abstain_pass = (c_abstain >= min_abstention) and (c_abstain >= base["abstention_correctness"])
    checks.append({
        "name": "Abstention Correctness",
        "baseline": f"{base['abstention_correctness']:.2f}",
        "candidate": f"{c_abstain:.2f}",
        "observed": f"{c_abstain:.2f}",
        "criterion": f">= {min_abstention:.2f} (and >= baseline)",
        "passed": abstain_pass,
    })

    # Print Table
    print("\n" + "=" * 80)
    print("  RAHBAR RELEASE GATE VERIFICATION")
    print(f"  Baseline:  {baseline_path}")
    print(f"  Candidate: {candidate_path}")
    print("=" * 80)
    print(f"{'CHECK':<24} | {'BASELINE':<10} | {'CANDIDATE':<10} | {'CRITERION':<20} | {'STATUS'}")
    print("-" * 80)
    for chk in checks:
        status_str = "PASS" if chk["passed"] else "FAIL"
        print(f"{chk['name']:<24} | {chk['baseline']:<10} | {chk['candidate']:<10} | {chk['criterion']:<20} | {status_str}")
    print("=" * 80)

    overall_pass = all(chk["passed"] for chk in checks)

    if overall_pass:
        print("\n>>> OVERALL RESULT: PASS <<<")
        print("Candidate meets all production release criteria.\n")
        return 0
    else:
        print("\n>>> OVERALL RESULT: FAIL <<<")
        failed_names = [chk["name"] for chk in checks if not chk["passed"]]
        print(f"Candidate rejected: failed {', '.join(failed_names)}.\n")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAHBAR release gate.")
    parser.add_argument("--baseline", required=True, help="Baseline eval results JSON")
    parser.add_argument("--candidate", required=True, help="Candidate eval results JSON")
    parser.add_argument("--thresholds", default="eval/thresholds.yaml", help="Path to thresholds.yaml")
    args = parser.parse_args()

    return run_gate(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        thresholds_path=Path(args.thresholds),
    )


if __name__ == "__main__":
    sys.exit(main())
