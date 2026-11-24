#!/usr/bin/env python3
"""Compares baseline vs candidate RAG evaluation results.

CLI:
    python eval/compare_runs.py --baseline eval/results/eval_baseline.json --candidate eval/results/eval_candidate.json

Loads thresholds from eval/thresholds.yaml, prints a side-by-side comparison table:
    metric | baseline | candidate | delta | pass/fail
Exits 0 if candidate passes all gates, 1 if any gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = HERE / "thresholds.yaml"


def load_metrics(path: Path | str) -> dict[str, float]:
    """Extract flat metrics dictionary from an eval result JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Result file not found: {path}")

    data = json.loads(p.read_text(encoding="utf-8"))

    # Extract from run1 or top-level
    r1 = data.get("run1", data)
    emb = r1.get("embedding", {})
    ret = r1.get("retrieval", {})
    ret_base = ret.get("baseline", ret)
    noise = ret.get("noise_test", {}).get("delta", {})
    typo = ret.get("typo_test", {}).get("delta", {})
    e2e = r1.get("e2e", {})
    stoch = data.get("stochasticity_deltas", {})

    return {
        "embedding_gap": float(emb.get("embedding_gap", 0.0)),
        "embed_time_per_chunk_sec": float(emb.get("embed_time_per_chunk_sec", 0.0)),
        "determinism_max_diff": float(emb.get("determinism_max_diff", 0.0)),
        "recall@3": float(ret_base.get("recall@3", 0.0)),
        "recall@5": float(ret_base.get("recall@5", 0.0)),
        "mrr": float(ret_base.get("mrr", 0.0)),
        "ndcg@5": float(ret_base.get("ndcg@5", 0.0)),
        "noise_recall5_delta": float(noise.get("recall@5_delta", 0.0)),
        "typo_recall5_delta": float(typo.get("recall@5_delta", 0.0)),
        "faithfulness_rate": float(e2e.get("faithfulness_rate", 0.0)),
        "abstention_rate": float(e2e.get("abstention_rate", 0.0)),
        "citation_validity": float(e2e.get("citation_validity", 0.0)),
        "latency_p50_sec": float(e2e.get("latency_p50_sec", 0.0)),
        "latency_p95_sec": float(e2e.get("latency_p95_sec", 0.0)),
        "llm_judge_agreement": float(e2e.get("llm_judge_agreement", 0.0)),
        "stochasticity_max_delta": float(max(stoch.values()) if stoch else 0.0),
    }


def compare(
    baseline_path: Path,
    candidate_path: Path,
    thresholds_path: Path = DEFAULT_THRESHOLDS,
) -> int:
    base_metrics = load_metrics(baseline_path)
    cand_metrics = load_metrics(candidate_path)

    thresholds: dict[str, Any] = {}
    if thresholds_path.is_file():
        try:
            thresholds = yaml.safe_load(thresholds_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass

    # Gate specifications: (metric_key, display_name, direction, threshold_key, default_threshold)
    # direction: ">=" means higher is better, "<=" means lower is better
    rules = [
        ("embedding_gap", "Embedding Gap", ">=", "embedding_gap_min", 0.15),
        ("embed_time_per_chunk_sec", "Embed Time/Chunk (s)", "<=", "embedding_time_per_chunk_max_sec", 0.5),
        ("determinism_max_diff", "Determinism Max Diff", "<=", "embedding_determinism_max_diff", 1e-5),
        ("recall@3", "Retrieval Recall@3", ">=", "retrieval_recall3_min", 0.50),
        ("recall@5", "Retrieval Recall@5", ">=", "retrieval_recall5_min", 0.60),
        ("mrr", "Retrieval MRR", ">=", "retrieval_mrr_min", 0.50),
        ("ndcg@5", "Retrieval nDCG@5", ">=", "retrieval_ndcg5_min", 0.50),
        ("noise_recall5_delta", "Noise Recall@5 Delta", ">=", "noise_recall5_drop_max", -0.20),
        ("typo_recall5_delta", "Typo Recall@5 Delta", ">=", "typo_recall5_drop_max", -0.25),
        ("faithfulness_rate", "Faithfulness Rate", ">=", "faithfulness_rate_min", 0.70),
        ("abstention_rate", "Abstention Rate", ">=", "abstention_rate_min", 0.80),
        ("citation_validity", "Citation Validity", ">=", "citation_validity_min", 0.85),
        ("latency_p50_sec", "Latency p50 (s)", "<=", "latency_p50_max_sec", 5.0),
        ("latency_p95_sec", "Latency p95 (s)", "<=", "latency_p95_max_sec", 12.0),
        ("llm_judge_agreement", "LLM-Judge Agreement", ">=", "llm_judge_agreement_min", 0.75),
        ("stochasticity_max_delta", "Stochasticity Delta", "<=", "stochasticity_max_delta", 0.05),
    ]

    all_passed = True
    table_rows = []

    for key, name, direction, thresh_key, default_thresh in rules:
        b_val = base_metrics.get(key, 0.0)
        c_val = cand_metrics.get(key, 0.0)
        delta = c_val - b_val

        # Get threshold value
        thresh_val = thresholds.get(thresh_key, default_thresh)
        if thresh_key == "noise_recall5_drop_max" and thresh_val > 0:
            thresh_val = -thresh_val
        if thresh_key == "typo_recall5_drop_max" and thresh_val > 0:
            thresh_val = -thresh_val

        # Evaluate gate
        is_pass = True
        if direction == ">=":
            if c_val < thresh_val:
                is_pass = False
            # Check maximum allowed drop from baseline for recall@5 (rubric gate: <= 0.05 drop)
            if key == "recall@5" and (b_val - c_val) > 0.05:
                is_pass = False
        else:
            if c_val > thresh_val:
                is_pass = False

        if not is_pass:
            all_passed = False

        status = "PASS" if is_pass else "FAIL"
        delta_str = f"{delta:+.4f}" if abs(delta) >= 1e-4 else "0.0000"

        table_rows.append({
            "name": name,
            "baseline": f"{b_val:.4f}",
            "candidate": f"{c_val:.4f}",
            "delta": delta_str,
            "status": status,
        })

    # Print Table
    print("\n" + "=" * 78)
    print(f"{'METRIC':<26} | {'BASELINE':<10} | {'CANDIDATE':<10} | {'DELTA':<10} | {'STATUS':<8}")
    print("=" * 78)
    for r in table_rows:
        print(f"{r['name']:<26} | {r['baseline']:<10} | {r['candidate']:<10} | {r['delta']:<10} | {r['status']:<8}")
    print("=" * 78 + "\n")

    if all_passed:
        print("🎉 Candidate PASSED all gate criteria relative to baseline and thresholds. (Exit code: 0)")
        return 0
    else:
        print("❌ Candidate FAILED one or more evaluation gate criteria. (Exit code: 1)")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Baseline vs Candidate RAG evaluations.")
    parser.add_argument("--baseline", required=True, help="Path to baseline evaluation JSON")
    parser.add_argument("--candidate", required=True, help="Path to candidate evaluation JSON")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS), help="Path to thresholds.yaml")
    args = parser.parse_args()

    return compare(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        thresholds_path=Path(args.thresholds),
    )


if __name__ == "__main__":
    sys.exit(main())
