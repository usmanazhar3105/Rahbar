#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set, and gate CI on the result (section 5).

    python eval/run_eval.py --api http://<ingress>                 # your corpus
    python eval/run_eval.py --api ... --golden eval/fixtures/golden.fixture.yaml
    python eval/run_eval.py --api ... --update-baseline            # after a deliberate improvement

Exit code 0 when every gate passes, 1 when any fails. CI relies on that exit code.

Given: talking to the API, reading the golden set, writing results.json, the exit
code. Implemented: the four functions recall_at_k, citation_validity, summarise, and gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent

# The gate from section 5.4. Change these only with a reason written in docs/PIPELINE.md.
MAX_RECALL5_DROP = 0.05        # recall@5 may fall at most 5 points below the baseline
MIN_CITATION_VALIDITY = 0.95   # at least 95% of citations must match retrieved passages
REQUIRED_ABSTENTION = 1.0      # every must-abstain question must be declined


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        sys.exit(f"no questions in {path}")
    return questions


def ask(client: httpx.Client, api: str, question: str) -> dict:
    """POST one question and return the AskResponse as a dict (see api/app/schemas.py)."""
    endpoint = f"{api.rstrip('/')}/api/ask"
    try:
        response = client.post(endpoint, json={"question": question})
        if response.status_code == 404:
            response = client.post(f"{api.rstrip('/')}/api/chat", json={"question": question})
    except httpx.HTTPError:
        response = client.post(f"{api.rstrip('/')}/api/chat", json={"question": question})
    response.raise_for_status()
    return response.json()


def _match_source(s1: str, s2: str) -> bool:
    if not s1 or not s2:
        return False
    p1 = Path(s1).name.lower()
    p2 = Path(s2).name.lower()
    if p1 == p2:
        return True
    stem1 = Path(p1).stem.replace("-", "_").lower()
    stem2 = Path(p2).stem.replace("-", "_").lower()
    return stem1 == stem2 or p1 in p2 or p2 in p1 or stem1 in stem2 or stem2 in stem1


def _match_section(s1: str, s2: str) -> bool:
    sec1 = (s1 or "").strip().lower()
    sec2 = (s2 or "").strip().lower()
    if not sec1 or not sec2:
        return True
    if sec1 == sec2 or sec1 in sec2 or sec2 in sec1:
        return True
    c1 = sec1.split()[0].rstrip(".:")
    c2 = sec2.split()[0].rstrip(".:")
    if c1 and c2 and c1 == c2:
        return True
    return False


def recall_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Fraction of the relevant (source, section) pairs found in the top k retrieved.
    Return None when the question has no relevant passages (must-abstain questions)."""
    if not relevant:
        return None
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    matched = 0
    for rel in relevant:
        rel_src = rel.get("source", "")
        rel_sec = rel.get("section", "")
        for ret in top_k:
            if _match_source(rel_src, ret.get("source", "")) and _match_section(rel_sec, ret.get("section", "")):
                matched += 1
                break

    return matched / len(relevant)


def citation_validity(response: dict) -> float | None:
    """Fraction of the response's citations that match a passage in response['retrieved'].
    Return None when there are no citations."""
    citations = response.get("citations")
    if not citations:
        return None

    retrieved = response.get("retrieved") or []
    valid = 0
    for cit in citations:
        cit_src = cit.get("source", "")
        cit_sec = cit.get("section", "")
        matched = False
        for ret in retrieved:
            if _match_source(cit_src, ret.get("source", "")) and _match_section(cit_sec, ret.get("section", "")):
                matched = True
                break
        if matched:
            valid += 1

    return valid / len(citations)


def summarise(rows: list[dict]) -> dict:
    """Overall recall@3, recall@5, citation validity and abstention correctness, plus
    recall@5 broken down by question type. This dict becomes results.json."""
    r3_scores: list[float] = []
    r5_scores: list[float] = []
    cit_scores: list[float] = []
    by_type: dict[str, list[float]] = {}
    type_counts: dict[str, int] = {}

    abstain_total = 0
    abstain_correct = 0

    for row in rows:
        q_type = row.get("type", "ordinary")
        type_counts[q_type] = type_counts.get(q_type, 0) + 1

        resp = row.get("response") or {}
        retrieved = resp.get("retrieved") or []
        relevant = row.get("relevant") or []

        r3 = recall_at_k(relevant, retrieved, 3)
        if r3 is not None:
            r3_scores.append(r3)

        r5 = recall_at_k(relevant, retrieved, 5)
        if r5 is not None:
            r5_scores.append(r5)
            by_type.setdefault(q_type, []).append(r5)

        cv = citation_validity(resp)
        if cv is not None:
            cit_scores.append(cv)

        if row.get("must_abstain", False) or not relevant:
            abstain_total += 1
            if resp.get("answered_from_context") is False:
                abstain_correct += 1

    mean_r3 = sum(r3_scores) / len(r3_scores) if r3_scores else 0.0
    mean_r5 = sum(r5_scores) / len(r5_scores) if r5_scores else 0.0
    mean_cit = sum(cit_scores) / len(cit_scores) if cit_scores else 1.0
    abst_corr = (abstain_correct / abstain_total) if abstain_total > 0 else 1.0

    by_type_summary: dict[str, dict] = {}
    for q_type, count in type_counts.items():
        type_r5_list = by_type.get(q_type, [])
        type_r5 = (sum(type_r5_list) / len(type_r5_list)) if type_r5_list else None
        by_type_summary[q_type] = {
            "count": count,
            "recall@5": round(type_r5, 4) if type_r5 is not None else None,
        }

    return {
        "recall@3": round(mean_r3, 4),
        "recall@5": round(mean_r5, 4),
        "citation_validity": round(mean_cit, 4),
        "abstention_correctness": round(abst_corr, 4),
        "by_type": by_type_summary,
    }


def gate(results: dict, baseline: dict | None) -> list[str]:
    """Return one message per failed gate (section 5.4); an empty list means pass."""
    failures: list[str] = []

    current_r5 = results.get("recall@5", 0.0)
    if baseline is not None and "recall@5" in baseline:
        baseline_r5 = baseline.get("recall@5", 0.0)
        drop = baseline_r5 - current_r5
        if drop > MAX_RECALL5_DROP:
            failures.append(
                f"recall@5 dropped by {drop:.4f} (baseline: {baseline_r5:.4f}, current: {current_r5:.4f}), "
                f"exceeding max allowable drop of {MAX_RECALL5_DROP}"
            )

    cit_val = results.get("citation_validity", 1.0)
    if cit_val < MIN_CITATION_VALIDITY:
        failures.append(
            f"citation_validity ({cit_val:.4f}) is below minimum threshold of {MIN_CITATION_VALIDITY}"
        )

    abst_corr = results.get("abstention_correctness", 1.0)
    if abst_corr < REQUIRED_ABSTENTION:
        failures.append(
            f"abstention_correctness ({abst_corr:.4f}) is below required {REQUIRED_ABSTENTION} "
            "(one or more must-abstain questions were answered instead of declined)"
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAHBAR against a golden set.")
    parser.add_argument("--api", required=True, help="base URL, e.g. http://192.168.49.2")
    parser.add_argument("--host", help="Host header, when your Ingress rule names a host")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--update-baseline", action="store_true",
                        help="write these results as the new baseline")
    args = parser.parse_args()

    questions = load_golden(args.golden)
    rows = []
    headers = {"Host": args.host} if args.host else {}
    with httpx.Client(timeout=120.0, headers=headers) as client:
        for q in questions:
            started = time.perf_counter()
            response = ask(client, args.api, q["question"])
            rel = q.get("relevant") or q.get("relevant_passages") or []
            must_ab = bool(q.get("must_abstain", False)) or q.get("type") in ("abstain", "must_abstain") or (not rel)
            rows.append({
                "id": q["id"],
                "type": q.get("type", "ordinary"),
                "must_abstain": must_ab,
                "relevant": rel,
                "expect": q.get("expect"),
                "response": response,
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"  {q['id']:<5} {rows[-1]['seconds']:>6.2f}s  {q['question'][:60]}")

    results = summarise(rows)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.exists() else None
    failures = gate(results, baseline)

    results["golden"] = str(args.golden)
    results["questions"] = len(rows)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {args.out}")
    if args.update_baseline:
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"baseline updated: {args.baseline}")

    if failures:
        print("\nGATE FAILED")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\ngate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
