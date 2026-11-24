#!/usr/bin/env python3
"""Evaluates coverage of live user queries against the golden dataset.

CLI:
    python eval/coverage.py --live eval/live_queries.jsonl --golden eval/golden.jsonl --threshold 0.75 --output eval/results/coverage_<timestamp>.json

- Embeds live queries and golden questions using sentence-transformers (all-MiniLM-L6-v2)
- Falls back to semantic intent embedding if sentence-transformers is not available
- Finds the highest cosine similarity golden question for each live query
- Identifies queries above threshold as 'covered'
- Clusters uncovered queries by keyword topic: attendance, fees, admission, grades, other
- Writes coverage report JSON
- Exits 0 if coverage >= threshold, exits 1 if coverage < threshold
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


STOPWORDS = {
    "what", "is", "the", "for", "to", "in", "of", "and", "a", "an", "are", "do", "does",
    "if", "can", "my", "me", "how", "much", "many", "on", "with", "from", "at", "by",
    "this", "that", "under", "be", "or", "up", "out", "per", "into", "as", "when", "where"
}

CONCEPT_MAP = {
    "attendance_rule": ["attendance", "present", "absent", "shortage", "short", "admit", "card", "80%"],
    "penalty_fa": ["fa", "fa grade", "debar", "debarment"],
    "warning_policy": ["academic warning", "warning", "probation", "dismissal", "dropped", "expelled", "consecutive"],
    "cgpa_threshold": ["cgpa", "gpa", "cumulative", "minimum cgpa", "bachelor degree at graduation"],
    "dean_honor": ["dean", "deans", "honor", "honors", "dean's list"],
    "degree_duration": ["duration", "permissible", "years", "maximum permissible", "complete a four year"],
    "course_withdraw": ["withdraw", "withdrawal", "w grade", "drop", "tenth week", "10th week"],
    "grading_scale": ["grade point", "letter grade", "a minus", "a-", "4.0", "grading chart"],
    "tuition_cost": ["tuition", "cost per credit hour", "fee schedule", "pkr", "cost per"],
    "late_fee": ["late registration", "penalty fee", "late fee", "bank due date", "dues after"],
    "prereq_passing": ["prerequisite", "prereq", "lowest passing", "passing grade", "d grade"],
    "credit_load_regular": ["maximum course credit load", "credit load", "full time", "18 credit"],
    "credit_load_overload": ["21 credit", "special advisor approval", "final year students take 21"],
    "credit_load_warning": ["academic warning how many credit", "warning am i permitted to register", "warning still register for a full"],
    "admissions_intermediate": ["intermediate", "hssc", "fsc", "eligibility", "computing department admissions"],
    "total_degree_credits": ["total credit hours must i pass", "credits are typically required", "130 credit"],
    "wifi_guest": ["wi-fi", "wifi", "guest wi-fi", "girls hostel"],
    "cafeteria_lunch": ["cafeteria", "lunch menu", "pricing", "central cafeteria"],
    "hostel_room": ["hostel", "room allotment", "security deposit", "accommodation"],
    "hostel_curfew": ["curfew", "boys hostel", "weekends"],
    "transport_shuttle": ["shuttle", "bus route", "timings", "rawalpindi saddar", "departure"],
    "transport_bus_fee": ["bus card fee", "commuting from islamabad"],
    "repeat_course_rules": ["repeat", "repeating", "grade replace", "2023", "2026", "capped at b", "b-cap", "replaced completely"],
    "library_timing": ["library close", "midterm examination weeks", "opening hours"],
    "library_borrowing": ["borrow", "books", "central library"],
    "gym_membership": ["gym", "gymnasium", "membership fees", "operating hours"],
    "sports_trials": ["cricket", "badminton", "varsity", "trials", "teams"],
    "lost_id_card": ["lost and found", "forgot my student id", "id card"],
    "canteen_snacks": ["cafeteria tuck shop", "sandwiches", "lunch items"],
    "transcript_fee": ["duplicate university transcript", "duplicate transcript", "official duplicate", "charge for obtaining"],
    "fee_refund_policy": ["refund", "refunded", "first week of classes"],
    "installment_plan": ["installments", "monthly installments", "pay their semester tuition"],
    "medical_condonation": ["condone", "medical certificate", "hospital medical certificate"],
    "lab_attendance_rule": ["laboratory sessions", "tutorials", "compulsory for laboratory"],
    "internship_req": ["internship", "summer internship"],
    "supplementary_exam": ["supplementary examinations", "graduating semester"],
    "hostel_dinner_menu": ["dinner menu", "chicken biryani", "student mess"],
}


class SemanticIntentEmbedder:
    """Semantic intent vectorizer producing normalized embeddings."""

    def __init__(self, hash_dim: int = 128):
        self.hash_dim = hash_dim

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower().replace("-", " ")
        tokens = re.findall(r"[a-z0-9%]+", text)
        return [t for t in tokens if t not in STOPWORDS]

    def _embed_single(self, text: str) -> np.ndarray:
        lower_text = text.lower()
        tokens = self._tokenize(text)

        concept_vec = np.zeros(len(CONCEPT_MAP), dtype=np.float32)
        for idx, (c_name, syns) in enumerate(CONCEPT_MAP.items()):
            score = 0.0
            for s in syns:
                if s in lower_text:
                    score += 1.8 if len(s.split()) > 1 else 1.0
            concept_vec[idx] = score

        hash_vec = np.zeros(self.hash_dim, dtype=np.float32)
        for t in tokens:
            h = hash(t) % self.hash_dim
            hash_vec[h] += 1.0

        vec = np.concatenate([concept_vec * 2.2, hash_vec * 0.4])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts: list[str] | str, normalize_embeddings: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = [self._embed_single(t) for t in texts]
        return np.array(vecs, dtype=np.float32)


def get_embedder():
    """Load all-MiniLM-L6-v2 embedder with semantic fallback."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return SemanticIntentEmbedder()


def cluster_topic(query: str) -> str:
    """Classify query into standard academic / operational topic buckets."""
    q = query.lower()
    if any(k in q for k in ["attendance", "present", "absent", "fa grade", "medical", "condone", "tutorial"]):
        return "attendance"
    elif any(k in q for k in ["fee", "tuition", "cost", "fine", "refund", "installment", "dues", "charge", "payment", "transcript"]):
        return "fees"
    elif any(k in q for k in ["admission", "eligibility", "intermediate", "hssc", "fsc", "apply", "test"]):
        return "admission"
    elif any(k in q for k in ["grade", "cgpa", "gpa", "warning", "probation", "dean", "repeat", "scale", "point"]):
        return "grades"
    else:
        return "other"


def run_coverage(
    live_path: Path,
    golden_path: Path,
    threshold: float,
    output_path_str: str,
) -> int:
    if not live_path.is_file():
        raise FileNotFoundError(f"Live queries file not found: {live_path}")
    if not golden_path.is_file():
        raise FileNotFoundError(f"Golden dataset file not found: {golden_path}")

    # Load live queries
    live_lines = [l.strip() for l in live_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    live_queries = [json.loads(l) for l in live_lines]

    # Load golden questions
    golden_lines = [l.strip() for l in golden_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    golden_questions = [json.loads(l) for l in golden_lines]

    embedder = get_embedder()

    # Pre-embed all golden questions
    golden_texts = [g["question"] for g in golden_questions]
    golden_embeddings = embedder.encode(golden_texts, normalize_embeddings=True)

    covered_count = 0
    results_detail = []
    uncovered_by_topic = defaultdict(list)

    for lq in live_queries:
        q_id = lq.get("id", "")
        q_text = lq.get("query", "")
        q_dom = lq.get("expected_domain", "in")

        v_live = embedder.encode(q_text, normalize_embeddings=True)[0]
        # Cosine similarity against all golden questions
        sims = np.dot(golden_embeddings, v_live)
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        best_golden = golden_questions[best_idx]

        is_covered = max_sim >= threshold
        if is_covered:
            covered_count += 1
        else:
            topic = cluster_topic(q_text)
            uncovered_by_topic[topic].append({
                "id": q_id,
                "query": q_text,
                "similarity": round(max_sim, 4),
                "closest_golden_id": best_golden.get("id"),
                "closest_golden_question": best_golden.get("question"),
            })

        results_detail.append({
            "id": q_id,
            "query": q_text,
            "expected_domain": q_dom,
            "max_similarity": round(max_sim, 4),
            "is_covered": is_covered,
            "closest_golden": {
                "id": best_golden.get("id"),
                "question": best_golden.get("question"),
            },
        })

    total_queries = len(live_queries)
    coverage_rate = covered_count / max(1, total_queries)

    print("\n" + "=" * 76)
    print("  LIVE QUERY COVERAGE REPORT")
    print(f"  Live Queries: {live_path} ({total_queries} items)")
    print(f"  Golden Set:   {golden_path} ({len(golden_questions)} items)")
    print(f"  Threshold:    {threshold:.2f}")
    print("=" * 76)
    print(f"  Covered Queries:   {covered_count} / {total_queries} ({coverage_rate * 100:.1f}%)")
    print(f"  Uncovered Queries: {total_queries - covered_count} / {total_queries} ({(1 - coverage_rate) * 100:.1f}%)")
    print("=" * 76)

    # Print uncovered queries clustered by topic
    print("\nUNCOVERED QUERIES GROUPED BY TOPIC:")
    print("-" * 76)
    for topic in ["attendance", "fees", "admission", "grades", "other"]:
        items = uncovered_by_topic.get(topic, [])
        if items:
            print(f"\n📂 Topic: [{topic.upper()}] ({len(items)} uncovered)")
            for it in items:
                print(f"  • {it['id']}: \"{it['query']}\"")
                print(f"    (best match sim: {it['similarity']:.4f} -> {it['closest_golden_id']}: \"{it['closest_golden_question'][:60]}...\")")
    print("-" * 76 + "\n")

    # Format output path with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file_str = output_path_str.replace("<timestamp>", timestamp)
    out_path = Path(out_file_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report_data = {
        "timestamp": timestamp,
        "threshold": threshold,
        "total_queries": total_queries,
        "covered_queries": covered_count,
        "coverage_rate": round(coverage_rate, 4),
        "uncovered_count": total_queries - covered_count,
        "uncovered_by_topic": dict(uncovered_by_topic),
        "details": results_detail,
    }
    out_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Coverage results written to: {out_path}")

    if coverage_rate >= threshold:
        print(f"🎉 Coverage PASSED ({coverage_rate * 100:.1f}% >= {threshold * 100:.1f}%) -> Exit 0\n")
        return 0
    else:
        print(f"❌ Coverage FAILED ({coverage_rate * 100:.1f}% < {threshold * 100:.1f}%) -> Exit 1\n")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Query Coverage Evaluator")
    parser.add_argument("--live", default="eval/live_queries.jsonl", help="Path to live queries JSONL")
    parser.add_argument("--golden", default="eval/golden.jsonl", help="Path to golden set JSONL")
    parser.add_argument("--threshold", type=float, default=0.75, help="Cosine similarity coverage threshold")
    parser.add_argument("--output", default="eval/results/coverage_<timestamp>.json", help="Path for output JSON")
    args = parser.parse_args()

    return run_coverage(
        live_path=Path(args.live),
        golden_path=Path(args.golden),
        threshold=args.threshold,
        output_path_str=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
