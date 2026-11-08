import pytest
from pathlib import Path
import sys

# Ensure eval directory is importable
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from run_eval import recall_at_k, citation_validity, summarise, gate, load_golden


def test_load_golden_file():
    golden_path = HERE / "golden.yaml"
    assert golden_path.exists()
    questions = load_golden(golden_path)
    assert len(questions) >= 15

    # Check question distribution
    types = [q.get("type") for q in questions]
    assert types.count("ordinary") >= 8
    assert "version_sensitive" in types or "version" in types
    assert "exact_term" in types or "exact" in types
    assert "abstain" in types or "must_abstain" in types
    assert "injection" in types


def test_recall_at_k_abstain_returns_none():
    assert recall_at_k([], [{"source": "rules.pdf", "section": "1.0"}], k=5) is None


def test_recall_at_k_matching():
    relevant = [
        {"source": "academic_rules_2026.pdf", "section": "1.21"},
        {"source": "academic_rules_2026.pdf", "section": "5.4"},
    ]
    retrieved = [
        {"source": "academic_rules_2026.pdf", "section": "1.21 Attendance requirement", "score": 0.9},
        {"source": "other.pdf", "section": "9.9", "score": 0.8},
        {"source": "academic_rules_2026.pdf", "section": "5.4 Academic Warning", "score": 0.7},
    ]

    # At k=1: only 1 of 2 relevant retrieved
    assert recall_at_k(relevant, retrieved, k=1) == 0.5
    # At k=2: still only 1 retrieved
    assert recall_at_k(relevant, retrieved, k=2) == 0.5
    # At k=3: both retrieved
    assert recall_at_k(relevant, retrieved, k=3) == 1.0


def test_citation_validity():
    # Empty citations -> None
    assert citation_validity({"citations": []}) is None
    assert citation_validity({}) is None

    response = {
        "citations": [
            {"source": "academic_rules_2026.pdf", "section": "1.21"},
            {"source": "hallucinated.pdf", "section": "99.0"},
        ],
        "retrieved": [
            {"source": "academic_rules_2026.pdf", "section": "1.21 Attendance", "score": 0.9},
        ],
    }
    # 1 out of 2 valid
    assert citation_validity(response) == 0.5

    valid_response = {
        "citations": [
            {"source": "academic_rules_2026.pdf", "section": "1.21"},
        ],
        "retrieved": [
            {"source": "academic_rules_2026.pdf", "section": "1.21 Attendance", "score": 0.9},
        ],
    }
    assert citation_validity(valid_response) == 1.0


def test_summarise():
    rows = [
        {
            "id": "q1",
            "type": "ordinary",
            "must_abstain": False,
            "relevant": [{"source": "rules.pdf", "section": "1.0"}],
            "response": {
                "answered_from_context": True,
                "citations": [{"source": "rules.pdf", "section": "1.0"}],
                "retrieved": [{"source": "rules.pdf", "section": "1.0"}],
            },
        },
        {
            "id": "q2",
            "type": "abstain",
            "must_abstain": True,
            "relevant": [],
            "response": {
                "answered_from_context": False,
                "citations": [],
                "retrieved": [],
            },
        },
    ]

    summary = summarise(rows)
    assert summary["recall@3"] == 1.0
    assert summary["recall@5"] == 1.0
    assert summary["citation_validity"] == 1.0
    assert summary["abstention_correctness"] == 1.0
    assert "ordinary" in summary["by_type"]
    assert "abstain" in summary["by_type"]
    assert summary["by_type"]["ordinary"]["recall@5"] == 1.0
    assert summary["by_type"]["abstain"]["recall@5"] is None


def test_gate_pass_and_fail():
    baseline = {
        "recall@5": 0.95,
        "citation_validity": 1.0,
        "abstention_correctness": 1.0,
    }

    # Pass case
    good_results = {
        "recall@5": 0.95,
        "citation_validity": 1.0,
        "abstention_correctness": 1.0,
    }
    assert gate(good_results, baseline) == []

    # Drop within tolerance (0.04 <= 0.05) passes
    acceptable_drop = {
        "recall@5": 0.91,
        "citation_validity": 1.0,
        "abstention_correctness": 1.0,
    }
    assert gate(acceptable_drop, baseline) == []

    # Drop exceeding 0.05 fails
    excessive_drop = {
        "recall@5": 0.85,
        "citation_validity": 1.0,
        "abstention_correctness": 1.0,
    }
    failures = gate(excessive_drop, baseline)
    assert len(failures) == 1
    assert "recall@5 dropped" in failures[0]

    # Citation validity below 0.95 fails
    bad_citation = {
        "recall@5": 0.95,
        "citation_validity": 0.90,
        "abstention_correctness": 1.0,
    }
    failures = gate(bad_citation, baseline)
    assert len(failures) == 1
    assert "citation_validity" in failures[0]

    # Abstention correctness below 1.0 fails
    bad_abstain = {
        "recall@5": 0.95,
        "citation_validity": 1.0,
        "abstention_correctness": 0.5,
    }
    failures = gate(bad_abstain, baseline)
    assert len(failures) == 1
    assert "abstention_correctness" in failures[0]

