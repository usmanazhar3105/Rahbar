"""Validation script for eval/golden.jsonl evaluation benchmark.

Asserts:
  - Exactly >= 25 entries
  - At least 15 answerable, 5 unanswerable, 5 hard
  - At least 8 human-written
  - All required keys present: id, question, type, source, ground_truth_doc, ground_truth_page, ground_truth_answer

Exits 0 if valid, 1 with an error message if not.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_KEYS = {
    "id",
    "question",
    "type",
    "source",
    "ground_truth_doc",
    "ground_truth_page",
    "ground_truth_answer",
}

VALID_TYPES = {"answerable", "unanswerable", "hard"}
VALID_SOURCES = {"human", "synthetic"}


def validate_golden(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"File not found: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if len(lines) < 25:
        raise ValueError(f"Expected at least 25 entries, found {len(lines)}")

    entries = []
    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for idx, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {idx}: Invalid JSON: {exc}") from exc

        if not isinstance(entry, dict):
            raise ValueError(f"Line {idx}: Entry must be a JSON object")

        # Key validation
        missing_keys = REQUIRED_KEYS - set(entry.keys())
        if missing_keys:
            raise ValueError(f"Line {idx} (ID: {entry.get('id', '?')}): Missing required keys: {sorted(missing_keys)}")

        entry_id = str(entry["id"]).strip()
        if not entry_id:
            raise ValueError(f"Line {idx}: 'id' must not be empty")
        if entry_id in seen_ids:
            raise ValueError(f"Line {idx}: Duplicate ID '{entry_id}'")
        seen_ids.add(entry_id)

        # Question validation
        question = str(entry["question"]).strip()
        if not question:
            raise ValueError(f"Line {idx} (ID: {entry_id}): 'question' must not be empty")

        # Type validation
        q_type = str(entry["type"]).strip().lower()
        if q_type not in VALID_TYPES:
            raise ValueError(f"Line {idx} (ID: {entry_id}): Invalid type '{q_type}', expected one of {sorted(VALID_TYPES)}")
        type_counts[q_type] += 1

        # Source validation
        source = str(entry["source"]).strip().lower()
        if source not in VALID_SOURCES:
            raise ValueError(f"Line {idx} (ID: {entry_id}): Invalid source '{source}', expected one of {sorted(VALID_SOURCES)}")
        source_counts[source] += 1

        # Type-specific rules
        if q_type == "unanswerable":
            if entry["ground_truth_doc"] is not None:
                raise ValueError(f"Line {idx} (ID: {entry_id}): 'ground_truth_doc' must be null for unanswerable question")
            if entry["ground_truth_page"] is not None:
                raise ValueError(f"Line {idx} (ID: {entry_id}): 'ground_truth_page' must be null for unanswerable question")
            if entry["ground_truth_answer"] is not None:
                raise ValueError(f"Line {idx} (ID: {entry_id}): 'ground_truth_answer' must be null for unanswerable question")
        else:
            if not entry["ground_truth_doc"]:
                raise ValueError(f"Line {idx} (ID: {entry_id}): 'ground_truth_doc' must not be empty for {q_type} question")
            if entry["ground_truth_page"] is None:
                raise ValueError(f"Line {idx} (ID: {entry_id}): 'ground_truth_page' must be specified for {q_type} question")
            if not entry["ground_truth_answer"]:
                raise ValueError(f"Line {idx} (ID: {entry_id}): 'ground_truth_answer' must not be empty for {q_type} question")

        entries.append(entry)

    # Check minimum distributions
    if type_counts["answerable"] < 15:
        raise ValueError(f"Insufficient answerable questions: found {type_counts['answerable']}, expected at least 15")
    if type_counts["unanswerable"] < 5:
        raise ValueError(f"Insufficient unanswerable questions: found {type_counts['unanswerable']}, expected at least 5")
    if type_counts["hard"] < 5:
        raise ValueError(f"Insufficient hard questions: found {type_counts['hard']}, expected at least 5")
    if source_counts["human"] < 8:
        raise ValueError(f"Insufficient human questions: found {source_counts['human']}, expected at least 8")

    print(f"Validation SUCCESS: {len(entries)} valid golden questions verified.")
    print(f"  Distribution: {type_counts['answerable']} answerable, {type_counts['unanswerable']} unanswerable, {type_counts['hard']} hard")
    print(f"  Sources: {source_counts['human']} human, {source_counts['synthetic']} synthetic")


def main() -> int:
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "golden.jsonl"
    try:
        validate_golden(target_path)
        return 0
    except Exception as exc:
        print(f"Validation FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
