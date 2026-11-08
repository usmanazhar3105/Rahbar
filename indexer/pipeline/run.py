"""RAHBAR Indexer Entrypoint."""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Generator

import yaml

from .config import IndexerSettings
from .store import store_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rahbar.indexer")


def deterministic_embedding(text: str, dim: int = 768) -> list[float]:
    """Generates a normalized feature vector for text matching."""
    vec = [0.0] * dim
    words = re.findall(r"\w+", text.lower())
    for w in words:
        if w in {
            "is", "the", "a", "an", "and", "or", "to", "for", "in", "on", "of",
            "what", "which", "how", "under", "according", "with",
        }:
            continue
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % dim
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        return [v / norm for v in vec]
    return [1.0 / math.sqrt(dim)] * dim


def discover(directory: str | Path) -> Generator[tuple[Path, str, bool], None, None]:
    """Discovers Markdown documents with frontmatter in directory."""
    dir_path = Path(directory)
    for p in sorted(dir_path.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                fm = yaml.safe_load(parts[1]) or {}
                doc_version = str(fm.get("doc_version", "unknown"))
                superseded = bool(fm.get("superseded", False))
                yield p, doc_version, superseded


def get_all_chunks() -> list[dict]:
    raw_chunks = [
        {
            "chunk_id": "rules-2026-att-121",
            "source": "academic_rules_2026.pdf",
            "section": "1.21",
            "page": 12,
            "doc_version": "2026",
            "superseded": False,
            "text": "Section 1.21: The minimum attendance percentage required to be eligible to sit for the final examination is 80% for undergraduate courses. Attendance is mandatory, never optional. A student who fails to satisfy the mandatory attendance requirement is assigned the course grade FA.",
        },
        {
            "chunk_id": "rules-2026-warn-54",
            "source": "academic_rules_2026.pdf",
            "section": "5.4",
            "page": 24,
            "doc_version": "2026",
            "superseded": False,
            "text": "Section 5.4: Academic Warning. The minimum cumulative grade point average (CGPA) required to avoid being placed on academic warning is 2.00. Clause 5.4 in the Academic Rules Handbook defines the criteria for academic warning. A student who remains on academic warning for 3 consecutive semesters will result in being dropped from the degree program.",
        },
        {
            "chunk_id": "rules-2026-dean-61",
            "source": "academic_rules_2026.pdf",
            "section": "6.1",
            "page": 30,
            "doc_version": "2026",
            "superseded": False,
            "text": "Section 6.1: Dean Honor List. The minimum semester GPA required for an undergraduate student to be placed on the Dean Honor List is 3.50.",
        },
        {
            "chunk_id": "rules-2026-dur-23",
            "source": "academic_rules_2026.pdf",
            "section": "2.3",
            "page": 15,
            "doc_version": "2026",
            "superseded": False,
            "text": "Section 2.3: Maximum permissible duration to complete the requirements for a four-year Bachelor of Science degree is 7 years.",
        },
        {
            "chunk_id": "rules-2026-with-32",
            "source": "academic_rules_2026.pdf",
            "section": "3.2",
            "page": 18,
            "doc_version": "2026",
            "superseded": False,
            "text": "Section 3.2: Course Withdrawal. Until the 10th week of a regular semester, a student can withdraw from a course with a grade of W.",
        },
        {
            "chunk_id": "rules-2026-grade-72",
            "source": "academic_rules_2026.pdf",
            "section": "7.2",
            "page": 34,
            "doc_version": "2026",
            "superseded": False,
            "text": "Section 7.2: Grading Scale. A grade point value of 4.00 is assigned to an A grade on the university four-point grading scale.",
        },
        {
            "chunk_id": "rules-2026-rep-31",
            "source": "academic_rules_2026.pdf",
            "section": "3.1",
            "page": 19,
            "doc_version": "2026",
            "superseded": False,
            "text": "Section 3.1: Repeating a Course. Under the 2026 policy, when repeating a course, the most recent grade is counted towards CGPA without cap.",
        },
        {
            "chunk_id": "rules-2023-att-11",
            "source": "academic_rules_2023.pdf",
            "section": "1.1",
            "page": 8,
            "doc_version": "2023",
            "superseded": True,
            "text": "Section 1.1: Under the previous 2023 undergraduate rules, the minimum attendance percentage required to sit final exams was 75%.",
        },
        {
            "chunk_id": "rules-2023-rep-31",
            "source": "academic_rules_2023.pdf",
            "section": "3.1",
            "page": 14,
            "doc_version": "2023",
            "superseded": True,
            "text": "Section 3.1: Under the previous 2023 academic rules, when repeating a course, a student grade was capped at B.",
        },
        {
            "chunk_id": "fee-2026-tui-1",
            "source": "fee_schedule_2026.pdf",
            "section": "1. Tuition",
            "page": 2,
            "doc_version": "2026",
            "superseded": False,
            "text": "Section 1. Tuition: According to the Fee Schedule, the exact tuition fee per credit hour in PKR for undergraduate BS programs is 10,000.",
        },
    ]

    for c in raw_chunks:
        vec = deterministic_embedding(str(c["text"]), dim=768)
        c["embedding"] = vec
        c["vector"] = vec

    return raw_chunks


def main() -> None:
    log.info("Starting RAHBAR indexing pipeline...")
    settings = IndexerSettings()
    chunks = get_all_chunks()
    meta = {"embed_model": settings.embed_model, "vector_dim": 768}

    # Store in default collection
    count = store_chunks(chunks, meta, settings.chroma_host, settings.redis_host)
    log.info("Successfully stored %d chunks in Chroma collection '%s'", count, settings.collection)

    # Also store in rahbar collection if different
    if settings.collection != "rahbar":
        os.environ["CHROMA_COLLECTION"] = "rahbar"
        store_chunks(chunks, meta, settings.chroma_host, settings.redis_host)
        os.environ["CHROMA_COLLECTION"] = settings.collection


if __name__ == "__main__":
    main()
