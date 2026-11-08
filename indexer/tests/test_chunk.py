"""Unit tests for the Chunk stage (indexer/pipeline/chunk.py).

Tests run offline with no model, no network, and test structure-aware chunking and overlap.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.chunk import chunk, chunk_pages, is_heading
from pipeline.clean import clean
from pipeline.extract import extract
from pipeline.models import Chunk


def test_is_heading_detection() -> None:
    assert is_heading("## 1. Attendance")
    assert is_heading("# University Policies")
    assert is_heading("1. Attendance")
    assert is_heading("5.4 Academic Warning")
    assert is_heading("ADMISSION REGULATIONS")

    # Regular body sentences are not headings
    assert not is_heading("1.1 A student must attend at least 80 percent of the lectures of each course to be allowed to sit the final examination.")
    assert not is_heading("This is a regular sentence describing policies.")
    assert not is_heading("")


def test_chunk_pages_contract() -> None:
    pages = [
        {
            "page_num": 1,
            "text": (
                "## 1. Attendance\n"
                "A student must attend at least 80 percent of lectures.\n\n"
                "## 2. Academic Warning\n"
                "A student whose CGPA is below 2.00 is placed on warning."
            ),
            "had_ocr": False,
            "source": "handbook.pdf",
        }
    ]

    chunks = chunk_pages(
        pages=pages,
        doc_version="2026-06",
        superseded=False,
        chunk_size=400,
        overlap=80,
    )

    assert len(chunks) == 2

    c1 = chunks[0]
    assert "chunk_id" in c1
    assert c1["source"] == "handbook.pdf"
    assert c1["section"] == "1. Attendance"
    assert c1["page"] == 1
    assert c1["doc_version"] == "2026-06"
    assert c1["superseded"] is False
    assert "80 percent" in c1["text"]

    c2 = chunks[1]
    assert c2["section"] == "2. Academic Warning"
    assert "CGPA is below 2.00" in c2["text"]


def test_chunk_overlap_across_multiple_chunks() -> None:
    # Generate a long section that will exceed chunk_size=50 tokens
    long_para1 = "Sentence one about academic regulations. Sentence two explaining attendance requirements. Sentence three covering grading policy."
    long_para2 = "Sentence four regarding semester examination rules. Sentence five detailing withdrawal deadlines. Sentence six covering fee refunds."

    pages = [
        {
            "page_num": 1,
            "text": f"## 1. Comprehensive Rules\n\n{long_para1}\n\n{long_para2}",
            "had_ocr": False,
            "source": "rules.pdf",
        }
    ]

    # Target small size to force multiple chunks with overlap
    chunks = chunk_pages(
        pages=pages,
        doc_version="2026-06",
        superseded=False,
        chunk_size=35,
        overlap=15,
    )

    assert len(chunks) >= 2
    # Second chunk must carry tokens from the end of the first chunk
    first_chunk_text = chunks[0]["text"]
    second_chunk_text = chunks[1]["text"]

    # Verify section is retained
    assert chunks[0]["section"] == "1. Comprehensive Rules"
    assert chunks[1]["section"] == "1. Comprehensive Rules"

    # Verify overlap text exists in second chunk
    tail_words = first_chunk_text.split()[-3:]
    assert any(w in second_chunk_text for w in tail_words)


def test_chunk_document_adapter_with_fixture() -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "eval" / "fixtures" / "academic-rules-2026.md"
    raw_doc = extract(str(fixture_path), doc_version="2026", superseded=False)
    cleaned_doc = clean(raw_doc)

    doc_chunks = chunk(cleaned_doc, size=400, overlap=80)

    assert len(doc_chunks) >= 3
    for c in doc_chunks:
        assert isinstance(c, Chunk)
        assert c.source == "academic-rules-2026.md"
        assert c.doc_version == "2026"
        assert c.superseded is False
        assert c.page == 1
        meta = c.metadata()
        assert set(meta.keys()) == {"source", "section", "page", "doc_version", "superseded"}

    # Check that sections match expected fixture headings
    sections = [c.section for c in doc_chunks]
    assert any("Attendance" in s for s in sections)
    assert any("Academic warning" in s for s in sections)
    assert any("Repeating a course" in s for s in sections)

