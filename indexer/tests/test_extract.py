"""Unit tests for the Extract stage (indexer/pipeline/extract.py).

Tests run offline with no model, no network, and test against fixture corpus and synthetic PDFs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pypdf

from pipeline.extract import (
    extract,
    extract_pdf,
    is_heading,
    strip_repeating_headers_footers,
)
from pipeline.models import Document


def test_is_heading_detection() -> None:
    # Preserves ALL CAPS
    assert is_heading("ACADEMIC RULES AND REGULATIONS")
    assert is_heading("SCHEDULE OF CHARGES")

    # Preserves Numbered Headings
    assert is_heading("1.2 Policy Name")
    assert is_heading("5.4 Academic Warning")
    assert is_heading("1. Attendance")
    assert is_heading("2.1.3 Registration Procedure")
    assert is_heading("# Margalla Institute Rules")
    assert is_heading("## 3. Repeating a course")

    # Regular text / headers / footers are not headings
    assert not is_heading("Margalla Institute of Technology")
    assert not is_heading("Page 1 of 53")
    assert not is_heading("A student must attend at least 80 percent of lectures.")
    assert not is_heading("")


def test_strip_repeating_headers_footers_preserves_headings() -> None:
    # 4 pages where "FAST-NUCES Header" appears on all 4 pages (>60%)
    # "Page Number Footer" appears on all 4 pages (>60%)
    # "1.2 Policy Heading" appears on 3 pages (>60%) but is a HEADING -> should be kept!
    page1 = (
        "FAST-NUCES Header\n"
        "1.2 Policy Heading\n"
        "This is unique content on page 1.\n"
        "Page Number Footer"
    )
    page2 = (
        "FAST-NUCES Header\n"
        "1.2 Policy Heading\n"
        "This is unique content on page 2.\n"
        "Page Number Footer"
    )
    page3 = (
        "FAST-NUCES Header\n"
        "1.2 Policy Heading\n"
        "This is unique content on page 3.\n"
        "Page Number Footer"
    )
    page4 = (
        "FAST-NUCES Header\n"
        "This is unique content on page 4 without heading.\n"
        "Page Number Footer"
    )

    pages = [page1, page2, page3, page4]
    cleaned = strip_repeating_headers_footers(pages, threshold=0.60)

    for idx, c in enumerate(cleaned):
        # Repeating non-heading header and footer should be stripped
        assert "FAST-NUCES Header" not in c
        assert "Page Number Footer" not in c
        # Unique content remains
        assert f"This is unique content on page {idx + 1}" in c

    # Heading was preserved on pages 1, 2, 3
    assert "1.2 Policy Heading" in cleaned[0]
    assert "1.2 Policy Heading" in cleaned[1]
    assert "1.2 Policy Heading" in cleaned[2]


def create_test_pdf(tmp_path: Path, num_pages: int = 2) -> Path:
    """Create a small multi-page PDF for testing."""
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    pdf_path = tmp_path / "sample_doc.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


def test_extract_pdf_contract_and_ocr_fallback(tmp_path: Path) -> None:
    pdf_file = create_test_pdf(tmp_path, num_pages=2)

    # Blank pages have 0 chars (< 50 threshold), which triggers OCR fallback
    with patch("pipeline.extract.ocr_page", return_value="1.1 Attendance Policy\nExtracted OCR Text") as mock_ocr:
        results = extract_pdf(str(pdf_file))

        assert len(results) == 2
        assert mock_ocr.call_count == 2

        for i, item in enumerate(results, start=1):
            assert item["page_num"] == i
            assert item["source"] == "sample_doc.pdf"
            assert item["had_ocr"] is True
            assert "Attendance Policy" in item["text"]


def test_extract_document_for_markdown_fixture() -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "eval" / "fixtures" / "academic-rules-2026.md"
    assert fixture_path.exists(), "eval fixture must exist"

    doc: Document = extract(str(fixture_path), doc_version="2026", superseded=False)

    assert doc.source == "academic-rules-2026.md"
    assert doc.doc_version == "2026"
    assert doc.superseded is False
    assert len(doc.pages) == 1

    page = doc.pages[0]
    assert page.page == 1
    assert page.ocr is False
    # Front matter should be stripped, body intact
    assert "---" not in page.text.splitlines()[:2]
    assert "## 1. Attendance" in page.text
    assert "80 percent" in page.text

