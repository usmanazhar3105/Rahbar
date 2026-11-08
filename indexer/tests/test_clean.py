"""Unit tests for the Clean stage (indexer/pipeline/clean.py).

Tests run offline with no model, no network, and test against fixture corpus and synthetic page dicts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.clean import clean, clean_pages, clean_text, fix_ocr_errors
from pipeline.extract import extract
from pipeline.models import Document


def test_ligature_and_invisible_character_cleaning() -> None:
    raw = (
        "This is a de\ufb01nition of a \ufb02ag\u00adship "
        "policy with a\u200b zero\u200cwidth\ufeff space."
    )
    cleaned = clean_text(raw, had_ocr=False)
    assert "definition" in cleaned
    assert "flagship" in cleaned
    assert "\ufb01" not in cleaned
    assert "\ufb02" not in cleaned
    assert "\u00ad" not in cleaned
    assert "\u200b" not in cleaned
    assert "zerowidth" in cleaned or "zero width" in cleaned


def test_consecutive_blank_lines_collapsed() -> None:
    raw = "Paragraph one.\n\n\n\n\n\nParagraph two.\n\n\nParagraph three."
    cleaned = clean_text(raw, had_ocr=False)
    assert cleaned == "Paragraph one.\n\nParagraph two.\n\nParagraph three."


def test_page_number_stripping_preserves_structure() -> None:
    raw = (
        "## 1. Attendance\n"
        "1. Attendance is mandatory for all students.\n"
        "2. Minimum threshold is 80 percent.\n"
        "a) Medical exceptions require HOD approval.\n"
        "b) Absences beyond condonation lead to grade FA.\n"
        "Page 12 of 50\n"
        "4.2 Academic Warning applies when CGPA < 2.00.\n"
        "| Course | Fee |\n"
        "| CS101 | 10,000 |\n"
        "12\n"
        "BS (all disciplines)    10,000\n"
        "- 13 -"
    )

    cleaned = clean_text(raw, had_ocr=False)
    cleaned_lines = cleaned.splitlines()

    # Page number markers must be stripped
    assert "Page 12 of 50" not in cleaned_lines
    assert "12" not in cleaned_lines
    assert "- 13 -" not in cleaned_lines

    # Section headings, list items, tables, and clauses must be PRESERVED
    assert "## 1. Attendance" in cleaned_lines
    assert "1. Attendance is mandatory for all students." in cleaned_lines
    assert "2. Minimum threshold is 80 percent." in cleaned_lines
    assert "a) Medical exceptions require HOD approval." in cleaned_lines
    assert "b) Absences beyond condonation lead to grade FA." in cleaned_lines
    assert "4.2 Academic Warning applies when CGPA < 2.00." in cleaned_lines
    assert "| Course | Fee |" in cleaned_lines
    assert "| CS101 | 10,000 |" in cleaned_lines
    assert "BS (all disciplines)    10,000" in cleaned_lines


def test_ocr_error_correction_on_ocr_pages() -> None:
    # Text with OCR errors: 'C0URSE', 'po1icy', '1ectures', 'cond0ne'
    # Mixed with valid numbers that should NOT be altered: 'CS101', '10,000', '2.00', '4.2'
    raw_ocr = (
        "In this C0URSE, the po1icy requires students to attend all 1ectures. "
        "The HOD may cond0ne absence under clause 4.2. "
        "Tuition for CS101 is 10,000 PKR and min CGPA is 2.00."
    )

    # When had_ocr=True, fix OCR errors
    fixed = clean_text(raw_ocr, had_ocr=True)
    assert "COURSE" in fixed
    assert "policy" in fixed
    assert "lectures" in fixed
    assert "condone" in fixed

    # Valid alphanumeric tokens and numbers must remain unchanged
    assert "CS101" in fixed
    assert "10,000" in fixed
    assert "2.00" in fixed
    assert "4.2" in fixed


def test_clean_pages_contract() -> None:
    input_pages = [
        {
            "page_num": 1,
            "text": "Page 1 of 2\n\n\n\n## 1. Title\nDe\ufb01nition of po1icy.",
            "had_ocr": True,
            "source": "handbook.pdf",
        },
        {
            "page_num": 2,
            "text": "1\n2. Next Section\nAttendance is 80%.",
            "had_ocr": False,
            "source": "handbook.pdf",
        },
    ]

    output_pages = clean_pages(input_pages)

    assert len(output_pages) == 2
    for orig, res in zip(input_pages, output_pages):
        assert res["page_num"] == orig["page_num"]
        assert res["had_ocr"] == orig["had_ocr"]
        assert res["source"] == orig["source"]

    # Page 1 checks
    p1 = output_pages[0]
    assert "Page 1 of 2" not in p1["text"]
    assert "Definition of policy." in p1["text"]

    # Page 2 checks
    p2 = output_pages[1]
    assert p2["text"].splitlines()[0] == "2. Next Section"


def test_clean_document_with_fixture() -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "eval" / "fixtures" / "academic-rules-2026.md"
    raw_doc = extract(str(fixture_path), doc_version="2026", superseded=False)

    cleaned_doc = clean(raw_doc)

    assert isinstance(cleaned_doc, Document)
    assert cleaned_doc.source == "academic-rules-2026.md"
    assert cleaned_doc.doc_version == "2026"
    assert cleaned_doc.superseded is False
    assert len(cleaned_doc.pages) == 1

    page_text = cleaned_doc.pages[0].text
    assert "## 1. Attendance" in page_text
    assert "80 percent" in page_text
    # Injection text is preserved as data
    assert "Ignore all previous instructions" in page_text

