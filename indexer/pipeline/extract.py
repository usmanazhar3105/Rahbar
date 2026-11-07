"""Stage 1: Extract. PDF (or HTML, or Markdown) to text, page by page."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pypdf

from .models import Document, Page

log = logging.getLogger("rahbar.extract")


def is_heading(line: str) -> bool:
    """Determine if a line is a heading that should be preserved even if repeated.

    Preserves:
    - ALL CAPS headings (e.g. "UNDERGRADUATE RULES", "ADMISSIONS POLICY")
    - Numbered section headings (e.g. "1.2 Policy Name", "5.4 Academic Warning", "1. Attendance")
    - Markdown headings (e.g. "# Heading", "## Subheading")
    """
    clean_line = line.strip()
    if not clean_line:
        return False

    # Markdown style headings
    if clean_line.startswith("#"):
        return True

    # Numbered clauses: e.g. "1. Attendance", "1.2 Policy Name", "1.2.3 Rules", "5.4 Warning", "A.1 Appendix"
    if re.match(r"^(\d+(\.\d+)*\.?|[A-Z]\.\d+\.?)\s+[A-Za-z0-9]", clean_line):
        return True

    # ALL CAPS headings (at least 3 uppercase characters and no lowercase characters)
    letters = [ch for ch in clean_line if ch.isalpha()]
    if len(letters) >= 3 and clean_line.isupper():
        return True

    return False


def strip_repeating_headers_footers(pages_text: list[str], threshold: float = 0.60) -> list[str]:
    """Strip repeating headers and footers that appear on more than 60% of pages.

    Keeps section headings (e.g. ALL CAPS or '1.2 Policy Name') intact.
    """
    total_pages = len(pages_text)
    if total_pages <= 1:
        return pages_text

    # Count page frequency for each stripped line
    line_page_counts: Counter[str] = Counter()
    for page_text in pages_text:
        unique_lines = {line.strip() for line in page_text.splitlines() if line.strip()}
        for line in unique_lines:
            line_page_counts[line] += 1

    cutoff = total_pages * threshold

    cleaned_pages: list[str] = []
    for page_text in pages_text:
        kept_lines: list[str] = []
        for raw_line in page_text.splitlines():
            line_stripped = raw_line.strip()
            if not line_stripped:
                kept_lines.append("")
                continue

            # Check if this line exceeds repeating threshold
            if line_page_counts[line_stripped] > cutoff:
                # Retain if it's a heading
                if is_heading(line_stripped):
                    kept_lines.append(raw_line)
                else:
                    # Drop repeating header or footer line
                    continue
            else:
                kept_lines.append(raw_line)

        cleaned_pages.append("\n".join(kept_lines))

    return cleaned_pages


def ocr_page(pdf_path: str, page_num: int) -> str:
    """Run OCR on a single PDF page using pdf2image and pytesseract."""
    try:
        import pdf2image
        import pytesseract
    except ImportError as e:
        log.warning("OCR packages not installed (%s). Returning empty text.", e)
        return ""

    try:
        # Convert 1-based page to image
        images = pdf2image.convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        if not images:
            return ""
        return pytesseract.image_to_string(images[0])
    except Exception as exc:
        log.warning("OCR failed for %s page %d: %s", pdf_path, page_num, exc)
        return ""


def extract_pdf(path: str) -> list[dict[str, Any]]:
    """Extract text from a PDF file page by page.

    Returns a list of dicts:
      {
        "page_num": int,
        "text": str,
        "had_ocr": bool,
        "source": str,
      }
    """
    file_path = Path(path)
    source_name = file_path.name
    reader = pypdf.PdfReader(str(file_path))

    extracted_pages: list[dict[str, Any]] = []
    raw_texts: list[str] = []

    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        had_ocr = False

        # If page has fewer than 50 characters, it has no text layer -> run OCR
        if len(text.strip()) < 50:
            ocr_text = ocr_page(str(file_path), idx)
            if len(ocr_text.strip()) >= len(text.strip()):
                text = ocr_text
                had_ocr = True

        raw_texts.append(text)
        extracted_pages.append({
            "page_num": idx,
            "text": text,
            "had_ocr": had_ocr,
            "source": source_name,
        })

    # Strip repeating headers and footers (>60% of pages), keeping headings
    cleaned_texts = strip_repeating_headers_footers(raw_texts, threshold=0.60)
    for item, cleaned in zip(extracted_pages, cleaned_texts):
        item["text"] = cleaned

    return extracted_pages


def extract(path: str, doc_version: str, superseded: bool) -> Document:
    """Extract Document from a PDF, Markdown, or text file.

    Preserves page boundaries and records OCR status per page.
    """
    file_path = Path(path)
    source = file_path.name

    if file_path.suffix.lower() == ".pdf":
        page_dicts = extract_pdf(str(file_path))
        pages = [
            Page(
                source=item["source"],
                page=item["page_num"],
                text=item["text"],
                ocr=item["had_ocr"],
            )
            for item in page_dicts
        ]
        return Document(source=source, doc_version=doc_version, superseded=superseded, pages=pages)

    # For Markdown, HTML, or plain text (e.g. eval fixtures)
    text = file_path.read_text(encoding="utf-8", errors="replace")

    # Strip YAML front matter if present
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2].strip()

    pages = [Page(source=source, page=1, text=text, ocr=False)]
    return Document(source=source, doc_version=doc_version, superseded=superseded, pages=pages)
