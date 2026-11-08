"""Stage 3: Chunk. Structure-aware, with overlap."""

from __future__ import annotations

import re
import uuid
from typing import Any

import tiktoken

from .models import Chunk, Document

# Fallback encoding if cl100k_base cannot be loaded
try:
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base, with whitespace fallback."""
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Approximation: ~1.3 tokens per word
    return int(len(text.split()) * 1.3)


def slice_tokens_tail(text: str, num_tokens: int) -> str:
    """Extract the last num_tokens from text and return as string."""
    if num_tokens <= 0 or not text.strip():
        return ""
    if _ENC is not None:
        tokens = _ENC.encode(text)
        if len(tokens) <= num_tokens:
            return text
        tail_tokens = tokens[-num_tokens:]
        return _ENC.decode(tail_tokens)
    words = text.split()
    target_words = max(1, int(num_tokens / 1.3))
    return " ".join(words[-target_words:])


def clean_heading_title(line: str) -> str:
    """Clean a heading line to produce a clean section name."""
    clean_line = line.strip()
    # Strip markdown hashes
    clean_line = re.sub(r"^#+\s*", "", clean_line)
    return clean_line.strip()


def is_heading(line: str) -> bool:
    """Identify if a line represents a section heading boundary.

    Headings are short titles (<= 12 words), not full sentence paragraphs.
    """
    clean_line = line.strip()
    if not clean_line or len(clean_line.split()) > 12:
        return False

    # Markdown headings (# Heading, ## Section)
    if clean_line.startswith("#"):
        return True

    # Numbered section clauses (e.g. 1. Attendance, 1.2 Policy, 5.4 Academic Warning)
    if re.match(r"^(\d+(\.\d+)*\.?|[A-Z]\.\d+\.?)\s+[A-Za-z0-9]", clean_line):
        return True

    # ALL CAPS lines (e.g. "MARGINAL ACADEMIC POLICY")
    letters = [ch for ch in clean_line if ch.isalpha()]
    if len(letters) >= 3 and clean_line.isupper():
        return True

    return False


def split_into_units(text: str, max_tokens: int) -> list[str]:
    """Split text into paragraphs, and long paragraphs into sentences."""
    paragraphs = re.split(r"\n\s*\n", text)
    units: list[str] = []

    for para in paragraphs:
        p_strip = para.strip()
        if not p_strip:
            continue

        if count_tokens(p_strip) <= max_tokens:
            units.append(p_strip)
        else:
            # Paragraph is too long -> split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", p_strip)
            for s in sentences:
                s_strip = s.strip()
                if s_strip:
                    units.append(s_strip)

    return units


def chunk_pages(
    pages: list[dict[str, Any]],
    doc_version: str,
    superseded: bool,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[dict[str, Any]]:
    """Structure-aware chunking with token overlap.

    Splits on section boundaries, then paragraphs, then sentences.
    Returns a list of dicts with keys:
      chunk_id, text, source, section, page, doc_version, superseded
    """
    if not pages:
        return []

    source_name = pages[0].get("source", "unknown")
    chunks: list[dict[str, Any]] = []

    # Parse document into structural sections across pages
    # Section format: (section_title, starting_page, section_body)
    sections: list[dict[str, Any]] = []
    current_sec_title = "General"
    current_sec_page = pages[0].get("page_num", 1)
    current_sec_lines: list[tuple[int, str]] = []

    for p in pages:
        page_num = p.get("page_num", 1)
        lines = p.get("text", "").splitlines()

        for line in lines:
            line_str = line.strip()
            if is_heading(line_str):
                # Flush previous section
                if current_sec_lines:
                    sections.append({
                        "section": current_sec_title,
                        "page": current_sec_page,
                        "lines": current_sec_lines,
                    })
                current_sec_title = clean_heading_title(line_str)
                current_sec_page = page_num
                current_sec_lines = [(page_num, line)]
            else:
                current_sec_lines.append((page_num, line))

    # Flush final section
    if current_sec_lines:
        sections.append({
            "section": current_sec_title,
            "page": current_sec_page,
            "lines": current_sec_lines,
        })

    # Chunk each section
    chunk_counter = 0

    for sec in sections:
        sec_title = sec["section"]
        sec_lines = sec["lines"]
        if not sec_lines:
            continue

        sec_start_page = sec["page"]
        full_sec_text = "\n".join(l_text for _, l_text in sec_lines).strip()
        if not full_sec_text:
            continue

        units = split_into_units(full_sec_text, max_tokens=chunk_size)
        if not units:
            continue

        current_chunk_parts: list[str] = []
        current_chunk_page = sec_start_page
        current_tokens = 0

        for unit in units:
            unit_tokens = count_tokens(unit)

            if current_chunk_parts and (current_tokens + unit_tokens > chunk_size):
                # Emit current chunk
                chunk_text = "\n\n".join(current_chunk_parts).strip()
                chunk_counter += 1
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_name}:{sec_title}:{chunk_counter}"))

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": source_name,
                    "section": sec_title,
                    "page": current_chunk_page,
                    "doc_version": doc_version,
                    "superseded": superseded,
                })

                # Carry overlap into next chunk
                overlap_text = slice_tokens_tail(chunk_text, overlap)
                current_chunk_parts = [overlap_text, unit] if overlap_text else [unit]
                current_tokens = count_tokens("\n\n".join(current_chunk_parts))
            else:
                current_chunk_parts.append(unit)
                current_tokens += unit_tokens

        # Flush any remaining text in the section
        if current_chunk_parts:
            chunk_text = "\n\n".join(current_chunk_parts).strip()
            # If chunk is not just an empty duplicate of overlap
            if chunk_text:
                chunk_counter += 1
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_name}:{sec_title}:{chunk_counter}"))
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": source_name,
                    "section": sec_title,
                    "page": current_chunk_page,
                    "doc_version": doc_version,
                    "superseded": superseded,
                })

    return chunks


def chunk(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Adapter bridging the indexer Document pipeline to chunk_pages."""
    page_dicts = [
        {
            "page_num": p.page,
            "text": p.text,
            "had_ocr": p.ocr,
            "source": p.source,
        }
        for p in doc.pages
    ]

    chunk_dicts = chunk_pages(
        pages=page_dicts,
        doc_version=doc.doc_version,
        superseded=doc.superseded,
        chunk_size=size,
        overlap=overlap,
    )

    return [
        Chunk(
            id=cd["chunk_id"],
            text=cd["text"],
            source=cd["source"],
            section=cd["section"],
            page=cd["page"],
            doc_version=cd["doc_version"],
            superseded=cd["superseded"],
        )
        for cd in chunk_dicts
    ]
