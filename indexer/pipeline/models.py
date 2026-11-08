from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Page:
    page: int
    text: str
    ocr: bool = False
    source: str = ""


@dataclass
class Document:
    source: str
    doc_version: str = "unknown"
    superseded: bool = False
    pages: list[Page] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    section: str
    page: int | None = None
    doc_version: str = "unknown"
    superseded: bool = False
    embedding: list[float] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "section": self.section,
            "page": self.page,
            "doc_version": self.doc_version,
            "superseded": self.superseded,
        }
