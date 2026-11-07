"""Facade module re-exporting the retrieval components."""

from __future__ import annotations

from .retrieval import (
    ChromaRetriever,
    Retriever,
    StubRetriever,
    get_retriever,
    verify_embed_model,
)
from .schemas import Passage, RetrievedChunk

__all__ = [
    "Retriever",
    "ChromaRetriever",
    "StubRetriever",
    "Passage",
    "RetrievedChunk",
    "get_retriever",
    "verify_embed_model",
]

