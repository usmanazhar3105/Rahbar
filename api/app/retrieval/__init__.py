"""Retrieval package."""

from __future__ import annotations

from ..schemas import Passage, RetrievedChunk
from .chroma import ChromaRetriever, should_filter_superseded, verify_embed_model
from .stub import StubRetriever

Retriever = ChromaRetriever


def get_retriever(settings=None):
    if settings and getattr(settings, "retriever", "stub") == "chroma":
        return ChromaRetriever(settings)
    return StubRetriever()


__all__ = [
    "ChromaRetriever",
    "Passage",
    "RetrievedChunk",
    "Retriever",
    "StubRetriever",
    "get_retriever",
    "should_filter_superseded",
    "verify_embed_model",
]
