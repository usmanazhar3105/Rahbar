from __future__ import annotations

from ..config import Settings
from .base import Retriever
from .chroma import ChromaRetriever, verify_embed_model
from .stub import StubRetriever


def get_retriever(settings: Settings) -> Retriever:
    if settings.retriever == "stub":
        return StubRetriever()
    if settings.retriever == "chroma":
        return ChromaRetriever(settings)
    raise ValueError(f"unknown RETRIEVER {settings.retriever!r}; expected stub or chroma")


__all__ = [
    "Retriever",
    "ChromaRetriever",
    "StubRetriever",
    "get_retriever",
    "verify_embed_model",
]
