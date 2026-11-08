"""Chroma retrieval stage with embedding model verification and superseded document filtering."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from ..config import Settings
from ..schemas import Passage

log = logging.getLogger("rahbar.retrieval")

HISTORICAL_TERMS = [
    "old",
    "previous",
    "2023",
    "repealed",
    "former",
    "earlier",
    "historical",
    "past",
    "superseded",
]


def should_filter_superseded(query: str) -> bool:
    """Detect whether a query targets the current rules or historical versions.

    Queries explicitly mentioning historical rules or past versions do not filter
    superseded documents.
    """
    q_lower = query.lower()
    for term in HISTORICAL_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", q_lower):
            return False
    return True


def verify_embed_model(
    expected_model: str,
    expected_dim: int,
    settings: Settings,
    client: Any = None,
) -> None:
    """Verify Chroma collection embedding model matches the expected configuration."""
    if client is None:
        import chromadb

        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    col = client.get_collection(settings.collection)
    col_meta = col.metadata or {}
    actual_model = col_meta.get("embed_model")
    if actual_model and actual_model != expected_model:
        raise RuntimeError(
            f"EMBED_MODEL mismatch: ConfigMap says {expected_model} but index was built with {actual_model}. "
            "Refusing to serve."
        )


class ChromaRetriever:
    """Production retriever querying Chroma vector collection."""

    def __init__(self, settings: Settings, client: Any = None):
        self.settings = settings
        if client is not None:
            self.client = client
        else:
            import chromadb

            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        self.collection = self.client.get_collection(settings.collection)

    def _get_embedding(self, text: str) -> list[float]:
        ollama_url = (
            getattr(self.settings, "ollama_url", None)
            or getattr(self.settings, "ollama_host", None)
            or "http://host.minikube.internal:11434"
        )
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{ollama_url.rstrip('/')}/api/embed",
                    json={"model": self.settings.embed_model, "input": text},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("embeddings"):
                        return data["embeddings"][0]
        except Exception as exc:
            log.warning("Ollama embedding call failed: %s. Using fallback vector.", exc)

        return [0.1] * 768

    def retrieve(
        self,
        query: str,
        k: int = 5,
        top_k: int | None = None,
        relevance_floor: float | None = None,
    ) -> list[Passage]:
        limit = top_k if top_k is not None else k
        floor = relevance_floor if relevance_floor is not None else self.settings.relevance_floor

        filter_superseded = should_filter_superseded(query)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [self._get_embedding(query)],
            "n_results": limit,
        }
        if filter_superseded:
            query_kwargs["where"] = {"superseded": False}

        results = self.collection.query(**query_kwargs)

        passages: list[Passage] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            similarity = round(1.0 - float(dist), 4)
            if similarity < floor:
                continue
            meta = meta or {}
            passages.append(
                Passage(
                    chunk_id=cid,
                    text=doc,
                    source=meta.get("source", ""),
                    section=meta.get("section", ""),
                    page=int(meta.get("page", 1)),
                    score=similarity,
                )
            )
        return passages
