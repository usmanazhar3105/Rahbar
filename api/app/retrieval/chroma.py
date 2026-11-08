"""The real retriever using ChromaDB and Ollama embedding.

Connects to ChromaDB, embeds queries with nomic-embed-text, filters superseded policies
unless requested, applies a relevance floor, and maps results to Passage objects.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from ..config import Settings, get_settings
from ..schemas import Passage

log = logging.getLogger("rahbar.retrieval.chroma")

# Regex to detect when a user explicitly requests older / superseded rules
SUPERSEDED_OVERRIDE_REGEX = re.compile(
    r"\b(old version|previous policy|earlier policy|2023 rules|previous rules|repealed|superseded|prior version)\b",
    re.IGNORECASE,
)


def should_filter_superseded(question: str) -> bool:
    """Return True if superseded documents should be filtered out (default True).

    Returns False if the user explicitly asks about historical or older versions.
    """
    return not bool(SUPERSEDED_OVERRIDE_REGEX.search(question))


def verify_embed_model(
    expected_model: str,
    expected_dim: int = 768,
    settings: Settings | None = None,
    client: Any | None = None,
) -> None:
    """Verify that collection metadata matches the pinned EMBED_MODEL and vector dimension.

    Raises RuntimeError on mismatch to prevent serving corrupt retrieval results.
    """
    if settings is None:
        settings = get_settings()

    if client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    try:
        collection = client.get_collection(settings.collection)
    except Exception as exc:
        raise RuntimeError(f"Could not load collection '{settings.collection}': {exc}") from exc

    meta = collection.metadata or {}
    actual_model = meta.get("embed_model")
    actual_dim = meta.get("dimension") or meta.get("vector_dim")

    if actual_model and actual_model != expected_model:
        raise RuntimeError(
            f"EMBED_MODEL mismatch: ConfigMap says {expected_model} but index was built with {actual_model}. "
            f"Refusing to serve."
        )

    if expected_dim and actual_dim and int(actual_dim) != int(expected_dim):
        raise RuntimeError(
            f"EMBED_MODEL dimension mismatch: expected dimension {expected_dim} but collection has {actual_dim}. "
            f"Refusing to serve."
        )


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        if client is not None:
            self.client = client
        else:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

    def embed_query(self, text: str) -> list[float]:
        """Embed the question using the pinned EMBED_MODEL via Ollama."""
        base_url = self.settings.ollama_url.rstrip("/")
        embed_url = f"{base_url}/api/embed"

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                embed_url,
                json={"model": self.settings.embed_model, "input": [text]},
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding query failed at {embed_url} with status {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise ValueError(f"No embeddings returned from {embed_url}")

        return embeddings[0]

    def retrieve(self, question: str, k: int | None = None) -> list[Passage]:
        """Retrieve the top k passages from ChromaDB.

        Embeds the query, filters superseded documents (unless asked),
        calculates similarity (1 - cosine distance), and applies the relevance floor.
        """
        top_k = k if k is not None else self.settings.top_k
        query_vector = self.embed_query(question)

        collection = self.client.get_collection(self.settings.collection)

        # Filter out superseded documents unless query asks for old versions
        filter_superseded = should_filter_superseded(question)
        where_filter = {"superseded": False} if filter_superseded else None

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        passages: list[Passage] = []
        floor = self.settings.relevance_floor

        for chunk_id, doc_text, meta, distance in zip(ids_list, docs_list, metas_list, dists_list):
            meta = meta or {}
            # Chroma cosine distance: similarity = 1 - distance
            similarity = 1.0 - float(distance)

            # Relevance floor: drop chunks below floor threshold
            if similarity < floor:
                continue

            page_val = meta.get("page")
            parsed_page = int(page_val) if page_val is not None else None

            passages.append(
                Passage(
                    chunk_id=str(chunk_id),
                    text=str(doc_text),
                    source=str(meta.get("source", "unknown")),
                    section=str(meta.get("section", "General")),
                    page=parsed_page,
                    score=round(similarity, 4),
                )
            )

        # Sort highest similarity first
        passages.sort(key=lambda p: p.score, reverse=True)
        return passages
