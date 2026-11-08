"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .config import IndexerSettings

DEFAULT_OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST",
    os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434"),
)
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
BATCH_SIZE = 32


class OllamaUnreachableError(RuntimeError):
    """Raised when the Ollama embedding service is unreachable."""


def embed_chunks(
    chunks: list[dict[str, Any]],
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_EMBED_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Embed chunks using Ollama HTTP API in batches of 32.

    Returns:
      (chunks_with_vectors, metadata_dict)
      where metadata_dict has {"embed_model": str, "vector_dim": int}.

    Raises:
      OllamaUnreachableError: if Ollama server cannot be reached at ollama_host.
    """
    if not chunks:
        return [], {"embed_model": model, "vector_dim": 768}

    base_url = ollama_host.rstrip("/")
    embed_url = f"{base_url}/api/embed"
    vector_dim = 768
    total_chunks = len(chunks)

    # Process in batches of 32
    for start_idx in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[start_idx : start_idx + BATCH_SIZE]
        batch_texts = [c.get("text", "") for c in batch]

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    embed_url,
                    json={"model": model, "input": batch_texts},
                )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.RequestError) as exc:
            raise OllamaUnreachableError(
                f"Ollama server is unreachable at {base_url}. "
                f"Ensure Ollama is running and accessible (check OLLAMA_HOST). Error: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama embedding request failed at {embed_url} with status {response.status_code}: {response.text}"
            )

        data = response.json()
        embeddings = data.get("embeddings") or []

        if len(embeddings) != len(batch):
            raise ValueError(
                f"Mismatch in embeddings count: expected {len(batch)}, received {len(embeddings)}"
            )

        if embeddings:
            vector_dim = len(embeddings[0])

        for chunk, emb in zip(batch, embeddings):
            chunk["vector"] = emb

    metadata = {
        "embed_model": model,
        "vector_dim": vector_dim,
    }
    return chunks, metadata


def embed(texts: list[str], settings: IndexerSettings) -> list[list[float]]:
    """Adapter for the indexer pipeline runner.

    Embeds raw string texts in batches using the configured embed_model and ollama_url.
    """
    if not texts:
        return []

    dummy_chunks = [{"text": t} for t in texts]
    embedded_chunks, _ = embed_chunks(
        chunks=dummy_chunks,
        ollama_host=settings.ollama_url,
        model=settings.embed_model,
    )
    return [c["vector"] for c in embedded_chunks]
