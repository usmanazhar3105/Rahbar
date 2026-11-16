"""Indexer configuration, from the same ConfigMap the API reads (section 4.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexerSettings:
    embed_model: str        # EMBED_MODEL: must equal the API's value (section 4.4)
    chroma_host: str
    chroma_port: int
    collection: str
    corpus_dir: str         # where the documents and _index.json sit inside the Job's pod
    chunk_size: int
    chunk_overlap: int
    ollama_url: str


def _int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if not val:
        return default
    if ":" in val:
        val = val.rsplit(":", 1)[-1]
    try:
        return int(val)
    except ValueError:
        return default


def get_settings() -> IndexerSettings:
    return IndexerSettings(
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
        chroma_port=_int("CHROMA_PORT", 8000),
        collection=os.environ.get("CHROMA_COLLECTION", "rahbar"),
        corpus_dir=os.environ.get("CORPUS_DIR", "/corpus"),
        chunk_size=_int("CHUNK_SIZE", 400),
        chunk_overlap=_int("CHUNK_OVERLAP", 50),
        ollama_url=os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434"),
    )
