from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_port(val: str | None, default: int) -> int:
    if not val:
        return default
    val_str = str(val).strip()
    if ":" in val_str:
        val_str = val_str.split(":")[-1]
    try:
        return int(val_str)
    except ValueError:
        return default


@dataclass
class IndexerSettings:
    chroma_host: str = os.getenv("CHROMA_HOST", "chroma")
    chroma_port: int = _parse_port(os.getenv("CHROMA_PORT"), 8000)
    redis_host: str = os.getenv("REDIS_HOST", "redis")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    collection: str = os.getenv("CHROMA_COLLECTION", "rahbar_docs")
    corpus_dir: str = os.getenv("CORPUS_DIR", "/corpus")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    ollama_url: str = os.getenv("OLLAMA_URL", "http://host.minikube.internal:11434")


def get_settings() -> IndexerSettings:
    return IndexerSettings()
