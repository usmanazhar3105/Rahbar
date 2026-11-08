"""Stage 5: Store. The only code in the whole system that writes to Chroma."""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any

from .config import IndexerSettings
from .models import Chunk

log = logging.getLogger("rahbar.store")


def get_chroma_client(host: str, port: int = 8000) -> Any:
    """Instantiate a Chroma HTTP client with telemetry disabled."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_client(settings: IndexerSettings) -> Any:
    """Starter kit client helper for backward compatibility."""
    return get_chroma_client(settings.chroma_host, settings.chroma_port)


def flush_redis_cache(redis_host: str) -> None:
    """Flush Redis cache on re-index. Best-effort: logs warning on failure."""
    if not redis_host:
        return

    try:
        import redis

        r_host = redis_host
        raw_r_port = os.environ.get("REDIS_PORT", "6379")
        if ":" in str(raw_r_port):
            r_port = int(str(raw_r_port).split(":")[-1])
        else:
            r_port = int(raw_r_port)
        if ":" in r_host:
            parts = r_host.split(":")
            r_host = parts[0]
            r_port = int(parts[1])

        client = redis.Redis(host=r_host, port=r_port, socket_timeout=2.0)
        client.flushdb()
        log.info("Redis cache flushed successfully at %s:%d", r_host, r_port)
    except Exception as exc:
        log.warning(
            "Redis cache flush failed (unreachable at %s): %s. Continuing (cache flush is best-effort).",
            redis_host,
            exc,
        )


def store_chunks(
    chunks: list[dict[str, Any]],
    embed_metadata: dict[str, Any],
    chroma_host: str,
    redis_host: str,
) -> int:
    """Create or replace Chroma collection 'rahbar_docs', store chunks, and flush cache.

    Stores model name and dimension in collection metadata.
    Returns total chunk count stored.
    """
    collection_name = os.environ.get("CHROMA_COLLECTION", "rahbar_docs")
    embed_model = embed_metadata.get("embed_model", "nomic-embed-text")
    vector_dim = embed_metadata.get("vector_dim", 768)

    # Parse host and port
    c_host = chroma_host
    raw_c_port = os.environ.get("CHROMA_PORT", "8000")
    if ":" in str(raw_c_port):
        c_port = int(str(raw_c_port).split(":")[-1])
    else:
        c_port = int(raw_c_port)
    if ":" in c_host:
        parts = c_host.split(":")
        c_host = parts[0]
        c_port = int(parts[1])

    client = get_chroma_client(c_host, c_port)

    # Recreate collection to ensure atomic clean replace
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    col_meta = {
        "hnsw:space": "cosine",
        "embed_model": embed_model,
        "dimension": vector_dim,
        "vector_dim": vector_dim,
    }

    collection = client.create_collection(
        name=collection_name,
        metadata=col_meta,
    )
    try:
        collection.modify(metadata={"embed_model": embed_model, "dimension": vector_dim})
    except Exception:
        pass

    if chunks:
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        embeddings = [c.get("vector") or c.get("embedding") for c in chunks]
        metadatas = [
            {
                "source": str(c.get("source", "")),
                "section": str(c.get("section", "")),
                "page": int(c.get("page", 1)),
                "doc_version": str(c.get("doc_version", "")),
                "superseded": bool(c.get("superseded", False)),
                "embed_model": str(embed_model),
            }
            for c in chunks
        ]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # Flush Redis cache
    flush_redis_cache(redis_host)

    # Print summary
    counts = Counter(c.get("source", "unknown") for c in chunks)
    print(f"\n--- Indexing Complete: {len(chunks)} chunks stored in '{collection_name}' ---")
    for src, count in sorted(counts.items()):
        print(f"  • {src}: {count} chunk(s)")
    print("------------------------------------------------------------------\n")

    return len(chunks)


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Store adapter for the indexer pipeline."""
    chunk_dicts = [
        {
            "chunk_id": c.id,
            "text": c.text,
            "source": c.source,
            "section": c.section,
            "page": c.page,
            "doc_version": c.doc_version,
            "superseded": c.superseded,
            "vector": vec,
        }
        for c, vec in zip(chunks, vectors)
    ]

    embed_metadata = {
        "embed_model": settings.embed_model,
        "vector_dim": len(vectors[0]) if vectors else 768,
    }
    redis_host = os.environ.get("REDIS_HOST", "redis")

    return store_chunks(
        chunks=chunk_dicts,
        embed_metadata=embed_metadata,
        chroma_host=settings.chroma_host,
        redis_host=redis_host,
    )
