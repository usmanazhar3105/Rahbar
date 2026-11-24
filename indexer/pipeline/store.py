"""Stage 5: Store. The only code in the whole system that writes to Chroma."""

from __future__ import annotations

import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .config import IndexerSettings
from .models import Chunk

log = logging.getLogger("rahbar.store")


def get_chroma_client(host: str, port: int = 8000) -> Any:
    """Instantiate a Chroma HTTP client with telemetry disabled."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    c_host = host
    c_port = port
    if ":" in c_host:
        parts = c_host.split(":")
        c_host = parts[0]
        try:
            c_port = int(parts[1])
        except ValueError:
            pass

    return chromadb.HttpClient(
        host=c_host,
        port=c_port,
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
        r_val = os.environ.get("REDIS_PORT", "6379")
        if ":" in r_val:
            r_val = r_val.rsplit(":", 1)[-1]
        try:
            r_port = int(r_val)
        except ValueError:
            r_port = 6379

        if ":" in r_host:
            parts = r_host.split(":")
            r_host = parts[0]
            try:
                r_port = int(parts[1])
            except ValueError:
                pass

        client = redis.Redis(host=r_host, port=r_port, socket_timeout=2.0)
        client.flushdb()
        log.info("Redis cache flushed successfully at %s:%d", r_host, r_port)
    except Exception as exc:
        log.warning(
            "Redis cache flush failed (unreachable at %s): %s. Continuing (cache flush is best-effort).",
            redis_host,
            exc,
        )


def get_or_create_collection(
    client: Any,
    collection_name: str | None = None,
    embed_model: str = "nomic-embed-text",
    vector_dim: int = 768,
) -> Any:
    """Get or create the ChromaDB collection preserving existing chunks."""
    name = collection_name or os.environ.get("CHROMA_COLLECTION", "rahbar_docs")
    col_meta = {
        "hnsw:space": "cosine",
        "embed_model": embed_model,
        "dimension": vector_dim,
        "vector_dim": vector_dim,
    }
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(name=name, metadata=col_meta)
    elif hasattr(client, "get_collection"):
        try:
            return client.get_collection(name)
        except Exception:
            return client.create_collection(name=name, metadata=col_meta)
    else:
        return client.create_collection(name=name, metadata=col_meta)


def get_existing_docs_in_chroma(collection: Any) -> dict[str, list[dict[str, Any]]]:
    """Retrieve all chunks currently stored in ChromaDB grouped by doc_id.

    Returns:
        dict mapping doc_id -> list of chunk items [{"id": id, "metadata": meta}]
    """
    try:
        data = collection.get(include=["metadatas"])
    except Exception as exc:
        log.warning("Could not fetch existing chunks from ChromaDB: %s", exc)
        return {}

    ids = data.get("ids") or []
    metas = data.get("metadatas") or []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk_id, meta in zip(ids, metas):
        meta = meta or {}
        doc_id = meta.get("doc_id") or meta.get("source", "")
        if doc_id not in grouped:
            grouped[doc_id] = []
        grouped[doc_id].append({"id": chunk_id, "metadata": meta})

    return grouped


def delete_doc_chunks(collection: Any, doc_id: str) -> int:
    """Delete all chunks belonging to doc_id from ChromaDB."""
    try:
        existing = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        chunk_count = len(existing.get("ids", []))
        if chunk_count > 0:
            collection.delete(where={"doc_id": doc_id})
        return chunk_count
    except Exception as exc:
        log.warning("Failed to delete chunks for doc_id '%s': %s", doc_id, exc)
        return 0


def archive_superseded_version(
    collection: Any,
    doc_id: str,
    old_version: str,
    old_file_path: str | Path | None = None,
    archive_base_dir: str = "/data/archive",
) -> int:
    """Archive an older version of a document.

    1. Moves old file to /data/archive/<doc_id>/<old_version>/
    2. Sets metadata field 'archived': True on all old chunks in ChromaDB.
    """
    # 1. Move old file to archive folder if it exists
    if old_file_path:
        old_path = Path(old_file_path)
        if old_path.is_file():
            dest_dir = Path(archive_base_dir) / doc_id / old_version
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / old_path.name
                shutil.move(str(old_path), str(dest_path))
                log.info("Archived old document file: %s -> %s", old_path, dest_path)
            except Exception as exc:
                log.warning("Could not move old file to archive: %s", exc)

    # 2. Update ChromaDB chunks for this doc_id and old_version to archived=True
    updated_count = 0
    try:
        existing = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids = existing.get("ids") or []
        metas = existing.get("metadatas") or []

        target_ids = []
        target_metas = []
        for c_id, meta in zip(ids, metas):
            meta = dict(meta or {})
            if meta.get("doc_version") == old_version and not meta.get("archived", False):
                meta["archived"] = True
                meta["superseded"] = True
                target_ids.append(c_id)
                target_metas.append(meta)

        if target_ids:
            collection.update(ids=target_ids, metadatas=target_metas)
            updated_count = len(target_ids)
            log.info("Archived %d chunk(s) for doc_id %s (version: %s)", updated_count, doc_id, old_version)
    except Exception as exc:
        log.warning("Failed to update archived chunks for doc_id '%s': %s", doc_id, exc)

    return updated_count


def rollback(
    doc_id: str,
    version: str,
    chroma_host: str = "chroma",
    chroma_port: int = 8000,
    collection_name: str | None = None,
) -> dict[str, int]:
    """Roll back a document: sets archived=false for version and archived=true for current.

    Effectively restores the specified older version as the active document.
    """
    client = get_chroma_client(chroma_host, chroma_port)
    col_name = collection_name or os.environ.get("CHROMA_COLLECTION", "rahbar_docs")
    collection = client.get_collection(col_name)

    existing = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    ids = existing.get("ids") or []
    metas = existing.get("metadatas") or []

    restored_ids = []
    restored_metas = []
    archived_ids = []
    archived_metas = []

    for c_id, meta in zip(ids, metas):
        meta = dict(meta or {})
        c_ver = str(meta.get("doc_version", ""))
        if c_ver == str(version):
            meta["archived"] = False
            meta["superseded"] = False
            restored_ids.append(c_id)
            restored_metas.append(meta)
        else:
            meta["archived"] = True
            meta["superseded"] = True
            archived_ids.append(c_id)
            archived_metas.append(meta)

    if restored_ids:
        collection.update(ids=restored_ids, metadatas=restored_metas)
    if archived_ids:
        collection.update(ids=archived_ids, metadatas=archived_metas)

    # Flush Redis cache on rollback
    flush_redis_cache(os.environ.get("REDIS_HOST", "redis"))

    print(f"Rollback complete for doc_id '{doc_id}': restored version '{version}' ({len(restored_ids)} chunks), archived current ({len(archived_ids)} chunks)")
    return {"restored": len(restored_ids), "archived": len(archived_ids)}


def store_chunks(
    chunks: list[dict[str, Any]],
    embed_metadata: dict[str, Any],
    chroma_host: str,
    redis_host: str,
    replace_all: bool = True,
) -> int:
    """Store chunks in Chroma collection 'rahbar_docs' and flush cache.

    If replace_all is True, drops and recreates collection (clean full replace).
    Otherwise, uses get_or_create_collection for incremental updates.
    """
    collection_name = os.environ.get("CHROMA_COLLECTION", "rahbar_docs")
    embed_model = embed_metadata.get("embed_model", "nomic-embed-text")
    vector_dim = embed_metadata.get("vector_dim", 768)

    client = get_chroma_client(chroma_host)

    if replace_all:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = get_or_create_collection(
        client=client,
        collection_name=collection_name,
        embed_model=embed_model,
        vector_dim=vector_dim,
    )
    if hasattr(collection, "modify"):
        try:
            collection.modify(metadata={
                "hnsw:space": "cosine",
                "embed_model": embed_model,
                "dimension": vector_dim,
                "vector_dim": vector_dim,
            })
        except Exception:
            pass

    if chunks:
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        embeddings = [c["vector"] for c in chunks]
        metadatas = [
            {
                "doc_id": str(c.get("doc_id", "")),
                "doc_version": str(c.get("doc_version", "")),
                "raw_text_hash": str(c.get("raw_text_hash", "")),
                "archived": bool(c.get("archived", False)),
                "source": str(c.get("source", "")),
                "section": str(c.get("section", "")),
                "page": int(c.get("page", 1)),
                "superseded": bool(c.get("superseded", False)),
                "embed_model": str(embed_model),
            }
            for c in chunks
        ]

        # Use upsert when available, fall back to add
        if hasattr(collection, "upsert"):
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
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
    print(f"\n--- Stored {len(chunks)} chunk(s) in '{collection_name}' ---")
    for src, count in sorted(counts.items()):
        print(f"  • {src}: {count} chunk(s)")
    print("------------------------------------------------------------------\n")

    return len(chunks)


def store(
    chunks: list[Chunk],
    vectors: list[list[float]],
    settings: IndexerSettings,
    replace_all: bool = True,
) -> int:
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
            "doc_id": c.doc_id,
            "raw_text_hash": c.raw_text_hash,
            "archived": c.archived,
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
        replace_all=replace_all,
    )
