"""Unit tests for the Store stage (indexer/pipeline/store.py).

All tests mock ChromaDB and Redis clients and run offline with no external services.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from pipeline.config import IndexerSettings
from pipeline.models import Chunk
from pipeline.store import flush_redis_cache, store, store_chunks


class FakeChromaCollection:
    def __init__(self, name: str, metadata: dict) -> None:
        self.name = name
        self.metadata = metadata
        self.added: dict[str, Any] = {}
        self.modified_metadata: dict[str, Any] = {}

    def modify(self, metadata: dict) -> None:
        self.modified_metadata = metadata

    def add(self, ids, documents, embeddings, metadatas) -> None:
        self.added = {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeChromaCollection] = {}
        self.deleted: list[str] = []

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)
        if name in self.collections:
            del self.collections[name]

    def create_collection(self, name: str, metadata: dict) -> FakeChromaCollection:
        col = FakeChromaCollection(name, metadata)
        self.collections[name] = col
        return col


def make_test_chunks(n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk-{i}",
            "text": f"Text of chunk {i}",
            "source": f"policy_{i % 2}.pdf",
            "section": f"Section {i}",
            "page": i + 1,
            "doc_version": "2026-06",
            "superseded": False,
            "vector": [0.1] * 768,
        }
        for i in range(n)
    ]


def test_store_chunks_creates_collection_and_stores_metadata() -> None:
    fake_client = FakeChromaClient()
    chunks = make_test_chunks(4)
    embed_meta = {"embed_model": "nomic-embed-text", "vector_dim": 768}

    with patch("pipeline.store.get_chroma_client", return_value=fake_client), \
         patch("pipeline.store.flush_redis_cache") as mock_flush:

        count = store_chunks(
            chunks=chunks,
            embed_metadata=embed_meta,
            chroma_host="chroma:8000",
            redis_host="redis:6379",
        )

        assert count == 4
        assert "rahbar_docs" in fake_client.deleted
        assert "rahbar_docs" in fake_client.collections

        col = fake_client.collections["rahbar_docs"]
        # Verify collection metadata
        assert col.metadata["embed_model"] == "nomic-embed-text"
        assert col.metadata["dimension"] == 768
        assert col.modified_metadata["embed_model"] == "nomic-embed-text"

        # Verify chunk entries
        assert len(col.added["ids"]) == 4
        assert len(col.added["documents"]) == 4
        assert len(col.added["embeddings"]) == 4
        assert len(col.added["metadatas"]) == 4

        first_meta = col.added["metadatas"][0]
        assert first_meta["source"] == "policy_0.pdf"
        assert first_meta["section"] == "Section 0"
        assert first_meta["page"] == 1
        assert first_meta["doc_version"] == "2026-06"
        assert first_meta["superseded"] is False
        assert first_meta["embed_model"] == "nomic-embed-text"

        mock_flush.assert_called_once_with("redis:6379")


def test_flush_redis_cache_handles_connection_error_gracefully() -> None:
    # When Redis is unreachable, flush_redis_cache must log a warning and not raise
    with patch("redis.Redis") as mock_redis_cls:
        instance = MagicMock()
        instance.flushdb.side_effect = ConnectionError("Could not connect to Redis")
        mock_redis_cls.return_value = instance

        # Should not raise exception
        flush_redis_cache("unreachable-redis:6379")
        instance.flushdb.assert_called_once()


def test_store_pipeline_adapter() -> None:
    fake_client = FakeChromaClient()
    chunk_objs = [
        Chunk(
            id="c1",
            text="Chunk text 1",
            source="rules.pdf",
            section="1. Rules",
            page=1,
            doc_version="2026",
            superseded=False,
        )
    ]
    vectors = [[0.2] * 768]
    settings = IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="chroma",
        chroma_port=8000,
        collection="rahbar_docs",
        corpus_dir="/corpus",
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://host.minikube.internal:11434",
    )

    with patch("pipeline.store.get_chroma_client", return_value=fake_client), \
         patch("pipeline.store.flush_redis_cache"):
        count = store(chunk_objs, vectors, settings)
        assert count == 1
        assert "rahbar_docs" in fake_client.collections

