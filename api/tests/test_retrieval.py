"""Unit tests for the Chroma retrieval stage and embedding pin guardrails (api/app/retrieval/chroma.py).

All tests mock ChromaDB and Ollama HTTP endpoints; zero network or model calls.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from app.config import Settings
from app.retrieval.chroma import (
    ChromaRetriever,
    should_filter_superseded,
    verify_embed_model,
)
from app.schemas import Passage, RetrievedChunk


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        retriever="chroma",
        embed_model="nomic-embed-text",
        chroma_host="chroma",
        chroma_port=8000,
        collection="rahbar_docs",
        top_k=5,
        relevance_floor=0.3,
        ollama_url="http://host.minikube.internal:11434",
    )


class FakeCollection:
    def __init__(self, name: str, metadata: dict | None = None) -> None:
        self.name = name
        self.metadata = metadata or {"embed_model": "nomic-embed-text", "dimension": 768}
        self.last_query_kwargs: dict = {}

    def query(self, **kwargs):
        self.last_query_kwargs = kwargs
        # Return 3 hits:
        # Hit 1: distance 0.1 -> similarity 0.9 (above floor 0.3)
        # Hit 2: distance 0.5 -> similarity 0.5 (above floor 0.3)
        # Hit 3: distance 0.85 -> similarity 0.15 (below floor 0.3, must be dropped)
        return {
            "ids": [["chunk-1", "chunk-2", "chunk-3"]],
            "documents": [["Document text 1", "Document text 2", "Document text 3"]],
            "metadatas": [[
                {"source": "rules-2026.pdf", "section": "1. Attendance", "page": 2, "superseded": False},
                {"source": "rules-2026.pdf", "section": "2. Warnings", "page": 5, "superseded": False},
                {"source": "noise.pdf", "section": "Irrelevant", "page": 1, "superseded": False},
            ]],
            "distances": [[0.1, 0.5, 0.85]],
        }


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self._collection = collection

    def get_collection(self, name: str):
        return self._collection


def test_should_filter_superseded_detection() -> None:
    # Standard questions filter out superseded documents
    assert should_filter_superseded("When is a student placed on academic warning?")
    assert should_filter_superseded("What is the fee per credit hour?")

    # Historical questions must NOT filter superseded documents
    assert not should_filter_superseded("What was the minimum attendance in the old version?")
    assert not should_filter_superseded("What was the previous policy on course repetition?")
    assert not should_filter_superseded("What were the 2023 rules for probation?")
    assert not should_filter_superseded("Show me the repealed regulations.")


def test_chroma_retriever_query_and_relevance_floor(mock_settings: Settings) -> None:
    fake_col = FakeCollection("rahbar_docs")
    fake_client = FakeClient(fake_col)

    retriever = ChromaRetriever(mock_settings, client=fake_client)

    # Mock Ollama embedding call
    def mock_post(self, url, json=None, **kwargs):
        return httpx.Response(200, json={"embeddings": [[0.1] * 768]})

    with patch.object(httpx.Client, "post", mock_post):
        # 1. Test ordinary query with superseded filter applied
        passages = retriever.retrieve("What is the minimum attendance?", k=5)

        # Hit 3 (similarity 0.15 < floor 0.3) must have been dropped
        assert len(passages) == 2
        assert isinstance(passages[0], Passage)
        assert isinstance(passages[0], RetrievedChunk)

        # Check mapping and similarity score (1 - distance)
        assert passages[0].chunk_id == "chunk-1"
        assert passages[0].score == 0.9  # 1 - 0.1
        assert passages[0].section == "1. Attendance"
        assert passages[0].page == 2

        assert passages[1].chunk_id == "chunk-2"
        assert passages[1].score == 0.5  # 1 - 0.5

        # Check where filter was applied
        assert fake_col.last_query_kwargs.get("where") == {"superseded": False}
        assert fake_col.last_query_kwargs.get("n_results") == 5


def test_chroma_retriever_removes_superseded_filter_for_old_policy_query(mock_settings: Settings) -> None:
    fake_col = FakeCollection("rahbar_docs")
    fake_client = FakeClient(fake_col)

    retriever = ChromaRetriever(mock_settings, client=fake_client)

    def mock_post(self, url, json=None, **kwargs):
        return httpx.Response(200, json={"embeddings": [[0.1] * 768]})

    with patch.object(httpx.Client, "post", mock_post):
        # Query asking for previous policy
        _ = retriever.retrieve("What was the rule in the 2023 rules?", k=5)

        # where clause must be omitted when historical queries are detected
        assert "where" not in fake_col.last_query_kwargs


def test_verify_embed_model_success(mock_settings: Settings) -> None:
    fake_col = FakeCollection(
        "rahbar_docs",
        metadata={"embed_model": "nomic-embed-text", "dimension": 768},
    )
    fake_client = FakeClient(fake_col)

    # Should succeed without exception
    verify_embed_model(
        expected_model="nomic-embed-text",
        expected_dim=768,
        settings=mock_settings,
        client=fake_client,
    )


def test_verify_embed_model_mismatch_raises_runtime_error(mock_settings: Settings) -> None:
    # Collection built with different model
    fake_col = FakeCollection(
        "rahbar_docs",
        metadata={"embed_model": "all-minilm", "dimension": 384},
    )
    fake_client = FakeClient(fake_col)

    with pytest.raises(RuntimeError) as exc_info:
        verify_embed_model(
            expected_model="nomic-embed-text",
            expected_dim=768,
            settings=mock_settings,
            client=fake_client,
        )

    msg = str(exc_info.value)
    assert "EMBED_MODEL mismatch" in msg
    assert "nomic-embed-text" in msg
    assert "all-minilm" in msg
    assert "Refusing to serve" in msg

