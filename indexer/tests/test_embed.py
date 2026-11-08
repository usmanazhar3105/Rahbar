"""Unit tests for the Embed stage (indexer/pipeline/embed.py).

All tests use mocks and run offline with no model, no cluster, and no network calls.
"""

from __future__ import annotations

import httpx
import pytest
from pipeline.config import IndexerSettings
from pipeline.embed import (
    OllamaUnreachableError,
    embed,
    embed_chunks,
)


def make_dummy_chunks(n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk-{i}",
            "text": f"This is chunk content number {i}",
            "source": "doc.pdf",
            "section": "1. Section",
            "page": 1,
            "doc_version": "2026-06",
            "superseded": False,
        }
        for i in range(n)
    ]


def test_embed_chunks_adds_vector_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = make_dummy_chunks(5)
    fake_dim = 768

    def mock_post(self, url, json=None, **kwargs):
        assert "api/embed" in str(url)
        assert json["model"] == "nomic-embed-text"
        input_texts = json["input"]
        # Return fake vectors of length fake_dim
        fake_embeddings = [[0.1] * fake_dim for _ in input_texts]
        return httpx.Response(200, json={"embeddings": fake_embeddings})

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    result_chunks, metadata = embed_chunks(
        chunks,
        ollama_host="http://localhost:11434",
        model="nomic-embed-text",
    )

    assert len(result_chunks) == 5
    for c in result_chunks:
        assert "vector" in c
        assert len(c["vector"]) == fake_dim

    assert metadata == {
        "embed_model": "nomic-embed-text",
        "vector_dim": fake_dim,
    }


def test_embed_chunks_batching_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    # 70 chunks with BATCH_SIZE=32 -> 3 batches (32, 32, 6)
    chunks = make_dummy_chunks(70)
    batch_calls: list[int] = []

    def mock_post(self, url, json=None, **kwargs):
        batch_size = len(json["input"])
        batch_calls.append(batch_size)
        fake_embeddings = [[0.05] * 768 for _ in range(batch_size)]
        return httpx.Response(200, json={"embeddings": fake_embeddings})

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    result_chunks, metadata = embed_chunks(
        chunks,
        ollama_host="http://localhost:11434",
        model="nomic-embed-text",
    )

    assert batch_calls == [32, 32, 6]
    assert len(result_chunks) == 70
    assert all("vector" in c for c in result_chunks)


def test_embed_chunks_unreachable_host_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = make_dummy_chunks(3)
    target_host = "http://unreachable.minikube.internal:11434"

    def mock_post(self, url, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    with pytest.raises(OllamaUnreachableError) as exc_info:
        embed_chunks(chunks, ollama_host=target_host, model="nomic-embed-text")

    err_msg = str(exc_info.value)
    # Must contain the host URL in the message
    assert target_host in err_msg
    assert "unreachable" in err_msg.lower()


def test_embed_adapter_for_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = ["Sample text 1", "Sample text 2"]
    fake_dim = 768

    def mock_post(self, url, json=None, **kwargs):
        return httpx.Response(200, json={"embeddings": [[0.5] * fake_dim for _ in json["input"]]})

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    settings = IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="chroma",
        chroma_port=8000,
        collection="rahbar",
        corpus_dir="/corpus",
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://host.minikube.internal:11434",
    )

    vectors = embed(texts, settings)
    assert len(vectors) == 2
    assert len(vectors[0]) == fake_dim
    assert len(vectors[1]) == fake_dim

