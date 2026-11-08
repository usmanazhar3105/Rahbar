"""Unit and integration tests for FastAPI routes (api/app/routes/ask.py, health.py, metrics.py).

Uses FastAPI TestClient with offline StubRetriever and StubProvider; no network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.providers.llm.stub import StubProvider
from app.retrieval.stub import StubRetriever


@pytest.fixture
def client() -> TestClient:
    settings = Settings(
        retriever="stub",
        llm_provider="stub",
        embed_model="nomic-embed-text",
        chroma_host="chroma",
        chroma_port=8000,
        collection="rahbar",
        top_k=3,
        relevance_floor=0.3,
        redis_host="redis",
    )
    with TestClient(app) as test_client:
        test_client.app.state.settings = settings
        test_client.app.state.retriever = StubRetriever()
        test_client.app.state.provider = StubProvider()
        yield test_client


def test_healthz_liveness(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_readyz_success(client: TestClient) -> None:
    # Mock Chroma collection with non-empty count and matching model
    mock_collection = MagicMock()
    mock_collection.count.return_value = 15
    mock_collection.metadata = {"embed_model": "nomic-embed-text", "dimension": 768}

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    with patch("app.routes.health.get_chroma_client", return_value=mock_chroma_client), \
         patch("redis.Redis", return_value=mock_redis):

        resp = client.get("/readyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["checks"]["chunks_indexed"] == 15


def test_readyz_fails_when_collection_empty(client: TestClient) -> None:
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.metadata = {"embed_model": "nomic-embed-text"}

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    with patch("app.routes.health.get_chroma_client", return_value=mock_chroma_client), \
         patch("redis.Redis"):

        resp = client.get("/readyz")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not ready"
        assert "collection 'rahbar' is empty" in data["reason"]


def test_readyz_fails_when_redis_unreachable(client: TestClient) -> None:
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.metadata = {"embed_model": "nomic-embed-text"}

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    mock_redis = MagicMock()
    mock_redis.ping.side_effect = ConnectionError("Redis unreachable")

    with patch("app.routes.health.get_chroma_client", return_value=mock_chroma_client), \
         patch("redis.Redis", return_value=mock_redis):

        resp = client.get("/readyz")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not ready"
        assert "redis" in data["reason"]


def test_readyz_fails_when_embed_model_mismatches(client: TestClient) -> None:
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.metadata = {"embed_model": "all-minilm"}  # mismatched model!

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    with patch("app.routes.health.get_chroma_client", return_value=mock_chroma_client), \
         patch("redis.Redis"):

        resp = client.get("/readyz")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not ready"
        assert "EMBED_MODEL mismatch" in data["reason"]


def test_post_chat_and_ask_success(client: TestClient) -> None:
    # 1. Test POST /api/chat
    chat_resp = client.post(
        "/api/chat",
        json={"question": "What is the minimum attendance required?"},
        headers={"X-Request-ID": "test-req-123"},
    )
    assert chat_resp.status_code == 200
    assert chat_resp.headers.get("X-Request-ID") == "test-req-123"
    chat_data = chat_resp.json()
    assert chat_data["request_id"] == "test-req-123"
    assert chat_data["answered_from_context"] is True
    assert len(chat_data["citations"]) > 0

    # 2. Test POST /api/ask
    ask_resp = client.post(
        "/api/ask",
        json={"question": "When is a student placed on academic warning?"},
    )
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert ask_data["answered_from_context"] is True
    assert "stub" in ask_data["provider"]


def test_post_chat_validation_length(client: TestClient) -> None:
    # Too short (< 3 chars)
    short_resp = client.post("/api/chat", json={"question": "hi"})
    assert short_resp.status_code == 422

    # Too long (> 500 chars)
    long_resp = client.post("/api/chat", json={"question": "a" * 501})
    assert long_resp.status_code == 422


def test_post_chat_honest_abstention_when_no_passages(client: TestClient) -> None:
    # Empty retriever
    empty_retriever = MagicMock()
    empty_retriever.name = "empty_mock"
    empty_retriever.retrieve.return_value = []
    client.app.state.retriever = empty_retriever

    resp = client.post("/api/chat", json={"question": "How do I apply for a hostel room?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answered_from_context"] is False
    assert data["confidence"] == "low"
    assert "does not contain information" in data["answer"]


def test_metrics_endpoint_exposes_all_required_metrics(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text

    required_metrics = [
        "rahbar_requests_total",
        "rahbar_request_latency_seconds",
        "rahbar_retrieval_latency_seconds",
        "rahbar_generation_latency_seconds",
        "rahbar_cache_hits_total",
        "rahbar_abstentions_total",
        "rahbar_fallbacks_total",
        "rahbar_tokens_in_total",
        "rahbar_tokens_out_total",
    ]

    for m in required_metrics:
        assert m in body, f"Metric '{m}' missing from /metrics output"
