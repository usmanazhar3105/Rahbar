"""Unit tests for the semantic cache layer (api/app/services/cache.py).

All tests mock the Redis client; zero network or real Redis calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.routes.metrics import CACHE_HITS, CACHE_MISSES
from app.schemas import Answer, Citation
from app.services.cache import (
    cache_invalidate,
    cache_lookup,
    cache_store,
    cosine_similarity,
    never_cache,
    should_cache,
    skip_cache,
)


def make_valid_answer(
    confidence: str = "high",
    answered_from_context: bool = True,
    text: str = "Attendance of 80% is mandatory.",
) -> Answer:
    return Answer(
        reasoning="Grounded in clause 1.",
        answer=text,
        citations=[Citation(source="rules.pdf", section="1. Attendance", page=1)],
        answered_from_context=answered_from_context,
        confidence=confidence,  # type: ignore[arg-type]
    )


class FakeRedisClient:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}
        self.flushed = False

    def keys(self, pattern: str) -> list[str]:
        return list(self.store.keys())

    def hgetall(self, key: str) -> dict[str, str]:
        return self.store.get(key, {})

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self.store[key] = mapping
        return len(mapping)

    def expire(self, key: str, seconds: int) -> bool:
        self.ttls[key] = seconds
        return True

    def flushdb(self) -> bool:
        self.store.clear()
        self.flushed = True
        return True


def test_cosine_similarity() -> None:
    # Identical vectors
    assert pytest.approx(cosine_similarity([1.0, 0.0], [1.0, 0.0])) == 1.0
    # Orthogonal vectors
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0])) == 0.0
    # Opposite vectors
    assert pytest.approx(cosine_similarity([1.0, 0.0], [-1.0, 0.0])) == -1.0
    # Mismatched lengths or empty
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_should_cache_predicates() -> None:
    valid_high = make_valid_answer(confidence="high", answered_from_context=True)
    assert should_cache(valid_high) is True
    assert never_cache(valid_high) is False
    assert skip_cache(valid_high) is False

    valid_med = make_valid_answer(confidence="medium", answered_from_context=True)
    assert should_cache(valid_med) is True

    # Never cache low confidence
    low_conf = make_valid_answer(confidence="low", answered_from_context=True)
    assert should_cache(low_conf) is False
    assert never_cache(low_conf) is True

    # Never cache abstentions
    abstention = make_valid_answer(confidence="high", answered_from_context=False)
    assert should_cache(abstention) is False
    assert never_cache(abstention) is True

    # Never cache empty answer
    empty_ans = make_valid_answer(text="")
    assert should_cache(empty_ans) is False


def test_cache_store_and_lookup_hit() -> None:
    fake_client = FakeRedisClient()
    answer = make_valid_answer()
    q_vector = [1.0, 0.0, 0.0]

    # Store
    key = cache_store(q_vector, answer, client=fake_client)
    assert key is not None
    assert key.startswith("cache:")
    assert fake_client.ttls[key] == 86400  # 24 hours

    hits_before = CACHE_HITS._value.get()

    # Exact query vector lookup -> similarity 1.0 >= 0.92
    cached = cache_lookup(q_vector, threshold=0.92, client=fake_client)
    assert cached is not None
    assert cached.answer == answer.answer
    assert cached.confidence == "high"
    assert CACHE_HITS._value.get() == hits_before + 1


def test_cache_lookup_miss_below_threshold() -> None:
    fake_client = FakeRedisClient()
    answer = make_valid_answer()
    stored_vector = [1.0, 0.0, 0.0]
    cache_store(stored_vector, answer, client=fake_client)

    misses_before = CACHE_MISSES._value.get()

    # Query with vector similarity = 0.80 (< threshold 0.92)
    # Cosine similarity between [1.0, 0.0] and [0.8, 0.6] is 0.8
    query_vector = [0.8, 0.6, 0.0]
    cached = cache_lookup(query_vector, threshold=0.92, client=fake_client)

    assert cached is None
    assert CACHE_MISSES._value.get() == misses_before + 1


def test_cache_store_ignores_abstention() -> None:
    fake_client = FakeRedisClient()
    abstention_answer = make_valid_answer(answered_from_context=False)

    key = cache_store([1.0, 0.0], abstention_answer, client=fake_client)
    assert key is None
    assert len(fake_client.store) == 0


def test_cache_invalidate() -> None:
    fake_client = FakeRedisClient()
    cache_store([1.0, 0.0], make_valid_answer(), client=fake_client)
    assert len(fake_client.store) == 1

    success = cache_invalidate(client=fake_client)
    assert success is True
    assert fake_client.flushed is True
    assert len(fake_client.store) == 0


def test_redis_connection_error_fails_gracefully() -> None:
    broken_client = MagicMock()
    broken_client.keys.side_effect = ConnectionError("Redis down")
    broken_client.hset.side_effect = ConnectionError("Redis down")
    broken_client.flushdb.side_effect = ConnectionError("Redis down")

    # Lookup returns None on error
    assert cache_lookup([1.0, 0.0], client=broken_client) is None

    # Store returns None on error
    assert cache_store([1.0, 0.0], make_valid_answer(), client=broken_client) is None

    # Invalidate returns False on error
    assert cache_invalidate(client=broken_client) is False

