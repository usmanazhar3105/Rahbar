"""Semantic cache for RAG question-answer pairs using Redis.

Stores question embedding vectors and Answer payloads as Redis hashes with a 24-hour TTL.
Lookup computes cosine similarity against stored vectors; hits above the threshold
bypass retrieval and model generation.

Threshold configuration:
0.92 chosen as the lowest threshold with zero false hits on our test questions.
Tried 0.85 (false hit: 'fees' matched 'scholarship amounts') and 0.90 (false hit:
'course drop deadline' matched 'semester freeze deadline').
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from typing import Any

from ..config import get_settings
from ..routes.metrics import CACHE_HITS, CACHE_MISSES
from ..schemas import Answer

log = logging.getLogger("rahbar.cache")

DEFAULT_THRESHOLD = float(os.environ.get("CACHE_THRESHOLD", "0.92"))
DEFAULT_TTL_SECONDS = 86400  # 24 hours


def get_redis_client() -> Any:
    """Connect to Redis at REDIS_HOST (port 6379 or parsed from host)."""
    import redis

    settings = get_settings()
    host_str = os.environ.get("REDIS_HOST", settings.redis_host)
    raw_port = os.environ.get("REDIS_PORT")
    if raw_port and ":" in str(raw_port):
        port = int(str(raw_port).split(":")[-1])
    elif raw_port:
        try:
            port = int(raw_port)
        except ValueError:
            port = settings.redis_port
    else:
        port = settings.redis_port

    if ":" in host_str:
        parts = host_str.split(":")
        host_str = parts[0]
        port = int(parts[1])

    return redis.Redis(
        host=host_str,
        port=port,
        socket_timeout=2.0,
        decode_responses=True,
    )


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)


def should_cache(answer: Answer) -> bool:
    """Determine whether an answer qualifies for semantic caching.

    Never-cache rules:
    - Never cache abstentions (answered_from_context is False)
    - Never cache answers with low confidence
    - Never cache empty or failed validation answers
    """
    if not answer.answered_from_context:
        return False

    if answer.confidence == "low":
        return False

    if not answer.answer or not answer.answer.strip():
        return False

    return True


def never_cache(answer: Answer) -> bool:
    """Inverted predicate for never-cache / skip-cache rule enforcement."""
    return not should_cache(answer)


skip_cache = never_cache


def cache_lookup(
    question_vector: list[float],
    threshold: float = DEFAULT_THRESHOLD,
    client: Any | None = None,
) -> Answer | None:
    """Look up a cached Answer by computing cosine similarity against stored vectors.

    Returns the stored Answer if similarity >= threshold, else None.
    """
    if not question_vector:
        return None

    try:
        r = client or get_redis_client()
        keys = r.keys("cache:*")
        if not keys:
            CACHE_MISSES.inc()
            return None

        best_score = -1.0
        best_answer_json: str | None = None

        for k in keys:
            data = r.hgetall(k)
            if not data or "vector" not in data or "answer" not in data:
                continue

            try:
                stored_vec = json.loads(data["vector"])
                score = cosine_similarity(question_vector, stored_vec)
                if score > best_score:
                    best_score = score
                    best_answer_json = data["answer"]
            except Exception:
                continue

        if best_score >= threshold and best_answer_json:
            CACHE_HITS.inc()
            log.info("Semantic cache hit (similarity=%.4f >= threshold=%.4f)", best_score, threshold)
            return Answer.model_validate_json(best_answer_json)

        CACHE_MISSES.inc()
        return None

    except Exception as exc:
        log.warning("Semantic cache lookup failed (Redis error): %s. Continuing with miss.", exc)
        CACHE_MISSES.inc()
        return None


def cache_store(
    question_vector: list[float],
    answer: Answer,
    client: Any | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> str | None:
    """Store a question vector and Answer in Redis with a 24-hour TTL."""
    if not should_cache(answer):
        log.debug("Skipping cache store: answer fails should_cache criteria")
        return None

    if not question_vector:
        return None

    try:
        r = client or get_redis_client()
        cache_id = str(uuid.uuid4())
        key = f"cache:{cache_id}"

        r.hset(
            key,
            mapping={
                "vector": json.dumps(question_vector),
                "answer": answer.model_dump_json(),
            },
        )
        r.expire(key, ttl_seconds)
        log.info("Stored answer in semantic cache with key %s (TTL=%ds)", key, ttl_seconds)
        return key

    except Exception as exc:
        log.warning("Semantic cache store failed (Redis error): %s", exc)
        return None


def cache_invalidate(client: Any | None = None) -> bool:
    """Invalidate semantic cache via FLUSHDB (called after re-indexing)."""
    try:
        r = client or get_redis_client()
        r.flushdb()
        log.info("Semantic cache invalidated (FLUSHDB)")
        return True
    except Exception as exc:
        log.warning("Semantic cache invalidation failed: %s", exc)
        return False


clear_cache = cache_invalidate
flush_cache = cache_invalidate

