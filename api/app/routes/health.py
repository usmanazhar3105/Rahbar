"""Liveness and readiness probes (section 4.3).

- /healthz: Liveness probe. Returns 200 immediately, never touches dependencies.
- /readyz: Readiness probe. Verifies Chroma is reachable and non-empty, Redis is reachable,
  and collection embedding metadata matches the pinned EMBED_MODEL.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Request, Response

log = logging.getLogger("rahbar.health")

router = APIRouter()


def get_chroma_client(host: str, port: int) -> Any:
    """Helper to obtain a Chroma HTTP client with telemetry disabled."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe. Must NEVER touch Chroma, Redis, or the LLM."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, object]:
    """Readiness probe.

    Verifies:
      1. Chroma is reachable and collection exists.
      2. Collection is non-empty (collection.count() > 0).
      3. Collection embedding model matches settings.embed_model.
      4. Redis is reachable via ping().
    """
    if os.getenv("CI_MODE") == "true":
        return {"status": "ready", "mode": "ci"}

    settings = getattr(request.app.state, "settings", None)
    if not settings:
        response.status_code = 503
        return {"status": "not ready", "reason": "settings not initialized"}

    # 1 & 2. Check ChromaDB and non-empty collection
    try:
        chroma_client = get_chroma_client(settings.chroma_host, settings.chroma_port)
        collection = chroma_client.get_collection(settings.collection)

        # Check collection is non-empty
        count = collection.count()
        if count == 0:
            response.status_code = 503
            return {
                "status": "not ready",
                "reason": f"chroma: collection '{settings.collection}' is empty",
                "checks": {"chroma": "empty", "count": 0},
            }

        # Check embedding model pin guardrail
        col_meta = collection.metadata or {}
        actual_model = col_meta.get("embed_model")
        if actual_model and actual_model != settings.embed_model:
            response.status_code = 503
            log.error(
                "EMBED_MODEL mismatch: ConfigMap says %s but index was built with %s. Refusing to serve.",
                settings.embed_model,
                actual_model,
            )
            return {
                "status": "not ready",
                "reason": (
                    f"EMBED_MODEL mismatch: ConfigMap says {settings.embed_model} "
                    f"but index was built with {actual_model}"
                ),
                "checks": {"chroma": "model_mismatch", "expected": settings.embed_model, "actual": actual_model},
            }

    except Exception as exc:
        response.status_code = 503
        return {
            "status": "not ready",
            "reason": f"chroma: {exc}",
            "checks": {"chroma": "unreachable"},
        }

    # 3. Check Redis reachability
    try:
        import redis

        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            socket_timeout=2.0,
        )
        r.ping()
    except Exception as exc:
        response.status_code = 503
        return {
            "status": "not ready",
            "reason": f"redis: {exc}",
            "checks": {"chroma": "ready", "redis": "unreachable"},
        }

    return {
        "status": "ready",
        "checks": {
            "chroma": "ready",
            "collection": settings.collection,
            "chunks_indexed": count,
            "embed_model": settings.embed_model,
            "redis": "ready",
        },
    }
