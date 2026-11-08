from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, Request

from .config import Settings
from .providers.llm.factory import get_provider
from .retrieval.stub import StubRetriever
from .routes import ask, health, metrics
from .routes.metrics import REQUESTS

app = FastAPI(title="RAHBAR API")


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    req_id = (
        request.headers.get("x-request-id")
        or request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    REQUESTS.labels(path=request.url.path, status=str(response.status_code)).inc()
    return response


settings = Settings(
    retriever=os.getenv("RETRIEVER", "stub"),
    llm_provider=os.getenv("LLM_PROVIDER", "stub"),
    chroma_host=os.getenv("CHROMA_HOST", "chroma"),
    redis_host=os.getenv("REDIS_HOST", "redis"),
    embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    top_k=int(os.getenv("TOP_K", "5")),
    relevance_floor=float(os.getenv("RELEVANCE_FLOOR", "0.3")),
    cache_threshold=float(os.getenv("CACHE_THRESHOLD", "0.92")),
    temperature=float(os.getenv("TEMPERATURE", "0.1")),
)

app.state.settings = settings
app.state.retriever = StubRetriever()
app.state.provider = get_provider(settings)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(metrics.router)
