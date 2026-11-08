"""Prometheus metrics at /metrics (section 4.8).

Exposes request counters, latency histograms, token usage, fallbacks,
and semantic cache metrics.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

router = APIRouter()

# HTTP & API latency metrics
REQUESTS = Counter("rahbar_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram(
    "rahbar_request_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
REQUEST_LATENCY = Histogram(
    "rahbar_request_latency_seconds",
    "Total request latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
RETRIEVAL_LATENCY = Histogram(
    "rahbar_retrieval_latency_seconds",
    "Retrieval stage latency in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
GENERATION_LATENCY = Histogram(
    "rahbar_generation_latency_seconds",
    "LLM generation stage latency in seconds",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)

# LLM generation, guardrail, and semantic cache metrics
FALLBACKS = Counter("rahbar_fallbacks_total", "Number of fallback responses triggered")
ABSTENTIONS = Counter("rahbar_abstentions_total", "Number of abstentions triggered")
TOKENS_IN = Counter("rahbar_tokens_in_total", "Total prompt tokens sent to LLMs")
TOKENS_OUT = Counter("rahbar_tokens_out_total", "Total completion tokens received from LLMs")
CACHE_HITS = Counter("rahbar_cache_hits_total", "Number of semantic cache hits")
CACHE_MISSES = Counter("rahbar_cache_misses_total", "Number of semantic cache misses")


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
