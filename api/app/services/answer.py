"""Turns a question into an AskResponse. Production RAG query pipeline.

Orchestrates semantic caching, retrieval with relevance floor cutoff, honest abstention,
LLM generation with retry/fallback, citation validation, and structured JSON observability.
"""

from __future__ import annotations

import logging
import re
import time

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..routes.metrics import (
    ABSTENTIONS,
    GENERATION_LATENCY,
    REQUEST_LATENCY,
    RETRIEVAL_LATENCY,
)
from ..schemas import AskResponse, Citation, Passage, RetrievedRef
from .cache import cache_lookup, cache_store, should_cache

log = logging.getLogger("rahbar.answer")

# Queries containing follow-up cues bypass the semantic cache
FOLLOW_UP_PATTERN = re.compile(
    r"\b(follow up|follow-up|previous answer|as you said|earlier|what about that)\b",
    re.IGNORECASE,
)


def validate_citations(citations: list[Citation], passages: list[Passage]) -> list[Citation]:
    """Ensure every citation strictly matches a retrieved passage by source and section."""
    if not passages:
        return []

    valid: list[Citation] = []
    for cit in citations:
        cit_src = cit.source.lower()
        cit_sec = cit.section.lower()

        matched = any(
            cit_src == p.source.lower()
            and (cit_sec in p.section.lower() or p.section.lower() in cit_sec)
            for p in passages
        )
        if matched:
            valid.append(cit)
        else:
            log.warning("Dropping ungrounded citation to %s (%s)", cit.source, cit.section)

    return valid


def answer_question(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    settings: Settings,
) -> AskResponse:
    """Execute the full RAG pipeline: cache check -> retrieval -> abstention -> generation -> cache store."""
    start_time = time.perf_counter()
    request_id = request_id_var.get()
    question_vector: list[float] | None = None

    # 1. Semantic Cache Check (skip if follow-up query)
    is_followup = bool(FOLLOW_UP_PATTERN.search(question))
    if not is_followup and hasattr(retriever, "embed_query"):
        try:
            question_vector = retriever.embed_query(question)
            cached_answer = cache_lookup(question_vector, threshold=settings.cache_threshold)
            if cached_answer is not None:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
                log.info(
                    "answered from cache",
                    extra={
                        "request_id": request_id,
                        "cached": True,
                        "abstained": not cached_answer.answered_from_context,
                        "latency_ms": elapsed_ms,
                    },
                )
                return AskResponse(
                    request_id=request_id,
                    answer=cached_answer.answer,
                    citations=cached_answer.citations,
                    answered_from_context=cached_answer.answered_from_context,
                    confidence=cached_answer.confidence,
                    retrieved=[],
                    provider=f"{provider.name}-cached",
                )
        except Exception as exc:
            log.warning("Cache check error: %s. Continuing with retrieval.", exc)

    # 2. Retrieval Stage with Latency Metric
    retrieval_start = time.perf_counter()
    passages = retriever.retrieve(question, k=settings.top_k)
    RETRIEVAL_LATENCY.observe(time.perf_counter() - retrieval_start)

    # Apply relevance floor filter
    passages = [p for p in passages if p.score >= settings.relevance_floor]

    # 3. Honest Abstention if no chunks meet the relevance floor
    if not passages:
        ABSTENTIONS.inc()
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        log.info(
            "answered with abstention (below relevance floor)",
            extra={
                "request_id": request_id,
                "top_k": settings.top_k,
                "chunk_ids": [],
                "similarity_scores": [],
                "abstained": True,
                "latency_ms": elapsed_ms,
            },
        )
        return AskResponse(
            request_id=request_id,
            answer="I am sorry, but the provided corpus does not contain information to answer this question.",
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=[],
            provider=provider.name,
        )

    # 4. LLM Generation Stage with Latency Metric
    gen_start = time.perf_counter()
    answer = provider.generate(question, passages)
    GENERATION_LATENCY.observe(time.perf_counter() - gen_start)

    if not answer.answered_from_context:
        ABSTENTIONS.inc()

    # 5. Citation Grounding Validation
    answer.citations = validate_citations(answer.citations, passages)

    # 6. Semantic Cache Store (if answer satisfies cacheability criteria)
    if question_vector and should_cache(answer):
        cache_store(question_vector, answer)

    # 7. Total Request Latency Metric and Structured Log
    total_elapsed = time.perf_counter() - start_time
    REQUEST_LATENCY.observe(total_elapsed)

    log.info(
        "answered",
        extra={
            "request_id": request_id,
            "retriever": retriever.name,
            "provider": provider.name,
            "top_k": settings.top_k,
            "chunk_ids": [p.chunk_id for p in passages],
            "similarity_scores": [round(p.score, 4) for p in passages],
            "abstained": not answer.answered_from_context,
            "latency_ms": round(total_elapsed * 1000, 1),
        },
    )

    return AskResponse(
        request_id=request_id,
        answer=answer.answer,
        citations=answer.citations,
        answered_from_context=answer.answered_from_context,
        confidence=answer.confidence,
        retrieved=[
            RetrievedRef(
                chunk_id=p.chunk_id,
                source=p.source,
                section=p.section,
                page=p.page,
                score=p.score,
            )
            for p in passages
        ],
        provider=provider.name,
    )
