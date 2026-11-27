"""Turns a question into an AskResponse. Production RAG query pipeline.

Orchestrates semantic caching, retrieval with relevance floor cutoff, honest abstention,
prompt injection guardrails, LLM generation with retry/fallback, citation validation,
Prometheus metrics, and structured D1 JSON line tracing.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..routes.metrics import (
    ABSTENTIONS,
    CACHE_LOOKUP_LATENCY,
    GENERATION_LATENCY,
    GUARDRAIL_LATENCY,
    INJECTION_ATTEMPTS,
    REQUEST_LATENCY,
    RETRIEVAL_LATENCY,
    TOP1_SIMILARITY,
)
from ..schemas import AskResponse, Citation, Passage, RetrievedRef
from .cache import cache_lookup, cache_store, should_cache
from .guardrails import apply_retrieval_guardrails, check_query_guardrail
from .tracing import build_prompt, build_trace_record, compute_prompt_hash, emit_trace

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


def _record_query_trace(
    *,
    trace_id: str,
    question: str,
    rewritten_query: str | None,
    embed_model: str,
    embed_model_version: str,
    collection_name: str,
    collection_version: str,
    passages: list[Passage],
    filters_applied: dict[str, Any],
    reranked_order: list[str],
    provider_name: str,
    llm_model: str,
    model_parameters: dict[str, Any],
    response_answer: str,
    answered_from_context: bool,
    confidence: str,
    citations: list[Citation],
    guardrail_actions: list[dict[str, Any]],
    latency_per_stage: dict[str, float],
    cache_hit: bool,
    cache_status: str,
    trace_dir: str,
    trace_stdout: bool,
) -> None:
    """Build prompt hash and emit a complete D1 JSON trace record."""
    sys_p, usr_p = build_prompt(question, passages, provider_name=provider_name)
    prompt_hash = compute_prompt_hash(sys_p, usr_p)
    trace_rec = build_trace_record(
        trace_id=trace_id,
        user_query=question,
        rewritten_query=rewritten_query,
        embedding_model=embed_model,
        embedding_model_version=embed_model_version,
        collection_name=collection_name,
        collection_version=collection_version,
        retrieved_passages=passages,
        filters_applied=filters_applied,
        reranked_order=reranked_order,
        system_prompt=sys_p,
        user_prompt=usr_p,
        prompt_hash=prompt_hash,
        model=llm_model,
        model_parameters=model_parameters,
        response_answer=response_answer,
        answered_from_context=answered_from_context,
        confidence=confidence,
        citations=citations,
        guardrail_actions=guardrail_actions,
        latency_per_stage=latency_per_stage,
        cache_hit=cache_hit,
        cache_status=cache_status,
    )
    emit_trace(trace_rec, trace_dir=trace_dir, write_stdout=trace_stdout)


def answer_question(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    settings: Settings,
) -> AskResponse:
    """Execute the full RAG pipeline with full D1 observability tracing."""
    start_time = time.perf_counter()
    raw_rid = request_id_var.get()
    request_id = raw_rid if raw_rid and raw_rid != "-" else uuid.uuid4().hex
    trace_id = request_id

    normalized_q = " ".join(question.strip().split())
    rewritten_query: str | None = normalized_q if normalized_q != question else None

    embed_model = getattr(settings, "embed_model", "nomic-embed-text")
    embed_model_version = getattr(settings, "embed_model_version", "1.0")
    collection_name = getattr(settings, "collection", "rahbar_docs")
    collection_version = getattr(settings, "collection_version", "v1")
    top_k = getattr(settings, "top_k", 5)
    relevance_floor = getattr(settings, "relevance_floor", 0.3)
    cache_threshold = getattr(settings, "cache_threshold", 0.92)
    temperature = getattr(settings, "temperature", 0.1)
    llm_model = (
        "stub"
        if provider.name == "stub"
        else getattr(settings, "llm_model", "llama-3.1-8b-instant")
    )
    trace_dir = getattr(settings, "trace_dir", "")
    trace_stdout = getattr(settings, "trace_stdout", True)

    model_parameters: dict[str, Any] = {
        "provider": provider.name,
        "temperature": temperature,
        "top_k": top_k,
        "relevance_floor": relevance_floor,
        "cache_threshold": cache_threshold,
    }
    guardrail_actions: list[dict[str, Any]] = []

    # 1. Query Guardrail Check (runs before cache or retrieval)
    q_guard_start = time.perf_counter()
    query_blocked, q_guard_actions = check_query_guardrail(question)
    q_guard_elapsed = time.perf_counter() - q_guard_start
    guardrail_actions.extend(q_guard_actions)

    if query_blocked:
        INJECTION_ATTEMPTS.inc()
        ABSTENTIONS.inc()
        GUARDRAIL_LATENCY.observe(q_guard_elapsed)
        total_elapsed = time.perf_counter() - start_time
        REQUEST_LATENCY.observe(total_elapsed)

        refusal = "Request blocked by security guardrail: prompt injection directives are not permitted."
        _record_query_trace(
            trace_id=trace_id,
            question=question,
            rewritten_query=rewritten_query,
            embed_model=embed_model,
            embed_model_version=embed_model_version,
            collection_name=collection_name,
            collection_version=collection_version,
            passages=[],
            filters_applied={"guardrail_blocked": True},
            reranked_order=[],
            provider_name=provider.name,
            llm_model=llm_model,
            model_parameters=model_parameters,
            response_answer=refusal,
            answered_from_context=False,
            confidence="low",
            citations=[],
            guardrail_actions=guardrail_actions,
            latency_per_stage={
                "cache_lookup_ms": 0.0,
                "retrieval_ms": 0.0,
                "guardrail_ms": round(q_guard_elapsed * 1000, 3),
                "generation_ms": 0.0,
                "total_ms": round(total_elapsed * 1000, 3),
            },
            cache_hit=False,
            cache_status="skipped_guardrail",
            trace_dir=trace_dir,
            trace_stdout=trace_stdout,
        )
        return AskResponse(
            request_id=request_id,
            answer=refusal,
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=[],
            provider=provider.name,
        )

    # 2. Semantic Cache Check (skip if follow-up query)
    question_vector: list[float] | None = None
    is_followup = bool(FOLLOW_UP_PATTERN.search(question))
    cache_status = "skipped_followup" if is_followup else "miss"
    cache_start = time.perf_counter()

    if not is_followup and hasattr(retriever, "embed_query"):
        try:
            question_vector = retriever.embed_query(question)
            cached_answer = cache_lookup(question_vector, threshold=cache_threshold)
            if cached_answer is not None:
                cache_elapsed = time.perf_counter() - cache_start
                CACHE_LOOKUP_LATENCY.observe(cache_elapsed)
                GUARDRAIL_LATENCY.observe(q_guard_elapsed)
                total_elapsed = time.perf_counter() - start_time
                REQUEST_LATENCY.observe(total_elapsed)

                guardrail_actions.append(
                    {
                        "rail": "semantic_cache",
                        "action": "cache_hit",
                        "threshold": cache_threshold,
                    }
                )
                _record_query_trace(
                    trace_id=trace_id,
                    question=question,
                    rewritten_query=rewritten_query,
                    embed_model=embed_model,
                    embed_model_version=embed_model_version,
                    collection_name=collection_name,
                    collection_version=collection_version,
                    passages=[],
                    filters_applied={"cache_threshold": cache_threshold},
                    reranked_order=[],
                    provider_name=provider.name,
                    llm_model=llm_model,
                    model_parameters=model_parameters,
                    response_answer=cached_answer.answer,
                    answered_from_context=cached_answer.answered_from_context,
                    confidence=cached_answer.confidence,
                    citations=cached_answer.citations,
                    guardrail_actions=guardrail_actions,
                    latency_per_stage={
                        "cache_lookup_ms": round(cache_elapsed * 1000, 3),
                        "retrieval_ms": 0.0,
                        "guardrail_ms": round(q_guard_elapsed * 1000, 3),
                        "generation_ms": 0.0,
                        "total_ms": round(total_elapsed * 1000, 3),
                    },
                    cache_hit=True,
                    cache_status="hit",
                    trace_dir=trace_dir,
                    trace_stdout=trace_stdout,
                )
                log.info(
                    "answered from cache",
                    extra={
                        "request_id": request_id,
                        "cached": True,
                        "abstained": not cached_answer.answered_from_context,
                        "latency_ms": round(total_elapsed * 1000, 1),
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
            cache_status = "error_fallback_miss"
            log.warning("Cache check error: %s. Continuing with retrieval.", exc)

    cache_elapsed = time.perf_counter() - cache_start
    CACHE_LOOKUP_LATENCY.observe(cache_elapsed)
    cache_lookup_ms = round(cache_elapsed * 1000, 3)

    # 3. Retrieval Stage with Latency Metric
    retrieval_start = time.perf_counter()
    if getattr(type(retriever), "supports_trace", False) is True:
        passages, retrieval_meta = retriever.retrieve_with_trace(  # type: ignore[attr-defined]
            question,
            k=top_k,
            query_vector=question_vector,
        )
        filters_applied = dict(retrieval_meta.get("filters_applied", {}))
        reranked_order = list(retrieval_meta.get("reranked_order", [p.chunk_id for p in passages]))
        collection_name = str(retrieval_meta.get("collection_name", collection_name))
        collection_version = str(retrieval_meta.get("collection_version", collection_version))
        embed_model = str(retrieval_meta.get("embed_model", embed_model))
        embed_model_version = str(retrieval_meta.get("embed_model_version", embed_model_version))
    else:
        raw_passages = retriever.retrieve(question, k=top_k)
        passages = [p for p in raw_passages if p.score >= relevance_floor]
        filters_applied = {
            "where": None,
            "superseded_excluded": False,
            "archived_excluded": True,
            "relevance_floor": relevance_floor,
            "dropped_below_floor": len(raw_passages) - len(passages),
        }
        reranked_order = [p.chunk_id for p in passages]

    retrieval_elapsed = time.perf_counter() - retrieval_start
    RETRIEVAL_LATENCY.observe(retrieval_elapsed)
    retrieval_ms = round(retrieval_elapsed * 1000, 3)

    if passages:
        TOP1_SIMILARITY.observe(max(p.score for p in passages))

    # 4. Retrieval Context Guardrails (Prompt Injection Detection & Sanitization)
    ctx_guard_start = time.perf_counter()
    passages, ctx_guard_actions, injection_count = apply_retrieval_guardrails(passages)
    guardrail_actions.extend(ctx_guard_actions)
    if injection_count > 0:
        INJECTION_ATTEMPTS.inc(injection_count)

    # 5. Honest Abstention if no chunks meet the relevance floor
    if not passages:
        ABSTENTIONS.inc()
        guardrail_actions.append(
            {
                "rail": "relevance_floor_gate",
                "action": "abstained",
                "relevance_floor": relevance_floor,
            }
        )
        ctx_guard_elapsed = time.perf_counter() - ctx_guard_start
        total_guard_elapsed = q_guard_elapsed + ctx_guard_elapsed
        GUARDRAIL_LATENCY.observe(total_guard_elapsed)
        total_elapsed = time.perf_counter() - start_time
        REQUEST_LATENCY.observe(total_elapsed)

        abstention_msg = "I am sorry, but the provided corpus does not contain information to answer this question."
        _record_query_trace(
            trace_id=trace_id,
            question=question,
            rewritten_query=rewritten_query,
            embed_model=embed_model,
            embed_model_version=embed_model_version,
            collection_name=collection_name,
            collection_version=collection_version,
            passages=[],
            filters_applied=filters_applied,
            reranked_order=[],
            provider_name=provider.name,
            llm_model=llm_model,
            model_parameters=model_parameters,
            response_answer=abstention_msg,
            answered_from_context=False,
            confidence="low",
            citations=[],
            guardrail_actions=guardrail_actions,
            latency_per_stage={
                "cache_lookup_ms": cache_lookup_ms,
                "retrieval_ms": retrieval_ms,
                "guardrail_ms": round(total_guard_elapsed * 1000, 3),
                "generation_ms": 0.0,
                "total_ms": round(total_elapsed * 1000, 3),
            },
            cache_hit=False,
            cache_status=cache_status,
            trace_dir=trace_dir,
            trace_stdout=trace_stdout,
        )
        log.info(
            "answered with abstention (below relevance floor)",
            extra={
                "request_id": request_id,
                "top_k": top_k,
                "chunk_ids": [],
                "similarity_scores": [],
                "abstained": True,
                "latency_ms": round(total_elapsed * 1000, 1),
            },
        )
        return AskResponse(
            request_id=request_id,
            answer=abstention_msg,
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=[],
            provider=provider.name,
        )

    guardrail_actions.append(
        {
            "rail": "relevance_floor_gate",
            "action": "passed",
            "relevance_floor": relevance_floor,
            "kept_chunks": len(passages),
        }
    )

    # 6. LLM Generation Stage with Latency Metric
    gen_start = time.perf_counter()
    answer = provider.generate(question, passages)
    gen_elapsed = time.perf_counter() - gen_start
    GENERATION_LATENCY.observe(gen_elapsed)
    generation_ms = round(gen_elapsed * 1000, 3)

    if not answer.answered_from_context:
        ABSTENTIONS.inc()

    # 7. Citation Grounding Validation
    raw_citation_count = len(answer.citations)
    answer.citations = validate_citations(answer.citations, passages)
    dropped_citations = raw_citation_count - len(answer.citations)
    guardrail_actions.append(
        {
            "rail": "citation_grounding_validator",
            "action": "stripped_ungrounded" if dropped_citations > 0 else "passed",
            "kept_citations": len(answer.citations),
            "dropped_citations": dropped_citations,
        }
    )
    guardrail_actions.append({"rail": "system_prompt_data_isolation", "action": "enforced"})

    ctx_guard_elapsed = time.perf_counter() - ctx_guard_start
    total_guard_elapsed = q_guard_elapsed + ctx_guard_elapsed
    GUARDRAIL_LATENCY.observe(total_guard_elapsed)

    # 8. Semantic Cache Store (if answer satisfies cacheability criteria)
    if question_vector and should_cache(answer):
        cache_store(question_vector, answer)

    # 9. Total Request Latency Metric, Trace Emission, and Structured Log
    total_elapsed = time.perf_counter() - start_time
    REQUEST_LATENCY.observe(total_elapsed)

    _record_query_trace(
        trace_id=trace_id,
        question=question,
        rewritten_query=rewritten_query,
        embed_model=embed_model,
        embed_model_version=embed_model_version,
        collection_name=collection_name,
        collection_version=collection_version,
        passages=passages,
        filters_applied=filters_applied,
        reranked_order=reranked_order,
        provider_name=provider.name,
        llm_model=llm_model,
        model_parameters=model_parameters,
        response_answer=answer.answer,
        answered_from_context=answer.answered_from_context,
        confidence=answer.confidence,
        citations=answer.citations,
        guardrail_actions=guardrail_actions,
        latency_per_stage={
            "cache_lookup_ms": cache_lookup_ms,
            "retrieval_ms": retrieval_ms,
            "guardrail_ms": round(total_guard_elapsed * 1000, 3),
            "generation_ms": generation_ms,
            "total_ms": round(total_elapsed * 1000, 3),
        },
        cache_hit=False,
        cache_status=cache_status,
        trace_dir=trace_dir,
        trace_stdout=trace_stdout,
    )

    log.info(
        "answered",
        extra={
            "request_id": request_id,
            "retriever": retriever.name,
            "provider": provider.name,
            "top_k": top_k,
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
                doc_id=p.doc_id,
                doc_version=p.doc_version,
            )
            for p in passages
        ],
        provider=provider.name,
    )
