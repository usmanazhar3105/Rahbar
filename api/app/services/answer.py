from __future__ import annotations

import uuid
from typing import Any

from ..schemas import AskResponse, RetrievedRef


def answer_question(
    question: str,
    retriever: Any,
    provider: Any,
    settings: Any,
    request_id: str | None = None,
) -> AskResponse:
    if not request_id:
        request_id = str(uuid.uuid4())
    top_k = getattr(settings, "top_k", 5)
    relevance_floor = getattr(settings, "relevance_floor", 0.3)

    passages = retriever.retrieve(question, top_k=top_k, relevance_floor=relevance_floor)
    retrieved_refs = [
        RetrievedRef(
            chunk_id=p.chunk_id,
            source=p.source,
            section=p.section,
            page=p.page,
            score=p.score,
        )
        for p in passages
    ]

    if not passages:
        return AskResponse(
            request_id=request_id,
            answer="The indexed corpus does not contain information to answer this question.",
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=[],
            provider=getattr(provider, "name", "stub"),
        )

    ans = provider.generate(question, passages)
    return AskResponse(
        request_id=request_id,
        answer=ans.answer,
        citations=ans.citations,
        answered_from_context=ans.answered_from_context,
        confidence=ans.confidence,
        retrieved=retrieved_refs,
        provider=getattr(provider, "name", "stub"),
    )
