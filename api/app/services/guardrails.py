"""Prompt injection and retrieval guardrails with structured trace logging (Part D4)."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..schemas import Passage

log = logging.getLogger("rahbar.guardrails")

# Patterns that indicate prompt injection attempts in queries or retrieved chunks.
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_system_prompt",
        re.compile(
            r"\b(?:output|reveal|print|show|repeat|leak|display)\s+(?:your\s+|the\s+)?(?:system\s+prompt|hidden\s+instructions?|developer\s+prompt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "override_policy",
        re.compile(
            r"\b(?:say\s+the\s+policy\s+permits\s+it|system\s*override|jailbreak|you\s+are\s+now\s+(?:DAN|unrestricted))\b",
            re.IGNORECASE,
        ),
    ),
]


def _find_matches(text: str) -> list[tuple[str, re.Pattern[str], str]]:
    """Return list of (pattern_name, compiled_regex, matched_substring) found in text."""
    hits: list[tuple[str, re.Pattern[str], str]] = []
    for name, pattern in INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            hits.append((name, pattern, m.group(0)))
    return hits


def check_query_guardrail(question: str) -> tuple[bool, list[dict[str, Any]]]:
    """Inspect user query for adversarial prompt injection directives.

    Returns:
        (is_blocked, guardrail_actions)
    """
    hits = _find_matches(question)
    if hits:
        matched_names = [h[0] for h in hits]
        action = {
            "rail": "query_injection_guardrail",
            "target": "user_query",
            "action": "blocked",
            "injection_detected": True,
            "matched_patterns": matched_names,
            "reason": "Adversarial prompt injection pattern detected in user query.",
        }
        log.warning("Query guardrail blocked injection attempt: %s", matched_names)
        return True, [action]

    return False, [
        {
            "rail": "query_injection_guardrail",
            "target": "user_query",
            "action": "passed",
            "injection_detected": False,
        }
    ]


def apply_retrieval_guardrails(
    passages: list[Passage],
) -> tuple[list[Passage], list[dict[str, Any]], int]:
    """Scan retrieved passages for embedded prompt injections and sanitize them.

    Preserves legitimate surrounding policy sentences while stripping adversarial directives
    so retrieved text is treated strictly as untrusted data.

    Returns:
        (sanitized_passages, guardrail_actions, injection_count)
    """
    sanitized: list[Passage] = []
    actions: list[dict[str, Any]] = []
    injection_count = 0

    for passage in passages:
        hits = _find_matches(passage.text)
        if not hits:
            sanitized.append(passage)
            continue

        injection_count += 1
        cleaned_text = passage.text
        matched_names: list[str] = []
        matched_snippets: list[str] = []

        for name, pattern, snippet in hits:
            matched_names.append(name)
            matched_snippets.append(snippet)
            # Strip full sentence containing the injection phrase if bounded, else replace phrase
            sentence_re = re.compile(
                rf"[^.!?\n]*{pattern.pattern}[^.!?\n]*[.!?]?",
                re.IGNORECASE,
            )
            cleaned_text = sentence_re.sub("[REDACTED_INJECTION_DIRECTIVE]", cleaned_text)

        cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()
        sanitized.append(passage.model_copy(update={"text": cleaned_text}))

        action = {
            "rail": "context_injection_guardrail",
            "target": "retrieved_chunk",
            "action": "flagged_and_sanitized",
            "injection_detected": True,
            "chunk_id": passage.chunk_id,
            "source": passage.source,
            "doc_version": passage.doc_version or "unknown",
            "matched_patterns": matched_names,
            "matched_snippet": matched_snippets[0],
            "detail": (
                "Detected adversarial instruction in retrieved chunk; sanitized directive "
                "before prompt assembly and enforced data-only delimiter."
            ),
        }
        actions.append(action)
        log.warning(
            "Retrieval guardrail flagged injection in chunk=%s source=%s patterns=%s",
            passage.chunk_id,
            passage.source,
            matched_names,
        )

    if injection_count == 0:
        actions.append(
            {
                "rail": "context_injection_guardrail",
                "target": "retrieved_chunks",
                "action": "passed",
                "injection_detected": False,
                "chunks_scanned": len(passages),
            }
        )

    return sanitized, actions, injection_count
