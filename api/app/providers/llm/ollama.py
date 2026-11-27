"""Ollama provider: ollama-host or ollama-incluster (section 3.5)."""

from __future__ import annotations

import json
import logging

import httpx

from ...config import Settings
from ...schemas import Answer, Citation, Passage
from ...services.tracing import OLLAMA_SYSTEM_PROMPT, build_ollama_user_message

log = logging.getLogger("rahbar.llm.ollama")


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings  # ollama_url, llm_model, temperature

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        """Generate an answer using local Ollama instance with structured JSON output."""
        base_url = self.settings.ollama_url.rstrip("/")
        endpoint = f"{base_url}/api/chat"

        messages = [
            {
                "role": "system",
                "content": OLLAMA_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_ollama_user_message(question, passages),
            },
        ]

        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {"temperature": self.settings.temperature},
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(endpoint, json=payload)
            if resp.status_code == 200:
                raw_content = resp.json().get("message", {}).get("content", "{}")
                parsed = self._parse_json_object(raw_content)
                return self._normalize_answer(parsed, passages)
        except Exception as exc:
            log.warning("Ollama generation failed: %s", exc)

        return Answer(
            reasoning="Ollama call failed or was unconfigured",
            answer="Service temporarily unavailable.",
            citations=[],
            answered_from_context=False,
            confidence="low",
        )

    @staticmethod
    def _parse_json_object(raw_content: str) -> dict:
        """Extract a JSON object from model output, tolerating code fences."""
        text = (raw_content or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        raise ValueError("Ollama response did not contain a valid JSON object")

    @staticmethod
    def _normalize_answer(data: dict, passages: list[Passage]) -> Answer:
        """Normalize local LLM output into a valid Answer with grounded citations."""
        has_passages = bool(passages)
        reasoning_text = str(data.get("reasoning", "")).strip()
        answer_text = str(data.get("answer") or "").strip()
        answer_lower = answer_text.lower()

        abstention_markers = (
            "does not cover",
            "doesn't cover",
            "not cover",
            "no information",
            "not mentioned",
            "not provided",
            "cannot answer",
            "cannot be provided",
            "no direct answer",
            "no evidence",
            "not contain",
            "not discussed",
            "outside the corpus",
        )
        text_says_abstain = any(m in answer_lower for m in abstention_markers) or (
            "does not cover" in reasoning_text.lower() and not answer_text
        )

        raw_flag = data.get("answered_from_context")
        if not has_passages or text_says_abstain:
            answered_from_context = False
        elif isinstance(raw_flag, bool):
            # Small local models may echo false even after answering from context.
            answered_from_context = True if (raw_flag or answer_text) else False
        elif isinstance(raw_flag, str):
            answered_from_context = raw_flag.strip().lower() in {"true", "yes", "1"} or bool(answer_text)
        else:
            answered_from_context = True

        confidence = str(data.get("confidence", "medium")).strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        citations: list[Citation] = []
        raw_citations = data.get("citations") or []
        if answered_from_context and isinstance(raw_citations, list):
            for item in raw_citations:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("source", "")).strip()
                sec = str(item.get("section", "")).strip() or "General"
                if not src:
                    continue
                raw_page = item.get("page")
                try:
                    page = int(raw_page) if raw_page is not None else None
                except (TypeError, ValueError):
                    page = None
                citations.append(Citation(source=src, section=sec, page=page))

        if answered_from_context and not citations and passages:
            top = passages[0]
            citations = [Citation(source=top.source, section=top.section, page=top.page)]

        if not answer_text:
            answer_text = (
                "The corpus does not cover this topic."
                if not answered_from_context
                else passages[0].text[:300]
            )

        return Answer(
            reasoning=reasoning_text,
            answer=answer_text,
            citations=citations if answered_from_context else [],
            answered_from_context=answered_from_context,
            confidence=confidence,
        )
