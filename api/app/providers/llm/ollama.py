"""Ollama provider: ollama-host or ollama-incluster (section 3.5)."""

from __future__ import annotations

import json
import logging

import httpx

from ...config import Settings
from ...schemas import Answer, Passage

log = logging.getLogger("rahbar.llm.ollama")


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings  # ollama_url, llm_model, temperature

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        """Generate an answer using local Ollama instance with structured JSON output."""
        base_url = self.settings.ollama_url.rstrip("/")
        endpoint = f"{base_url}/api/chat"

        evidence_lines = [
            f"[{i}] {p.text} (source: {p.source}, section: {p.section})"
            for i, p in enumerate(passages, start=1)
        ]
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "No evidence retrieved."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a grounded assistant. Answer only from the evidence. "
                    "Output JSON matching the Answer schema."
                ),
            },
            {
                "role": "user",
                "content": f"Evidence:\n{evidence_text}\n\nQuestion: {question}",
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
                return Answer.model_validate(json.loads(raw_content))
        except Exception as exc:
            log.warning("Ollama generation failed: %s", exc)

        return Answer(
            reasoning="Ollama call failed or was unconfigured",
            answer="Service temporarily unavailable.",
            citations=[],
            answered_from_context=False,
            confidence="low",
        )
