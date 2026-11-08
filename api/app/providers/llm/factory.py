"""Factory for instantiating LLM providers."""

from __future__ import annotations

import os

from ...config import Settings
from .base import LLMProvider
from .hosted import GroqProvider
from .stub import StubProvider


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Return configured LLM provider.

    Reads LLM_PROVIDER from settings or environment.
    Supported: groq (default), hosted, stub.
    """
    raw_name = (settings.llm_provider if settings else None) or os.getenv("LLM_PROVIDER", "groq")
    provider_name = (raw_name or "groq").lower()

    if provider_name in ("groq", "hosted"):
        return GroqProvider(settings=settings)
    elif provider_name == "stub":
        return StubProvider()
    elif provider_name in ("ollama-host", "ollama-incluster"):
        from .ollama import OllamaProvider

        return OllamaProvider(settings or Settings())
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider_name}. Supported: groq, stub, hosted"
        )

