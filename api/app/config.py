from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    retriever: str = Field(default="stub")
    llm_provider: str = Field(default="stub")
    llm_model: str = Field(default="llama-3.1-8b-instant")
    hosted_base_url: str = Field(default="https://api.groq.com/openai/v1")
    embed_model: str = Field(default="nomic-embed-text")
    chroma_host: str = Field(default="chroma")
    chroma_port: int = Field(default=8000)
    collection: str = Field(default="rahbar")
    top_k: int = Field(default=5)
    relevance_floor: float = Field(default=0.3)
    redis_host: str = Field(default="redis")
    redis_port: int = Field(default=6379)
    cache_threshold: float = Field(default=0.92)
    temperature: float = Field(default=0.1)
    ollama_host: str = Field(default="")

    def __getattr__(self, item: str) -> Any:
        return ""


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
