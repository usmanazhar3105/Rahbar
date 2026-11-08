import os

import pytest  # noqa: F401

os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("GROQ_API_KEY", "stub-key-for-ci")
os.environ.setdefault("TOP_K", "5")
os.environ.setdefault("RELEVANCE_FLOOR", "0.3")
os.environ.setdefault("CACHE_THRESHOLD", "0.92")
os.environ.setdefault("TEMPERATURE", "0.1")
