"""Facade module for API route routers."""

from __future__ import annotations

from .routes import ask, health, metrics

__all__ = ["ask", "health", "metrics"]

