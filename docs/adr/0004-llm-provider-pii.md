# ADR 0004: Hosted LLM Provider Selection and PII Isolation

## Status: Accepted

## Context
Deploying an on-cluster 8B LLM requires high GPU/VRAM allocations not available on standard student Minikube environments. We evaluated hosted LLM providers (Groq) with strict data privacy constraints.

## Decision
We select **Groq hosted API (`llama-3.1-8b-instant`)** with privacy guardrails:
1. Grounded answering only: system prompt strictly commands answering from numbered context passages.
2. Context isolation: retrieved context is demarcated with XML tags (`<context>`) to neutralize prompt injection.
3. Zero PII transmission: user credentials, student IDs, and non-contextual metadata are never forwarded in the request payload.
4. Secret injection: `GROQ_API_KEY` is injected strictly via Kubernetes Secret into container environment variables.

## Consequences
- **Positive**: High inference speed ($<1.2\text{s}$ latency), zero local GPU requirement, deterministic low temperature ($0.1$).
- **Negative**: Requires outbound internet egress from the API pod to `api.groq.com`.

