# ADR 0002: Semantic Cache Store Using Redis

## Status: Accepted

## Context
Serving LLM completions for repetitive or semantically equivalent questions introduces unnecessary inference cost and high latency ($\approx 1.5\text{s}$ per call). A semantic cache stores `(question_vector, Answer)` pairs and returns pre-computed responses when incoming query similarity exceeds a threshold ($0.92$).

We evaluated two architectural approaches for the semantic cache backing store:
1. **A secondary Chroma collection** dedicated to cached query vectors.
2. **A dedicated Redis instance** storing cached embeddings and JSON payloads.

## Decision
We choose **Redis** as the semantic cache storage engine rather than a second Chroma collection:

1. **Native TTL (Time-To-Live) Support**:
   - Semantic cache entries must expire to prevent permanent retention of outdated guidance. Redis provides native, per-key TTL (`SETEX` / `EXPIRE` for 24 hours), automatically evicting expired entries without scheduled cron jobs.
   - Chroma collections lack native TTL or per-document expiration; implementing eviction in Chroma would require polling and manual `collection.delete()`.
2. **Atomic, Instant Invalidation (`FLUSHDB`)**:
   - Whenever the corpus is updated and re-indexed, cached answers must be immediately cleared to avoid serving stale policies.
   - In Redis, the indexer triggers a non-blocking `FLUSHDB` in milliseconds. In Chroma, clearing and reconstructing an active collection requires re-initializing collection metadata and vector indices, introducing index lock contention with live reader processes.
3. **Operational Simplicity for Small Key Count**:
   - A semantic cache typically stores hundreds to low thousands of entries. At this volume, computing cosine similarity across cached Redis vectors in Python takes $<1\text{ms}$.
   - Running a second Chroma collection doubles SQLite metadata overhead, disk volume allocations, and memory footprint in the vector store container.

## Consequences
- **Positive**:
  - Independent lifecycle: Chroma remains a pure document store; Redis remains a volatile cache.
  - Sub-millisecond lookup and native TTL expiration.
  - Zero risk of cross-polluting document embeddings with cached user questions.
- **Negative**:
  - Requires maintaining an additional lightweight Redis ClusterIP service in Kubernetes.
  - Redis cache miss falls back to Chroma retrieval; Redis downtime degrades performance to uncached latency but does not break the service.

