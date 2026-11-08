# ADR 0002: Redis Semantic Cache Architecture and Invalidation Strategy

## Status
Accepted · November 2026

## Context
Repeated or semantically equivalent questions to the RAHBAR API incur latency and token costs on the LLM provider (Groq API). A simple exact-string cache (e.g. keying on query string) fails on minor phrasing differences (e.g. *"What is minimum attendance?"* vs *"What attendance percentage is required?"*).

We need a semantic cache that:
1. Compares query intent via embedding vector similarity.
2. Serves responses with sub-10ms latency.
3. Protects against stale responses when the underlying documents change.
4. Strictly forbids caching incorrect, low-confidence, or abstained responses.

## Decision
1. **Redis Hash Store with 24-hour TTL**:
   - Cache entries are stored as Redis hashes with key `cache:{uuid4()}` containing `vector` (JSON serialized float array) and `answer` (JSON serialized `Answer` schema).
   - Each entry has a 24-hour TTL (`EXPIRE key 86400`).
2. **Similarity Threshold of 0.92**:
   - Evaluated 0.85 (caused false positives between fee queries and scholarship amounts) and 0.90 (caused false positives between course drop and semester freeze deadlines).
   - 0.92 delivered zero false hits while recognizing equivalent student queries.
3. **Never-Cache Invariant**:
   - `should_cache(answer)` strictly returns `False` when `answered_from_context=False` (abstentions) or when `confidence="low"`.
4. **Invalidation via Indexer-Triggered FLUSHDB**:
   - Upon completing index storage in Chroma, the indexer Job executes `FLUSHDB` against Redis.

## Options Considered
1. **In-Memory Python Dict / LRU Cache**: Rejected because in-memory state is lost on pod restart, not shared across multiple API replicas, and consumes pod memory.
2. **ChromaDB as Vector Cache**: Considered storing question embeddings in a secondary Chroma collection. However, Redis provides much lower latency (<2ms vs 15-20ms) and native TTL expiration.

## Consequences
- **Positive**: Drastically reduced API latency for common queries, eliminates redundant Groq API calls, shared across all API replicas, and guarantees zero stale responses after re-indexing.
- **Negative / Trade-offs**: Cache lookup scales linearly with number of cached items ($O(N)$ similarity checks). At large scale (10,000+ cached queries), a dedicated vector indexing plugin (e.g. Redis RediSearch HNSW) would replace brute-force vector comparison.

