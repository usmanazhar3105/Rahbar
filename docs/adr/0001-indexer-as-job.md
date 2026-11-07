# ADR 0001: Indexer as a Kubernetes Job

## Status: Accepted

## Context
In a RAG architecture, document extraction, cleaning, chunking, embedding, and vector database ingestion must occur before retrieval queries can be served. A naive implementation triggers indexing in the API container's startup hook (`@app.on_event("startup")` or `lifespan`).

Running indexing on API startup causes four catastrophic production failure modes:
1. **Race Condition across Replicas**: With 3 API replicas running against a single Chroma volume, all three pods attempt to extract and write to the same collection simultaneously. Chroma's SQLite metadata layer locks (`sqlite3.OperationalError: database is locked`), resulting in index corruption and crash loops.
2. **Every Pod Restart Triggers Re-indexing**: Any container restart (from an OOMKill, node drain, rolling update, or readiness probe blip) causes the newly spawned pod to re-read and re-embed the entire corpus, generating unnecessary Ollama load and stalling container start.
3. **Readiness Becomes a Lie**: The API cannot serve traffic until indexing finishes, but indexing 50+ pages takes minutes. Kubernetes readiness probes will fail repeatedly; if initial probe delays are misconfigured, Kubernetes marks the pod unready and kills slow-starting pods before indexing can ever complete.
4. **Cannot Re-index Without Redeploying**: When a new academic policy is published, the cluster cannot update the vector database without redeploying or terminating live API pods, creating artificial service disruption for users.

## Decision
We decouple indexing completely from the API lifecycle:
1. The **API is strictly a reader**: It connects to Chroma at startup, verifies collection existence and embedding model metadata, and serves queries. It never extracts documents or writes to vector collections.
2. The **Indexer runs as a standalone Kubernetes Job**: Document processing runs inside a dedicated batch container with `restartPolicy: OnFailure` and explicit `backoffLimit: 2`.
3. Periodic and manual indexing is orchestrated through a Kubernetes `CronJob` and ad-hoc `Job` triggers.

## Consequences
- **Positive**:
  - Zero write contention on Chroma storage; exactly one writer runs at any given time.
  - API pods boot in $<1$ second with lightweight readiness checks.
  - Decoupled resource limits: The indexer is allocated burst memory (1Gi) for PDF/OCR rendering, while API replicas stay lean (256Mi).
  - Corpus updates occur asynchronously in the background without dropping live HTTP connections.
- **Negative**:
  - The API must handle queries arriving before the initial index job has populated the collection (handled via `/readyz` returning HTTP 503 until `collection.count() > 0`).

## The command to re-index without downtime:
```bash
kubectl create job --from=cronjob/reindex manual-reindex-1 -n rahbar
```
