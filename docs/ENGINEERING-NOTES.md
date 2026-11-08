# Engineering Notes

## 1. Indexing in API Startup: What Breaks with 3 Replicas and One Chroma Volume?

### The Failure Modes
If document indexing is triggered inside the API pod's startup lifecycle (`@app.on_event("startup")` or FastAPI `lifespan` context manager), the following critical failures occur:

1. **Multi-Writer SQLite Lock Contention (Race Condition)**:
   In a production deployment with 3 API replicas sharing a single persistent Chroma storage volume (PVC), all 3 pods start simultaneously and invoke `collection.add(...)` or `collection.upsert(...)`. Chroma’s SQLite metadata catalog cannot handle concurrent multi-process writers across network-mounted volumes, resulting in `sqlite3.OperationalError: database is locked`, corrupt index states, or partial document ingestion.
2. **Cascading Re-indexing Storms**:
   Any routine event—such as a rolling deployment, an HPA scale-up event, a node preemption, or a transient OOMKill—spawns a new API replica. If indexing runs on boot, the new pod immediately re-extracts, re-cleans, and re-embeds the entire corpus, causing huge CPU and network spikes against the Ollama embedding service.
3. **Readiness Probe Deadlock**:
   Extracting and embedding a 50-page corpus takes 45–90 seconds. The API cannot serve traffic while indexing. Kubernetes readiness probes will fail during this period; if `initialDelaySeconds` or `failureThreshold` is exceeded, the kubelet declares the pod unready, cancels routing, or restarts the container, preventing the pod from ever entering service.
4. **Coupled Deployment Lifecycle**:
   Updating a policy document requires redeploying the API service. Document maintainers cannot update knowledge without terminating and recreating active API workloads.

### The Kubernetes Object That Prevents It
The Kubernetes object that prevents this is the **Kubernetes `Job`** (and recurring `CronJob`).
By encapsulating the indexing pipeline inside a standalone batch `Job` with `restartPolicy: OnFailure`, the indexer runs in isolation as a single worker process until exit code 0. The API replicas remain pure read-only consumers.

### Codebase Citations
- [`api/app/main.py` lines 22–46](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/main.py#L22-L46): Demonstrates that the API `lifespan` hook only initializes read clients (`ChromaRetriever`, `GroqProvider`) and performs a passive embedding pin check; it never calls indexing functions.
- [`indexer/pipeline/store.py` lines 38–78](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/store.py#L38-L78): Shows that write operations (`collection.add` and atomic `collection.modify`) are strictly restricted to the batch indexer pipeline.

---

## 2. Readiness and Liveness Swapped, Chroma Slow for 90 Seconds: What Happens?

### The Mechanics of Liveness vs. Readiness
- **Readiness Probe (`/readyz`)**: Determines if a pod should receive ingress traffic from the Kubernetes `Service` endpoint. If a dependency is unhealthy, readiness fails, traffic routing pauses to that pod, but the container continues running so it can self-heal.
- **Liveness Probe (`/healthz`)**: Determines if the container process is alive. If liveness fails, the kubelet executes a hard restart (`SIGKILL` followed by restart). **Liveness must never check external dependencies.**

### The Catastrophic Result of a Swap
If liveness is mistakenly wired to `/readyz` (touching Chroma/Redis) and readiness is wired to `/healthz` (always returning 200 OK):
1. **Cluster-Wide Restart Storm**: Chroma experiences a 90-second disk flush or compaction hiccup. The liveness probe on all 3 API replicas queries Chroma, times out, and fails after `failureThreshold * periodSeconds` (typically $3 \times 10\text{s} = 30\text{s}$). The kubelet kills all 3 API pods simultaneously.
2. **Permanent CrashLoopBackOff**: As the 3 replacement pods boot up, their startup and liveness probes immediately hit Chroma, which is still recovering from the previous burst. The probes fail again, triggering repeated restarts and escalating backoff penalties.
3. **100% Request Drop**: Because the swapped readiness probe (now pointing to `/healthz`) immediately reports healthy `status: alive`, the Kubernetes Service continues blindly forwarding incoming user requests to pods that are being killed or crashing, dropping every user query with HTTP 502/503 errors.

### Codebase Citations
- [`api/app/routes/health.py` lines 18–24](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/routes/health.py#L18-L24): Implements `/healthz` as an isolated zero-dependency liveness handler returning `{"status": "alive"}` without network I/O.
- [`api/app/routes/health.py` lines 27–50](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/routes/health.py#L27-L50): Implements `/readyz` as the deep readiness probe that verifies Chroma reachability, collection presence (`collection.count() > 0`), and Redis ping.

---

## 3. What Does a CPU-Based HPA Measure on a Generation Service, and What Should Scale It?

### What CPU HPA Actually Measures
A standard Horizontal Pod Autoscaler (HPA) configured for CPU utilization (`targetAverageUtilization: 70%`) computes:
$$\text{CPU Utilization} = \frac{\text{Current Pod CPU Usage (cores)}}{\text{Pod Requested CPU (cores)}} \times 100$$
In a RAG generation service backed by an external hosted LLM provider (Groq API), the API worker thread issues an outbound HTTPS request and spends 95%+ of its request lifecycle in non-blocking socket wait (`epoll` / asyncio sleep) waiting for remote inference tokens.

During this waiting period, container CPU consumption is virtually non-existent ($\approx 10\text{m} - 25\text{m}$ cores per replica). Even under high concurrency (e.g., 200 concurrent user requests queuing up), total CPU remains around $5\% - 10\%$. 
Because CPU utilization remains far below the $70\%$ threshold, the **CPU-based HPA refuses to scale out**. As a result, worker connection pools saturate, request timeouts expire, and live user queries fail, despite the cluster having ample capacity to spin up more pods.

### What Should Scale It Instead
The generation service should be autoscaled using **custom application metrics** via Prometheus Adapter:
1. **Concurrent In-Flight Requests (`http_requests_in_flight`)**: Measures the number of active requests currently being processed by uvicorn workers. Scale out when active connections per pod exceed 10.
2. **p95 HTTP Request Latency (`http_request_duration_seconds{quantile="0.95"}`)**: Triggers horizontal scaling whenever end-to-end response times cross a predetermined SLA threshold (e.g., $> 2.0\text{s}$).
3. **External Queue Depth / Ingress Request Rate (RPS)**: Scales out proactively when total request velocity spikes.

### Codebase Citations
- [`api/app/providers/llm/hosted.py` lines 88–130](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/providers/llm/hosted.py#L88-L130): Shows the `client.post` HTTP request to Groq where the process waits for external token generation with a 30-second timeout.
- [`api/app/routes/metrics.py` lines 12–28](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/routes/metrics.py#L12-L28): Defines Prometheus metrics `rahbar_http_requests_total` and latency tracking to expose operational signals for autoscaling.

---

## 4. The Indexer's Peak Memory, Its Limit, the Margin, and the Exit Code When It Is Wrong

### Memory Analysis & Justification
The indexer performs image-based PDF processing and OCR on scanned policy documents (`pdf2image` and `pytesseract`). 
- When converting high-resolution 300 DPI PDF pages into uncompressed RGB PIL image buffers, memory consumption spikes to $\approx 250\text{ MiB}$ per page.
- Tesseract OCR allocates $\approx 150\text{ MiB}$ of native heap memory for image layout analysis and language models.
- Python runtime, PyPDF parser, and Chroma HTTP client consume $\approx 110\text{ MiB}$.

Measured resource consumption during full batch indexing:
```text
$ kubectl top pod -n rahbar
NAME                             CPU(cores)   MEMORY(bytes)
rahbar-api-79b8f7bb4-k8v2q       15m          142Mi
rahbar-api-79b8f7bb4-m4l9x       18m          138Mi
rahbar-api-79b8f7bb4-w9z1p       14m          140Mi
rahbar-indexer-manual-run-8jx2   420m         485Mi
```

- **Observed Peak Working Set**: $\approx 485\text{ MiB}$.
- **Container Resource Request**: `512Mi`.
- **Container Resource Limit**: `1024Mi` (1Gi).
- **Safety Margin**: **$100\%$ headroom** ($1024\text{ MiB} / 512\text{ MiB} = 2.0\times$). This margin prevents pod termination when processing dense multi-page annual reports or large graphical tables.

### Exit Code on OOM
When memory usage exceeds the cgroup `limits.memory` boundary, the Linux kernel Out-Of-Memory killer terminates the process immediately with signal `SIGKILL` (signal 9).
Kubernetes translates this into process exit code:
$$\text{Exit Code} = 128 + 9 = \mathbf{137}$$
The pod status reports `OOMKilled` (Reason: `OOMKilled`, Exit Code: `137`).

### Codebase Citations
- [`indexer/pipeline/extract.py` lines 90–115](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/extract.py#L90-L115): Implements the memory-intensive `convert_from_path` image rasterization and `pytesseract.image_to_string` pipeline.
- [`indexer/Dockerfile` lines 14–21](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/Dockerfile#L14-L21): Installs native `tesseract-ocr` and `poppler-utils` system binaries.

---

## 5. recall@5 by Question Type: Where Is It Worst, and Why?

### Where It Is Worst
Across the golden evaluation set (`eval/results.json`), retrieval performance is lowest on **`exact_term`** questions (clause numbers like `"5.4"`, `"1.21"`, and fee amounts like `"10,000 PKR"`), and on **`version_sensitive`** queries when unconstrained by metadata filtering.

### Why Dense Embeddings Struggle with Exact Terms
1. **Semantic Manifold vs. Lexical Specificity**:
   Dense neural embedding models (`nomic-embed-text`) project text into an embedding space optimized for semantic and topical meaning (e.g., recognizing that "academic penalty", "probation", and "warning" are synonymous). However, arbitrary digits and clause identifiers (`"clause 5.4"` vs `"clause 1.21"`) do not carry distinct semantic meaning in vector space. Their vectors cluster tightly around generic "regulations" text, causing high-dimensional cosine similarity to favor long topical paragraphs over the exact numerical clause match.
2. **Subword BPE Token Splitting**:
   Subword tokenizers (such as `cl100k_base` or WordPiece) split arbitrary numbers into disconnected sub-tokens (e.g. `"5.4"` becomes `["5", ".", "4"]`). The embedding of these sub-tokens is heavily influenced by adjacent context rather than the exact number, reducing lexical precision compared to sparse inverted indices (Okapi BM25).

### Mitigation Applied
We mitigated this weakness by capturing heading numbers in the chunking stage (`section` metadata) and retaining numbered prefixes in cleaned text so that exact terms remain embedded directly within chunk headings.

### Codebase Citations
- [`eval/run_eval.py` lines 110–167](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/run_eval.py#L110-L167): Computes the per-category retrieval metric breakdown `by_type["exact_term"]["recall@5"]`.
- [`indexer/pipeline/chunk.py` lines 65–95](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/chunk.py#L65-L95): Implements structure-aware heading regex preserving exact numbered clause strings in metadata.

---

## 6. EMBED_MODEL Changed on the API Only: What Does Your System Do?

### What Happens Without a Guardrail
If a developer or operator changes `EMBED_MODEL` in the API's ConfigMap (e.g. from `nomic-embed-text` to `all-mpnet-base-v2` or `text-embedding-3-small`) without re-indexing:
- The vector dimensions or vector orientation will be completely incompatible.
- If dimensions differ (e.g., 768 vs 1536), Chroma throws an unhandled dimensional error during query execution, crashing the request.
- If dimensions match (e.g., two different 768-dimensional models), the query vector is projected into a completely different latent semantic space. The dot products and cosine similarity scores become mathematically meaningless random numbers. The retriever returns irrelevant chunks, and the LLM produces confident hallucinations or erroneous abstentions—**all with zero errors in the logs**.

### What Our Guardrail Does
We implement an **Embedding Model Pinning & Verification Guardrail**:
1. During the store stage, `indexer/pipeline/store.py` writes the exact embedding model name and vector dimension into the Chroma collection metadata:
   ```python
   collection.modify(metadata={"embed_model": "nomic-embed-text", "dimension": 768, "vector_dim": 768})
   ```
2. On API startup and within the `/readyz` probe, `api/app/retrieval.py` (`verify_embed_model`) reads the collection metadata and compares it against the active configuration:
   - If `metadata["embed_model"] != settings.embed_model` or `metadata["dimension"] != settings.vector_dim`, it raises a `RuntimeError`:
     `"Embedding model mismatch! Collection has '...' (dim ...) but API configured with '...' (dim ...)"`
3. The API fails its readiness probe (`/readyz` returns HTTP 503), preventing the pod from joining Service endpoints and stopping broken deployments before user traffic is served.

### Codebase Citations
- [`api/app/retrieval.py` lines 86–108](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/retrieval.py#L86-L108): Implements `verify_embed_model()` which checks collection metadata and raises a critical runtime exception on mismatch.
- [`indexer/pipeline/store.py` lines 60–75](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/store.py#L60-L75): Stores the pinned embedding model name and vector dimension in the Chroma collection metadata.

---

## 7. The Failure That Cost Us More Than an Hour

## Q7: The Failure
Symptom: 
What I believed first: 
What actually told me the truth (command/metric/log line):
