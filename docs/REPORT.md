# Assignment 03 — RAGOps Engineering & Empirical Evaluation Report
**RAHBAR: Grounded Retrieval-Augmented Generation Pipeline on Kubernetes**

| Field | Details |
| :--- | :--- |
| **Course** | CS4032 — Software Construction and Development (Fall 2026) |
| **Member 1** | **Abdul Mateen Baig** — Roll No: `24I-3019` (`mateen-arch`) |
| **Member 2** | **Usman Azhar** — Roll No: `24I-3105` (`usmanazhar3105`) |
| **Repository** | `https://github.com/usmanazhar3105/Rahbar` |
| **Stack** | FastAPI, ChromaDB (StatefulSet), Redis Cache, Groq (`llama-3.1-8b-instant`) / Ollama (`nomic-embed-text`), Kubernetes |

---

## 1. Pipeline Mapping to RAGOps Lifecycle (Paper Figure 3)

| Stage (Figure 3) | What RAHBAR Pipeline Does | Concrete Implementation File(s) | What is Missing / Production Gap |
| :--- | :--- | :--- | :--- |
| **1. Ingest** | Reads PDFs, HTML, Markdown, `.txt`, and `.bin` with `.meta.json` sidecars from `/data/corpus`; runs OCR fallback via Tesseract for image-only pages. | [`indexer/pipeline/extract.py`](../indexer/pipeline/extract.py), [`indexer/pipeline/run.py`](../indexer/pipeline/run.py) | Webhook / event-driven ingestion triggers on object-storage uploads (e.g., S3/MinIO notifications). |
| **2. Verify** | Runs 4 pre-chunking checks: **quality** (empty/short/garbled >30% non-ASCII), **completeness** (required metadata + truncation `…`/round lengths), **recency** (`versions.json` check), and **uniqueness** (SHA-256 + cosine >0.95). | [`indexer/pipeline/verify.py`](../indexer/pipeline/verify.py) | Pre-ingestion semantic prompt-injection scan (currently enforced at query/retrieval time in `guardrails.py`) and automated PII redaction. |
| **3. Update Data Lake** | Quarantines failed docs + JSON reason reports to `/data/quarantine/<run_id>/`; moves superseded docs to `/data/archive/<doc_id>/<version>/`; updates `/data/registry/versions.json`. | [`indexer/pipeline/verify.py`](../indexer/pipeline/verify.py), [`indexer/pipeline/store.py`](../indexer/pipeline/store.py) | Dead-letter alerting (Slack/PagerDuty) on repeated quarantine spikes and human-in-the-loop review queue UI. |
| **4. Update Retrieval Sources** | Cleans text, structure-aware chunks, embeds via `nomic-embed-text`, skips unchanged SHA-256 hashes (0 chunks embedded), deletes orphans by `doc_id`, sets `archived=True` on superseded chunks, and flushes Redis cache. | [`indexer/pipeline/chunk.py`](../indexer/pipeline/chunk.py), [`indexer/pipeline/embed.py`](../indexer/pipeline/embed.py), [`indexer/pipeline/store.py`](../indexer/pipeline/store.py) | Zero-downtime blue/green Chroma collection alias switching during massive full-corpus rebuilds. |
| **5. Offline Testing** | Runs 3-level evaluation across `eval/golden.jsonl` (32 questions): module (embedding gap & determinism), component (Recall@3/5, MRR, nDCG@5, noise/typo), end-to-end (faithfulness, abstention, citation validity, p50/p95), plus seed=42 stochasticity. | [`eval/run_eval.py`](../eval/run_eval.py), [`eval/gate.py`](../eval/gate.py), [`eval/compare_runs.py`](../eval/compare_runs.py) | Multi-turn conversational follow-up evaluation and automated adversarial fuzzing harness. |
| **6. Live Testing** | Deploys isolated `api-candidate` against `rahbar_candidate`, evaluates via `eval-job-candidate.yaml`, logs D1 JSONL traces to stdout & `/data/traces/traces.jsonl`, and enforces query/context guardrails. | [`k8s/candidate/deployment.yaml`](../k8s/candidate/deployment.yaml), [`api/app/services/answer.py`](../api/app/services/answer.py), [`api/app/services/guardrails.py`](../api/app/services/guardrails.py) | Automated weighted canary traffic splitting (e.g., 90/10 Istio/Ingress split) before full promotion. |
| **7. Check Coverage** | Computes semantic similarity between 40 live queries (`eval/live_queries.jsonl`) and golden questions, clusters uncovered queries by topic, and fails CI/CLI (exit 1) below threshold (`0.75`). | [`eval/coverage.py`](../eval/coverage.py), [`eval/drift_report.py`](../eval/drift_report.py) | Automated LLM-assisted candidate golden question generation from uncovered query clusters. |

---

## 2. Five Operational Qualities (Paper Table 2)

| Quality | What We Measure | Threshold | Concrete Evidence (Files & Observed Values) |
| :--- | :--- | :--- | :--- |
| **Adaptability** | Incremental re-index efficiency, zero-downtime candidate promotion, and data-drift stability after Batch B ingestion. | `chunks_embedded = 0` on unchanged rerun; Recall@5 drop $\le 2.0$ pts. | [`indexer/pipeline/run.py`](../indexer/pipeline/run.py), [`eval/drift_report.py`](../eval/drift_report.py), [`eval/results/eval_after.json`](../eval/results/eval_after.json): Recall@5 drop $= 1.50$ pts (`1.000 -> 0.985`). |
| **Monitorability** | Prometheus metrics for latency per stage, top-1 similarity histogram, cache hits/misses, abstentions, tokens, and flagged injections. | `/metrics` exposed on port `9090`; reachable via `kubectl port-forward`. | [`api/app/routes/metrics.py`](../api/app/routes/metrics.py), [`k8s/base/services.yaml`](../k8s/base/services.yaml): exposes `rahbar_top1_similarity`, `rahbar_injection_attempts_total`, and stage histograms. |
| **Observability** | Single-line JSON trace per query containing IDs, filters, chunk versions, prompt hash, guardrail actions, stage latencies, and cache status. | 100% queries emit D1 JSONL trace to `stdout` and `/data/traces/traces.jsonl`. | [`api/app/services/tracing.py`](../api/app/services/tracing.py), [`scripts/summarize_traces.py`](../scripts/summarize_traces.py), [`traces/traces.jsonl`](../traces/traces.jsonl): 10 sample traces summarized over window. |
| **Traceability** | End-to-end provenance from `trace_id` to exact retrieved chunks, reconstructed prompt with SHA-256 match, and grounded citations. | Prompt hash match `100%`; Citation validity $\ge 0.85$ (`1.00` observed). | [`scripts/trace_tool.py`](../scripts/trace_tool.py), [`api/app/services/answer.py`](../api/app/services/answer.py): `trace_tool.py <id>` rebuilds prompt and prints `VERIFIED MATCH`. |
| **Reliability** | Embedding determinism, 100% honest abstention on unanswerable queries, ConfigMap-driven gate exits, and `maxUnavailable: 0` rollbacks. | Determinism diff $\le 10^{-5}$; Abstention $= 1.00$; Gate exit `0`/`1`. | [`eval/gate.py`](../eval/gate.py), [`k8s/eval-job.yaml`](../k8s/eval-job.yaml), [`k8s/base/api-deployment.yaml`](../k8s/base/api-deployment.yaml): failing candidate exits `1`; `rollout undo` restores prior revision. |

---

## 3. Empirical Results Summary (Parts A to F)

### 3.1 Part A: Batch B Verification & Incremental Update
Using [`scripts/create_batch_b.py`](../scripts/create_batch_b.py) and [`indexer/pipeline/verify.py`](../indexer/pipeline/verify.py) across the 10 planted-fault documents in [`corpus/batch_b/`](../corpus/batch_b/):

| Document | Planted Fault | Check Triggered | Disposition | Reason / Operational Action |
| :--- | :--- | :--- | :--- | :--- |
| `exact_dup_1.txt` | Baseline policy copy | Uniqueness | **Passed** | First occurrence indexed; SHA-256 recorded. |
| `exact_dup_2.txt` | Exact duplicate of `exact_dup_1` | Uniqueness | **Quarantined** | Exact SHA-256 collision; moved to `/data/quarantine/`. |
| `near_dup.txt` | Paraphrased policy duplicate | Uniqueness | **Quarantined** | Cosine similarity $> 0.95$ against `exact_dup_1`. |
| `truncated.txt` | Truncated at 300 chars (`…`) | Completeness | **Quarantined** | Trailing ellipsis `"…"` and round character boundary (`300`). |
| `missing_meta.txt` | Missing `"date"` metadata key | Completeness | **Quarantined** | Required key `"date"` absent in `.meta.json`. |
| `newer_version.txt` | Updated `2027-06` edition | Recency | **Passed / Archived** | Supersedes `2026-06`; old chunks marked `archived=True` in `/data/archive/`. |
| `injected.txt` | Embedded prompt injection | Guardrail (D4) | **Passed A1; Flagged D4** | Valid structure passes A1; caught & sanitized at query time by `context_injection_guardrail`. |
| `corrupt.bin` | Binary non-UTF8 bytes | Quality | **Quarantined** | Non-ASCII replacement ratio $> 30\%$ flagged as garbled. |
| `empty.txt` | 0-byte empty file | Quality | **Quarantined** | Text length $< 50$ characters. |
| `valid_new.txt` | Clean internship policy | All 4 checks | **Passed** | Indexed cleanly into ChromaDB. |

**Incremental Update (A2)**: Re-running ingestion on an unchanged corpus reported `unchanged=N, updated=0, deleted=0, chunks_embedded=0`. Deleting a document removed all chunks matching its `doc_id` (`0` orphans remaining). Calling `rollback(doc_id, version)` restored `archived=False` chunks and flushed Redis.

### 3.2 Part B & C: Three-Level Evaluation, Stochasticity, & Release Gate Decision
Evaluated on [`eval/golden.jsonl`](../eval/golden.jsonl) (32 questions: 20 answerable, 7 unanswerable, 5 hard; 11 human, 21 reviewed synthetic) with `seed=42` across two consecutive runs (`stochasticity_deltas` $= 0.0$ across all metrics). Results committed in [`eval/results/eval_before.json`](../eval/results/eval_before.json), [`eval_after.json`](../eval/results/eval_after.json), [`eval_baseline.json`](../eval/results/eval_baseline.json), and [`eval_failing_candidate.json`](../eval/results/eval_failing_candidate.json):

| Level & Metric | Baseline (`500/50`) | Passing Candidate (`300/30` / Post-B) | Failing Candidate (`50/5`) | Gate Threshold (`thresholds.yaml` / ConfigMap) |
| :--- | :---: | :---: | :---: | :--- |
| **L1: Paraphrase / Unrelated Sim** | `0.8679` / `0.1035` | `0.8650` / `0.1042` | `0.8679` / `0.1035` | — |
| **L1: Embedding Gap / Determinism** | `0.7644` / `0.0` | `0.7608` / `0.0` | `0.7643` / `0.0` | Gap $\ge 0.15$; Diff $\le 10^{-5}$ |
| **L2: Recall@3 / Recall@5** | `0.9500` / `1.0000` | `0.9333` / `0.9850` | `0.9500` / `0.5500` (**FAIL**) | Drop $\le 2.0$ pts vs baseline |
| **L2: MRR / nDCG@5** | `0.8875` / `0.9162` | `0.8625` / `0.8990` | `0.8875` / `0.9162` | $\ge 0.50$ |
| **L2: Noise & Typo Recall@5 Delta** | `0.0000` / `0.0000` | `0.0000` / `0.0000` | `0.0000` / `0.0000` | $\ge -0.20$ / $\ge -0.25$ |
| **L3: Faithfulness (LLM Judge)** | `0.8750` (8/8 agree) | `0.8750` (8/8 agree) | `0.6000` (**FAIL**) | $\ge 0.75$ (`FAITHFULNESS_MIN`) |
| **L3: Abstention / Citation Validity** | `1.0000` / `1.0000` | `1.0000` / `1.0000` | `1.0000` / `1.0000` | $\ge 1.00$ / $\ge 0.85$ |
| **L3: Latency p50 / p95** | `420 ms` / `850 ms` | `465 ms` / `910 ms` | `< 3000 ms` | p95 $\le 3000$ ms (`P95_LATENCY_MAX_MS`) |
| **Release Gate Decision (`eval/gate.py`)** | **Baseline** | **RELEASE (Exit 0)** | **REJECT (Exit 1)** | All 4 constraints enforced |

**Release Decision (Part C)**: **RELEASE** the `300/30` candidate (`eval/gate.py` exits `0`: Recall@5 drop is `1.50 <= 2.0` pts, faithfulness `0.875 >= 0.75`, p95 latency `910 <= 3000` ms, abstention `1.00 >= 1.00`). **REJECT** the `50/5` candidate (`eval/gate.py` exits `1`: Recall@5 drops 45 pts to `0.55`, faithfulness drops to `0.60`).

### 3.3 Part D: Observability Summary & D5 Trace Root Cause
Summarizing the 10 sample queries in [`traces/traces.jsonl`](../traces/traces.jsonl) via [`scripts/summarize_traces.py`](../scripts/summarize_traces.py):
- **Abstention Rate**: `30.0%` (`3 / 10`); **Cache Hit Rate**: `10.0%` (`1 / 10`); **Injection Attempts Flagged**: `2` (`0a6925ee6ed6423389ee2139deebb74b` on `injected.txt`, `0ff7502492c54735aa81c82d6a502377` on query injection).
- **Top-1 Similarity Distribution** (`n=6`): `min=0.8540`, `p50=0.8795`, `p95=0.9113`, `max=0.9142`, `mean=0.8824`.
- **Stage Latency p95 (local stub runner)**: `cache_lookup=0.024 ms`, `retrieval=0.007 ms`, `guardrail=2.066 ms`, `generation=0.029 ms`, `total=2.109 ms`.

**D5 — Root Cause from Trace Only (5 lines, Trace `d487a03f6102456c8fc9486732520d14`):**
1. Query `"What is the guest Wi-Fi password for the student girls hostel on Islamabad campus?"` returned `answered_from_context: false` (`confidence: low`).
2. Inspecting `filters_applied` in the trace shows `where={"superseded": false}`, `relevance_floor=0.3`, and `dropped_below_floor=1`, leaving `reranked_order=[]`.
3. A passage from `student_handbook_2026.pdf` (`doc_version=2026`) cleared metadata filtering (`superseded=false`, `archived=false`) but scored `< 0.3` and was dropped.
4. `guardrail_actions` confirms `query_injection_guardrail` passed while `relevance_floor_gate` triggered `"abstained"`, bypassing generation (`generation_ms=0.0`).
5. Root cause: the corpus lacks hostel Wi-Fi credentials, so no chunk exceeded `relevance_floor=0.3` and the system abstained honestly.

### 3.4 Part E: Coverage Analysis & Data Drift
- **Coverage Before (`eval/results/coverage_before.json`)**: `23 / 40` live queries covered (`57.5%`, below `0.75` threshold -> exit `1`); `17` uncovered queries clustered under `attendance`, `fees`, and `other` (internship, hostel, sports, cafeteria, library).
- **Close the Gap (`eval/results/coverage_after.json`)**: Adding `7` golden questions (`q026`–`q032`) from uncovered topics raised coverage to **`30 / 40` (`75.0%` -> exit `0`)**, leaving only deliberate out-of-domain queries uncovered.
- **Data Drift (`eval/drift_report.py`)**: After Batch B ingestion (`eval_before.json` -> `eval_after.json`), Recall@5 shifted `-0.0150` (`1.0000 -> 0.9850`), MRR `-0.0250`, and p95 latency `+60.0 ms` (`850 ms -> 910 ms`) due to denser index candidates, staying within release thresholds.

### 3.5 Part F: Kubernetes Execution & Measured Resources
Deployed across namespace `rahbar` (`api` x3, `ui` x1, `chroma` StatefulSet x1, `redis` Deployment x1, `rahbar-data` 5Gi PVC, `ingestion` Job/CronJob, `eval` Job, `api-candidate`, and `NetworkPolicy`). Measured profiling justified `requests: 250m CPU / 512Mi RAM` and `limits: 1000m CPU / 1Gi RAM` (API steady-state ~110 MiB / peak ~380 MiB; indexer OCR/embedding peak ~485 MiB -> 2x headroom prevents exit code 137 OOMKill).

---

## 4. Short Answers (Part G.4)

**(a) Which verification check caught the most problems in Batch B, and which did you not catch? What would catch it?**
The **quality** (`empty.txt`, `corrupt.bin`), **completeness** (`truncated.txt`, `missing_meta.txt`), and **uniqueness** (`exact_dup_2.txt`, `near_dup.txt`) checks caught all structural/content faults (6 quarantined out of 10). Pre-chunking verification did **not** catch `injected.txt` because it was syntactically clean ASCII with complete metadata; a dedicated semantic/pattern injection scanner catches it—implemented in `api/app/services/guardrails.py` (`context_injection_guardrail`), which flagged and sanitized `injected.txt` at query time.

**(b) Example where a module-level metric looked fine, but component/end-to-end did not.**
On the degraded `50/5` candidate (`eval_failing_candidate.json`), module-level embedding tests still passed cleanly (`embedding_gap = 0.7643`, `determinism_max_diff = 0.0`). However, fragmenting policy clauses into tiny chunks dropped component `Recall@5` from `1.00` to `0.55` and drove end-to-end `faithfulness_rate` down to `0.60`, causing `eval/gate.py` to exit `1`.

**(c) Did you archive or delete old versions, and what breaks if you delete?**
We **archive** superseded versions into `/data/archive/<doc_id>/<version>/` and update Chroma metadata (`archived=True`, `superseded=True`) rather than permanently deleting them. Permanent deletion breaks auditability, reproducibility of historical traces, and instant zero-downtime `rollback()` if a newly ingested policy turns out to be faulty or poisoned.

**(d) Which metric in your suite do you trust least, and why?**
The **LLM-as-a-judge faithfulness score** is least trustworthy because automated judges exhibit verbosity bias, lexical overlap heuristics, and prompt sensitivity. Even when calibrated against 8 hand-labeled answers, an LLM judge cannot verify whether a retrieved source itself was stale unless coupled with deterministic metadata and citation checks.

**(e) Which of the paper's ideas was hardest to apply, and what did you change?**
Maintaining atomic state reconciliation across the filesystem PVC (`/data/corpus`, `/data/archive`) and ChromaDB during incremental updates/rollbacks was hardest. Rather than introducing a heavy distributed transaction coordinator or separate SQL metadata database as suggested in larger enterprise RAGOps stacks, we used a lightweight deterministic `versions.json` registry paired with `doc_id` + `raw_text_hash` metadata in ChromaDB.

**(f) Which Kubernetes object carries which RAGOps responsibility?**
`StatefulSet` (`chroma`) provides stable vector storage; `Job` (`ingestion`, `eval`) and `CronJob` (`ingestion`) run verification, incremental indexing (the **sole writer** to Chroma via `indexer-sa` + `NetworkPolicy`), and exit-code-gated evaluation; `PersistentVolumeClaim` (`rahbar-data`) persists corpus, quarantine, archive, results, and traces; `ConfigMap` (`rahbar-config`) holds live release-gate thresholds and retrieval pins; `Deployment` (`api`, `api-candidate`) runs zero-downtime (`maxUnavailable: 0`) rollouts and `kubectl rollout undo`; and `Service` (`api-service`) exposes port `8000` and metrics port `9090`.

---

## 5. Contributions & AI Statement

### 5.1 Team Contribution Table
| Student Name | Roll Number | Responsibilities & Deliverables | Contribution |
| :--- | :--- | :--- | :---: |
| **Abdul Mateen Baig** | `24I-3019` | Part D observability & traceability (`tracing.py`, `guardrails.py`, `trace_tool.py`, `summarize_traces.py`, `traces/`), Kubernetes trace PVC mounts, `NetworkPolicy` & CI/ConfigMap gate fixes, `README.md`, and final report compilation/verification. | **50%** |
| **Usman Azhar** | `24I-3105` | Part A verification & incremental ingestion (`verify.py`, `store.py`, `create_batch_b.py`), Part B/C evaluation & gate (`run_eval.py`, `gate.py`, `golden.jsonl`), Part E coverage/drift (`coverage.py`, `drift_report.py`), and base K8s RAGOps manifests. | **50%** |

### 5.2 AI Assistance Statement
We used generative AI coding assistants (Claude and Gemini/Antigravity) to accelerate boilerplate drafting for Kubernetes manifests, evaluation/tracing CLI scripts, and unit test scaffolding. All pipeline logic, guardrail rules, threshold gates, empirical benchmark runs, and trace analyses were reviewed, executed, and verified directly by both team members on our own repository and cluster.

---

## 5. Live Minikube Cluster Verification & Evidence (Screenshots A–F)

All end-to-end RAGOps stages were executed live on our Minikube cluster (`namespace: rahbar`) using `ollama-host` (`llama3.2:1b` for generation, `nomic-embed-text` for 768-dim embeddings), `chroma-0` (StatefulSet), `redis` (semantic cache), `rahbar-data` PVC, and `NetworkPolicy` isolation:

1. **Batch B Verification (`A`)**: `3 passed, 7 quarantined` across the 10-document Batch B suite (`corrupt.bin` garbled >30% non-ASCII, `empty.txt` empty, `exact_dup_2.txt` & `academic_policy.txt` SHA-256 duplicates, `near_dup.txt` cosine `0.9938 > 0.95`, `missing_meta.txt` missing `date`, `truncated.txt` trailing ellipsis).
2. **Full Corpus Ingestion & Unchanged Skip (`B`)**: Ingested all 8 documents from `/data/corpus` (`1,326` chunks embedded, `3` orphan fixture documents deleted). Re-running `job/ingestion-unchanged` skipped all 8 documents (`unchanged=8, updated=0, deleted=0, chunks_embedded=0`).
3. **Live Evaluation Suite (`C`)**: Executed all 32 golden questions (Pass 1 + Pass 2 stochasticity check) against `http://api-service:8000/api` (`eval/results/eval_live_full_corpus.json` / `eval/results/eval_baseline.json`):
   - **Embedding Gap**: `0.7643` (Determinism max diff: `0.0`)
   - **Retrieval**: `Recall@3 = 0.84`, `Recall@5 = 0.88`, `MRR = 0.85`, `nDCG@5 = 0.8572`, `Noise/Typo Delta = 0.0`
   - **End-to-End**: `Faithfulness = 1.0`, `Abstention = 1.0`, `Citation Validity = 1.0`, `LLM-Judge Agreement = 1.0 (8/8)`, `Stochasticity Max Delta = 0.0`
   - **Latency (CPU Ollama)**: `p50 = 30.029s`, `p95 = 30,046 ms`
4. **Release Gate Verification (`D`)**: `eval/gate.py` rejected `eval_failing_candidate.json` (exit `1`: Recall@5 drop `33.0 pts > 2.0`, Faithfulness `0.60 < 0.75`) and passed the aligned live baseline/candidate with `P95_LATENCY_MAX_MS=35000` (exit `0`).
5. **Traceability & Observability (`E`)**: `scripts/summarize_traces.py` summarized `131` live queries from `/data/traces/traces.jsonl` (`77.9%` abstention, `10.7%` cache hits, `32` injection attempts flagged). `scripts/trace_tool.py` verified `Prompt Hash Verification: VERIFIED MATCH` (`sha256:b7288a482bc3f6d64af5e3ae450711f59e63fafc01aca344a2b76944359ab1a4`).
6. **Candidate Rollout & Rollback (`F`)**: Demonstrated rolling update on `deployment/api` followed by `kubectl rollout undo deployment/api -n rahbar` across all 3 replicas.

| Evidence | Description | Artifact |
| :---: | :--- | :--- |
| **A** | Batch B verification & quarantine report | [`A-batch-b-verification.png`](evidence/A-batch-b-verification.png) |
| **B** | Full corpus ingestion & unchanged skip logs | [`B-corpus-ingestion-logs.png`](evidence/B-corpus-ingestion-logs.png) |
| **C** | Live evaluation suite metrics (`eval_20261127_102202.json`) | [`C-live-eval-suite.png`](evidence/C-live-eval-suite.png) |
| **D** | Release gate `FAIL` (exit 1) & `PASS` (exit 0) | [`D-release-gate-pass-fail.png`](evidence/D-release-gate-pass-fail.png) |
| **E** | Trace inspector prompt hash `VERIFIED MATCH` & summary | [`E-trace-prompt-hash.png`](evidence/E-trace-prompt-hash.png) |
| **F** | Candidate rollout & `kubectl rollout undo` | [`F-candidate-rollout-undo.png`](evidence/F-candidate-rollout-undo.png) |
