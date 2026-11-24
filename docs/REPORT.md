# RAGOps Engineering & Empirical Evaluation Report
## FAST-NUCES Academic Policy RAG System (RAHBAR)

---

### Executive Summary & Report Outline
- **Target System**: RAHBAR (Retrieval-Augmented Generation for Academic Policies)
- **Architecture**: ChromaDB (Vector Store), Groq Llama-3.1-8b-Instant (LLM Generator), Sentence-Transformers `all-MiniLM-L6-v2` / `nomic-embed-text` (Embeddings), FastAPI (Query Engine), Kubernetes / RAGOps Lifecycle.
- **Report Structure**:
  1. **Section 1**: Mapping RAHBAR to the RAGOps Lifecycle (Paper Figure 3)
  2. **Section 2**: Empirical Evaluation of the Five Operational Qualities (Paper Table 2)
  3. **Section 3**: Comprehensive Results Summary (Verification, Multi-Level Eval, Release Gate)
  4. **Section 4**: Deep-Dive Engineering Analysis & Short Answers
  5. **Section 5**: Team Contribution Matrix & Academic AI Declaration

---

## Section 1: Pipeline Stage Mapping

The table below maps the RAGOps operational lifecycle (as formalized in Figure 3 of the RAGOps reference architecture) to the concrete implementation stages of the RAHBAR system:

| Stage (Figure 3) | What RAHBAR Pipeline Does | Concrete Implementation File | What is Missing / Production Gap |
| :--- | :--- | :--- | :--- |
| **1. Ingest** | Reads source PDFs, DOCX, and raw text files from Persistent Volume `/data/corpus`; parses raw text, extracts metadata (`source`, `title`, `date`, `page_count`). | [`indexer/pipeline/extract.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/extract.py)<br>[`indexer/pipeline/run.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/run.py) | Dynamic web scraping hooks, real-time change detection listeners (e.g., S3/GCS bucket notifications). |
| **2. Verify** | Performs 4 pre-chunking gates on raw docs: quality (ASCII/length/garble), completeness (required metadata & truncation `…`), recency (registry version check), uniqueness (exact SHA-256 & cosine $>0.95$). | [`indexer/pipeline/verify.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/verify.py) | Prompt injection payload scanning, automated PII redaction, and semantic schema contract enforcement. |
| **3. Update Data Lake** | Quarantines corrupted files to `/data/quarantine/`; archives superseded documents to `/data/archive/<doc_id>/<version>/`; updates version registry `/data/registry/versions.json`. | [`indexer/pipeline/run.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/run.py)<br>[`indexer/pipeline/verify.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/verify.py) | Automated dead-letter queue (DLQ) alerts, automated human review ticketing integration. |
| **4. Update Retrieval Sources** | Cleans text, chunks documents (configurable size/overlap), embeds via `all-MiniLM-L6-v2`, skips unchanged hashes, deletes orphans, and upserts chunks to ChromaDB collection. | [`indexer/pipeline/chunk.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/chunk.py)<br>[`indexer/pipeline/embed.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/embed.py)<br>[`indexer/pipeline/store.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/store.py) | Blue/Green vector index switching with zero-downtime hot swapping at the storage layer. |
| **5. Offline Testing** | Evaluates embedding space (gap, determinism), component retrieval (Recall@3/5, MRR, nDCG@5, noise/typo sensitivity), and end-to-end generation (faithfulness, abstention, latency). | [`eval/run_eval.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/run_eval.py)<br>[`eval/gate.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/gate.py) | Multi-turn conversational evaluation, adversarial red-teaming fuzzing suite. |
| **6. Live Testing** | Routes simulated and candidate traffic to `api-candidate-service` (port 8001); verifies canary stability before promoting via `kubectl set image`. | [`k8s/candidate/deployment.yaml`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/k8s/candidate/deployment.yaml)<br>[`k8s/candidate/eval-job-candidate.yaml`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/k8s/candidate/eval-job-candidate.yaml) | Dynamic traffic splitting (e.g., Istio/Envoy weighted canary: 90/10 split) and live shadow routing. |
| **7. Check Coverage** | Compares live production queries against the golden evaluation set using semantic cosine embeddings; clusters uncovered queries by academic topic and gates on coverage $\ge 0.75$. | [`eval/coverage.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/coverage.py)<br>[`scripts/create_live_queries.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/scripts/create_live_queries.py) | Automated clustering with LLM-assisted synthetic golden question auto-generation from uncovered clusters. |

---

## Section 2: Five Operational Qualities (Paper Table 2)

| Operational Quality | What We Measure | Target Threshold | Concrete Evidence (File & Metric) |
| :--- | :--- | :--- | :--- |
| **Adaptability** | Ability to ingest new policy versions without regression, absorb incremental chunk changes, and configure chunking parameters dynamically. | Delta Recall@5 $\le 2.0$ pts drop;<br>Skipping unchanged chunks $= 100\%$. | [`eval/results/eval_candidate.json`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/results/eval_candidate.json)<br>[`eval/drift_report.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/drift_report.py): Recall drop $-0.015$; logged *"Skipping unchanged"* in `store.py`. |
| **Monitorability** | Real-time tracking of latency histograms, token usage, cache hit/miss ratio, and fallbacks via open telemetry standards. | Scrape latency $< 100$ ms;<br>Port-forward 9090 functional. | [`api/app/routes/metrics.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/routes/metrics.py)<br>[`k8s/base/services.yaml`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/k8s/base/services.yaml): Named port `metrics` on port 9090 exposing Prometheus metrics. |
| **Observability** | Structured request tracing linking question input to retrieved chunk IDs, similarity scores, abstention decisions, and latency. | $100\%$ requests carry `request_id`;<br>JSON lines emitted to `stdout`. | [`api/app/logjson.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/logjson.py)<br>[`api/app/services/answer.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/services/answer.py): `log.info("answered", extra={...})` readable via `kubectl logs`. |
| **Traceability** | Provenance linking every generated sentence to verified document source metadata, section title, and page number. | Citation validity $\ge 85\%$ ($1.00$ observed);<br>Ungrounded citations $= 0$. | [`api/app/services/answer.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/api/app/services/answer.py): `validate_citations()` strips hallucinations; [`eval/run_eval.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/run_eval.py) verifies citations against metadata. |
| **Reliability** | Deterministic embedding output, graceful abstention on out-of-domain queries, and automated gate enforcement on candidate releases. | Determinism diff $\le 10^{-5}$;<br>Abstention rate $= 100\%$;<br>Gate exit $= 0$. | [`eval/gate.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/gate.py)<br>[`eval/thresholds.yaml`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/thresholds.yaml): Strict assertions preventing failing candidates from deploying. |

---

## Section 3: Empirical Results Summary

### 3.1 Verification Results (Batch B Stress Testing)
The verification pipeline ([`indexer/pipeline/verify.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/indexer/pipeline/verify.py)) processed the 10 stress-test documents from Batch B with the following outcomes:

| Document | Trigger Target | Check Triggered | Disposition | Quarantine Reason / Action |
| :--- | :--- | :--- | :--- | :--- |
| `exact_dup_1.txt` | Baseline copy | Uniqueness | **Passed** | First copy indexed; SHA-256 recorded in memory. |
| `exact_dup_2.txt` | Exact duplicate | Uniqueness | **Quarantined** | Exact SHA-256 duplicate of `exact_dup_1.txt`. |
| `near_dup.txt` | Paraphrased sentences | Uniqueness | **Quarantined** | Cosine similarity $0.963 > 0.95$ threshold against existing doc. |
| `truncated.txt` | Cut off at 300 chars | Completeness | **Quarantined** | Text terminates with ellipsis `"…"`. |
| `missing_meta.txt`| Missing date metadata | Completeness | **Quarantined** | Required metadata key `"date"` missing or empty. |
| `newer_version.txt`| 2027 revision date | Recency | **Passed/Replaced** | Supersedes 2026 edition; archives old version to `/data/archive/`. |
| `injected.txt` | Prompt injection payload| Guardrail | **Passed (Leak)** | Structural checks passed; payload requires LLM semantic guardrail. |
| `corrupt.bin` | Binary file with `.txt`| Quality | **Quarantined** | Non-ASCII character ratio $> 30\%$ (flagged as garbled). |
| `empty.txt` | 0-byte document | Quality | **Quarantined** | Empty raw text body (length $< 50$ characters). |
| `scanned_image.pdf`| Image-only document | Quality | **Quarantined** | Zero extracted textual characters prior to OCR processing. |

---

### 3.2 Evaluation Metrics (All Three Levels)

```
========================================================================================================
LEVEL 1: MODULE-LEVEL EVALUATION (Embedding Space & Determinism)
--------------------------------------------------------------------------------------------------------
Metric                                    Baseline (500/50)       Candidate (300/30)      Threshold Gate
--------------------------------------------------------------------------------------------------------
Paraphrase Pair Mean Cosine Similarity     0.8679                  0.8650                  -
Unrelated Pair Mean Cosine Similarity      0.1035                  0.1042                  -
Embedding Separation Gap                   0.7644                  0.7608                  >= 0.1500 (PASS)
Chunk Embedding Latency (Avg)              0.36 ms                 0.38 ms                 < 20.0 ms (PASS)
Embedding Determinism Max Difference       0.000000                0.000000                <= 1e-5   (PASS)
========================================================================================================
LEVEL 2: COMPONENT-LEVEL EVALUATION (ChromaDB Retrieval Quality)
--------------------------------------------------------------------------------------------------------
Metric                                    Baseline (500/50)       Candidate (300/30)      Tolerance Gate
--------------------------------------------------------------------------------------------------------
Recall@3                                  0.9500 (95.0%)          0.9333 (93.3%)          -
Recall@5                                  1.0000 (100.0%)         0.9850 (98.5%)          Max drop <= 2.0 pts (PASS)
Mean Reciprocal Rank (MRR)                0.8875                  0.8625                  -
Normalized DCG (nDCG@5)                   0.9162                  0.8990                  -
Recall@5 Under Injection of 3 Noise Docs   1.0000 (Delta: 0.0)     0.9850 (Delta: 0.0)     Delta <= 0.05 (PASS)
Recall@5 Under Typo Perturbation (p=0.1)  1.0000 (Delta: 0.0)     0.9850 (Delta: 0.0)     Delta <= 0.05 (PASS)
========================================================================================================
LEVEL 3: END-TO-END EVALUATION (FastAPI + Groq Llama-3.1-8b)
--------------------------------------------------------------------------------------------------------
Metric                                    Baseline (500/50)       Candidate (300/30)      Production Gate
--------------------------------------------------------------------------------------------------------
Faithfulness Rate (LLM-Judge Grounded)    0.8750 (87.5%)          0.8750 (87.5%)          >= 0.7500 (PASS)
Abstention Correctness (Out-of-Domain)    1.0000 (100.0%)         1.0000 (100.0%)         >= 1.0000 (PASS)
Citation Validity Rate                    1.0000 (100.0%)         1.0000 (100.0%)         >= 0.8500 (PASS)
Latency p50 (Median End-to-End)           420.0 ms                465.0 ms                -
Latency p95 (95th Percentile Tail)        850.0 ms                910.0 ms                <= 3000 ms (PASS)
LLM-Judge vs. Human Agreement (8 Samples) 1.0000 (100.0%)         1.0000 (100.0%)         >= 0.8000 (PASS)
========================================================================================================
```

---

### 3.3 Release Gate Verification Decision

The release gate script ([`eval/gate.py`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/gate.py)) applied the codified thresholds in [`eval/thresholds.yaml`](file:///c:/Users/AA/Downloads/scd%20Assingment%202/eval/thresholds.yaml):

| Gate Check | Baseline | Candidate | ConfigMap Threshold Criterion | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@5 Drop** | 100.0% | 98.5% | $\le 2.0$ points drop (Observed: 1.50 pts) | **PASS** |
| **Faithfulness Minimum** | 0.88 | 0.88 | $\ge 0.75$ | **PASS** |
| **p95 Latency Ceiling** | 850 ms | 910 ms | $\le 3000$ ms | **PASS** |
| **Abstention Correctness** | 1.00 | 1.00 | $\ge 1.00$ (and $\ge$ Baseline) | **PASS** |
| **OVERALL RELEASE STATUS**| — | — | **All 4 Hard Constraints Satisfied** | **PASS (Exit 0)** |

*Contrast with Failing Candidate (`chunk_size=50`)*: Destroyed semantic continuity, dropping Recall@5 to 0.450 (55.0 pt drop), causing `eval/gate.py` to trigger immediate failure (`Exit 1`).

---

## Section 4: Engineering Analysis & Short Answers

### (a) Which verification check caught the most problems in Batch B, and which problem did you not catch?
The **completeness and uniqueness checks** jointly intercepted the highest number of defects in Batch B: completeness rejected truncated files terminating with `"…"` and documents missing mandatory timestamp fields, while uniqueness caught identical bit-level clones (SHA-256) and paraphrased plagiarisms (cosine similarity $> 0.95$). Conversely, the pre-chunking verification stage failed to catch `injected.txt`, which contained a prompt injection attack buried inside legitimate academic prose. Because the injection payload comprised syntactically valid ASCII characters and satisfied length and metadata requirements, it bypassed structural verification entirely, illustrating that adversarial prompt injections require dedicated semantic guardrails rather than static structural hygiene checks.

### (b) Give an example where a module-level metric looked fine but component/end-to-end did not.
When evaluating our deliberately degraded candidate indexing run (`chunk_size=50`, `chunk_overlap=5`), the module-level embedding tests passed cleanly with an embedding separation gap of 0.764 and zero determinism variance, proving that `all-MiniLM-L6-v2` vectorized short tokens consistently. However, at the component retrieval level, chunking sentences into 50-character fragments stripped necessary semantic context, causing ChromaDB Recall@5 to plummet from 100% to 45.0%. Consequently, the end-to-end generator received fragmented partial clauses, driving faithfulness down to 0.50 and triggering widespread hallucination; this divergence proves that isolated representation quality does not guarantee downstream retrieval or generation viability.

### (c) Archive vs. delete — what breaks if you delete?
Permanently deleting superseded document chunks (`collection.delete()`) breaks auditability, state rollback, and active user session reproducibility. If an erroneous, poisoned, or malformed policy update is ingested and the previous version has been permanently expunged, rolling back to the known-good state is impossible without reconstructing the corpus from external backups and incurring substantial vector re-indexing downtime. Archiving old versions into `/data/archive/<doc_id>/<version>/` and toggling `archived=True` in ChromaDB isolates deprecated chunks from live user queries via filter predicates (`{"archived": False}`) while enabling instant zero-downtime rollback and continuous evaluation against historical policy baselines.

### (d) Which metric do you trust least and why?
The **faithfulness metric evaluated via LLM-as-a-judge** is the least trustworthy metric in the evaluation suite. Automated LLM judges frequently suffer from position bias, sycophancy, and severe sensitivity to prompt phrasing, often scoring answers as "faithful" merely because they share lexical n-grams with retrieved passages rather than validating true logical entailment. Furthermore, if a retrieved passage itself contains contradictory or outdated information from a superseded policy, the judge rewards the model for faithfully reproducing erroneous facts, confounding factual correctness with context faithfulness.

### (e) Which RAGOps idea was hardest to apply?
Implementing **incremental state reconciliation with transactional multi-version archiving and orphan removal** was the most technically demanding RAGOps concept. Correlating physical documents on a Kubernetes Persistent Volume with embedded chunks in ChromaDB required computing dual-level SHA-256 fingerprints (source-title fingerprint for identity, raw text hash for content change), orchestrating conditional chunk upserts, and handling collections without breaking ongoing reads. Ensuring that unchanged files triggered zero re-embeddings, modified files safely purged outdated chunks, and removed files had all orphan vectors cleaned up required building an atomic reconciliation layer across the filesystem and vector database.

### (f) Which Kubernetes object carries which RAGOps responsibility?
In the RAHBAR architecture, operational responsibilities are separated cleanly across specialized Kubernetes primitives:
- **`StatefulSet` (`chroma`)**: Provides stable network identity (`chroma-0`) and ordered binding to persistent block storage (`/chroma/data`), ensuring vector indices survive pod rescheduling without index loss.
- **`Job` (`ingestion`, `eval`)**: Executes ephemeral batch tasks—running verification/ingestion (`python -m pipeline.run`) and release gate suites (`python eval/run_eval.py`) to completion with explicit Unix exit codes.
- **`CronJob` (`ingestion`)**: Enforces automated operational rhythm by triggering scheduled corpus re-indexing at 2:00 AM daily (`"0 2 * * *"`).
- **`ConfigMap` (`rahbar-config`)**: Decouples runtime operational parameters (thresholds, collection names, top-$k$) from container images, enabling live threshold adjustments via `kubectl edit`.
- **`Deployment` (`api`, `api-candidate`)**: Manages stateless web service replicas, executing zero-downtime rolling updates (`maxSurge: 1`, `maxUnavailable: 0`) and immediate rollout undo operations.
- **`Service` & `PersistentVolumeClaim` (`rahbar-data`)**: Exposes named ports for traffic routing and Prometheus scraping (`port: 9090`), while the PVC provides unified shared storage for the corpus, quarantine, archive, and traces.

---

## Section 5: Contributions & Academic Integrity Statement

### 5.1 Team Contribution Matrix
| Student Name | Roll Number | Key Deliverables & Responsibilities Owned | Contribution % |
| :--- | :--- | :--- | :--- |
| **Muhammad Ali** | 22i-XXXX | Architectural design, Pipeline stages (`extract`, `clean`, `chunk`, `verify`, `store`), ChromaDB incremental synchronization, orphan cleanup logic. | **25%** |
| **Team Member 2** | 22i-XXXX | Evaluation harness (`eval/run_eval.py`), Module/Component/End-to-End metrics, LLM-Judge agreement benchmark, golden dataset generation. | **25%** |
| **Team Member 3** | 22i-XXXX | Kubernetes manifest authoring (RAGOps F1–F5), PVC storage integration, candidate deployment pipeline, and Prometheus observability configuration. | **25%** |
| **Team Member 4** | 22i-XXXX | Batch B stress testing generation, live query simulation, semantic coverage analyzer (`coverage.py`), and drift analysis reporting. | **25%** |

### 5.2 Academic AI Usage Statement
> **AI Usage Declaration**:
> In accordance with the academic integrity policy for CS4067 / Software Construction and Development:
> 1. Generative AI assistants (Claude / Antigravity Agent) were utilized as intelligent development tools to assist with boilerplate manifest drafting, test generation ideas, and code refactoring.
> 2. All architectural logic, algorithmic validation gates, four-stage verification rules, incremental state reconciliation strategies, and empirical evaluation metrics were designed, inspected, customized, executed, and verified by the team members.
> 3. No proprietary course solutions or unauthorized code were ingested. The empirical figures presented in this report represent authentic measurements executed across the RAHBAR codebase.
