# RAHBAR Kubernetes (RAGOps) Operations Guide

This guide details the deployment, configuration, release gate workflow, controlled candidate rollout, observability, and rollback procedures for RAHBAR in Kubernetes.

- **Namespace**: `rahbar`
- **Images**:
  - API (Production): `rahbar-api:latest`
  - API (Candidate): `rahbar-api:candidate`
  - Indexer / Evaluator: `rahbar-indexer:latest`
  - Vector Store: `chromadb/chroma:latest`
  - Cache: `redis:7-alpine`

---

## 1. Directory Structure

```
k8s/
├── README.md
├── base/
│   ├── namespace.yaml
│   ├── serviceaccounts.yaml
│   ├── secret-placeholder.yaml
│   ├── pvc.yaml                   # 5Gi ReadWriteOnce (rahbar-data mounted at /data)
│   ├── configmap.yaml             # Core settings & release gate thresholds
│   ├── services.yaml              # ClusterIP services (includes named metrics port 9090)
│   ├── service.yaml               # Service alias for api-service
│   ├── api-deployment.yaml        # API Deployment (mounts rahbar-data at /data read-only)
│   ├── chroma-statefulset.yaml    # ChromaDB StatefulSet (mounts /chroma/data)
│   ├── redis-deployment.yaml      # Redis Cache Deployment & PVC
│   ├── ui-deployment.yaml         # Frontend Web UI Deployment
│   ├── ingestion-job.yaml         # One-off ingestion job (python -m pipeline.run)
│   ├── ingestion-cronjob.yaml     # Daily ingestion job ("0 2 * * *")
│   ├── eval-job.yaml              # Production evaluation job (python eval/run_eval.py)
│   ├── networkpolicy.yaml         # Ingress/egress isolation rules
│   ├── ingress.yaml               # Ingress routing
│   ├── hpa.yaml                   # Horizontal pod autoscaling
│   └── kustomization.yaml         # Base resource kustomization
└── candidate/
    ├── deployment.yaml            # api-candidate deployment & api-candidate-service (8001)
    └── eval-job-candidate.yaml    # Candidate evaluation job pointing to port 8001
```

---

## 2. Exact `kubectl apply` Order

Apply the manifests in the following exact dependency order:

```bash
# Step 1: Create namespace
kubectl apply -f k8s/base/namespace.yaml

# Step 2: Create service accounts and security permissions
kubectl apply -f k8s/base/serviceaccounts.yaml

# Step 3: Create shared persistent volume claim (rahbar-data)
kubectl apply -f k8s/base/pvc.yaml

# Step 4: Create configuration and secrets
kubectl apply -f k8s/base/configmap.yaml
kubectl apply -f k8s/base/secret-placeholder.yaml

# Step 5: Deploy stateful infrastructure (Vector DB & Cache)
kubectl apply -f k8s/base/chroma-statefulset.yaml
kubectl apply -f k8s/base/redis-deployment.yaml

# Step 6: Create services (including HTTP and metrics port 9090)
kubectl apply -f k8s/base/services.yaml

# Step 7: Deploy API and UI workloads
kubectl apply -f k8s/base/api-deployment.yaml
kubectl apply -f k8s/base/ui-deployment.yaml

# Step 8: Apply ingress, network policies, and autoscaling
kubectl apply -f k8s/base/networkpolicy.yaml
kubectl apply -f k8s/base/ingress.yaml
kubectl apply -f k8s/base/hpa.yaml

# Step 9: Register scheduled CronJob for daily 2:00 AM ingestion
kubectl apply -f k8s/base/ingestion-cronjob.yaml
```

Alternatively, deploy the complete base stack via Kustomize:
```bash
kubectl apply -k k8s/base
```

---

## 3. Running Operations Jobs

### A. Run Ingestion Pipeline
To trigger manual document ingestion across `/data/corpus` with verification, incremental hashing, and orphan cleanup:
```bash
kubectl apply -f k8s/base/ingestion-job.yaml
```
To monitor the ingestion run:
```bash
# Follow logs
kubectl logs -f job/ingestion -n rahbar

# Check job status
kubectl get job ingestion -n rahbar
```

### B. Run Production Evaluation Suite
To execute the automated evaluation suite against the active production service (`http://api-service:8000/api`) with release gate verification:
```bash
kubectl apply -f k8s/base/eval-job.yaml
```
To inspect evaluation logs and gate outcome:
```bash
# Follow eval run logs
kubectl logs -f job/eval -n rahbar

# Verify exit status (0 = PASS, 1 = FAIL)
kubectl get job eval -n rahbar -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}'
```

---

## 4. How to Change a Release Gate Threshold

All gate thresholds are externalized in `rahbar-config` ConfigMap and read at runtime from environment variables:
- `RECALL_AT_5_MIN_DROP`: Maximum allowed drop in Recall@5 vs baseline (points) [default: `"2.0"`]
- `FAITHFULNESS_MIN`: Absolute minimum faithfulness threshold [default: `"0.75"`]
- `P95_LATENCY_MAX_MS`: p95 latency ceiling in milliseconds [default: `"3000"`]
- `ABSTENTION_CORRECTNESS_MIN`: Minimum abstention correctness rate [default: `"1.0"`]

To edit thresholds live without rebuilding any container image:
```bash
kubectl edit configmap rahbar-config -n rahbar
```
Modify the desired keys under `data:`, save and exit. Subsequent runs of `eval-job` and `gate.py` immediately reflect the updated thresholds.

---

## 5. Controlled Candidate Release and Rollback (F3)

To test a candidate model or indexing configuration (e.g. chunk size 300, overlap 30) safely against the isolated candidate collection `rahbar_candidate`:

### Step 1: Deploy Candidate Stack
```bash
kubectl apply -f k8s/candidate/deployment.yaml
```
This spins up:
- Deployment `api-candidate` running `rahbar-api:candidate` with `COLLECTION_NAME=rahbar_candidate`
- Service `api-candidate-service` accessible internally at port `8001`

### Step 2: Run Evaluation Against Candidate
```bash
kubectl apply -f k8s/candidate/eval-job-candidate.yaml
```
Monitor the evaluation:
```bash
kubectl logs -f job/eval-candidate -n rahbar
```

### Step 3: Promote Candidate if Gate Passes
If `job/eval-candidate` succeeds (exit code 0), promote the candidate image to production:
```bash
# Release candidate if gate passes
kubectl set image deployment/api api=rahbar-api:candidate -n rahbar
kubectl rollout status deployment/api -n rahbar --timeout=120s
```

### Step 4: Emergency Rollback
If any unexpected latency or quality regression occurs in production after promotion:
```bash
# Rollback to previous deployment revision
kubectl rollout undo deployment/api -n rahbar
kubectl rollout status deployment/api -n rahbar --timeout=120s
```

---

## 6. Observability (F4)

### A. Structured Trace Logs
Every request logs a single JSON line to `stdout` containing:
- `ts`: ISO-8601 UTC timestamp
- `request_id`: unique trace identifier propagated through all pipeline stages
- `retriever`: retriever provider name (e.g. `chroma`)
- `provider`: generation provider (e.g. `groq`)
- `top_k`: retrieved passages count
- `chunk_ids`: retrieved chunk identifiers
- `similarity_scores`: vector similarity scores
- `abstained`: boolean indicating whether the query abstained
- `latency_ms`: end-to-end request latency

View live structured logs via `kubectl logs`:
```bash
kubectl logs -l app=api -n rahbar -f --tail=100
```

### B. Prometheus Metrics & Port-Forward
The Prometheus `/metrics` endpoint is exposed on port `8000` inside the container and mapped to named port `metrics` on port `9090` of `api-service`.

To scrape or inspect metrics locally:
```bash
kubectl port-forward svc/api-service 9090:9090 -n rahbar
```
Then visit or curl:
```bash
curl http://localhost:9090/metrics
```

---

## 7. Resource Allocation and Limits (F5)

All Deployments and Jobs specify standard baseline resource constraints:
```yaml
resources:
  requests:
    cpu: "250m"
    memory: "512Mi"
  limits:
    cpu: "1000m"
    memory: "1Gi"
```

### Measured Profiling and Sizing Justification
- **API Pods**:
  - Steady-state idle: ~110 MiB memory, ~15m CPU.
  - Active chat/retrieval throughput under load: ~280–380 MiB memory, ~180–320m CPU.
  - Sizing: Request `250m` / `512Mi` ensures guaranteed cluster scheduling. Limit `1000m` / `1Gi` provides ample headroom for concurrent requests while preventing out-of-memory cascading kills.
- **Indexer & Eval Pods**:
  - Embedding computation and PDF parsing peak at ~485 MiB memory (observed via `kubectl top pod`).
  - Request `250m` / `512Mi` and limit `1000m` / `1Gi` safely accommodate batch tokenization and OCR rendering without triggering OOMKill (exit code 137).

To inspect live resource usage in a cluster:
```bash
kubectl top pod -n rahbar
```
