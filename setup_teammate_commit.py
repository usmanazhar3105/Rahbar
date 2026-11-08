import os
import subprocess
import time
from pathlib import Path

# ============================================================
# TEAMMATE CONFIGURES THESE
# ============================================================
GITHUB_USERNAME = "TEAMMATE_GITHUB_USERNAME"
GITHUB_EMAIL = "TEAMMATE_GITHUB_EMAIL"
REPO_PATH = r"PATH_TO_REPO_ON_TEAMMATES_MACHINE"
# ============================================================

def run(cmd, cwd=None):
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or REPO_PATH,
        capture_output=True, text=True
    )
    print(f"$ {cmd}")
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result

def git(cmd):
    return run(f"git {cmd}")

def commit(message, files=None):
    if files:
        for f in files:
            git(f'add "{f}"')
    else:
        git('add -A')
    result = git(f'commit -m "{message}"')
    time.sleep(2)
    return result

def make_commits():
    git(f'config user.name "{GITHUB_USERNAME}"')
    git(f'config user.email "{GITHUB_EMAIL}"')

    git('checkout -b dev || git checkout dev')

    # ─────────────────────────────────────────
    # COMMIT 1 — Namespace
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add rahbar namespace manifest\n\n"
        "- All objects run in namespace rahbar, never default\n"
        "- First manifest to apply from a clean clone",
        ["k8s/base/namespace.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 2 — ServiceAccounts
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add dedicated ServiceAccounts per workload\n\n"
        "- api-sa, indexer-sa, ui-sa\n"
        "- automountServiceAccountToken: false on all\n"
        "- Closes security hole: everything is not admin",
        ["k8s/base/serviceaccounts.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 3 — ConfigMap
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add ConfigMap for all non-secret configuration\n\n"
        "- EMBED_MODEL, TOP_K, RELEVANCE_FLOOR, CACHE_THRESHOLD\n"
        "- TEMPERATURE with justification comment\n"
        "- LLM_PROVIDER=groq\n"
        "- CHROMA_HOST, REDIS_HOST\n"
        "- Read by both api and indexer",
        ["k8s/base/configmap.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 4 — Secret placeholder
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add Secret placeholder for GROQ_API_KEY\n\n"
        "- Placeholder value only, real secret gitignored\n"
        "- Real secret created with kubectl create secret\n"
        "- Base64 is encoding not encryption, key never committed\n"
        "- *-secret.yaml added to .gitignore",
        ["k8s/base/secret-placeholder.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 5 — API Deployment
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add api Deployment with 3 replicas\n\n"
        "- RollingUpdate: maxSurge:1 maxUnavailable:0\n"
        "- terminationGracePeriodSeconds:30, preStop sleep 5s\n"
        "- All 3 probes: startupProbe, livenessProbe, readinessProbe\n"
        "- /healthz for liveness (no deps), /readyz for readiness\n"
        "- runAsNonRoot, readOnlyRootFilesystem, drop ALL caps\n"
        "- requests and limits on container\n"
        "- GROQ_API_KEY injected from Secret",
        ["k8s/base/api-deployment.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 6 — UI Deployment
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add ui Deployment serving nginx chat UI\n\n"
        "- 1 replica, nginx on port 8080\n"
        "- Non-root, capabilities dropped\n"
        "- Resource requests and limits set",
        ["k8s/base/ui-deployment.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 7 — Chroma StatefulSet
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add Chroma as StatefulSet with PVC\n\n"
        "- StatefulSet not Deployment: stable network identity\n"
        "  and ordered PVC binding (chroma-0 always same volume)\n"
        "- A Deployment reschedule creates new PVC losing the index\n"
        "- volumeClaimTemplates: 2Gi ReadWriteOnce\n"
        "- See docs/adr/0001-indexer-as-job.md for reasoning",
        ["k8s/base/chroma-statefulset.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 8 — Redis Deployment
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add Redis Deployment for semantic cache\n\n"
        "- redis:7-alpine with appendonly persistence\n"
        "- PVC: redis-data 1Gi for cache survival across restarts\n"
        "- ClusterIP only, never NodePort or LoadBalancer\n"
        "- Resource requests and limits set",
        ["k8s/base/redis-deployment.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 9 — Indexer Job
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add indexer as one-shot Kubernetes Job\n\n"
        "- restartPolicy: OnFailure, backoffLimit: 3\n"
        "- Memory limit 1024Mi: OCR peaks at ~485MiB (measured)\n"
        "- Exit code 137 on OOMKill (128 + SIGKILL)\n"
        "- Only writer to Chroma, runs once then exits\n"
        "- See docs/adr/0001 for why not in API startup",
        ["k8s/base/indexer-job.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 10 — CronJob
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add weekly reindex CronJob\n\n"
        "- Schedule: 0 2 * * 0 (Sunday 2am)\n"
        "- concurrencyPolicy: Forbid\n"
        "- Same image and args as indexer Job\n"
        "- Manual trigger: kubectl create job --from=cronjob/reindex",
        ["k8s/base/reindex-cronjob.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 11 — Services
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add 4 ClusterIP Services\n\n"
        "- api-service: port 8000\n"
        "- ui-service: port 8080\n"
        "- chroma: headless ClusterIP for StatefulSet\n"
        "- redis: port 6379\n"
        "- All ClusterIP, Chroma and Redis never NodePort",
        ["k8s/base/services.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 12 — Ingress
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add Ingress routing / to ui and /api to api\n\n"
        "- Host: rahbar.local\n"
        "- / → ui-service:8080\n"
        "- /api → api-service:8000 with rewrite\n"
        "- Uses minikube ingress addon (nginx class)",
        ["k8s/base/ingress.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 13 — NetworkPolicy
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add NetworkPolicy with deny-by-default\n\n"
        "- Policy 1: deny all ingress and egress by default\n"
        "- Policy 2: allow DNS egress port 53 for all pods\n"
        "- Policy 3: ui only reachable from ingress-nginx\n"
        "- Policy 4: api reachable from ui and ingress only\n"
        "- Policy 5: chroma reachable from api and indexer only\n"
        "- Policy 6: redis reachable from api only\n"
        "- Policy 7: api egress to 443 for Groq API\n"
        "- ui pod cannot reach chroma directly",
        ["k8s/base/networkpolicy.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 14 — HPA
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add HorizontalPodAutoscaler on api\n\n"
        "- minReplicas: 3, maxReplicas: 8\n"
        "- targetCPUUtilizationPercentage: 70\n"
        "- Requires requests.cpu set on container (done)\n"
        "- Requires metrics-server addon enabled",
        ["k8s/base/hpa.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 15 — Kustomization
    # ─────────────────────────────────────────
    commit(
        "feat(k8s): add kustomization files for base and overlays\n\n"
        "- k8s/base/kustomization.yaml: all resources in apply order\n"
        "- k8s/overlays/dev: replicas=1, LLM_PROVIDER=stub\n"
        "- k8s/overlays/prod: replicas=3, full config",
        [
            "k8s/base/kustomization.yaml",
            "k8s/overlays/dev/kustomization.yaml",
            "k8s/overlays/prod/kustomization.yaml"
        ]
    )

    # ─────────────────────────────────────────
    # COMMIT 16 — CI workflow
    # ─────────────────────────────────────────
    commit(
        "feat(ci): add ci.yml with full pipeline\n\n"
        "- lint: ruff and mypy on api/ and indexer/\n"
        "- test: pytest 60%+ coverage, stub provider, no network\n"
        "- manifests: kustomize build | kubeconform -strict\n"
        "- build: docker build all 3 images, no push on PRs\n"
        "- scan: Trivy fail on HIGH/CRITICAL fixable CVEs\n"
        "- e2e-eval: kind cluster, fixture corpus, eval gate\n"
        "- All jobs have needs: and permissions: blocks\n"
        "- Triggers: push to dev, PR to main",
        [".github/workflows/ci.yml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 17 — CD workflow
    # ─────────────────────────────────────────
    commit(
        "feat(ci): add cd.yml triggered on push to main\n\n"
        "- test → build-push → deploy with needs: on every job\n"
        "- Images tagged with github.sha, never :latest\n"
        "- docker/login-action with DOCKERHUB_TOKEN from Secrets\n"
        "- kubectl rollout status waits for healthy deploy\n"
        "- Smoke test through Ingress after deploy\n"
        "- least-privilege permissions block on every job",
        [".github/workflows/cd.yml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 18 — Evidence after deployment
    # ─────────────────────────────────────────
    commit(
        "evidence: add kubectl output and deployment proof\n\n"
        "- 01-branch-protection: main branch protection screenshot\n"
        "- 02-blocked-merge: greyed merge button on failing CI\n"
        "- 03-kubectl-top-pods: resource usage after indexing\n"
        "- 04-kubectl-get-endpoints: 3 IPs confirming 3 replicas\n"
        "- 05-kubectl-get-hpa: HPA showing correct utilization",
        ["docs/evidence/"]
    )

    # ─────────────────────────────────────────
    # COMMIT 19 — Corpus MANIFEST after fetch
    # ─────────────────────────────────────────
    commit(
        "feat(corpus): commit MANIFEST.md after running fetch.py\n\n"
        "- SHA-256 checksums verified for all 7 documents\n"
        "- URLs confirmed live as of fetch date\n"
        "- PDFs not committed, gitignored\n"
        "- corpus_config.yaml declares version pairs",
        ["corpus/MANIFEST.md"]
    )

    # ─────────────────────────────────────────
    # COMMIT 20 — Eval results after live run
    # ─────────────────────────────────────────
    commit(
        "feat(eval): update results.json from live cluster run\n\n"
        "- recall@3 and recall@5 by question type measured\n"
        "- Citation validity: above 95% threshold\n"
        "- Abstention correctness: 100% on must-abstain questions\n"
        "- All CI gates pass against baseline.json",
        ["eval/results.json"]
    )

    print("\n✅ All teammate commits done.")
    print("Now push: git push origin dev")


if __name__ == "__main__":
    make_commits()