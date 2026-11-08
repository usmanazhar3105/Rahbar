#!/usr/bin/env python3
"""Submission checklist verification script for RAHBAR.

Verifies:
1. Architecture and manifests exist (k8s/base, k8s/overlays)
2. CI/CD GitHub Actions workflows exist (.github/workflows)
3. Corpus and manifest exist (corpus/MANIFEST.md)
4. Evidence documents exist (docs/evidence/)
5. Evaluation results exist and meet required gates (eval/results.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check() -> int:
    failures: list[str] = []
    print("Checking RAHBAR submission criteria...\n")

    # 1. Manifests
    required_manifests = [
        "k8s/base/api-deployment.yaml",
        "k8s/base/chroma-statefulset.yaml",
        "k8s/base/redis-deployment.yaml",
        "k8s/base/ui-deployment.yaml",
        "k8s/base/services.yaml",
        "k8s/base/ingress.yaml",
        "k8s/base/hpa.yaml",
        "k8s/base/networkpolicy.yaml",
        "k8s/base/configmap.yaml",
        "k8s/base/reindex-cronjob.yaml",
        "k8s/base/kustomization.yaml",
        "k8s/overlays/dev/kustomization.yaml",
        "k8s/overlays/prod/kustomization.yaml",
    ]
    for m in required_manifests:
        p = ROOT / m
        if not p.is_file():
            failures.append(f"Missing manifest: {m}")
    print(f"  [✓] Verified {len(required_manifests)} Kubernetes manifests")

    # 2. GitHub Actions Workflows
    required_workflows = [
        ".github/workflows/ci.yml",
        ".github/workflows/cd.yml",
    ]
    for w in required_workflows:
        p = ROOT / w
        if not p.is_file():
            failures.append(f"Missing workflow: {w}")
    print(f"  [✓] Verified {len(required_workflows)} GitHub Actions workflows")

    # 3. Corpus
    manifest_path = ROOT / "corpus" / "MANIFEST.md"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        failures.append("Corpus manifest missing or empty: corpus/MANIFEST.md")
    else:
        print("  [✓] Verified corpus/MANIFEST.md is present and populated")

    # 4. Evidence
    evidence_files = [
        "docs/evidence/03-kubectl-top-pods.txt",
        "docs/evidence/04-kubectl-get-endpoints.txt",
        "docs/evidence/05-kubectl-get-hpa.txt",
    ]
    for ef in evidence_files:
        p = ROOT / ef
        if not p.is_file() or p.stat().st_size == 0:
            failures.append(f"Missing or empty evidence file: {ef}")
    print(f"  [✓] Verified {len(evidence_files)} kubectl evidence output files")

    # 5. Evaluation Results
    results_path = ROOT / "eval" / "results.json"
    if not results_path.is_file():
        failures.append("Missing evaluation results: eval/results.json")
    else:
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            r5 = data.get("recall@5", 0.0)
            cit = data.get("citation_validity", 0.0)
            abst = data.get("abstention_correctness", 0.0)

            if r5 < 0.90:
                failures.append(f"recall@5 too low: {r5:.2f} < 0.90")
            if cit < 0.95:
                failures.append(f"citation_validity too low: {cit:.2f} < 0.95")
            if abst < 1.0:
                failures.append(f"abstention_correctness too low: {abst:.2f} < 1.0")

            print(f"  [✓] Evaluation results: recall@5={r5:.2f}, citation_validity={cit:.2f}, abstention={abst:.2f}")
        except Exception as exc:
            failures.append(f"Error reading eval/results.json: {exc}")

    print()
    if failures:
        print("SUBMISSION CHECKS FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("ALL SUBMISSION CHECKS PASSED SUCCESSFULLY (0 failures).")
    return 0


if __name__ == "__main__":
    sys.exit(check())
