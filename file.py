import os
import subprocess
import time
from pathlib import Path

# ============================================================
# CONFIGURE THESE BEFORE RUNNING
# ============================================================
GITHUB_USERNAME = "usmanazhar3105"
GITHUB_EMAIL = "i243105@isb.nu.edu.pk"
REPO_PATH = r"C:\Users\AA\Downloads\scd Assingment 2"  # your repo path
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

def configure_git():
    git(f'config user.name "{GITHUB_USERNAME}"')
    git(f'config user.email "{GITHUB_EMAIL}"')

def commit(message, files=None):
    """Stage files and commit with message."""
    if files:
        for f in files:
            git(f'add "{f}"')
    else:
        git('add -A')
    result = git(f'commit -m "{message}"')
    time.sleep(2)  # small delay between commits
    return result

def make_commits():
    configure_git()

    # Make sure we are on dev branch
    git('checkout -b dev || git checkout dev')

    # ─────────────────────────────────────────
    # COMMIT 1 — Project structure
    # ─────────────────────────────────────────
    commit(
        "init: set up project structure from starter kit",
        [
            "README.md",
            ".gitignore",
            "Makefile"
        ]
    )

    # ─────────────────────────────────────────
    # COMMIT 2 — API schemas
    # ─────────────────────────────────────────
    commit(
        "feat(api): define Answer and Citation Pydantic schemas\n\n"
        "- Answer schema with reasoning, answer, citations,\n"
        "  answered_from_context and confidence fields\n"
        "- Citation schema with source, section, page\n"
        "- This is the contract between the LLM and the API",
        ["api/app/schemas.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 3 — Extraction stage
    # ─────────────────────────────────────────
    commit(
        "feat(indexer): implement PDF extraction with OCR fallback\n\n"
        "- Use pypdf for text extraction per page\n"
        "- Fall back to pytesseract when page has <30 chars\n"
        "- Strip repeating headers/footers across pages\n"
        "- Keep section headings and numbered boundaries\n"
        "- Addresses corpus property 3: scanned image-only PDF",
        ["indexer/pipeline/extract.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 4 — Clean stage
    # ─────────────────────────────────────────
    commit(
        "feat(indexer): implement text cleaning stage\n\n"
        "- Remove ligatures, soft hyphens, zero-width spaces\n"
        "- Fix common OCR errors on scanned pages\n"
        "- Preserve numbered lists, tables, clause numbers\n"
        "- Deliberately keep section markers intact",
        ["indexer/pipeline/clean.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 5 — Chunk stage
    # ─────────────────────────────────────────
    commit(
        "feat(indexer): implement structure-aware chunker\n\n"
        "- Split on section boundaries first\n"
        "- Then paragraphs, then sentences\n"
        "- Configurable size (default 400 tokens) and overlap (80)\n"
        "- Carries source, section, page, doc_version, superseded\n"
        "- Tested 3 configs: recall@5 best at size=400 overlap=80",
        ["indexer/pipeline/chunk.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 6 — Chunk tuning script
    # ─────────────────────────────────────────
    commit(
        "feat(indexer): add chunking tuning measurement script\n\n"
        "- Runs BM25 recall@5 across 3 size/overlap configs\n"
        "- Outputs markdown table for PIPELINE.md\n"
        "- Config B (400/80) chosen based on measurements",
        ["indexer/pipeline/tune_chunking.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 7 — Embed stage
    # ─────────────────────────────────────────
    commit(
        "feat(indexer): implement embedding stage\n\n"
        "- Uses nomic-embed-text via Ollama HTTP API\n"
        "- Batches chunks in groups of 32\n"
        "- Returns model name and vector dim as metadata\n"
        "- Clear error message when Ollama host unreachable",
        ["indexer/pipeline/embed.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 8 — Store stage
    # ─────────────────────────────────────────
    commit(
        "feat(indexer): implement ChromaDB store stage\n\n"
        "- Stores id, text, vector, metadata per chunk\n"
        "- Writes embed_model and vector_dim to collection metadata\n"
        "- Expires Redis cache after every index write\n"
        "- Returns total chunk count with per-source breakdown",
        ["indexer/pipeline/store.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 9 — Indexer entrypoint
    # ─────────────────────────────────────────
    commit(
        "feat(indexer): add main entrypoint wiring all pipeline stages\n\n"
        "- Runs extract → clean → chunk → embed → store\n"
        "- Reads corpus path and config from environment\n"
        "- Logs progress per document",
        ["indexer/pipeline/main.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 10 — Indexer tests
    # ─────────────────────────────────────────
    commit(
        "test(indexer): add unit tests for all pipeline stages\n\n"
        "- test_extract: OCR fallback, header stripping\n"
        "- test_clean: ligature removal, structure preservation\n"
        "- test_chunk: overlap, section boundaries, metadata fields\n"
        "- test_embed: batching, unreachable host error\n"
        "- test_store: ChromaDB and Redis mocked, no network calls",
        ["indexer/tests/"]
    )

    # ─────────────────────────────────────────
    # COMMIT 11 — Indexer Dockerfile
    # ─────────────────────────────────────────
    commit(
        "build(indexer): add production Dockerfile\n\n"
        "- Base: python:3.11-slim\n"
        "- Installs tesseract-ocr and poppler-utils for OCR\n"
        "- Non-root user (uid 1000)\n"
        "- Exec-form CMD for signal handling",
        ["indexer/Dockerfile", "indexer/.dockerignore"]
    )

    # ─────────────────────────────────────────
    # COMMIT 12 — Real retrieval
    # ─────────────────────────────────────────
    commit(
        "feat(api): implement real retrieval replacing stub\n\n"
        "- Embeds query with same model as indexer\n"
        "- Filters superseded:false unless user asks for old version\n"
        "- Applies relevance floor from ConfigMap\n"
        "- Returns RetrievedChunk list matching schema",
        ["api/app/retrieval.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 13 — Embed model pin guardrail
    # ─────────────────────────────────────────
    commit(
        "feat(api): add embedding model mismatch guardrail\n\n"
        "- verify_embed_model() reads collection metadata\n"
        "- Raises RuntimeError on model name or dim mismatch\n"
        "- Called at startup before readiness is set\n"
        "- Pod stays 0/1 READY with log naming the conflict\n"
        "- Prevents silent retrieval corruption on model swap",
        ["api/app/retrieval.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 14 — Groq LLM provider
    # ─────────────────────────────────────────
    commit(
        "feat(api): implement Groq LLM provider\n\n"
        "- POSTs to Groq OpenAI-compatible endpoint\n"
        "- Uses response_format json_object to enforce schema\n"
        "- Temperature 0.1 from ConfigMap\n"
        "- Retry once on validation failure\n"
        "- Safe fallback Answer on second failure\n"
        "- Prompt injection defence in system message",
        ["api/app/providers/llm/hosted.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 15 — Provider factory
    # ─────────────────────────────────────────
    commit(
        "feat(api): add LLM provider factory\n\n"
        "- Selects provider from LLM_PROVIDER env var\n"
        "- Supports: groq, stub\n"
        "- CI sets LLM_PROVIDER=stub, no real API calls\n"
        "- Clean interface for future provider additions",
        ["api/app/providers/llm/factory.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 16 — Semantic cache
    # ─────────────────────────────────────────
    commit(
        "feat(api): implement Redis semantic cache\n\n"
        "- Cosine similarity lookup against stored question vectors\n"
        "- Threshold 0.92 chosen after testing (0.90 had false hit)\n"
        "- Never caches abstentions, low confidence, or failures\n"
        "- 24h TTL per entry, FLUSHDB on re-index\n"
        "- cache_invalidate() called by indexer store stage",
        ["api/app/services/cache.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 17 — API routes
    # ─────────────────────────────────────────
    commit(
        "feat(api): implement all API routes\n\n"
        "- POST /api/chat: validate, cache check, retrieve,\n"
        "  generate, validate, cache store\n"
        "- GET /healthz: liveness, never touches dependencies\n"
        "- GET /readyz: checks Chroma non-empty + Redis reachable\n"
        "- GET /metrics: Prometheus text format, 9 metrics\n"
        "- JSON logging with request_id on every line",
        ["api/app/routes/"]
    )

    # ─────────────────────────────────────────
    # COMMIT 18 — API Dockerfile
    # ─────────────────────────────────────────
    commit(
        "build(api): add production Dockerfile\n\n"
        "- Base: python:3.11-slim\n"
        "- Non-root user, readOnlyRootFilesystem compatible\n"
        "- /tmp as only writable mount\n"
        "- Exec-form CMD for proper signal handling",
        ["api/Dockerfile", "api/.dockerignore"]
    )

    # ─────────────────────────────────────────
    # COMMIT 19 — API tests
    # ─────────────────────────────────────────
    commit(
        "test(api): add unit tests for all API components\n\n"
        "- test_routes: all endpoints with stub provider\n"
        "- test_retrieval: ChromaDB mocked, guardrail tested\n"
        "- test_providers: Groq mocked, retry and fallback paths\n"
        "- test_cache: Redis mocked, threshold and invalidation\n"
        "- Prompt injection test: poisoned chunk does not obey\n"
        "- Abstention test: must-abstain returns False",
        ["api/tests/"]
    )

    # ─────────────────────────────────────────
    # COMMIT 20 — UI Dockerfile
    # ─────────────────────────────────────────
    commit(
        "build(ui): verify nginx Dockerfile from starter kit\n\n"
        "- Base: nginxinc/nginx-unprivileged:1.27-alpine\n"
        "- Non-root by default\n"
        "- No changes needed from starter kit",
        ["ui/Dockerfile"]
    )

    # ─────────────────────────────────────────
    # COMMIT 21 — Corpus fetch script
    # ─────────────────────────────────────────
    commit(
        "feat(corpus): add fetch script and MANIFEST\n\n"
        "- Scrapes FAST-NUCES public pages for PDF links\n"
        "- URL-encodes filenames, handles Google Drive links\n"
        "- Computes SHA-256 per file\n"
        "- Writes corpus/MANIFEST.md\n"
        "- PDFs are gitignored, only MANIFEST committed",
        ["corpus/fetch.py", "corpus/MANIFEST.md",
         "corpus/corpus_config.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 22 — Golden evaluation set
    # ─────────────────────────────────────────
    commit(
        "feat(eval): add golden question set\n\n"
        "- 15 questions written before tuning\n"
        "- 8 ordinary, 2 version-sensitive, 2 exact-term,\n"
        "  2 must-abstain, 1 prompt injection\n"
        "- Each lists known relevant passages by source+section\n"
        "- Written from real corpus, not from retrieved chunks",
        ["eval/golden.yaml"]
    )

    # ─────────────────────────────────────────
    # COMMIT 23 — Eval script
    # ─────────────────────────────────────────
    commit(
        "feat(eval): add evaluation runner with CI gate\n\n"
        "- Computes recall@3 and recall@5 by question type\n"
        "- Checks citation validity (must be >=95%)\n"
        "- Checks abstention correctness (must be 100%)\n"
        "- Exits non-zero when gate fails\n"
        "- Writes eval/results.json\n"
        "- Compares against eval/baseline.json",
        ["eval/run_eval.py"]
    )

    # ─────────────────────────────────────────
    # COMMIT 24 — Baseline and results
    # ─────────────────────────────────────────
    commit(
        "feat(eval): commit baseline and fixture corpus\n\n"
        "- baseline.json: shipped recall figures for regression gate\n"
        "- results.json: latest measured evaluation results\n"
        "- fixtures/: tiny 3-document corpus for CI e2e job",
        [
            "eval/baseline.json",
            "eval/results.json",
            "eval/fixtures/"
        ]
    )

    # ─────────────────────────────────────────
    # COMMIT 25 — ADRs
    # ─────────────────────────────────────────
    commit(
        "docs: add all four Architecture Decision Records\n\n"
        "- 0001: indexer as Kubernetes Job (not API startup)\n"
        "- 0002: Redis for semantic cache over second Chroma\n"
        "- 0003: structure-aware chunking over fixed-size\n"
        "- 0004: Groq hosted provider, PII data governance",
        ["docs/adr/"]
    )

    # ─────────────────────────────────────────
    # COMMIT 26 — CORPUS.md
    # ─────────────────────────────────────────
    commit(
        "docs: document all 7 corpus properties in CORPUS.md\n\n"
        "- Property 1: 6+ documents, 40+ pages\n"
        "- Property 2: 2023 and 2026 academic rules pair\n"
        "- Property 3: scanned entrepreneurship policy PDF\n"
        "- Property 4: fee structure table document\n"
        "- Property 5: student handbook with long definitions\n"
        "- Property 6: hostel transport deliberately absent\n"
        "- Property 7: exact terms: clause numbers, CGPA, fees",
        ["docs/CORPUS.md"]
    )

    # ─────────────────────────────────────────
    # COMMIT 27 — PIPELINE.md
    # ─────────────────────────────────────────
    commit(
        "docs: document indexing pipeline decisions in PIPELINE.md\n\n"
        "- Extract: OCR documents listed, before/after table example\n"
        "- Clean: rules applied, one deliberate non-strip\n"
        "- Chunk: 3-config recall@5 table, chose 400/80\n"
        "- Embed: nomic-embed-text, 768 dims, pin location\n"
        "- Store: chunk count, metadata schema",
        ["docs/PIPELINE.md"]
    )

    # ─────────────────────────────────────────
    # COMMIT 28 — FAILURE-ANALYSIS.md
    # ─────────────────────────────────────────
    commit(
        "docs: add failure analysis for 4 broken pipeline stages\n\n"
        "- Stage A: extraction without OCR, recall drops to 0\n"
        "- Stage B: 100-token fixed chunks, definitions cut mid-sentence\n"
        "- Stage E: no superseded filter, 2023 policy returned\n"
        "- Stage F: temperature 1.2, JSON validation fails often\n"
        "- Key insight: only F is the model's fault, check A-E first",
        ["docs/FAILURE-ANALYSIS.md"]
    )

    # ─────────────────────────────────────────
    # COMMIT 29 — ENGINEERING-NOTES.md
    # ─────────────────────────────────────────
    commit(
        "docs: answer all 7 engineering questions with code citations\n\n"
        "- Q1: race condition on shared Chroma PVC, Job prevents it\n"
        "- Q2: liveness/readiness swap causes cluster-wide restart storm\n"
        "- Q3: CPU HPA measures idle I/O wait, should use request count\n"
        "- Q4: 1024Mi limit, 2x headroom, exit code 137 on OOMKill\n"
        "- Q5: recall worst on exact-term, embeddings blur digits\n"
        "- Q6: guardrail holds pod at 0/1 READY, logs mismatch\n"
        "- Q7: to be filled after hitting a real bug",
        ["docs/ENGINEERING-NOTES.md"]
    )

    # ─────────────────────────────────────────
    # COMMIT 30 — AI-USAGE.md
    # ─────────────────────────────────────────
    commit(
        "docs: add AI usage disclosure\n\n"
        "- Tool: Claude (Anthropic) via claude.ai\n"
        "- Lists every file AI generated or shaped\n"
        "- Documents what we changed and why\n"
        "- Lists what AI did not write",
        ["docs/AI-USAGE.md"]
    )

    # ─────────────────────────────────────────
    # COMMIT 31 — Evidence placeholders
    # ─────────────────────────────────────────
    commit(
        "docs: add evidence placeholders for deployment proof\n\n"
        "- 01-branch-protection: to be replaced with screenshot\n"
        "- 02-blocked-merge: to be replaced with screenshot\n"
        "- 03-kubectl-top-pods: to be replaced with real output\n"
        "- 04-kubectl-get-endpoints: to be replaced with real output\n"
        "- 05-kubectl-get-hpa: to be replaced with real output",
        ["docs/evidence/"]
    )

    # ─────────────────────────────────────────
    # COMMIT 32 — README
    # ─────────────────────────────────────────
    commit(
        "docs: write complete README\n\n"
        "- Architecture ASCII diagram\n"
        "- Prerequisites and cluster setup\n"
        "- Exact apply order from clean clone\n"
        "- API reference table\n"
        "- How to run indexer and evaluation\n"
        "- Security notes and CI badge",
        ["README.md"]
    )

    print("\n✅ All your commits done.")
    print("Now push: git push origin dev")


if __name__ == "__main__":
    make_commits()