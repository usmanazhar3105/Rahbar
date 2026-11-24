"""Stage 1.5: Verification. Quality, completeness, recency, and uniqueness checks.

Runs AFTER extraction but BEFORE chunking.
Quarantines defective documents and outputs verification reports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("rahbar.verify")

# Lazy-loaded sentence-transformers model
_SENTENCE_TRANSFORMER = None


def get_sentence_transformer() -> Any | None:
    """Lazy load all-MiniLM-L6-v2 embedding model for near-duplicate detection."""
    global _SENTENCE_TRANSFORMER
    if _SENTENCE_TRANSFORMER is not None:
        return _SENTENCE_TRANSFORMER
    try:
        from sentence_transformers import SentenceTransformer

        _SENTENCE_TRANSFORMER = SentenceTransformer("all-MiniLM-L6-v2")
        return _SENTENCE_TRANSFORMER
    except Exception as exc:
        log.warning("sentence-transformers (all-MiniLM-L6-v2) not available (%s).", exc)
        return None


def is_cut_at_round_chars(text: str) -> bool:
    """Detect if text was truncated at a round character buffer limit."""
    t_len = len(text)
    t_strip = len(text.strip())

    round_thresholds = {
        100, 200, 250, 300, 400, 500, 1000, 2000, 2500, 5000, 10000,
        256, 512, 1024, 2048, 4096,
    }
    if t_len in round_thresholds or t_strip in round_thresholds:
        return True
    if t_len >= 100 and (t_len % 100 == 0 or t_strip % 100 == 0):
        return True
    return False


def load_registry(registry_path: str) -> dict[str, Any]:
    """Load versions.json registry if it exists."""
    path = Path(registry_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to load versions registry at %s: %s", registry_path, exc)
        return {}


def check_recency(source: str, title: str, doc_date: str, registry: dict[str, Any]) -> str | None:
    """Compare document's (source + title) fingerprint against versions registry.

    Returns an error reason string if a newer version (higher date) already exists.
    """
    if not registry or not doc_date:
        return None

    # Supported key formats in registry
    candidate_keys = [
        f"{source}:{title}",
        f"{source}::{title}",
        f"{source}_{title}",
        f"{source}|{title}",
        source,
        title,
    ]

    reg_date = None
    for k in candidate_keys:
        if k in registry:
            val = registry[k]
            if isinstance(val, dict):
                reg_date = str(val.get("date") or val.get("version") or "")
            elif isinstance(val, str):
                reg_date = val
            break

    # If registry is a list of entries
    if reg_date is None and isinstance(registry, list):
        for entry in registry:
            if isinstance(entry, dict):
                s = entry.get("source", "")
                t = entry.get("title", "")
                if (s == source and t == title) or s == source or t == title:
                    reg_date = str(entry.get("date") or entry.get("version") or "")
                    break

    if reg_date and reg_date > str(doc_date):
        return f"Newer version already exists in registry ({reg_date} > {doc_date})"

    return None


def verify(
    documents: list[dict[str, Any]],
    run_id: str | None = None,
    registry_path: str = "/data/registry/versions.json",
    quarantine_dir: str = "/data/quarantine",
    results_dir: str = "/data/results",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify extracted documents against quality, completeness, recency, and uniqueness.

    Args:
        documents: List of dicts, each with keys 'text' and 'metadata'
                   where metadata contains 'source', 'title', 'date', 'page_count'.
        run_id: Optional run identifier.
        registry_path: Path to versions.json registry.
        quarantine_dir: Directory where quarantined docs are copied.
        results_dir: Directory where verification reports are written.

    Returns:
        (passed_docs, quarantined_docs)
    """
    if not run_id:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    registry = load_registry(registry_path)
    model = get_sentence_transformer()

    passed_docs: list[dict[str, Any]] = []
    quarantined_docs: list[dict[str, Any]] = []
    per_doc_results: list[dict[str, Any]] = []

    seen_hashes: set[str] = set()
    seen_embeddings: list[Any] = []

    for idx, doc in enumerate(documents):
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        source = meta.get("source", f"doc_{idx}") if isinstance(meta, dict) else f"doc_{idx}"
        title = meta.get("title", "") if isinstance(meta, dict) else ""
        date = meta.get("date", "") if isinstance(meta, dict) else ""

        quarantine_reason: dict[str, str] | None = None

        # ---------------------------------------------------------------------
        # 1. Quality Check
        # ---------------------------------------------------------------------
        if not text or len(text.strip()) == 0:
            quarantine_reason = {"check": "quality", "reason": "empty text"}
        elif len(text.strip()) < 50:
            quarantine_reason = {"check": "quality", "reason": f"text shorter than 50 chars ({len(text.strip())} chars)"}
        else:
            total_chars = len(text)
            non_ascii_count = sum(1 for ch in text if ord(ch) > 127)
            if total_chars > 0 and (non_ascii_count / total_chars) > 0.30:
                pct = round((non_ascii_count / total_chars) * 100, 1)
                quarantine_reason = {
                    "check": "quality",
                    "reason": f"garbled text: {pct}% non-ASCII characters (>30%)",
                }

        # ---------------------------------------------------------------------
        # 2. Completeness Check
        # ---------------------------------------------------------------------
        if quarantine_reason is None:
            if not isinstance(meta, dict):
                quarantine_reason = {
                    "check": "completeness",
                    "reason": "metadata must be a dictionary",
                }
            else:
                required_keys = ["source", "title", "date", "page_count"]
                missing = [
                    k for k in required_keys
                    if k not in meta or meta[k] is None or str(meta[k]).strip() == ""
                ]
                if missing:
                    quarantine_reason = {
                        "check": "completeness",
                        "reason": f"missing or empty metadata fields: {', '.join(missing)}",
                    }
                elif text.rstrip().endswith("…") or text.rstrip().endswith("..."):
                    quarantine_reason = {
                        "check": "completeness",
                        "reason": "text ends with ellipsis (cut-off text)",
                    }
                elif is_cut_at_round_chars(text):
                    quarantine_reason = {
                        "check": "completeness",
                        "reason": f"text cut at a round number of characters (length: {len(text)})",
                    }

        # ---------------------------------------------------------------------
        # 3. Recency Check
        # ---------------------------------------------------------------------
        if quarantine_reason is None:
            recency_error = check_recency(source=source, title=title, doc_date=str(date), registry=registry)
            if recency_error:
                quarantine_reason = {"check": "recency", "reason": recency_error}

        # ---------------------------------------------------------------------
        # 4. Uniqueness Check
        # ---------------------------------------------------------------------
        if quarantine_reason is None:
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                quarantine_reason = {
                    "check": "uniqueness",
                    "reason": "exact duplicate text (SHA-256 match)",
                }
            else:
                # Near-duplicate check with sentence-transformers cosine similarity > 0.95
                if model is not None and seen_embeddings:
                    try:
                        import numpy as np

                        emb = model.encode([text], normalize_embeddings=True)[0]
                        max_sim = 0.0
                        for prev_emb in seen_embeddings:
                            sim = float(np.dot(emb, prev_emb))
                            if sim > max_sim:
                                max_sim = sim
                        if max_sim > 0.95:
                            quarantine_reason = {
                                "check": "uniqueness",
                                "reason": f"near-duplicate text (cosine similarity {max_sim:.4f} > 0.95)",
                            }
                    except Exception as exc:
                        log.warning("Near-duplicate embedding check failed: %s", exc)

        # ---------------------------------------------------------------------
        # Disposition
        # ---------------------------------------------------------------------
        if quarantine_reason is not None:
            doc_copy = dict(doc)
            doc_copy["quarantine_reason"] = quarantine_reason
            quarantined_docs.append(doc_copy)
            per_doc_results.append({
                "source": source,
                "title": title,
                "status": "quarantined",
                "quarantine_reason": quarantine_reason,
            })
        else:
            # Passed all 4 checks
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            seen_hashes.add(text_hash)
            if model is not None:
                try:
                    emb = model.encode([text], normalize_embeddings=True)[0]
                    seen_embeddings.append(emb)
                except Exception:
                    pass

            passed_docs.append(doc)
            per_doc_results.append({
                "source": source,
                "title": title,
                "status": "passed",
            })

    # -------------------------------------------------------------------------
    # Copy quarantined docs to /data/quarantine/<run_id>/
    # -------------------------------------------------------------------------
    target_quarantine_dir = Path(quarantine_dir) / run_id
    try:
        target_quarantine_dir.mkdir(parents=True, exist_ok=True)
        for i, q_doc in enumerate(quarantined_docs):
            safe_source = Path(str(q_doc.get("metadata", {}).get("source", f"doc_{i}"))).name
            q_file = target_quarantine_dir / f"quarantined_{i}_{safe_source}.json"
            q_data = {
                "text": q_doc.get("text", ""),
                "metadata": q_doc.get("metadata", {}),
                "quarantine_reason": q_doc.get("quarantine_reason"),
            }
            q_file.write_text(json.dumps(q_data, indent=2, ensure_ascii=False), encoding="utf-8")

            # If original file exists on filesystem, copy file as well
            raw_path = q_doc.get("path") or q_doc.get("metadata", {}).get("source")
            if raw_path and Path(str(raw_path)).is_file():
                try:
                    shutil.copy2(raw_path, target_quarantine_dir / Path(str(raw_path)).name)
                except Exception:
                    pass
    except Exception as exc:
        log.warning("Could not write quarantined docs to %s: %s", target_quarantine_dir, exc)

    # -------------------------------------------------------------------------
    # Write verification report to /data/results/verify_<run_id>.json
    # -------------------------------------------------------------------------
    report_data = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(documents),
        "passed": len(passed_docs),
        "quarantined": len(quarantined_docs),
        "results": per_doc_results,
    }

    target_results_dir = Path(results_dir)
    try:
        target_results_dir.mkdir(parents=True, exist_ok=True)
        report_path = target_results_dir / f"verify_{run_id}.json"
        report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not write verification report to %s: %s", results_dir, exc)

    return passed_docs, quarantined_docs
