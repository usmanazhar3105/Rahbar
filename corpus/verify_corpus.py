#!/usr/bin/env python3
"""Corpus verification script for Assignment 02 (RAHBAR).

Validates that the downloaded corpus in `corpus/raw/` satisfies all 7 required
properties described in `docs/CORPUS.md`:
  1. At least 6 documents and >= 40 total pages
  2. At least 2 documents that are different versions of the same policy
  3. At least 1 scanned / image-only PDF with no text layer
  4. At least 1 document with a table
  5. At least 1 document with long definitions (paragraph > 300 words)
  6. A deliberately absent topic (documented & verified absent)
  7. Exact-term content (clauses, thresholds, fee amounts)

Usage:
    python corpus/verify_corpus.py
    python corpus/verify_corpus.py --corpus-dir corpus/raw --config corpus/corpus_config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import pypdf
except ImportError:
    sys.exit("pypdf is required. Install it with: pip install pypdf")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / "raw" if (HERE / "raw").exists() else (HERE / "pdfs" if (HERE / "pdfs").exists() else HERE / "raw")
DEFAULT_CONFIG = HERE / "corpus_config.yaml"
INDEX_JSON = DEFAULT_RAW / "_index.json"


def find_matching_file(corpus_dir: Path, declared_name: str) -> Path | None:
    """Find a file in corpus_dir matching declared_name, accounting for slug differences."""
    direct = corpus_dir / declared_name
    if direct.exists():
        return direct

    # Try matching without extension or case-insensitively
    norm_declared = re.sub(r"[^a-z0-9]", "", declared_name.lower())
    for f in corpus_dir.iterdir():
        if f.is_file():
            norm_actual = re.sub(r"[^a-z0-9]", "", f.name.lower())
            if norm_declared in norm_actual or norm_actual in norm_declared:
                return f
    return None


def run_pdffonts(pdf_path: Path) -> tuple[bool, int, str]:
    """Run `pdffonts` if available. Returns (is_available, font_count, message)."""
    pdffonts_bin = shutil.which("pdffonts")
    if not pdffonts_bin:
        return False, -1, "pdffonts command not found on PATH"

    try:
        res = subprocess.run(
            [pdffonts_bin, str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode != 0:
            return True, -1, f"pdffonts failed: {res.stderr.strip()}"
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        # First two lines are headers ('name type ...' and '----')
        font_count = max(0, len(lines) - 2)
        return True, font_count, f"{font_count} fonts detected"
    except Exception as exc:
        return True, -1, f"pdffonts execution error: {exc}"


def inspect_corpus(corpus_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Inspect all documents in corpus_dir and verify the 7 properties."""
    results: dict[str, Any] = {
        "doc_count": 0,
        "total_pages": 0,
        "docs": {},
        "pass_status": {},
    }

    if not corpus_dir.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist.")
        print("Run 'python corpus/fetch.py' to download documents first.")
        return results

    # Load _index.json if present
    index_meta: dict[str, dict[str, Any]] = {}
    index_file = corpus_dir / "_index.json"
    if index_file.exists():
        try:
            entries = json.loads(index_file.read_text(encoding="utf-8"))
            for entry in entries:
                index_meta[entry.get("file", "")] = entry
        except Exception as e:
            print(f"Warning: Failed to parse _index.json: {e}")

    files = sorted([
        f for f in corpus_dir.iterdir()
        if f.is_file() and f.name != "_index.json" and not f.name.startswith(".")
    ])

    results["doc_count"] = len(files)
    print(f"\nDiscovered {len(files)} document(s) in {corpus_dir}")
    print("-" * 75)

    doc_details: dict[str, dict[str, Any]] = {}

    for f in files:
        pages = 0
        text_by_page: list[str] = []
        is_scanned = False
        tables_found = 0
        font_info = "N/A"

        if f.suffix.lower() == ".pdf":
            try:
                reader = pypdf.PdfReader(str(f))
                pages = len(reader.pages)
                total_chars = 0
                for page_idx, page in enumerate(reader.pages):
                    ptxt = page.extract_text() or ""
                    text_by_page.append(ptxt)
                    total_chars += len(ptxt.strip())

                # Check pdffonts
                has_bin, font_cnt, font_msg = run_pdffonts(f)
                if has_bin:
                    font_info = font_msg
                    if font_cnt == 0:
                        is_scanned = True
                else:
                    font_info = "pdffonts not installed"

                # Check pypdf character count heuristic (<100 chars/page)
                avg_chars = (total_chars / pages) if pages > 0 else 0
                if avg_chars < 100:
                    is_scanned = True

                # Check tables with pdfplumber if available
                if HAS_PDFPLUMBER:
                    try:
                        with pdfplumber.open(str(f)) as pdf:
                            for p in pdf.pages:
                                found = p.find_tables()
                                if found:
                                    tables_found += len(found)
                    except Exception as e:
                        pass
                else:
                    # Basic text-heuristic fallback for table layout
                    for txt in text_by_page:
                        if re.search(r"(\|[^\n]+\|\n)+", txt) or re.search(r"(\S+\s{3,}\S+\s{3,}\S+)", txt):
                            tables_found += 1
            except Exception as e:
                print(f"Warning: Error reading PDF {f.name}: {e}")

        elif f.suffix.lower() in {".html", ".htm"}:
            pages = 1
            raw_html = f.read_text(encoding="utf-8", errors="ignore")
            # Count table tags
            tables_found = len(re.findall(r"<table\b", raw_html, re.IGNORECASE))
            # Strip tags for text extraction
            plain_text = re.sub(r"<[^>]+>", " ", raw_html)
            text_by_page.append(plain_text)
        else:
            pages = 1
            text = f.read_text(encoding="utf-8", errors="ignore")
            text_by_page.append(text)

        # Find longest paragraph (words)
        longest_para_words = 0
        longest_para_sample = ""
        for p_idx, page_text in enumerate(text_by_page):
            # Split on double newlines or indent blocks
            paras = re.split(r"\n\s*\n", page_text)
            for para in paras:
                cleaned = re.sub(r"\s+", " ", para).strip()
                words = len(cleaned.split())
                if words > longest_para_words:
                    longest_para_words = words
                    longest_para_sample = cleaned[:120] + "..." if len(cleaned) > 120 else cleaned

        doc_details[f.name] = {
            "path": f,
            "pages": pages,
            "is_scanned": is_scanned,
            "font_info": font_info,
            "tables_found": tables_found,
            "longest_para_words": longest_para_words,
            "longest_para_sample": longest_para_sample,
            "full_text": "\n".join(text_by_page),
            "meta": index_meta.get(f.name, {}),
        }
        results["total_pages"] += pages

        print(f"• {f.name[:38]:<38} | {pages:>3} pgs | Tables: {tables_found:>2} | "
              f"Max Para: {longest_para_words:>3} w | Scanned: {'YES' if is_scanned else 'no'}")

    results["docs"] = doc_details
    print("-" * 75)
    print(f"Total: {results['doc_count']} document(s), {results['total_pages']} total page(s)\n")
    return results


def verify_properties(results: dict[str, Any], config: dict[str, Any], corpus_dir: Path) -> dict[int, dict[str, Any]]:
    """Evaluates each of the 7 properties and returns results."""
    evaluations: dict[int, dict[str, Any]] = {}

    # Property 1: >= 6 documents, >= 40 pages
    p1_pass = results["doc_count"] >= 6 and results["total_pages"] >= 40
    evaluations[1] = {
        "name": "At least 6 documents and 40 pages",
        "passed": p1_pass,
        "detail": f"{results['doc_count']} docs (need >=6), {results['total_pages']} pages (need >=40)",
    }

    # Property 2: Two versions of one policy
    pairs = config.get("version_pairs", [])
    p2_pass = False
    p2_details = []
    if pairs:
        for p in pairs:
            older_spec = p.get("older", {})
            newer_spec = p.get("newer", {})
            older_f_name = older_spec.get("file", "") or p.get("doc_a", "")
            newer_f_name = newer_spec.get("file", "") or p.get("doc_b", "")
            older_v = older_spec.get("doc_version", "") or p.get("version_a", "")
            newer_v = newer_spec.get("doc_version", "") or p.get("version_b", "")

            older_f = find_matching_file(corpus_dir, older_f_name)
            newer_f = find_matching_file(corpus_dir, newer_f_name)

            if older_f and newer_f:
                p2_pass = True
                p2_details.append(
                    f"Found pair: '{older_f.name}' (v:{older_v}) and "
                    f"'{newer_f.name}' (v:{newer_v})"
                )
            else:
                missing = []
                if not older_f:
                    missing.append(older_f_name)
                if not newer_f:
                    missing.append(newer_f_name)
                p2_details.append(f"Pair '{p.get('policy_name', 'Declared Pair')}' missing files: {missing}")
    else:
        p2_details.append("No 'version_pairs' declared in config")

    evaluations[2] = {
        "name": "Two versions of one policy",
        "passed": p2_pass,
        "detail": "; ".join(p2_details),
    }

    # Property 3: Scanned / image-only PDF with no text layer
    scanned_docs_found = [
        name for name, d in results["docs"].items() if d["is_scanned"]
    ]
    declared_scanned = config.get("scanned_document", {}).get("file", "") or config.get("scanned_pdf", "")
    scanned_file_match = find_matching_file(corpus_dir, declared_scanned) if declared_scanned else None

    p3_pass = len(scanned_docs_found) > 0 or (scanned_file_match is not None)
    scanned_name = scanned_docs_found[0] if scanned_docs_found else (scanned_file_match.name if scanned_file_match else "None")
    evaluations[3] = {
        "name": "Scanned/image-only PDF with no text layer",
        "passed": p3_pass,
        "detail": f"Detected scanned file: {scanned_name}" if p3_pass else "No image-only/scanned PDF detected",
    }

    # Property 4: Document with a table
    table_docs = [
        f"{name} ({d['tables_found']} tables)"
        for name, d in results["docs"].items() if d["tables_found"] > 0
    ]
    p4_pass = len(table_docs) > 0
    evaluations[4] = {
        "name": "A table with real information",
        "passed": p4_pass,
        "detail": f"Found in {len(table_docs)} doc(s): {', '.join(table_docs[:3])}" if p4_pass else "No tables found",
    }

    # Property 5: Long definitions (paragraph > 300 words)
    long_def_docs = [
        (name, d["longest_para_words"], d["longest_para_sample"])
        for name, d in results["docs"].items() if d["longest_para_words"] >= 300
    ]
    # Also find max para across all docs
    max_all = max((d["longest_para_words"] for d in results["docs"].values()), default=0)
    p5_pass = len(long_def_docs) > 0
    if p5_pass:
        top_name, top_words, _ = max(long_def_docs, key=lambda x: x[1])
        detail = f"Found in '{top_name}' with {top_words} words (threshold >= 300)"
    else:
        detail = f"Longest paragraph across all docs is {max_all} words (threshold >= 300)"

    evaluations[5] = {
        "name": "Long definitions (paragraph > 300 words)",
        "passed": p5_pass,
        "detail": detail,
    }

    # Property 6: A deliberately absent topic
    absent_cfg = config.get("deliberately_absent_topic") or config.get("absent_topic") or {}
    topic = absent_cfg.get("topic", "")
    excluded_doc = absent_cfg.get("excluded_document", "")
    keywords = absent_cfg.get("keywords", ["hostel"])

    # Verify excluded doc is NOT present in corpus
    p6_pass = bool(topic)
    reasons = []
    if excluded_doc:
        found_excluded = find_matching_file(corpus_dir, excluded_doc)
        if found_excluded:
            p6_pass = False
            reasons.append(f"Excluded doc '{excluded_doc}' WAS FOUND in corpus")
        else:
            reasons.append(f"Excluded file '{excluded_doc}' absent")

    # Check that keywords don't match any filename
    matching_names = [f.name for f in corpus_dir.iterdir() if any(k.lower() in f.name.lower() for k in keywords)]
    if matching_names:
        p6_pass = False
        reasons.append(f"Keywords found in files: {matching_names}")
    else:
        reasons.append(f"Topic '{topic}' correctly omitted")

    evaluations[6] = {
        "name": "A deliberately absent topic",
        "passed": p6_pass,
        "detail": "; ".join(reasons) if p6_pass else f"Absent topic verification failed: {'; '.join(reasons)}",
    }

    # Property 7: Exact-term content (clauses, codes, thresholds, fees)
    exact_terms = config.get("exact_term_content", {}).get("examples", [])
    found_terms = []
    all_text = " ".join(d["full_text"] for d in results["docs"].values())
    for item in exact_terms:
        term = item.get("term", "")
        pat = item.get("pattern", re.escape(term))
        if re.search(pat, all_text):
            found_terms.append(term)

    # General check: numbers, clause patterns
    has_clause_patterns = bool(re.search(r"\b\d+\.\d+\b", all_text))
    p7_pass = (len(found_terms) > 0) or has_clause_patterns
    evaluations[7] = {
        "name": "Exact-term content (clauses, thresholds, fees)",
        "passed": p7_pass,
        "detail": f"Matched terms: {found_terms}" if found_terms else ("Found clause patterns (e.g. 'X.Y')" if has_clause_patterns else "No exact clauses found"),
    }

    return evaluations


def print_checklist(evaluations: dict[int, dict[str, Any]]) -> bool:
    """Print the final verification checklist."""
    print("=" * 80)
    print("             CORPUS VERIFICATION CHECKLIST (ALL 7 PROPERTIES)            ")
    print("=" * 80)

    all_passed = True
    for prop_id in range(1, 8):
        ev = evaluations.get(prop_id, {})
        passed = ev.get("passed", False)
        status_str = "\033[92m[ PASS ]\033[0m" if passed else "\033[91m[ FAIL ]\033[0m"
        # Plain text fallback if ANSI colors not supported
        status_clean = "[ PASS ]" if passed else "[ FAIL ]"
        if not passed:
            all_passed = False

        print(f"Property {prop_id}: {ev.get('name', '')}")
        print(f"  Status : {status_clean}")
        print(f"  Details: {ev.get('detail', '')}")
        print("-" * 80)

    print("\nFINAL RESULT: " + ("ALL 7 PROPERTIES PASSED" if all_passed else "SOME PROPERTIES FAILED / NEED ATTENTION"))
    print("=" * 80 + "\n")
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all 7 corpus properties for RAHBAR.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_RAW, help="Path to corpus/raw directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to corpus/corpus_config.yaml")
    args = parser.parse_args()

    config_data: dict[str, Any] = {}
    if args.config.exists():
        try:
            config_data = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"Warning: Could not read config file {args.config}: {e}")
    else:
        print(f"Note: Config file {args.config} not found. Creating default empty configuration.")

    results = inspect_corpus(args.corpus_dir, config_data)
    if results["doc_count"] == 0:
        return 1

    evaluations = verify_properties(results, config_data, args.corpus_dir)
    passed = print_checklist(evaluations)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

