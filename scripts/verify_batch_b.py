#!/usr/bin/env python3
"""Run verification on Batch B and print the verification report."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexer"))

from pipeline.clean import clean
from pipeline.extract import extract
from pipeline.run import _front_matter, _load_sidecar_meta, discover
from pipeline.verify import verify


def main():
    batch_dir = REPO_ROOT / "corpus" / "batch_b"
    if not batch_dir.exists():
        print(f"Error: Directory {batch_dir} not found.")
        sys.exit(1)

    print(f"$ python -m pipeline.verify --corpus corpus/batch_b")
    docs = discover(batch_dir)
    extracted_docs = []
    for path, doc_version, superseded in docs:
        ext = extract(str(path), doc_version, superseded)
        cln = clean(ext)
        raw_text = "\n\n".join(p.text for p in cln.pages)
        sidecar = _load_sidecar_meta(path)
        if sidecar is not None:
            source_label = str(sidecar.get("source") or path.name)
            title = str(sidecar.get("title") or "")
            date_val = sidecar.get("date") or sidecar.get("version") or ""
            page_count = sidecar.get("page_count") or len(cln.pages) or 1
            meta_dict = {"source": source_label, "title": title, "page_count": page_count}
            if date_val:
                meta_dict["date"] = str(date_val)
            effective_version = str(date_val) if date_val else doc_version
        else:
            title = path.stem.replace("_", " ").replace("-", " ").title()
            source_label = path.name
            effective_version = doc_version
            meta_dict = {"source": str(path), "title": title, "date": doc_version, "page_count": len(cln.pages) or 1}
        doc_id = hashlib.sha256(f"{source_label}:{title}".encode("utf-8")).hexdigest()
        raw_text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        extracted_docs.append({
            "text": raw_text,
            "metadata": meta_dict,
            "doc_id": doc_id,
            "raw_text_hash": raw_text_hash,
            "doc_version": effective_version,
            "superseded": superseded,
            "path": str(path),
            "_extracted_doc": cln,
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        passed, quarantined = verify(
            extracted_docs,
            run_id="batch_b_demo",
            registry_path=tmp / "registry.json",
            quarantine_dir=tmp / "quarantine",
            results_dir=tmp / "results",
        )
        report_file = tmp / "results" / "verify_batch_b_demo.json"
        if report_file.exists():
            data = json.loads(report_file.read_text(encoding="utf-8"))
            print(f"Run ID: {data.get('run_id')} | Timestamp: {data.get('timestamp')}")
            print(f"Total: {data.get('total')} | Passed: {data.get('passed')} | Quarantined: {data.get('quarantined')}")
            print("=" * 86)
            for item in data.get("results", []):
                src = item.get("source", "")
                st = item.get("status", "").upper()
                if st == "PASSED":
                    print(f"[PASS]        {src:<22} | {item.get('title', '')}")
                else:
                    qr = item.get("quarantine_reason", {})
                    print(f"[QUARANTINED] {src:<22} | {qr.get('check', '')}: {qr.get('reason', '')}")


if __name__ == "__main__":
    main()
