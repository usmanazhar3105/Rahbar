"""The indexer Job's entry point. It runs once and exits (section 4.2).

    python -m pipeline.run --dummy    # week 1: three fake chunks, no model needed
    python -m pipeline.run            # the real pipeline: extract, verify, clean, chunk, embed, store

Exit code 0 means the index was written; anything else fails the Job, and
Kubernetes retries it up to the Job's backoffLimit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import IndexerSettings, get_settings
from .models import Chunk


def log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer", "msg": msg, **fields}), flush=True)


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers (used by eval/fixtures)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def _load_sidecar_meta(path: Path) -> dict[str, Any] | None:
    """Load `<filename>.meta.json` or `<stem>.meta.json` if present alongside a corpus file."""
    candidates = [
        path.with_name(f"{path.name}.meta.json"),
        path.with_suffix(".meta.json"),
    ]
    for cand in candidates:
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return None


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document to index.

    corpus/fetch.py writes _index.json next to the files it downloads; use it when
    present. Otherwise discover .pdf/.md/.html/.txt/.bin files and read frontmatter or
    `.meta.json` sidecar metadata when present.
    """
    root = Path(corpus_dir)
    index = root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        return [(root / e["file"], str(e.get("doc_version", "unknown")),
                 bool(e.get("superseded", False))) for e in entries]
    found = []
    for path in sorted(root.glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt", ".bin"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        sidecar = _load_sidecar_meta(path)
        if sidecar is not None:
            doc_ver = str(sidecar.get("date") or sidecar.get("version") or "unknown")
            sup = bool(sidecar.get("superseded", False))
            found.append((path, doc_ver, sup))
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        found.append((path, meta.get("doc_version", "unknown"),
                      meta.get("superseded", "false").lower() == "true"))
    return found


def _update_registry(registry_path: str, passed_docs: list[dict[str, Any]]) -> None:
    """Persist verified document versions to registry JSON for recency tracking."""
    reg_file = Path(registry_path)
    try:
        existing: dict[str, Any] = {}
        if reg_file.is_file():
            loaded = json.loads(reg_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        for doc in passed_docs:
            meta = doc.get("metadata", {})
            src = str(meta.get("source", ""))
            title = str(meta.get("title", ""))
            date = str(meta.get("date", ""))
            if src and date:
                key = f"{src}:{title}" if title else src
                prev = existing.get(key, {})
                prev_date = str(prev.get("date", "")) if isinstance(prev, dict) else str(prev)
                if not prev_date or date >= prev_date:
                    existing[key] = {"source": src, "title": title, "date": date}
        reg_file.parent.mkdir(parents=True, exist_ok=True)
        reg_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as exc:
        log("warn_registry_update_failed", error=str(exc))


def run_pipeline(settings: IndexerSettings) -> int:
    from .chunk import chunk
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import (
        archive_superseded_version,
        get_chroma_client,
        get_existing_docs_in_chroma,
        get_or_create_collection,
        store,
    )
    from .verify import verify

    run_id = os.environ.get("RUN_ID") or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    registry_path = os.environ.get("REGISTRY_PATH", "/data/registry/versions.json")
    quarantine_dir = os.environ.get("QUARANTINE_DIR", "/data/quarantine")
    archive_dir = os.environ.get("ARCHIVE_DIR", "/data/archive")
    results_dir = os.environ.get("RESULTS_DIR", "/data/results")

    documents = discover(settings.corpus_dir)
    if not documents:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")

    # -------------------------------------------------------------------------
    # 1. Extraction stage & document dictionary construction
    # -------------------------------------------------------------------------
    extracted_docs: list[dict[str, Any]] = []

    for path, doc_version, superseded in documents:
        extracted_doc = extract(str(path), doc_version, superseded)
        cleaned_doc = clean(extracted_doc)

        raw_text = "\n\n".join(p.text for p in cleaned_doc.pages)
        sidecar = _load_sidecar_meta(path)

        if sidecar is not None:
            source_label = str(sidecar.get("source") or path.name)
            title = str(sidecar.get("title") or "")
            date_val = sidecar.get("date") or sidecar.get("version") or ""
            page_count = sidecar.get("page_count") or len(cleaned_doc.pages) or 1
            meta_dict: dict[str, Any] = {
                "source": source_label,
                "title": title,
                "page_count": page_count,
            }
            if date_val:
                meta_dict["date"] = str(date_val)
            effective_version = str(date_val) if date_val else doc_version
        else:
            title = path.stem.replace("_", " ").replace("-", " ").title()
            if path.suffix.lower() == ".md":
                fm = _front_matter(path)
                title = fm.get("title", title)
                doc_version = fm.get("doc_version", doc_version)
            source_label = path.name
            effective_version = doc_version
            meta_dict = {
                "source": str(path),
                "title": title,
                "date": doc_version,
                "page_count": len(cleaned_doc.pages) or 1,
            }

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
            "_extracted_doc": cleaned_doc,
        })

    # -------------------------------------------------------------------------
    # 2. Verification stage (AFTER extraction, BEFORE chunking)
    # -------------------------------------------------------------------------
    passed_docs, quarantined_docs = verify(
        extracted_docs,
        run_id=run_id,
        registry_path=registry_path,
        quarantine_dir=quarantine_dir,
        results_dir=results_dir,
    )
    print(f"Verification: {len(passed_docs)} passed, {len(quarantined_docs)} quarantined")

    # Connect to ChromaDB
    client = get_chroma_client(settings.chroma_host, settings.chroma_port)
    collection = get_or_create_collection(
        client=client,
        collection_name=settings.collection,
        embed_model=settings.embed_model,
    )

    # -------------------------------------------------------------------------
    # 3. DELETED documents: remove orphans from ChromaDB
    # -------------------------------------------------------------------------
    existing_in_chroma = get_existing_docs_in_chroma(collection)
    corpus_doc_ids = {d["doc_id"] for d in passed_docs}

    deleted_count = 0
    for db_doc_id, chunks_for_doc in existing_in_chroma.items():
        if db_doc_id and db_doc_id not in corpus_doc_ids:
            num_chunks = len(chunks_for_doc)
            collection.delete(where={"doc_id": db_doc_id})
            print(f"Orphan removed: {db_doc_id}, {num_chunks} chunks deleted")
            deleted_count += 1

    # -------------------------------------------------------------------------
    # 4. Ingestion: UNCHANGED, CHANGED, and SUPERSEDED version handling
    # -------------------------------------------------------------------------
    unchanged_count = 0
    updated_count = 0
    chunks_embedded_count = 0

    docs_to_embed: list[dict[str, Any]] = []

    for doc in passed_docs:
        d_id = doc["doc_id"]
        current_hash = doc["raw_text_hash"]
        current_ver = str(doc["doc_version"])

        if d_id in existing_in_chroma:
            chunks = existing_in_chroma[d_id]
            first_meta = chunks[0]["metadata"] if chunks else {}
            db_hash = str(first_meta.get("raw_text_hash", ""))
            db_ver = str(first_meta.get("doc_version", ""))
            is_archived = bool(first_meta.get("archived", False))

            # 1. UNCHANGED documents (same doc_id and same SHA-256 hash of raw text)
            if db_hash == current_hash and not is_archived:
                log(f"Skipping unchanged: {d_id}")
                unchanged_count += 1
                continue

            # 4. SUPERSEDED versions: when a new version of a document arrives (newer date)
            if current_ver > db_ver and db_ver != "":
                archive_superseded_version(
                    collection=collection,
                    doc_id=d_id,
                    old_version=db_ver,
                    old_file_path=doc.get("path"),
                    archive_base_dir=archive_dir,
                )
                updated_count += 1
                docs_to_embed.append(doc)
            else:
                # 2. CHANGED documents: delete existing chunks and re-embed
                collection.delete(where={"doc_id": d_id})
                updated_count += 1
                docs_to_embed.append(doc)
        else:
            # New document not in ChromaDB
            updated_count += 1
            docs_to_embed.append(doc)

    # -------------------------------------------------------------------------
    # 5. Chunk and Embed only passed, non-skipped documents
    # -------------------------------------------------------------------------
    all_new_chunks: list[Chunk] = []
    for doc in docs_to_embed:
        pieces = chunk(doc, settings.chunk_size, settings.chunk_overlap)
        log("chunked", source=doc.get("metadata", {}).get("source", ""), chunks=len(pieces))
        all_new_chunks.extend(pieces)

    if all_new_chunks:
        vectors = embed([c.text for c in all_new_chunks], settings)
        store(all_new_chunks, vectors, settings, replace_all=False)
        chunks_embedded_count = len(all_new_chunks)

    _update_registry(registry_path, passed_docs)

    # -------------------------------------------------------------------------
    # 6. Summary and ingest report
    # -------------------------------------------------------------------------
    summary = (
        f"unchanged={unchanged_count}  updated={updated_count}  "
        f"deleted={deleted_count}  chunks_embedded={chunks_embedded_count}"
    )
    print(summary)

    res_path = Path(results_dir)
    try:
        res_path.mkdir(parents=True, exist_ok=True)
        report_file = res_path / f"ingest_{run_id}.json"
        report_content = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "unchanged": unchanged_count,
            "updated": updated_count,
            "deleted": deleted_count,
            "chunks_embedded": chunks_embedded_count,
        }
        report_file.write_text(json.dumps(report_content, indent=2), encoding="utf-8")
    except Exception as exc:
        log("warn_ingest_report_failed", error=str(exc))

    return chunks_embedded_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in tokens")
    parser.add_argument("--collection", type=str, default=None, help="Target ChromaDB collection name")
    args = parser.parse_args()
    settings = get_settings()

    updates: dict[str, Any] = {}
    if args.chunk_size is not None:
        updates["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        updates["chunk_overlap"] = args.chunk_overlap

    if args.collection is not None:
        updates["collection"] = args.collection
    elif args.chunk_size is not None or args.chunk_overlap is not None:
        updates["collection"] = "rahbar_candidate"

    if updates:
        settings = replace(settings, **updates)

    os.environ["CHROMA_COLLECTION"] = settings.collection

    log("indexer starting", dummy=args.dummy, embed_model=settings.embed_model,
        collection=settings.collection, chroma=f"{settings.chroma_host}:{settings.chroma_port}",
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        pod=os.environ.get("HOSTNAME", "-"))
    started = time.perf_counter()
    try:
        if args.dummy:
            from .dummy import write_dummy_index
            from .store import get_client

            count = write_dummy_index(settings, get_client(settings))
        else:
            count = run_pipeline(settings)
    except NotImplementedError as exc:
        log("indexer stage not implemented yet", error=str(exc))
        return 2
    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
