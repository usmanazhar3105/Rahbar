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
from pathlib import Path
from typing import Any

from .config import IndexerSettings, get_settings
from .models import Chunk


def log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer", "msg": msg, **fields}), flush=True)


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers (used by eval/fixtures)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document to index.

    corpus/fetch.py writes _index.json next to the files it downloads; use it when
    present. Otherwise read Markdown front matter (the CI fixtures work this way).
    """
    root = Path(corpus_dir)
    index = root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        return [(root / e["file"], str(e.get("doc_version", "unknown")),
                 bool(e.get("superseded", False))) for e in entries]
    found = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        found.append((path, meta.get("doc_version", "unknown"),
                      meta.get("superseded", "false").lower() == "true"))
    return found


def run_pipeline(settings: IndexerSettings) -> int:
    from .chunk import chunk
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import (
        archive_superseded_version,
        flush_redis_cache,
        get_chroma_client,
        get_existing_docs_in_chroma,
        get_or_create_collection,
        store,
    )
    from .verify import verify

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

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

        # Concatenate full page text
        raw_text = "\n\n".join(p.text for p in cleaned_doc.pages)
        title = path.stem.replace("_", " ").replace("-", " ").title()
        if path.suffix.lower() == ".md":
            fm = _front_matter(path)
            title = fm.get("title", title)
            doc_version = fm.get("doc_version", doc_version)

        # doc_id: SHA-256 hash of the source path + title
        doc_id = hashlib.sha256(f"{str(path)}:{title}".encode("utf-8")).hexdigest()
        raw_text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        extracted_docs.append({
            "text": raw_text,
            "metadata": {
                "source": str(path),
                "title": title,
                "date": doc_version,
                "page_count": len(cleaned_doc.pages) or 1,
            },
            "doc_id": doc_id,
            "raw_text_hash": raw_text_hash,
            "doc_version": doc_version,
            "superseded": superseded,
            "path": str(path),
            "_extracted_doc": cleaned_doc,
        })

    # -------------------------------------------------------------------------
    # 2. Verification stage (AFTER extraction, BEFORE chunking)
    # -------------------------------------------------------------------------
    passed_docs, quarantined_docs = verify(extracted_docs, run_id=run_id)
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
    corpus_doc_ids = {d["doc_id"] for d in extracted_docs}

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
                    archive_base_dir="/data/archive",
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

    # -------------------------------------------------------------------------
    # 6. Summary and ingest report
    # -------------------------------------------------------------------------
    summary = f"unchanged={unchanged_count}  updated={updated_count}  deleted={deleted_count}  chunks_embedded={chunks_embedded_count}"
    print(summary)

    results_dir = Path("/data/results")
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        report_file = results_dir / f"ingest_{run_id}.json"
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
        log.warning("Could not write ingest results report: %s", exc)

    return chunks_embedded_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in tokens")
    parser.add_argument("--collection", type=str, default=None, help="Target ChromaDB collection name")
    args = parser.parse_args()
    settings = get_settings()

    if args.chunk_size is not None:
        settings.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        settings.chunk_overlap = args.chunk_overlap

    if args.collection is not None:
        settings.collection = args.collection
    elif args.chunk_size is not None or args.chunk_overlap is not None:
        # Candidate parameters provided -> write to rahbar_candidate
        settings.collection = "rahbar_candidate"

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
