#!/usr/bin/env python3
"""Download the corpus listed in corpus/sources.yaml, politely, and record it.

    python corpus/fetch.py                       # download what is missing
    python corpus/fetch.py --force               # download everything again
    python corpus/fetch.py --verify              # re-hash local files against MANIFEST.md
    python corpus/fetch.py --list URL            # list the documents linked from an index page

Writes:
    corpus/raw/<file>         the documents (gitignored: never commit someone else's PDFs)
    corpus/pdfs/<file>        the downloaded PDF copies (gitignored)
    corpus/raw/_index.json    file -> doc_version, superseded; the indexer reads this
    corpus/MANIFEST.md        filename, url, sha256, date_fetched, pages (commit this)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import httpx
import yaml

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
PDFS = HERE / "pdfs"
SOURCES = HERE / "sources.yaml"
MANIFEST = HERE / "MANIFEST.md"
USER_AGENT = "RAHBAR-coursework-fetcher/1.0 (CS4032 student project; one request at a time)"
PAUSE_SECONDS = 2.0
DRIVE_VIEW = re.compile(r"https?://drive\.google\.com/file/d/([^/?#]+)")


def normalise(url: str) -> str:
    """Encode spaces and other unsafe characters, and turn Drive viewer links into
    direct downloads. Already-encoded %XX sequences are left alone."""
    match = DRIVE_VIEW.match(url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return quote(url, safe=":/?&=%#+,;@!$'()*~")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:80] or "document"


def guess_extension(url: str, content_type: str) -> str:
    content_type = content_type.lower()
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type:
        return ".html"
    suffix = Path(urlsplit(url).path).suffix.lower()
    return suffix if suffix in {".pdf", ".html", ".htm", ".docx", ".txt", ".md"} else ".bin"


def get(client: httpx.Client, url: str, attempts: int = 3) -> httpx.Response:
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url)
            if response.status_code < 500:
                return response
            reason = f"HTTP {response.status_code}"
        except httpx.TransportError as exc:
            reason = type(exc).__name__
        if attempt < attempts:
            wait = PAUSE_SECONDS * 2 ** attempt
            print(f"    {reason}; retrying in {wait:.0f}s ({attempt}/{attempts - 1})")
            time.sleep(wait)
    raise RuntimeError(f"gave up after {attempts} attempts: {reason}")


def count_pages(content: bytes, filename: str) -> int:
    """Determine the number of pages in a document (using pypdf for PDFs)."""
    if filename.lower().endswith(".pdf") and HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            return len(reader.pages)
        except Exception:
            return 1
    return 1


def load_sources() -> list[dict]:
    if not SOURCES.exists():
        sys.exit("corpus/sources.yaml not found. Copy corpus/sources.example.yaml to "
                 "corpus/sources.yaml and keep the documents you want.")
    data = yaml.safe_load(SOURCES.read_text(encoding="utf-8")) or {}
    docs = data.get("documents") or []
    for i, doc in enumerate(docs):
        if not doc.get("name") or not doc.get("url"):
            sys.exit(f"sources.yaml entry {i + 1} needs both name and url")
    return docs


def fetch(force: bool) -> int:
    docs = load_sources()
    RAW.mkdir(parents=True, exist_ok=True)
    PDFS.mkdir(parents=True, exist_ok=True)
    index_path = RAW / "_index.json"
    previous = {e["name"]: e for e in json.loads(index_path.read_text(encoding="utf-8"))} \
        if index_path.exists() else {}

    entries, failures = [], []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True,
                      timeout=httpx.Timeout(60.0, connect=15.0)) as client:
        for doc in docs:
            name, url = doc["name"], normalise(doc["url"])
            doc_file = doc.get("file")
            old = previous.get(name)
            expected_file = doc_file or (old.get("file") if old else None)

            if old and not force and expected_file and (RAW / expected_file).exists() and old.get("url") == url:
                print(f"  have    {expected_file}")
                entries.append({
                    **old,
                    "file": expected_file,
                    "doc_version": doc.get("doc_version", "unknown"),
                    "superseded": bool(doc.get("superseded", False)),
                })
                # Ensure copy exists in pdfs
                if (RAW / expected_file).exists() and not (PDFS / expected_file).exists():
                    (PDFS / expected_file).write_bytes((RAW / expected_file).read_bytes())
                continue

            print(f"  fetch   {name}")
            try:
                response = get(client, url)
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                content_type = response.headers.get("content-type", "")
                if "drive.google.com" in url and "html" in content_type:
                    raise RuntimeError("Google Drive returned a web page, not the file "
                                       "(too large for direct download, or not shared publicly)")

                file_ext = guess_extension(url, content_type)
                file = doc_file if doc_file else (slug(name) + file_ext)

                # Write to both RAW and PDFS folders
                (RAW / file).write_bytes(response.content)
                (PDFS / file).write_bytes(response.content)

                pages = count_pages(response.content, file)
                sha256_hash = hashlib.sha256(response.content).hexdigest()
                date_fetched = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                entries.append({
                    "name": name,
                    "url": url,
                    "file": file,
                    "bytes": len(response.content),
                    "pages": pages,
                    "sha256": sha256_hash,
                    "content_type": content_type.split(";")[0],
                    "date_fetched": date_fetched,
                    "doc_version": doc.get("doc_version", "unknown"),
                    "superseded": bool(doc.get("superseded", False)),
                })
                print(f"          {file}, {len(response.content):,} bytes, {pages} page(s)")
            except Exception as exc:  # report every failure, keep going
                failures.append((name, str(exc)))
                print(f"  FAILED  {name}: {exc}")
            time.sleep(PAUSE_SECONDS)

    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    (PDFS / "_index.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    write_manifest(entries)
    print(f"\n{len(entries)} documents recorded in corpus/MANIFEST.md and stored in corpus/pdfs/ and corpus/raw/")
    if failures:
        print(f"{len(failures)} failed; fix their URLs in sources.yaml and run again:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        return 1
    return 0


def write_manifest(entries: list[dict]) -> None:
    lines = [
        "# Corpus manifest",
        "",
        "Generated by `corpus/fetch.py`. The documents themselves live in `corpus/raw/` and `corpus/pdfs/`, "
        "which are gitignored; run `python corpus/fetch.py` to download them, then `--verify` to check "
        "you have the same bytes.",
        "",
        "| filename | url | sha256 | date_fetched | pages |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for e in entries:
        filename = e.get("file", "")
        url = e.get("url", "")
        sha256 = e.get("sha256", "")
        date_fetched = e.get("date_fetched", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        pages = e.get("pages", 1)
        lines.append(f"| `{filename}` | {url} | `{sha256}` | {date_fetched} | {pages} |")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify() -> int:
    if not MANIFEST.exists():
        sys.exit("no corpus/MANIFEST.md yet; run fetch first")

    # Parse rows: filename | url | sha256 | date_fetched | pages
    content = MANIFEST.read_text(encoding="utf-8")
    rows = re.findall(r"\|\s*`([^`]+)`\s*\|\s*(https?://[^\s|]+)\s*\|\s*`([0-9a-f]{64})`\s*\|\s*[^|]+\|\s*(\d+)\s*\|", content)
    if not rows:
        # Fallback to legacy format if needed
        rows = [(m[0], "", m[1], 1) for m in re.findall(r"\| `([^`]+)` \| [\d,]+ \| `([0-9a-f]{64})` \|", content)]

    bad = 0
    for file, url, expected, _ in rows:
        path = (RAW / file) if (RAW / file).exists() else (PDFS / file)
        if not path.exists():
            print(f"  missing   {file}")
            bad += 1
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            print(f"  changed   {file}  (the website may have published a new version)")
            bad += 1
        else:
            print(f"  ok        {file}")
    print(f"\n{len(rows) - bad} of {len(rows)} match the manifest")
    return 1 if bad else 0


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None


def list_documents(index_url: str) -> int:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True,
                      timeout=30.0) as client:
        page = get(client, index_url)
    parser = _Links()
    parser.feed(page.text)
    seen = set()
    for href, text in parser.links:
        url = urljoin(index_url, href)
        is_doc = urlsplit(url).path.lower().endswith(".pdf") or DRIVE_VIEW.match(url)
        if is_doc and url not in seen:
            seen.add(url)
            name = text or Path(urlsplit(url).path).name
            print(f"  - name: {json.dumps(name)}\n    url: {json.dumps(url)}")
    print(f"\n{len(seen)} document links on {index_url}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true", help="download everything again")
    group.add_argument("--verify", action="store_true", help="check local files against MANIFEST.md")
    group.add_argument("--list", metavar="URL", help="list document links on an index page")
    args = parser.parse_args()
    if args.verify:
        return verify()
    if args.list:
        return list_documents(args.list)
    return fetch(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
