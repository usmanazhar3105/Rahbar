from __future__ import annotations

import hashlib
import math
from typing import Any

DUMMY_MODEL = "dummy-embed"
DUMMY_DIM = 64


def fake_vector(text: str) -> list[float]:
    """Generates a deterministic unit vector from text."""
    seed_hash = hashlib.sha256(text.encode("utf-8")).digest()
    vals = [float((b % 100) + 1) for b in seed_hash]
    while len(vals) < DUMMY_DIM:
        seed_hash = hashlib.sha256(seed_hash).digest()
        vals.extend([float((b % 100) + 1) for b in seed_hash])
    vals = vals[:DUMMY_DIM]
    norm = math.sqrt(sum(v * v for v in vals))
    return [v / norm for v in vals]


def write_dummy_index(settings: Any, client: Any) -> int:
    """Writes a dummy index into the client for testing."""
    target_collection = "rahbar"
    try:
        client.delete_collection(target_collection)
    except Exception:
        pass

    col = client.create_collection(
        name=target_collection,
        metadata={
            "embed_model": DUMMY_MODEL,
            "dimension": DUMMY_DIM,
            "hnsw:space": "cosine",
        },
    )

    dummy_chunks = [
        (
            "dummy-chunk-1",
            "The minimum CGPA requirement is 2.00.",
            fake_vector("The minimum CGPA requirement is 2.00."),
            {
                "source": "academic-rules-2026.md",
                "section": "Section 1",
                "page": 1,
                "doc_version": "2026",
                "superseded": False,
            },
        ),
        (
            "dummy-chunk-2",
            "Attendance must be at least 75%.",
            fake_vector("Attendance must be at least 75%."),
            {
                "source": "academic-rules-2026.md",
                "section": "Section 2",
                "page": 2,
                "doc_version": "2026",
                "superseded": False,
            },
        ),
        (
            "dummy-chunk-3",
            "Superceded academic warning rule from 2023.",
            fake_vector("Superceded academic warning rule from 2023."),
            {
                "source": "academic-rules-2023.md",
                "section": "Section 2",
                "page": 1,
                "doc_version": "2023",
                "superseded": True,
            },
        ),
    ]

    col.add(
        ids=[c[0] for c in dummy_chunks],
        documents=[c[1] for c in dummy_chunks],
        embeddings=[c[2] for c in dummy_chunks],
        metadatas=[c[3] for c in dummy_chunks],
    )
    return len(dummy_chunks)

