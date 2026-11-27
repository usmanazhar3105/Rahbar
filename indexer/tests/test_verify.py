"""Unit tests for indexer/pipeline/verify.py and rollback in store.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline.store import rollback
from pipeline.verify import verify


def test_verify_quality_checks(tmp_path: Path) -> None:
    quarantine_dir = tmp_path / "quarantine"
    results_dir = tmp_path / "results"

    docs = [
        # 1. Empty text
        {
            "text": "   ",
            "metadata": {"source": "f1.pdf", "title": "F1", "date": "2026", "page_count": 1},
        },
        # 2. Shorter than 50 chars
        {
            "text": "Too short text here.",
            "metadata": {"source": "f2.pdf", "title": "F2", "date": "2026", "page_count": 1},
        },
        # 3. >30% non-ASCII garbled
        {
            "text": "Valid start " + ("\u0627\u0644\u0639\u0631\u0628\u064a\u0629" * 10),
            "metadata": {"source": "f3.pdf", "title": "F3", "date": "2026", "page_count": 1},
        },
        # 4. Valid doc
        {
            "text": "This is a completely valid policy text that has more than fifty characters and normal ASCII letters and words.",
            "metadata": {"source": "valid.pdf", "title": "Valid Policy", "date": "2026", "page_count": 2},
        },
    ]

    passed, quarantined = verify(
        docs,
        run_id="test_run_1",
        quarantine_dir=str(quarantine_dir),
        results_dir=str(results_dir),
    )

    assert len(passed) == 1
    assert len(quarantined) == 3
    assert quarantined[0]["quarantine_reason"]["check"] == "quality"
    assert quarantined[1]["quarantine_reason"]["check"] == "quality"
    assert quarantined[2]["quarantine_reason"]["check"] == "quality"

    # Verify report was written
    report_file = results_dir / "verify_test_run_1.json"
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["total"] == 4
    assert report["passed"] == 1
    assert report["quarantined"] == 3


def test_verify_completeness_checks(tmp_path: Path) -> None:
    quarantine_dir = tmp_path / "quarantine"
    results_dir = tmp_path / "results"

    docs = [
        # Missing metadata field
        {
            "text": "This text is definitely long enough to pass quality check, but metadata is missing date.",
            "metadata": {"source": "f1.pdf", "title": "F1", "page_count": 1},
        },
        # Text ends with ellipsis
        {
            "text": "This text is definitely long enough to pass quality check, but it was cut off with an ellipsis…",
            "metadata": {"source": "f2.pdf", "title": "F2", "date": "2026", "page_count": 1},
        },
        # Text cut at round number of chars (e.g. 100)
        {
            "text": "a" * 100,
            "metadata": {"source": "f3.pdf", "title": "F3", "date": "2026", "page_count": 1},
        },
    ]

    passed, quarantined = verify(
        docs,
        run_id="test_run_2",
        quarantine_dir=str(quarantine_dir),
        results_dir=str(results_dir),
    )

    assert len(passed) == 0
    assert len(quarantined) == 3
    for q in quarantined:
        assert q["quarantine_reason"]["check"] == "completeness"


def test_verify_recency_check(tmp_path: Path) -> None:
    quarantine_dir = tmp_path / "quarantine"
    results_dir = tmp_path / "results"
    registry_file = tmp_path / "versions.json"

    registry_file.write_text(
        json.dumps({
            "rules.pdf:Academic Rules": "2026-06",
        }),
        encoding="utf-8",
    )

    docs = [
        # Older version (2023 < 2026-06)
        {
            "text": "This text is definitely long enough to pass quality check and completeness checks cleanly.",
            "metadata": {"source": "rules.pdf", "title": "Academic Rules", "date": "2023-08", "page_count": 1},
        },
        # Newer version (2026-08 > 2026-06)
        {
            "text": "This text is also long enough and represents a brand new updated policy document.",
            "metadata": {"source": "rules.pdf", "title": "Academic Rules", "date": "2026-08", "page_count": 1},
        },
    ]

    passed, quarantined = verify(
        docs,
        run_id="test_run_3",
        registry_path=str(registry_file),
        quarantine_dir=str(quarantine_dir),
        results_dir=str(results_dir),
    )

    assert len(quarantined) == 1
    assert quarantined[0]["quarantine_reason"]["check"] == "recency"
    assert len(passed) == 1


def test_verify_uniqueness_check(tmp_path: Path) -> None:
    quarantine_dir = tmp_path / "quarantine"
    results_dir = tmp_path / "results"

    sample_text = "This is a specific distinct policy body that is well over fifty characters long for duplicate testing."

    docs = [
        {
            "text": sample_text,
            "metadata": {"source": "original.pdf", "title": "Original", "date": "2026", "page_count": 1},
        },
        # Exact duplicate
        {
            "text": sample_text,
            "metadata": {"source": "copy.pdf", "title": "Copy", "date": "2026", "page_count": 1},
        },
    ]

    passed, quarantined = verify(
        docs,
        run_id="test_run_4",
        quarantine_dir=str(quarantine_dir),
        results_dir=str(results_dir),
    )

    assert len(passed) == 1
    assert len(quarantined) == 1
    assert quarantined[0]["quarantine_reason"]["check"] == "uniqueness"


def test_rollback_function() -> None:
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["c1", "c2"],
        "metadatas": [
            {"doc_id": "doc123", "doc_version": "2023", "archived": True},
            {"doc_id": "doc123", "doc_version": "2026", "archived": False},
        ],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    from unittest.mock import patch

    with patch("pipeline.store.get_chroma_client", return_value=mock_client), \
         patch("pipeline.store.flush_redis_cache"):
        result = rollback(doc_id="doc123", version="2023", chroma_host="chroma", chroma_port=8000)

        assert result["restored"] == 1
        assert result["archived"] == 1
        assert mock_collection.update.call_count >= 1


def test_run_pipeline_batch_b_verification_and_frozen_settings_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from unittest.mock import patch

    from pipeline.config import get_settings
    from pipeline.run import discover, run_pipeline

    batch_b_dir = Path(__file__).resolve().parent.parent.parent / "corpus" / "batch_b"
    if not batch_b_dir.is_dir():
        pytest.skip("corpus/batch_b not present")

    discovered = discover(str(batch_b_dir))
    names = {p.name for p, _, _ in discovered}
    assert "corrupt.bin" in names
    assert "missing_meta.txt" in names
    assert "exact_dup_1.txt" in names

    monkeypatch.setenv("CORPUS_DIR", str(batch_b_dir))
    monkeypatch.setenv("QUARANTINE_DIR", str(tmp_path / "quarantine"))
    monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setenv("REGISTRY_PATH", str(tmp_path / "registry" / "versions.json"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))

    settings = replace(get_settings(), chunk_size=300, chunk_overlap=30, collection="rahbar_candidate")
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": [], "metadatas": []}
    mock_col.metadata = {"embed_model": settings.embed_model, "dimension": 768}
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col

    with (
        patch("pipeline.store.get_chroma_client", return_value=mock_client),
        patch("pipeline.embed.embed", side_effect=lambda texts, _s: [[0.1] * 768 for _ in texts]),
        patch("pipeline.store.flush_redis_cache"),
    ):
        embedded_count = run_pipeline(settings)

    assert embedded_count > 0
    verify_reports = list((tmp_path / "results").glob("verify_*.json"))
    assert len(verify_reports) == 1
    report = json.loads(verify_reports[0].read_text(encoding="utf-8"))
    # Empty, corrupt.bin, truncated, missing_meta, exact_dup_2 must be quarantined
    assert report["quarantined"] >= 5
