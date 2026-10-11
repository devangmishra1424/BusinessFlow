"""Tests for the two maintenance functions in rag/ingest.py that clean up
after _supersede_existing_chunks' own deliberate design: it marks a chunk
superseded rather than deleting it (so a correction never erases the
record of what a document said before), which means nothing ever
reclaims that space on its own. Real Postgres/pgvector store, real
docling/embedding calls, same tempfile-based pattern test_retriever.py's
own ingestion tests already use.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from businessflow.accounts.db import get_connection
from businessflow.rag.ingest import ingest_document, purge_orphaned_chunks, purge_superseded_chunks
from businessflow.rag.store import backdate_chunks, delete_chunks_for_document, get_chunk_ids_for_document, normalize_source_document

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres/pgvector",
)


def _write_temp_doc(text: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(text)
        return f.name


def test_purge_superseded_chunks_deletes_only_chunks_older_than_the_cutoff():
    temp_path = _write_temp_doc("# Old version\n\nThe grace period is 3 days.")
    try:
        ingest_document(temp_path, document_type="policy")
        # Re-ingesting supersedes the first generation -- exactly the
        # real-world trigger this function exists to clean up after.
        ingest_document(temp_path, document_type="policy")

        all_chunks = get_chunk_ids_for_document(temp_path)
        superseded_before_ids = [r["id"] for r in all_chunks if r["superseded_at"]]
        assert superseded_before_ids, "expected a real superseded chunk from the re-ingestion above"

        # Not old enough yet -- a generous cutoff must leave it alone.
        deleted = purge_superseded_chunks(older_than_days=30)
        assert deleted == 0
        still_there = get_chunk_ids_for_document(temp_path)
        assert {r["id"] for r in still_there} >= set(superseded_before_ids)

        # Backdate it directly, the same way _supersede_existing_chunks
        # itself sets superseded_at -- simulating "this became superseded
        # 60 days ago" without waiting 60 real days.
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=60)
        backdate_chunks(superseded_before_ids, old_timestamp)

        deleted = purge_superseded_chunks(older_than_days=30)
        assert deleted == len(superseded_before_ids)
        gone = get_chunk_ids_for_document(temp_path)
        assert not (set(superseded_before_ids) & {r["id"] for r in gone})

        # The still-active generation must never be touched by this.
        active = get_chunk_ids_for_document(temp_path)
        assert all(not r["superseded_at"] for r in active)
    finally:
        delete_chunks_for_document(temp_path)
        os.unlink(temp_path)


def test_purge_orphaned_chunks_deletes_chunks_whose_file_is_gone():
    temp_path = _write_temp_doc("# Orphan candidate\n\nSome real policy text.")
    ingest_document(temp_path, document_type="policy")
    os.unlink(temp_path)  # the file is gone, but its chunks are still in the store

    try:
        deleted = purge_orphaned_chunks()
        assert deleted >= 1

        remaining = get_chunk_ids_for_document(temp_path)
        assert remaining == []
    finally:
        delete_chunks_for_document(temp_path)


def test_purge_orphaned_chunks_never_touches_a_document_still_on_disk():
    temp_path = _write_temp_doc("# Still real\n\nThis file is not going anywhere.")
    try:
        ingest_document(temp_path, document_type="policy")

        purge_orphaned_chunks()

        still_there = get_chunk_ids_for_document(temp_path)
        assert still_there, "a document whose file still exists on disk must never be purged"
    finally:
        delete_chunks_for_document(temp_path)
        os.unlink(temp_path)


def test_ingest_document_extracts_text_from_a_scanned_image_only_pdf_via_ocr():
    """Docling's bundled OCR engine (RapidOCR) actually runs and correctly
    reads a genuinely image-only PDF -- no embedded text layer at all, the
    fixture is a plain PIL-rendered image saved as a PDF (a raster image
    in the content stream, no text operators -- mechanically identical to
    a real scanned document, not a real-world scan itself).

    Found live while checking whether OCR was a real gap for this project
    (assumed missing, listed as a Tier 3 item to build): it wasn't --
    DocumentConverter() already runs with docling's own do_ocr=True default,
    confirmed both via extract_document_text() and this full ingest_document()
    path, then cleaned up. This test exists so that fact stays true, not
    just observed once -- a future docling upgrade or config change that
    silently disables OCR would otherwise have no test coverage to catch it."""
    fixture_path = str(_FIXTURES_DIR / "scanned_loan_agreement.pdf")
    try:
        chunks_stored = ingest_document(fixture_path, document_type="loan_agreement", account_id="BF-1001")
        assert chunks_stored == 1

        rows = get_connection().execute(
            "select document_text from document_chunks where source_document = %s and superseded_at is null",
            (normalize_source_document(fixture_path),),
        ).fetchall()
        assert len(rows) == 1
        text = rows[0]["document_text"]
        # Every real figure from the source image -- confirms OCR actually
        # read the numbers correctly, not just "found some text somewhere".
        assert "14.75" in text
        assert "500000" in text
        assert "36" in text
    finally:
        delete_chunks_for_document(fixture_path)


# ---------------------------------------------------------------------------------------------------------------------
# All-or-nothing ingestion, serialised per document, and a retriever that is one consistent snapshot.
# ---------------------------------------------------------------------------------------------------------------------

import threading  # noqa: E402

import psycopg  # noqa: E402

from businessflow.rag import ingest as ingest_module  # noqa: E402
from businessflow.rag.ingest import ChunkRecord, UnreadableDocumentError  # noqa: E402
from businessflow.rag.retriever import DocumentRetriever, corpus_fingerprint  # noqa: E402


def _active_texts(path: str) -> list[str]:
    rows = get_connection().execute(
        "select document_text from document_chunks where source_document = %s and superseded_at is null order by chunk_index",
        (normalize_source_document(path),),
    ).fetchall()
    return [r["document_text"] for r in rows]


def test_a_failure_while_writing_the_new_version_leaves_the_previous_one_active(monkeypatch):
    path = _write_temp_doc("# Terms v1\n\nThe original waiver is exactly twelve percent of the outstanding balance.\n")
    try:
        ingest_document(path, document_type="loan_agreement")
        before = _active_texts(path)
        assert before and "twelve percent" in before[0]

        # A NUL byte cannot be stored in a Postgres text column: the insert fails INSIDE the transaction, after the
        # old generation was marked superseded. Everything must roll back, not leave the document retired.
        monkeypatch.setattr(ingest_module, "parse_and_chunk", lambda p: [ChunkRecord(text="corrupt \x00 chunk text of some length", headings="")])
        with pytest.raises(psycopg.Error):
            ingest_document(path, document_type="loan_agreement")

        assert _active_texts(path) == before
        assert len(get_chunk_ids_for_document(path)) == len(before), "a rolled-back ingest must not leave rows behind either"
    finally:
        delete_chunks_for_document(path)
        os.unlink(path)


def test_an_unreadable_replacement_is_refused_and_the_previous_version_stays_active(monkeypatch):
    path = _write_temp_doc("# Terms v1\n\nThe original waiver is exactly twelve percent of the outstanding balance.\n")
    try:
        ingest_document(path, document_type="loan_agreement")
        before = _active_texts(path)

        def unreadable(file_path):
            raise UnreadableDocumentError("no readable text could be extracted")

        monkeypatch.setattr(ingest_module, "parse_and_chunk", unreadable)
        with pytest.raises(UnreadableDocumentError):
            ingest_document(path, document_type="loan_agreement")

        assert _active_texts(path) == before
    finally:
        delete_chunks_for_document(path)
        os.unlink(path)


def test_concurrent_uploads_of_the_same_document_leave_exactly_one_active_version():
    path = _write_temp_doc("# Terms\n\nThe waiver clause applies once per loan and is recorded in the schedule.\n")
    errors = []

    def upload():
        try:
            ingest_document(path, document_type="loan_agreement")
        except Exception as e:  # noqa: BLE001 - the test reports whatever a racing upload raises
            errors.append(e)

    try:
        threads = [threading.Thread(target=upload) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)

        assert not errors, errors
        rows = get_chunk_ids_for_document(path)
        active = [r for r in rows if r["superseded_at"] is None]
        assert len(active) == 1, f"{len(active)} active chunks after 4 racing uploads of a one-chunk document"
        assert len(rows) == 4, "each upload writes its own generation; all but the last are superseded"
    finally:
        delete_chunks_for_document(path)
        os.unlink(path)


def test_the_corpus_fingerprint_changes_when_a_document_is_added_replaced_or_removed():
    path = _write_temp_doc("# Terms\n\nThe first wording of this clause is long enough to be a real chunk.\n")
    try:
        start = corpus_fingerprint()
        ingest_document(path, document_type="loan_agreement")
        added = corpus_fingerprint()
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Terms\n\nThe second wording of this clause is long enough to be a real chunk.\n")
        ingest_document(path, document_type="loan_agreement")
        replaced = corpus_fingerprint()
        assert len({start, added, replaced}) == 3
    finally:
        delete_chunks_for_document(path)
        os.unlink(path)
    assert corpus_fingerprint() != replaced


def test_a_retriever_built_before_an_upload_answers_from_its_snapshot_and_never_crashes():
    # The bug this design fixes: the embedding search ran live against the table while the text came from a snapshot,
    # so a chunk ingested after the snapshot was found by the search, was missing from the snapshot, and raised.
    path = _write_temp_doc("# Terms\n\nA special quarterly rebate of exactly seven point five percent applies to this loan.\n")
    try:
        snapshot = DocumentRetriever()
        ingest_document(path, document_type="loan_agreement", account_id="BF-1001")

        results = snapshot.retrieve("what is the quarterly rebate percentage", top_k=3, account_id="BF-1001")  # must not raise
        assert not any("seven point five" in r["text"] for r in results), "the old snapshot cannot know a later upload"

        assert snapshot.fingerprint != corpus_fingerprint(), "...which is exactly what the fingerprint check detects"
        fresh = DocumentRetriever().retrieve("what is the quarterly rebate percentage", top_k=3, account_id="BF-1001")
        assert any("seven point five" in r["text"] for r in fresh)
    finally:
        delete_chunks_for_document(path)
        os.unlink(path)
