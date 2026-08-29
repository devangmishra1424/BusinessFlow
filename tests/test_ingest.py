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

from businessflow.rag.ingest import ingest_document, purge_orphaned_chunks, purge_superseded_chunks
from businessflow.rag.store import backdate_chunks, delete_chunks_for_document, get_chunk_ids_for_document


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
