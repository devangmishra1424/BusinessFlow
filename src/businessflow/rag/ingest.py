"""Document ingestion: parse a real file (PDF, markdown, docx -- whatever
Docling supports) into structure-aware chunks, embed them, and add them to
the persistent vector store. Each chunk keeps its heading path as metadata
(e.g. "Restructuring options > One-time settlement"), not just raw text --
that's what "contextual chunking" concretely means here.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from businessflow.accounts.db import get_connection
from businessflow.rag.embeddings import embed_passages
from businessflow.rag.store import embedding_literal


def ingest_document(file_path: str, document_type: str, account_id: str | None = None) -> int:
    """Parses and chunks file_path, embeds each chunk, and stores it.

    document_type is e.g. "policy", "loan_agreement", "regulatory".
    account_id scopes a document to one specific borrower (their own signed
    loan agreement, say) -- leave it None for documents that apply to
    every borrower, like general policy.

    Re-ingesting the same file_path is safe to run twice: any existing
    (not-already-superseded) chunks for that file are marked superseded --
    via _supersede_existing_chunks, which sets a real superseded_at
    timestamp on them in place -- rather than deleted, so a correction to a
    policy/loan document doesn't erase the record of what it said before.
    DocumentRetriever only ever considers chunks without a real
    superseded_at value, so this is invisible to retrieval; it's only
    reachable via a direct query against document_chunks by source_document.

    Returns the number of NEW chunks stored (matches the old contract:
    superseded chunks from previous generations aren't counted).
    """
    doc = DocumentConverter().convert(file_path).document
    chunks = list(HybridChunker().chunk(doc))
    conn = get_connection()

    # One timestamp for this whole call: it's both the superseded_at value
    # stamped on the old generation below, and (see the id comment in the
    # loop) folded into the new generation's ids.
    now = datetime.now(timezone.utc)
    _supersede_existing_chunks(conn, file_path, now)

    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_passages(texts)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Folds `now` in, not just file_path + index: the old id scheme
        # (file_path + index alone) is what let re-ingestion overwrite the
        # same ids in place, which was correct back when re-ingestion
        # deleted-then-inserted. Now that the previous generation is kept
        # around (marked superseded, not deleted), reusing those same ids
        # here would upsert straight over the rows _supersede_existing_
        # chunks just wrote above -- silently erasing the superseded
        # marking this whole feature exists to preserve. Folding in `now`
        # guarantees this generation's ids can't collide with the one it's
        # superseding.
        chunk_id = hashlib.sha1(f"{file_path}::{i}::{now.isoformat()}".encode()).hexdigest()
        conn.execute(
            """
            insert into document_chunks
                (id, source_document, document_type, account_id, chunk_index, headings, document_text, embedding)
            values (%s, %s, %s, %s, %s, %s, %s, %s::vector)
            """,
            (
                chunk_id, file_path, document_type, account_id or "general", i,
                " > ".join(chunk.meta.headings or []), chunk.text, embedding_literal(embedding),
            ),
        )
    return len(chunks)


def _supersede_existing_chunks(conn, file_path: str, superseded_at: datetime) -> None:
    """Marks every currently-active chunk stored for file_path as
    superseded, in place, ahead of ingest_document() inserting a new
    generation.

    Only touches chunks that are currently active (superseded_at is null)
    -- a chunk from an even older generation that's already superseded is
    left completely alone, so its original superseded_at timestamp (when
    it actually stopped being current) is never overwritten by a later,
    unrelated re-ingestion.

    Does nothing if file_path has never been ingested before, or every
    chunk for it is already superseded -- nothing active to mark.
    """
    conn.execute(
        "update document_chunks set superseded_at = %s where source_document = %s and superseded_at is null",
        (superseded_at, file_path),
    )


def purge_superseded_chunks(older_than_days: int = 30) -> int:
    """The real cleanup _supersede_existing_chunks defers: that function
    marks a chunk superseded_at instead of deleting it, so a correction
    never erases the record of what a document said before -- but nothing
    ever actually reclaimed that space, and real usage shows it adds up
    fast (one KB doc alone accumulated 76 superseded chunks from repeated
    re-ingestion during development). A chunk superseded this long ago has
    had every reasonable chance to matter for a compliance/history look-
    back; keeping it forever is unbounded growth, not a retention policy.

    A plain `<` comparison against a real timestamptz column now, not a
    Python-side filter -- Chroma's filter grammar couldn't express this
    directly (retriever.py's own comments document why), Postgres always
    could. Returns the number of chunks actually deleted."""
    conn = get_connection()
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    rows = conn.execute(
        "delete from document_chunks where superseded_at is not null and superseded_at < %s returning id",
        (cutoff,),
    ).fetchall()
    return len(rows)


def purge_orphaned_chunks() -> int:
    """Deletes chunks whose source_document no longer exists on disk --
    found live via a real, active (not superseded) chunk from a test
    upload-size-cap probe file, still retrievable for that account months
    after the probe file itself was deleted. Deliberately narrow: a chunk
    is only orphaned when its file is verifiably gone, never based on
    content or age, so this can never delete a real, currently-valid
    document's chunks. Returns the number of chunks actually deleted."""
    conn = get_connection()
    rows = conn.execute("select distinct source_document from document_chunks").fetchall()
    orphaned_documents = [r["source_document"] for r in rows if not Path(r["source_document"]).exists()]
    if not orphaned_documents:
        return 0
    deleted = conn.execute(
        "delete from document_chunks where source_document = any(%s) returning id",
        (orphaned_documents,),
    ).fetchall()
    return len(deleted)


def extract_document_text(file_path: str) -> str:
    """Parses file_path with the same Docling converter ingest_document()
    uses, and returns the WHOLE document as one markdown string -- for a
    caller that wants the full text (e.g. to hand to an LLM extraction
    pass, see rag/extraction.py's extract_loan_terms), not the chunked/
    embedded form ingest_document() stores.

    Parses file_path independently of ingest_document() -- a caller that
    needs both (an account document upload wanting both RAG ingestion and
    structured-field extraction) parses the same file twice. Acceptable
    for a single-document, ops-driven upload, not a hot path -- and it
    keeps ingest_document's own signature, return type, and already-
    tested contract completely unchanged rather than bolting a second
    concern onto it.

    export_to_markdown() (verified against the installed docling version
    via DoclingDocument's real export_to_* methods) rather than
    export_to_text(), since markdown keeps table structure -- relevant
    for a loan agreement's numeric terms, which sometimes appear in a
    table rather than a sentence.
    """
    doc = DocumentConverter().convert(file_path).document
    return doc.export_to_markdown()
