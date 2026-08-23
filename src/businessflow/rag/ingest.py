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

from businessflow.rag.embeddings import embed_passages
from businessflow.rag.store import get_collection


def ingest_document(file_path: str, document_type: str, account_id: str | None = None) -> int:
    """Parses and chunks file_path, embeds each chunk, and stores it.

    document_type is e.g. "policy", "loan_agreement", "regulatory".
    account_id scopes a document to one specific borrower (their own signed
    loan agreement, say) -- leave it None for documents that apply to
    every borrower, like general policy.

    Re-ingesting the same file_path is safe to run twice: any existing
    (not-already-superseded) chunks for that file are marked superseded --
    via _supersede_existing_chunks, which sets a real superseded_at ISO
    timestamp on them in place -- rather than deleted, so a correction to a
    policy/loan document doesn't erase the record of what it said before.
    DocumentRetriever only ever considers chunks without a real
    superseded_at value, so this is invisible to retrieval; it's only
    reachable via a direct collection.get(where={"source_document": ...}).

    Returns the number of NEW chunks stored (matches the old contract:
    superseded chunks from previous generations aren't counted).
    """
    doc = DocumentConverter().convert(file_path).document
    chunks = list(HybridChunker().chunk(doc))
    collection = get_collection()

    # One timestamp for this whole call: it's both the superseded_at value
    # stamped on the old generation below, and (see the id comment in the
    # loop) folded into the new generation's ids.
    now = datetime.now(timezone.utc).isoformat()
    _supersede_existing_chunks(collection, file_path, now)

    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_passages(texts)

    ids = []
    metadatas = []
    for i, chunk in enumerate(chunks):
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
        chunk_id = hashlib.sha1(f"{file_path}::{i}::{now}".encode()).hexdigest()
        ids.append(chunk_id)
        metadatas.append({
            "source_document": file_path,
            "document_type": document_type,
            "account_id": account_id or "general",
            "chunk_index": i,
            "headings": " > ".join(chunk.meta.headings or []),
            # superseded_at is deliberately omitted here: absent means
            # active (see DocumentRetriever). A later re-ingestion of this
            # same file_path is what adds it, via _supersede_existing_
            # chunks, once this generation is itself superseded.
        })

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def _supersede_existing_chunks(collection, file_path: str, superseded_at: str) -> None:
    """Marks every currently-active chunk stored for file_path as
    superseded, in place, ahead of ingest_document() upserting a new
    generation.

    Chroma has no "patch just this one metadata key on an existing row"
    operation -- upsert is the only in-place-update primitive, and it
    requires the full record (embedding + document text), which is why
    those are fetched back out via collection.get() rather than
    reconstructed. Re-upserting at the SAME ids these chunks already have
    updates them in place instead of creating duplicates.

    Only touches chunks that are currently active (no superseded_at, or
    a prior ingest never having set one) -- a chunk from an even older
    generation that's already superseded is left completely alone, so
    its original superseded_at timestamp (when it actually stopped being
    current) is never overwritten by a later, unrelated re-ingestion.

    Does nothing if file_path has never been ingested before, or every
    chunk for it is already superseded -- nothing active to mark.
    """
    existing = collection.get(
        where={"source_document": file_path}, include=["documents", "metadatas", "embeddings"]
    )
    active = [
        (chunk_id, document, metadata, embedding)
        for chunk_id, document, metadata, embedding in zip(
            existing["ids"], existing["documents"], existing["metadatas"], existing["embeddings"]
        )
        if not metadata.get("superseded_at")
    ]
    if not active:
        return

    ids, documents, metadatas, embeddings = (list(field) for field in zip(*active))
    superseded_metadatas = [{**metadata, "superseded_at": superseded_at} for metadata in metadatas]
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=superseded_metadatas)


def purge_superseded_chunks(older_than_days: int = 30) -> int:
    """The real cleanup _supersede_existing_chunks defers: that function
    marks a chunk superseded_at instead of deleting it, so a correction
    never erases the record of what a document said before -- but nothing
    ever actually reclaimed that space, and real usage shows it adds up
    fast (one KB doc alone accumulated 76 superseded chunks from repeated
    re-ingestion during development). A chunk superseded this long ago has
    had every reasonable chance to matter for a compliance/history look-
    back; keeping it forever is unbounded growth, not a retention policy.

    Filtered in Python, not via a Chroma `where` clause -- retriever.py's
    own comments already found Chroma's filter grammar unreliable for
    anything beyond exact-match/$in/$nin on a known value set, and a
    "$lt this ISO timestamp" comparison isn't one of those. Returns the
    number of chunks actually deleted."""
    collection = get_collection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    corpus = collection.get(include=["metadatas"])
    stale_ids = [
        chunk_id
        for chunk_id, metadata in zip(corpus["ids"], corpus["metadatas"])
        if metadata.get("superseded_at") and metadata["superseded_at"] < cutoff
    ]
    if not stale_ids:
        return 0
    collection.delete(ids=stale_ids)
    return len(stale_ids)


def purge_orphaned_chunks() -> int:
    """Deletes chunks whose source_document no longer exists on disk --
    found live via a real, active (not superseded) chunk from a test
    upload-size-cap probe file, still retrievable for that account months
    after the probe file itself was deleted. Deliberately narrow: a chunk
    is only orphaned when its file is verifiably gone, never based on
    content or age, so this can never delete a real, currently-valid
    document's chunks. Returns the number of chunks actually deleted."""
    collection = get_collection()
    corpus = collection.get(include=["metadatas"])
    orphaned_ids = [
        chunk_id
        for chunk_id, metadata in zip(corpus["ids"], corpus["metadatas"])
        if metadata.get("source_document") and not Path(metadata["source_document"]).exists()
    ]
    if not orphaned_ids:
        return 0
    collection.delete(ids=orphaned_ids)
    return len(orphaned_ids)


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
