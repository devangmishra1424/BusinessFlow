"""Document ingestion: parse a real file (PDF, markdown, docx -- whatever
Docling supports) into structure-aware chunks, embed them, and add them to
the persistent vector store. Each chunk keeps its heading path as metadata
(e.g. "Restructuring options > One-time settlement"), not just raw text --
that's what "contextual chunking" concretely means here.
"""

import hashlib

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

    Re-ingesting the same file_path is safe to run twice: existing chunks
    for that file are deleted first, so edits to a document don't leave
    stale orphaned chunks behind from a previous, longer version of it.

    Returns the number of chunks stored.
    """
    doc = DocumentConverter().convert(file_path).document
    chunks = list(HybridChunker().chunk(doc))
    collection = get_collection()

    collection.delete(where={"source_document": file_path})
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_passages(texts)

    ids = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        # Derived from file_path + index, so re-ingestion overwrites the
        # same ids rather than accumulating duplicates.
        chunk_id = hashlib.sha1(f"{file_path}::{i}".encode()).hexdigest()
        ids.append(chunk_id)
        metadatas.append({
            "source_document": file_path,
            "document_type": document_type,
            "account_id": account_id or "general",
            "chunk_index": i,
            "headings": " > ".join(chunk.meta.headings or []),
        })

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)
