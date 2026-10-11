"""Document ingestion: parse a real file (PDF, markdown, docx -- whatever
Docling supports) into structure-aware chunks, embed them, and add them to
the persistent vector store. Each chunk keeps its heading path as metadata
(e.g. "Restructuring options > One-time settlement").

Ingestion is all-or-nothing. Everything that can fail (parsing, the
readability check, embedding) runs BEFORE the database is touched, and the
database part is a single transaction: a document is never left retired
with no replacement, or half-inserted.
"""

import hashlib
import logging
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import ConversionResult, DocumentConverter

from businessflow.accounts.db import get_connection
from businessflow.rag.embeddings import embed_passages
from businessflow.rag.store import embedding_literal, normalize_source_document, resolve_source_document
from businessflow.rag.tokenize import with_headings

logger = logging.getLogger(__name__)


class UnreadableDocumentError(ValueError):
    """The file opened, but no usable text came out of it: a blank page, a scan the OCR could not read, a language the
    bundled OCR does not know. Ingesting what little (or what garbage) did come out would silently replace a good
    document with nothing, so the upload is refused instead."""


# Measured on rendered test scans (see tests/test_ingest.py and the README): the bundled OCR reads English scans
# at 0.93-0.99 confidence and fails on Devanagari at 0.50-0.57; a scan it finds nothing in yields no text at all.
_MIN_READABLE_CHARS = 20
_MIN_OCR_CONFIDENCE = 0.75
_PLACEHOLDER = re.compile(r"<!--.*?-->")  # Docling writes "<!-- image -->" for a picture it could not read text from


# Whether the heading path goes in front of the text that is embedded. Tried (eval/retrieval_benchmark.py, "headings
# everywhere"): it lifted Recall@5 on the dev questions but did not hold on the held-out ones or the loan-agreement
# facts, so by the benchmark's own decision rule it is not used. A change here means re-embedding every stored chunk.
EMBED_WITH_HEADINGS = False


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    headings: str  # " > "-joined heading path, "" when the chunk sits under no heading

    @property
    def embedding_text(self) -> str:
        return with_headings(self.headings, self.text)


# One converter and one chunker for the whole process. Building a DocumentConverter loads the layout, table and OCR
# models; doing that on every upload made the process grow by ~100-190 MB per upload (485 -> 1366 MB over three
# uploads, measured) and cost ~2 s each. Conversions are serialised: the models are not documented as thread-safe,
# and one at a time is also what keeps peak memory bounded on a small VM.
_parse_lock = threading.Lock()
_recent_conversions: "OrderedDict[tuple, ConversionResult]" = OrderedDict()
_RECENT_CONVERSIONS_KEPT = 2


@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    return DocumentConverter()


@lru_cache(maxsize=1)
def _chunker() -> HybridChunker:
    return HybridChunker()


def _convert(file_path: str) -> ConversionResult:
    """Docling's conversion of file_path. The last couple are remembered by (path, mtime, size), so the loan-term
    extraction that follows ingestion in the upload endpoint does not parse the same file a second time."""
    stat = Path(file_path).stat()
    key = (str(Path(file_path).resolve()), stat.st_mtime_ns, stat.st_size)
    with _parse_lock:
        result = _recent_conversions.get(key)
        if result is None:
            result = _converter().convert(file_path)
            _recent_conversions[key] = result
            while len(_recent_conversions) > _RECENT_CONVERSIONS_KEPT:
                _recent_conversions.popitem(last=False)
        else:
            _recent_conversions.move_to_end(key)
        return result


def _require_readable(result: ConversionResult, records: list[ChunkRecord], file_path: str) -> None:
    body = _PLACEHOLDER.sub(" ", " ".join(r.text for r in records))
    readable = sum(ch.isalnum() for ch in body)
    if readable < _MIN_READABLE_CHARS:
        raise UnreadableDocumentError(
            f"no readable text could be extracted from {Path(file_path).name} (found {readable} letters and digits) -- "
            f"is it blank, or a scan too poor for OCR to read?"
        )
    confidence = getattr(getattr(result, "confidence", None), "ocr_score", math.nan)
    if not math.isnan(confidence) and confidence < _MIN_OCR_CONFIDENCE:
        raise UnreadableDocumentError(
            f"{Path(file_path).name} could not be read reliably by OCR (confidence {confidence:.2f}, needs "
            f"{_MIN_OCR_CONFIDENCE}) -- the bundled OCR reads English scans, not Hindi/Devanagari ones. Upload the "
            f"text version of the document instead."
        )


def parse_and_chunk(file_path: str) -> list[ChunkRecord]:
    """The file as structure-aware chunks, or UnreadableDocumentError. Touches no database."""
    result = _convert(file_path)
    with _parse_lock:
        chunks = list(_chunker().chunk(result.document))
    records = [ChunkRecord(text=c.text, headings=" > ".join(c.meta.headings or [])) for c in chunks]
    _require_readable(result, records, file_path)
    return records


def embed_records(records: list[ChunkRecord]):
    return embed_passages([r.embedding_text if EMBED_WITH_HEADINGS else r.text for r in records])


def ingest_document(file_path: str, document_type: str, account_id: str | None = None) -> int:
    """Parses and chunks file_path, embeds each chunk, and stores it.

    document_type is e.g. "policy", "loan_agreement", "regulatory".
    account_id scopes a document to one specific borrower (their own signed
    loan agreement, say) -- leave it None for documents that apply to
    every borrower, like general policy.

    Re-ingesting the same file_path is safe to run twice: any existing
    (not-already-superseded) chunks for that file are marked superseded
    rather than deleted, so a correction to a policy/loan document doesn't
    erase the record of what it said before. DocumentRetriever only ever
    considers chunks without a superseded_at value, so this is invisible to
    retrieval; it's only reachable via a direct query by source_document.
    Superseding and inserting happen in ONE transaction, serialised per
    document by an advisory lock, so a failure leaves the previous version
    exactly as it was and two concurrent uploads of the same file cannot
    leave two active generations.

    Returns the number of NEW chunks stored. Raises UnreadableDocumentError
    (a ValueError) for a file with no usable text, before anything is
    written -- the previous version stays active.
    """
    records = parse_and_chunk(file_path)
    embeddings = embed_records(records)

    # Stored (and matched against) as a repo-root-relative path -- see
    # normalize_source_document's docstring for why an absolute one
    # would be wrong now that this store is shared across machines.
    source_document = normalize_source_document(file_path)
    now = datetime.now(timezone.utc)

    rows = [
        (
            # Folds `now` in, not just source_document + index: the previous generation is kept (marked
            # superseded, not deleted), so this generation's ids must not collide with it.
            hashlib.sha1(f"{source_document}::{i}::{now.isoformat()}".encode()).hexdigest(),
            source_document, document_type, account_id or "general", i, record.headings, record.text,
            embedding_literal(embedding),
        )
        for i, (record, embedding) in enumerate(zip(records, embeddings))
    ]
    with get_connection().transaction() as tx:
        tx.execute("select pg_advisory_xact_lock(hashtext(%s))", (source_document,))
        _supersede_existing_chunks(tx, source_document, now)
        with tx.cursor() as cursor:
            cursor.executemany(
                """
                insert into document_chunks
                    (id, source_document, document_type, account_id, chunk_index, headings, document_text, embedding)
                values (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                rows,
            )
    logger.info("ingested %s: %d chunks (document_type=%s, account_id=%s)", source_document, len(rows), document_type, account_id or "general")
    return len(rows)


def _supersede_existing_chunks(conn, source_document: str, superseded_at: datetime) -> None:
    """Marks every currently-active chunk stored for source_document (an
    already-normalized, repo-root-relative path -- see
    normalize_source_document) as superseded, in place, ahead of
    ingest_document() inserting a new generation.

    Only touches chunks that are currently active (superseded_at is null)
    -- a chunk from an even older generation that's already superseded is
    left completely alone, so its original superseded_at timestamp (when
    it actually stopped being current) is never overwritten by a later,
    unrelated re-ingestion.

    Does nothing if source_document has never been ingested before, or
    every chunk for it is already superseded -- nothing active to mark.
    """
    conn.execute(
        "update document_chunks set superseded_at = %s where source_document = %s and superseded_at is null",
        (superseded_at, source_document),
    )


def purge_superseded_chunks(older_than_days: int = 30) -> int:
    """The real cleanup _supersede_existing_chunks defers: that function
    marks a chunk superseded_at instead of deleting it, so a correction
    never erases the record of what a document said before -- but nothing
    reclaims that space until this runs (scripts/purge_chunks.py). A chunk
    superseded this long ago has had every reasonable chance to matter for
    a compliance/history look-back; keeping it forever is unbounded growth,
    not a retention policy. Returns the number of chunks actually deleted."""
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
    document's chunks. Returns the number of chunks actually deleted.

    Resolves each stored (repo-root-relative) source_document against the
    repo root before checking -- checking the raw relative string against
    whatever the process's own cwd happens to be would work by convention
    (every entry point here runs from the repo root) but resolving
    explicitly doesn't depend on that convention holding everywhere."""
    conn = get_connection()
    rows = conn.execute("select distinct source_document from document_chunks").fetchall()
    orphaned_documents = [
        r["source_document"] for r in rows if not resolve_source_document(r["source_document"]).exists()
    ]
    if not orphaned_documents:
        return 0
    deleted = conn.execute(
        "delete from document_chunks where source_document = any(%s) returning id",
        (orphaned_documents,),
    ).fetchall()
    return len(deleted)


def extract_document_text(file_path: str) -> str:
    """The WHOLE document as one markdown string -- for a caller that wants
    the full text (e.g. to hand to an LLM extraction pass, see
    rag/extraction.py's extract_loan_terms), not the chunked/embedded form
    ingest_document() stores. Uses the same conversion ingest_document()
    just made (see _convert), so an upload does not parse its file twice.

    export_to_markdown() rather than export_to_text(), since markdown keeps
    table structure -- relevant for a loan agreement's numeric terms, which
    sometimes appear in a table rather than a sentence.
    """
    return _convert(file_path).document.export_to_markdown()
