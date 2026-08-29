"""The persistent vector store every ingested document lands in, shared by
ingestion and retrieval -- so what gets ingested (once, whenever a document
is uploaded) is exactly what's searchable later, in a different process,
across restarts.

Backed by pgvector on the same Supabase Postgres project as every other
table (see accounts/db.py) -- not a separate local ChromaDB file anymore.
That local file was a genuine gap: it's a per-machine artifact nothing in
the deploy process ever rebuilt, so a fresh VM deploy silently ran with an
empty index instead of failing loudly (found live: the deployed bot
answered a real policy question by fabricating a fake-looking policy ID
rather than citing real KB text, because retrieval had nothing to find).
Putting chunks in the same database as everything else means "did this
box's index get rebuilt" stops being a category of bug -- ingestion is now
a plain INSERT into a table that's backed up exactly like accounts/payments
already are.

This module intentionally does NOT mimic chromadb's Collection interface
(.get/.upsert/.query with its where-filter DSL) -- ingest.py and
retriever.py speak plain SQL against get_connection() directly instead.
Chroma's filter grammar took several rounds of "confirmed empirically" to
pin down (see the git history of retriever.py/ingest.py before this
change) for exactly the kind of thing a WHERE clause expresses directly.
"""

from datetime import datetime
from pathlib import Path

from businessflow.accounts.db import get_connection

# store.py lives at src/businessflow/rag/store.py -- three parents up is
# the repo root, the same offset ops/api.py's own _DOCUMENTS_DIR already
# uses from its own location one level shallower in the tree.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def embedding_literal(vector) -> str:
    """pgvector accepts its input type as a string literal ("[0.1,0.2,...]"
    cast via ::vector) -- this avoids pulling in the separate `pgvector`
    package just to register a psycopg type adapter on a connection pool
    that accounts/db.py already owns and configures."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def normalize_source_document(file_path: str) -> str:
    """Every source_document is stored (and matched against) as a path
    relative to the repo root, never the absolute path a caller happened
    to pass in.

    Found live: document_chunks is one shared table every machine that
    runs this app writes into (dev laptops, CI runners, the VM) -- an
    absolute path bakes in whichever machine ingested it, so the exact
    same logical file (say data/kb/grace_period.md) ingested from a
    Windows checkout and a Linux CI checkout was tracked as two entirely
    unrelated documents, and purge_orphaned_chunks (which checks
    resolve_source_document(...).exists() against the CALLING machine's
    own filesystem) deleted the other machine's perfectly real chunks
    just because that exact absolute path doesn't exist locally. A
    repo-root-relative path is identical across every checkout, so
    supersede-matching and the orphan check both become correct
    regardless of which machine ingested or later re-checks a document.

    Falls back to the absolute, resolved path when file_path isn't under
    the repo at all (e.g. a test's tempfile) -- there's no portable
    relative form for that case, and it only ever affects a single
    machine's own temporary data, never a real committed document.
    """
    resolved = Path(file_path).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_source_document(source_document: str) -> Path:
    """The inverse of normalize_source_document -- a relative value is
    resolved against the repo root (so it's checked the same way on every
    machine); an absolute one (the tempfile-outside-the-repo case) is used
    as-is."""
    p = Path(source_document)
    return p if p.is_absolute() else _REPO_ROOT / p


# Everything below is test/inspection support only -- ingest.py and
# retriever.py never call these, they speak SQL against get_connection()
# directly for their own real read/write paths. These exist so tests can
# inspect or manipulate raw chunk state (a superseded_at timestamp, whether
# a chunk still exists) without every test file hand-rolling the same SQL.
# Each takes the SAME raw path a test passed to ingest_document and
# normalizes it the same way ingest_document itself does, so a test never
# needs to know or care that storage keys on the normalized form.


def get_chunk_ids_for_document(source_document: str) -> list[dict]:
    """id + superseded_at for every chunk (active or superseded) currently
    stored for source_document."""
    return get_connection().execute(
        "select id, superseded_at from document_chunks where source_document = %s",
        (normalize_source_document(source_document),),
    ).fetchall()


def get_chunk_texts_for_document(source_document: str) -> list[dict]:
    """document_text + superseded_at for every chunk (active or superseded)
    currently stored for source_document -- test/inspection support for
    verifying supersede-in-place behavior directly, bypassing
    DocumentRetriever's own active-only filtering."""
    return get_connection().execute(
        "select document_text, superseded_at from document_chunks where source_document = %s",
        (normalize_source_document(source_document),),
    ).fetchall()


def backdate_chunks(ids: list[str], superseded_at: datetime) -> None:
    """Simulate a chunk having become superseded at an arbitrary point in
    the past, without waiting for it -- e.g. to test purge_superseded_
    chunks' cutoff without a real 60-day wait."""
    get_connection().execute(
        "update document_chunks set superseded_at = %s where id = any(%s)",
        (superseded_at, ids),
    )


def delete_chunks_for_document(source_document: str) -> None:
    """Removes every chunk (active or superseded) for source_document --
    test cleanup, the equivalent of the old collection.delete(where=...)."""
    get_connection().execute(
        "delete from document_chunks where source_document = %s",
        (normalize_source_document(source_document),),
    )
