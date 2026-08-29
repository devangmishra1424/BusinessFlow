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

from businessflow.accounts.db import get_connection


def embedding_literal(vector) -> str:
    """pgvector accepts its input type as a string literal ("[0.1,0.2,...]"
    cast via ::vector) -- this avoids pulling in the separate `pgvector`
    package just to register a psycopg type adapter on a connection pool
    that accounts/db.py already owns and configures."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


# Everything below is test/inspection support only -- ingest.py and
# retriever.py never call these, they speak SQL against get_connection()
# directly for their own real read/write paths. These exist so tests can
# inspect or manipulate raw chunk state (a superseded_at timestamp, whether
# a chunk still exists) without every test file hand-rolling the same SQL.


def get_chunk_ids_for_document(source_document: str) -> list[dict]:
    """id + superseded_at for every chunk (active or superseded) currently
    stored for source_document."""
    return get_connection().execute(
        "select id, superseded_at from document_chunks where source_document = %s",
        (source_document,),
    ).fetchall()


def get_chunk_texts_for_document(source_document: str) -> list[dict]:
    """document_text + superseded_at for every chunk (active or superseded)
    currently stored for source_document -- test/inspection support for
    verifying supersede-in-place behavior directly, bypassing
    DocumentRetriever's own active-only filtering."""
    return get_connection().execute(
        "select document_text, superseded_at from document_chunks where source_document = %s",
        (source_document,),
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
    get_connection().execute("delete from document_chunks where source_document = %s", (source_document,))
