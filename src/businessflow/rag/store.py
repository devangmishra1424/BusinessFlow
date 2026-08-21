"""The persistent vector store every ingested document lands in, shared by
ingestion and retrieval -- so what gets ingested (once, whenever a document
is uploaded) is exactly what's searchable later, in a different process,
across restarts. This replaced an earlier in-memory-only chromadb.Client()
that lost its index every time the process restarted.
"""

from pathlib import Path

import chromadb

_CHROMA_PATH = Path(__file__).resolve().parents[3] / "data" / "chroma_db"
_COLLECTION_NAME = "documents"


def get_collection():
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    return client.get_or_create_collection(_COLLECTION_NAME)
