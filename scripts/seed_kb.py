"""One-time (well, re-runnable -- ingestion is idempotent) seeding: ingests
the hand-written policy KB docs into the persistent vector store through
the same ingestion pipeline real uploaded documents (PDFs, loan
agreements) will go through. Run whenever a KB doc's content changes.

Run: python scripts/seed_kb.py
"""

from pathlib import Path

from businessflow.rag.ingest import ingest_document

_KB_DIR = Path(__file__).resolve().parents[1] / "data" / "kb"


def main():
    for path in sorted(_KB_DIR.glob("*.md")):
        count = ingest_document(str(path), document_type="policy")
        print(f"{path.name}: {count} chunks")


if __name__ == "__main__":
    main()
