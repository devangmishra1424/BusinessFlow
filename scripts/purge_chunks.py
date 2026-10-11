"""Retention for the document store: removes old superseded chunks and chunks whose source file is gone.

Re-uploading a document never deletes what it replaced (the old wording is kept, marked superseded, as a compliance
trail), so without this the table only ever grows. Nothing runs it automatically -- deleting history is a decision, so
it is a command someone chooses to run, and it does nothing unless told to.

  superseded  chunks superseded more than --older-than-days ago (default 30)
  orphaned    chunks whose source_document no longer exists on this machine's disk. CAUTION: run this on the machine
              that holds data/documents (the VM), not a laptop whose copy is partial -- on any other machine every
              uploaded document looks "gone".

Run: python -m scripts.purge_chunks                                  # dry run: counts only
     python -m scripts.purge_chunks --apply                          # delete old superseded chunks
     python -m scripts.purge_chunks --apply --orphans                # ...and orphaned ones
     python -m scripts.purge_chunks --older-than-days 90 --apply
"""

import argparse
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from businessflow.accounts.db import get_connection  # noqa: E402
from businessflow.rag.ingest import purge_orphaned_chunks, purge_superseded_chunks  # noqa: E402
from businessflow.rag.store import resolve_source_document  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--older-than-days", type=int, default=30)
    parser.add_argument("--orphans", action="store_true", help="also remove chunks whose source file is gone from this disk")
    parser.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    args = parser.parse_args()

    conn = get_connection()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than_days)
    n_old = conn.execute(
        "select count(*) as n from document_chunks where superseded_at is not null and superseded_at < %s", (cutoff,)
    ).fetchone()["n"]
    n_active, n_superseded = (
        conn.execute("select count(*) filter (where superseded_at is null) a, count(*) filter (where superseded_at is not null) s from document_chunks").fetchone()
    ).values()
    print(f"{n_active} active chunks, {n_superseded} superseded ({n_old} superseded more than {args.older_than_days} days ago)")

    orphaned = 0
    if args.orphans:
        sources = [r["source_document"] for r in conn.execute("select distinct source_document from document_chunks").fetchall()]
        missing = [s for s in sources if not resolve_source_document(s).exists()]
        orphaned = conn.execute("select count(*) as n from document_chunks where source_document = any(%s)", (missing,)).fetchone()["n"]
        print(f"{orphaned} chunks across {len(missing)} documents whose file is not on this disk")

    if not args.apply:
        print("dry run -- nothing deleted. Re-run with --apply to delete.")
        return
    print(f"deleted {purge_superseded_chunks(args.older_than_days)} old superseded chunks")
    if args.orphans:
        print(f"deleted {purge_orphaned_chunks()} orphaned chunks")


if __name__ == "__main__":
    main()
