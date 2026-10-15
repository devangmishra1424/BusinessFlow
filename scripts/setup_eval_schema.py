"""Builds, or refreshes, the evals' own copy of the tables: schema DATABASE_SCHEMA (default "eval") in the same
database, so the nightly check (scripts/run_eval_monitor.py) can reseed demo accounts and log test conversations
without touching the live ones. Found live: run against the live tables at 21:00 UTC, every suite's reseed wiped what
people had done on the demo that day -- a payment, chats, requests.

Each step is safe to re-run:
1. Creates the schema and, when it has no tables yet, every table and sequence in schema.sql inside it, in one
   transaction. A schema with only some of the live tables is refused, not patched: drop it and re-run.
2. Mirrors the live index's ACTIVE document chunks into it (the policy KB and each demo account's loan agreement), same
   ids and embeddings, so nothing is re-embedded; chunks no longer active in the live index are marked superseded in
   the copy too. Re-run after a KB doc or loan agreement changes.
3. Seeds the demo accounts into it (scripts/seed_accounts.py, which follows DATABASE_SCHEMA) and copies their interest
   rates from the live accounts (set by scripts/seed_loan_agreements.py, which this does not re-run).

Reads the live tables in steps 2 and 3; never writes them.

Run: python -m scripts.setup_eval_schema
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import tuple_row

from businessflow.accounts.db import database_schema
from scripts import seed_accounts

load_dotenv()

_SCHEMA_SQL = Path(__file__).resolve().parents[1] / "schema.sql"
# Named, not "select *": the live table's column order is whatever its history made it, and need not match schema.sql's.
_CHUNK_COLUMNS = (
    "id", "source_document", "document_type", "account_id", "chunk_index", "headings", "document_text", "embedding",
    "superseded_at", "created_at",
)


def _tables(conn, schema: str) -> set[str]:
    rows = conn.execute(
        "select table_name from information_schema.tables where table_schema = %s and table_type = 'BASE TABLE'",
        (schema,),
    ).fetchall()
    return {r[0] for r in rows}


def main() -> dict:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set -- copy .env.example to .env and fill it in")
    schema = database_schema()
    if schema is None:
        raise SystemExit("DATABASE_SCHEMA must name the evals' copy (default 'eval'), never the live 'public' tables")
    target = sql.Identifier(schema)

    with psycopg.connect(database_url, connect_timeout=15, row_factory=tuple_row) as conn:
        live_tables = _tables(conn, "public")
        created = False
        if not _tables(conn, schema):
            with conn.transaction():
                conn.execute(sql.SQL("create schema if not exists {}").format(target))
                # public stays on the path for the pgvector type; every table schema.sql creates lands in the copy.
                conn.execute(sql.SQL("set local search_path to {}, public").format(target))
                conn.execute(_SCHEMA_SQL.read_text(encoding="utf-8"))
            created = True
        missing = live_tables - _tables(conn, schema)
        if missing:
            raise SystemExit(
                f"schema {schema!r} has some tables but not {sorted(missing)} -- drop it (drop schema {schema} cascade) "
                "and run this again"
            )

        columns = sql.SQL(", ").join(sql.Identifier(c) for c in _CHUNK_COLUMNS)
        with conn.transaction():
            copied = conn.execute(
                sql.SQL(
                    "insert into {}.document_chunks ({cols}) select {cols} from public.document_chunks "
                    "where superseded_at is null on conflict (id) do nothing"
                ).format(target, cols=columns)
            ).rowcount
            retired = conn.execute(
                sql.SQL(
                    "update {}.document_chunks set superseded_at = now() where superseded_at is null "
                    "and id not in (select id from public.document_chunks where superseded_at is null)"
                ).format(target)
            ).rowcount

    # seed_accounts deletes and rewrites the demo accounts in whatever DATABASE_SCHEMA names -- checked again right
    # before it runs, because on the live tables that is exactly the wipe this script exists to prevent.
    if database_schema() != schema:
        raise SystemExit("DATABASE_SCHEMA changed while setting up the copy -- not seeding")
    seed_accounts.main()  # follows DATABASE_SCHEMA: the copy, never the live accounts

    with psycopg.connect(database_url, connect_timeout=15) as conn:
        rates = conn.execute(
            sql.SQL(
                "update {}.accounts e set interest_rate_pct = p.interest_rate_pct from public.accounts p "
                "where p.account_id = e.account_id and p.interest_rate_pct is not null"
            ).format(target)
        ).rowcount

    summary = {"schema": schema, "created": created, "chunks_copied": copied, "chunks_retired": retired, "interest_rates": rates}
    print(summary)
    return summary


if __name__ == "__main__":
    # Only the command line defaults the schema: importing this module must not switch anything else to the copy.
    os.environ.setdefault("DATABASE_SCHEMA", "eval")
    main()
