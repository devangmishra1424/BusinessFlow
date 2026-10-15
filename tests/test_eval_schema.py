"""The evals' own copy of the tables (accounts/db.py's DATABASE_SCHEMA, scripts/setup_eval_schema.py).

Found live: the nightly eval check ran against the live tables at 21:00 UTC, and every suite's reseed wiped what people
had done on the demo that day -- a borrower's payment was gone the next morning. These pin down that the evals' writes
land in their copy, that an incomplete copy is refused instead of falling through to the live tables, and that the
live tables are left exactly as they were.
"""

import importlib
import os

import psycopg
import pytest

from businessflow.accounts import db

_pg_skip = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set -- these tests hit real Postgres")


# ------------------------------------------------------------------------------------------------- no database


@pytest.mark.parametrize("value, expected", [("", None), ("public", None), (" eval ", "eval"), ("eval_2", "eval_2")])
def test_database_schema_names_the_copy_or_nothing(monkeypatch, value, expected):
    monkeypatch.setenv("DATABASE_SCHEMA", value)
    assert db.database_schema() == expected


@pytest.mark.parametrize("value", ["Eval", "eval; drop schema public", "1eval", 'ev"al', "e" * 64])
def test_database_schema_refuses_anything_but_a_plain_name(monkeypatch, value):
    monkeypatch.setenv("DATABASE_SCHEMA", value)
    with pytest.raises(RuntimeError, match="not a plain lowercase schema name"):
        db.database_schema()


def test_the_evals_default_to_their_copy_against_the_production_database(monkeypatch):
    import eval as eval_package

    monkeypatch.setenv("DATABASE_URL", "postgresql://user@aws-0-ap-southeast-2.pooler.supabase.com:5432/postgres")
    monkeypatch.setenv("DATABASE_SCHEMA", "")  # unset as far as the check goes; restored after the test
    importlib.reload(eval_package)

    assert os.environ["DATABASE_SCHEMA"] == "eval"


def test_the_evals_leave_any_other_database_and_an_explicit_choice_alone(monkeypatch):
    import eval as eval_package

    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/businessflow_ci")
    monkeypatch.setenv("DATABASE_SCHEMA", "")
    importlib.reload(eval_package)
    assert os.environ["DATABASE_SCHEMA"] == ""

    monkeypatch.setenv("DATABASE_URL", "postgresql://user@x.pooler.supabase.com:5432/postgres")
    monkeypatch.setenv("DATABASE_SCHEMA", "eval_other")
    importlib.reload(eval_package)
    assert os.environ["DATABASE_SCHEMA"] == "eval_other"


# ------------------------------------------------------------------------------------------------ real Postgres


def _drop(schema: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        conn.execute(f'drop schema if exists "{schema}" cascade')


@_pg_skip
def test_an_incomplete_copy_is_refused_before_anything_can_reach_the_live_tables(monkeypatch):
    schema = "evaltest_incomplete"
    _drop(schema)
    try:
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
            conn.execute(f'create schema "{schema}"')
            conn.execute(f'create table "{schema}".accounts (account_id text primary key)')  # one table of many
        monkeypatch.setenv("DATABASE_SCHEMA", schema)

        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
            with pytest.raises(RuntimeError, match="has no copy of"):
                db.use_schema(conn)
            # The search_path was never switched.
            assert not conn.execute("show search_path").fetchone()[0].startswith(f'"{schema}"')
    finally:
        _drop(schema)


def _live_fingerprint(conn) -> tuple:
    """What a reseed of the live tables would change: the demo accounts' rows and the events log."""
    accounts = conn.execute("select account_id, updated_at from public.accounts order by account_id").fetchall()
    events = conn.execute("select count(*), max(id) from public.events").fetchone()
    payments = conn.execute("select count(*), max(id) from public.payment_history").fetchone()
    return tuple(accounts), tuple(events), tuple(payments)


@_pg_skip
def test_the_copy_is_built_and_reseeded_without_touching_the_live_tables(monkeypatch):
    from scripts import seed_accounts, setup_eval_schema

    schema = "evaltest_copy"
    _drop(schema)
    monkeypatch.setenv("DATABASE_SCHEMA", schema)
    try:
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
            before = _live_fingerprint(conn)
            live_chunks = conn.execute("select count(*) from public.document_chunks where superseded_at is null").fetchone()[0]

        first = setup_eval_schema.main()
        again = setup_eval_schema.main()  # safe to re-run: nothing is copied twice

        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
            assert _live_fingerprint(conn) == before
            copied_accounts = conn.execute(f'select count(*) from "{schema}".accounts').fetchone()[0]
            copied_chunks = conn.execute(
                f'select count(*) from "{schema}".document_chunks where superseded_at is null'
            ).fetchone()[0]
            db.use_schema(conn)  # a complete copy is accepted...
            assert conn.execute("show search_path").fetchone()[0].startswith(f'"{schema}"')
            # ...and from then on unqualified names are the copy's.
            assert conn.execute("select count(*) from accounts").fetchone()[0] == copied_accounts

        assert first["created"] is True and again["created"] is False
        assert copied_accounts == len(seed_accounts._ACCOUNT_IDS)
        assert copied_chunks == live_chunks and again["chunks_copied"] == 0
    finally:
        _drop(schema)


@_pg_skip
def test_setup_refuses_to_target_the_live_tables(monkeypatch):
    from scripts import setup_eval_schema

    monkeypatch.setenv("DATABASE_SCHEMA", "public")

    with pytest.raises(SystemExit):
        setup_eval_schema.main()
