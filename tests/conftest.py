"""Shared pytest fixtures."""

import contextlib
import io
import os

from dotenv import load_dotenv

# MUST run before any other project import in this file, and before pytest
# imports any test module -- conftest.py loads first, and every test
# file's own DATABASE_URL skip check (pytest.mark.skipif(not
# os.environ.get("DATABASE_URL"), ...)) is evaluated at module-import time
# during collection, right after this. A real incident: .env's
# DATABASE_URL points at the SAME production Supabase database the
# deployed app reads from -- there's no separate local/test database --
# so every local pytest run was silently writing real test data (and,
# via reseed_accounts, resetting demo accounts a human had deliberately
# removed) straight into production.
#
# Two things have to both be right here, not just one:
#
# 1. load_dotenv() is called explicitly, HERE, before checking anything --
#    without this, os.environ may not have DATABASE_URL populated from
#    .env yet at all (nothing has loaded it), so the "does this look like
#    Supabase" check below would silently see nothing and guard nothing,
#    purely depending on which module happened to import
#    accounts/db.py (which also calls load_dotenv()) first.
#
# 2. The Supabase URL is overwritten with an EMPTY STRING, never popped/
#    deleted. Found live, the hard way: popping the key made it look
#    "unset", and accounts/db.py's own load_dotenv() call -- triggered by
#    the very first thing that imports it, whenever that happens to be --
#    silently restored the real Supabase URL right back, since
#    load_dotenv() only ever fills in variables that are genuinely
#    missing, not ones already present (even at an empty string). A test
#    file gated only on GROQ_API_KEY (not DATABASE_URL), like
#    test_pipeline.py's real end-to-end round-trip tests, then went
#    ahead and hit the real database anyway, mid-session, invisibly.
#    An empty string is still a "set" key as far as load_dotenv() is
#    concerned, so it survives every later load_dotenv() call for the
#    rest of the process -- and reads as falsy everywhere else
#    (os.environ.get("DATABASE_URL") in every skip check, accounts/db.py's
#    own "if not database_url: raise RuntimeError(...)" guard), so every
#    DATABASE_URL-gated test skips cleanly and anything ungated fails
#    loudly and immediately instead of silently reaching Supabase.
#
# CI is unaffected either way: its DATABASE_URL points at its own
# disposable pgvector container, never Supabase, so this never fires there.
load_dotenv()
if "supabase" in os.environ.get("DATABASE_URL", "").lower():
    os.environ["DATABASE_URL"] = ""

import pytest

from scripts.seed_accounts import main as _reseed_demo_accounts


@pytest.fixture
def reseed_accounts():
    """Resets BF-1001..1004 (and their payment history / promises /
    disputes / escalations / events) to the canonical seed state before a
    test runs. Needed by any test that calls a tool which mutates real
    Postgres state -- otherwise one test's side effect (e.g. opening a
    dispute) leaks into a later test that assumes a clean account, the
    same bug that first showed up in eval/tool_calling_benchmark.py.

    Second layer of the same guard as the top of this file (which already
    clears a Supabase-pointed DATABASE_URL before collection even starts,
    so in practice this branch shouldn't be reachable) -- kept here too in
    case something re-sets DATABASE_URL mid-session, so this fixture can
    never silently reseed BF-1001..1004 back into production regardless."""
    database_url = os.environ.get("DATABASE_URL", "")
    if "supabase" in database_url.lower():
        pytest.skip(
            "reseed_accounts refuses to run against a Supabase DATABASE_URL -- "
            "this fixture writes real account data, and .env's DATABASE_URL points "
            "at the production database the live app uses, not a disposable test one. "
            "Run this against a real local/CI-only Postgres instead."
        )
    with contextlib.redirect_stdout(io.StringIO()):
        _reseed_demo_accounts()
