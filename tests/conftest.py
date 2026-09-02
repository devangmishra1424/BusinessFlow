"""Shared pytest fixtures."""

import contextlib
import io
import os

# MUST run before any other project import in this file, and before pytest
# imports any test module -- conftest.py loads first, and every test
# file's own DATABASE_URL skip check (pytest.mark.skipif(not
# os.environ.get("DATABASE_URL"), ...)) is evaluated at module-import time
# during collection, right after this. A real incident: .env's
# DATABASE_URL points at the SAME production Supabase database the
# deployed app reads from -- there's no separate local/test database --
# so every local pytest run was silently writing real test data (and,
# via reseed_accounts, resetting demo accounts a human had deliberately
# removed) straight into production. This clears DATABASE_URL from the
# environment entirely whenever it looks like Supabase, so every
# DATABASE_URL-gated test across the whole suite skips cleanly instead --
# not just the ones using reseed_accounts below. CI is unaffected: its
# DATABASE_URL points at its own disposable pgvector container, never
# Supabase, so this never fires there.
if "supabase" in os.environ.get("DATABASE_URL", "").lower():
    os.environ.pop("DATABASE_URL")

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
