"""Shared pytest fixtures."""

import contextlib
import io

import pytest

from scripts.seed_accounts import main as _reseed_demo_accounts


@pytest.fixture
def reseed_accounts():
    """Resets BF-1001..1004 (and their payment history / promises /
    disputes / escalations / events) to the canonical seed state before a
    test runs. Needed by any test that calls a tool which mutates real
    Postgres state -- otherwise one test's side effect (e.g. opening a
    dispute) leaks into a later test that assumes a clean account, the
    same bug that first showed up in eval/tool_calling_benchmark.py."""
    with contextlib.redirect_stdout(io.StringIO()):
        _reseed_demo_accounts()
