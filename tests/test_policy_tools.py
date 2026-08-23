"""Thin test for the check_policy MCP tool wrapper -- retrieval quality
itself is exhaustively covered by test_retriever.py; this just confirms
the tool correctly shapes DocumentRetriever's output as {query, results}.
Assumes scripts/seed_kb.py has already been run against the persistent
store this test session points at (same assumption as test_retriever.py).
"""

import time

import businessflow.tools.policy_tools as policy_tools
from businessflow.tools.policy_tools import check_policy


def test_check_policy_returns_query_and_grounded_results():
    result = check_policy(query="can I get a few more days to pay")

    assert result["query"] == "can I get a few more days to pay"
    assert len(result["results"]) > 0
    assert all("text" in r and "source_document" in r for r in result["results"])


def test_check_policy_with_no_matching_account_scoped_docs_still_returns_general_results():
    # account_id given but this borrower has no documents of their own
    # uploaded -- general policy docs must still come back, not an empty
    # result silently masking a real "found nothing" case.
    result = check_policy(query="what is the grace period", account_id="BF-1001")

    assert len(result["results"]) > 0


class _FakeRetriever:
    """Stand-in for DocumentRetriever that skips the real embedding/rerank
    model loading -- these tests are about _retriever()'s own caching
    logic (reuse vs. rebuild), not retrieval quality, so the real,
    slow-to-construct class would only add noise here."""

    instances_built = 0

    def __init__(self):
        _FakeRetriever.instances_built += 1


def _reset_retriever_cache(monkeypatch):
    """Puts businessflow.tools.policy_tools's module-level cache into a
    known, empty state, and points its DocumentRetriever reference at
    _FakeRetriever so a rebuild is cheap and countable. monkeypatch undoes
    all of this at the end of the test regardless of outcome."""
    _FakeRetriever.instances_built = 0
    monkeypatch.setattr(policy_tools, "DocumentRetriever", _FakeRetriever)
    monkeypatch.setattr(policy_tools, "_cached_retriever", None)
    monkeypatch.setattr(policy_tools, "_last_built_at", 0.0)


def test_retriever_reuses_the_cached_instance_within_the_refresh_interval(monkeypatch):
    _reset_retriever_cache(monkeypatch)

    first = policy_tools._retriever()
    second = policy_tools._retriever()

    assert first is second
    assert _FakeRetriever.instances_built == 1


def test_retriever_rebuilds_once_the_refresh_interval_has_elapsed(monkeypatch):
    _reset_retriever_cache(monkeypatch)

    first = policy_tools._retriever()
    assert _FakeRetriever.instances_built == 1

    # Simulate 10+ minutes having passed since the last build, without
    # actually sleeping: push the module's stored last-built timestamp far
    # enough into the past that _retriever()'s real time.time() call on
    # the next line reads as more than _REFRESH_INTERVAL_SECONDS after it.
    monkeypatch.setattr(
        policy_tools,
        "_last_built_at",
        time.time() - policy_tools._REFRESH_INTERVAL_SECONDS - 1,
    )

    second = policy_tools._retriever()

    assert second is not first
    assert _FakeRetriever.instances_built == 2
