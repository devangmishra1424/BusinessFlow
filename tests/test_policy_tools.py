"""Thin test for the check_policy MCP tool wrapper -- retrieval quality
itself is exhaustively covered by test_retriever.py; this just confirms
the tool correctly shapes DocumentRetriever's output as {query, results}.
Assumes scripts/seed_kb.py has already been run against the persistent
store this test session points at (same assumption as test_retriever.py).

The second half tests the snapshot cache (_retriever) with a fake retriever and a fake fingerprint, so it needs no
database and no models: when the snapshot is reused, when it is rebuilt, and what happens when the database or the
rebuild fails.
"""

import logging
import os
import threading

import psycopg
import pytest

import businessflow.tools.policy_tools as policy_tools
from businessflow.tools.policy_tools import check_policy

_pg_skip = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres/pgvector via check_policy",
)


@_pg_skip
def test_check_policy_returns_query_and_grounded_results():
    result = check_policy(query="can I get a few more days to pay")

    assert result["query"] == "can I get a few more days to pay"
    assert len(result["results"]) > 0
    assert all("text" in r and "source_document" in r for r in result["results"])


@_pg_skip
def test_check_policy_with_no_matching_account_scoped_docs_still_returns_general_results():
    # account_id given but this borrower has no documents of their own
    # uploaded -- general policy docs must still come back, not an empty
    # result silently masking a real "found nothing" case.
    result = check_policy(query="what is the grace period", account_id="BF-1001")

    assert len(result["results"]) > 0


class _FakeRetriever:
    """Stand-in for DocumentRetriever that skips the real embedding/rerank model loading and the database -- these
    tests are about _retriever()'s own caching logic, not retrieval quality. `fingerprint_on_build` is what the fake
    database would report at the moment of the build."""

    instances_built = 0
    fingerprint_on_build: tuple = ("v1",)
    fail_next_build = False

    def __init__(self):
        if _FakeRetriever.fail_next_build:
            raise ValueError("a malformed row")
        _FakeRetriever.instances_built += 1
        self.fingerprint = _FakeRetriever.fingerprint_on_build
        self._ids = ["a", "b"]


@pytest.fixture
def cache(monkeypatch):
    """A known, empty cache; the fake retriever; a controllable fingerprint query (`state["fingerprint"]`, or
    `state["error"]` to make it raise) and a controllable clock. monkeypatch undoes all of it after the test."""
    _FakeRetriever.instances_built = 0
    _FakeRetriever.fingerprint_on_build = ("v1",)
    _FakeRetriever.fail_next_build = False
    state = {"fingerprint": ("v1",), "error": None, "now": 1000.0, "fingerprint_queries": 0}

    def fingerprint():
        state["fingerprint_queries"] += 1
        if state["error"]:
            raise state["error"]
        return state["fingerprint"]

    monkeypatch.setattr(policy_tools, "DocumentRetriever", _FakeRetriever)
    monkeypatch.setattr(policy_tools, "corpus_fingerprint", fingerprint)
    monkeypatch.setattr(policy_tools, "_cached_retriever", None)
    monkeypatch.setattr(policy_tools, "_last_checked_at", 0.0)
    monkeypatch.setattr(policy_tools.time, "monotonic", lambda: state["now"])
    return state


def _later(cache):
    cache["now"] += policy_tools._FINGERPRINT_CHECK_INTERVAL_SECONDS + 1


def test_the_snapshot_is_reused_without_touching_the_database_inside_the_interval(cache):
    first = policy_tools._retriever()
    cache["now"] += policy_tools._FINGERPRINT_CHECK_INTERVAL_SECONDS - 1
    second = policy_tools._retriever()

    assert first is second
    assert _FakeRetriever.instances_built == 1
    assert cache["fingerprint_queries"] == 0


def test_an_unchanged_corpus_is_checked_but_not_rebuilt(cache):
    first = policy_tools._retriever()
    _later(cache)

    assert policy_tools._retriever() is first
    assert cache["fingerprint_queries"] == 1
    assert _FakeRetriever.instances_built == 1


def test_a_changed_corpus_is_picked_up_within_one_interval(cache):
    first = policy_tools._retriever()
    cache["fingerprint"] = ("v2",)
    _FakeRetriever.fingerprint_on_build = ("v2",)
    _later(cache)

    second = policy_tools._retriever()

    assert second is not first and second.fingerprint == ("v2",)
    assert _FakeRetriever.instances_built == 2
    # and it settles: the next interval finds nothing new, so no third build
    _later(cache)
    assert policy_tools._retriever() is second
    assert _FakeRetriever.instances_built == 2


def test_when_the_database_is_down_the_old_snapshot_keeps_answering_and_the_failure_is_logged(cache, caplog):
    first = policy_tools._retriever()
    cache["error"] = psycopg.OperationalError("server closed the connection")
    _later(cache)

    with caplog.at_level(logging.WARNING, logger="businessflow.tools.policy_tools"):
        assert policy_tools._retriever() is first
    assert any("could not check whether the policy documents changed" in r.getMessage() for r in caplog.records)
    # one failed check per interval, not one per request
    assert policy_tools._retriever() is first
    assert cache["fingerprint_queries"] == 1


def test_a_rebuild_that_fails_leaves_the_working_snapshot_in_place_and_is_logged_loudly(cache, caplog):
    first = policy_tools._retriever()
    cache["fingerprint"] = ("v2",)
    _FakeRetriever.fail_next_build = True
    _later(cache)

    with caplog.at_level(logging.ERROR, logger="businessflow.tools.policy_tools"):
        assert policy_tools._retriever() is first
    record = next(r for r in caplog.records if r.levelno == logging.ERROR)
    assert "rebuilding the policy snapshot failed" in record.getMessage()
    assert record.exc_info and record.exc_info[0] is ValueError


def test_a_caller_never_waits_for_a_rebuild_someone_else_is_running(cache):
    first = policy_tools._retriever()
    cache["fingerprint"] = ("v2",)
    _later(cache)

    with policy_tools._refresh_lock:  # another thread is in the middle of its check/rebuild
        seen = []
        worker = threading.Thread(target=lambda: seen.append(policy_tools._retriever()))
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), "the caller blocked on a refresh that was already in progress"
    assert seen == [first]
    assert cache["fingerprint_queries"] == 0


def test_first_use_builds_exactly_one_snapshot_even_when_callers_arrive_together(cache):
    barrier = threading.Barrier(6)
    results = []

    def call():
        barrier.wait()
        results.append(policy_tools._retriever())

    threads = [threading.Thread(target=call) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert _FakeRetriever.instances_built == 1
    assert len({id(r) for r in results}) == 1
