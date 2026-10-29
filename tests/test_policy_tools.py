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
    searches: list = []  # (query, keyword arguments) of every retrieve() call

    def __init__(self):
        if _FakeRetriever.fail_next_build:
            raise ValueError("a malformed row")
        _FakeRetriever.instances_built += 1
        self.fingerprint = _FakeRetriever.fingerprint_on_build
        self._ids = ["a", "b"]

    def retrieve(self, query, **kwargs):
        _FakeRetriever.searches.append((query, kwargs))
        return []


@pytest.fixture
def cache(monkeypatch):
    """A known, empty cache; the fake retriever; a controllable fingerprint query (`state["fingerprint"]`, or
    `state["error"]` to make it raise) and a controllable clock. monkeypatch undoes all of it after the test."""
    _FakeRetriever.instances_built = 0
    _FakeRetriever.fingerprint_on_build = ("v1",)
    _FakeRetriever.fail_next_build = False
    _FakeRetriever.searches = []
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


# --- warm_up: the first policy question after a restart should not pay for loading the search models ---------------

def test_warm_up_builds_the_snapshot_and_runs_one_search_so_the_first_question_does_not(cache, caplog):
    with caplog.at_level(logging.INFO, logger="businessflow.tools.policy_tools"):
        policy_tools.warm_up()

    assert _FakeRetriever.instances_built == 1
    assert [query for query, _ in _FakeRetriever.searches] == [policy_tools._WARM_UP_QUERY]
    assert any("policy search warmed up in" in r.getMessage() and "2 active chunks" in r.getMessage() for r in caplog.records)
    # the question that comes next uses that same snapshot and builds nothing
    assert policy_tools._retriever() is policy_tools._cached_retriever
    assert _FakeRetriever.instances_built == 1


@pytest.mark.parametrize("where, error", [
    ("build", psycopg.OperationalError("connection refused")),
    ("search", OSError("cannot reach huggingface.co")),
    ("search", RuntimeError("onnxruntime could not create a session")),
])
def test_a_warm_up_that_cannot_reach_the_database_or_the_models_says_what_to_check_and_does_not_raise(
    cache, monkeypatch, caplog, where, error,
):
    class Failing(_FakeRetriever):
        def __init__(self):
            if where == "build":
                raise error
            super().__init__()

        def retrieve(self, query, **kwargs):
            raise error

    monkeypatch.setattr(policy_tools, "DocumentRetriever", Failing)

    with caplog.at_level(logging.INFO, logger="businessflow.tools.policy_tools"):
        policy_tools.warm_up()  # must return: the service starts either way

    record = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "warm-up failed" in record.getMessage() and "DATABASE_URL" in record.getMessage()
    assert record.exc_info and record.exc_info[1] is error
    assert not any("warmed up in" in r.getMessage() for r in caplog.records)  # never reported as done


def test_after_a_failed_warm_up_the_first_question_loads_the_snapshot_as_it_always_did(cache, monkeypatch):
    class Down(_FakeRetriever):
        def __init__(self):
            raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(policy_tools, "DocumentRetriever", Down)
    policy_tools.warm_up()
    assert policy_tools._cached_retriever is None  # nothing half-built is left behind

    monkeypatch.setattr(policy_tools, "DocumentRetriever", _FakeRetriever)  # the database is back
    assert policy_tools._retriever() is not None
    assert _FakeRetriever.instances_built == 1


def test_a_warm_up_hit_by_an_error_nobody_expected_does_not_hide_it(cache, monkeypatch):
    class Broken(_FakeRetriever):
        def retrieve(self, query, **kwargs):
            raise ValueError("a malformed row")

    monkeypatch.setattr(policy_tools, "DocumentRetriever", Broken)
    with pytest.raises(ValueError, match="a malformed row"):
        policy_tools.warm_up()


def test_warm_up_in_the_background_returns_at_once_and_does_not_hold_the_service_up(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow_warm_up():
        started.set()
        release.wait(timeout=10)

    monkeypatch.setattr(policy_tools, "warm_up", slow_warm_up)

    thread = policy_tools.warm_up_in_background()  # comes back while the warm-up is still running
    try:
        assert started.wait(timeout=5)
        assert thread.is_alive() and thread.daemon and thread.name == "policy-warm-up"
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


@_pg_skip
def test_warm_up_loads_the_real_snapshot_and_both_models(monkeypatch):
    """The point of it, with the real thing: afterwards the first question has nothing left to load. Slow (it loads the
    two models again, about 25 s on a laptop). Assumes scripts/seed_kb.py has been run, like the tests above."""
    from businessflow.rag import embeddings, retriever

    monkeypatch.setattr(policy_tools, "_cached_retriever", None)
    embeddings._model.cache_clear()
    retriever._load_reranker.cache_clear()

    policy_tools.warm_up()

    assert policy_tools._cached_retriever is not None and policy_tools._cached_retriever._ids
    assert embeddings._model.cache_info().currsize == 1
    assert retriever._load_reranker.cache_info().currsize == 1
