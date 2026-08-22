"""Unit tests for accounts/store.py's log_event -- specifically the
foreign-key fallback found live via eval/reasoning_accuracy.py: a
general (no verified account) conversation let the model pass a
hallucinated account_id to check_policy (which doesn't itself validate
one), and the generic "log every tool call" instrumentation in
agent/loop.py then crashed the entire turn on a ForeignKeyViolation --
a logging side-effect taking down a tool call that had already
succeeded. Real Postgres, no LLM.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

from businessflow.accounts import store
from businessflow.observability.metrics import event_counts_since

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_log_event_with_a_real_account_id_logs_normally(reseed_accounts):
    since = datetime.now(timezone.utc) - timedelta(seconds=2)
    store.log_event("BF-1001", "test_store_marker", {"n": 1})

    row = store.get_connection().execute(
        "select account_id from events where event_type = 'test_store_marker' and created_at >= %s order by id desc limit 1",
        (since,),
    ).fetchone()
    assert row["account_id"] == "BF-1001"


def test_log_event_with_a_nonexistent_account_id_does_not_raise(reseed_accounts):
    # This must not crash -- it's exactly what happened live when the
    # model passed a hallucinated account_id to a tool that doesn't
    # validate one, and the generic tool-call logger tried to log it.
    store.log_event("BF-9999", "test_store_marker", {"n": 1})


def test_log_event_with_a_nonexistent_account_id_falls_back_to_no_specific_borrower(reseed_accounts):
    since = datetime.now(timezone.utc) - timedelta(seconds=2)
    store.log_event("BF-9999", "test_store_marker", {"n": 1})

    row = store.get_connection().execute(
        "select account_id from events where event_type = 'test_store_marker' and created_at >= %s order by id desc limit 1",
        (since,),
    ).fetchone()
    assert row["account_id"] is None  # not "BF-9999" -- that account doesn't exist


def test_log_event_fallback_is_still_visible_in_aggregate_metrics(reseed_accounts):
    # The event isn't silently dropped -- it's still real signal an
    # operator would see via observability/metrics.py, just not
    # attributed to a (nonexistent) borrower.
    since = datetime.now(timezone.utc) - timedelta(seconds=2)
    store.log_event("BF-9999", "test_store_marker_for_metrics", {"n": 1})

    counts = event_counts_since(since)
    assert counts.get("test_store_marker_for_metrics", 0) >= 1
