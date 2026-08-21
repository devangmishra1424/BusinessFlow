"""Unit tests for observability/metrics.py -- real Postgres, no LLM.

Uses a `since` timestamp set immediately before logging a small, known
batch of events, rather than asserting exact totals -- the events table
accumulates real activity across this whole project's testing/eval
history (with no per-test cleanup for account_id=None events in
particular), so an exact global count would be flaky by design.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

from businessflow.accounts import store
from businessflow.observability.metrics import escalation_rate, event_counts_since

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_event_counts_since_reflects_newly_logged_events(reseed_accounts):
    since = datetime.now(timezone.utc) - timedelta(seconds=2)
    store.log_event("BF-1001", "test_metrics_marker", {"n": 1})
    store.log_event("BF-1001", "test_metrics_marker", {"n": 2})

    counts = event_counts_since(since)

    assert counts.get("test_metrics_marker", 0) >= 2


def test_event_counts_since_empty_for_a_window_with_nothing_in_it(reseed_accounts):
    far_future = datetime.now(timezone.utc) + timedelta(days=365)
    assert event_counts_since(far_future) == {}


def test_escalation_rate_computed_from_real_tool_called_events(reseed_accounts):
    since = datetime.now(timezone.utc) - timedelta(seconds=2)
    store.log_event("BF-1001", "tool_called", {"tool": "get_payment_status", "arguments": {}, "result": {}})
    store.log_event("BF-1001", "tool_called", {"tool": "escalate_to_human", "arguments": {}, "result": {}})
    store.log_event("BF-1001", "tool_called", {"tool": "log_promise_to_pay", "arguments": {}, "result": {}})
    store.log_event("BF-1001", "tool_called", {"tool": "escalate_to_human", "arguments": {}, "result": {}})

    rate = escalation_rate(since)

    assert rate == pytest.approx(0.5)  # 2 of 4 in this exact window


def test_escalation_rate_zero_for_a_window_with_no_tool_calls(reseed_accounts):
    far_future = datetime.now(timezone.utc) + timedelta(days=365)
    assert escalation_rate(far_future) == 0.0
