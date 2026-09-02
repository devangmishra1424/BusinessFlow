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


def test_set_telegram_chat_id_persists_on_the_real_account_row(reseed_accounts):
    store.set_telegram_chat_id("BF-1001", 900123)

    assert store.get_account_or_raise("BF-1001").telegram_chat_id == 900123


def test_approve_restructuring_applies_the_real_proposed_changes(reseed_accounts):
    from businessflow.tools.escalation_tools import propose_restructuring

    proposal = propose_restructuring(account_id="BF-1001", extra_months=3)

    result = store.approve_restructuring(proposal["escalation_id"])

    assert result["account_id"] == "BF-1001"
    assert result["new_months_remaining"] == 17
    assert result["new_emi_amount"] == 10294.12

    account = store.get_account_or_raise("BF-1001")
    assert account.months_remaining == 17
    assert account.emi_amount == 10294.12

    escalation = store.get_escalation(proposal["escalation_id"])
    assert escalation.status == "approved"
    assert escalation.resolved_at is not None


def test_approve_restructuring_closes_a_plain_escalation_with_no_account_change(reseed_accounts):
    # Regression test for a real bug found live: escalate_to_human (an
    # open dispute, a broken-promise pattern, or the agent just being
    # unsure) creates an escalation with proposed_changes=None -- the vast
    # majority of real escalations, unlike propose_restructuring's
    # structured ones. Approving one of these used to raise ValueError
    # unconditionally, an unhandled 500 in the ops dashboard the instant
    # anyone clicked Approve on an ordinary escalation.
    from businessflow.tools.escalation_tools import escalate_to_human

    escalation = escalate_to_human(account_id="BF-1001", reason="Borrower has a general question, unsure how to help.")
    before = store.get_account_or_raise("BF-1001")

    result = store.approve_restructuring(escalation["escalation_id"])

    assert result == {"escalation_id": escalation["escalation_id"], "account_id": "BF-1001"}
    after = store.get_account_or_raise("BF-1001")
    assert after.months_remaining == before.months_remaining  # untouched -- nothing to apply
    assert after.emi_amount == before.emi_amount

    resolved = store.get_escalation(escalation["escalation_id"])
    assert resolved.status == "approved"
    assert resolved.resolved_at is not None


def test_approve_restructuring_raises_on_unknown_escalation(reseed_accounts):
    with pytest.raises(store.EscalationNotFoundError):
        store.approve_restructuring("ESC-9999999")


def test_approve_restructuring_raises_on_an_already_resolved_escalation(reseed_accounts):
    from businessflow.tools.escalation_tools import propose_restructuring

    proposal = propose_restructuring(account_id="BF-1001", extra_months=3)
    store.approve_restructuring(proposal["escalation_id"])

    # A double-click/retry must not silently double-apply the change --
    # it should error loudly instead.
    with pytest.raises(store.EscalationAlreadyResolvedError):
        store.approve_restructuring(proposal["escalation_id"])


def test_reject_restructuring_never_touches_the_account(reseed_accounts):
    from businessflow.tools.escalation_tools import propose_restructuring

    proposal = propose_restructuring(account_id="BF-1001", extra_months=3)

    result = store.reject_restructuring(proposal["escalation_id"], reason="Borrower already 2 EMIs behind")

    assert result["account_id"] == "BF-1001"
    assert result["reason"] == "Borrower already 2 EMIs behind"

    # Nothing was ever applied to the real account.
    account = store.get_account_or_raise("BF-1001")
    assert account.months_remaining == 14
    assert account.emi_amount == 12500

    escalation = store.get_escalation(proposal["escalation_id"])
    assert escalation.status == "rejected"
    assert escalation.resolution_reason == "Borrower already 2 EMIs behind"


def test_reject_restructuring_reason_is_optional(reseed_accounts):
    from businessflow.tools.escalation_tools import propose_restructuring

    proposal = propose_restructuring(account_id="BF-1001", extra_months=3)

    result = store.reject_restructuring(proposal["escalation_id"], reason=None)

    assert result["reason"] is None
    assert store.get_escalation(proposal["escalation_id"]).resolution_reason is None
