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


def test_get_account_by_telegram_chat_id_finds_the_real_account(reseed_accounts):
    store.set_telegram_chat_id("BF-1001", 900124)

    account = store.get_account_by_telegram_chat_id(900124)

    assert account is not None
    assert account.account_id == "BF-1001"


def test_get_account_by_telegram_chat_id_returns_none_for_an_unmapped_chat(reseed_accounts):
    assert store.get_account_by_telegram_chat_id(999999999) is None


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


def test_record_payment_treats_emi_plus_the_late_fee_as_a_regular_payment(reseed_accounts):
    # BF-1002 is seeded 11 days past due (EMI 22,000, past the 3-day grace period). Paying EMI + the
    # 500 fee -- exactly what the overdue "Pay" button and every reminder mint -- used to be recorded as
    # a 500 overpayment credited to the NEXT EMI.
    before = store.get_account_or_raise("BF-1002")

    result = store.record_payment("BF-1002", 22_500)

    assert result["kind"] == "regular"
    assert result["late_fee_paid"] == 500.0
    after = store.get_account_or_raise("BF-1002")
    assert after.months_remaining == before.months_remaining - 1
    assert after.pending_emi_credit == 0
    assert after.payment_history[-1].kind == "regular" and after.payment_history[-1].amount == 22_500


def test_record_payment_reduce_tenure_removes_whole_emis_against_the_real_check_constraint(reseed_accounts):
    # Writes a principal_prepayment_* kind through the REAL payment_history CHECK constraint (schema.sql) --
    # the production database had to be migrated by hand for these kinds, so CI is where drift shows up.
    before = store.get_account_or_raise("BF-1002")  # 20 months left, EMI 22,000
    assert before.months_remaining == 20

    # A deliberate 50,000 extra on top of the EMI -- NOT EMI + the late fee, so it is a real prepayment
    # even though this account is past its grace period (only an overage of exactly the fee is the fee).
    result = store.record_payment("BF-1002", 22_000 + 50_000, payment_scheme="reduce_tenure")

    assert result["kind"] == "principal_prepayment_reduce_tenure"
    assert result["late_fee_paid"] == 0.0
    after = store.get_account_or_raise("BF-1002")
    assert after.months_remaining == 17  # one for this EMI, two whole EMIs from the 50,000
    assert after.pending_emi_credit == 6_000  # the remainder is kept, not dropped
    assert after.payment_history[-1].kind == "principal_prepayment_reduce_tenure"


def test_get_borrower_messages_lists_everything_sent_to_the_borrower_newest_first(reseed_accounts):
    store.log_event("BF-1001", "clarification_request_sent", {"message": "Please call us", "delivered_via_telegram": False})
    store.log_event("BF-1001", "reminder_sent", {"kind": "due_now", "message": "Your EMI is due", "delivered_via_telegram": False})
    store.log_event("BF-1001", "restructuring_decision_notified", {"approved": True, "message": "Approved", "delivered_via_telegram": True})
    store.log_event("BF-1001", "dispute_resolution_notified", {"message": "Dispute closed", "delivered_via_telegram": False})
    store.log_event("BF-1001", "user_message", {"content": "not something we SENT"})
    store.log_event("BF-1002", "reminder_sent", {"kind": "due_now", "message": "someone else's", "delivered_via_telegram": False})

    messages = store.get_borrower_messages("BF-1001")

    assert [m["kind"] for m in messages] == ["dispute", "decision", "reminder", "message"]
    assert [m["message"] for m in messages] == ["Dispute closed", "Approved", "Your EMI is due", "Please call us"]
    assert messages[1]["delivered_via_telegram"] is True and messages[0]["delivered_via_telegram"] is False
