"""One case, one conversation (accounts/cases.py) -- the pure parts, no database: case references, how event rows
become threads, and how each dashboard builds a case from them.

Found live: "i have paid my emi, cn you update it in the profile" was filed as a plain dispute, the borrower read our
reply in a different panel from their request, and ops wrote replies that weren't attached to anything.
"""

from datetime import datetime, timedelta, timezone

import pytest

from businessflow.accounts import cases
from businessflow.accounts import escalation_kinds as kinds
from businessflow.accounts.models import Escalation

T0 = datetime(2026, 10, 15, 9, tzinfo=timezone.utc)


# --------------------------------------------------------------------------------------------------- references


@pytest.mark.parametrize("ref, parsed", [
    ("D-12", ("dispute", 12)),
    ("D-1", ("dispute", 1)),
    ("ESC-0012", ("escalation", "ESC-0012")),
    ("ESC-123456", ("escalation", "ESC-123456")),
    (" D-7 ", ("dispute", 7)),
])
def test_a_case_reference_names_a_dispute_or_an_escalation(ref, parsed):
    assert cases.parse_case_ref(ref) == parsed


@pytest.mark.parametrize("ref", [
    "", None, "D-", "D-0", "D-012", "D-12; drop table disputes", "d-12", "ESC-12", "esc-0012", "ESC-0012x",
    "D-" + "9" * 19, "12", "dispute-2026-10-01T09:00:00+00:00",
])
def test_anything_else_is_not_a_case_reference(ref):
    assert cases.parse_case_ref(ref) is None


def test_a_dispute_reference_round_trips():
    assert cases.parse_case_ref(cases.dispute_case_ref(42)) == ("dispute", 42)


# ------------------------------------------------------------------------------------------------------ threads


def _row(event_type, minutes, **details):
    return {"event_type": event_type, "details": details, "created_at": T0 + timedelta(minutes=minutes)}


def test_event_rows_become_one_thread_per_case_oldest_first():
    rows = [  # newest first, as the store reads them
        _row("dispute_resolution_notified", 30, case_ref="D-1", message="Good news: we've found your payment", delivered_via_telegram=True),
        _row(cases.CASE_MESSAGE_EVENT, 20, case_ref="D-1", author="borrower", message="UTR 4123"),
        _row(cases.CASE_MESSAGE_EVENT, 10, case_ref="D-1", author="ops", message="Could you share the reference?"),
        _row("restructuring_decision_notified", 5, case_ref="ESC-0003", approved=True, message="Approved"),
    ]

    threads = cases.group_threads(rows)

    assert [(m["author"], m["message"]) for m in threads["D-1"]] == [
        ("ops", "Could you share the reference?"),
        ("borrower", "UTR 4123"),
        ("ops", "Good news: we've found your payment"),  # an outcome notice is from the team
    ]
    assert threads["D-1"][-1]["delivered_via_telegram"] is True
    assert [m["message"] for m in threads["ESC-0003"]] == ["Approved"]


def test_rows_without_a_case_or_a_message_are_not_thread_messages():
    rows = [
        _row("reminder_sent", 1, message="Your EMI is due"),  # no case: a general message
        _row(cases.CASE_MESSAGE_EVENT, 2, case_ref="D-1", author="ops", message="   "),
        _row(cases.CASE_MESSAGE_EVENT, 3, case_ref="D-1", author="borrower"),
    ]

    assert cases.group_threads(rows) == {}


# ------------------------------------------------------------------------------- what each dashboard builds from them


def _escalation(i, reason, status="queued_for_human", changes=None, hours_ago=1):
    created = T0 - timedelta(hours=hours_ago)
    return Escalation(
        escalation_id=f"ESC-{i:04d}", account_id="BF-TEST", reason=reason, status=status, created_at=created,
        resolved_at=None if status == "queued_for_human" else created + timedelta(minutes=30),
        proposed_changes=changes, resolution_reason=None,
    )


def _dispute(i, kind, status="open", hours_ago=1, note=None):
    opened = T0 - timedelta(hours=hours_ago)
    return {
        "id": i, "kind": kind, "reason": f"words {i}", "status": status, "opened_at": opened,
        "resolved_at": None if status == "open" else opened + timedelta(hours=1), "resolution_note": note,
    }


def test_the_borrower_sees_a_payment_claim_as_one_conversation_they_can_add_to():
    from businessflow.channels.browser_api import _build_requests

    threads = cases.group_threads([
        _row(cases.CASE_MESSAGE_EVENT, 5, case_ref="D-7", author="ops", message="Could you share the reference?"),
    ])

    (claim,) = _build_requests([], [_dispute(7, cases.PAYMENT_CLAIM)], threads)

    assert claim["id"] == "D-7" and claim["kind"] == "payment_claim"
    assert claim["title"] == "Payment you reported" and claim["status_label"] == "Being checked"
    assert claim["detail"] == "words 7"  # their own words open the conversation
    assert [m["message"] for m in claim["thread"]] == ["Could you share the reference?"]
    assert claim["can_reply"] is True


def test_a_closed_request_takes_no_more_notes():
    from businessflow.channels.browser_api import _build_requests

    requests = _build_requests(
        [_escalation(1, "Borrower requested a human agent", status="approved")],
        [_dispute(2, cases.DISPUTE, status="resolved")],
        {},
    )

    assert [r["can_reply"] for r in requests] == [False, False]
    assert next(r for r in requests if r["kind"] == "dispute")["status_label"] == "Resolved"


def test_the_borrower_is_told_a_payment_claim_is_being_checked_not_that_they_disputed_something():
    from businessflow.accounts.policy import PAYMENT_CLAIM_REVIEW_WORKING_DAYS
    from businessflow.channels.browser_api import _build_warnings

    class _Flag:
        label, reason = "disputed", "staff-toned reason"

    status = {"days_past_due": 0, "late_fee_applicable": False, "late_fee_amount": None, "broken_promise_count": 0}
    (claim,) = _build_warnings([_Flag()], status, open_claim_kind=cases.PAYMENT_CLAIM)
    (dispute,) = _build_warnings([_Flag()], status, open_claim_kind=cases.DISPUTE)

    assert "checking the payment you reported" in claim["text"]
    assert f"{PAYMENT_CLAIM_REVIEW_WORKING_DAYS} working days" in claim["text"] and "paused" in claim["text"]
    assert "open dispute" in dispute["text"]


def test_the_ops_inbox_puts_a_fraud_claim_first_then_claims_and_requests_then_system_notices():
    from businessflow.ops.api import _dispute_case, _escalation_case, _sorted_cases

    items = [
        _escalation_case(_escalation(1, f"{kinds.BROKEN_PROMISE_PATTERN_PREFIX} -- 2 broken", hours_ago=50), "A", {}),
        _escalation_case(_escalation(2, "Borrower requested a human agent", hours_ago=40), "B", {}),
        _dispute_case(_dispute(3, cases.PAYMENT_CLAIM, hours_ago=2), "BF-3", "C", {}),
        _escalation_case(_escalation(4, f"{kinds.FRAUD_REASON_PREFIX}: not my loan", hours_ago=1), "D", {}),
        _dispute_case(_dispute(5, cases.DISPUTE, hours_ago=30), "BF-5", "E", {}),
        _dispute_case(_dispute(6, cases.DISPUTE, status="resolved", hours_ago=60), "BF-6", "F", {}),
    ]

    ordered = _sorted_cases(items)

    assert [c.case_ref for c in ordered] == ["ESC-0004", "D-5", "D-3", "ESC-0002", "ESC-0001", "D-6"]
    assert ordered[0].kind == "fraud" and ordered[-1].open is False


def test_ops_case_titles_say_what_each_case_is():
    from businessflow.ops.api import _case_title

    assert _case_title(cases.PAYMENT_CLAIM, "paid on the 3rd") == "Says they already paid"
    assert _case_title(kinds.SYSTEM, f"{kinds.BROKEN_PROMISE_PATTERN_PREFIX} -- 2") == "Broken promises"
    assert _case_title(kinds.SYSTEM, kinds.CHRONIC_DELINQUENCY_REASON) == "Chronically overdue"
    assert _case_title(kinds.SYSTEM, f"{kinds.GUARDRAIL_REASON_PREFIX} x") == "Agent reply blocked"
    assert _case_title(kinds.SYSTEM, "something new") == "System notice"


def test_a_system_notice_cannot_be_replied_to_because_the_borrower_never_sees_it():
    from businessflow.ops.api import _escalation_case

    notice = _escalation_case(_escalation(1, kinds.CHRONIC_DELINQUENCY_REASON), "A", {})
    request = _escalation_case(_escalation(2, "Borrower requested a human agent"), "B", {})

    assert notice.can_reply is False and request.can_reply is True
