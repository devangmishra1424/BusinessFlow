"""Unit tests for escalate_to_human, request_closure_certificate, and
propose_restructuring -- direct function calls, real Postgres."""

import os
import re

import pytest

from businessflow.accounts import store
from businessflow.tools.escalation_tools import (
    _CLOSURE_CERTIFICATE_REASON,
    escalate_to_human,
    propose_restructuring,
    request_closure_certificate,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_escalate_to_human_creates_a_real_escalation(reseed_accounts):
    result = escalate_to_human(account_id="BF-1003", reason="Borrower demands a human, dispute unresolved")

    assert result["account_id"] == "BF-1003"
    assert result["reason"] == "Borrower demands a human, dispute unresolved"
    assert result["status"] == "queued_for_human"
    # escalation_seq is a shared, monotonically increasing Postgres sequence
    # -- never reset by re-seeding -- so only the format is stable across
    # runs, not a specific number.
    assert re.match(r"^ESC-\d+$", result["escalation_id"])


def test_escalate_to_human_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        escalate_to_human(account_id="BF-9999", reason="anything")


def test_escalate_to_human_is_idempotent_against_an_identical_repeat_call(reseed_accounts):
    # A retry, or the model escalating for the same stated reason twice in
    # one turn, should produce one ticket for a human to work, not two.
    first = escalate_to_human(account_id="BF-1003", reason="Borrower demands a human, dispute unresolved")
    second = escalate_to_human(account_id="BF-1003", reason="Borrower demands a human, dispute unresolved")

    assert second["escalation_id"] == first["escalation_id"]


def test_escalate_to_human_with_a_different_reason_opens_a_genuinely_new_ticket(reseed_accounts):
    # Idempotency must only collapse true repeats -- a second, distinct
    # reason to escalate the same account is real new information for
    # the human on the other end, not noise to dedupe away.
    first = escalate_to_human(account_id="BF-1003", reason="Borrower demands a human, dispute unresolved")
    second = escalate_to_human(account_id="BF-1003", reason="Borrower also asked about a totally separate loan")

    assert second["escalation_id"] != first["escalation_id"]


def test_request_closure_certificate_is_not_eligible_before_fully_repaid(reseed_accounts):
    # BF-1001's real seeded months_remaining is 14 -- nowhere near fully
    # repaid, so there's nothing yet for a human to act on.
    result = request_closure_certificate(account_id="BF-1001")

    assert result == {
        "account_id": "BF-1001",
        "eligible": False,
        "months_remaining": 14,
    }
    # No escalation should have been created for this -- confirmed against
    # the real escalations table, not just the returned dict.
    assert store.get_escalations_for_account("BF-1001") == []


def test_request_closure_certificate_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        request_closure_certificate(account_id="BF-9999")


def test_request_closure_certificate_escalates_once_fully_repaid(reseed_accounts):
    # No seeded demo account has months_remaining == 0 -- set one up
    # directly against the real row (the same direct-SQL-via-
    # store.get_connection() pattern test_account_tools.py's dispute test
    # already uses to inspect real state) rather than adding a whole new
    # store function just for this one test's setup.
    store.get_connection().execute(
        "update accounts set months_remaining = 0 where account_id = %s", ("BF-1001",)
    )

    result = request_closure_certificate(account_id="BF-1001")

    assert result["account_id"] == "BF-1001"
    assert result["eligible"] is True
    assert result["months_remaining"] == 0
    assert result["status"] == "queued_for_human"
    assert result["reason"] == _CLOSURE_CERTIFICATE_REASON
    assert re.match(r"^ESC-\d+$", result["escalation_id"])

    escalations = store.get_escalations_for_account("BF-1001")
    assert len(escalations) == 1
    assert escalations[0].reason == _CLOSURE_CERTIFICATE_REASON
    assert "fully repaid" in escalations[0].reason
    assert escalations[0].status == "queued_for_human"


def test_request_closure_certificate_is_idempotent_against_a_repeat_call(reseed_accounts):
    # Same idempotency contract as escalate_to_human itself (this reuses
    # store.create_escalation directly) -- a retry once fully repaid must
    # not open a second ticket for the same real request.
    store.get_connection().execute(
        "update accounts set months_remaining = 0 where account_id = %s", ("BF-1001",)
    )

    first = request_closure_certificate(account_id="BF-1001")
    second = request_closure_certificate(account_id="BF-1001")

    assert second["escalation_id"] == first["escalation_id"]
    assert len(store.get_escalations_for_account("BF-1001")) == 1


def test_propose_restructuring_queues_a_real_pending_approval(reseed_accounts):
    # BF-1001: emi_amount=12500, months_remaining=14 -- same math as
    # calculate_hypothetical's own test (175000 / 17 = 10294.117...).
    result = propose_restructuring(account_id="BF-1001", extra_months=3)

    assert result["account_id"] == "BF-1001"
    assert result["status"] == "pending_approval"
    assert result["type"] == "extend_tenure"
    assert result["new_months_remaining"] == 17
    assert result["new_emi_amount"] == 10294.12
    assert re.match(r"^ESC-\d+$", result["escalation_id"])

    # The account itself must NOT have changed yet -- this only queues a
    # human approval, it never applies anything on its own.
    account = store.get_account_or_raise("BF-1001")
    assert account.months_remaining == 14
    assert account.emi_amount == 12500

    escalations = store.get_escalations_for_account("BF-1001")
    assert len(escalations) == 1
    assert escalations[0].status == "queued_for_human"
    assert escalations[0].proposed_changes == {
        "type": "extend_tenure", "extra_months": 3,
        "new_months_remaining": 17, "new_emi_amount": 10294.12,
    }


def test_propose_restructuring_blocked_when_dispute_is_open(reseed_accounts):
    # BF-1003 has an open dispute (seeded) -- same block
    # calculate_hypothetical itself enforces, and no escalation should
    # be created for a proposal that was never actually eligible.
    result = propose_restructuring(account_id="BF-1003", extra_months=2)

    assert result["eligible"] is False
    assert "dispute" in result["reason"]
    assert store.get_escalations_for_account("BF-1003") == []


def test_propose_restructuring_rejects_extra_months_beyond_policy_cap(reseed_accounts):
    with pytest.raises(ValueError, match="extra_months"):
        propose_restructuring(account_id="BF-1001", extra_months=4)


def test_propose_restructuring_two_different_proposals_open_two_distinct_tickets(reseed_accounts):
    # Different real numbers -> different reason text -> no false dedup,
    # unlike create_escalation's normal exact-repeat collapsing.
    first = propose_restructuring(account_id="BF-1001", extra_months=1)
    second = propose_restructuring(account_id="BF-1001", extra_months=2)

    assert first["escalation_id"] != second["escalation_id"]
    assert len(store.get_escalations_for_account("BF-1001")) == 2
