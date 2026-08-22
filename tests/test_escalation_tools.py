"""Unit tests for escalate_to_human -- direct function call, real Postgres."""

import os
import re

import pytest

from businessflow.tools.escalation_tools import escalate_to_human

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
