"""Unit tests for the 3 payment_tools -- the policy-enforcing tools
(minimum partial-payment threshold, dispute/broken-promise blocking,
restructuring math). Direct function calls, real seeded Postgres, no LLM.
"""

import os

import pytest

from businessflow.tools.payment_tools import calculate_hypothetical, generate_payment_link, propose_partial_payment

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_generate_payment_link_returns_synthetic_link_with_the_given_amount(reseed_accounts):
    result = generate_payment_link(account_id="BF-1004", amount=28000)

    assert result["account_id"] == "BF-1004"
    assert result["amount"] == 28000
    assert result["synthetic"] is True
    assert "BF-1004" in result["payment_link"]
    assert "28000" in result["payment_link"]


def test_generate_payment_link_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        generate_payment_link(account_id="BF-9999", amount=1000)


def test_propose_partial_payment_accepts_amount_exactly_at_policy_minimum(reseed_accounts):
    # BF-1002: emi_amount=22000, MIN_PARTIAL_PAYMENT_PCT=0.70 -> minimum=15400.
    # The boundary itself (not just comfortably-above) is the case worth
    # pinning down: `< minimum` must not reject a proposal equal to it.
    result = propose_partial_payment(account_id="BF-1002", proposed_amount=15400)

    assert result["eligible"] is True
    assert result["accepted_amount"] == 15400
    assert result["minimum_amount"] == 15400


def test_propose_partial_payment_rejects_amount_just_below_policy_minimum(reseed_accounts):
    result = propose_partial_payment(account_id="BF-1002", proposed_amount=15399.99)

    assert result["eligible"] is False
    assert result["minimum_amount"] == 15400
    assert "below the policy minimum" in result["reason"]


def test_propose_partial_payment_blocked_when_dispute_is_open(reseed_accounts):
    # BF-1003 has an open dispute (seeded) -- automated restructuring must
    # be refused regardless of the amount proposed, even a generous one.
    result = propose_partial_payment(account_id="BF-1003", proposed_amount=35000)

    assert result["eligible"] is False
    assert "dispute" in result["reason"]


def test_propose_partial_payment_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        propose_partial_payment(account_id="BF-9999", proposed_amount=1000)


def test_calculate_hypothetical_extend_tenure_math(reseed_accounts):
    # BF-1001: emi_amount=12500, months_remaining=14.
    # remaining_principal = 12500*14 = 175000; +2 months -> new_tenure=16
    # new_emi = 175000/16 = 10937.5
    result = calculate_hypothetical(account_id="BF-1001", restructuring_type="extend_tenure", extra_months=2)

    assert result["new_months_remaining"] == 16
    assert result["new_emi_amount"] == 10937.5


def test_calculate_hypothetical_rejects_extra_months_beyond_policy_cap(reseed_accounts):
    # MAX_RESTRUCTURING_EXTENSION_MONTHS = 3 -- 4 must be rejected.
    with pytest.raises(ValueError, match="extra_months"):
        calculate_hypothetical(account_id="BF-1001", restructuring_type="extend_tenure", extra_months=4)


def test_calculate_hypothetical_rejects_missing_extra_months_for_extend_tenure(reseed_accounts):
    with pytest.raises(ValueError, match="extra_months"):
        calculate_hypothetical(account_id="BF-1001", restructuring_type="extend_tenure")


def test_calculate_hypothetical_one_time_settlement_math(reseed_accounts):
    # BF-1004: emi_amount=28000, months_remaining=22.
    # remaining_principal = 28000*22 = 616000; 5% discount -> 585200.0
    result = calculate_hypothetical(account_id="BF-1004", restructuring_type="one_time_settlement")

    assert result["remaining_principal"] == 616000
    assert result["settlement_amount"] == 585200.0


def test_calculate_hypothetical_rejects_invalid_restructuring_type(reseed_accounts):
    with pytest.raises(ValueError, match="restructuring_type"):
        calculate_hypothetical(account_id="BF-1001", restructuring_type="waive_it_all")


def test_calculate_hypothetical_blocked_when_dispute_is_open(reseed_accounts):
    result = calculate_hypothetical(account_id="BF-1003", restructuring_type="one_time_settlement")

    assert result["eligible"] is False
    assert "dispute" in result["reason"]
