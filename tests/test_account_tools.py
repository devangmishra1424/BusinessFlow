"""Unit tests for the 3 account_tools -- direct function calls against the
real, seeded Postgres accounts (BF-1001..1004), no LLM involved. Every
test that mutates state uses the reseed_accounts fixture so it starts
from the canonical seed data, not whatever a previous test left behind.
"""

import os

import pytest

from businessflow.accounts import store
from businessflow.tools.account_tools import flag_dispute, get_payment_status, log_promise_to_pay

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_get_payment_status_returns_real_account_fields(reseed_accounts):
    result = get_payment_status(account_id="BF-1001")

    assert result["account_id"] == "BF-1001"
    assert result["borrower_name"] == "Priya Sharma"
    assert result["principal_amount"] == 250_000.0
    assert result["emi_amount"] == 12500.0
    assert result["emi_due_date"] == "2026-08-18"
    assert result["days_past_due"] == 3  # DEMO_TODAY (2026-08-21) - emi_due_date
    assert result["tenure_months"] == 24
    assert result["months_remaining"] == 14
    assert result["outstanding_balance_approx"] == 175_000.0  # 12500 * 14, same approximation calculate_hypothetical uses
    assert result["dispute_open"] is False
    assert result["broken_promise_count"] == 0


def test_get_payment_status_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        get_payment_status(account_id="BF-9999")


def test_log_promise_to_pay_persists_a_real_row(reseed_accounts):
    result = log_promise_to_pay(account_id="BF-1001", promised_date="2026-08-25", promised_amount=12500)

    assert result == {
        "account_id": "BF-1001",
        "promised_date": "2026-08-25",
        "promised_amount": 12500.0,
        "tolerance_days": 2,
        "logged": True,
        "already_logged": False,
    }
    account = store.get_account_or_raise("BF-1001")
    assert any(p.promised_date.isoformat() == "2026-08-25" and p.promised_amount == 12500.0 for p in account.promises)


def test_log_promise_to_pay_is_idempotent_against_an_identical_repeat_call(reseed_accounts):
    # A retry, or the model calling this twice in one turn, should log the
    # promise once -- not create two rows for what was meant to be one
    # real-world commitment.
    log_promise_to_pay(account_id="BF-1001", promised_date="2026-08-25", promised_amount=12500)
    result = log_promise_to_pay(account_id="BF-1001", promised_date="2026-08-25", promised_amount=12500)

    assert result["already_logged"] is True
    account = store.get_account_or_raise("BF-1001")
    matching = [p for p in account.promises if p.promised_date.isoformat() == "2026-08-25" and p.promised_amount == 12500.0]
    assert len(matching) == 1


def test_log_promise_to_pay_with_a_different_amount_is_a_genuinely_new_promise(reseed_accounts):
    # Idempotency must only collapse true repeats -- a borrower revising
    # their own promise (different amount) has to still go through.
    log_promise_to_pay(account_id="BF-1001", promised_date="2026-08-25", promised_amount=12500)
    result = log_promise_to_pay(account_id="BF-1001", promised_date="2026-08-25", promised_amount=8000)

    assert result["already_logged"] is False
    account = store.get_account_or_raise("BF-1001")
    assert len(account.promises) == 2


def test_log_promise_to_pay_raises_on_malformed_date(reseed_accounts):
    # The tool assumes the caller (the model) always produces a clean ISO
    # date -- documenting the current failure mode rather than silently
    # tolerating it: a non-ISO date string blows up with an unhandled
    # ValueError, not a graceful "please clarify the date" response.
    with pytest.raises(ValueError):
        log_promise_to_pay(account_id="BF-1001", promised_date="next Tuesday", promised_amount=12500)


def test_log_promise_to_pay_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        log_promise_to_pay(account_id="BF-9999", promised_date="2026-08-25", promised_amount=100)


def test_flag_dispute_opens_a_real_dispute(reseed_accounts):
    result = flag_dispute(account_id="BF-1001", reason="Already paid via UPI, not reflected")

    assert result == {
        "account_id": "BF-1001",
        "dispute_open": True,
        "reason": "Already paid via UPI, not reflected",
        "already_open": False,
    }
    assert store.get_account_or_raise("BF-1001").dispute_open is True


def test_flag_dispute_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        flag_dispute(account_id="BF-9999", reason="anything")


def test_flag_dispute_is_idempotent_against_a_repeat_call(reseed_accounts):
    # A second flag_dispute call on an already-disputed account (a retry,
    # or the model raising the same dispute twice) must not create a
    # second disputes log entry for the same underlying claim.
    flag_dispute(account_id="BF-1001", reason="Already paid via UPI, not reflected")
    result = flag_dispute(account_id="BF-1001", reason="Already paid via UPI, not reflected")

    assert result["already_open"] is True
    rows = store.get_connection().execute(
        "select id from disputes where account_id = %s", ("BF-1001",)
    ).fetchall()
    assert len(rows) == 1
