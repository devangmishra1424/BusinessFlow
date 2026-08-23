"""Unit tests for the 4 account_tools -- direct function calls against the
real, seeded Postgres accounts (BF-1001..1004), no LLM involved. Every
test that mutates state uses the reseed_accounts fixture so it starts
from the canonical seed data, not whatever a previous test left behind.
"""

import os

import pytest

from businessflow.accounts import store
from businessflow.tools.account_tools import (
    _MAX_PAYMENT_HISTORY_LIMIT,
    flag_dispute,
    get_payment_history,
    get_payment_status,
    log_promise_to_pay,
)

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
    # Null by default -- the seeded demo accounts have never had a loan
    # agreement uploaded/parsed, so the column is at its NULL default.
    assert result["interest_rate_pct"] is None
    assert result["nach_mandate_active"] is True  # BF-1001's real seeded value
    # 3 days past due == GRACE_PERIOD_DAYS exactly -- still WITHIN the grace
    # period (grace_period.md: no late fee applies within the 3 days), not
    # past it, so no late fee yet.
    assert result["late_fee_applicable"] is False
    assert result["late_fee_amount"] is None
    assert result["dispute_open"] is False
    assert result["broken_promise_count"] == 0


def test_get_payment_status_applies_late_fee_once_past_the_grace_period(reseed_accounts):
    # BF-1002 is 11 days past due (real seeded emi_due_date 2026-08-10 vs
    # DEMO_TODAY 2026-08-21) -- well past GRACE_PERIOD_DAYS (3), so this is
    # the real seeded account that should trip late_fee_applicable. It also
    # has nach_mandate_active=False in the real seed data, so this same
    # account covers both new fields' "true"/non-null branches together.
    result = get_payment_status(account_id="BF-1002")

    assert result["days_past_due"] == 11
    assert result["nach_mandate_active"] is False
    assert result["late_fee_applicable"] is True
    assert result["late_fee_amount"] == 500.0


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


def test_get_payment_history_returns_real_seeded_records_most_recent_first(reseed_accounts):
    # BF-1001's real seeded payment_history has exactly 3 rows, oldest to
    # newest in the seed data (2026-05-18, 06-18, 07-18) -- this checks the
    # tool actually reverses that into most-recent-first, not just passes
    # the store's own ascending order through.
    result = get_payment_history(account_id="BF-1001")

    assert result["account_id"] == "BF-1001"
    assert result["payment_history"] == [
        {"date": "2026-07-18", "amount": 12500.0, "on_time": True},
        {"date": "2026-06-18", "amount": 12500.0, "on_time": True},
        {"date": "2026-05-18", "amount": 12500.0, "on_time": True},
    ]


def test_get_payment_history_respects_a_smaller_limit(reseed_accounts):
    result = get_payment_history(account_id="BF-1001", limit=2)

    assert [r["date"] for r in result["payment_history"]] == ["2026-07-18", "2026-06-18"]


def test_get_payment_history_clamps_a_limit_above_the_real_maximum(reseed_accounts):
    # BF-1001 only has 3 real seeded payment records -- a huge limit
    # wouldn't actually prove the cap is enforced unless there are more
    # than _MAX_PAYMENT_HISTORY_LIMIT rows to cap away in the first place.
    # Insert enough synthetic rows (cleaned up by the next test's
    # reseed_accounts, same as any other test's leaked mutation here) to
    # exceed the cap for real, then check the result stops at the cap, not
    # at whatever the caller asked for.
    conn = store.get_connection()
    for day in range(1, 26):  # 25 extra rows -- comfortably past the cap of 20
        conn.execute(
            "insert into payment_history (account_id, payment_date, amount, on_time) values (%s, %s, %s, %s)",
            ("BF-1001", f"2025-01-{day:02d}", 12500, True),
        )

    result = get_payment_history(account_id="BF-1001", limit=1000)

    assert len(result["payment_history"]) == _MAX_PAYMENT_HISTORY_LIMIT


def test_get_payment_history_clamps_a_non_positive_limit_up_to_one(reseed_accounts):
    # A borrower/model asking for 0 (or a negative limit) should never
    # error and should never come back genuinely empty -- clamp up to the
    # smallest real answer (the single most recent payment) instead.
    result = get_payment_history(account_id="BF-1001", limit=0)

    assert len(result["payment_history"]) == 1
    assert result["payment_history"][0]["date"] == "2026-07-18"


def test_get_payment_history_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        get_payment_history(account_id="BF-9999")
