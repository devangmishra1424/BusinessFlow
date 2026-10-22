"""Unit tests for the 4 account_tools -- direct function calls against the
real, seeded Postgres accounts (BF-1001..1004), no LLM involved. Every
test that mutates state uses the reseed_accounts fixture so it starts
from the canonical seed data, not whatever a previous test left behind.
"""

import os
from datetime import date, timedelta

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


def _due_date(days_past_due: int) -> str:
    """The real emi_due_date for an account currently `days_past_due`
    overdue -- mirrors scripts/seed_accounts.py's _DAYS_PAST_DUE, which
    seeds every date relative to date.today() rather than a fixed literal."""
    return (store.current_date() - timedelta(days=days_past_due)).isoformat()


def _days_before(due_iso: str, days: int) -> str:
    from datetime import date

    return (date.fromisoformat(due_iso) - timedelta(days=days)).isoformat()


# An arbitrary date safely in the future relative to whenever these tests
# actually run -- these tests only care that log_promise_to_pay accepts
# and persists *some* promised_date, not this specific one.
_FUTURE_PROMISE_DATE = (store.current_date() + timedelta(days=5)).isoformat()

# BF-1001's 3 real seeded payments -- offsets (31/61/92 days before its
# emi_due_date) match scripts/seed_accounts.py's _PAYMENT_HISTORY_OFFSETS
# for BF-1001 exactly, most-recent-first.
_BF1001_DUE = _due_date(3)
_BF1001_PAYMENT_RECENT = _days_before(_BF1001_DUE, 31)
_BF1001_PAYMENT_MID = _days_before(_BF1001_DUE, 61)
_BF1001_PAYMENT_OLDEST = _days_before(_BF1001_DUE, 92)


def test_get_payment_status_returns_real_account_fields(reseed_accounts):
    result = get_payment_status(account_id="BF-1001")

    assert result["account_id"] == "BF-1001"
    assert result["borrower_name"] == "Priya Sharma"
    assert result["principal_amount"] == 250_000.0
    assert result["emi_amount"] == 12500.0
    assert result["emi_due_date"] == _due_date(3)
    assert result["days_past_due"] == 3  # seeded 3 days before store.current_date()
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
    # BF-1002 is seeded 11 days past due (relative to store.current_date(),
    # see scripts/seed_accounts.py) -- well past GRACE_PERIOD_DAYS (3), so
    # this is the real seeded account that should trip late_fee_applicable. It also
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


def _promise_confirmed(**terms):
    """The chat flow: the terms are read back first, then recorded on the borrower's yes."""
    first = log_promise_to_pay(**terms)
    assert first["logged"] is False and first["read_back"]
    return log_promise_to_pay(**terms, borrower_confirmed=True)


def test_log_promise_to_pay_reads_the_terms_back_before_recording_anything(reseed_accounts):
    result = log_promise_to_pay(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=12500)

    assert result["logged"] is False
    assert result["read_back"].startswith("₹12,500 by ")
    assert store.get_account_or_raise("BF-1001").promises == []


def test_a_confirmation_without_a_read_back_is_not_recorded(reseed_accounts):
    # borrower_confirmed=True on the first call is not enough: the terms were never read back to them.
    result = log_promise_to_pay(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=12500,
                                borrower_confirmed=True)

    assert result["logged"] is False
    assert store.get_account_or_raise("BF-1001").promises == []


def test_confirming_different_terms_from_the_ones_read_back_is_not_recorded(reseed_accounts):
    log_promise_to_pay(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=12500)
    result = log_promise_to_pay(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=10000,
                                borrower_confirmed=True)

    assert result["logged"] is False and result["read_back"].startswith("₹10,000 by ")
    assert store.get_account_or_raise("BF-1001").promises == []


def test_log_promise_to_pay_persists_a_real_row(reseed_accounts):
    result = _promise_confirmed(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=12500)

    assert result["logged"] is True and result["already_logged"] is False and result["replaced_earlier_promise"] is False
    account = store.get_account_or_raise("BF-1001")
    [row] = account.promises
    assert (row.promised_date.isoformat(), row.promised_amount, row.made_via) == (_FUTURE_PROMISE_DATE, 12500.0, "chat")
    assert row.terms_text == result["terms"]


def test_a_promise_in_parts_is_one_plan_with_each_part_recorded(reseed_accounts):
    first_date = date.fromisoformat(_FUTURE_PROMISE_DATE)
    second = (first_date + timedelta(days=7)).isoformat()
    result = _promise_confirmed(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=5000,
                                more_parts=[{"date": second, "amount": 7500}])

    assert ", then ₹7,500 by " in result["terms"]
    parts = store.get_account_or_raise("BF-1001").promises
    assert [(p.part_no, p.parts_total, p.promised_amount, p.plan_id) for p in parts] == [
        (1, 2, 5000.0, result["plan_id"]), (2, 2, 7500.0, result["plan_id"])]
    view = get_payment_status("BF-1001")["promise"]
    assert view["status"] == "on_track" and view["next_part"] == {"date": _FUTURE_PROMISE_DATE, "amount_left": 5000.0}


def test_a_promise_with_dates_out_of_order_is_refused(reseed_accounts):
    earlier = (date.fromisoformat(_FUTURE_PROMISE_DATE) - timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="must come after"):
        log_promise_to_pay(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=5000,
                           more_parts=[{"date": earlier, "amount": 5000}])


def test_log_promise_to_pay_is_idempotent_against_an_identical_repeat_call(reseed_accounts):
    # A retry, or the model calling this twice in one turn, should log the
    # promise once -- not create two rows for what was meant to be one
    # real-world commitment.
    _promise_confirmed(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=12500)
    result = log_promise_to_pay(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=12500,
                                borrower_confirmed=True)

    assert result["already_logged"] is True
    account = store.get_account_or_raise("BF-1001")
    matching = [p for p in account.promises if p.promised_date.isoformat() == _FUTURE_PROMISE_DATE and p.promised_amount == 12500.0]
    assert len(matching) == 1


def test_a_new_promise_replaces_the_open_one_without_counting_it_broken(reseed_accounts):
    # A borrower revising their own promise before its date renegotiated; they did not break it.
    _promise_confirmed(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=12500)
    result = _promise_confirmed(account_id="BF-1001", promised_date=_FUTURE_PROMISE_DATE, promised_amount=8000)

    assert result["already_logged"] is False and result["replaced_earlier_promise"] is True
    account = store.get_account_or_raise("BF-1001")
    assert [(p.promised_amount, p.resolution) for p in account.promises] == [(12500.0, "void"), (8000.0, None)]
    assert account.broken_promise_count() == 0


def test_log_promise_to_pay_raises_on_malformed_date(reseed_accounts):
    # The tool assumes the caller (the model) always produces a clean ISO
    # date -- documenting the current failure mode rather than silently
    # tolerating it: a non-ISO date string blows up with an unhandled
    # ValueError, not a graceful "please clarify the date" response.
    with pytest.raises(ValueError):
        log_promise_to_pay(account_id="BF-1001", promised_date="next Tuesday", promised_amount=12500)


def test_log_promise_to_pay_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        log_promise_to_pay(account_id="BF-9999", promised_date=_FUTURE_PROMISE_DATE, promised_amount=100)


def test_flag_dispute_opens_a_real_dispute(reseed_accounts):
    result = flag_dispute(account_id="BF-1001", reason="The late fee was added although I paid on time")

    assert {k: result[k] for k in ("account_id", "dispute_open", "reason", "already_open", "kind")} == {
        "account_id": "BF-1001",
        "dispute_open": True,
        "reason": "The late fee was added although I paid on time",
        "already_open": False,
        "kind": "dispute",
    }
    assert result["case_ref"].startswith("D-")
    assert "review_within_working_days" not in result  # only a payment claim promises a check by a date
    assert store.get_account_or_raise("BF-1001").dispute_open is True


def test_a_payment_claim_is_its_own_kind_and_says_when_it_will_be_checked(reseed_accounts):
    from businessflow.accounts.policy import PAYMENT_CLAIM_REVIEW_WORKING_DAYS

    result = flag_dispute(account_id="BF-1001", reason="Paid 12,500 by UPI on the 3rd", kind="payment_claim")

    assert result["kind"] == "payment_claim" and result["already_open"] is False
    assert result["review_within_working_days"] == PAYMENT_CLAIM_REVIEW_WORKING_DAYS
    row = store.get_connection().execute(
        "select kind, reason from disputes where account_id = %s and status = 'open'", ("BF-1001",)
    ).fetchone()
    assert (row["kind"], row["reason"]) == ("payment_claim", "Paid 12,500 by UPI on the 3rd")


def test_more_details_on_an_open_claim_are_added_to_it_not_dropped(reseed_accounts):
    first = flag_dispute(account_id="BF-1001", reason="Paid by UPI on the 3rd", kind="payment_claim")
    second = flag_dispute(account_id="BF-1001", reason="UTR 412345678901", kind="payment_claim")

    assert second["already_open"] is True and second["case_ref"] == first["case_ref"]
    thread = store.get_case_threads("BF-1001")[first["case_ref"]]
    assert [(m["author"], m["message"]) for m in thread] == [("borrower", "UTR 412345678901")]
    rows = store.get_connection().execute("select id from disputes where account_id = %s", ("BF-1001",)).fetchall()
    assert len(rows) == 1


def test_flag_dispute_refuses_an_unknown_kind(reseed_accounts):
    with pytest.raises(ValueError, match="kind must be one of"):
        flag_dispute(account_id="BF-1001", reason="anything", kind="refund")


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
    # A retry with the same words is not added to the case as a second message either.
    assert store.get_case_threads("BF-1001").get(result["case_ref"], []) == []


def test_get_payment_history_returns_real_seeded_records_most_recent_first(reseed_accounts):
    # BF-1001's real seeded payment_history has exactly 3 rows, oldest to
    # newest in the seed data -- this checks the tool actually reverses
    # that into most-recent-first, not just passes the store's own
    # ascending order through.
    result = get_payment_history(account_id="BF-1001")

    assert result["account_id"] == "BF-1001"
    assert result["payment_history"] == [
        {"date": _BF1001_PAYMENT_RECENT, "amount": 12500.0, "on_time": True, "kind": "regular", "charges_paid": 0.0},
        {"date": _BF1001_PAYMENT_MID, "amount": 12500.0, "on_time": True, "kind": "regular", "charges_paid": 0.0},
        {"date": _BF1001_PAYMENT_OLDEST, "amount": 12500.0, "on_time": True, "kind": "regular", "charges_paid": 0.0},
    ]


def test_get_payment_history_respects_a_smaller_limit(reseed_accounts):
    result = get_payment_history(account_id="BF-1001", limit=2)

    assert [r["date"] for r in result["payment_history"]] == [_BF1001_PAYMENT_RECENT, _BF1001_PAYMENT_MID]


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
    assert result["payment_history"][0]["date"] == _BF1001_PAYMENT_RECENT


def test_get_payment_history_raises_on_unknown_account(reseed_accounts):
    with pytest.raises(ValueError, match="No account found"):
        get_payment_history(account_id="BF-9999")


def test_get_payment_status_reports_what_is_due_now(reseed_accounts):
    within_grace = get_payment_status(account_id="BF-1001")  # 3 days late: no fee yet
    past_grace = get_payment_status(account_id="BF-1002")  # 11 days late: EMI 22,000 + the 500 fee

    assert within_grace["amount_due_now"] == 12_500.0
    assert past_grace["amount_due_now"] == 22_500.0


def test_get_payment_status_says_nothing_is_late_or_due_on_a_fully_repaid_loan(reseed_accounts):
    store.get_connection().execute("update accounts set months_remaining = 0 where account_id = %s", ("BF-1002",))

    result = get_payment_status(account_id="BF-1002")  # its (seeded) due date is long past, but nothing is left to pay

    assert result["days_past_due"] == 0
    assert result["late_fee_applicable"] is False and result["late_fee_amount"] is None
    assert result["amount_due_now"] == 0.0
