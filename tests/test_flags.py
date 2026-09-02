"""Tests for ops/flags.py -- the boundary conditions (exactly at grace
period, exactly at broken-promise threshold) matter more here than the
obvious cases, since those are exactly where an off-by-one would
silently mislabel a real account. Pure unit tests against hand-built
Account objects (no DB needed) plus integration tests against the real
seeded demo accounts, which are known-good ground truth from earlier
this session (BF-1001 clean, BF-1002 overdue only, BF-1003 all three
flags, BF-1004 overdue only).
"""

import os
from datetime import date

import pytest

from businessflow.accounts.models import Account, PromiseToPay
from businessflow.accounts.policy import BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION, GRACE_PERIOD_DAYS
from businessflow.ops.flags import compute_flags, is_clean

_TODAY = date(2026, 8, 21)

# compute_flags looks up the real dispute reason text via a real Postgres
# call (store.get_latest_open_dispute_reason) even for a hand-built
# Account with dispute_open=True set directly -- not actually pure,
# despite this file's own module docstring calling every test here "no DB
# needed"; only true for the tests that never hit a dispute-open account.
_pg_skip = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- compute_flags looks up the real dispute reason via Postgres",
)


def _account(**overrides) -> Account:
    defaults = dict(
        account_id="TEST-0001",
        borrower_name="Test Borrower",
        business_name="Test Business",
        phone_number="+919800000000",
        language_preference="en",
        loan_type="Working Capital Loan",
        principal_amount=100_000,
        emi_amount=10_000,
        tenure_months=12,
        months_remaining=6,
        emi_due_date=date(2026, 8, 18),  # 3 days before _TODAY
        nach_mandate_active=True,
        dispute_open=False,
        risk_tier="low",
    )
    defaults.update(overrides)
    return Account(**defaults)


def test_exactly_at_grace_period_boundary_is_not_flagged_overdue():
    # emi_due_date is exactly GRACE_PERIOD_DAYS before _TODAY -- the real
    # policy text (verified via check_policy earlier this session) says
    # the grace period covers day 3 itself, "after day 3" is overdue.
    account = _account(emi_due_date=date(2026, 8, 18))
    assert (_TODAY - account.emi_due_date).days == GRACE_PERIOD_DAYS

    flags = compute_flags(account, as_of=_TODAY)

    assert not any(f.label == "overdue" for f in flags)


def test_one_day_past_grace_period_is_flagged_overdue():
    account = _account(emi_due_date=date(2026, 8, 17))  # GRACE_PERIOD_DAYS + 1

    flags = compute_flags(account, as_of=_TODAY)

    assert any(f.label == "overdue" for f in flags)
    assert f"{GRACE_PERIOD_DAYS}-day grace period" in next(f for f in flags if f.label == "overdue").reason


def test_not_yet_due_is_never_flagged_overdue():
    account = _account(emi_due_date=date(2026, 9, 1))  # in the future relative to _TODAY

    flags = compute_flags(account, as_of=_TODAY)

    assert not any(f.label == "overdue" for f in flags)


@_pg_skip
def test_open_dispute_is_flagged():
    account = _account(dispute_open=True)

    flags = compute_flags(account, as_of=_TODAY)

    assert any(f.label == "disputed" for f in flags)


def test_one_broken_promise_below_threshold_is_not_flagged():
    account = _account(promises=[
        PromiseToPay(made_on=date(2026, 7, 1), promised_date=date(2026, 7, 5), promised_amount=10_000, kept=False),
    ])
    assert account.broken_promise_count() == 1
    assert BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION == 2  # this test's premise depends on the real threshold

    flags = compute_flags(account, as_of=_TODAY)

    assert not any(f.label == "broken_promises" for f in flags)


def test_broken_promises_at_threshold_is_flagged():
    account = _account(promises=[
        PromiseToPay(made_on=date(2026, 6, 1), promised_date=date(2026, 6, 5), promised_amount=10_000, kept=False),
        PromiseToPay(made_on=date(2026, 7, 1), promised_date=date(2026, 7, 5), promised_amount=10_000, kept=False),
    ])

    flags = compute_flags(account, as_of=_TODAY)

    assert any(f.label == "broken_promises" for f in flags)


def test_kept_promises_dont_count_as_broken():
    account = _account(promises=[
        PromiseToPay(made_on=date(2026, 6, 1), promised_date=date(2026, 6, 5), promised_amount=10_000, kept=True),
        PromiseToPay(made_on=date(2026, 7, 1), promised_date=date(2026, 7, 5), promised_amount=10_000, kept=True),
    ])

    flags = compute_flags(account, as_of=_TODAY)

    assert not any(f.label == "broken_promises" for f in flags)


def test_unresolved_promise_kept_is_none_and_doesnt_count_as_broken():
    # kept=None means the promised_date hasn't passed and been evaluated
    # yet -- must not be treated as a broken promise just because it's
    # not explicitly True.
    account = _account(promises=[
        PromiseToPay(made_on=date(2026, 8, 20), promised_date=date(2026, 8, 25), promised_amount=10_000, kept=None),
    ])

    flags = compute_flags(account, as_of=_TODAY)

    assert not any(f.label == "broken_promises" for f in flags)


def test_clean_account_has_no_flags():
    account = _account(emi_due_date=date(2026, 8, 19), dispute_open=False)  # 2 days past due, within grace

    assert is_clean(account, as_of=_TODAY)


@_pg_skip
def test_an_account_can_carry_multiple_flags_at_once():
    account = _account(
        emi_due_date=date(2026, 8, 1),  # 20 days past due
        dispute_open=True,
        promises=[
            PromiseToPay(made_on=date(2026, 6, 1), promised_date=date(2026, 6, 5), promised_amount=10_000, kept=False),
            PromiseToPay(made_on=date(2026, 7, 1), promised_date=date(2026, 7, 5), promised_amount=10_000, kept=False),
        ],
    )

    flags = {f.label for f in compute_flags(account, as_of=_TODAY)}

    assert flags == {"overdue", "disputed", "broken_promises"}


# --- Integration: against the real seeded demo accounts ---

pytestmark_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


@pytestmark_db
def test_real_seeded_accounts_match_known_flag_distribution(reseed_accounts):
    from businessflow.accounts import store

    expected = {
        "BF-1001": set(),  # 3 days past due -- exactly at the grace boundary, clean
        "BF-1002": {"overdue"},
        "BF-1003": {"overdue", "disputed", "broken_promises"},
        "BF-1004": {"overdue"},
    }
    for account_id, expected_labels in expected.items():
        account = store.get_account_or_raise(account_id)
        actual_labels = {f.label for f in compute_flags(account)}
        assert actual_labels == expected_labels, f"{account_id}: expected {expected_labels}, got {actual_labels}"
