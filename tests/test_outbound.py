"""Tests for the proactive outbound feature. decide.py is pure logic
against hand-built Account objects (no DB needed), same style as
test_flags.py -- boundary conditions matter most here, since that's
exactly where an off-by-one would silently miss or double-send a real
reminder. run.py's idempotency and compose.py's message generation need
real Postgres (and, for compose, a real Groq call).
"""

import os
from datetime import date

import pytest

from businessflow.accounts.models import Account
from businessflow.accounts.policy import GRACE_PERIOD_DAYS, HEADS_UP_DAYS_BEFORE_DUE
from businessflow.outbound.decide import decide_reminder, decide_reminders

_TODAY = date(2026, 8, 21)


def _account(**overrides) -> Account:
    defaults = dict(
        account_id="TEST-0001", borrower_name="Test Borrower", business_name="Test Business",
        phone_number="+919800000000", language_preference="en", loan_type="Working Capital Loan",
        principal_amount=100_000, emi_amount=10_000, tenure_months=12, months_remaining=6,
        emi_due_date=date(2026, 8, 23), nach_mandate_active=True, dispute_open=False, risk_tier="low",
    )
    defaults.update(overrides)
    return Account(**defaults)


def test_exactly_at_the_heads_up_window_edge_gets_a_heads_up():
    account = _account(emi_due_date=date(2026, 8, 23))  # HEADS_UP_DAYS_BEFORE_DUE=2 -> 2026-08-23
    assert (account.emi_due_date - _TODAY).days == HEADS_UP_DAYS_BEFORE_DUE

    reminder = decide_reminder(account, as_of=_TODAY)

    assert reminder is not None
    assert reminder.kind == "heads_up"
    assert reminder.days == HEADS_UP_DAYS_BEFORE_DUE


def test_one_day_beyond_the_heads_up_window_gets_nothing_yet():
    account = _account(emi_due_date=date(2026, 8, 24))  # 3 days out, beyond the 2-day window

    assert decide_reminder(account, as_of=_TODAY) is None


def test_due_today_is_not_a_heads_up_case():
    # days_until_due == 0 -- due today isn't "before due" any more, and
    # isn't past the grace period yet either.
    account = _account(emi_due_date=_TODAY)

    assert decide_reminder(account, as_of=_TODAY) is None


def test_exactly_at_the_grace_period_boundary_gets_no_follow_up():
    account = _account(emi_due_date=date(2026, 8, 18))  # 3 days past due, exactly at GRACE_PERIOD_DAYS
    assert (_TODAY - account.emi_due_date).days == GRACE_PERIOD_DAYS

    assert decide_reminder(account, as_of=_TODAY) is None


def test_one_day_past_the_grace_period_gets_a_follow_up():
    account = _account(emi_due_date=date(2026, 8, 17))  # 4 days past due

    reminder = decide_reminder(account, as_of=_TODAY)

    assert reminder is not None
    assert reminder.kind == "follow_up"
    assert reminder.days == GRACE_PERIOD_DAYS + 1


def test_an_open_dispute_suppresses_the_reminder_even_when_overdue():
    account = _account(emi_due_date=date(2026, 8, 1), dispute_open=True)  # 20 days past due

    assert decide_reminder(account, as_of=_TODAY) is None


def test_comfortably_within_the_cycle_gets_nothing():
    account = _account(emi_due_date=date(2026, 9, 15))  # weeks away

    assert decide_reminder(account, as_of=_TODAY) is None


def test_decide_reminder_only_fires_for_accounts_that_actually_need_one():
    accounts = [
        _account(account_id="A", emi_due_date=date(2026, 8, 23)),  # heads-up
        _account(account_id="B", emi_due_date=date(2026, 9, 15)),  # nothing
        _account(account_id="C", emi_due_date=date(2026, 8, 1)),  # follow-up
    ]

    reminders = [r for r in (decide_reminder(a, as_of=_TODAY) for a in accounts) if r is not None]

    assert {r.account_id for r in reminders} == {"A", "C"}


_pg_skip = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set -- these tests hit real Postgres",
)
_groq_skip = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set -- copy .env.example to .env and fill it in",
)


@_pg_skip
def test_decide_reminders_against_real_seeded_accounts(reseed_accounts):
    reminders = decide_reminders()
    by_id = {r.account_id: r for r in reminders}

    # BF-1001: 3 days past due -- exactly at the grace boundary, no reminder.
    assert "BF-1001" not in by_id
    # BF-1003: has an open dispute -- suppressed regardless of days past due.
    assert "BF-1003" not in by_id


@_pg_skip
def test_has_recent_event_with_detail_finds_a_real_logged_reminder(reseed_accounts):
    from datetime import datetime, timedelta, timezone

    from businessflow.accounts import store

    since = datetime.now(timezone.utc) - timedelta(seconds=2)
    store.log_event("BF-1001", "reminder_sent", {"kind": "heads_up", "message": "test"})

    assert store.has_recent_event_with_detail("BF-1001", "reminder_sent", since, "kind", "heads_up") is True
    assert store.has_recent_event_with_detail("BF-1001", "reminder_sent", since, "kind", "follow_up") is False


@_pg_skip
@_groq_skip
def test_run_daily_outbound_pass_does_not_resend_the_same_reminder_twice_in_one_day(reseed_accounts):
    from businessflow.outbound.decide import decide_reminders
    from businessflow.outbound.run import run_daily_outbound_pass

    # BF-1002 and BF-1004 are both real, overdue-only seeded accounts --
    # confirm at least one genuinely qualifies today before relying on it.
    due_today = {r.account_id for r in decide_reminders(["BF-1002", "BF-1004"])}
    assert due_today, "expected at least one of BF-1002/BF-1004 to have a real reminder due today"
    target = next(iter(due_today))

    first_pass = run_daily_outbound_pass([target])
    second_pass = run_daily_outbound_pass([target])

    assert len(first_pass) == 1
    assert first_pass[0]["account_id"] == target
    assert second_pass == []  # same day, same kind -- already sent, must not resend


@_pg_skip
@_groq_skip
def test_compose_message_produces_a_real_grounded_message():
    from businessflow.outbound.compose import compose_message
    from businessflow.outbound.decide import OutboundReminder

    account = _account(account_id="BF-TEST", borrower_name="Priya Sharma", emi_amount=12500, emi_due_date=date(2026, 8, 23))
    reminder = OutboundReminder(account_id="BF-TEST", kind="heads_up", days=2)

    message = compose_message(account, reminder)

    assert len(message) > 0
    assert "12500" in message.replace(",", "") or "12,500" in message
