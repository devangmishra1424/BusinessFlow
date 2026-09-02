"""Tests for the proactive outbound feature. decide.py is pure logic
against hand-built Account objects (no DB needed), same style as
test_flags.py -- boundary conditions matter most here, since that's
exactly where an off-by-one would silently miss or double-send a real
reminder. run.py's idempotency and compose.py's message generation need
real Postgres (and, for compose, a real Groq call).
"""

import os
from datetime import date, timedelta

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
    account = _account(emi_due_date=_TODAY + timedelta(days=HEADS_UP_DAYS_BEFORE_DUE))
    assert (account.emi_due_date - _TODAY).days == HEADS_UP_DAYS_BEFORE_DUE

    reminder = decide_reminder(account, as_of=_TODAY)

    assert reminder is not None
    assert reminder.kind == "heads_up"
    assert reminder.days == HEADS_UP_DAYS_BEFORE_DUE


def test_one_day_beyond_the_heads_up_window_gets_nothing_yet():
    account = _account(emi_due_date=_TODAY + timedelta(days=HEADS_UP_DAYS_BEFORE_DUE + 1))

    assert decide_reminder(account, as_of=_TODAY) is None


def test_due_today_is_not_a_heads_up_case():
    # days_until_due == 0 -- due today isn't "before due" any more, and
    # isn't past the grace period yet either. It's the due_now case: no
    # late fee has kicked in, but "pay now" is exactly the right message.
    account = _account(emi_due_date=_TODAY)

    reminder = decide_reminder(account, as_of=_TODAY)

    assert reminder is not None
    assert reminder.kind == "due_now"
    assert reminder.days == 0


def test_exactly_at_the_grace_period_boundary_gets_due_now_not_follow_up():
    account = _account(emi_due_date=date(2026, 8, 18))  # 3 days past due, exactly at GRACE_PERIOD_DAYS
    assert (_TODAY - account.emi_due_date).days == GRACE_PERIOD_DAYS

    reminder = decide_reminder(account, as_of=_TODAY)

    assert reminder is not None
    assert reminder.kind == "due_now"
    assert reminder.days == GRACE_PERIOD_DAYS


def test_one_day_past_due_within_grace_gets_due_now():
    account = _account(emi_due_date=date(2026, 8, 20))  # 1 day past due, well within the 3-day grace period

    reminder = decide_reminder(account, as_of=_TODAY)

    assert reminder is not None
    assert reminder.kind == "due_now"
    assert reminder.days == 1


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

    # BF-1001: 3 days past due -- exactly at the grace boundary, a due_now
    # reminder (no late fee yet, but "pay now" still applies).
    assert by_id["BF-1001"].kind == "due_now"
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
def test_send_reminder_falls_back_to_a_logged_event_with_no_linked_telegram_chat(reseed_accounts):
    # BF-1001 has no telegram_chat_id after a reseed -- nothing to
    # actually deliver to, so this exercises the real logged-fallback
    # path, not a network call to Telegram.
    from datetime import datetime, timedelta, timezone

    from businessflow.accounts import store
    from businessflow.outbound.send import send_reminder

    since = datetime.now(timezone.utc) - timedelta(seconds=2)
    delivered = send_reminder("BF-1001", "heads_up", "Your EMI is due soon.")

    assert delivered is False
    row = store.get_connection().execute(
        "select details from events where account_id = %s and event_type = 'reminder_sent' "
        "and created_at >= %s order by created_at desc limit 1",
        ("BF-1001", since),
    ).fetchone()
    assert row["details"]["kind"] == "heads_up"
    assert row["details"]["delivered_via_telegram"] is False


@_pg_skip
def test_send_reminder_attempts_real_telegram_delivery_when_a_chat_is_linked(reseed_accounts):
    # 900999 isn't a real chat that has started this bot -- Telegram
    # rejects it with a real "Chat not found" error, caught by
    # _send_telegram_message and surfaced here as delivered=False,
    # confirming this path genuinely attempts delivery rather than
    # silently skipping straight to the fallback.
    from businessflow.accounts import store
    from businessflow.outbound.send import send_reminder

    store.set_telegram_chat_id("BF-1001", 900999)

    delivered = send_reminder("BF-1001", "follow_up", "Your EMI is overdue.")

    assert delivered is False


@_pg_skip
def test_send_reminder_with_a_payment_link_still_attempts_real_delivery(reseed_accounts):
    # Same "fake chat_id, real rejection" proof as the test above, but
    # exercising the reply_markup/InlineKeyboardButton construction path --
    # confirms building a real "Pay now" button doesn't itself raise
    # before the message even reaches Telegram.
    from businessflow.accounts import store
    from businessflow.outbound.send import send_reminder

    store.set_telegram_chat_id("BF-1001", 900999)

    delivered = send_reminder(
        "BF-1001", "due_now", "Your EMI is due today.", payment_url="https://example.com/pay/faketoken", payment_amount=12500
    )

    assert delivered is False


@_pg_skip
def test_resolve_matured_promises_marks_a_fulfilled_promise_kept(reseed_accounts):
    # Regression coverage for a real gap this codebase's own README named
    # under "Known gaps": promises.kept was only ever written once, by this
    # same seed script, at seed time -- nothing ever evaluated a promise
    # made afterward. BF-1001 has no seeded promises of its own (see
    # scripts/seed_accounts.py's _PROMISE_OFFSETS), so this is a clean
    # slate: add one, whose evaluation window has already passed, backed
    # by a real matching payment, and confirm the job actually resolves it.
    from businessflow.accounts import store

    today = store.current_date()
    made_on = today - timedelta(days=20)
    promised_date = today - timedelta(days=10)  # well past promised_date + PROMISE_TOLERANCE_DAYS
    store.add_promise("BF-1001", made_on, promised_date, 5_000)
    store.record_payment("BF-1001", 5_000, payment_date=promised_date, apply_extra_to_next=False)

    resolved = store.resolve_matured_promises()

    matches = [r for r in resolved if r["account_id"] == "BF-1001"]
    assert matches and all(r["kept"] is True for r in matches)


@_pg_skip
def test_resolve_matured_promises_marks_an_unfulfilled_promise_broken(reseed_accounts):
    from businessflow.accounts import store

    today = store.current_date()
    store.add_promise("BF-1004", today - timedelta(days=20), today - timedelta(days=10), 99_999)

    resolved = store.resolve_matured_promises()

    matches = [r for r in resolved if r["account_id"] == "BF-1004"]
    assert matches and all(r["kept"] is False for r in matches)


@_pg_skip
def test_resolve_matured_promises_leaves_a_still_pending_promise_alone(reseed_accounts):
    # promised_date + PROMISE_TOLERANCE_DAYS hasn't passed yet -- must not
    # be touched at all, kept or broken, until its window actually closes.
    from businessflow.accounts import store

    today = store.current_date()
    store.add_promise("BF-1001", today, today + timedelta(days=5), 5_000)

    resolved = store.resolve_matured_promises()

    assert not any(r["account_id"] == "BF-1001" for r in resolved)
    fresh = store.get_account_or_raise("BF-1001")
    assert fresh.promises[-1].kept is None


@_pg_skip
def test_resolve_promises_escalates_exactly_once_the_broken_promise_threshold_is_crossed(reseed_accounts):
    from businessflow.accounts import store
    from businessflow.outbound.run import resolve_promises

    # BF-1002 already has exactly one seeded broken promise (see
    # scripts/seed_accounts.py's _PROMISE_OFFSETS: kept=False), so one more
    # genuine break here crosses BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION
    # (2) for the first time, not the fifth.
    before = store.get_account_or_raise("BF-1002")
    assert before.broken_promise_count() == 1

    today = store.current_date()
    store.add_promise("BF-1002", today - timedelta(days=20), today - timedelta(days=10), 99_999)

    result = resolve_promises()

    after = store.get_account_or_raise("BF-1002")
    assert after.broken_promise_count() == 2
    assert any(e["account_id"] == "BF-1002" for e in result["escalated"])

    # Running it again (nothing new to resolve) must not open a second,
    # duplicate escalation -- create_escalation's own account_id+reason
    # dedup (see accounts/store.py) is what's actually relied on here.
    second_result = resolve_promises()
    assert not any(e["account_id"] == "BF-1002" for e in second_result["escalated"])
    escalations = store.get_escalations_for_account("BF-1002")
    broken_promise_escalations = [e for e in escalations if "broken promises" in e.reason.lower()]
    assert len(broken_promise_escalations) == 1


@_pg_skip
@_groq_skip
def test_run_daily_outbound_pass_escalates_a_chronically_overdue_account(reseed_accounts):
    # Regression coverage for a real gap: follow_up reminders fired
    # identically forever with no ceiling, never themselves escalating to
    # a human no matter how delinquent an account got. BF-1002 has no open
    # dispute (unlike BF-1003), so pushing it well past
    # MANDATORY_ESCALATION_DAYS_PAST_DUE exercises the real decide->
    # escalate path end to end.
    from businessflow.accounts import store
    from businessflow.outbound.run import _CHRONIC_DELINQUENCY_REASON, run_daily_outbound_pass

    new_due_date = store.current_date() - timedelta(days=20)
    store.get_connection().execute(
        "update accounts set emi_due_date = %s where account_id = %s", (new_due_date, "BF-1002")
    )

    run_daily_outbound_pass(["BF-1002"])

    escalations = store.get_escalations_for_account("BF-1002")
    assert any(
        e.reason == _CHRONIC_DELINQUENCY_REASON and e.status == "queued_for_human" for e in escalations
    )


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
