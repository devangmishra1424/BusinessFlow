"""Promise plans (accounts/promises.py): a promise is judged by its own words -- each part, an amount by a date, paid in
full by that date or the promise is broken. Pure logic first (no DB), then the store and the reminder pause against
real Postgres, same split as test_outbound.py.
"""

from datetime import date, timedelta

import pytest

from businessflow.accounts import store
from businessflow.accounts.models import Account, PromiseToPay
from businessflow.accounts.promises import Part, evaluate_plan, plan_view, terms_text, validate_parts
from businessflow.outbound.decide import decide_reminder

_D = date(2026, 10, 1)


def _day(n: int) -> date:
    return _D + timedelta(days=n)


# --- judging a plan -------------------------------------------------------------------------------------------------

def test_a_part_paid_in_pieces_by_its_date_is_kept():
    states = evaluate_plan([Part(_day(10), 10_000)], made_on=_D, payments=[(_day(3), 4_000), (_day(10), 6_000)], as_of=_day(11))
    assert [s.resolution for s in states] == ["kept"]


def test_a_part_one_day_late_is_broken_no_extra_window():
    states = evaluate_plan([Part(_day(10), 10_000)], made_on=_D, payments=[(_day(11), 10_000)], as_of=_day(11))
    assert [s.resolution for s in states] == ["broken"]


def test_a_part_whose_date_is_today_is_still_open_not_broken():
    states = evaluate_plan([Part(_day(10), 10_000)], made_on=_D, payments=[], as_of=_day(10))
    assert [s.resolution for s in states] == [None]


def test_a_plan_broken_at_its_second_part_voids_the_rest():
    parts = [Part(_day(5), 5_000), Part(_day(10), 5_000), Part(_day(15), 5_000)]
    states = evaluate_plan(parts, made_on=_D, payments=[(_day(4), 5_000), (_day(9), 2_000)], as_of=_day(11))
    assert [s.resolution for s in states] == ["kept", "broken", "void"]
    assert states[1].paid_toward == 2_000


def test_money_paid_before_the_promise_does_not_count_toward_it():
    states = evaluate_plan([Part(_day(10), 10_000)], made_on=_day(2), payments=[(_day(1), 10_000)], as_of=_day(11))
    assert [s.resolution for s in states] == ["broken"]


def test_the_view_shows_what_is_left_of_the_next_part():
    rows = [
        PromiseToPay(made_on=_D, promised_date=_day(5), promised_amount=5_000, plan_id="P-1", part_no=1, parts_total=2),
        PromiseToPay(made_on=_D, promised_date=_day(10), promised_amount=7_500, plan_id="P-1", part_no=2, parts_total=2),
    ]
    view = plan_view(rows, payments=[(_day(2), 6_000)], as_of=_day(6))
    assert view["status"] == "on_track"
    assert [p["status"] for p in view["parts"]] == ["kept", "open"]
    assert view["next_part"] == {"date": _day(10).isoformat(), "amount_left": 6_500.0}


def test_terms_are_written_out_in_full():
    assert terms_text([Part(date(2026, 10, 25), 10_000), Part(date(2026, 11, 2), 12_500)]) == (
        "₹10,000 by 25 Oct 2026, then ₹12,500 by 2 Nov 2026")


@pytest.mark.parametrize("parts, message", [
    ([Part(_day(i + 1), 100) for i in range(5)], "1 to 4 parts"),
    ([Part(_day(-1), 100)], "in the past"),
    ([Part(_day(91), 100)], "more than 90 days"),
    ([Part(_day(5), 100), Part(_day(5), 100)], "must come after"),
    ([Part(_day(5), 0)], "more than zero"),
    ([Part(_day(5), 60_000)], "more than this account could owe"),
])
def test_a_plan_that_is_not_a_real_promise_is_refused(parts, message):
    with pytest.raises(ValueError, match=message):
        validate_parts(parts, today=_D, ceiling=50_000)


# --- reminders while a promise runs ---------------------------------------------------------------------------------

def _account_with_plan(due_in_days: int, part_in_days: int) -> Account:
    return Account(
        account_id="TEST-0001", borrower_name="Test", business_name="Test", phone_number="+919800000000",
        language_preference="en", loan_type="Working Capital Loan", principal_amount=100_000, emi_amount=10_000,
        tenure_months=12, months_remaining=6, emi_due_date=_day(due_in_days), nach_mandate_active=True,
        dispute_open=False, risk_tier="low",
        promises=[PromiseToPay(made_on=_D, promised_date=_day(part_in_days), promised_amount=10_000, plan_id="P-1")],
    )


def test_an_open_promise_replaces_the_overdue_follow_up():
    account = _account_with_plan(due_in_days=-10, part_in_days=5)  # 10 days past due, promised in 5
    assert decide_reminder(account, as_of=_D) is None


@pytest.mark.parametrize("days_before", [1, 0])
def test_the_borrower_is_reminded_of_their_own_promise_the_day_before_and_on_the_day(days_before):
    account = _account_with_plan(due_in_days=-10, part_in_days=days_before)
    reminder = decide_reminder(account, as_of=_D)
    assert (reminder.kind, reminder.days) == ("promise_due", days_before)


def test_a_broken_promise_ends_the_pause():
    account = _account_with_plan(due_in_days=-10, part_in_days=-1)
    account.promises[0].kept, account.promises[0].resolution = False, "broken"
    assert decide_reminder(account, as_of=_D).kind == "follow_up"


# --- the store, against real Postgres -------------------------------------------------------------------------------

def test_a_payment_that_completes_a_part_keeps_it_at_once(reseed_accounts):
    today = store.current_date()
    store.add_promise_plan("BF-1001", [Part(today + timedelta(days=3), 2_000)], made_on=today, made_via="ops")

    result = store.record_payment("BF-1001", 2_000, payment_date=today, apply_extra_to_next=False)

    assert [(r["part_no"], r["resolution"]) for r in result["promises_resolved"]] == [(1, "kept")]
    assert store.get_account_or_raise("BF-1001").promises[-1].kept is True


def test_a_plan_broken_at_part_two_counts_as_one_broken_promise(reseed_accounts):
    today = store.current_date()
    made = today - timedelta(days=20)
    parts = [Part(today - timedelta(days=15), 1_000), Part(today - timedelta(days=5), 1_000), Part(today + timedelta(days=5), 1_000)]
    store.add_promise_plan("BF-1001", parts, made_on=made, made_via="ops")
    store.record_payment("BF-1001", 1_000, payment_date=today - timedelta(days=16), apply_extra_to_next=False)

    store.resolve_promises()

    account = store.get_account_or_raise("BF-1001")
    assert [p.resolution for p in account.promises] == ["kept", "broken", "void"]
    assert account.broken_promise_count() == 1


def test_resolving_twice_changes_nothing_the_second_time(reseed_accounts):
    today = store.current_date()
    store.add_promise_plan("BF-1001", [Part(today - timedelta(days=2), 1_000)], made_on=today - timedelta(days=5), made_via="ops")

    assert len(store.resolve_promises()) >= 1
    assert [r for r in store.resolve_promises() if r["account_id"] == "BF-1001"] == []
