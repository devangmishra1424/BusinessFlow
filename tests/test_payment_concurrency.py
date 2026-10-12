"""Payments under concurrency, against real Postgres (no LLM).

The bug these pin: redeem_payment_token checked used_at, then marked the link
used, then recorded the payment, as separate autocommit statements with no
condition on used_at. Two confirms of the same link at the same moment (a
double tap, a retry while the first request was still running) could both
pass the check and record the payment twice. Separately, record_payment read
the account unlocked and wrote it back, so two payments on one account at the
same moment both started from the same balance and one update was lost.

Each test starts its workers at the same instant with a Barrier, so the
requests really overlap.
"""

import os
import threading
from datetime import timedelta

import pytest

from businessflow.accounts import store

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


@pytest.fixture
def account():
    """A real account, fully torn down afterwards (never a seeded demo account)."""
    acct, _ = store.create_account(
        borrower_name="Concurrency Test Borrower", business_name="Concurrency Test Co", phone_number="+919800022222",
        language_preference="en", loan_type="Test Loan", principal_amount=60_000, emi_amount=3_000,
        tenure_months=20, emi_due_date=store.current_date() + timedelta(days=10), nach_mandate_active=True, risk_tier="low",
    )
    try:
        yield acct
    finally:
        conn = store.get_connection()
        conn.execute("delete from payment_tokens where account_id = %s", (acct.account_id,))
        conn.execute("delete from payment_history where account_id = %s", (acct.account_id,))
        conn.execute("delete from events where account_id = %s", (acct.account_id,))
        conn.execute("delete from accounts where account_id = %s", (acct.account_id,))


def _run_at_once(n, fn):
    """Runs fn(i) in n threads released together; returns each result or the exception it raised."""
    barrier = threading.Barrier(n)
    outcomes = [None] * n

    def worker(i):
        barrier.wait()
        try:
            outcomes[i] = fn(i)
        except Exception as e:  # collected and asserted on by the test, never swallowed
            outcomes[i] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a payment call hung"
    return outcomes


def test_simultaneous_confirms_of_one_link_record_exactly_one_payment(account):
    token = store.create_payment_token(account.account_id, 3000)

    outcomes = _run_at_once(4, lambda _i: store.redeem_payment_token(token))

    successes = [o for o in outcomes if isinstance(o, dict)]
    refusals = [o for o in outcomes if isinstance(o, store.PaymentTokenAlreadyUsedError)]
    assert len(successes) == 1, outcomes
    assert len(refusals) == 3, outcomes
    updated = store.get_account_or_raise(account.account_id)
    assert len(updated.payment_history) == 1
    assert updated.months_remaining == account.months_remaining - 1
    assert store.get_payment_token_info(token)["status"] == "used"


def test_two_payments_on_one_account_at_the_same_moment_are_both_applied(account):
    # Two different links for the same account, confirmed together: both are
    # legitimate, so both must land, one after the other (no lost update).
    tokens = [store.create_payment_token(account.account_id, 3000) for _ in range(2)]

    outcomes = _run_at_once(2, lambda i: store.redeem_payment_token(tokens[i]))

    assert all(isinstance(o, dict) for o in outcomes), outcomes
    updated = store.get_account_or_raise(account.account_id)
    assert len(updated.payment_history) == 2
    assert updated.months_remaining == account.months_remaining - 2
    assert updated.emi_due_date == store.add_one_month(store.add_one_month(account.emi_due_date))


def test_a_failure_while_recording_leaves_the_link_usable(account, monkeypatch):
    # Marking the link used and recording the payment commit together: if the
    # payment cannot be written, the link must not be burned with nothing paid.
    token = store.create_payment_token(account.account_id, 3000)

    def broken(*_args, **_kwargs):
        raise RuntimeError("simulated database failure while recording the payment")

    monkeypatch.setattr(store, "_record_payment_in", broken)
    with pytest.raises(RuntimeError):
        store.redeem_payment_token(token)
    assert store.get_payment_token_info(token)["status"] == "pending"
    assert len(store.get_account_or_raise(account.account_id).payment_history) == 0

    monkeypatch.undo()
    result = store.redeem_payment_token(token)
    assert result["amount"] == 3000.0
    assert store.get_payment_token_info(token)["status"] == "used"


def test_a_short_payment_without_a_decision_still_leaves_the_link_usable(account):
    token = store.create_payment_token(account.account_id, 1000)  # less than the 3,000 due

    with pytest.raises(store.ExtraPaymentDecisionRequiredError):
        store.redeem_payment_token(token)
    assert store.get_payment_token_info(token)["status"] == "pending"

    result = store.redeem_payment_token(token, apply_extra_to_next=True)
    assert result["kind"] == "extra_applied"
