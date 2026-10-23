"""A payment within the grace period is on time (accounts/policy.py GRACE_PERIOD_DAYS; policy BFF-POL-02 section 3):
the payment record, and everything that reads it -- the borrower's history, the risk score, Telegram, ops voice --
must not call it late. Found 2026-10-23 building the demo: a payment two days after the due date was stored as late.
"""

from datetime import timedelta

from businessflow.accounts import store
from businessflow.accounts.policy import GRACE_PERIOD_DAYS


def test_paid_on_the_last_day_of_grace_is_on_time_and_the_day_after_is_late(reseed_accounts):
    for account_id, days_after_due, expected in (("BF-1001", GRACE_PERIOD_DAYS, True), ("BF-1002", GRACE_PERIOD_DAYS + 1, False)):
        account = store.get_account_or_raise(account_id)
        paid_on = account.emi_due_date + timedelta(days=days_after_due)
        result = store.record_payment(account_id, account.emi_amount - account.pending_emi_credit, payment_date=paid_on)
        assert result["on_time"] is expected, (account_id, days_after_due)
        assert store.get_account_or_raise(account_id).payment_history[-1].on_time is expected
