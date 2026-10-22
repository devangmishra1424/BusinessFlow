"""The late fee as a real charge: when it is assessed, how a payment is applied, waivers, and every surface agreeing.

Pure tests of accounts/dues.py's allocate_payment first (no database), then the store and the APIs against real
Postgres (no LLM): assess_late_fees, record_payment's order (EMI first, then the fee), the within-grace waiver,
waive_charge, the ops waive endpoint, get_payment_status and the pay page's breakdown.
"""

import os
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from businessflow.accounts import dues, store
from businessflow.accounts.models import Account, Charge
from businessflow.accounts.policy import GRACE_PERIOD_DAYS, LATE_FEE_FLAT_AMOUNT

TODAY = date(2026, 10, 22)


def _account(due: date, emi: float = 22_000, credit: float = 0.0, months: int = 20, charges=()) -> Account:
    return Account(
        account_id="BF-T", borrower_name="T", business_name="T", phone_number="+910000000000", language_preference="en",
        loan_type="Test", principal_amount=500_000, emi_amount=emi, tenure_months=36, months_remaining=months,
        emi_due_date=due, nach_mandate_active=False, dispute_open=False, risk_tier="low", pending_emi_credit=credit,
        charges=list(charges),
    )


def _fee(charge_id: int, for_due: date, paid: float = 0.0, status: str = "due") -> Charge:
    return Charge(id=charge_id, kind="late_fee", for_due_date=for_due, amount=500.0, amount_paid=paid, assessed_on=for_due, status=status)


# --- allocate_payment: the order a payment is applied in (pure) ---

class TestAllocate:
    overdue = TODAY - timedelta(days=19)

    def test_emi_and_fee_exactly_pays_both_nothing_extra(self):
        a = dues.allocate_payment(_account(self.overdue, charges=[_fee(1, self.overdue)]), 22_500, TODAY)
        assert (a.to_emi, a.emi_retired, a.to_charges, a.extra, a.short_of_emi) == (22_000, True, ((1, 500.0),), 0.0, 0.0)

    def test_the_emi_alone_retires_the_emi_and_leaves_the_fee_owed(self):
        a = dues.allocate_payment(_account(self.overdue, charges=[_fee(1, self.overdue)]), 22_000, TODAY)
        assert a.emi_retired and a.to_charges == () and a.extra == 0

    def test_short_of_an_overdue_emi_pays_toward_the_emi_never_the_fee(self):
        a = dues.allocate_payment(_account(self.overdue, charges=[_fee(1, self.overdue)]), 10_000, TODAY)
        assert (a.to_emi, a.emi_retired, a.short_of_emi, a.to_charges) == (10_000, False, 12_000, ())

    def test_before_the_next_emi_is_due_the_open_fee_comes_first(self):
        next_due = TODAY + timedelta(days=12)
        account = _account(next_due, charges=[_fee(1, self.overdue)])
        only_fee = dues.allocate_payment(account, 500, TODAY)
        assert (only_fee.to_charges, only_fee.to_emi, only_fee.short_of_emi, only_fee.extra) == (((1, 500.0),), 0.0, 0.0, 0.0)
        both = dues.allocate_payment(account, 22_500, TODAY)
        assert both.to_charges == ((1, 500.0),) and both.emi_retired and both.extra == 0

    def test_a_fee_can_be_paid_in_part_and_oldest_fees_first(self):
        older, newer = self.overdue - timedelta(days=31), self.overdue
        account = _account(TODAY + timedelta(days=5), charges=[_fee(1, older, paid=200), _fee(2, newer)])
        a = dues.allocate_payment(account, 450, TODAY)
        assert a.to_charges == ((1, 300.0), (2, 150.0)) and a.extra == 0

    def test_a_repaid_loan_with_a_fee_left_pays_the_fee_then_keeps_the_rest_as_extra(self):
        a = dues.allocate_payment(_account(self.overdue, months=0, charges=[_fee(1, self.overdue)]), 800, TODAY)
        assert (a.to_emi, a.to_charges, a.extra) == (0.0, ((1, 500.0),), 300.0)

    def test_paid_and_waived_fees_are_not_owed(self):
        account = _account(self.overdue, charges=[_fee(1, self.overdue, paid=500, status="paid"), _fee(2, self.overdue, status="waived")])
        assert dues.charges_due(account) == 0
        assert dues.amount_due_now(account) == 22_000

    def test_amount_due_now_is_the_emi_less_credit_plus_open_fees(self):
        assert dues.amount_due_now(_account(self.overdue, credit=2_000, charges=[_fee(1, self.overdue)])) == 20_500


# --- against real Postgres ---

db = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set -- these tests hit real Postgres")


@pytest.fixture
def overdue_account():
    """A real account whose current EMI was due 19 days ago (past the grace period), torn down afterwards."""
    acct, _ = store.create_account(
        borrower_name="Fee Test Borrower", business_name="Fee Test Co", phone_number="+919800033333",
        language_preference="en", loan_type="Test Loan", principal_amount=500_000, emi_amount=22_000,
        tenure_months=36, emi_due_date=store.current_date() - timedelta(days=19), nach_mandate_active=False, risk_tier="medium",
    )
    try:
        yield acct
    finally:
        conn = store.get_connection()
        for table in ("payment_tokens", "payment_history", "events", "escalations", "disputes", "promises"):
            conn.execute(f"delete from {table} where account_id = %s", (acct.account_id,))
        conn.execute("delete from accounts where account_id = %s", (acct.account_id,))


def _charges(account_id):
    return store.get_account_or_raise(account_id).charges


@db
class TestAssess:
    def test_a_fee_is_assessed_once_for_the_missed_emi_however_often_it_runs(self, overdue_account):
        assert store.assess_late_fees(account_id=overdue_account.account_id) == 1
        assert store.assess_late_fees(account_id=overdue_account.account_id) == 0
        assert store.assess_late_fees() >= 0  # the all-accounts form adds nothing more for this account
        [fee] = _charges(overdue_account.account_id)
        assert (fee.kind, fee.amount, fee.for_due_date, fee.status) == ("late_fee", LATE_FEE_FLAT_AMOUNT, overdue_account.emi_due_date, "due")

    def test_no_fee_within_the_grace_period(self, overdue_account):
        within = overdue_account.emi_due_date + timedelta(days=GRACE_PERIOD_DAYS)
        assert store.assess_late_fees(within, overdue_account.account_id) == 0
        assert store.assess_late_fees(within + timedelta(days=1), overdue_account.account_id) == 1


@db
class TestRecordPayment:
    def test_emi_first_then_the_fee_stays_owed_until_paid(self, overdue_account):
        aid = overdue_account.account_id
        first = store.record_payment(aid, 22_000)
        assert (first["kind"], first["charges_paid"], first["months_remaining"]) == ("regular", 0, 35)
        [fee] = _charges(aid)
        assert fee.status == "due"
        assert dues.charges_due(store.get_account_or_raise(aid)) == 500

        second = store.record_payment(aid, 500)
        assert (second["kind"], second["charges_paid"], second["late_fee_paid"], second["months_remaining"]) == ("charge", 500, 500, 35)
        [fee] = _charges(aid)
        assert (fee.status, fee.amount_paid) == ("paid", 500)
        history = store.get_account_or_raise(aid).payment_history
        assert [(p.kind, p.charges_paid) for p in history] == [("regular", 0), ("charge", 500)]

    def test_emi_plus_fee_in_one_payment_pays_both(self, overdue_account):
        result = store.record_payment(overdue_account.account_id, 22_500)
        assert (result["kind"], result["charges_paid"], result["late_fee_paid"], result["pending_emi_credit"]) == ("regular", 500, 500, 0.0)
        assert [c.status for c in _charges(overdue_account.account_id)] == ["paid"]

    def test_a_payment_dated_within_the_grace_period_removes_a_fee_already_charged(self, overdue_account):
        aid = overdue_account.account_id
        store.assess_late_fees(account_id=aid)
        paid_on = overdue_account.emi_due_date + timedelta(days=2)
        result = store.record_payment(aid, 22_000, payment_date=paid_on)
        assert result["late_fee_waived_paid_in_grace"] is True
        [fee] = _charges(aid)
        assert (fee.status, fee.waived_by) == ("waived", "system")
        assert paid_on.isoformat() in fee.waived_reason

    def test_a_short_payment_of_an_overdue_emi_still_needs_the_credit_decision_and_never_touches_the_fee(self, overdue_account):
        with pytest.raises(store.ExtraPaymentDecisionRequiredError):
            store.record_payment(overdue_account.account_id, 10_000)
        result = store.record_payment(overdue_account.account_id, 10_000, apply_extra_to_next=True)
        assert (result["kind"], result["charges_paid"], result["pending_emi_credit"]) == ("extra_applied", 0, 10_000)
        assert [c.status for c in _charges(overdue_account.account_id)] == ["due"]


@db
class TestWaive:
    def test_waiving_records_who_and_why_and_cannot_happen_twice(self, overdue_account):
        aid = overdue_account.account_id
        store.assess_late_fees(account_id=aid)
        [fee] = _charges(aid)
        waived = store.waive_charge(aid, fee.id, "Asha (ops)", "Bank holiday delayed the NACH debit")
        assert (waived.status, waived.waived_by, waived.waived_reason, waived.outstanding) == ("waived", "Asha (ops)", "Bank holiday delayed the NACH debit", 0.0)
        assert dues.charges_due(store.get_account_or_raise(aid)) == 0
        with pytest.raises(store.ChargeNotOpenError):
            store.waive_charge(aid, fee.id, "Asha (ops)", "again")

    def test_a_charge_of_another_account_is_not_found(self, overdue_account):
        store.assess_late_fees(account_id=overdue_account.account_id)
        [fee] = _charges(overdue_account.account_id)
        with pytest.raises(store.ChargeNotFoundError):
            store.waive_charge("BF-0000", fee.id, "Asha (ops)", "wrong account")


@db
class TestSurfaces:
    def test_get_payment_status_lists_the_fee_and_includes_it_in_what_is_due(self, overdue_account):
        from businessflow.tools.account_tools import get_payment_status
        status = get_payment_status(overdue_account.account_id)
        assert (status["late_fee_applicable"], status["late_fee_amount"], status["amount_due_now"]) == (True, 500, 22_500)
        assert status["charges"] == [{"kind": "late_fee", "for_emi_due_date": overdue_account.emi_due_date.isoformat(), "amount": 500.0, "amount_still_owed": 500.0}]

        store.record_payment(overdue_account.account_id, 22_000)  # EMI paid, fee left: still owed, next EMI not due yet
        after = get_payment_status(overdue_account.account_id)
        assert (after["days_past_due"], after["late_fee_amount"], after["amount_due_now"]) == (0, 500, 22_500)

    def test_the_ops_waive_endpoint_needs_a_name_and_reason_tells_the_borrower_and_refuses_twice(self, overdue_account, monkeypatch):
        monkeypatch.setenv("OPS_API_KEY", os.environ.get("OPS_API_KEY") or "local-test-key")
        from businessflow.ops import api as ops_api
        client = TestClient(ops_api.app)
        headers = {"X-API-Key": os.environ["OPS_API_KEY"]}
        aid = overdue_account.account_id
        detail = client.get(f"/accounts/{aid}", headers=headers).json()
        [fee] = [c for c in detail["charges"] if c["status"] == "due"]
        url = f"/accounts/{aid}/charges/{fee['id']}/waive"

        assert client.post(url, headers=headers, json={"waived_by": "A", "reason": "no"}).status_code == 422
        assert client.post(f"/accounts/{aid}/charges/999999999/waive", headers=headers, json={"waived_by": "Asha", "reason": "goodwill"}).status_code == 404
        ok = client.post(url, headers=headers, json={"waived_by": "Asha", "reason": "First delay in 18 months"})
        assert ok.status_code == 200 and ok.json()["charge"]["status"] == "waived"
        assert client.post(url, headers=headers, json={"waived_by": "Asha", "reason": "again please"}).status_code == 409

        messages = store.get_borrower_messages(aid)
        assert any(m["kind"] == "fee" and "removed the late fee of ₹500" in m["message"] and "First delay in 18 months" in m["message"] for m in messages)
        assert client.get(f"/accounts/{aid}", headers=headers).json()["amount_due_now"] == 22_000

    def test_the_pay_page_says_what_the_payment_covers_before_confirming(self, overdue_account):
        from businessflow.channels import browser_api
        client = TestClient(browser_api.app)
        token = store.create_payment_token(overdue_account.account_id, 22_500)
        info = client.get(f"/pay/{token}/info").json()
        assert info["late_fee_due"] == 500
        b = info["breakdown"]
        assert (b["to_emi"], b["emi_retired"], b["extra"], b["short_of_emi"]) == (22_000, True, 0, 0)
        assert [c["amount"] for c in b["to_charges"]] == [500] and b["to_charges"][0]["label"].startswith("Late fee for the ")
        confirmed = client.post(f"/pay/{token}/confirm", json={}).json()
        assert (confirmed["late_fee_paid"], confirmed["kind"]) == (500, "regular")
