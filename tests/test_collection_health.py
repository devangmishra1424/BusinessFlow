"""Collection health (accounts/health.py) and the one computed risk (outbound/risk_engine.py) as ops see them. Pure
numbers first on hand-built accounts, then the ops endpoints against real Postgres.
"""

import os
from datetime import date, timedelta

from businessflow.accounts.health import collection_efficiency, cure_rate, dpd_buckets, promises_kept
from businessflow.accounts.models import Account, Charge, PaymentRecord, PromiseToPay
from businessflow.outbound.risk_engine import RiskTier, calculate_account_risk

_TODAY = date(2026, 10, 24)


def _account(account_id="A-1", dpd=0, **kw) -> Account:
    base = dict(
        account_id=account_id, borrower_name="B", business_name="Biz", phone_number="+919800000000",
        language_preference="en", loan_type="WC", principal_amount=100_000, emi_amount=10_000, tenure_months=12,
        months_remaining=6, emi_due_date=_TODAY - timedelta(days=dpd) if dpd else _TODAY + timedelta(days=5),
        nach_mandate_active=True, dispute_open=False, risk_tier="low",
    )
    base.update(kw)
    return Account(**base)


def test_every_live_account_lands_in_exactly_one_bucket():
    accounts = [_account("A", 0), _account("B", 1), _account("C", 30), _account("D", 31), _account("E", 95),
                _account("F", 0, months_remaining=0)]  # F is repaid: not on the book
    buckets = {b["key"]: b["account_ids"] for b in dpd_buckets(accounts, _TODAY)}
    assert buckets == {"current": ["A"], "1_30": ["B", "C"], "31_60": ["D"], "61_90": [], "90_plus": ["E"]}


def test_collection_efficiency_is_money_in_over_money_in_plus_owed():
    paid = _account("A", 0, payment_history=[PaymentRecord(_TODAY - timedelta(days=3), 10_000, True)])
    owing = _account("B", 10)  # owes its 10,000 EMI now
    ce = collection_efficiency([paid, owing], _TODAY)
    assert (ce["collected"], ce["still_owed"], ce["value"]) == (10_000, 10_000, 0.5)


def test_a_rate_with_nothing_to_measure_is_none_not_zero():
    assert promises_kept([_account()], _TODAY)["value"] is None
    assert cure_rate([_account()], _TODAY)["value"] is None


def test_promises_kept_counts_plans_not_parts_and_skips_replaced_ones():
    d = _TODAY - timedelta(days=10)
    kept_plan = [PromiseToPay(d, d, 100, kept=True, plan_id="P-1", part_no=1),
                 PromiseToPay(d, d + timedelta(days=2), 100, kept=True, plan_id="P-1", part_no=2)]
    broken_plan = [PromiseToPay(d, d, 100, kept=False, plan_id="P-2", resolution="broken"),
                   PromiseToPay(d, d + timedelta(days=3), 100, plan_id="P-2", part_no=2, resolution="void")]
    replaced = [PromiseToPay(d, d + timedelta(days=1), 100, plan_id="P-3", resolution="void")]
    pk = promises_kept([_account(promises=kept_plan + broken_plan + replaced)], _TODAY)
    assert (pk["kept"], pk["broken"], pk["value"]) == (1, 1, 0.5)


def test_cure_rate_is_late_accounts_now_current():
    fee = Charge(1, "late_fee", _TODAY - timedelta(days=40), 500, 500, _TODAY - timedelta(days=36), "paid")
    cured = _account("A", 0, charges=[fee])
    still_late = _account("B", 20, charges=[Charge(2, "late_fee", _TODAY - timedelta(days=20), 500, 0, _TODAY - timedelta(days=16), "due")])
    cr = cure_rate([cured, still_late], _TODAY)
    assert (cr["cured"], cr["went_late"], cr["cured_ids"]) == (1, 2, ["A"])


def test_the_risk_lists_every_point_it_adds():
    profile = calculate_account_risk(_account(dpd=20, risk_tier="high", dispute_open=True), _TODAY)
    assert sum(f["points"] for f in profile.breakdown) == profile.score
    labels = [f["label"] for f in profile.breakdown]
    assert "Rated high risk at onboarding" in labels and "Active dispute flagged on account" in labels


def test_an_open_promise_lowers_the_risk():
    plain = calculate_account_risk(_account(dpd=20), _TODAY)
    promised = calculate_account_risk(_account(dpd=20, promises=[PromiseToPay(_TODAY, _TODAY + timedelta(days=3), 10_000, plan_id="P-9")]), _TODAY)
    assert promised.score == plain.score - 10
    assert ("A promise to pay is open and on track", -10.0) in [(f["label"], f["points"]) for f in promised.breakdown]


def test_the_onboarding_rating_alone_never_makes_a_current_account_high_risk():
    assert calculate_account_risk(_account(risk_tier="high"), _TODAY).tier == RiskTier.LOW


# --- through the ops API --------------------------------------------------------------------------------------------

def test_ops_sees_the_computed_risk_and_the_health_numbers(reseed_accounts):
    from fastapi.testclient import TestClient

    from businessflow.ops.api import app

    client = TestClient(app)
    headers = {"X-API-Key": os.environ["OPS_API_KEY"]}

    accounts = client.get("/accounts", headers=headers).json()
    for a in accounts:
        assert a["risk"]["tier"] == a["risk_tier"]
        assert a["onboarding_risk_tier"] in ("low", "medium", "high")
        assert abs(sum(f["points"] for f in a["risk"]["factors"]) - a["risk"]["score"]) < 0.01 or a["risk"]["score"] in (0, 100)

    health = client.get("/portfolio/health", headers=headers).json()
    assert 0 < sum(b["accounts"] for b in health["dpd_buckets"]) <= len(accounts)  # repaid loans are not on the book
    assert {"collection_efficiency", "promises_kept", "cure_rate", "containment"} <= set(health)
    for key in ("collection_efficiency", "promises_kept", "cure_rate", "containment"):
        assert health[key]["how"]
