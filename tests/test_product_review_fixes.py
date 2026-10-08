"""Regression tests for the defects found in the product review (see the commit
message): each one pins a behaviour that was wrong, using the real product code
and hand-built accounts. Nothing here needs a database, a network or Groq -- the
few functions that write (store.record_payment, outbound.run) are exercised
through an in-memory stand-in for the one boundary they cross (the DB
connection), so the logic under test is the real logic.
"""

import asyncio
import dataclasses
from datetime import date, datetime, timedelta, timezone

import pytest

from businessflow.accounts import dues, escalation_kinds as kinds, store
from businessflow.accounts.models import Account, Escalation, PaymentRecord, PromiseToPay
from businessflow.ops.flags import compute_flags
from businessflow.outbound.decide import decide_reminder
from businessflow.outbound.risk_engine import RiskTier, ToneStrategy, calculate_account_risk

TODAY = date(2026, 10, 8)


def _account(**overrides) -> Account:
    defaults = dict(
        account_id="BF-TEST", borrower_name="Test Borrower", business_name="Test Biz", phone_number="+919800000000",
        language_preference="en", loan_type="Working Capital Loan", principal_amount=500_000, emi_amount=22_000,
        tenure_months=36, months_remaining=20, emi_due_date=TODAY + timedelta(days=2), nach_mandate_active=True,
        dispute_open=False, risk_tier="low",
    )
    defaults.update(overrides)
    return Account(**defaults)


class _FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params=None):
        self.sink.append((" ".join(sql.split()), params))
        return self


def _pay(monkeypatch, account: Account, amount: float, payment_date: date = TODAY, **kw):
    """Runs the REAL store.record_payment against `account` with the DB
    connection replaced by an in-memory recorder; returns (result, account as
    the row would look afterwards, the SQL it issued)."""
    sink = []
    monkeypatch.setattr(store, "get_connection", lambda: _FakeConn(sink))
    monkeypatch.setattr(store, "get_account_or_raise", lambda _id: account)
    result = store.record_payment(account.account_id, amount, payment_date=payment_date, **kw)
    after = dataclasses.replace(
        account, principal_amount=result["principal_amount"], emi_amount=result["emi_amount"],
        months_remaining=result["months_remaining"], emi_due_date=date.fromisoformat(result["next_emi_due_date"]),
        pending_emi_credit=result["pending_emi_credit"],
    )
    return result, after, sink


# ---------------------------------------------------------------------------
# A fully repaid loan must never be chased
# ---------------------------------------------------------------------------


def test_the_final_emi_leaves_a_phantom_due_date_but_nothing_is_ever_due_on_it(monkeypatch):
    account = _account(months_remaining=1, emi_due_date=TODAY + timedelta(days=2))

    result, repaid, _ = _pay(monkeypatch, account, 22_000)

    assert result["months_remaining"] == 0
    assert repaid.emi_due_date > TODAY  # record_payment still rolls the date forward a month

    for offset in (-3, 0, 5, 20):  # heads-up window, due date, past grace, long after
        as_of = repaid.emi_due_date + timedelta(days=offset)
        assert decide_reminder(repaid, as_of) is None
        assert [f.label for f in compute_flags(repaid, as_of)] == []
        assert repaid.days_past_due(as_of) == 0
        assert dues.amount_due_now(repaid, as_of) == 0.0


def test_a_loan_with_months_left_is_still_chased_normally():
    account = _account(months_remaining=3, emi_due_date=TODAY - timedelta(days=5))

    assert decide_reminder(account, TODAY).kind == "follow_up"
    assert [f.label for f in compute_flags(account, TODAY)] == ["overdue"]


# ---------------------------------------------------------------------------
# One definition of "due now"
# ---------------------------------------------------------------------------


def test_amount_due_now_is_emi_less_credit_plus_the_late_fee_once_it_applies():
    within_grace = _account(emi_due_date=TODAY - timedelta(days=3))  # exactly at the grace boundary: no fee yet
    overdue = _account(emi_due_date=TODAY - timedelta(days=4))
    credited = _account(emi_due_date=TODAY - timedelta(days=10), pending_emi_credit=9_500)
    fully_credited = _account(emi_due_date=TODAY + timedelta(days=1), pending_emi_credit=30_000)

    assert dues.amount_due_now(within_grace, TODAY) == 22_000
    assert dues.amount_due_now(overdue, TODAY) == 22_500
    assert dues.amount_due_now(credited, TODAY) == 12_500 + 500
    assert dues.emi_due_now(fully_credited) == 0.0  # credit can exceed the EMI; never negative


def test_reminder_amounts_follow_the_reminder_kind_not_a_clock():
    account = _account(pending_emi_credit=2_000)

    assert dues.reminder_amounts(account, "heads_up") == (20_000, 0.0, 20_000)
    assert dues.reminder_amounts(account, "due_now") == (20_000, 0.0, 20_000)
    assert dues.reminder_amounts(account, "follow_up") == (20_000, 500.0, 20_500)


# ---------------------------------------------------------------------------
# record_payment: prepayment schemes and the late fee
# ---------------------------------------------------------------------------


def test_reduce_tenure_with_a_small_extra_still_retires_the_emi_just_paid(monkeypatch):
    account = _account(account_id="BF-1003", emi_amount=35_000, principal_amount=800_000, months_remaining=11)

    result, after, _ = _pay(monkeypatch, account, 35_500, payment_scheme="reduce_tenure")

    # Before the fix: months_remaining stayed 11 (the paid EMI vanished from the
    # schedule while the due date advanced) and the extra was silently dropped.
    assert result["months_remaining"] == 10
    assert result["kind"] == "overpayment_applied"  # less than one whole EMI of extra -> kept as credit, not "shortened"
    assert result["pending_emi_credit"] == 500.0


def test_reduce_tenure_removes_whole_emis_and_keeps_the_remainder_as_credit(monkeypatch):
    account = _account(emi_amount=22_000, months_remaining=20, principal_amount=500_000)

    result, _, _ = _pay(monkeypatch, account, 22_000 + 50_000, payment_scheme="reduce_tenure")

    # 20 months left, this EMI retires one (19), 50,000 pays two whole EMIs of 22,000 (17), 6,000 is carried.
    assert result["kind"] == "principal_prepayment_reduce_tenure"
    assert result["months_remaining"] == 17
    assert result["pending_emi_credit"] == 6_000.0
    assert result["principal_amount"] == 500_000 - 44_000
    assert result["emi_amount"] == 22_000


def test_reduce_tenure_never_removes_more_months_than_remain(monkeypatch):
    account = _account(emi_amount=1_000, months_remaining=3, principal_amount=10_000)

    result, _, _ = _pay(monkeypatch, account, 1_000 + 50_000, payment_scheme="reduce_tenure")

    assert result["months_remaining"] == 0
    assert result["pending_emi_credit"] == 50_000 - 2_000  # only two EMIs were left to prepay; the rest is not lost


def test_reduce_emi_applies_the_whole_extra_to_the_balance_that_is_actually_left(monkeypatch):
    account = _account(emi_amount=22_000, months_remaining=20, principal_amount=500_000)

    result, _, _ = _pay(monkeypatch, account, 22_000 + 50_000, payment_scheme="reduce_emi")

    months_after = 19
    assert result["kind"] == "principal_prepayment_reduce_emi"
    assert result["months_remaining"] == months_after
    # value is conserved: (remaining EMIs x new EMI) + the extra == remaining EMIs x old EMI
    assert result["emi_amount"] * months_after + 50_000 == pytest.approx(22_000 * months_after, abs=0.1 * months_after)
    assert result["pending_emi_credit"] == 0.0


def test_paying_emi_plus_the_late_fee_is_a_payment_not_a_prepayment(monkeypatch):
    account = _account(account_id="BF-1003", emi_amount=35_000, months_remaining=11, emi_due_date=TODAY - timedelta(days=20))

    result, after, _ = _pay(monkeypatch, account, 35_500)

    assert result["kind"] == "regular"
    assert result["late_fee_paid"] == 500.0
    assert result["pending_emi_credit"] == 0.0  # the fee must not come back as credit toward next month
    assert result["months_remaining"] == 10
    assert after.days_past_due(TODAY) == 0


@pytest.mark.parametrize("scheme", ["credit_next_emi", "reduce_emi", "reduce_tenure"])
def test_the_late_fee_is_recognised_whatever_scheme_the_page_sent(monkeypatch, scheme):
    account = _account(emi_amount=35_000, months_remaining=11, emi_due_date=TODAY - timedelta(days=20))

    result, _, _ = _pay(monkeypatch, account, 35_500, payment_scheme=scheme)

    assert result["kind"] == "regular" and result["late_fee_paid"] == 500.0


def test_an_extra_of_the_same_size_is_a_real_extra_when_no_fee_applies(monkeypatch):
    account = _account(emi_amount=35_000, months_remaining=11, emi_due_date=TODAY - timedelta(days=2))  # inside grace

    result, _, _ = _pay(monkeypatch, account, 35_500)

    assert result["kind"] == "overpayment_applied"
    assert result["late_fee_paid"] == 0.0
    assert result["pending_emi_credit"] == 500.0


def test_a_larger_deliberate_extra_on_an_overdue_account_is_still_a_prepayment(monkeypatch):
    account = _account(emi_amount=35_000, months_remaining=11, emi_due_date=TODAY - timedelta(days=20))

    result, _, _ = _pay(monkeypatch, account, 35_000 + 5_000)  # not EMI + fee: the borrower chose to pay more

    assert result["kind"] == "overpayment_applied"
    assert result["late_fee_paid"] == 0.0
    assert result["pending_emi_credit"] == 5_000.0


def test_a_part_payment_still_needs_an_explicit_decision_and_does_not_retire_the_cycle(monkeypatch):
    account = _account(emi_amount=35_000, months_remaining=11, emi_due_date=TODAY - timedelta(days=20))

    with pytest.raises(store.ExtraPaymentDecisionRequiredError):
        _pay(monkeypatch, account, 24_500)

    result, after, _ = _pay(monkeypatch, account, 24_500, apply_extra_to_next=True)
    assert result["kind"] == "extra_applied"
    assert result["months_remaining"] == 11
    assert dues.emi_due_now(after) == 10_500  # what is still due for THIS EMI


# ---------------------------------------------------------------------------
# Tone: strikes alone never make a current borrower a "final notice"
# ---------------------------------------------------------------------------


def _strikes(n: int):
    return [
        PromiseToPay(made_on=TODAY - timedelta(days=200 - i * 30), promised_date=TODAY - timedelta(days=195 - i * 30),
                     promised_amount=10_000, kept=False)
        for i in range(n)
    ]


def test_old_broken_promises_do_not_make_a_current_borrowers_heads_up_a_final_notice():
    history = [PaymentRecord(date=TODAY - timedelta(days=d), amount=22_000, on_time=True) for d in (120, 90, 60, 30)]
    account = _account(promises=_strikes(2), payment_history=history, emi_due_date=TODAY + timedelta(days=2))

    profile = calculate_account_risk(account, TODAY)

    assert decide_reminder(account, TODAY).kind == "heads_up"
    assert profile.recommended_tone not in (ToneStrategy.FINAL_NOTICE, ToneStrategy.FIRM_URGENT)
    assert profile.tier not in (RiskTier.CRITICAL, RiskTier.HIGH)


def test_two_broken_promises_on_a_delinquent_account_still_escalate_the_tone():
    account = _account(promises=_strikes(2), emi_due_date=TODAY - timedelta(days=15))

    profile = calculate_account_risk(account, TODAY)

    assert profile.tier == RiskTier.CRITICAL and profile.recommended_tone == ToneStrategy.FINAL_NOTICE


def test_one_broken_promise_alone_does_not_force_a_firm_tone():
    account = _account(promises=_strikes(1), emi_due_date=TODAY + timedelta(days=2))

    assert calculate_account_risk(account, TODAY).recommended_tone == ToneStrategy.EMPATHETIC


def test_risk_engine_defaults_to_the_same_utc_today_as_the_rest_of_the_product(monkeypatch):
    from businessflow.outbound import risk_engine

    monkeypatch.setattr(risk_engine, "current_date", lambda: TODAY)
    account = _account(emi_due_date=TODAY - timedelta(days=10))

    assert calculate_account_risk(account).factors[0].startswith("10 days past due")


# ---------------------------------------------------------------------------
# Escalation kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason, changes, expected",
    [
        (f"{kinds.GUARDRAIL_REASON_PREFIX} amount(s) not from any real tool result: [12500.0]", None, kinds.SYSTEM),
        (kinds.CHRONIC_DELINQUENCY_REASON, None, kinds.SYSTEM),
        (f"{kinds.BROKEN_PROMISE_PATTERN_PREFIX} -- 2 broken promises on record", None, kinds.SYSTEM),
        ("Repeated failed contact attempt over the phone", None, kinds.SYSTEM),
        ("SUSPECTED FRAUD/IDENTITY CLAIM -- she never took this loan", None, kinds.FRAUD),
        (kinds.CLOSURE_CERTIFICATE_REASON, None, kinds.CLOSURE),
        ("Borrower requested a human agent from the dashboard", None, kinds.CALLBACK),
        ("Borrower requested a human agent directly via /agent", None, kinds.CALLBACK),
        ("Borrower agreed to extend tenure by 3 month(s)", {"type": "extend_tenure"}, kinds.RESTRUCTURING),
        # agent-written text is in the borrower's language: never guessed at
        ("ग्रेवेंस: सेवा के बारे में असंतोष", None, kinds.REQUEST),
        ("customer asked for a human directly", None, kinds.REQUEST),
    ],
)
def test_classify_escalation(reason, changes, expected):
    assert kinds.classify_escalation(reason, changes) == expected


def test_fraud_sorts_ahead_of_everything_else_and_system_notices_last():
    ranked = sorted(kinds.URGENCY_RANK, key=kinds.URGENCY_RANK.get)

    assert ranked[0] == kinds.FRAUD
    assert ranked[-1] == kinds.SYSTEM


def test_the_producers_use_the_same_strings_the_classifier_matches():
    from businessflow.outbound import run
    from businessflow.tools import escalation_tools

    assert run._CHRONIC_DELINQUENCY_REASON == kinds.CHRONIC_DELINQUENCY_REASON
    assert escalation_tools._CLOSURE_CERTIFICATE_REASON == kinds.CLOSURE_CERTIFICATE_REASON


# ---------------------------------------------------------------------------
# What the borrower sees (browser_api builders)
# ---------------------------------------------------------------------------


def _escalation(i, reason, status="queued_for_human", changes=None, resolution=None, hours_ago=1):
    created = datetime(2026, 10, 8, 12, tzinfo=timezone.utc) - timedelta(hours=hours_ago)
    return Escalation(
        escalation_id=f"ESC-{i:04d}", account_id="BF-TEST", reason=reason, status=status, created_at=created,
        resolved_at=None if status == "queued_for_human" else created + timedelta(minutes=30),
        proposed_changes=changes, resolution_reason=resolution,
    )


def test_the_borrower_never_sees_system_tickets_or_raw_staff_wording():
    from businessflow.channels.browser_api import _build_requests

    escalations = [
        _escalation(1, f"{kinds.GUARDRAIL_REASON_PREFIX} amount(s) not from any real tool result: [12500.0]"),
        _escalation(2, kinds.CHRONIC_DELINQUENCY_REASON),
        _escalation(3, f"{kinds.BROKEN_PROMISE_PATTERN_PREFIX} -- 2 broken promises on record"),
        _escalation(4, "Borrower requested a human agent from the dashboard", hours_ago=5),
    ]

    requests = _build_requests(escalations, [])

    assert [r["id"] for r in requests] == ["ESC-0004"]
    assert requests[0]["title"] == "Call-back request"
    blob = " ".join(str(v) for r in requests for v in r.values())
    assert "Guardrail" not in blob and "Chronically" not in blob and "Broken promise pattern" not in blob


def test_requests_use_borrower_terms_for_status_and_carry_the_humans_outcome():
    from businessflow.channels.browser_api import _build_requests

    terms = {"type": "extend_tenure", "extra_months": 3, "new_months_remaining": 25, "new_emi_amount": 24_640.0}
    escalations = [
        _escalation(1, "x", status="approved", changes=terms, hours_ago=10),
        _escalation(2, "x", status="rejected", changes=terms, resolution="Account is current enough.", hours_ago=20),
        _escalation(3, "Borrower requested a human agent from the dashboard", status="approved", hours_ago=30),
        _escalation(4, kinds.CLOSURE_CERTIFICATE_REASON, status="approved", hours_ago=40),
    ]

    by_id = {r["id"]: r for r in _build_requests(escalations, [])}

    assert by_id["ESC-0001"]["status_label"] == "Approved" and "new EMI ₹24,640.00 over 25 months" in by_id["ESC-0001"]["detail"]
    assert by_id["ESC-0002"]["status_label"] == "Not approved" and by_id["ESC-0002"]["status"] == "declined"
    assert by_id["ESC-0002"]["outcome"] == "Account is current enough."
    assert by_id["ESC-0003"]["status_label"] == "Closed"  # a hand-off is "closed", not "approved"
    assert by_id["ESC-0004"]["status_label"] == "Approved"


def test_disputes_appear_in_the_borrowers_requests_with_their_outcome():
    from businessflow.channels.browser_api import _build_requests

    opened = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    disputes = [
        {"reason": "An incorrectly applied late fee", "status": "resolved", "opened_at": opened,
         "resolved_at": opened + timedelta(days=2), "resolution_note": "Fee reviewed; it stands."},
        {"reason": "I already paid on the 3rd", "status": "open", "opened_at": opened + timedelta(days=5),
         "resolved_at": None, "resolution_note": None},
    ]

    requests = _build_requests([_escalation(1, "Borrower requested a human agent from the dashboard", hours_ago=1)], disputes)

    assert [r["kind"] for r in requests] == ["callback", "dispute", "dispute"]  # newest first
    open_dispute, resolved_dispute = requests[1], requests[2]
    assert open_dispute["status_label"] == "Under review" and open_dispute["outcome"] is None
    assert resolved_dispute["status_label"] == "Resolved" and resolved_dispute["outcome"] == "Fee reviewed; it stands."


class _Flag:
    def __init__(self, label):
        self.label, self.reason = label, "staff-toned reason"


def test_overdue_banner_states_the_fee_as_applied_not_as_avoidable():
    from businessflow.channels.browser_api import _build_warnings

    status = {"days_past_due": 20, "late_fee_applicable": True, "late_fee_amount": 500.0, "broken_promise_count": 2}
    texts = {w["label"]: w["text"] for w in _build_warnings([_Flag("overdue"), _Flag("disputed"), _Flag("broken_promises")], status)}

    assert "20 days overdue" in texts["overdue"] and "late fee of ₹500 has been added" in texts["overdue"]
    assert "avoid" not in texts["overdue"]
    assert "open dispute" in texts["disputed"] and "reviewing" in texts["disputed"]
    assert "2 earlier payment promises" in texts["broken_promises"]
    assert "on record" not in texts["broken_promises"]


def test_timeline_labels_grace_and_overdue_differently_and_knows_the_new_payment_kinds():
    from businessflow.channels.browser_api import _build_emi_timeline

    history = [
        PaymentRecord(date=TODAY - timedelta(days=60), amount=22_000, on_time=True, kind="principal_prepayment_reduce_emi"),
        PaymentRecord(date=TODAY - timedelta(days=30), amount=72_000, on_time=True, kind="principal_prepayment_reduce_tenure"),
    ]
    in_grace = _account(months_remaining=2, emi_due_date=TODAY - timedelta(days=3), payment_history=history)
    overdue = _account(months_remaining=2, emi_due_date=TODAY - timedelta(days=9))

    grace_rows = _build_emi_timeline(in_grace, 3)
    overdue_rows = _build_emi_timeline(overdue, 9)

    assert "EMI lowered" in grace_rows[0]["label"] and "loan shortened" in grace_rows[1]["label"]
    assert grace_rows[2]["label"] == "Due 3d ago — within the grace period"
    assert overdue_rows[0]["label"] == "Overdue — 9d past due"
    assert overdue_rows[1]["label"] == "Scheduled"


# ---------------------------------------------------------------------------
# What ops says to the borrower
# ---------------------------------------------------------------------------


def test_only_a_real_restructuring_says_good_news_and_it_says_what_is_still_due(monkeypatch):
    from businessflow.ops.api import _approval_message

    monkeypatch.setattr(store, "current_date", lambda: TODAY)
    result = {"account_id": "BF-TEST", "new_months_remaining": 25, "new_emi_amount": 24_640.0}
    overdue = _account(emi_due_date=TODAY - timedelta(days=9))
    current = _account(emi_due_date=TODAY + timedelta(days=9))

    on_overdue = _approval_message(kinds.RESTRUCTURING, result, overdue)
    on_current = _approval_message(kinds.RESTRUCTURING, result, current)

    assert on_overdue.startswith("Good news") and "25 months remaining" in on_overdue and "₹24,640.00" in on_overdue
    assert "overdue instalment is still due" in on_overdue
    assert "still due" not in on_current


@pytest.mark.parametrize("kind", [kinds.CALLBACK, kinds.CLOSURE, kinds.REQUEST, kinds.FRAUD])
def test_closing_any_other_ticket_never_says_good_news_or_approved_for_a_complaint(kind):
    from businessflow.ops.api import _approval_message

    message = _approval_message(kind, {"account_id": "BF-TEST"}, None)

    assert message and "Good news" not in message
    if kind in (kinds.REQUEST, kinds.FRAUD):
        assert "approved" not in message  # a grievance was once told "your request has been approved"


def test_a_system_ticket_never_messages_the_borrower():
    from businessflow.ops.api import _approval_message, _rejection_message

    assert _approval_message(kinds.SYSTEM, {"account_id": "BF-TEST"}, None) is None
    assert _rejection_message(kinds.SYSTEM, "whatever") is None


def test_a_rejected_restructuring_says_what_the_borrower_can_still_do():
    from businessflow.ops.api import _rejection_message

    message = _rejection_message(kinds.RESTRUCTURING, "Already two EMIs behind")

    assert "more time could not be approved" in message and "Already two EMIs behind" in message
    assert "still pay what's due" in message


# ---------------------------------------------------------------------------
# Reminders: right amount, right words
# ---------------------------------------------------------------------------


class _Captured:
    def __init__(self):
        self.messages = None


def _fake_groq(monkeypatch, captured: _Captured):
    from businessflow.outbound import compose

    class _Completions:
        def create(self, **kwargs):
            captured.messages = kwargs["messages"]
            return type("C", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": " ok "})()})()]})()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(compose, "groq_client", lambda: _Client())


def test_a_follow_up_quotes_the_fee_and_the_total_it_asks_for(monkeypatch):
    from businessflow.outbound.compose import compose_message
    from businessflow.outbound.decide import OutboundReminder

    captured = _Captured()
    _fake_groq(monkeypatch, captured)
    account = _account(emi_amount=35_000, pending_emi_credit=0, emi_due_date=TODAY - timedelta(days=20))

    compose_message(account, OutboundReminder("BF-TEST", "follow_up", 20))

    user_facts = captured.messages[1]["content"]
    assert "35,000 rupees is now 20 day(s) past due" in user_facts
    assert "late fee of 500 rupees applies" in user_facts and "35,500 rupees in total is due now" in user_facts


def test_a_heads_up_quotes_what_is_actually_due_after_credit(monkeypatch):
    from businessflow.outbound.compose import compose_message
    from businessflow.outbound.decide import OutboundReminder

    captured = _Captured()
    _fake_groq(monkeypatch, captured)
    account = _account(emi_amount=45_000, pending_emi_credit=9_500)

    compose_message(account, OutboundReminder("BF-TEST", "heads_up", 3))

    assert "EMI of 35,500 rupees is due in 3 day(s)" in captured.messages[1]["content"]


def test_the_reminder_prompt_forbids_threats_even_for_the_firmest_tone(monkeypatch):
    from businessflow.outbound.compose import compose_message
    from businessflow.outbound.decide import OutboundReminder

    captured = _Captured()
    _fake_groq(monkeypatch, captured)
    profile = calculate_account_risk(_account(promises=_strikes(2), emi_due_date=TODAY - timedelta(days=20)), TODAY)
    assert profile.recommended_tone == ToneStrategy.FINAL_NOTICE

    compose_message(_account(), OutboundReminder("BF-TEST", "follow_up", 20), risk_profile=profile)

    system_prompt = captured.messages[0]["content"]
    assert "never threatening and never abusive" in system_prompt
    assert "never threaten" in system_prompt  # the final-notice tone line itself


def _stub_outbound_run(monkeypatch, account: Account, kind: str):
    from businessflow.outbound import run
    from businessflow.outbound.decide import OutboundReminder

    minted, sent = [], []
    monkeypatch.setattr(run, "decide_reminders", lambda ids=None: [OutboundReminder(account.account_id, kind, 20)])
    monkeypatch.setattr(run, "_already_sent_today", lambda *_: False)
    monkeypatch.setattr(run.store, "get_account_or_raise", lambda _id: account)
    monkeypatch.setattr(run, "escalate_to_human", lambda *a, **k: None)
    monkeypatch.setattr(run, "compose_message", lambda *a, **k: "reminder text")
    monkeypatch.setattr(run, "generate_payment_link", lambda aid, amount: minted.append(amount) or {"payment_link": "http://x/pay/t"})
    monkeypatch.setattr(run, "send_reminder", lambda aid, kind, msg, url, amount: sent.append(amount) or False)
    return run, minted, sent


def test_the_reminder_link_and_button_are_minted_for_what_is_due_not_the_bare_emi(monkeypatch):
    account = _account(emi_amount=35_000, pending_emi_credit=5_000, emi_due_date=TODAY - timedelta(days=20))
    run, minted, sent = _stub_outbound_run(monkeypatch, account, "follow_up")

    result = run.run_daily_outbound_pass()

    assert minted == [30_000 + 500] and sent == [30_500]  # EMI less credit, plus the late fee
    assert len(result) == 1


def test_no_reminder_is_sent_when_credit_already_covers_this_months_emi(monkeypatch):
    account = _account(emi_amount=22_000, pending_emi_credit=30_000, emi_due_date=TODAY + timedelta(days=2))
    run, minted, sent = _stub_outbound_run(monkeypatch, account, "heads_up")

    result = run.run_daily_outbound_pass()

    assert result == [] and minted == [] and sent == []


# ---------------------------------------------------------------------------
# Telegram: default language and default pay amount
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preference, expected", [("hi", "hi"), ("en", "en"), ("hinglish", "en")])
def test_telegram_defaults_to_the_accounts_own_language(monkeypatch, preference, expected):
    from businessflow.channels import telegram_bot

    monkeypatch.setattr(telegram_bot, "_language_choice", {})
    monkeypatch.setattr(telegram_bot.store, "get_account", lambda _id: _account(language_preference=preference))

    assert telegram_bot._language_for(1, "BF-TEST") == expected


def test_an_explicit_language_choice_beats_the_account_preference(monkeypatch):
    from businessflow.channels import telegram_bot

    monkeypatch.setattr(telegram_bot, "_language_choice", {1: "en"})
    monkeypatch.setattr(telegram_bot.store, "get_account", lambda _id: _account(language_preference="hi"))

    assert telegram_bot._language_for(1, "BF-TEST") == "en"


def test_telegram_language_for_an_unknown_account_is_english(monkeypatch):
    from businessflow.channels import telegram_bot

    monkeypatch.setattr(telegram_bot, "_language_choice", {})
    monkeypatch.setattr(telegram_bot.store, "get_account", lambda _id: None)

    assert telegram_bot._language_for(1, "BF-NOPE") == "en"
    assert telegram_bot._language_for(1, None) == "en"


def test_pay_with_no_amount_pays_what_is_due_now(monkeypatch):
    from businessflow.channels import telegram_bot

    account = _account(emi_amount=35_000, pending_emi_credit=0, emi_due_date=TODAY - timedelta(days=20))
    monkeypatch.setattr(telegram_bot, "_sessions", {7: {"account_id": "BF-TEST", "language": "en", "messages": []}})
    monkeypatch.setattr(telegram_bot.store, "get_account_or_raise", lambda _id: account)
    monkeypatch.setattr(telegram_bot.store, "current_date", lambda: TODAY)
    monkeypatch.setattr(telegram_bot.store, "log_event", lambda *a, **k: None)
    minted = []
    monkeypatch.setattr(
        telegram_bot, "generate_payment_link",
        lambda aid, amount: minted.append(amount) or {"payment_link": "http://x/pay/token"},
    )

    reply = asyncio.run(telegram_bot._run_pay(7, None))

    assert minted == [35_500.0]
    assert "₹35,500.00" in reply and "http://x/pay/token" in reply


def test_pay_with_no_amount_says_so_when_nothing_is_due(monkeypatch):
    from businessflow.channels import telegram_bot

    account = _account(months_remaining=0)
    monkeypatch.setattr(telegram_bot, "_sessions", {7: {"account_id": "BF-TEST", "language": "en", "messages": []}})
    monkeypatch.setattr(telegram_bot.store, "get_account_or_raise", lambda _id: account)
    monkeypatch.setattr(telegram_bot.store, "current_date", lambda: TODAY)
    monkeypatch.setattr(telegram_bot, "generate_payment_link", lambda *a: pytest.fail("must not mint a link for nothing"))

    reply = asyncio.run(telegram_bot._run_pay(7, None))

    assert "Nothing is due" in reply


def test_pay_with_an_explicit_amount_is_unchanged(monkeypatch):
    from businessflow.channels import telegram_bot

    monkeypatch.setattr(telegram_bot, "_sessions", {7: {"account_id": "BF-TEST", "language": "en", "messages": []}})
    monkeypatch.setattr(telegram_bot.store, "log_event", lambda *a, **k: None)
    minted = []
    monkeypatch.setattr(
        telegram_bot, "generate_payment_link",
        lambda aid, amount: minted.append(amount) or {"payment_link": "http://x/pay/token"},
    )

    reply = asyncio.run(telegram_bot._run_pay(7, ["5000"]))

    assert minted == [5000.0] and "₹5,000.00" in reply
