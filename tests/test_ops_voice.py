"""Ops voice, without a database or a model: ops/voice.py's handling of what
the model chose, and ops/voice_reply.py's answers and confirmation cards over
a stand-in for the dashboard's data (FakeData below, the same shapes the ops
endpoints return). Endpoint wiring and real Postgres are in
test_ops_voice_api.py.

What these pin down: answers are computed from the data, never taken from the
model; an amount, phone number or drafted figure the operator didn't say is
never put on a card; an ambiguous name is asked about, not guessed; and a
question never turns into an action.
"""

import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from businessflow.ops import voice, voice_reply
from businessflow.ops.voice import NotUnderstood, parse_plan

TODAY = date(2026, 10, 15)
NOW = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)


def _summary(account_id, name, business, *, dpd=0, flags=(), risk="low", emi=10000.0, due_now=0.0, telegram=True):
    return {
        "account_id": account_id, "borrower_name": name, "business_name": business, "loan_type": "Working Capital Loan",
        "principal_amount": 300000.0, "emi_amount": emi, "days_past_due": dpd, "risk_tier": risk,
        "flags": [{"label": f, "reason": f} for f in flags], "telegram_linked": telegram, "amount_due_now": due_now,
    }


def _case(ref, kind, account_id, name, *, source=None, opened_days_ago=1, recent_payments=(), proposed=None):
    return {
        "case_ref": ref, "source": source or ("dispute" if ref.startswith("D-") else "escalation"), "kind": kind,
        "title": {"payment_claim": "Says they already paid", "restructuring": "Asks for more time",
                  "callback": "Asks to talk to a person", "dispute": "Dispute"}.get(kind, "Request"),
        "account_id": account_id, "borrower_name": name, "open": True, "status": "open",
        "opened_at": NOW - timedelta(days=opened_days_ago), "closed_at": None, "borrower_words": "I paid by UPI on Monday",
        "proposed_changes": proposed, "outcome": None, "thread": [], "can_reply": kind != "system",
        "recent_payments": list(recent_payments),
    }


ACCOUNTS = [
    _summary("BF-1001", "Ravi Kumar", "Kumar Traders", dpd=12, flags=("overdue",), risk="medium", emi=12000, due_now=12500),
    _summary("BF-1002", "Asha Verma", "Verma Textiles", risk="low"),
    _summary("BF-1003", "Fatima Khan", "Khan Kirana", dpd=45, flags=("overdue", "disputed", "broken_promises"), risk="high", emi=12500, due_now=13250, telegram=False),
    _summary("BF-1004", "Imran Khan", "Imran Auto Parts", dpd=5, flags=("overdue",), risk="high", emi=8000.5, due_now=8000.5),
]


def _detail(summary, **extra):
    return {
        **summary, "phone_number": "+919812345001", "language_preference": "en", "tenure_months": 24, "months_remaining": 14,
        "emi_due_date": TODAY - timedelta(days=summary["days_past_due"]), "nach_mandate_active": True,
        "dispute_open": "disputed" in [f["label"] for f in summary["flags"]], "pending_emi_credit": 0.0,
        "payment_history": [], "promises": [], "disputes": [], "escalations": [], "clarification_requests": [],
        "call_log": [], "cases": [], "messages": [], **extra,
    }


class FakeData:
    def __init__(self, accounts=ACCOUNTS, details=None, inbox=(), metrics=None):
        self._accounts = list(accounts)
        self._details = details or {a["account_id"]: _detail(a) for a in accounts}
        self._inbox = list(inbox)
        self._metrics = metrics or {"since_hours": 24, "event_counts": {}, "escalation_rate": 0.0}
        self.metrics_asked = []

    def accounts(self):
        return self._accounts

    def account(self, account_id):
        return self._details.get(account_id)

    def inbox(self):
        return self._inbox

    def metrics(self, since_hours):
        self.metrics_asked.append(since_hours)
        return self._metrics

    def documents(self, account_id):
        return []

    def conversation(self, account_id):
        return []

    def decision_message(self, case, account, approved, reason):
        return "APPROVED-MESSAGE" if approved else "REJECTED-MESSAGE"

    def claim_confirmed_message(self, amount, paid_on, note):
        return f"CLAIM-CONFIRMED {amount} {paid_on}"

    def case_closed_message(self, kind, note):
        return f"CLOSED {kind}"


def _ask(tool, heard="", data=None, screen_account_id=None, **arguments):
    return voice_reply.respond(parse_plan(tool, arguments), heard, data or FakeData(), screen_account_id, TODAY, NOW)


# --- understanding --------------------------------------------------------------------------------------------------

def _completion(name=None, arguments="{}"):
    calls = [SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))] if name else []
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))])


def test_understand_asks_the_voice_model_for_exactly_one_menu_choice(monkeypatch):
    sent = {}

    def fake_completion(**kwargs):
        sent.update(kwargs)
        return _completion("record_payment", json.dumps({"account": "Fatima", "amount": 12500, "paid_on": "2026-10-14"}))

    monkeypatch.setattr(voice, "_create_completion", fake_completion)
    monkeypatch.delenv("OPS_VOICE_MODEL", raising=False)
    plan = voice.understand("record twelve and a half thousand from fatima, paid yesterday", TODAY, "the Portfolio")

    assert plan.tool == "record_payment"
    assert plan.args.amount == 12500 and plan.args.paid_on == date(2026, 10, 14)
    # Its own daily allowance: not the borrower chat's model.
    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["tool_choice"] == "required" and sent["tools"] is voice.MENU
    assert "2026-10-15" in sent["messages"][1]["content"] and "the Portfolio" in sent["messages"][1]["content"]


def test_the_voice_model_can_be_changed_without_a_code_change(monkeypatch):
    sent = {}
    monkeypatch.setattr(voice, "_create_completion", lambda **kw: sent.update(kw) or _completion("help"))
    monkeypatch.setenv("OPS_VOICE_MODEL", "some/other-model")
    voice.understand("help", TODAY, "the Portfolio")
    assert sent["model"] == "some/other-model"
    # .env.example leaves it empty: that must mean the default, not a model named ""
    monkeypatch.setenv("OPS_VOICE_MODEL", "")
    voice.understand("help", TODAY, "the Portfolio")
    assert sent["model"] == "openai/gpt-oss-120b"


@pytest.mark.parametrize("completion", [
    _completion(),  # the model answered in words instead of choosing
    _completion("delete_everything"),  # not on the menu
    _completion("record_payment", "{not json"),
    _completion("record_payment", "[1, 2]"),
    _completion("log_call", json.dumps({"account": "BF-1001", "outcome": "argued"})),  # not an allowed outcome
])
def test_an_unusable_model_choice_is_not_understood_rather_than_guessed(monkeypatch, completion):
    monkeypatch.setattr(voice, "_create_completion", lambda **kw: completion)
    with pytest.raises(NotUnderstood):
        voice.understand("something", TODAY, "the Portfolio")


def test_details_the_model_leaves_blank_count_as_not_said():
    plan = parse_plan("record_payment", {"account": "BF-1001", "amount": "", "paid_on": None})
    assert plan.args.amount is None and plan.args.paid_on is None


def test_every_menu_entry_has_an_argument_model_and_the_other_way_round():
    # A workflow's steps are asked for by the page, never offered to the model.
    assert {t["function"]["name"] for t in voice.MENU} == set(voice.ARGS) - voice.PAGE_ONLY_TOOLS


def test_the_model_cannot_pick_a_step_only_the_page_asks_for(monkeypatch):
    monkeypatch.setattr(voice, "_create_completion", lambda **kw: _completion("call_step", json.dumps({"account": "BF-1001"})))
    with pytest.raises(NotUnderstood):
        voice.understand("next borrower", TODAY, "the call list")


# --- answers --------------------------------------------------------------------------------------------------------

def test_portfolio_overview_is_computed_from_the_data():
    inbox = [_case("D-1", "payment_claim", "BF-1003", "Fatima Khan")]
    reply = _ask("portfolio_overview", data=FakeData(inbox=inbox))
    assert reply.kind == "answer"
    # 3 overdue, EMIs 12000 + 12500 + 8000.5 = 32500.5, shown whole like the Overdue card
    assert reply.say == (
        "4 accounts on the books, 3 needing attention. 3 overdue, with ₹32,500 in EMIs at risk. "
        "1 open case waiting on a person. Risk mix: 2 high, 1 medium and 1 low risk."
    )
    assert reply.screen.view == "overview" and reply.screen.filter == "all"


def test_who_to_call_first_lists_the_most_overdue_first_and_shows_that_list():
    reply = _ask("show_accounts", group="overdue")
    assert reply.say.startswith("3 overdue accounts, with ₹32,500 in EMIs at risk.")
    assert "Most overdue: Fatima Khan, 45 days; Ravi Kumar, 12 days; Imran Khan, 5 days." in reply.say
    assert (reply.screen.filter, reply.screen.sort) == ("overdue", "urgency")


def test_an_empty_group_says_so():
    reply = _ask("show_accounts", data=FakeData(accounts=[ACCOUNTS[1]]), group="overdue")
    assert reply.say == "Nothing is overdue."


def test_inbox_counts_by_kind_and_says_what_has_gone_stale():
    inbox = [
        _case("D-1", "payment_claim", "BF-1003", "Fatima Khan", opened_days_ago=5),
        _case("ESC-0002", "restructuring", "BF-1001", "Ravi Kumar", opened_days_ago=1),
    ]
    reply = _ask("show_inbox", data=FakeData(inbox=inbox))
    assert reply.say.startswith("2 open cases: 1 payment claim and 1 request for more time.")
    assert "1 case has waited more than 3 days." in reply.say
    assert reply.screen.view == "inbox"


def test_escalation_rate_uses_the_window_asked_for_and_says_when_there_is_no_activity():
    data = FakeData(metrics={"since_hours": 168, "event_counts": {"tool_called": 25}, "escalation_rate": 0.12})
    reply = _ask("escalation_rate", data=data, window="7d")
    assert data.metrics_asked == [168]
    assert reply.say == "In the last 7 days, 12% of the AI assistant's 25 tool calls were handed to a person."
    assert reply.screen.since_hours == 168
    quiet = _ask("escalation_rate", window="24h")
    assert quiet.say == "No AI assistant activity in the last 24 hours."


def test_what_an_account_owes_includes_the_late_fee_from_the_data():
    reply = _ask("account_details", account="BF-1003", topic="dues")
    assert reply.say.startswith("Fatima Khan, BF-1003: ₹13,250 due now, including a ₹750 late fee.")
    assert (reply.screen.account_id, reply.screen.tab) == ("BF-1003", "overview")


# --- which account --------------------------------------------------------------------------------------------------

def test_this_account_means_the_profile_on_screen():
    reply = _ask("account_details", screen_account_id="BF-1002", account="her")
    assert reply.say.startswith("Asha Verma, BF-1002:")


def test_this_account_with_no_profile_open_is_asked_about():
    reply = _ask("account_details", account="current")
    assert reply.kind == "clarify" and "Which account?" in reply.say


def test_a_misheard_name_still_finds_the_account():
    assert _ask("account_details", account="Fatma Kahn").say.startswith("Fatima Khan, BF-1003:")


def test_two_close_names_are_asked_about_not_guessed():
    reply = _ask("account_details", account="Khan", topic="dues")
    assert reply.kind == "clarify"
    assert {c.arguments["account"] for c in reply.choices} == {"BF-1003", "BF-1004"}
    # each choice is the same request with the account settled, run later without the model
    assert all(c.tool == "account_details" and c.arguments["topic"] == "dues" for c in reply.choices)


def test_an_unknown_account_is_said_to_be_unknown():
    assert _ask("account_details", account="BF-9999").say == "I couldn't find account BF-9999."
    assert "couldn't find an account" in _ask("account_details", account="Zebediah Quill").say


# --- actions: never without what was said ---------------------------------------------------------------------------

def test_a_payment_the_operator_said_becomes_a_card_with_the_forms_own_request():
    reply = _ask("record_payment", "record 12,500 from Fatima, paid yesterday",
                 account="Fatima", amount=12500, paid_on="2026-10-14")
    assert reply.kind == "proposal"
    p = reply.proposal
    assert (p.method, p.path) == ("POST", "/accounts/BF-1003/payments")
    assert p.body == {"amount": 12500, "payment_date": "2026-10-14"}
    assert not p.credit_choice  # 12,500 covers this cycle's EMI
    assert reply.say.endswith("Press Confirm to go ahead.")


def test_an_amount_the_operator_never_said_is_not_put_on_a_card():
    reply = _ask("record_payment", "record a payment from Fatima", account="Fatima", amount=5000)
    assert reply.kind == "clarify" and reply.proposal is None


def test_no_amount_at_all_is_asked_for():
    reply = _ask("record_payment", "record a payment from Fatima", account="Fatima")
    assert reply.kind == "clarify" and "How much" in reply.say


def test_the_full_amount_due_comes_from_the_database_not_the_model():
    reply = _ask("record_payment", "record her full amount", account="BF-1003", full_amount_due=True, amount=99)
    assert reply.proposal.body["amount"] == 13250


def test_a_short_payment_asks_whether_to_credit_it_before_confirm():
    reply = _ask("record_payment", "record 5000 from Ravi", account="Ravi", amount=5000)
    assert reply.proposal.credit_choice and "apply_extra_to_next" not in reply.proposal.body
    said = _ask("record_payment", "record 5000 from Ravi and credit it", account="Ravi", amount=5000, credit_toward_next=True)
    assert not said.proposal.credit_choice and said.proposal.body["apply_extra_to_next"] is True


def test_a_payment_dated_in_the_future_is_refused():
    reply = _ask("record_payment", "record 12500 from Ravi paid tomorrow", account="Ravi", amount=12500, paid_on="2026-10-16")
    assert reply.kind == "clarify" and "future" in reply.say


def test_a_payment_like_one_recorded_days_ago_carries_a_warning():
    details = {a["account_id"]: _detail(a) for a in ACCOUNTS}
    details["BF-1001"]["payment_history"] = [{"date": TODAY - timedelta(days=2), "amount": 12000.0, "on_time": True, "kind": "regular"}]
    reply = _ask("record_payment", "record 12000 from Ravi", data=FakeData(details=details), account="Ravi", amount=12000)
    assert reply.proposal.warnings and "already recorded" in reply.proposal.warnings[0]


def test_a_message_with_a_figure_the_operator_did_not_say_is_not_proposed():
    reply = _ask("message_borrower", "tell her we got her payment", account="BF-1003",
                 message="We have received your payment of ₹5,000.")
    assert reply.kind == "clarify" and "5000" in reply.say


def test_a_message_card_shows_the_exact_text_and_needs_it_to_stay_filled_in():
    reply = _ask("message_borrower", "tell her we got her documents", account="BF-1003",
                 message="We have received your documents.")
    p = reply.proposal
    assert p.borrower_told == "We have received your documents." == p.body["message"]
    assert (p.editable, p.editable_required) == ("message", True)
    assert ("To", "Fatima Khan (BF-1003)") in [(d.label, d.value) for d in p.details]


def test_confirming_a_claim_nobody_filed_records_the_payment_without_the_drafted_note():
    # eval/ops_voice_benchmark.py's one miss: "she paid the full amount today" read as confirming a claim, with a note
    # to the borrower nobody dictated.
    reply = _ask("confirm_payment_claim", "she paid the full amount today", screen_account_id="BF-1003",
                 account="current", full_amount_due=True, note="We have received your payment in full today. Thank you.")
    p = reply.proposal
    assert (p.action, p.path) == ("record_payment", "/accounts/BF-1003/payments")
    assert p.body == {"amount": 13250, "payment_date": "2026-10-15"} and p.borrower_told is None


def test_recording_a_payment_while_a_claim_is_open_points_at_the_claim():
    details = {a["account_id"]: _detail(a) for a in ACCOUNTS}
    details["BF-1003"]["cases"] = [_case("D-7", "payment_claim", "BF-1003", "Fatima Khan")]
    reply = _ask("record_payment", "record 12500 from fatima", data=FakeData(details=details), account="BF-1003", amount=12500)
    assert any("open payment claim" in w for w in reply.proposal.warnings)


def test_confirming_a_claim_targets_that_claim_and_previews_the_borrowers_message():
    details = {a["account_id"]: _detail(a) for a in ACCOUNTS}
    details["BF-1003"]["cases"] = [_case("D-7", "payment_claim", "BF-1003", "Fatima Khan")]
    reply = _ask("confirm_payment_claim", "confirm fatima's 12500 from yesterday", data=FakeData(details=details),
                 account="BF-1003", amount=12500, paid_on="2026-10-14")
    p = reply.proposal
    assert p.path == "/accounts/BF-1003/cases/D-7/confirm-payment"
    assert p.borrower_told == "CLAIM-CONFIRMED 12500.0 2026-10-14"


def test_closing_a_payment_claim_without_a_reason_is_asked_about():
    details = {a["account_id"]: _detail(a) for a in ACCOUNTS}
    details["BF-1003"]["cases"] = [_case("D-7", "payment_claim", "BF-1003", "Fatima Khan")]
    reply = _ask("close_case", "close fatima's claim", data=FakeData(details=details), account="BF-1003")
    assert reply.kind == "clarify" and "needs a reason" in reply.say


def test_two_open_requests_are_asked_about_before_approving():
    details = {a["account_id"]: _detail(a) for a in ACCOUNTS}
    details["BF-1001"]["cases"] = [
        _case("ESC-0001", "restructuring", "BF-1001", "Ravi Kumar"),
        _case("ESC-0002", "callback", "BF-1001", "Ravi Kumar"),
    ]
    reply = _ask("approve_request", "approve ravi's request", data=FakeData(details=details), account="Ravi")
    assert reply.kind == "clarify"
    assert [c.arguments["case"] for c in reply.choices] == ["ESC-0001", "ESC-0002"]
    chosen = _ask("approve_request", "approve ravi's request", data=FakeData(details=details), **reply.choices[0].arguments)
    assert chosen.proposal.path == "/escalations/ESC-0001/approve" and chosen.proposal.borrower_told == "APPROVED-MESSAGE"


def test_a_new_phone_number_must_have_been_said_digit_for_digit():
    misheard = _ask("update_account", "change her number to nine eight one two three", account="BF-1002", phone_number="9812345678")
    assert misheard.kind == "clarify"
    said = _ask("update_account", "change her number to 98123 45678", account="BF-1002", phone_number="9812345678")
    assert said.proposal.method == "PATCH" and said.proposal.body == {"phone_number": "+919812345678"}


def test_a_message_to_a_group_goes_to_exactly_that_group():
    reply = _ask("message_group", "message everyone overdue to call us", group="overdue", message="Please call us.")
    assert reply.proposal.path == "/clarification-requests/bulk"
    assert reply.proposal.body["account_ids"] == ["BF-1003", "BF-1001", "BF-1004"]


@pytest.mark.parametrize("tool, arguments", [
    ("portfolio_overview", {}), ("show_accounts", {"group": "overdue"}), ("show_inbox", {}),
    ("escalation_rate", {"window": "3d"}), ("account_details", {"account": "BF-1003", "topic": "payments"}),
    ("open_page", {"page": "inbox"}), ("help", {}), ("cannot_help", {"reason": "weather"}),
])
def test_a_question_never_turns_into_an_action(tool, arguments):
    reply = _ask(tool, "record 12500 from fatima", **arguments)
    assert reply.kind != "proposal" and reply.proposal is None
