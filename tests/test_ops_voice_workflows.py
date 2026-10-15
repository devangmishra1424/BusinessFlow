"""Ops voice workflows (ops/voice_workflows.py) and the structured answers they share with the rest of the voice
channel (rows, suggestions, the help menu), over test_ops_voice.py's stand-in for the dashboard's data -- no
database, no model. What these pin down: the briefing counts what the cards count; the call list leaves out an
account whose dispute is open; a step offers the right next actions for its item; a call outcome and the promise
made on it go on one card, each figure checked against what was said; a case closed meanwhile is passed over; and
"next" / "skip" with nothing running is said to be so, not acted on."""

from datetime import date, timedelta

import pytest

from businessflow.ops import voice, voice_reply, voice_workflows
from businessflow.ops.voice import parse_plan
from tests.test_ops_voice import ACCOUNTS, NOW, TODAY, FakeData, _ask, _case, _detail


def _details(**per_account):
    details = {a["account_id"]: _detail(a) for a in ACCOUNTS}
    for account_id, extra in per_account.items():
        details[account_id].update(extra)
    return details


# --- the menu and structured answers --------------------------------------------------------------------------------

def test_the_help_menu_is_grouped_and_every_entry_either_runs_or_fills_the_box():
    reply = _ask("help")
    groups = [s.group for s in reply.suggestions]
    assert groups[0] == "Start here" and {"Work through", "Ask", "Do (you confirm each)"} <= set(groups)
    for s in reply.suggestions:
        assert (s.tool is None) != (s.prompt is None), s.label
        if s.tool:
            parse_plan(s.tool, s.arguments or {})  # every runnable entry is a valid plan


def test_an_overdue_list_comes_with_rows_to_tap_and_the_call_list_offered():
    reply = _ask("show_accounts", group="overdue")
    assert [r.label for r in reply.rows] == ["Fatima Khan", "Ravi Kumar", "Imran Khan"]
    assert reply.rows[0].account_id == "BF-1003" and reply.rows[0].tone == "danger"
    assert reply.rows[0].value == "45 days · ₹13,250 due"
    assert reply.suggestions[0].tool == "start_call_list"


def test_an_inbox_row_opens_the_case_on_the_borrowers_profile():
    inbox = [_case("D-1", "payment_claim", "BF-1003", "Fatima Khan", opened_days_ago=5)]
    reply = _ask("show_inbox", data=FakeData(inbox=inbox))
    assert (reply.rows[0].account_id, reply.rows[0].tab, reply.rows[0].tone) == ("BF-1003", "cases", "danger")
    assert [s.tool for s in reply.suggestions] == ["start_inbox", "start_inbox"]  # all, then only payment claims


def test_an_accounts_summary_has_a_phone_number_that_dials():
    reply = _ask("account_details", account="BF-1002")
    phone = next(r for r in reply.rows if r.label == "Phone")
    assert phone.href == "tel:+919812345001"
    assert any(s.prompt and "record a payment from Asha Verma" in s.prompt for s in reply.suggestions)


# --- the briefing ---------------------------------------------------------------------------------------------------

def test_the_briefing_counts_calls_and_cases_and_offers_where_to_start():
    inbox = [_case("D-1", "payment_claim", "BF-1003", "Fatima Khan", opened_days_ago=5),
             _case("ESC-0002", "restructuring", "BF-1001", "Ravi Kumar", opened_days_ago=1)]
    reply = _ask("daily_briefing", data=FakeData(inbox=inbox))
    # 3 overdue, but Fatima's dispute is open: 2 to call (Ravi 12,000 + Imran 8,000.5 in EMIs)
    assert "2 borrowers to call, with ₹20,000 overdue and 2 open cases in the inbox, 1 waiting over 3 days." in reply.say
    calls = reply.rows[0]
    assert (calls.label, calls.value, calls.tool) == ("Calls to make", "2", "start_call_list")
    assert "1 disputed account left out" in calls.detail
    assert [s.tool for s in reply.suggestions] == ["start_call_list", "start_inbox", "start_inbox"]


def test_a_quiet_day_says_so():
    reply = _ask("daily_briefing", data=FakeData(accounts=[ACCOUNTS[1]]))
    assert reply.say.endswith("Nothing is overdue and the inbox is clear.")
    assert reply.suggestions == []


# --- the call list --------------------------------------------------------------------------------------------------

def test_the_call_list_is_most_overdue_first_without_an_open_dispute():
    reply = _ask("start_call_list")
    assert reply.workflow.kind == "call_list" and reply.workflow.step_tool == "call_step"
    assert [i.key for i in reply.workflow.items] == ["BF-1001", "BF-1004"]  # Fatima (disputed) left out
    assert "1 account with an open dispute or payment claim left out" in reply.say
    assert reply.screen.account_id == "BF-1001"  # the first borrower's profile opens


def test_nobody_to_call_starts_no_run():
    reply = _ask("start_call_list", data=FakeData(accounts=[ACCOUNTS[1]]))
    assert reply.workflow is None and reply.say.startswith("No one to call.")


def test_a_call_step_shows_the_number_and_offers_the_outcomes():
    reply = _ask("call_step", account="BF-1001")
    assert reply.rows[0].href == "tel:+919812345001"
    tools = [(s.label, s.tool, (s.arguments or {}).get("outcome")) for s in reply.suggestions]
    assert ("No answer", "log_call", "no_answer") in tools and ("Skip", "workflow_control", None) in tools
    assert reply.say.endswith("Tell me how the call goes.")


def test_a_borrower_already_called_today_puts_skip_first():
    details = _details(**{"BF-1001": {"call_log": [{"outcome": "no_answer", "note": None, "created_at": NOW}]}})
    reply = _ask("call_step", data=FakeData(details=details), account="BF-1001")
    assert "Already called today: no answer." in reply.say
    assert reply.suggestions[0].label == "Skip"


def test_a_call_outcome_with_a_promise_is_one_card_with_two_requests():
    reply = _ask("log_call", "reached ravi, he will pay 8000 by the 25th", screen_account_id="BF-1001",
                 account="current", outcome="reached", promised_amount=8000, promised_date="2026-10-25")
    p = reply.proposal
    assert (p.path, p.body["outcome"]) == ("/accounts/BF-1001/call-log", "reached")
    assert [(r.path, r.body) for r in p.then] == [
        ("/accounts/BF-1001/promises", {"promised_date": "2026-10-25", "promised_amount": 8000}),
    ]
    assert p.account_id == "BF-1001"  # in the call list, confirming moves on from Ravi


def test_a_promised_amount_nobody_said_is_not_put_on_the_card():
    reply = _ask("log_call", "reached ravi, he will pay by the 25th", account="BF-1001", outcome="reached",
                 promised_amount=8000, promised_date="2026-10-25")
    assert reply.kind == "clarify" and reply.proposal is None


def test_a_promise_without_a_date_is_asked_about():
    reply = _ask("log_call", "reached ravi, he will pay 8000", account="BF-1001", outcome="reached", promised_amount=8000)
    assert reply.kind == "clarify" and "By what date" in reply.say


# --- the inbox ------------------------------------------------------------------------------------------------------

def _inbox_and_details():
    claim = _case("D-7", "payment_claim", "BF-1003", "Fatima Khan", opened_days_ago=4)
    more_time = _case("ESC-0012", "restructuring", "BF-1001", "Ravi Kumar", opened_days_ago=1,
                      proposed={"type": "extend_tenure", "extra_months": 3, "new_emi_amount": 10800.0, "new_months_remaining": 21})
    details = _details(**{"BF-1003": {"cases": [claim]}, "BF-1001": {"cases": [more_time]}})
    return FakeData(details=details, inbox=[claim, more_time])


def test_working_the_inbox_walks_its_cases_in_order():
    reply = _ask("start_inbox", data=_inbox_and_details())
    assert [(i.key, i.arguments) for i in reply.workflow.items] == [
        ("D-7", {"account": "BF-1003", "case": "D-7"}), ("ESC-0012", {"account": "BF-1001", "case": "ESC-0012"}),
    ]
    assert reply.say.startswith("2 open cases, most urgent first. First, Says they already paid from Fatima Khan")


def test_only_payment_claims_when_asked():
    reply = _ask("start_inbox", data=_inbox_and_details(), kind="payment_claim")
    assert [i.key for i in reply.workflow.items] == ["D-7"]


def test_a_payment_claim_step_offers_confirm_ask_for_the_reference_and_close():
    reply = _ask("case_step", data=_inbox_and_details(), account="BF-1003", case="D-7")
    labels = [s.label for s in reply.suggestions]
    assert labels[:3] == ["Confirm the payment…", "Ask for the reference", "Close: not found…"]
    ask = reply.suggestions[1]
    # the reference request is the case card's own wording, sent as a reply on this claim -- a card like any other
    card = _ask(ask.tool, data=_inbox_and_details(), **ask.arguments)
    assert card.proposal.path == "/accounts/BF-1003/cases/D-7/reply"
    assert card.proposal.body["message"] == voice_workflows.ASK_REFERENCE_TEXT
    assert "No payment recorded on the account in the last few days." in reply.say


def test_a_request_for_more_time_step_states_the_terms_and_offers_approve():
    reply = _ask("case_step", data=_inbox_and_details(), account="BF-1001", case="ESC-0012")
    assert "new EMI ₹10,800 over 21 months" in reply.say
    assert reply.suggestions[0].tool == "approve_request"


def test_approve_it_in_an_inbox_run_means_the_case_on_screen():
    data = _inbox_and_details()
    data._details["BF-1001"]["cases"].append(_case("ESC-0013", "callback", "BF-1001", "Ravi Kumar"))
    plan = parse_plan("approve_request", {"account": "current"})
    reply = voice_reply.respond(plan, "approve it", data, "BF-1001", TODAY, NOW, screen_case="ESC-0012", workflow="inbox")
    assert reply.proposal.path == "/escalations/ESC-0012/approve"  # not "which one?" between the two


def test_close_it_on_a_request_closes_the_request_and_says_the_note_is_not_sent():
    # Found in the preview: "I called her back, close it" on a call-back was read as closing a dispute.
    callback = _case("ESC-0014", "callback", "BF-1002", "Asha Verma")
    data = FakeData(details=_details(**{"BF-1002": {"cases": [callback]}}))
    plan = parse_plan("close_case", {"account": "current", "note": "called her back"})
    reply = voice_reply.respond(plan, "I called her back, close it", data, "BF-1002", TODAY, NOW,
                                screen_case="ESC-0014", workflow="inbox")
    assert reply.proposal.path == "/escalations/ESC-0014/approve"
    assert any("isn't sent" in w for w in reply.proposal.warnings)


def test_a_case_closed_since_the_run_started_is_passed_over():
    reply = _ask("case_step", data=FakeData(), account="BF-1003", case="D-7")
    assert reply.workflow_command == "skip" and "already closed" in reply.say


# --- inside a run ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("command", ["next", "skip", "previous", "stop", "repeat"])
def test_run_commands_are_passed_to_the_page_when_a_run_is_going(command):
    plan = parse_plan("workflow_control", {"command": command})
    reply = voice_reply.respond(plan, command, FakeData(), None, TODAY, NOW, workflow="call_list")
    assert reply.workflow_command == command


def test_next_with_nothing_running_says_so_and_offers_a_start():
    reply = _ask("workflow_control", command="next")
    assert reply.kind == "clarify" and reply.workflow_command is None
    assert [s.tool for s in reply.suggestions] == ["daily_briefing", "start_call_list", "start_inbox"]


_SAMPLE_ARGS = {
    "portfolio_overview": {}, "show_accounts": {}, "show_inbox": {}, "escalation_rate": {}, "help": {}, "cannot_help": {},
    "account_details": {"account": "BF-1001"}, "open_page": {"page": "inbox"}, "daily_briefing": {},
    "start_call_list": {}, "start_inbox": {}, "workflow_control": {"command": "next"},
    "call_step": {"account": "BF-1001"}, "case_step": {"account": "BF-1001", "case": "ESC-0012"},
}


def test_no_question_or_workflow_ability_ever_proposes_an_action():
    # Covers every read-only ability: one added to voice.QUESTION_TOOLS without a sample here fails this test.
    assert set(_SAMPLE_ARGS) == voice.QUESTION_TOOLS
    for tool, args in _SAMPLE_ARGS.items():
        reply = _ask(tool, "record 12500 from ravi", data=_inbox_and_details(), **args)
        assert reply.kind != "proposal" and reply.proposal is None, tool


def test_promises_due_by_a_date_already_passed_are_not_shown_as_open():
    details = _details(**{"BF-1001": {"promises": [
        {"made_on": TODAY - timedelta(days=10), "promised_date": TODAY - timedelta(days=2), "promised_amount": 5000.0, "kept": None},
        {"made_on": TODAY, "promised_date": date(2026, 10, 25), "promised_amount": 8000.0, "kept": None},
    ]}})
    reply = _ask("call_step", data=FakeData(details=details), account="BF-1001")
    assert "They promised ₹8,000 by Sun 25 Oct." in reply.say
