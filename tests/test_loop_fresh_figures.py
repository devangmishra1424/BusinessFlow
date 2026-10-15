"""The agent is told when the account changed after it last looked it up (agent/loop.py's _account_change_note).

Found live: a borrower paid through the link the agent had just sent, then asked "what is my payment history, and do I
have any dues right now?" -- and was told "21 days past due, ₹35,500 due today": the agent fetched only the history and
reused its status from before the payment. No model or database here: the store and the turn are stood in for.
"""

import json

from businessflow.agent import loop

ACCOUNT = "BF-1003"


def _conversation(with_status_lookup: bool) -> list[dict]:
    messages = [{"role": "system", "content": "prompt"}, {"role": "user", "content": "what do I owe?"}]
    if with_status_lookup:
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_payment_status", "arguments": json.dumps({"account_id": ACCOUNT})}},
        ]})
        messages.append({"role": "tool", "tool_call_id": "c1", "content": '{"days_past_due": 21}'})
        messages.append({"role": "assistant", "content": "21 days past due."})
    messages.append({"role": "user", "content": "do I have any dues right now?"})
    return messages


def _changes(monkeypatch, payments=0, restructurings=0):
    calls = []

    def fake(account_id):
        calls.append(account_id)
        return {"since": None, "payments": payments, "restructurings": restructurings}

    monkeypatch.setattr(loop.store, "changes_since_last_status_check", fake)
    return calls


def test_a_payment_after_the_last_lookup_tells_the_agent_to_look_again(monkeypatch):
    _changes(monkeypatch, payments=1)

    note = loop._account_change_note(_conversation(True), ACCOUNT)

    assert note["role"] == "system"
    assert "a payment was recorded" in note["content"]
    assert "Call get_payment_status again" in note["content"]
    assert "₹" not in note["content"]  # figures must come from a tool result, never from the note


def test_an_approved_restructuring_counts_too(monkeypatch):
    _changes(monkeypatch, restructurings=1)

    assert "restructuring was approved" in loop._account_change_note(_conversation(True), ACCOUNT)["content"]


def test_nothing_changed_means_no_note(monkeypatch):
    _changes(monkeypatch)

    assert loop._account_change_note(_conversation(True), ACCOUNT) is None


def test_no_lookup_in_this_conversation_means_no_note_and_no_database_call(monkeypatch):
    calls = _changes(monkeypatch, payments=1)

    assert loop._account_change_note(_conversation(False), ACCOUNT) is None
    assert loop._account_change_note(_conversation(True), None) is None
    assert calls == []


def test_the_note_is_in_this_turn_only_and_never_kept_in_the_conversation(monkeypatch):
    _changes(monkeypatch, payments=1)
    monkeypatch.setattr(loop.conversation_memory, "log_turn", lambda *args: None)
    seen = {}

    def fake_run_turn(conversation, verified_account_id=None, anonymous=False):
        seen["messages"] = list(conversation)
        return conversation + [{"role": "assistant", "content": "Nothing is overdue."}], "Nothing is overdue."

    monkeypatch.setattr(loop, "run_turn", fake_run_turn)
    conversation = _conversation(True)

    after, reply = loop.run_turn_with_memory(conversation, ACCOUNT)

    sent = seen["messages"]
    assert sent[-1]["content"] == "do I have any dues right now?"
    assert sent[-2]["role"] == "system" and "Account update" in sent[-2]["content"]  # right before the question
    assert not any("Account update" in (m.get("content") or "") for m in after)
    assert reply == "Nothing is overdue."


def test_a_correction_by_the_team_counts_as_a_change(monkeypatch):
    monkeypatch.setattr(
        loop.store, "changes_since_last_status_check",
        lambda account_id: {"since": None, "payments": 0, "restructurings": 0, "corrections": 1},
    )

    assert "our team corrected the account" in loop._account_change_note(_conversation(True), ACCOUNT)["content"]
