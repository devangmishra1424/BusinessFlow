"""Answering a message from our team (web and Telegram), and keeping our internal words out of messages to borrowers.
Against real Postgres; model and Telegram calls are replaced.
"""

import os
from types import SimpleNamespace

import pytest

from businessflow.accounts import store
from businessflow.outbound import compose


def _team_message(account_id: str, text: str = "Please tell us why the EMI is late.", **extra) -> int:
    store.log_event(account_id, "clarification_request_sent", {"message": text, "delivered_via_telegram": False, **extra})
    return store.get_connection().execute(
        "select id from events where account_id = %s and event_type = 'clarification_request_sent' order by id desc limit 1",
        (account_id,),
    ).fetchone()["id"]


@pytest.fixture
def web(reseed_accounts):
    from fastapi.testclient import TestClient

    from businessflow.channels.browser_api import app

    client = TestClient(app)
    conv = client.post("/conversations", json={"language": "en", "account_id": "BF-1002", "access_key": "716044"}).json()
    return client, conv["conversation_id"]


def test_a_reply_becomes_a_request_and_a_second_one_joins_it(web):
    client, cid = web
    mid = _team_message("BF-1002")

    first = client.post(f"/conversations/{cid}/messages-from-us/{mid}/reply", json={"message": "Shop was shut for Diwali."})
    assert first.status_code == 200 and first.json()["opened"] is True
    case_ref = first.json()["case_ref"]
    second = client.post(f"/conversations/{cid}/messages-from-us/{mid}/reply", json={"message": "I can pay on the 2nd."}).json()
    assert second == {"case_ref": case_ref, "opened": False}

    dash = client.get(f"/conversations/{cid}/dashboard").json()
    msg = next(m for m in dash["messages"] if m["id"] == mid)
    assert msg["replied_case_ref"] == case_ref
    request = next(r for r in dash["requests"] if r["id"] == case_ref)
    assert request["title"] == "Your reply to our message" and "Please tell us why" in request["detail"] and "Diwali" in request["detail"]
    assert [t["message"] for t in request["thread"] if t["author"] == "borrower"] == ["I can pay on the 2nd."]
    from datetime import datetime, timedelta, timezone
    sent = datetime.now(timezone(timedelta(hours=5, minutes=30)))  # quoted in the borrower's (Indian) date, not UTC's
    assert request["detail"].startswith(f"{sent.day} {sent.strftime('%b')} ")


def test_only_a_team_message_on_this_account_can_be_answered(web):
    client, cid = web
    store.log_event("BF-1002", "reminder_sent", {"message": "Your EMI is due.", "kind": "heads_up"})
    reminder = store.get_connection().execute(
        "select id from events where account_id = 'BF-1002' and event_type = 'reminder_sent' order by id desc limit 1").fetchone()["id"]
    other = _team_message("BF-1001")
    for mid in (reminder, other):
        assert client.post(f"/conversations/{cid}/messages-from-us/{mid}/reply", json={"message": "hi"}).status_code == 404


def test_a_telegram_reply_to_a_team_message_goes_to_the_team(reseed_accounts):
    from businessflow.channels import telegram_bot

    store.set_telegram_chat_id("BF-1003", 920001)
    _team_message("BF-1003", telegram_message_id=4242)

    answer = telegram_bot._reply_to_team(920001, 4242, "Payment went on the 20th, UTR 556677889900")
    assert answer.startswith("Thanks -- your reply went to our team as request ")
    assert telegram_bot._reply_to_team(920001, 9999, "hello") is None  # not a team message: the assistant answers it
    assert telegram_bot._reply_to_team(920002, 4242, "hello") is None  # another chat's message


# --- our words, not the borrower's ---------------------------------------------------------------------------------

def test_internal_words_are_found():
    assert compose.internal_terms_in("The only open flag on your account; we escalated it.") == ["flag", "escalated"]
    assert compose.internal_terms_in("Your EMI is 19 days late. Please reply.") == []


def _fake_groq(drafts, calls):
    def create(**kwargs):
        calls.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=drafts[len(calls) - 1]))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_a_draft_with_internal_words_is_rewritten_once_then_warned_about(monkeypatch):
    calls = []
    monkeypatch.setattr(compose, "groq_client", lambda: _fake_groq(["Your account is flagged.", "Your EMI is late; please reply."], calls))
    assert compose.draft_clarification_message("Arjun", "Mehta Hardware", ["19 days past due"], "ask why") == {
        "draft": "Your EMI is late; please reply.", "warnings": []}
    assert "flagged" in calls[1][-1]["content"]

    calls.clear()
    monkeypatch.setattr(compose, "groq_client", lambda: _fake_groq(["A flag is open.", "The flag remains."], calls))
    result = compose.draft_clarification_message("Arjun", "Mehta Hardware", [], "ask why")
    assert result["warnings"] and "flag" in result["warnings"][0]


def test_the_ops_draft_endpoint_passes_the_warnings_on(reseed_accounts, monkeypatch):
    from fastapi.testclient import TestClient

    from businessflow.ops import api

    monkeypatch.setattr(api, "draft_clarification_message", lambda **kw: {"draft": "d", "warnings": ["w"]})
    r = TestClient(api.app).post("/accounts/BF-1002/clarification-requests/draft", json={"operator_note": "x"},
                                 headers={"X-API-Key": os.environ["OPS_API_KEY"]})
    assert r.json() == {"draft": "d", "warnings": ["w"]}
