"""Telegram linking: one chat can hold several loans (switch with /loans), every pushed message says which loan when it
does, an account moved to a new chat leaves the old one (which is told), and ops can unlink. Against real Postgres;
Telegram sends are replaced by recorders.
"""

import asyncio
import os

import pytest

from businessflow.accounts import store


def _chat_of(account_id: str):
    return store.get_account_or_raise(account_id).telegram_chat_id


def test_a_chat_can_hold_two_loans_and_the_latest_is_active(reseed_accounts):
    store.set_telegram_chat_id("BF-1001", 700001)
    result = store.set_telegram_chat_id("BF-1002", 700001)

    assert result["previous_chat_id"] is None and set(result["loans_in_chat"]) == {"BF-1001", "BF-1002"}
    assert (_chat_of("BF-1001"), _chat_of("BF-1002")) == (700001, 700001)
    assert store.get_account_by_telegram_chat_id(700001).account_id == "BF-1002"
    assert store.touch_telegram_active("BF-1001", 700001) is True
    assert store.get_account_by_telegram_chat_id(700001).account_id == "BF-1001"
    assert store.touch_telegram_active("BF-1003", 700001) is False  # switching never links anything


def test_moving_an_account_to_a_new_chat_names_the_old_one(reseed_accounts):
    store.set_telegram_chat_id("BF-1001", 700002)
    assert store.set_telegram_chat_id("BF-1001", 700003)["previous_chat_id"] == 700002
    assert store.set_telegram_chat_id("BF-1001", 700003)["previous_chat_id"] is None  # same chat again: nothing moved


@pytest.fixture
def bot(monkeypatch):
    from businessflow.channels import telegram_bot

    sent = []

    async def fake(chat_id, text):
        sent.append((chat_id, text))
        return True

    monkeypatch.setattr(telegram_bot, "notify_telegram_chat", fake)
    telegram_bot._sessions.clear()
    telegram_bot.sent = sent
    return telegram_bot


def test_a_second_loans_credentials_add_it_to_the_chat_and_loans_switches(reseed_accounts, bot):
    assert "Verified" in bot.handle_incoming_message(810001, "BF-1001 482913")
    reply = bot.handle_incoming_message(810001, "BF-1002 716044")
    assert "2 loans" in reply and "/loans" in reply
    assert bot._sessions[810001]["account_id"] == "BF-1002"
    assert "already verified" in bot.handle_incoming_message(810001, "BF-1002 716044")
    assert "doesn't match" in bot.handle_incoming_message(810001, "BF-1003 000000")
    assert bot._sessions[810001]["account_id"] == "BF-1002"  # a wrong key changes nothing

    assert bot.switch_loan(810001, "BF-1001") == "Switched to BF-1001. What can I help you with?"
    assert bot._sessions[810001]["account_id"] == "BF-1001"
    assert "isn't linked" in bot.switch_loan(810001, "BF-1003")
    active, loans = bot.loans_in_chat(810001)
    assert active == "BF-1001" and {a.account_id for a in loans} == {"BF-1001", "BF-1002"}


def test_verifying_from_a_new_chat_tells_the_old_chat_and_closes_its_session(reseed_accounts, bot):
    assert "Verified" in bot.handle_incoming_message(800001, "BF-1001 482913")
    assert "Verified" in bot.handle_incoming_message(800002, "BF-1001 482913")

    assert [chat for chat, _ in bot.sent] == [800001] and "BF-1001" in bot.sent[0][1]
    assert 800001 not in bot._sessions and _chat_of("BF-1001") == 800002


def test_a_message_to_a_chat_with_several_loans_says_which(reseed_accounts, monkeypatch):
    from businessflow.outbound import send

    out = []

    async def fake_send(chat_id, text, *args, **kwargs):
        out.append(text)
        return True

    monkeypatch.setattr(send, "_send_telegram_message", fake_send)
    store.set_telegram_chat_id("BF-1001", 700010)
    asyncio.run(send._deliver_and_log("BF-1001", "Your EMI is due.", "reminder_sent", {"kind": "heads_up"}))
    store.set_telegram_chat_id("BF-1002", 700010)
    asyncio.run(send._deliver_and_log("BF-1001", "Your EMI is due.", "reminder_sent", {"kind": "heads_up"}))

    assert out[0] == "Your EMI is due."
    assert out[1].startswith("[BF-1001 · Cotton Threads Boutique]\n")


def test_ops_can_unlink_and_the_chat_is_told(reseed_accounts, monkeypatch):
    from fastapi.testclient import TestClient

    from businessflow.ops import api

    told = []

    async def fake(chat_id, text):
        told.append(chat_id)
        return True

    monkeypatch.setattr(api.notify, "notify_telegram_chat", fake)
    store.set_telegram_chat_id("BF-1003", 700009)
    client = TestClient(api.app)
    headers = {"X-API-Key": os.environ["OPS_API_KEY"]}

    r = client.post("/accounts/BF-1003/telegram/unlink", headers=headers).json()
    assert r == {"account_id": "BF-1003", "was_linked": True, "chat_told": True} and told == [700009]
    assert _chat_of("BF-1003") is None
    assert client.post("/accounts/BF-1003/telegram/unlink", headers=headers).json()["was_linked"] is False
    assert client.post("/accounts/BF-9999/telegram/unlink", headers=headers).status_code == 404
