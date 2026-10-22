"""Telegram linking: one chat speaks for one account, the chat an account moves away from is told, and ops can unlink.
Against real Postgres; the Telegram send is replaced by a recorder.
"""

import os

import pytest

from businessflow.accounts import store


def _chat_of(account_id: str):
    return store.get_account_or_raise(account_id).telegram_chat_id


def test_a_chat_that_verifies_a_second_account_leaves_the_first(reseed_accounts):
    store.set_telegram_chat_id("BF-1001", 700001)
    result = store.set_telegram_chat_id("BF-1002", 700001)

    assert result == {"previous_chat_id": None, "unlinked_accounts": ["BF-1001"]}
    assert (_chat_of("BF-1001"), _chat_of("BF-1002")) == (None, 700001)
    assert store.get_account_by_telegram_chat_id(700001).account_id == "BF-1002"


def test_moving_an_account_to_a_new_chat_names_the_old_one(reseed_accounts):
    store.set_telegram_chat_id("BF-1001", 700002)
    assert store.set_telegram_chat_id("BF-1001", 700003)["previous_chat_id"] == 700002
    assert store.set_telegram_chat_id("BF-1001", 700003)["previous_chat_id"] is None  # same chat again: nothing moved


@pytest.fixture
def sent(monkeypatch):
    from businessflow.channels import telegram_bot

    messages = []

    async def fake(chat_id, text):
        messages.append((chat_id, text))
        return True

    monkeypatch.setattr(telegram_bot, "notify_telegram_chat", fake)
    return messages


def test_verifying_from_a_new_chat_tells_the_old_chat_and_closes_its_session(reseed_accounts, sent):
    from businessflow.channels import telegram_bot

    telegram_bot._sessions.clear()
    assert "Verified" in telegram_bot.handle_incoming_message(800001, "BF-1001 482913")
    assert "Verified" in telegram_bot.handle_incoming_message(800002, "BF-1001 482913")

    assert [chat for chat, _ in sent] == [800001] and "BF-1001" in sent[0][1]
    assert 800001 not in telegram_bot._sessions and _chat_of("BF-1001") == 800002


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
    again = client.post("/accounts/BF-1003/telegram/unlink", headers=headers).json()
    assert again["was_linked"] is False
    assert client.post("/accounts/BF-9999/telegram/unlink", headers=headers).status_code == 404
