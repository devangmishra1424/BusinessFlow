"""The "what I can help with" buttons on Telegram (channels/abilities.py is the shared list): the keyboard, what a tap does,
what /start, /help and a fresh verification show, and that a database hiccup cannot stop the buttons appearing.

The routing helpers take plain arguments. The few handlers tested here are driven with minimal stand-ins for the message and
query objects (only the methods the handler calls), because what they add -- the order of the messages, which tap goes to the
assistant and which does not -- is exactly what needs checking. No database, model or Telegram network is involved."""

import asyncio
import logging
from types import SimpleNamespace

import psycopg
import pytest

from businessflow.channels import telegram_bot
from businessflow.channels.abilities import abilities_for

CHAT = 987001


class FakeMessage:
    def __init__(self):
        self.sent = []  # (text, reply_markup)
        self.parse_modes = []  # one per sent message: "HTML" for an agent reply, None for plain text
        self.voices = 0

    async def reply_text(self, text, reply_markup=None, parse_mode=None):
        self.sent.append((text, reply_markup))
        self.parse_modes.append(parse_mode)

    async def reply_voice(self, voice):
        self.voices += 1


def _callbacks(markup):
    return [button.callback_data for row in markup.inline_keyboard for button in row]


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(telegram_bot, "_sessions", {})
    monkeypatch.setattr(telegram_bot, "_language_choice", {})
    monkeypatch.setattr(telegram_bot, "_voice_preference", {})
    # An unlinked chat: no database is reached when a chat has no in-memory account.
    monkeypatch.setattr(telegram_bot.store, "get_account_by_telegram_chat_id", lambda chat_id: None)


def _verify(chat_id=CHAT, language="en"):
    telegram_bot._sessions[chat_id] = {"account_id": "BF-1001", "language": language, "messages": []}


# ---- the keyboard ----------------------------------------------------------------------------------------------------
def test_the_keyboard_has_one_button_per_row_carrying_only_the_entry_id_and_ends_with_more_questions():
    markup = telegram_bot._ability_keyboard("en", verified=True)

    entries = abilities_for("en", True)
    assert [len(row) for row in markup.inline_keyboard] == [1] * len(entries)
    assert _callbacks(markup) == [f"ab:{a['id']}" for a in entries]
    assert _callbacks(markup)[-1] == "ab:more"
    assert [row[0].text for row in markup.inline_keyboard] == [a["label"] for a in entries]


@pytest.mark.parametrize("verified", [True, False])
@pytest.mark.parametrize("language", ["en", "hi"])
def test_every_callback_fits_telegrams_64_byte_limit(language, verified):
    for data in _callbacks(telegram_bot._ability_keyboard(language, verified)):
        assert len(data.encode("utf-8")) <= 64


# ---- what a tap resolves to ------------------------------------------------------------------------------------------
def test_a_tap_in_a_verified_chat_finds_its_entry_in_the_chats_language():
    _verify(language="hi")

    language, ability = telegram_bot._ability_for_tap(CHAT, "status")

    assert language == "hi"
    assert ability["prompt"] == abilities_for("hi", True)[0]["prompt"]


def test_a_button_an_unverified_chat_may_not_use_finds_nothing():
    language, ability = telegram_bot._ability_for_tap(CHAT, "status")  # e.g. left over from before a /reset

    assert language == "en" and ability is None
    assert telegram_bot._ability_for_tap(CHAT, "policy")[1]["prompt"]  # the general ones still work
    assert telegram_bot._ability_for_tap(CHAT, "made-up-id")[1] is None


def test_a_database_failure_falls_back_to_the_general_buttons_and_says_so_in_the_log(monkeypatch, caplog):
    def _down(chat_id):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(telegram_bot.store, "get_account_by_telegram_chat_id", _down)

    with caplog.at_level(logging.WARNING, logger=telegram_bot.logger.name):
        language, verified = telegram_bot._ability_context(CHAT)

    assert (language, verified) == ("en", False)
    assert any("could not look up" in r.getMessage() for r in caplog.records)  # visible, not swallowed


def test_an_unexpected_error_is_not_swallowed(monkeypatch):
    def _bug(chat_id):
        raise RuntimeError("not a database problem")

    monkeypatch.setattr(telegram_bot.store, "get_account_by_telegram_chat_id", _bug)

    with pytest.raises(RuntimeError):
        telegram_bot._ability_context(CHAT)


# ---- the handlers ----------------------------------------------------------------------------------------------------
def _tap(ability_id, monkeypatch, reply="the assistant's answer"):
    calls = []

    async def fake_agent_reply(context, chat_id, text):
        calls.append(text)
        return reply

    monkeypatch.setattr(telegram_bot, "_agent_reply", fake_agent_reply)
    query = SimpleNamespace(message=FakeMessage())
    asyncio.run(telegram_bot._run_ability_tap(query, None, CHAT, ability_id))
    return query.message, calls


def test_tapping_a_prompt_button_shows_the_question_then_the_assistants_answer(monkeypatch):
    _verify()

    message, calls = _tap("status", monkeypatch)

    prompt = abilities_for("en", True)[0]["prompt"]
    assert calls == [prompt]  # went through the normal assistant turn, exactly as if typed
    assert [text for text, _ in message.sent] == [f"You asked: {prompt}", "the assistant's answer"]


def test_tapping_more_questions_is_answered_without_an_assistant_turn(monkeypatch):
    message, calls = _tap("more", monkeypatch)

    assert calls == []
    assert len(message.sent) == 1 and "I'll say so" in message.sent[0][0]


def test_tapping_how_to_verify_tells_an_unverified_chat_and_costs_no_assistant_turn(monkeypatch):
    message, calls = _tap("verify", monkeypatch)

    assert calls == []
    assert "6-digit access key" in message.sent[0][0]


def test_a_stale_button_gets_a_pointer_to_help_and_never_reaches_the_assistant(monkeypatch):
    message, calls = _tap("status", monkeypatch)  # unverified chat, account entry

    assert calls == []
    assert len(message.sent) == 1 and "/help" in message.sent[0][0]


def test_a_tap_in_a_hindi_chat_is_echoed_and_answered_in_hindi(monkeypatch):
    _verify(language="hi")

    message, calls = _tap("history", monkeypatch)

    assert message.sent[0][0].startswith("आपने पूछा: ")
    assert calls == [abilities_for("hi", True)[2]["prompt"]]


def test_with_voice_replies_on_the_answer_is_also_spoken(monkeypatch):
    _verify()
    telegram_bot._voice_preference[CHAT] = True
    spoken = []

    async def fake_speak(message, chat_id, reply):
        spoken.append((message, reply))

    monkeypatch.setattr(telegram_bot, "_send_spoken_reply", fake_speak)

    message, _ = _tap("status", monkeypatch)

    assert spoken == [(message, "the assistant's answer")]


def test_start_shows_the_general_buttons_to_an_unverified_chat_and_promises_nothing_big(monkeypatch):
    monkeypatch.setattr(telegram_bot, "_ability_context", lambda chat_id: ("en", False))
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=CHAT), message=FakeMessage())

    asyncio.run(telegram_bot.on_start(update, SimpleNamespace(args=[])))

    (text, markup), = update.message.sent
    assert _callbacks(markup) == ["ab:policy", "ab:verify", "ab:more"]
    assert "Tap an option below" in text
    assert "everything" not in text.lower()


def test_help_shows_the_intro_and_the_chats_buttons(monkeypatch):
    monkeypatch.setattr(telegram_bot, "_ability_context", lambda chat_id: ("en", True))
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=CHAT), message=FakeMessage())

    asyncio.run(telegram_bot.on_help(update, None))

    (text, markup), = update.message.sent
    assert text.startswith("Here are a few things")
    assert "ab:status" in _callbacks(markup) and _callbacks(markup)[-1] == "ab:more"


def test_typing_credentials_that_verify_the_chat_shows_the_full_list_once(monkeypatch):
    async def verifying_reply(context, chat_id, text):
        _verify(chat_id)  # what a successful verification does to the in-memory session
        return "Verified -- you're chatting about account BF-1001."

    monkeypatch.setattr(telegram_bot, "_agent_reply", verifying_reply)
    monkeypatch.setattr(telegram_bot, "_ability_context", lambda chat_id: ("en", True))
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=CHAT), message=FakeMessage())
    update.message.text = "BF-1001 482913"

    asyncio.run(telegram_bot.on_text_message(update, None))
    (text, markup), = update.message.sent
    assert text.startswith("Verified")
    assert "ab:status" in _callbacks(markup)

    # a later message in the same, now-verified chat gets no buttons attached
    async def plain_reply(context, chat_id, text):
        return "Your EMI is due on the 5th."

    monkeypatch.setattr(telegram_bot, "_agent_reply", plain_reply)
    update2 = SimpleNamespace(effective_chat=SimpleNamespace(id=CHAT), message=FakeMessage())
    update2.message.text = "when is it due"
    asyncio.run(telegram_bot.on_text_message(update2, None))
    assert update2.message.sent == [("Your EMI is due on the 5th.", None)]


def test_an_ordinary_message_in_an_unverified_chat_gets_no_buttons_attached(monkeypatch):
    async def reply(context, chat_id, text):
        return "Late fees apply after the due date."

    monkeypatch.setattr(telegram_bot, "_agent_reply", reply)
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=CHAT), message=FakeMessage())
    update.message.text = "what is the late fee"

    asyncio.run(telegram_bot.on_text_message(update, None))

    assert update.message.sent == [("Late fees apply after the due date.", None)]


# ---- the agent's markdown on Telegram (businessflow/markup.py) -------------------------------------------------------

def test_an_agent_reply_is_sent_with_its_markdown_as_telegram_formatting(monkeypatch):
    async def reply(context, chat_id, text):
        return "1. **Extend the tenure** by up to 3 months"

    monkeypatch.setattr(telegram_bot, "_agent_reply", reply)
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=CHAT), message=FakeMessage())
    update.message.text = "what are my options"

    asyncio.run(telegram_bot.on_text_message(update, None))

    assert update.message.sent == [("1. <b>Extend the tenure</b> by up to 3 months", None)]
    assert update.message.parse_modes == ["HTML"]


def test_if_telegram_refuses_the_formatting_the_reply_still_arrives_as_plain_text(monkeypatch):
    from telegram.error import BadRequest

    class RefusesHtml(FakeMessage):
        async def reply_text(self, text, reply_markup=None, parse_mode=None):
            if parse_mode:
                raise BadRequest("Can't parse entities")
            await super().reply_text(text, reply_markup=reply_markup)

    async def reply(context, chat_id, text):
        return "**Pay today** to avoid the late fee"

    monkeypatch.setattr(telegram_bot, "_agent_reply", reply)
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=CHAT), message=RefusesHtml())
    update.message.text = "should I pay"

    asyncio.run(telegram_bot.on_text_message(update, None))

    assert update.message.sent == [("Pay today to avoid the late fee", None)]
    assert update.message.parse_modes == [None]
