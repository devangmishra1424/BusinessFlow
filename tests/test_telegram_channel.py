"""Tests for the Telegram channel. Exercises the pure/sync routing helpers
directly with plain (chat_id, text) or (raw_bytes, ...) arguments -- never
via mocked python-telegram-bot Update/Context objects, matching
telegram_bot.py's own stated reason for keeping handle_incoming_message,
_decode_and_transcribe_voice_note, and handle_incoming_voice decoupled
from python-telegram-bot's types.

Real Postgres via the `reseed_accounts` fixture, gated behind `_pg_skip`
on DATABASE_URL (same convention as test_browser_api.py / test_auth.py).
The one path that makes a real Groq call -- anonymous chat's LLM reply --
is gated on GROQ_API_KEY instead, same convention as test_agent_loop.py
and test_pipeline.py's text_roundtrip (which also drives a real tool-
calling turn under nothing but a GROQ_API_KEY gate).

The voice decode/resample/VAD path uses a real, local Silero VAD model
(no network, no API key) fed a synthesized in-memory OGG/Opus buffer --
same "real model over mock" convention as test_vad.py's
test_returns_empty_on_pure_silence, just carried through the OGG decode
step this channel adds.
"""

import asyncio
import io
import os
import time

import numpy as np
import pytest
import soundfile as sf

from businessflow.channels import telegram_bot
from businessflow.channels.credentials import looks_like_credentials
from businessflow.channels.telegram_bot import (
    _decode_and_transcribe_voice_note,
    _handle_start_payload,
    _sessions,
    _transcript_echo,
    handle_incoming_message,
    handle_incoming_voice,
)

_pg_skip = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)
_groq_skip = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set -- copy .env.example to .env and fill it in to run this",
)


@pytest.fixture(autouse=True)
def _clear_channel_state():
    """_sessions and _language_choice are process-global in-memory dicts
    (see telegram_bot.py's module docstring on why) -- without clearing
    them, one test's chat_id could leak into a later test that reuses the
    same chat_id number."""
    _sessions.clear()
    telegram_bot._language_choice.clear()
    telegram_bot._session_locks.clear()
    yield
    _sessions.clear()
    telegram_bot._language_choice.clear()
    telegram_bot._session_locks.clear()


def _silence_ogg_bytes(seconds: float = 1.0, sample_rate: int = 48000) -> bytes:
    """A short digital-silence clip, encoded as real OGG/Opus bytes --
    the same container/codec Telegram voice notes actually arrive in --
    via soundfile, so _decode_and_transcribe_voice_note's own decode step
    is exercised for real, not skipped."""
    silence = np.zeros(int(seconds * sample_rate), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, silence, sample_rate, format="OGG", subtype="OPUS")
    return buf.getvalue()


class _FakeTelegramFile:
    """Duck-typed stand-in for telegram.File: handle_incoming_voice only
    ever calls download_as_bytearray() on it, per its own docstring."""

    def __init__(self, raw_bytes: bytes):
        self._raw_bytes = raw_bytes
        self.download_called = False

    async def download_as_bytearray(self):
        self.download_called = True
        return bytearray(self._raw_bytes)


# --- looks_like_credentials -------------------------------------------------


def test_looks_like_credentials_true_for_account_and_six_digit_key():
    assert looks_like_credentials("BF-1001 482913") is True


def test_looks_like_credentials_false_for_plain_sentence():
    assert looks_like_credentials("how much do I owe this month") is False


def test_looks_like_credentials_false_for_five_digit_key():
    assert looks_like_credentials("BF-1001 12345") is False


# --- handle_incoming_message -------------------------------------------------


@_groq_skip
@_pg_skip
def test_anonymous_message_starts_general_chat_with_real_reply():
    chat_id = 900001
    reply = handle_incoming_message(chat_id, "What documents do I need to close out my loan early?")

    assert chat_id in _sessions
    assert _sessions[chat_id]["account_id"] is None
    assert len(reply.strip()) > 0


@_pg_skip
def test_correct_credentials_verify_session_without_an_llm_call(reseed_accounts):
    chat_id = 900002
    reply = handle_incoming_message(chat_id, "BF-1001 482913")

    assert "Verified" in reply
    assert chat_id in _sessions
    assert _sessions[chat_id]["account_id"] == "BF-1001"
    # Verification is not a turn -- browser_api.py's separate POST
    # /conversations doesn't run one either, and neither does this path.
    assert [m["role"] for m in _sessions[chat_id]["messages"]] == ["system"]


@_pg_skip
def test_resending_credentials_on_an_already_verified_session_does_not_reach_the_llm(reseed_accounts):
    # Found live: an already-verified borrower who (redundantly) re-sent
    # "BF-1001 482913" got it forwarded straight to the LLM as an ordinary
    # message, with no idea it was ever a credentials pair -- the model
    # pattern-matched the bare 6-digit number in a financial conversation
    # and hallucinated a payment intent instead. Must short-circuit here,
    # exactly like the real verification branch does, and touch neither
    # the session's account_id nor its message history.
    chat_id = 900007
    handle_incoming_message(chat_id, "BF-1001 482913")
    messages_before = list(_sessions[chat_id]["messages"])

    reply = handle_incoming_message(chat_id, "BF-1001 482913")

    assert "already verified" in reply
    assert "BF-1001" in reply
    assert _sessions[chat_id]["account_id"] == "BF-1001"
    assert _sessions[chat_id]["messages"] == messages_before


@_pg_skip
@_groq_skip
def test_a_fresh_process_rehydrates_a_previously_verified_chat(reseed_accounts):
    # Simulates exactly the failure mode found live: the in-memory session
    # is gone (a deploy restart, or an OOM kill -- chat_id is deliberately
    # NOT in _sessions here) but the durable telegram_chat_id -> account_id
    # mapping written at the ORIGINAL verification survives it. An ordinary
    # follow-up message must resume as BF-1001, not demand re-verification.
    from businessflow.accounts import store

    chat_id = 900008
    store.set_telegram_chat_id("BF-1001", chat_id)
    assert chat_id not in _sessions

    reply = handle_incoming_message(chat_id, "How many months do I have left on my loan?")

    assert reply.startswith("Welcome back -- I've resumed account BF-1001.")
    assert _sessions[chat_id]["account_id"] == "BF-1001"


@_pg_skip
@_groq_skip
def test_a_never_verified_chat_id_still_gets_a_genuinely_anonymous_session(reseed_accounts):
    # No durable mapping exists for this chat_id at all -- must fall back
    # to today's existing anonymous-chat behavior, unchanged, not treat
    # "no mapping found" as an error or a false rehydration.
    chat_id = 900009

    reply = handle_incoming_message(chat_id, "What documents do I need to close out my loan early?")

    assert not reply.startswith("Welcome back")
    assert _sessions[chat_id]["account_id"] is None


@_pg_skip
def test_correct_credentials_persist_the_telegram_chat_id_on_the_account(reseed_accounts):
    # Durable, unlike _sessions above -- a decision made later on the ops
    # dashboard (approving/rejecting a restructuring request) needs this
    # to reach the borrower even after the in-memory session is gone.
    from businessflow.accounts import store

    chat_id = 900006
    handle_incoming_message(chat_id, "BF-1001 482913")

    assert store.get_account_or_raise("BF-1001").telegram_chat_id == chat_id


@_pg_skip
def test_wrong_credentials_deny_and_create_no_session(reseed_accounts):
    chat_id = 900003
    reply = handle_incoming_message(chat_id, "BF-1001 000000")

    assert "doesn't match" in reply
    assert chat_id not in _sessions


@_pg_skip
def test_followup_after_wrong_credentials_still_hits_verification_path(reseed_accounts):
    chat_id = 900004
    first_reply = handle_incoming_message(chat_id, "BF-1001 000000")

    assert "doesn't match" in first_reply
    assert chat_id not in _sessions  # no session existed yet after the failed attempt

    second_reply = handle_incoming_message(chat_id, "BF-1001 482913")

    assert "Verified" in second_reply  # so this is still routed as a verification attempt, not an existing turn
    assert _sessions[chat_id]["account_id"] == "BF-1001"


@_pg_skip
@_groq_skip
def test_credentials_sent_after_anonymous_chat_still_verify(reseed_accounts):
    # Regression test for a real bug found live: a borrower who chats
    # anonymously first (e.g. "what is my standing loan amount", with no
    # account attached yet) gets an anonymous session -- send valid
    # credentials AFTER that and they must still verify, not get silently
    # forwarded to the LLM as plain text (which tried to use the whole
    # "BF-1001 482913" string as a literal account_id and failed).
    chat_id = 900005
    first_reply = handle_incoming_message(chat_id, "what is my standing loan amount")

    assert _sessions[chat_id]["account_id"] is None  # anonymous session, as expected
    assert first_reply  # a real LLM reply, not an error

    second_reply = handle_incoming_message(chat_id, "BF-1001 482913")

    assert "Verified" in second_reply
    assert _sessions[chat_id]["account_id"] == "BF-1001"


@_pg_skip
def test_account_locks_out_after_repeated_failures_through_the_text_path(reseed_accounts):
    # Same lockout enforced in test_auth.py's
    # test_verify_and_start_conversation_locks_out_after_repeated_wrong_keys,
    # just reached through handle_incoming_message's text path instead of
    # calling verify_and_start_conversation directly.
    chat_id = 900005
    for _ in range(5):
        reply = handle_incoming_message(chat_id, "BF-1001 000000")
        assert "doesn't match" in reply

    reply = handle_incoming_message(chat_id, "BF-1001 482913")  # even the REAL key is now blocked

    assert "too many failed access attempts" in reply
    assert chat_id not in _sessions


# --- _decode_and_transcribe_voice_note ---------------------------------------


def test_decode_and_transcribe_returns_none_on_silence():
    raw_bytes = _silence_ogg_bytes()

    result = _decode_and_transcribe_voice_note(raw_bytes, "en")

    assert result is None


def test_decode_and_transcribe_resamples_a_non_16k_source():
    # Telegram voice notes are commonly 48kHz (see telegram_bot.py's
    # module docstring) -- confirms the resample path runs without
    # raising, for a rate that genuinely differs from the 16kHz VAD/ASR
    # expect, rather than only ever exercising the already-16kHz case.
    raw_bytes = _silence_ogg_bytes(seconds=1.0, sample_rate=48000)

    result = _decode_and_transcribe_voice_note(raw_bytes, None)

    assert result is None  # still silence -- just proves resample+VAD ran without crashing


def test_decode_and_transcribe_enforces_the_real_decoded_duration_not_just_a_claim(monkeypatch):
    # Regression test for a real bug an adversarial review caught: the
    # duration cap must be checked against the ACTUAL decoded audio length,
    # not a caller-supplied duration_seconds that a spoofed/malformed
    # container could under-report -- otherwise the cap does nothing.
    # Silence alone can't prove this (trim_to_speech would return None for
    # silence regardless of any duration check existing), so this
    # monkeypatches trim_to_speech/transcribe to prove they're never even
    # reached once the real decoded length exceeds the cap.
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("VAD/ASR must not run once the real decoded duration exceeds the cap")

    monkeypatch.setattr(telegram_bot, "trim_to_speech", _must_not_be_called)
    monkeypatch.setattr(telegram_bot, "transcribe", _must_not_be_called)

    over_cap_seconds = telegram_bot._MAX_VOICE_NOTE_SECONDS + 5
    raw_bytes = _silence_ogg_bytes(seconds=over_cap_seconds, sample_rate=16000)

    result = _decode_and_transcribe_voice_note(raw_bytes, "en")

    assert result is None


# --- _transcript_echo -----------------------------------------------------


def test_transcript_echo_redacts_a_credential_shaped_transcript():
    # Regression test for a real bug an adversarial review caught: the
    # "You said: ..." confirmation was echoing a spoken account_id +
    # 6-digit access key verbatim into the chat transcript, even though
    # handle_incoming_voice correctly refused to verify it.
    echo = _transcript_echo("BF-1001 482913")

    assert echo == "You said: [account details -- redacted]"
    assert "482913" not in echo


def test_transcript_echo_passes_through_a_normal_transcript():
    echo = _transcript_echo("what is my current EMI amount")

    assert echo == "You said: what is my current EMI amount"


# --- _handle_start_payload (Telegram deep-link onboarding) -----------------


def test_handle_start_payload_returns_none_for_a_malformed_payload():
    # on_start's own contract: None means "not a real deep-link payload,
    # fall back to the generic help text" -- must never reach verification
    # (and never mutate any session) for garbage input.
    result = asyncio.run(_handle_start_payload(900030, "not-a-real-payload"))

    assert result is None
    assert 900030 not in _sessions


@_pg_skip
def test_handle_start_payload_verifies_a_real_deep_link_tap(reseed_accounts):
    from businessflow.channels.credentials import build_telegram_start_payload

    chat_id = 900031
    payload = build_telegram_start_payload("BF-1001", "482913")

    result = asyncio.run(_handle_start_payload(chat_id, payload))

    assert "Verified" in result
    assert _sessions[chat_id]["account_id"] == "BF-1001"


@_pg_skip
def test_handle_start_payload_denies_a_wrong_key_the_same_way_typed_credentials_would(reseed_accounts):
    from businessflow.channels.credentials import build_telegram_start_payload

    chat_id = 900032
    payload = build_telegram_start_payload("BF-1001", "000000")

    result = asyncio.run(_handle_start_payload(chat_id, payload))

    assert "doesn't match" in result
    assert chat_id not in _sessions


# --- handle_incoming_voice ----------------------------------------------------


def test_handle_incoming_voice_rejects_too_long_before_downloading():
    fake_file = _FakeTelegramFile(b"unused")

    transcript, reply = asyncio.run(
        handle_incoming_voice(900010, fake_file, duration_seconds=121, language_hint="en")
    )

    assert transcript is None
    assert "too long" in reply
    assert fake_file.download_called is False  # cap must be enforced before spending a download on it


def test_handle_incoming_voice_reports_no_speech_detected():
    fake_file = _FakeTelegramFile(_silence_ogg_bytes())

    transcript, reply = asyncio.run(
        handle_incoming_voice(900011, fake_file, duration_seconds=2, language_hint="en")
    )

    assert transcript is None
    assert "couldn't hear anything" in reply
    assert fake_file.download_called is True


def test_handle_incoming_voice_blocks_credential_shaped_transcript_from_verification(monkeypatch):
    # The seam described in the implementer's report: a credential-shaped
    # transcript from a fresh chat_id must never reach
    # handle_incoming_message (no failed-attempt side effect from a
    # misheard digit) and must not create a session. The decode step is
    # monkeypatched here specifically because getting real ASR to
    # transcribe synthesized audio into an exact "BF-1001 482913" string
    # would be unreliable -- unlike the no-speech-on-silence case above,
    # which needs no patching because it's exactly what the real decode
    # path already does on silence.
    monkeypatch.setattr(telegram_bot, "_decode_and_transcribe_voice_note", lambda raw, lang: "BF-1001 482913")

    calls = []
    monkeypatch.setattr(telegram_bot, "handle_incoming_message", lambda chat_id, text: calls.append((chat_id, text)))

    chat_id = 900012
    fake_file = _FakeTelegramFile(b"unused")

    transcript, reply = asyncio.run(
        handle_incoming_voice(chat_id, fake_file, duration_seconds=3, language_hint="en")
    )

    assert transcript == "BF-1001 482913"
    assert "TYPE" in reply
    assert chat_id not in _sessions
    assert calls == []  # handle_incoming_message must never see this transcript


def test_handle_incoming_voice_allows_credential_shaped_transcript_once_a_session_exists(monkeypatch):
    # The guard is specifically "no session yet" -- once chat_id already
    # has a session, a credential-shaped voice transcript is just an
    # ordinary message and should flow through normally (e.g. a borrower
    # reading back numbers that happen to match the pattern mid-chat).
    _sessions[900013] = {"account_id": "BF-1001", "language": "en", "messages": [{"role": "system", "content": "x"}]}
    monkeypatch.setattr(telegram_bot, "_decode_and_transcribe_voice_note", lambda raw, lang: "BF-1001 482913")

    calls = []
    monkeypatch.setattr(
        telegram_bot,
        "handle_incoming_message",
        lambda chat_id, text: calls.append((chat_id, text)) or "handled",
    )

    fake_file = _FakeTelegramFile(b"unused")
    transcript, reply = asyncio.run(
        handle_incoming_voice(900013, fake_file, duration_seconds=3, language_hint="en")
    )

    assert transcript == "BF-1001 482913"
    assert reply == "handled"
    assert calls == [(900013, "BF-1001 482913")]


def test_two_voice_turns_for_the_same_chat_never_run_concurrently(monkeypatch):
    # Regression test for a real bug found live, not caught by any
    # existing test here: handle_incoming_voice calls handle_incoming_message
    # via asyncio.to_thread, mutating the shared, unlocked session["messages"]
    # list. A slow turn (real STT + LLM + TTS is easily several seconds)
    # still in flight when the borrower's next voice note or text message
    # arrives raced against it in a separate worker thread -- whichever
    # turn finished last silently overwrote the other's appended messages,
    # which is exactly what made a borrower's question appear to vanish or
    # get answered out of order. _get_session_lock now serializes every
    # call into handle_incoming_message per chat_id; this proves it by
    # firing two voice turns for the same chat concurrently and recording
    # how many were ever "inside" handle_incoming_message at once.
    chat_id = 900020
    concurrent_count = 0
    max_concurrent_seen = 0
    order = []

    def fake_handle_incoming_message(cid, text):
        nonlocal concurrent_count, max_concurrent_seen
        concurrent_count += 1
        max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
        time.sleep(0.15)  # stand-in for real STT+LLM+TTS latency
        order.append(text)
        concurrent_count -= 1
        return f"reply to {text}"

    monkeypatch.setattr(telegram_bot, "handle_incoming_message", fake_handle_incoming_message)
    monkeypatch.setattr(telegram_bot, "_decode_and_transcribe_voice_note", lambda raw, lang: raw.decode())

    async def run_both():
        return await asyncio.gather(
            handle_incoming_voice(chat_id, _FakeTelegramFile(b"first question"), duration_seconds=3, language_hint="en"),
            handle_incoming_voice(chat_id, _FakeTelegramFile(b"second question"), duration_seconds=3, language_hint="en"),
        )

    results = asyncio.run(run_both())

    assert max_concurrent_seen == 1  # never more than one turn in flight for this chat
    assert order == ["first question", "second question"]  # strictly sequential, not interleaved
    assert {r[1] for r in results} == {"reply to first question", "reply to second question"}


@_groq_skip
@_pg_skip
def test_handle_incoming_voice_reaches_the_real_agent_loop_without_the_nested_event_loop_bug(monkeypatch):
    # Regression test for a real bug found live (not in any test): every
    # other handle_incoming_voice test monkeypatches handle_incoming_message
    # itself, so none of them ever actually called into agent.loop.run_turn,
    # which does asyncio.run(...) internally. That's harmless from a plain
    # sync caller (which is all any other test is), but handle_incoming_voice
    # is a coroutine -- when it's driven by python-telegram-bot's own running
    # event loop (as it is for real, unlike asyncio.run() in a test), calling
    # straight into that asyncio.run() raises "asyncio.run() cannot be
    # called from a running event loop", and the borrower gets no reply at
    # all. Only monkeypatching the decode step here (not
    # handle_incoming_message) is what makes this test actually exercise the
    # real chain and would have caught the bug.
    monkeypatch.setattr(
        telegram_bot, "_decode_and_transcribe_voice_note",
        lambda raw, lang: "what is my current EMI amount",
    )
    fake_file = _FakeTelegramFile(b"unused")

    transcript, reply = asyncio.run(
        handle_incoming_voice(900014, fake_file, duration_seconds=3, language_hint="en")
    )

    assert transcript == "what is my current EMI amount"
    assert reply  # a real, non-empty reply from the real agent loop
