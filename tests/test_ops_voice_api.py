"""Ops voice over HTTP: the ops endpoints (/voice/*), the chat service's
internal speech endpoints they borrow Whisper and Piper through, and the wire
between the two (ops/speech_client.py against a real chat app on loopback).

The model call (voice.understand) and the speech models are replaced with
stand-ins here -- they're covered in test_ops_voice.py and test_asr.py /
test_tts.py -- so these run fast and without Groq. The tests marked _pg_skip
use real Postgres (CI) to check that an answer matches the dashboard's own
numbers, and that a confirmation card's request, sent as-is, does what the
card said -- and that nothing was written before it was sent.
"""

import json
import logging
import os
import socket
import threading
import time
from types import SimpleNamespace

import groq
import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from businessflow.audio.tts import Speech
from businessflow.channels import browser_api
from businessflow.ops import api as ops_api
from businessflow.ops import speech_client, voice
from businessflow.ops.voice import parse_plan
from businessflow.rate_limit import RateLimiter

_KEY = "test-ops-voice-key"
ops = TestClient(ops_api.app)
chat = TestClient(browser_api.app)

_pg_skip = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set -- these tests hit real Postgres")


@pytest.fixture(autouse=True)
def _ops_key_and_fresh_limiters(monkeypatch):
    monkeypatch.setenv("OPS_API_KEY", _KEY)
    monkeypatch.setattr(ops_api, "_ops_key_brute_force_limiter", RateLimiter(max_requests=5, window_seconds=300))
    monkeypatch.setattr(ops_api, "_voice_limiter", RateLimiter(max_requests=12, window_seconds=60))
    monkeypatch.setattr(ops_api, "_speech_limiter", RateLimiter(max_requests=30, window_seconds=60))


def _auth():
    return {"X-API-Key": _KEY}


def _wav_bytes() -> bytes:
    with open(os.path.join(os.path.dirname(__file__), "fixtures", "sample_speech_en.wav"), "rb") as f:
        return f.read()


def _understood_as(monkeypatch, tool, **arguments):
    seen = []
    monkeypatch.setattr(voice, "understand", lambda text, today, on_screen: seen.append(text) or parse_plan(tool, arguments))
    return seen


# --- the ops endpoints, no database ---------------------------------------------------------------------------------

def test_voice_endpoints_need_the_ops_key():
    assert ops.post("/voice/ask-text", json={"text": "help"}).status_code == 401
    assert ops.post("/voice/speak", json={"text": "hi"}, headers={"X-API-Key": "wrong"}).status_code == 401


def test_a_typed_request_is_understood_and_answered(monkeypatch):
    seen = _understood_as(monkeypatch, "help")
    response = ops.post("/voice/ask-text", json={"text": "  what can you do?  "}, headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert seen == ["what can you do?"]
    assert body["kind"] == "answer" and body["heard"] == "what can you do?" and "confirm" in body["say"]


def test_a_spoken_request_goes_through_the_speech_service_then_the_same_path(monkeypatch):
    sent = []
    monkeypatch.setattr(speech_client, "transcribe", lambda wav: sent.append(wav) or "what can you do")
    _understood_as(monkeypatch, "help")
    response = ops.post("/voice/ask", headers=_auth(), files={"audio": ("q.wav", _wav_bytes(), "audio/wav")},
                        data={"screen": json.dumps({"view": "inbox"})})
    assert response.status_code == 200
    assert sent == [_wav_bytes()]
    assert response.json()["heard"] == "what can you do"


def test_silence_is_said_to_be_silence_without_asking_the_model(monkeypatch):
    monkeypatch.setattr(speech_client, "transcribe", lambda wav: None)
    monkeypatch.setattr(voice, "understand", lambda *a: pytest.fail("nothing was said, so the model must not be asked"))
    response = ops.post("/voice/ask", headers=_auth(), files={"audio": ("q.wav", b"RIFF....", "audio/wav")})
    assert response.status_code == 200
    assert response.json()["kind"] == "clarify" and "didn't hear anything" in response.json()["say"]


def test_speech_service_down_means_voice_unavailable_not_a_crash(monkeypatch):
    def down(wav):
        raise speech_client.SpeechServiceUnavailable("speech service unreachable")

    monkeypatch.setattr(speech_client, "transcribe", down)
    response = ops.post("/voice/ask", headers=_auth(), files={"audio": ("q.wav", b"RIFF....", "audio/wav")})
    assert response.status_code == 503
    assert "Typing still works" in response.json()["detail"]


def test_a_recording_the_speech_service_cannot_read_is_a_400(monkeypatch):
    def unreadable(wav):
        raise speech_client.UnreadableRecording("refused (400)")

    monkeypatch.setattr(speech_client, "transcribe", unreadable)
    response = ops.post("/voice/ask", headers=_auth(), files={"audio": ("q.wav", b"not audio", "audio/wav")})
    assert response.status_code == 400


def test_a_recording_over_the_upload_limit_is_refused_before_transcription(monkeypatch):
    monkeypatch.setattr(ops_api, "_MAX_VOICE_UPLOAD_BYTES", 10)
    monkeypatch.setattr(speech_client, "transcribe", lambda wav: pytest.fail("must not be transcribed"))
    response = ops.post("/voice/ask", headers=_auth(), files={"audio": ("q.wav", b"x" * 11, "audio/wav")})
    assert response.status_code == 413


def test_a_screen_that_is_not_json_is_a_400():
    response = ops.post("/voice/ask", headers=_auth(), files={"audio": ("q.wav", b"RIFF", "audio/wav")}, data={"screen": "nope"})
    assert response.status_code == 400


def test_a_model_choice_that_cannot_be_used_asks_again_instead_of_failing(monkeypatch):
    def not_understood(*a):
        raise voice.NotUnderstood("the model called no function")

    monkeypatch.setattr(voice, "understand", not_understood)
    response = ops.post("/voice/ask-text", json={"text": "blorp"}, headers=_auth())
    assert response.status_code == 200 and response.json()["kind"] == "clarify"


def test_out_of_ai_allowance_is_a_clear_503(monkeypatch):
    def rate_limited(*a):
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        raise groq.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)

    monkeypatch.setattr(voice, "understand", rate_limited)
    response = ops.post("/voice/ask-text", json={"text": "how many overdue"}, headers=_auth())
    assert response.status_code == 503 and "allowance" in response.json()["detail"]


def test_ai_service_unreachable_is_a_502(monkeypatch):
    def unreachable(*a):
        raise groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))

    monkeypatch.setattr(voice, "understand", unreachable)
    assert ops.post("/voice/ask-text", json={"text": "how many overdue"}, headers=_auth()).status_code == 502


def test_voice_requests_are_capped_per_client(monkeypatch):
    monkeypatch.setattr(ops_api, "_voice_limiter", RateLimiter(max_requests=2, window_seconds=60))
    _understood_as(monkeypatch, "help")
    codes = [ops.post("/voice/ask-text", json={"text": "help"}, headers=_auth()).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_the_operators_words_are_not_logged(monkeypatch, caplog):
    _understood_as(monkeypatch, "help")
    with caplog.at_level(logging.INFO, logger="businessflow.ops.api"):
        ops.post("/voice/ask-text", json={"text": "what does Fatima Khan owe"}, headers=_auth())
    assert "ability=help" in caplog.text
    assert "Fatima" not in caplog.text


def test_a_choice_sent_back_runs_without_the_model_and_is_checked_against_the_menu(monkeypatch):
    monkeypatch.setattr(voice, "understand", lambda *a: pytest.fail("a choice must not call the model"))
    assert ops.post("/voice/plan", json={"tool": "help"}, headers=_auth()).json()["kind"] == "answer"
    assert ops.post("/voice/plan", json={"tool": "drop_tables"}, headers=_auth()).status_code == 400
    bad = ops.post("/voice/plan", json={"tool": "log_call", "arguments": {"outcome": "argued"}}, headers=_auth())
    assert bad.status_code == 400


def test_speak_returns_the_speech_services_audio(monkeypatch):
    monkeypatch.setattr(speech_client, "speak", lambda text: b"OggS-fake")
    response = ops.post("/voice/speak", json={"text": "Three accounts are overdue."}, headers=_auth())
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/ogg" and response.content == b"OggS-fake"


def test_speak_with_the_speech_service_down_is_a_503(monkeypatch):
    def down(text):
        raise speech_client.SpeechServiceUnavailable("down")

    monkeypatch.setattr(speech_client, "speak", down)
    assert ops.post("/voice/speak", json={"text": "hello"}, headers=_auth()).status_code == 503


# --- the chat service's internal speech endpoints -------------------------------------------------------------------

def test_internal_speech_is_refused_when_it_came_through_the_public_proxy_even_with_the_key():
    headers = {**_auth(), "X-Forwarded-For": "203.0.113.9"}
    assert chat.post("/internal/ops-voice/transcribe", content=_wav_bytes(), headers=headers).status_code == 403
    assert chat.post("/internal/ops-voice/speak", json={"text": "hi"}, headers=headers).status_code == 403


def test_internal_speech_needs_the_ops_key():
    assert chat.post("/internal/ops-voice/transcribe", content=_wav_bytes()).status_code == 401
    assert chat.post("/internal/ops-voice/speak", json={"text": "hi"}, headers={"X-API-Key": "wrong"}).status_code == 401


def test_internal_transcribe_refuses_what_is_not_audio_and_what_is_too_big(monkeypatch):
    assert chat.post("/internal/ops-voice/transcribe", content=b"definitely not a wav", headers=_auth()).status_code == 400
    monkeypatch.setattr(browser_api, "_OPS_VOICE_MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr(browser_api, "_decode_and_transcribe_voice", lambda *a, **k: pytest.fail("must not be transcribed"))
    assert chat.post("/internal/ops-voice/transcribe", content=b"x" * 101, headers=_auth()).status_code == 413


def test_internal_transcribe_is_english_and_capped_at_30_seconds(monkeypatch):
    calls = []
    monkeypatch.setattr(browser_api, "_decode_and_transcribe_voice",
                        lambda raw, language, max_seconds: calls.append((language, max_seconds)) or "  how many are overdue ")
    response = chat.post("/internal/ops-voice/transcribe", content=_wav_bytes(), headers=_auth())
    assert response.json() == {"transcript": "how many are overdue"}
    assert calls == [("en", 30)]


def test_internal_speak_limits_the_text_and_returns_ogg(monkeypatch):
    import torch

    monkeypatch.setattr(browser_api, "speak_english", lambda text: Speech(audio=torch.zeros(2205), sample_rate=22050))
    assert chat.post("/internal/ops-voice/speak", json={"text": "x" * 401}, headers=_auth()).status_code == 422
    response = chat.post("/internal/ops-voice/speak", json={"text": "₹12,500 is due."}, headers=_auth())
    assert response.status_code == 200 and response.content[:4] == b"OggS"


# --- the wire: speech_client against a real chat app on loopback ----------------------------------------------------

@pytest.fixture
def chat_on_loopback(monkeypatch):
    """The chat app served for real on 127.0.0.1 (lifespan off: no voice download), and speech_client pointed at it."""
    server = uvicorn.Server(uvicorn.Config(browser_api.app, host="127.0.0.1", port=0, lifespan="off", log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            pytest.fail("the chat app did not start on loopback")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    monkeypatch.setenv("CHAT_INTERNAL_URL", f"http://127.0.0.1:{port}")
    yield
    server.should_exit = True
    thread.join(timeout=10)


def test_speech_client_reaches_the_chat_service_directly_with_the_key(monkeypatch, chat_on_loopback):
    import torch

    monkeypatch.setattr(browser_api, "_decode_and_transcribe_voice", lambda raw, language, max_seconds: f"heard {len(raw)} bytes")
    monkeypatch.setattr(browser_api, "speak_english", lambda text: Speech(audio=torch.zeros(2205), sample_rate=22050))
    # a direct loopback call carries no X-Forwarded-For, so it passes the gate the public route can't
    assert speech_client.transcribe(_wav_bytes()) == f"heard {len(_wav_bytes())} bytes"
    assert speech_client.speak("Three accounts are overdue.")[:4] == b"OggS"


def test_speech_client_with_a_different_key_than_the_chat_service_is_unavailable(monkeypatch, chat_on_loopback):
    import torch

    # Both apps read OPS_API_KEY from this one process's environment, so the chat side is given its own.
    monkeypatch.setattr(browser_api, "os", SimpleNamespace(environ={"OPS_API_KEY": "the-chat-services-key"}))
    monkeypatch.setattr(browser_api, "speak_english", lambda text: Speech(audio=torch.zeros(2205), sample_rate=22050))
    with pytest.raises(speech_client.SpeechServiceUnavailable, match="answered 401"):
        speech_client.speak("hello")


def test_an_empty_chat_internal_url_means_the_default_address(monkeypatch):
    monkeypatch.setenv("CHAT_INTERNAL_URL", "")  # how .env.example leaves it
    assert speech_client._base_url() == "http://127.0.0.1:8000"


def test_speech_client_with_nothing_listening_is_unavailable(monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    monkeypatch.setenv("CHAT_INTERNAL_URL", f"http://127.0.0.1:{free_port}")
    with pytest.raises(speech_client.SpeechServiceUnavailable):
        speech_client.transcribe(b"RIFF")


# --- against real Postgres ------------------------------------------------------------------------------------------

@_pg_skip
def test_an_answer_matches_the_dashboards_own_numbers(monkeypatch, reseed_accounts):
    _understood_as(monkeypatch, "show_accounts", group="overdue")
    overdue = ops.get("/accounts?flag=overdue", headers=_auth()).json()
    reply = ops.post("/voice/ask-text", json={"text": "who is overdue"}, headers=_auth()).json()
    assert reply["kind"] == "answer"
    if overdue:
        assert reply["say"].startswith(f"{len(overdue)} overdue account")
        worst = max(overdue, key=lambda a: a["days_past_due"])
        assert worst["borrower_name"] in reply["say"]


@_pg_skip
def test_a_payment_card_writes_nothing_until_its_request_is_sent_and_then_records_it(monkeypatch, reseed_accounts):
    account = ops.get("/accounts/BF-1001", headers=_auth()).json()
    amount = account["emi_amount"]
    payments_before = len(account["payment_history"])
    _understood_as(monkeypatch, "record_payment", account="BF-1001", amount=amount)

    reply = ops.post("/voice/ask-text", json={"text": f"record {amount} from BF-1001"}, headers=_auth()).json()
    assert reply["kind"] == "proposal"
    assert len(ops.get("/accounts/BF-1001", headers=_auth()).json()["payment_history"]) == payments_before

    card = reply["proposal"]
    sent = ops.request(card["method"], card["path"], json=card["body"], headers=_auth())
    assert sent.status_code == 200, sent.text
    assert sent.json()["amount"] == amount
    assert len(ops.get("/accounts/BF-1001", headers=_auth()).json()["payment_history"]) == payments_before + 1
