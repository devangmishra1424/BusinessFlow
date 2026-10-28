"""The voice-service client (audio/remote.py) against a real HTTP server on loopback, and the two call sites that use it
(audio/asr.py's transcribe, audio/tts.py's _piper_chunks). The failure paths are the point: a rejected key, a server
error, a refused request and an unreachable host must each fall back to the local model, say why in the log, and
pause the service only when it looks down."""

import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import torch

from businessflow.audio import asr, remote, tts


class _Service:
    """A stand-in voice service: answers each path with the next canned (status, body) and records the requests."""

    def __init__(self):
        self.answers: dict[str, tuple[int, bytes]] = {}
        self.requests: list[dict] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                service.requests.append({"path": self.path, "key": self.headers.get("X-API-Key"), "body": body,
                                         "content_type": self.headers.get("Content-Type")})
                status, payload = service.answers.get(self.path, (500, b"no canned answer"))
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # keep pytest output clean
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def service(monkeypatch):
    s = _Service()
    monkeypatch.setenv("VOICE_SERVICE_URL", s.url)
    monkeypatch.setenv("VOICE_SERVICE_API_KEY", "test-key")
    monkeypatch.setattr(remote, "_paused_until", 0.0)
    monkeypatch.setattr(remote, "_last_error", None)
    monkeypatch.setattr(remote, "_failures", 0)
    yield s
    s.close()


def _stt_answer(text="नमस्ते"):
    return 200, json.dumps({"text": text, "language": "hi", "avg_logprob": -0.4, "no_speech_prob": 0.01}).encode()


def _synth_answer(chunks, rate=22050):
    encoded = [base64.b64encode(np.asarray(c, dtype="<f4").tobytes()).decode() for c in chunks]
    return 200, json.dumps({"sample_rate": rate, "chunks": encoded}).encode()


AUDIO = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)


def test_not_configured_without_a_url(monkeypatch):
    monkeypatch.delenv("VOICE_SERVICE_URL", raising=False)
    assert remote.configured() is False
    monkeypatch.setenv("VOICE_SERVICE_URL", "  ")
    assert remote.configured() is False


def test_transcribe_sends_the_key_and_the_samples_and_reads_the_answer(service):
    service.answers["/v1/stt"] = _stt_answer("मेरी EMI कब है")
    result = remote.transcribe(AUDIO, "hi")
    assert result == remote.Transcript("मेरी EMI कब है", -0.4, 0.01, "hi")
    request = service.requests[0]
    assert request["key"] == "test-key"
    assert request["content_type"].startswith("multipart/form-data; boundary=")
    assert b'name="language"\r\n\r\nhi' in request["body"]
    assert b"RIFF" in request["body"]  # a WAV with the audio, not an empty upload


def test_synth_round_trips_float32_chunks_exactly(service):
    first, second = np.random.default_rng(0).random(500, dtype=np.float32), np.random.default_rng(1).random(300, dtype=np.float32)
    service.answers["/v1/synth"] = _synth_answer([first, second], rate=22050)
    chunks, rate = remote.synth("Hello. World.", "en")
    assert rate == 22050 and len(chunks) == 2
    np.testing.assert_array_equal(chunks[0], first)
    np.testing.assert_array_equal(chunks[1], second)
    assert json.loads(service.requests[0]["body"]) == {"text": "Hello. World.", "voice": "en"}


def test_synth_of_nothing_speakable_is_no_chunks_not_an_error(service):
    service.answers["/v1/synth"] = 200, json.dumps({"sample_rate": None, "chunks": []}).encode()
    assert remote.synth("???", "en") == ([], None)


@pytest.mark.parametrize("code", [401, 403, 500, 503])
def test_a_rejected_key_or_server_error_raises_logs_and_pauses_the_service(service, caplog, code):
    service.answers["/v1/stt"] = code, b"nope"
    with caplog.at_level(logging.ERROR, logger="businessflow.audio.remote"):
        with pytest.raises(remote.VoiceServiceUnavailable):
            remote.transcribe(AUDIO, None)
    assert any("voice service" in r.message and r.levelno == logging.ERROR for r in caplog.records)
    # Paused: the next call does not even reach the server.
    with pytest.raises(remote.VoiceServiceUnavailable, match="paused"):
        remote.transcribe(AUDIO, None)
    assert len(service.requests) == 1
    assert remote.status()["paused"] is True and remote.status()["failures"] == 1


@pytest.mark.parametrize("code", [400, 413, 422, 429])
def test_a_refused_request_raises_but_does_not_pause_the_service(service, code):
    service.answers["/v1/stt"] = code, b"refused"
    with pytest.raises(remote.VoiceServiceUnavailable):
        remote.transcribe(AUDIO, None)
    service.answers["/v1/stt"] = _stt_answer()
    assert remote.transcribe(AUDIO, None).text == "नमस्ते"  # the next call goes through
    assert remote.status()["paused"] is False


def test_an_unreachable_service_raises_and_pauses(monkeypatch):
    monkeypatch.setenv("VOICE_SERVICE_URL", "http://127.0.0.1:1")  # nothing listens on port 1
    monkeypatch.setenv("VOICE_SERVICE_API_KEY", "k")
    monkeypatch.setattr(remote, "_paused_until", 0.0)
    with pytest.raises(remote.VoiceServiceUnavailable, match="unreachable"):
        remote.synth("hello", "en")
    assert remote.status()["paused"] is True


def test_a_missing_key_is_a_logged_fallback_not_a_crash(service, monkeypatch, caplog):
    monkeypatch.delenv("VOICE_SERVICE_API_KEY")
    with caplog.at_level(logging.ERROR, logger="businessflow.audio.remote"):
        with pytest.raises(remote.VoiceServiceUnavailable):
            remote.synth("hello", "en")
    assert "VOICE_SERVICE_API_KEY" in caplog.text
    assert service.requests == []


def test_an_unreadable_answer_is_unavailable(service):
    service.answers["/v1/synth"] = 200, b"<html>not json</html>"
    with pytest.raises(remote.VoiceServiceUnavailable, match="unreadable"):
        remote.synth("hello", "en")


# --- the call sites -------------------------------------------------------------------------------------------------

def test_transcribe_uses_the_service_and_marks_the_trace_remote(service, monkeypatch):
    service.answers["/v1/stt"] = _stt_answer("remote text")
    monkeypatch.setattr(asr, "_decode_local", lambda *a: pytest.fail("the local model must not run"))
    assert asr.transcribe(torch.from_numpy(AUDIO), language="hi") == "remote text"


def test_transcribe_falls_back_to_the_local_model_when_the_service_is_down(service, monkeypatch):
    service.answers["/v1/stt"] = 500, b"boom"
    monkeypatch.setattr(asr, "_decode_local", lambda audio, size, lang: remote.Transcript("local text", -0.2, 0.0, "en"))
    assert asr.transcribe(torch.from_numpy(AUDIO), language="en") == "local text"


def test_a_caller_that_names_its_own_model_always_decodes_locally(service, monkeypatch):
    monkeypatch.setattr(asr, "_decode_local", lambda audio, size, lang: remote.Transcript(f"local {size}", None, None, None))
    assert asr.transcribe(torch.from_numpy(AUDIO), model_size="small", language="en") == "local small"
    assert service.requests == []


class _FakeLocalVoice:
    def __init__(self, arrays, rate):
        self.arrays, self.rate = arrays, rate

    def synthesize(self, text):
        class Chunk:
            pass
        out = []
        for a in self.arrays:
            c = Chunk()
            c.audio_float_array, c.sample_rate = a, self.rate
            out.append(c)
        return out


def test_piper_chunks_come_from_the_service_when_it_answers(service):
    audio = np.ones(10, dtype=np.float32) * 0.25
    service.answers["/v1/synth"] = _synth_answer([audio], rate=22050)
    chunks, rate = tts._piper_chunks("hello", "en", lambda: pytest.fail("the local voice must not load"))
    assert rate == 22050
    np.testing.assert_array_equal(chunks[0], audio)


def test_piper_chunks_fall_back_to_the_local_voice_when_the_service_is_down(service):
    service.answers["/v1/synth"] = 500, b"boom"
    local = np.full(7, 0.5, dtype=np.float32)
    chunks, rate = tts._piper_chunks("hello", "en", lambda: _FakeLocalVoice([local], 16000))
    assert rate == 16000
    np.testing.assert_array_equal(chunks[0], local)


def test_a_hindi_voice_the_service_does_not_hold_is_spoken_locally(service):
    local = np.full(5, 0.1, dtype=np.float32)
    chunks, rate = tts._piper_chunks("नमस्ते", "hi", lambda: _FakeLocalVoice([local], 22050), use_service=False)
    assert rate == 22050 and service.requests == []


def test_without_a_service_the_local_voice_is_used_as_before(monkeypatch):
    monkeypatch.delenv("VOICE_SERVICE_URL", raising=False)
    local = np.full(5, 0.1, dtype=np.float32)
    chunks, rate = tts._piper_chunks("hello", "en", lambda: _FakeLocalVoice([local], 22050))
    assert rate == 22050 and chunks[0] is local


def test_english_with_nothing_speakable_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("VOICE_SERVICE_URL", raising=False)
    monkeypatch.setattr(tts, "_piper_voice", lambda: _FakeLocalVoice([], 22050))
    with pytest.raises(ValueError, match="nothing speakable"):
        tts._speak_english("???")


# --- /health ----------------------------------------------------------------------------------------------------------

def test_chat_health_says_nothing_about_voice_when_no_service_is_configured(monkeypatch):
    from fastapi.testclient import TestClient

    from businessflow.channels import browser_api

    monkeypatch.delenv("VOICE_SERVICE_URL", raising=False)
    monkeypatch.setattr(browser_api, "check_database", lambda *a, **k: None)
    assert TestClient(browser_api.app).get("/health").json() == {"status": "ok", "database": "ok"}


def test_chat_health_shows_when_voice_has_fallen_back_to_the_local_models(service, monkeypatch):
    from fastapi.testclient import TestClient

    from businessflow.channels import browser_api

    monkeypatch.setattr(browser_api, "check_database", lambda *a, **k: None)
    client = TestClient(browser_api.app)
    assert client.get("/health").json()["voice"] == "remote"

    service.answers["/v1/synth"] = 500, b"boom"
    with pytest.raises(remote.VoiceServiceUnavailable):
        remote.synth("hello", "en")
    response = client.get("/health")
    assert response.status_code == 200  # voice being down never fails the service
    assert response.json()["voice"] == "local-fallback"
    assert "boom" not in response.text and "127.0.0.1" not in response.text  # no reasons or addresses in a public body
