"""The voice service's HTTP behaviour (voice_service/app.py), with the two models replaced by small fakes so no model
file is needed: what matters here is the edge (key, language, sizes, voices, busy) and the shapes it returns. The one
test that needs a real Piper voice is skipped when the model file is not on the machine."""

import base64
import io
import os
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from businessflow.audio import asr
from voice_service import app as voice_app

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}
# PIPER_EN_MODEL lets a worktree without data/models point at another checkout's copy.
_PIPER_EN = Path(os.environ.get("PIPER_EN_MODEL") or Path(__file__).resolve().parents[1] / "data" / "models" / "piper" / "en_US-lessac-medium.onnx")


def _wav(seconds=1.0, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.sin(np.linspace(0, 440 * seconds, int(rate * seconds))) * 8000).astype("<i2").tobytes())
    return buf.getvalue()


class _FakeAsr:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        segs = [SimpleNamespace(text=" नमस्ते ", avg_logprob=-0.5, no_speech_prob=0.1),
                SimpleNamespace(text="EMI", avg_logprob=-0.1, no_speech_prob=0.3)]
        return iter(segs), SimpleNamespace(language=kwargs.get("language") or "hi")


class _FakeVoice:
    def __init__(self, arrays, rate=22050):
        self.arrays, self.rate = arrays, rate

    def synthesize(self, text):
        return [SimpleNamespace(audio_float_array=a, audio_int16_bytes=(a * 32767).astype("<i2").tobytes(),
                                sample_rate=self.rate) for a in self.arrays]


@pytest.fixture
def client(monkeypatch):
    fake_asr = _FakeAsr()
    sentences = [np.array([0.1, -0.2, 0.3], dtype=np.float32), np.array([0.5, 0.25], dtype=np.float32)]
    monkeypatch.setattr(voice_app, "KEYS", [KEY])
    monkeypatch.setattr(voice_app, "waiting", 0)
    monkeypatch.setitem(voice_app.state, "asr", fake_asr)
    monkeypatch.setitem(voice_app.state, "asr_name", "whisper-ft")
    monkeypatch.setitem(voice_app.state, "tts", {"en": _FakeVoice(sentences), "hi": _FakeVoice(sentences, 22050)})
    c = TestClient(voice_app.app)  # not a context manager: the lifespan would load the real models
    c.fake_asr, c.sentences = fake_asr, sentences
    return c


def _post_stt(client, data=None, headers=HEADERS, **fields):
    return client.post("/v1/stt", files={"file": ("a.wav", data if data is not None else _wav(), "audio/wav")},
                       data=fields, headers=headers)


def test_decode_settings_match_businessflows_own():
    assert voice_app.DECODE_KWARGS == asr._DECODE_KWARGS


def test_health_reports_the_live_model_and_voices(client):
    assert client.get("/health").json() == {"ok": True, "asr": True, "asr_model": "whisper-ft", "tts": ["en", "hi"]}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_every_endpoint_refuses_a_missing_or_wrong_key(client, headers):
    assert _post_stt(client, headers=headers).status_code == 401
    assert client.post("/v1/tts", json={"text": "hi"}, headers=headers).status_code == 401
    assert client.post("/v1/synth", json={"text": "hi", "voice": "en"}, headers=headers).status_code == 401


def test_with_no_key_configured_the_service_refuses_everything(client, monkeypatch):
    monkeypatch.setattr(voice_app, "KEYS", [])
    assert _post_stt(client).status_code == 503


def test_each_app_key_works_on_its_own(client, monkeypatch):
    monkeypatch.setattr(voice_app, "KEYS", ["key-a", "key-b"])
    assert _post_stt(client, headers={"X-API-Key": "key-b"}).status_code == 200
    assert _post_stt(client, headers={"X-API-Key": "key-a"}).status_code == 200
    assert _post_stt(client, headers={"X-API-Key": KEY}).status_code == 401  # a revoked key is gone


def test_stt_decodes_with_businessflows_settings_and_reports_confidence(client):
    r = _post_stt(client, language="hi")
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "नमस्ते EMI" and body["language"] == "hi"
    # Weighted by the raw segment text length (" नमस्ते " is 8 characters with its spaces), as asr.py does.
    assert body["avg_logprob"] == pytest.approx((-0.5 * 8 + -0.1 * 3) / 11, abs=1e-3)
    assert body["no_speech_prob"] == 0.3
    call = client.fake_asr.calls[0]
    assert call["language"] == "hi" and call["vad_filter"] is False
    assert {k: call[k] for k in voice_app.DECODE_KWARGS} == voice_app.DECODE_KWARGS


def test_stt_can_ask_whisper_to_drop_silence_for_raw_browser_recordings(client):
    assert _post_stt(client, vad_filter="true").status_code == 200
    assert client.fake_asr.calls[0]["vad_filter"] is True


def test_stt_rejects_a_language_it_does_not_serve(client):
    assert _post_stt(client, language="fr").status_code == 400
    assert client.fake_asr.calls == []


def test_stt_rejects_bytes_that_are_not_audio(client):
    assert _post_stt(client, data=b"this is not audio").status_code == 400
    assert client.fake_asr.calls == []


def test_stt_rejects_an_oversized_upload_before_decoding(client, monkeypatch):
    monkeypatch.setattr(voice_app, "MAX_AUDIO_BYTES", 1000)
    assert _post_stt(client, data=b"\0" * 1001).status_code == 413


def test_stt_rejects_a_recording_longer_than_the_cap(client, monkeypatch):
    monkeypatch.setattr(voice_app, "MAX_AUDIO_SECONDS", 1)
    assert _post_stt(client, data=_wav(seconds=2.0)).status_code == 413
    assert client.fake_asr.calls == []


def test_a_busy_service_says_so_instead_of_queueing_forever(client, monkeypatch):
    monkeypatch.setattr(voice_app, "waiting", voice_app.MAX_WAITING)
    assert _post_stt(client).status_code == 429
    assert client.post("/v1/synth", json={"text": "hi", "voice": "en"}, headers=HEADERS).status_code == 429


def test_synth_returns_each_sentence_as_exact_float32(client):
    r = client.post("/v1/synth", json={"text": "One. Two.", "voice": "hi"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["sample_rate"] == 22050 and len(body["chunks"]) == 2
    for encoded, expected in zip(body["chunks"], client.sentences):
        np.testing.assert_array_equal(np.frombuffer(base64.b64decode(encoded), dtype="<f4"), expected)


@pytest.mark.parametrize("text", ["???", "...", "   ?! "])
def test_synth_of_text_with_nothing_to_say_is_empty_not_an_error(client, text):
    r = client.post("/v1/synth", json={"text": text, "voice": "en"}, headers=HEADERS)
    assert r.status_code == 200 and r.json() == {"sample_rate": None, "chunks": []}


def test_synth_rejects_an_unknown_voice_and_bad_lengths(client):
    assert client.post("/v1/synth", json={"text": "hi", "voice": "fr"}, headers=HEADERS).status_code == 400
    assert client.post("/v1/synth", json={"text": "", "voice": "en"}, headers=HEADERS).status_code == 400
    too_long = "a" * (voice_app.MAX_TTS_CHARS + 1)
    assert client.post("/v1/synth", json={"text": too_long, "voice": "en"}, headers=HEADERS).status_code == 400


def test_tts_returns_a_playable_wav_and_refuses_unspeakable_text(client):
    r = client.post("/v1/tts", json={"text": "Hello there", "lang": "en"}, headers=HEADERS)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 22050 and w.getnframes() == 5  # both sentences, joined
    assert client.post("/v1/tts", json={"text": "???", "lang": "en"}, headers=HEADERS).status_code == 422
    assert client.post("/v1/tts", json={"text": "hi", "lang": "fr"}, headers=HEADERS).status_code == 400


@pytest.mark.skipif(not _PIPER_EN.exists(), reason="the Piper English voice file is not on this machine")
def test_synth_carries_what_a_real_piper_voice_produced_without_change(client, monkeypatch):
    """Piper's output differs a little from call to call (the VITS model adds random noise), so two syntheses cannot
    be compared; this records what the service's own call returned and checks the response holds exactly that."""
    from piper import PiperVoice

    class Recording:
        def __init__(self, voice):
            self.voice, self.seen = voice, []

        def synthesize(self, text):
            self.seen = list(self.voice.synthesize(text))
            return self.seen

    voice = Recording(PiperVoice.load(str(_PIPER_EN)))
    monkeypatch.setitem(voice_app.state, "tts", {"en": voice, "hi": voice})
    body = client.post("/v1/synth", json={"text": "Your EMI is due. Please pay today.", "voice": "en"}, headers=HEADERS).json()
    assert len(body["chunks"]) == len(voice.seen) > 1 and body["sample_rate"] == voice.seen[0].sample_rate
    for encoded, chunk in zip(body["chunks"], voice.seen):
        np.testing.assert_array_equal(np.frombuffer(base64.b64decode(encoded), dtype="<f4"), chunk.audio_float_array)
