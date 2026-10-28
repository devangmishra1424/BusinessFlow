"""Client for the voice service (voice_service/app.py): the Whisper decode and the Piper synthesis run on the voice VM
instead of in this process. Only the model calls move; silence trimming, script mixing, sentence gaps, level limiting
and Opus encoding stay in audio/asr.py, audio/tts.py and the channels, so what comes out is the same as running the
models here.

Configured by VOICE_SERVICE_URL and VOICE_SERVICE_API_KEY. With the URL unset, configured() is False and nothing in
this module is used: a machine that holds the models itself (the dev laptop, CI) behaves exactly as before.

When the service cannot answer, a call raises VoiceServiceUnavailable and the caller uses its local model. That is a
designed fallback, not a hidden one: the failure is logged at ERROR with what to check, and status() reports it.
Nothing is retried -- a retry would mostly fail the same way while a caller waits -- but after a failure that looks
like the service being down (unreachable, timeout, 5xx, a rejected key) calls skip it for _PAUSE_SECONDS, so each
turn does not also pay the connect timeout. A request the service refused as bad input (400/413/422) or as busy (429)
does not pause it. Plain urllib, like ops/speech_client.py: a few small POSTs with a timeout need no new dependency.
"""

import base64
import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_STT_TIMEOUT_SECONDS = 30.0  # a 60 s recording is the service's cap; a typical 5 s one takes 2-3 s there
_SYNTH_TIMEOUT_SECONDS = 20.0
_PAUSE_SECONDS = 30.0
_SAMPLE_RATE = 16000

# No proxy, ever: an HTTP_PROXY set on the machine for something else would otherwise route this call through it.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_lock = threading.Lock()
_paused_until = 0.0
_last_error: str | None = None
_failures = 0


class VoiceServiceUnavailable(Exception):
    """The voice service did not give a usable answer; the caller falls back to its local model."""


@dataclass
class Transcript:
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None
    language: str | None


def configured() -> bool:
    return bool(os.environ.get("VOICE_SERVICE_URL", "").strip())


def status() -> dict:
    """For a health page: whether the service is in use, whether calls are paused, and the last failure."""
    with _lock:
        paused = _paused_until > time.monotonic()
        return {"configured": configured(), "paused": paused, "failures": _failures, "last_error": _last_error}


def _fail(message: str, *, pause: bool) -> VoiceServiceUnavailable:
    global _paused_until, _last_error, _failures
    with _lock:
        _last_error = message
        _failures += 1
        if pause:
            _paused_until = time.monotonic() + _PAUSE_SECONDS
    return VoiceServiceUnavailable(message)


def _request(path: str, body: bytes, content_type: str, timeout: float) -> bytes:
    url_base = os.environ["VOICE_SERVICE_URL"].strip().rstrip("/")
    key = os.environ.get("VOICE_SERVICE_API_KEY", "").strip()
    if not key:
        logger.error(
            "VOICE_SERVICE_URL is set but VOICE_SERVICE_API_KEY is not -- set both in .env; using the local models, "
            "and pausing the service for %ds", _PAUSE_SECONDS,
        )
        raise _fail("VOICE_SERVICE_API_KEY is not set", pause=True)
    with _lock:
        if _paused_until > time.monotonic():
            raise VoiceServiceUnavailable(f"paused after: {_last_error}")
    request = urllib.request.Request(
        url_base + path, data=body, method="POST", headers={"Content-Type": content_type, "X-API-Key": key},
    )
    try:
        with _opener.open(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as e:  # before URLError: HTTPError is its subclass
        if e.code in (401, 403):
            logger.error(
                "voice service at %s rejected the key (%d) -- VOICE_SERVICE_API_KEY here must be one of VOICE_API_KEYS "
                "in /etc/voice.env on the voice VM; using the local models, and pausing the service for %ds",
                url_base, e.code, _PAUSE_SECONDS,
            )
            raise _fail(f"key rejected ({e.code})", pause=True) from e
        if e.code >= 500:
            logger.error(
                "voice service at %s failed (%d) on %s -- check `journalctl -u voice` on the voice VM; using the local "
                "models, and pausing the service for %ds", url_base, e.code, path, _PAUSE_SECONDS,
            )
            raise _fail(f"service error ({e.code})", pause=True) from e
        logger.error("voice service at %s refused %s (%d); using the local model for this call", url_base, path, e.code)
        raise _fail(f"request refused ({e.code})", pause=False) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        logger.error(
            "voice service unreachable at %s (%s) -- is voice-vm running, and does VOICE_SERVICE_URL match its DNS "
            "name? using the local models, and pausing the service for %ds",
            url_base, getattr(e, "reason", e), _PAUSE_SECONDS,
        )
        raise _fail(f"unreachable ({getattr(e, 'reason', e)})", pause=True) from e


def _multipart(fields: dict[str, str], filename: str, content: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode() + content + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(audio: np.ndarray, language: str | None) -> Transcript:
    """audio: mono float32 at 16 kHz (what asr.transcribe already has). Sent as a float WAV so the service decodes
    exactly these samples."""
    buf = io.BytesIO()
    sf.write(buf, audio, _SAMPLE_RATE, format="WAV", subtype="FLOAT")
    body, content_type = _multipart({"language": language} if language else {}, "audio.wav", buf.getvalue(), "audio/wav")
    raw = _request("/v1/stt", body, content_type, _STT_TIMEOUT_SECONDS)
    try:
        answer = json.loads(raw)
        return Transcript(
            text=answer["text"], avg_logprob=answer.get("avg_logprob"), no_speech_prob=answer.get("no_speech_prob"),
            language=answer.get("language"),
        )
    except (ValueError, KeyError) as e:
        raise _fail(f"unreadable answer ({type(e).__name__})", pause=True) from e


def synth(text: str, voice: str) -> tuple[list[np.ndarray], int | None]:
    """Piper's own output for text in the "en" or "hi" voice: one float32 array per sentence, and the sample rate
    (None when there was nothing to say)."""
    raw = _request("/v1/synth", json.dumps({"text": text, "voice": voice}).encode("utf-8"), "application/json", _SYNTH_TIMEOUT_SECONDS)
    try:
        answer = json.loads(raw)
        chunks = [np.frombuffer(base64.b64decode(c), dtype="<f4").astype(np.float32) for c in answer["chunks"]]
        return chunks, answer["sample_rate"]
    except (ValueError, KeyError, TypeError) as e:
        raise _fail(f"unreadable answer ({type(e).__name__})", pause=True) from e
