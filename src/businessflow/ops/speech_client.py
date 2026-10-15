"""Speech-to-text and text-to-speech for the ops dashboard's voice questions,
borrowed from the chat service instead of loaded here: the chat process
already holds Whisper and Piper, and a second copy of each in this process
would cost about 265 MB on a 3.8 GB VM (see the "Speech for the ops
dashboard" section of channels/browser_api.py, which serves these calls and
explains the two gates on them).

The two services run side by side on one machine, so this talks to the chat
service directly over loopback (CHAT_INTERNAL_URL, default
http://127.0.0.1:8000 -- the chat app's port in README's run instructions),
never through Caddy. Plain urllib, deliberately: two small POSTs with a
timeout don't need a new HTTP-client dependency.

Nothing is retried. Over loopback, a refused connection means the chat
service is down or restarting (a deploy restarts it), and a retry a moment
later would mostly fail the same way while the operator waits; the caller
says "voice is unavailable" at once instead, and the rest of the dashboard is
unaffected.
"""

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_DEFAULT_CHAT_INTERNAL_URL = "http://127.0.0.1:8000"
# The first question after a restart also loads Whisper in the chat process (1.7 s on the dev machine, longer on the
# VM's 2 vCPUs), and a borrower's voice message being transcribed at the same moment goes first. 60 s is a generous
# bound on that, not an expected wait: a 5 s question takes about 1 s to transcribe on the dev machine.
_TRANSCRIBE_TIMEOUT_SECONDS = 60.0
_SPEAK_TIMEOUT_SECONDS = 30.0

# No proxy, ever: an HTTP_PROXY set on the machine for something else would otherwise route this loopback call through it.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class SpeechServiceUnavailable(Exception):
    """The chat service didn't answer, or failed. Voice is unavailable until it does; nothing else is affected."""


class UnreadableRecording(ValueError):
    """The chat service read the request and refused the audio itself (not decodable, or too large)."""


def _base_url() -> str:
    return (os.environ.get("CHAT_INTERNAL_URL") or _DEFAULT_CHAT_INTERNAL_URL).rstrip("/")  # set but empty, as .env.example leaves it, means the default


def _post(path: str, body: bytes, content_type: str, timeout: float) -> bytes:
    key = os.environ.get("OPS_API_KEY")
    if not key:
        raise RuntimeError("OPS_API_KEY is not set -- copy .env.example to .env and fill it in")
    url = f"{_base_url()}{path}"
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": content_type, "X-API-Key": key},
    )
    try:
        with _opener.open(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as e:  # before URLError: HTTPError is its subclass
        if e.code in (400, 413):
            raise UnreadableRecording(f"the speech service refused the audio ({e.code})") from e
        logger.error(
            "ops voice: speech service at %s answered %d -- if 401/403, the chat service's OPS_API_KEY differs from "
            "this one's, or CHAT_INTERNAL_URL points through the public proxy", url, e.code,
        )
        raise SpeechServiceUnavailable(f"speech service answered {e.code}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        logger.error(
            "ops voice: speech service unreachable at %s (%s) -- is the chat service running on that address?",
            url, getattr(e, "reason", e),
        )
        raise SpeechServiceUnavailable("speech service unreachable") from e


def transcribe(wav_bytes: bytes) -> str | None:
    """What was said in a WAV recording (English), or None when no speech was found in it (or it ran past 30 s)."""
    raw = _post("/internal/ops-voice/transcribe", wav_bytes, "audio/wav", _TRANSCRIBE_TIMEOUT_SECONDS)
    return json.loads(raw)["transcript"]


def speak(text: str) -> bytes:
    """OGG/Opus audio of text, in the same English voice the borrower chat uses."""
    body = json.dumps({"text": text}).encode("utf-8")
    return _post("/internal/ops-voice/speak", body, "application/json", _SPEAK_TIMEOUT_SECONDS)
