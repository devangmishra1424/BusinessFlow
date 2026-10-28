"""Speech-to-text and text-to-speech as one small HTTP service, run on its own VM so the apps that need a voice
(BusinessFlow, VeriBook) do not each carry Whisper and Piper in their own memory. See voice_service/README.md for
how it is deployed and called.

Interior functions assume validated input; every endpoint validates at the edge (key, sizes, language, voice) before
any model runs. One model job runs at a time (the VM has 2 vCPUs), up to MAX_WAITING more may queue, and the rest
are told to come back later (429) rather than piling up.
"""

import asyncio
import base64
import glob
import io
import os
import re
import secrets
import time
import wave
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
MAX_TTS_CHARS = 2000
MAX_WAITING = 4
LANGUAGES = ("en", "hi")
VOICES = ("en", "hi")

# The same decode settings BusinessFlow uses in-process (businessflow/audio/asr.py, _DECODE_KWARGS, with the reasons
# for each). tests/test_voice_service.py fails if the two drift apart.
DECODE_KWARGS = dict(
    repetition_penalty=1.3, no_repeat_ngram_size=3, condition_on_previous_text=False,
    beam_size=10, best_of=10,
)

WHISPER_DIR = os.environ.get("WHISPER_DIR", "/opt/voice/models/whisper-base")
PIPER_DIR = os.environ.get("PIPER_DIR", "/opt/voice/models/piper")
KEYS = [k.strip() for k in os.environ.get("VOICE_API_KEYS", "").split(",") if k.strip()]

state: dict = {}
sem = asyncio.Semaphore(1)  # one heavy job at a time on 2 vCPUs
waiting = 0


@asynccontextmanager
async def lifespan(_app):
    from faster_whisper import WhisperModel
    from piper import PiperVoice

    state["asr"] = WhisperModel(WHISPER_DIR, device="cpu", compute_type="int8", cpu_threads=2)
    state["asr_name"] = os.path.basename(WHISPER_DIR.rstrip("/"))
    state["tts"] = {
        "en": PiperVoice.load(glob.glob(PIPER_DIR + "/**/en_US-lessac-medium.onnx", recursive=True)[0]),
        "hi": PiperVoice.load(glob.glob(PIPER_DIR + "/**/hi_IN-priyamvada-medium.onnx", recursive=True)[0]),
    }
    yield


app = FastAPI(title="voice", lifespan=lifespan)


def check_key(key: str | None) -> None:
    if not KEYS:
        raise HTTPException(503, "no API key configured")
    if not key or not any(secrets.compare_digest(key, k) for k in KEYS):
        raise HTTPException(401, "bad API key")


async def take_slot() -> None:
    global waiting
    if waiting >= MAX_WAITING:
        raise HTTPException(429, "busy, try again")
    waiting += 1
    try:
        await sem.acquire()
    finally:
        waiting -= 1


def decode(data: bytes):
    from faster_whisper.audio import decode_audio

    return decode_audio(io.BytesIO(data), sampling_rate=16000)


@app.get("/health")
def health():
    return {
        "ok": "asr" in state, "asr": "asr" in state, "asr_model": state.get("asr_name"),
        "tts": sorted(state.get("tts", {})),
    }


@app.post("/v1/stt")
async def stt(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    vad_filter: bool = Form(False),
    x_api_key: str | None = Header(None),
):
    """Transcribes one recording. vad_filter asks Whisper to drop silence itself, for callers (a browser mic) that
    send raw recordings; BusinessFlow trims silence before it sends and leaves it off."""
    check_key(x_api_key)
    if language not in (None, *LANGUAGES):
        raise HTTPException(400, f"language must be one of {LANGUAGES}")
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio too large")
    try:
        audio = decode(data)
    except Exception as e:  # noqa: BLE001 -- PyAV raises a different type per container/codec; all mean "not audio we can read"
        raise HTTPException(400, f"cannot decode audio: {type(e).__name__}") from e
    seconds = len(audio) / 16000
    if seconds > MAX_AUDIO_SECONDS:
        raise HTTPException(413, f"audio longer than {MAX_AUDIO_SECONDS} s")
    await take_slot()
    try:
        t0 = time.perf_counter()

        def run():
            segments, info = state["asr"].transcribe(audio, language=language, vad_filter=vad_filter, **DECODE_KWARGS)
            segments = list(segments)
            # Weighted by segment length, the same measure businessflow/audio/asr.py records.
            lengths = [max(len(s.text), 1) for s in segments]
            avg_logprob = no_speech_prob = None
            if segments:
                avg_logprob = round(sum(s.avg_logprob * n for s, n in zip(segments, lengths)) / sum(lengths), 3)
                no_speech_prob = round(max(s.no_speech_prob for s in segments), 3)
            return " ".join(s.text.strip() for s in segments), info.language, avg_logprob, no_speech_prob

        text, detected, avg_logprob, no_speech_prob = await asyncio.to_thread(run)
        return {
            "text": text, "language": detected, "audio_seconds": round(seconds, 2),
            "elapsed_s": round(time.perf_counter() - t0, 2), "avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob,
        }
    finally:
        sem.release()


class TtsIn(BaseModel):
    text: str
    lang: str = "en"


class SynthIn(BaseModel):
    text: str
    voice: str


def _checked_text(text: str) -> str:
    text = text.strip()
    if not text or len(text) > MAX_TTS_CHARS:
        raise HTTPException(400, f"text must be 1-{MAX_TTS_CHARS} characters")
    return text


def _speakable(text: str) -> bool:
    """False for text with no letter or digit in it ("???"), which a voice has nothing to say for."""
    return re.search(r"\w", text) is not None


@app.post("/v1/tts")
async def tts(body: TtsIn, x_api_key: str | None = Header(None)):
    """A finished WAV, for callers that just want to play the text (no sentence gaps or script mixing: that is
    BusinessFlow's own, and it uses /v1/synth)."""
    check_key(x_api_key)
    if body.lang not in VOICES:
        raise HTTPException(400, f"lang must be one of {VOICES}")
    text = _checked_text(body.text)
    if not _speakable(text):
        raise HTTPException(422, "nothing speakable in the text")
    await take_slot()
    try:
        def run():
            chunks = list(state["tts"][body.lang].synthesize(text))
            if not chunks:
                return None
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(chunks[0].sample_rate)
                w.writeframes(b"".join(c.audio_int16_bytes for c in chunks))
            return buf.getvalue()

        wav = await asyncio.to_thread(run)
        if wav is None:
            raise HTTPException(422, "nothing speakable in the text")
        return Response(content=wav, media_type="audio/wav")
    finally:
        sem.release()


@app.post("/v1/synth")
async def synth(body: SynthIn, x_api_key: str | None = Header(None)):
    """What the Piper voice produced for the text, untouched: one float32 little-endian array per sentence (base64)
    and the sample rate. BusinessFlow's own code joins them, mixes Hindi and English voices and limits the level, so
    the audio is the same as when it runs Piper itself. Text with nothing speakable returns no chunks, not an error."""
    check_key(x_api_key)
    if body.voice not in VOICES:
        raise HTTPException(400, f"voice must be one of {VOICES}")
    text = _checked_text(body.text)
    if not _speakable(text):
        return {"sample_rate": None, "chunks": []}
    await take_slot()
    try:
        def run():
            chunks = list(state["tts"][body.voice].synthesize(text))
            encoded = [base64.b64encode(c.audio_float_array.astype("<f4").tobytes()).decode("ascii") for c in chunks]
            return {"sample_rate": chunks[0].sample_rate if chunks else None, "chunks": encoded}

        return await asyncio.to_thread(run)
    finally:
        sem.release()
