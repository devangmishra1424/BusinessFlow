"""Text-to-speech: Piper for English (a local ONNX voice file, downloaded
separately into data/models/piper/ -- not bundled with pip install), Meta
MMS-TTS for Hindi (via transformers, auto-downloaded from Hugging Face on
first use). Both return the same shape -- a mono float32 tensor + its
sample rate -- so downstream code doesn't care which engine produced it.
"""

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from piper import PiperVoice
from transformers import AutoTokenizer, VitsModel

_PIPER_VOICE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "piper" / "en_US-lessac-medium.onnx"
)


@dataclass
class Speech:
    audio: torch.Tensor
    sample_rate: int


@lru_cache(maxsize=1)
def _piper_voice() -> PiperVoice:
    return PiperVoice.load(str(_PIPER_VOICE_PATH))


@lru_cache(maxsize=1)
def _mms_hindi():
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
    model = VitsModel.from_pretrained("facebook/mms-tts-hin")
    model.eval()
    return tokenizer, model


def speak_english(text: str) -> Speech:
    """English TTS via Piper. Fully local -- no network call at synthesis
    time, since the voice file is already on disk."""
    voice = _piper_voice()
    chunks = list(voice.synthesize(text))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    return Speech(audio=torch.from_numpy(audio), sample_rate=chunks[0].sample_rate)


def speak_hindi(text: str) -> Speech:
    """Hindi TTS via Meta's MMS-TTS. Downloads model weights on first call
    (cached by huggingface_hub after that, not re-downloaded per call)."""
    tokenizer, model = _mms_hindi()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    return Speech(audio=output.waveform[0], sample_rate=model.config.sampling_rate)


# Opus (the codec every channel encodes spoken replies as, for compact
# voice-note-sized files) only accepts these five sample rates -- nothing
# else, full stop. Found live: Piper's en_US-lessac-medium voice outputs
# 22050Hz, which isn't one of them, so every English spoken reply (on
# Telegram AND the browser channel) was raising LibsndfileError("Opus only
# supports sample rates of...") and silently dropping the voice reply
# (telegram_bot.py's _send_spoken_reply/on_voice_message both catch/log
# broadly around their TTS call) or 500ing (browser_api.py's speech
# endpoint, which has no such catch -- callers see a real error instead of
# a silently missing reply).
_OPUS_SAMPLE_RATES = (8000, 12000, 16000, 24000, 48000)


def encode_ogg_opus(speech: Speech) -> bytes:
    """The one place a Speech becomes OGG/Opus bytes for every channel
    (Telegram, browser) to reuse, instead of each one calling sf.write
    directly and risking the crash above. Resamples up to the smallest
    Opus-supported rate that's still >= speech.sample_rate whenever the
    engine's native rate isn't already one of them -- never down, so this
    never throws away resolution the model actually produced."""
    audio, sample_rate = speech.audio, speech.sample_rate
    if sample_rate not in _OPUS_SAMPLE_RATES:
        target_rate = min((r for r in _OPUS_SAMPLE_RATES if r >= sample_rate), default=48000)
        audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=target_rate)
        sample_rate = target_rate
    buf = io.BytesIO()
    sf.write(buf, audio.numpy(), sample_rate, format="OGG", subtype="OPUS")
    return buf.getvalue()
