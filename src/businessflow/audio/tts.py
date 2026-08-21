"""Text-to-speech: Piper for English (a local ONNX voice file, downloaded
separately into data/models/piper/ -- not bundled with pip install), Meta
MMS-TTS for Hindi (via transformers, auto-downloaded from Hugging Face on
first use). Both return the same shape -- a mono float32 tensor + its
sample rate -- so downstream code doesn't care which engine produced it.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
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
