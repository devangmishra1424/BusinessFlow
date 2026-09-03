"""Text-to-speech: Piper for English (a local ONNX voice file, downloaded
separately into data/models/piper/ -- not bundled with pip install), Meta
MMS-TTS for Hindi (via transformers, auto-downloaded from Hugging Face on
first use). Both return the same shape -- a mono float32 tensor + its
sample rate -- so downstream code doesn't care which engine produced it.
"""

import io
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from piper import PiperVoice
from transformers import AutoTokenizer, VitsModel

_PIPER_VOICE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "piper" / "en_US-lessac-medium.onnx"
)

# Neither engine puts any gap between sentences on its own: Piper already
# synthesizes one audio chunk per sentence, but speak_english used to
# concatenate them back-to-back with zero pause; MMS-TTS synthesizes
# whatever text it's given in a single pass, so a multi-sentence reply
# came out as one continuous, unpaused utterance. Both read as
# rushed/run-on rather than a natural conversational cadence. 0.2s is a
# reasonable middle value for an inter-sentence pause, not yet
# empirically tuned against a real naturalness score -- see
# eval/voice_naturalness_benchmark.py, which exists specifically to check
# a change like this actually helps rather than just assuming it does.
_SENTENCE_GAP_SECONDS = 0.2


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


def _join_with_gaps(chunks: list[torch.Tensor], sample_rate: int) -> torch.Tensor:
    """Concatenates chunks with _SENTENCE_GAP_SECONDS of silence between
    each pair -- never a leading or trailing gap, and no gap at all for a
    single chunk (the common one-sentence reply case is unchanged)."""
    if not chunks:
        return torch.zeros(0)
    gap = torch.zeros(int(_SENTENCE_GAP_SECONDS * sample_rate))
    parts = [chunks[0]]
    for chunk in chunks[1:]:
        parts.append(gap)
        parts.append(chunk)
    return torch.cat(parts)


def speak_english(text: str) -> Speech:
    """English TTS via Piper. Fully local -- no network call at synthesis
    time, since the voice file is already on disk. Piper synthesizes one
    audio chunk per sentence -- joined here with a deliberate pause
    between them (see _SENTENCE_GAP_SECONDS) rather than Piper's own
    back-to-back concatenation."""
    voice = _piper_voice()
    chunks = list(voice.synthesize(text))
    sample_rate = chunks[0].sample_rate
    audio = _join_with_gaps([torch.from_numpy(c.audio_float_array) for c in chunks], sample_rate)
    return Speech(audio=audio, sample_rate=sample_rate)


# Hindi's own sentence terminator is the danda ("।"), not a period -- "."
# is deliberately not split on here: Hindi doesn't conventionally use it
# to end a sentence, and any decimal point that reaches this function
# survives only because verbalizer.py missed it (a bug to fix there, not
# something this should paper over by mis-splitting a number in half).
# "?"/"!" are also split on for a borrowed-punctuation or mixed-script
# sentence. The terminator itself stays attached to its own sentence (a
# lookbehind, not a capturing split) so TTS still "hears" it.
_HINDI_SENTENCE_SPLIT = re.compile(r"(?<=[।?!])\s+")


def _split_hindi_sentences(text: str) -> list[str]:
    return [s for s in (part.strip() for part in _HINDI_SENTENCE_SPLIT.split(text.strip())) if s]


def speak_hindi(text: str) -> Speech:
    """Hindi TTS via Meta's MMS-TTS. Downloads model weights on first call
    (cached by huggingface_hub after that, not re-downloaded per call).
    Synthesized one sentence at a time (see _split_hindi_sentences) and
    joined with the same deliberate inter-sentence pause speak_english
    uses, instead of one single-shot pass over the whole reply."""
    tokenizer, model = _mms_hindi()
    sample_rate = model.config.sampling_rate
    sentences = _split_hindi_sentences(text) or [text]  # whitespace-only input has no sentence boundary at all

    chunks = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        chunks.append(output.waveform[0])
    audio = _join_with_gaps(chunks, sample_rate)
    return Speech(audio=audio, sample_rate=sample_rate)


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
