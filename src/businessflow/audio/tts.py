"""Text-to-speech: Piper for English (a local ONNX voice file, downloaded
separately into data/models/piper/ -- not bundled with pip install). Hindi
is Piper's hi_IN voice by default (auto-downloaded from Hugging Face on first
use, about 60 MB) with Meta MMS-TTS (via transformers, also auto-downloaded) as
the fallback, or the first choice with TTS_HINDI_ENGINE=mms. Every engine
returns the same shape -- a mono float32 tensor + its sample rate -- so
downstream code doesn't care which produced it.

Why Piper for Hindi (measured 2026-10-10 on the replies this bot speaks, CPU with
2 threads, kaggle/tts-comparison-kernel): about 5x faster than MMS, clearer to a
speech recogniser (character error 6.5-7% against MMS's 10.8%), ready in about
6 s instead of 28 s, and it needs no extra memory beyond the English voice,
where MMS held another ~420 MB in each process on a VM with 3.8 GB. A listener
ranked Piper's priyamvada voice above MMS. Kokoro Hindi ranked first by ear and
was the clearest (4-5%), but it costs roughly 700 MB and runs at about half real
time on those 2 threads, so it was not adopted for this VM.
"""

import io
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from piper import PiperVoice
from transformers import AutoTokenizer, VitsModel

logger = logging.getLogger(__name__)

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


_HINDI_ENGINES = ("piper", "mms")
_PIPER_HINDI_REPO = "rhasspy/piper-voices"


def _hindi_engine_order() -> list[str]:
    """TTS_HINDI_ENGINE picks the first choice; the other is the fallback. Read at call time so a deployment can
    flip it; a typo fails loudly rather than silently using the default."""
    preferred = os.environ.get("TTS_HINDI_ENGINE", "piper").strip().lower()
    if preferred not in _HINDI_ENGINES:
        raise ValueError(f"TTS_HINDI_ENGINE must be one of {_HINDI_ENGINES}, got {preferred!r}")
    return [preferred] + [e for e in _HINDI_ENGINES if e != preferred]


def _piper_hindi_voice_name() -> str:
    return os.environ.get("PIPER_HINDI_VOICE", "priyamvada").strip()  # rohan, pratham and priyamvada exist (medium)


@lru_cache(maxsize=4)
def _piper_hindi(voice_name: str) -> PiperVoice:
    """Piper's Hindi voice, fetched from Hugging Face on first use and cached by huggingface_hub after that (the
    same way the MMS weights are). The config file has to sit next to the model, so both are fetched."""
    filename = f"hi/hi_IN/{voice_name}/medium/hi_IN-{voice_name}-medium.onnx"
    model_path = hf_hub_download(_PIPER_HINDI_REPO, filename)
    hf_hub_download(_PIPER_HINDI_REPO, filename + ".json")
    return PiperVoice.load(model_path)


def _join_with_gaps(chunks: list[torch.Tensor], sample_rate: int, gap_seconds: float = _SENTENCE_GAP_SECONDS) -> torch.Tensor:
    """Concatenates chunks with gap_seconds (default _SENTENCE_GAP_SECONDS)
    of silence between each pair -- never a leading or trailing gap, and no
    gap at all for a single chunk (the common one-sentence reply case is
    unchanged)."""
    if not chunks:
        return torch.zeros(0)
    gap = torch.zeros(int(gap_seconds * sample_rate))
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


# The Hindi voice (MMS-TTS) only knows Devanagari: its tokenizer silently drops every Latin letter and
# digit, and a sentence with nothing it knows becomes an empty token sequence that crashes the model
# ("narrow(): length must be non-negative"). Before this, a Hindi reply written in Roman script
# ("Aapki agli kist...") therefore produced NO voice at all (Telegram logged the crash and said
# nothing), and a Latin word inside a Devanagari sentence ("UPI", "WhatsApp") was skipped without a
# trace. Now each sentence is split into runs by script: Devanagari runs go to the Hindi voice and Latin
# runs to the English voice, joined into one utterance.
_DEVANAGARI = re.compile(r"[\u0900-\u0963\u0970-\u097F]")  # letters and signs, not the danda (0964-0965) or digits (0966-096F)
# One or more consecutive Latin words (a word may hold digits, apostrophes, hyphens or &).
_LATIN_RUN = re.compile(r"([A-Za-z][A-Za-z0-9'\u2019&-]*(?:[ \t]+[A-Za-z][A-Za-z0-9'\u2019&-]*)*)")
# A short pause where the voice changes inside a sentence; shorter than the pause between sentences.
_RUN_GAP_SECONDS = 0.05


def _script_runs(sentence: str) -> list[tuple[str, str]]:
    """[("hi" | "en", text)] in order. Pieces with no letters of either script (punctuation, a bare
    digit run, a lone danda) are dropped: there is nothing to say, and verbalize() has already turned
    the numbers that matter into words."""
    runs: list[tuple[str, str]] = []
    for i, piece in enumerate(_LATIN_RUN.split(sentence)):
        piece = piece.strip()
        if not piece:
            continue
        if i % 2 == 1:
            runs.append(("en", piece))
        elif _DEVANAGARI.search(piece):
            runs.append(("hi", piece))
    return runs


def _synthesize_mms(tokenizer, model, text: str) -> torch.Tensor | None:
    """None when the Hindi voice has nothing it can say in `text` (only punctuation it does not know)."""
    inputs = tokenizer(text, return_tensors="pt")
    if inputs["input_ids"].numel() == 0:
        return None
    with torch.no_grad():
        return model(**inputs).waveform[0]


def _speak_hindi_run(run: str) -> tuple[torch.Tensor, int] | None:
    """One Devanagari run through the first Hindi engine that works: (audio, sample rate), or None when the engine has
    nothing it can say. The engines fail in unrelated ways (a Hugging Face download, onnxruntime, torch), so any
    failure of one means "try the next"; each is logged with its traceback, and if every engine fails the last error
    is raised."""
    last_error: Exception | None = None
    for engine in _hindi_engine_order():
        try:
            if engine == "piper":
                chunks = list(_piper_hindi(_piper_hindi_voice_name()).synthesize(run))
                if not chunks:
                    return None
                return torch.cat([torch.from_numpy(c.audio_float_array) for c in chunks]), chunks[0].sample_rate
            tokenizer, model = _mms_hindi()
            audio = _synthesize_mms(tokenizer, model, run)
            return None if audio is None else (audio, model.config.sampling_rate)
        except Exception as e:  # noqa: BLE001 -- see the docstring: any engine failure means "fall back", never "stay silent"
            logger.warning("Hindi voice engine %r failed for a run (%s); trying the next", engine, e, exc_info=True)
            last_error = e
    raise RuntimeError("no Hindi voice engine could speak this text") from last_error


def speak_hindi(text: str) -> Speech:
    """Hindi TTS (Piper's hi_IN voice, falling back to Meta's MMS-TTS; see the module docstring), with any Latin-script runs (English words, acronyms, or a whole
    reply the model wrote in Roman script) spoken by the English voice instead of being dropped.
    Downloads model weights on first call (cached by huggingface_hub after that, not re-downloaded
    per call). Synthesized one sentence at a time (see _split_hindi_sentences) and joined with the same
    deliberate inter-sentence pause speak_english uses.

    Raises ValueError when the text has nothing speakable at all (no letters), so a caller can tell the
    user the voice is unavailable instead of sending silence."""
    sentences = _split_hindi_sentences(text) or [text]  # whitespace-only input has no sentence boundary at all

    spoken: list[list[tuple[torch.Tensor, int]]] = []  # per sentence: [(audio, sample rate)] per run
    english_runs = 0
    for sentence in sentences:
        pieces: list[tuple[torch.Tensor, int]] = []
        for script, run in _script_runs(sentence):
            if script == "hi":
                hindi = _speak_hindi_run(run)
                if hindi is not None:
                    pieces.append(hindi)
            else:
                english = speak_english(run)
                pieces.append((english.audio, english.sample_rate))
                english_runs += 1
        if pieces:
            spoken.append(pieces)
    if not spoken:
        raise ValueError(f"speak_hindi: nothing speakable in {text[:60]!r} (no Devanagari or Latin letters)")
    if english_runs:
        logger.info("speak_hindi: %d Latin-script run(s) spoken with the English voice", english_runs)
    if not _DEVANAGARI.search(text):
        logger.warning("speak_hindi: the reply had no Devanagari at all (written in Roman script?); spoken entirely by the English voice")

    # One sample rate for the whole utterance: the higher of the two voices', so nothing is downsampled.
    target = max(rate for pieces in spoken for _, rate in pieces)

    def at_target(audio: torch.Tensor, rate: int) -> torch.Tensor:
        return audio if rate == target else torchaudio.functional.resample(audio, orig_freq=rate, new_freq=target)

    sentence_audio = [
        _join_with_gaps([at_target(a, r) for a, r in pieces], target, _RUN_GAP_SECONDS) for pieces in spoken
    ]
    return Speech(audio=_join_with_gaps(sentence_audio, target), sample_rate=target)


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
