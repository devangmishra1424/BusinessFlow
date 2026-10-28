"""Text-to-speech: Piper for English (a local ONNX voice file, downloaded
separately into data/models/piper/ -- not bundled with pip install). Hindi
is Piper's hi_IN voice by default (fetched from Hugging Face when a service
starts, about 60 MB) with Meta MMS-TTS (via transformers, also auto-downloaded) as
the fallback, or the first choice with TTS_HINDI_ENGINE=mms. Every engine
returns the same shape -- a mono float32 tensor + its sample rate -- so
downstream code doesn't care which produced it.

Why Piper for Hindi (eval/tts_benchmark.py, results in eval/results/tts_benchmark.md; 40 sentences
x 3 repeats (32 Devanagari scored for clarity, 8 Roman-script for robustness), criteria fixed beforehand, read back by Whisper large-v3): character error 7.3%
against MMS's 12.8% (the gain is in numbers and acronyms), 7x faster on 2 CPU threads (real-time factor
0.10 vs 0.74), steadier from one repeat to the next, and +138 MB in a fresh process against MMS's +175 MB
on a VM with 3.8 GB. A listener ranked Piper's priyamvada voice above MMS. Kokoro Hindi ranked first by
ear and was the clearest, but it costs roughly 700 MB and runs at about half real time on those 2
threads, so it was not adopted for this VM. The Hindi voice file (~63 MB) is fetched by prefetch_voices()
when a service starts, not during a reply.
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
from huggingface_hub.errors import LocalEntryNotFoundError
from piper import PiperVoice
from transformers import AutoTokenizer, VitsModel

from businessflow.audio import remote
from businessflow.audio.verbalizer import has_unverbalized_pattern
from businessflow.observability import trace

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


def _fetch_piper_hindi(voice_name: str, *, local_only: bool) -> str:
    """Path of the Hindi voice's model file; the config file has to sit next to it, so both are fetched."""
    filename = f"hi/hi_IN/{voice_name}/medium/hi_IN-{voice_name}-medium.onnx"
    model_path = hf_hub_download(_PIPER_HINDI_REPO, filename, local_files_only=local_only)
    hf_hub_download(_PIPER_HINDI_REPO, filename + ".json", local_files_only=local_only)
    return model_path


@lru_cache(maxsize=4)
def _piper_hindi(voice_name: str) -> PiperVoice:
    """Piper's Hindi voice. The cached copy is used without a network call (hf_hub_download otherwise asks
    Hugging Face whether the file changed, and waits out its timeout when the site is unreachable); only a voice
    that has never been fetched goes to the network -- normally prefetch_voices() did that at service start."""
    try:
        model_path = _fetch_piper_hindi(voice_name, local_only=True)
    except LocalEntryNotFoundError:
        model_path = _fetch_piper_hindi(voice_name, local_only=False)
    return PiperVoice.load(model_path)


def prefetch_voices() -> None:
    """Called once when a service starts: puts the Hindi voice file on disk so the first Hindi reply does not
    download 63 MB (in the Telegram bot that would also freeze every chat while it ran). Downloads only -- the
    model is not loaded, so this costs disk, not memory. Never raises for a download problem: the text replies
    do not depend on it and speak_hindi falls back to MMS-TTS. A bad TTS_HINDI_ENGINE does raise, here at start
    instead of at the first Hindi reply."""
    if _hindi_engine_order()[0] != "piper":
        return
    voice_name = _piper_hindi_voice_name()
    try:
        _fetch_piper_hindi(voice_name, local_only=False)
    except OSError:  # huggingface_hub's errors (no network, 404, timeout) all derive from OSError
        logger.error(
            "Could not fetch the Piper Hindi voice %r: Hindi replies will use the MMS-TTS fallback until it can be. "
            "Check that this machine can reach huggingface.co (or PIPER_HINDI_VOICE), then restart the service.",
            voice_name, exc_info=True,
        )
    else:
        logger.info("Piper Hindi voice %r is on disk", voice_name)


# Piper hands back audio normalised to a peak of exactly 1.0 (full scale), which leaves no headroom: after the lossy
# Opus step every voice reply goes through, the decoded waveform overshoots 1.0 on most clips (7 of 12 measured, up to
# 1.009) and a player or a 16-bit conversion clips it. Peaks are kept at -1 dBFS. Only ever turned DOWN, never up.
# -1 dBFS turned out not to be enough for Opus on its own: encode_ogg_opus limits further (_OPUS_PEAK_LIMIT).
_PEAK_LIMIT = 0.891  # 10 ** (-1 / 20)


def _limit_peak(audio: torch.Tensor, limit: float = _PEAK_LIMIT) -> torch.Tensor:
    """audio scaled down so its largest sample is at most `limit`; unchanged if it is already quieter."""
    if audio.numel() == 0:
        return audio
    peak = float(audio.abs().max())
    return audio if peak <= limit else audio * (limit / peak)


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


def _traced(engine: str, text: str, synthesize) -> Speech:
    """Runs one synthesis as a "tts" span (observability/trace.py): the engine, how much text, how much audio came out,
    and whether symbols reached the voice unconverted."""
    with trace.span("tts", engine=engine, chars=len(text), unverbalized=has_unverbalized_pattern(text)) as rec:
        speech = synthesize(text)
        rec["audio_seconds"] = round(speech.audio.shape[-1] / speech.sample_rate, 2)
        return speech


_SERVICE_HINDI_VOICE = "priyamvada"


def _piper_chunks(text: str, voice: str, local_voice, use_service: bool = True) -> tuple[list, int | None]:
    """Piper's output for text, one float32 array per sentence, and the sample rate (None when there was nothing to
    say). From the voice service when VOICE_SERVICE_URL is set (audio/remote.py); from `local_voice()`, the Piper voice
    loaded in this process, when it is not or when the service cannot answer. Both are the same model file, so it is the
    same voice either way (Piper adds a little random noise, so two syntheses of one text never match sample for
    sample, here or there)."""
    if use_service and remote.configured():
        try:
            return remote.synth(text, voice)
        except remote.VoiceServiceUnavailable:
            pass  # remote.py logged why at ERROR; the local voice below is the designed fallback
    chunks = list(local_voice().synthesize(text))
    return [c.audio_float_array for c in chunks], (chunks[0].sample_rate if chunks else None)


def speak_english(text: str) -> Speech:
    return _traced("piper_en", text, _speak_english)


def _speak_english(text: str) -> Speech:
    """English TTS via Piper. Fully local -- no network call at synthesis
    time, since the voice file is already on disk. Piper synthesizes one
    audio chunk per sentence -- joined here with a deliberate pause
    between them (see _SENTENCE_GAP_SECONDS) rather than Piper's own
    back-to-back concatenation."""
    chunks, sample_rate = _piper_chunks(text, "en", _piper_voice)
    if not chunks:
        raise ValueError(f"speak_english: nothing speakable in {text[:60]!r}")
    audio = _join_with_gaps([torch.from_numpy(c) for c in chunks], sample_rate)
    return Speech(audio=_limit_peak(audio), sample_rate=sample_rate)


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
                voice_name = _piper_hindi_voice_name()
                # The voice service holds priyamvada only; any other PIPER_HINDI_VOICE is spoken here.
                chunks, rate = _piper_chunks(
                    run, "hi", lambda: _piper_hindi(voice_name), use_service=voice_name == _SERVICE_HINDI_VOICE,
                )
                if not chunks:
                    return None
                return torch.cat([torch.from_numpy(c) for c in chunks]), rate
            tokenizer, model = _mms_hindi()
            audio = _synthesize_mms(tokenizer, model, run)
            return None if audio is None else (audio, model.config.sampling_rate)
        except Exception as e:  # noqa: BLE001 -- see the docstring: any engine failure means "fall back", never "stay silent"
            logger.warning("Hindi voice engine %r failed for a run (%s); trying the next", engine, e, exc_info=True)
            last_error = e
    raise RuntimeError("no Hindi voice engine could speak this text") from last_error


def speak_hindi(text: str) -> Speech:
    return _traced("hindi", text, _speak_hindi)


def _speak_hindi(text: str) -> Speech:
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
                english = _speak_english(run)  # untraced: part of this Hindi reply's one "tts" span
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
    return Speech(audio=_limit_peak(_join_with_gaps(sentence_audio, target)), sample_rate=target)


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

# Found in CI (2026-10-15, PR #18's run, a change that never touched audio): -1 dBFS was not enough headroom for Opus.
# A Hindi clip limited to 0.891 decoded at 1.021, so that voice note would clip on playback. Measured on 470 Piper
# clips: Opus alone raises the peak by up to 12.6% (the CI clip rose 14.6% in all), encode_ogg_opus's resample can
# add up to 4.5% before it, and a player decoding at 48 kHz peaks up to 3.8% above the 24 kHz decode soundfile
# gives -- by that measure 13 of the 470 (all Hindi) clipped on a phone at -1 dBFS. Limited to -2 dBFS AFTER the
# resample, the loudest of the 470 decoded at 0.914 (about 0.95 at 48 kHz). Voice notes are about 1 dB quieter than
# at -1 dBFS; agreed with the owner over -3 dBFS (2 dB quieter) and -1.5 dBFS (too little margin).
_OPUS_PEAK_LIMIT = 0.794  # 10 ** (-2 / 20)


def encode_ogg_opus(speech: Speech) -> bytes:
    """The one place a Speech becomes OGG/Opus bytes for every channel
    (Telegram, browser) to reuse, instead of each one calling sf.write
    directly and risking the crash above. Resamples up to the smallest
    Opus-supported rate that's still >= speech.sample_rate whenever the
    engine's native rate isn't already one of them -- never down, so this
    never throws away resolution the model actually produced. Then turns
    the peak down to _OPUS_PEAK_LIMIT (never up), so the decoded voice
    note stays under full scale whatever level the Speech arrives at."""
    audio, sample_rate = speech.audio, speech.sample_rate
    if sample_rate not in _OPUS_SAMPLE_RATES:
        target_rate = min((r for r in _OPUS_SAMPLE_RATES if r >= sample_rate), default=48000)
        audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=target_rate)
        sample_rate = target_rate
    audio = _limit_peak(audio, _OPUS_PEAK_LIMIT)
    buf = io.BytesIO()
    sf.write(buf, audio.numpy(), sample_rate, format="OGG", subtype="OPUS")
    return buf.getvalue()
