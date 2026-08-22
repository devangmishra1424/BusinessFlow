"""Speech-to-text via faster-whisper, CPU int8 -- the compute_type already
verified against the project's RAM budget (see the blueprint's benchmark
citation). Model size ("small" vs "base") is a parameter, not hardcoded,
specifically so scripts/compare_whisper_sizes.py can load and compare both
in the same process.
"""

from functools import lru_cache

import numpy as np
import torch
from faster_whisper import WhisperModel


@lru_cache(maxsize=None)
def _model(model_size: str) -> WhisperModel:
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(audio: torch.Tensor, model_size: str = "small", language: str | None = None) -> str:
    """Transcribes a mono float32 audio tensor (as produced by
    businessflow.audio.io.load_wav_as_tensor or vad.trim_to_speech) to text.
    language=None lets Whisper auto-detect; pass "hi" or "en" to force it.

    repetition_penalty/no_repeat_ngram_size/condition_on_previous_text=False
    are set explicitly rather than left at faster-whisper's defaults
    (1, 0, True) -- found live against the fine-tuned model on real MUCS
    audio: without them, several utterances degenerated into long
    repeated-syllable hallucination loops (e.g. "प्प्रिंट्ट प्रिंटे टे
    नलोग्वा इलोग..."), which also happened to swallow the English
    loanwords the reference transcripts correctly keep in Latin script.
    With these set, the loops stopped and more loanwords came back
    correctly in Latin script too -- these aren't proven to be two
    separate bugs, the decoding-time fix may address both."""
    audio_np: np.ndarray = audio.numpy() if isinstance(audio, torch.Tensor) else audio
    segments, _info = _model(model_size).transcribe(
        audio_np, language=language,
        repetition_penalty=1.3, no_repeat_ngram_size=3, condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments)
