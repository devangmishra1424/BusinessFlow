"""Speech-to-text via faster-whisper, CPU int8 -- the compute_type already
verified against the project's RAM budget (see the blueprint's benchmark
citation). model_size is a parameter, not hardcoded, specifically so
scripts/compare_whisper_sizes.py can load and compare several in the same
process -- but transcribe()'s own default is the real fine-tuned model
(see _DEFAULT_MODEL_SIZE below), not a bare size name like "small"/"base".
"""

import os
from functools import lru_cache

import numpy as np
import torch
from faster_whisper import WhisperModel

# The real fine-tuned model (Hindi-English code-switched, full corpus,
# LoRA rank 32) -- trained on Kaggle, converted to CTranslate2, benchmarked
# against the base "small" model on real MUCS test audio (see
# eval/wer_benchmark.py's results): WER 0.532 vs the base model's 1.008,
# roughly a 47% relative reduction. That fine-tune had existed for a while
# but was never actually wired into transcribe()'s default -- the live
# Telegram bot was calling transcribe() with no model_size override at
# all, silently falling back to the generic, un-fine-tuned "small" this
# whole time. Hosted on the Hub (not committed to git -- 278MB of binary
# weights) since faster-whisper/huggingface_hub can load a CTranslate2
# model directly by repo id, private-repo auth handled automatically via
# the HF_TOKEN environment variable already required for other ASR/dataset
# access (see .env.example). Override via WHISPER_MODEL_SIZE for anything
# that genuinely wants a different model (the comparison scripts already
# pass their own model_size explicitly and are unaffected by this).
_DEFAULT_MODEL_SIZE = os.environ.get(
    "WHISPER_MODEL_SIZE", "CaffeinatedCoding/businessflow-whisper-hi-en"
)

# repetition_penalty/no_repeat_ngram_size/condition_on_previous_text=False were
# found live against the fine-tuned model on real MUCS audio: without them,
# several utterances degenerated into long repeated-syllable hallucination
# loops (e.g. "प्प्रिंट्ट प्रिंटे टे नलोग्वा इलोग..."), which also swallowed
# the English loanwords the reference transcripts correctly keep in Latin
# script. beam_size/best_of=10 (faster-whisper's default is 5) is the
# separate, later "wider_beam" win from eval/decode_params_sweep.json --
# confirmed on the same r32 model and the same MUCS test set to cut WER a
# further ~4.5% relative (53.2% -> 50.8%) on top of the settings above, at
# the cost of ~29% more CPU time per transcription. That cost is real on
# this project's CPU-only int8 deployment, but still small next to the
# ~40-55s a real conversation turn spends on the LLM/tool-calling round trip
# (see eval/results/latency_benchmark.json), so the WER win is taken. Kept
# in one dict, not inlined into transcribe(), so a test can pin these
# without loading a real model (see test_asr.py).
_DECODE_KWARGS = dict(
    repetition_penalty=1.3, no_repeat_ngram_size=3, condition_on_previous_text=False,
    beam_size=10, best_of=10,
)


@lru_cache(maxsize=None)
def _model(model_size: str) -> WhisperModel:
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(audio: torch.Tensor, model_size: str = _DEFAULT_MODEL_SIZE, language: str | None = None) -> str:
    """Transcribes a mono float32 audio tensor (as produced by
    businessflow.audio.io.load_wav_as_tensor or vad.trim_to_speech) to text.
    language=None lets Whisper auto-detect; pass "hi" or "en" to force it.
    See _DECODE_KWARGS above for why each decode-time override is set."""
    audio_np: np.ndarray = audio.numpy() if isinstance(audio, torch.Tensor) else audio
    segments, _info = _model(model_size).transcribe(audio_np, language=language, **_DECODE_KWARGS)
    return " ".join(segment.text.strip() for segment in segments)
