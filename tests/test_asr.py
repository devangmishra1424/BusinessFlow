"""Tests for the ASR stage. Two fixtures: a clean English sentence (SAPI's
default voice handles plain English well, so this checks real transcription
accuracy, not just "did it crash"), and the Hindi-in-Roman-script sentence
reused from the VAD test (SAPI reads Hindi words with an English voice, so
this only checks robustness -- that ASR returns *something* reasonable on
code-switched-looking input, not an exact transcript match).
"""

from pathlib import Path

import pytest

from businessflow.audio.asr import _DECODE_KWARGS, _DEFAULT_MODEL_SIZE, transcribe
from businessflow.audio.io import load_wav_as_tensor

_FIXTURES = Path(__file__).parent / "fixtures"


def test_default_model_is_the_real_fine_tune_not_a_bare_size_name():
    # Regression test for a real bug: transcribe()'s default silently
    # fell back to the generic, un-fine-tuned "small" model, because
    # nothing overrode it -- the live Telegram bot's own call site never
    # passed model_size at all. A bare size name ("small"/"base"/"medium"/
    # etc, not a HF repo id containing "/") would mean this has regressed
    # back to that.
    assert "/" in _DEFAULT_MODEL_SIZE
    assert _DEFAULT_MODEL_SIZE not in {"tiny", "base", "small", "medium", "large"}


def test_decode_kwargs_use_the_confirmed_wider_beam_config():
    # Regression test for eval/decode_params_sweep.json's confirmed win:
    # beam_size=10/best_of=10 (faster-whisper's own default is 5/5) cut
    # WER ~4.5% relative (53.2% -> 50.8%) over the production settings
    # alone, on the same r32 model and MUCS test set. Nothing pinned this
    # before -- a future edit to transcribe() could silently drop back to
    # faster-whisper's plain defaults with nothing to catch it.
    assert _DECODE_KWARGS["beam_size"] == 10
    assert _DECODE_KWARGS["best_of"] == 10
    # And the earlier repetition-loop fix (see asr.py's own comment) must
    # survive alongside it, not get dropped when the dict was introduced.
    assert _DECODE_KWARGS["repetition_penalty"] == 1.3
    assert _DECODE_KWARGS["no_repeat_ngram_size"] == 3
    assert _DECODE_KWARGS["condition_on_previous_text"] is False


@pytest.mark.skipif(
    not (_FIXTURES / "sample_speech_en.wav").exists(),
    reason="fixture not present -- generated once locally via Windows SAPI TTS (scripts don't ship it; "
           "*.wav is gitignored), and voice is out of scope for now anyway",
)
def test_transcribes_clean_english_accurately():
    audio = load_wav_as_tensor(str(_FIXTURES / "sample_speech_en.wav"))
    text = transcribe(audio, model_size="small", language="en").lower()

    assert "payment" in text
    assert "rupees" in text or "rupee" in text
    assert "three" in text


@pytest.mark.skipif(
    not (_FIXTURES / "sample_speech.wav").exists(),
    reason="fixture not present -- generated once locally via Windows SAPI TTS (scripts don't ship it; "
           "*.wav is gitignored), and voice is out of scope for now anyway",
)
def test_transcribes_something_on_code_switched_sample():
    audio = load_wav_as_tensor(str(_FIXTURES / "sample_speech.wav"))
    text = transcribe(audio, model_size="small")

    assert len(text.strip()) > 0
