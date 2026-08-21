"""Tests for the ASR stage. Two fixtures: a clean English sentence (SAPI's
default voice handles plain English well, so this checks real transcription
accuracy, not just "did it crash"), and the Hindi-in-Roman-script sentence
reused from the VAD test (SAPI reads Hindi words with an English voice, so
this only checks robustness -- that ASR returns *something* reasonable on
code-switched-looking input, not an exact transcript match).
"""

from pathlib import Path

from businessflow.audio.asr import transcribe
from businessflow.audio.io import load_wav_as_tensor

_FIXTURES = Path(__file__).parent / "fixtures"


def test_transcribes_clean_english_accurately():
    audio = load_wav_as_tensor(str(_FIXTURES / "sample_speech_en.wav"))
    text = transcribe(audio, model_size="small", language="en").lower()

    assert "payment" in text
    assert "rupees" in text or "rupee" in text
    assert "three" in text


def test_transcribes_something_on_code_switched_sample():
    audio = load_wav_as_tensor(str(_FIXTURES / "sample_speech.wav"))
    text = transcribe(audio, model_size="small")

    assert len(text.strip()) > 0
