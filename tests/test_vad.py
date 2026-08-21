"""Tests for the VAD stage. Uses a real synthesized speech sample (Windows
SAPI TTS, saved once to tests/fixtures/) rather than a synthetic tone --
Silero VAD is a real neural speech detector, so a sine wave wouldn't
reliably register as speech and would prove nothing.
"""

from pathlib import Path

import torch

from businessflow.audio.io import load_wav_as_tensor
from businessflow.audio.vad import trim_to_speech

_FIXTURE = Path(__file__).parent / "fixtures" / "sample_speech.wav"


def test_trims_real_speech_and_keeps_most_of_it():
    audio = load_wav_as_tensor(str(_FIXTURE), expected_sampling_rate=16000)
    trimmed = trim_to_speech(audio, sampling_rate=16000)

    assert trimmed.numel() > 0, "expected real speech to be detected"
    assert trimmed.numel() <= audio.numel(), "trimming should never add samples"
    # The sample is ~3.9s of continuous speech with no long silence gaps --
    # trimming should keep the bulk of it, not shave off nearly everything.
    assert trimmed.numel() > audio.numel() * 0.5


def test_returns_empty_on_pure_silence():
    silence = torch.zeros(16000 * 2)  # 2 seconds of digital silence
    trimmed = trim_to_speech(silence, sampling_rate=16000)

    assert trimmed.numel() == 0