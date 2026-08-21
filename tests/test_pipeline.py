"""End-to-end round-trip tests. These make a real Groq API call, so they're
skipped entirely if GROQ_API_KEY isn't set rather than mocking the call --
the whole point of this stage is proving the real round-trip works, and a
mock can't tell us that.
"""

import os

import pytest

from businessflow.audio.io import load_wav_as_tensor
from businessflow.pipeline import text_roundtrip, voice_roundtrip

_FIXTURE_EN = "tests/fixtures/sample_speech_en.wav"

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set -- copy .env.example to .env and fill it in to run this",
)


def test_text_roundtrip_grounds_its_reply_in_a_real_tool_call():
    result = text_roundtrip("How many days past due is my payment?", language="en", account_id="BF-1001")

    assert result.transcript is None  # text path never transcribes anything
    assert "3" in result.reply_text  # Priya Sharma (BF-1001) is genuinely 3 days past due
    assert result.speech.audio.numel() > 0
    assert result.speech.sample_rate > 0


def test_voice_roundtrip_transcribes_and_speaks():
    audio = load_wav_as_tensor(_FIXTURE_EN)
    result = voice_roundtrip(audio, language="en", account_id="BF-1001")

    assert result.transcript is not None
    assert "payment" in result.transcript.lower()
    assert len(result.reply_text.strip()) > 0
    assert result.speech.audio.numel() > 0
