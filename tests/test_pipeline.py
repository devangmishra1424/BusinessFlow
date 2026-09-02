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

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set -- copy .env.example to .env and fill it in to run this",
    ),
    # Both tests below look up a real account (BF-1001) -- CI relies on an
    # earlier-run test file's reseed_accounts call having already seeded
    # it into CI's own disposable database (a single container shared for
    # the whole `pytest tests/` run, not reset between files); this file
    # never seeds it itself. Locally, DATABASE_URL is deliberately empty
    # whenever it would otherwise point at Supabase (see conftest.py) --
    # without this skip, that surfaced as a confusing "no account found"
    # failure instead of a clean skip like every other DB-dependent test.
    pytest.mark.skipif(
        not os.environ.get("DATABASE_URL"),
        reason="DATABASE_URL not set -- these tests hit a real account (BF-1001)",
    ),
]

# The real ASR default is a private model on the Hub (see audio/asr.py's
# _DEFAULT_MODEL_SIZE) -- transcribe() needs HF_TOKEN to download it.
# Real bug found live: CI has no HF_TOKEN configured, so
# test_voice_roundtrip_transcribes_and_speaks (which goes through the
# real voice_roundtrip -> transcribe() path, no mocking, same as every
# other test in this file) failed CI outright with a 401 the moment the
# model stopped being a bare, public size name. Consistent with this
# file's own GROQ_API_KEY convention: skip cleanly rather than fail when
# the credential isn't there, not mock the call.
_asr_skip = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set -- the real ASR default is a private model on the Hub and needs it to download",
)


def test_text_roundtrip_grounds_its_reply_in_a_real_tool_call():
    result = text_roundtrip("How many days past due is my payment?", language="en", account_id="BF-1001")

    assert result.transcript is None  # text path never transcribes anything
    assert "3" in result.reply_text  # Priya Sharma (BF-1001) is genuinely 3 days past due
    assert result.speech.audio.numel() > 0
    assert result.speech.sample_rate > 0


@_asr_skip
def test_voice_roundtrip_transcribes_and_speaks():
    audio = load_wav_as_tensor(_FIXTURE_EN)
    result = voice_roundtrip(audio, language="en", account_id="BF-1001")

    assert result.transcript is not None
    assert "payment" in result.transcript.lower()
    assert len(result.reply_text.strip()) > 0
    assert result.speech.audio.numel() > 0
