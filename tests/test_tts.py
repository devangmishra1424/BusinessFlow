"""Tests for audio/tts.py's encode_ogg_opus -- the shared OGG/Opus encoder
every voice-reply channel (Telegram, browser) now goes through instead of
calling sf.write directly.

Regression coverage for a real bug found live via the browser voice
widget: Piper's en_US-lessac-medium voice outputs 22050Hz, which Opus
doesn't support at all (it only accepts 8000/12000/16000/24000/48000hz) --
sf.write(..., format="OGG", subtype="OPUS") raised LibsndfileError for
every English spoken reply on BOTH channels. Telegram's own call sites
caught the exception broadly and just silently dropped the voice reply
(never surfaced as a bug); the browser's speech endpoint had no such
catch, so it turned into a real 500 -- that's what surfaced this live.

Most of these need no real model -- encode_ogg_opus is exercised directly
against a synthetic Speech, no Piper/MMS-TTS loading required. The one
real-engine test is gated on the Piper voice file actually being present
(it's a large binary asset, not committed to git -- see .gitignore/git
history -- so CI won't have it; this repo's dev setup docs cover fetching
it locally).
"""

import io

import soundfile as sf
import torch

from businessflow.audio.tts import _OPUS_SAMPLE_RATES, _PIPER_VOICE_PATH, Speech, encode_ogg_opus, speak_english


def _silence_speech(sample_rate: int, seconds: float = 0.5) -> Speech:
    return Speech(audio=torch.zeros(int(sample_rate * seconds)), sample_rate=sample_rate)


def test_encode_ogg_opus_resamples_a_sample_rate_opus_does_not_support():
    # 22050Hz is exactly Piper's real, native rate -- the one that broke
    # live. Must not raise, and the encoded audio must come back at a rate
    # Opus actually supports.
    speech = _silence_speech(22050)

    result = encode_ogg_opus(speech)

    data, sr = sf.read(io.BytesIO(result))
    assert sr in _OPUS_SAMPLE_RATES
    assert len(data) > 0


def test_encode_ogg_opus_picks_the_smallest_sufficient_rate_not_always_the_max():
    # 22050 -> 24000 is the correct nearest-up choice; blindly clamping to
    # 48000 (the max) would still "work" but needlessly doubles the file
    # size for every English reply, so this pins the exact expected rate.
    speech = _silence_speech(22050)

    result = encode_ogg_opus(speech)

    _data, sr = sf.read(io.BytesIO(result))
    assert sr == 24000


def test_encode_ogg_opus_leaves_an_already_opus_valid_rate_unchanged():
    speech = _silence_speech(16000)

    result = encode_ogg_opus(speech)

    _data, sr = sf.read(io.BytesIO(result))
    assert sr == 16000


def test_encode_ogg_opus_handles_every_boundary_rate_opus_defines():
    # Every rate Opus itself supports must pass straight through with no
    # resample attempted (min() with strict ">=" must include equality).
    for rate in _OPUS_SAMPLE_RATES:
        result = encode_ogg_opus(_silence_speech(rate, seconds=0.1))
        _data, sr = sf.read(io.BytesIO(result))
        assert sr == rate


def test_speak_english_output_encodes_without_crashing():
    if not _PIPER_VOICE_PATH.exists():
        import pytest

        pytest.skip(f"Piper voice file not present at {_PIPER_VOICE_PATH} -- see dev setup docs")

    speech = speak_english("Your EMI is due soon.")

    result = encode_ogg_opus(speech)

    data, sr = sf.read(io.BytesIO(result))
    assert sr in _OPUS_SAMPLE_RATES
    assert len(data) > 0
