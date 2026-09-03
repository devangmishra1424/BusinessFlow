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
against a synthetic Speech, no Piper/MMS-TTS loading required. Real-engine
tests that need Piper specifically are gated on the voice file actually
being present locally (it's a large binary asset, not committed to git --
see .gitignore/git history); CI fetches it itself as a workflow step, so
those aren't skipped there. speak_hindi needs no such gate -- MMS-TTS
auto-downloads from Hugging Face on first use with no local asset to
check for, the same as this project's ASR model.

_join_with_gaps/_split_hindi_sentences (the inter-sentence pause logic
added for prosody -- see tts.py's own docstring) are tested against
synthetic tensors/plain strings, not real synthesis output: Piper and
MMS-TTS are both genuinely non-deterministic model-to-model (confirmed
live -- two calls with identical text produced different sample counts),
so asserting an exact sample count against a real second synthesis call
would be flaky. The real-engine tests below stay at the "doesn't crash,
plausible shape" smoke-test level instead.
"""

import io

import pytest
import soundfile as sf
import torch

from businessflow.audio.tts import (
    _OPUS_SAMPLE_RATES,
    _PIPER_VOICE_PATH,
    _SENTENCE_GAP_SECONDS,
    Speech,
    _join_with_gaps,
    _split_hindi_sentences,
    encode_ogg_opus,
    speak_english,
    speak_hindi,
)


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


_piper_skip = pytest.mark.skipif(
    not _PIPER_VOICE_PATH.exists(), reason=f"Piper voice file not present at {_PIPER_VOICE_PATH} -- see dev setup docs",
)


def test_speak_english_output_encodes_without_crashing():
    if not _PIPER_VOICE_PATH.exists():
        pytest.skip(f"Piper voice file not present at {_PIPER_VOICE_PATH} -- see dev setup docs")

    speech = speak_english("Your EMI is due soon.")

    result = encode_ogg_opus(speech)

    data, sr = sf.read(io.BytesIO(result))
    assert sr in _OPUS_SAMPLE_RATES
    assert len(data) > 0


# ---------------------------------------------------------------------------
# _join_with_gaps -- pure logic, synthetic tensors, no model involved
# ---------------------------------------------------------------------------


def test_join_with_gaps_inserts_silence_between_every_pair_of_chunks():
    chunks = [torch.ones(3), torch.full((3,), 2.0), torch.full((3,), 3.0)]
    sample_rate = 10  # 0.2s gap * 10Hz = 2 samples per gap

    result = _join_with_gaps(chunks, sample_rate)

    expected = torch.cat([torch.ones(3), torch.zeros(2), torch.full((3,), 2.0), torch.zeros(2), torch.full((3,), 3.0)])
    assert torch.equal(result, expected)


def test_join_with_gaps_adds_no_gap_for_a_single_chunk():
    # The common one-sentence reply case must come back byte-identical to
    # its own input -- no leading/trailing silence invented out of thin air.
    chunk = torch.ones(5)

    result = _join_with_gaps([chunk], sample_rate=16000)

    assert torch.equal(result, chunk)


def test_join_with_gaps_returns_empty_for_no_chunks():
    result = _join_with_gaps([], sample_rate=16000)

    assert len(result) == 0


def test_join_with_gaps_gap_length_matches_the_configured_seconds():
    chunks = [torch.ones(1), torch.ones(1)]
    sample_rate = 48000

    result = _join_with_gaps(chunks, sample_rate)

    assert len(result) == 2 + int(_SENTENCE_GAP_SECONDS * sample_rate)


# ---------------------------------------------------------------------------
# _split_hindi_sentences -- pure logic, plain strings, no model involved
# ---------------------------------------------------------------------------


def test_split_hindi_sentences_splits_on_the_danda():
    text = "आपकी ईएमआई जल्द देय है। कृपया समय पर भुगतान करें।"

    result = _split_hindi_sentences(text)

    assert result == ["आपकी ईएमआई जल्द देय है।", "कृपया समय पर भुगतान करें।"]


def test_split_hindi_sentences_also_splits_on_question_and_exclamation_marks():
    text = "क्या आप ठीक हैं? हाँ, मैं ठीक हूँ!"

    result = _split_hindi_sentences(text)

    assert result == ["क्या आप ठीक हैं?", "हाँ, मैं ठीक हूँ!"]


def test_split_hindi_sentences_does_not_split_on_a_bare_period():
    # Hindi doesn't conventionally end a sentence with "." -- any decimal
    # point reaching here survives only because verbalizer.py missed it,
    # and this must not mis-split that number in half either way.
    text = "राशि 12.5 है।"

    result = _split_hindi_sentences(text)

    assert result == ["राशि 12.5 है।"]


def test_split_hindi_sentences_returns_one_item_with_no_sentence_boundary_at_all():
    result = _split_hindi_sentences("no danda here")

    assert result == ["no danda here"]


def test_split_hindi_sentences_returns_empty_for_whitespace_only():
    assert _split_hindi_sentences("   ") == []


# ---------------------------------------------------------------------------
# Real-engine smoke tests -- doesn't crash, plausible output shape. No
# exact-duration assertions against a second real synthesis call: see this
# module's own docstring for why that would be flaky.
# ---------------------------------------------------------------------------


@_piper_skip
def test_speak_english_multi_sentence_input_produces_longer_audio_than_one_sentence_alone():
    single = speak_english("Your EMI is due soon.")
    multi = speak_english("Your EMI is due soon. Please pay on time. Thank you.")

    assert len(multi.audio) > len(single.audio)
    assert multi.sample_rate == single.sample_rate


def test_speak_hindi_multi_sentence_input_produces_longer_audio_than_one_sentence_alone():
    single = speak_hindi("आपकी ईएमआई जल्द देय है।")
    multi = speak_hindi("आपकी ईएमआई जल्द देय है। कृपया समय पर भुगतान करें।")

    assert len(multi.audio) > len(single.audio)
    assert multi.sample_rate == single.sample_rate


def test_speak_hindi_output_encodes_without_crashing():
    speech = speak_hindi("आपकी ईएमआई जल्द देय है। कृपया समय पर भुगतान करें।")

    result = encode_ogg_opus(speech)

    data, sr = sf.read(io.BytesIO(result))
    assert sr in _OPUS_SAMPLE_RATES
    assert len(data) > 0
