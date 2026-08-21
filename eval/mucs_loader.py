"""Loads MUCS 2021's official held-out test split (Kaldi format: a
`segments` file mapping utterance IDs to time ranges within long
recordings, and a `text` file mapping utterance IDs to transcripts).

This is the test split MUCS ships separately from train -- never touched
by fine-tuning, which is what makes a before/after WER comparison honest.

Split into a cheap metadata listing and a per-utterance audio loader
(rather than one generator that loads everything) so callers can sample a
subset of utterance IDs before loading any audio -- holding all ~3,136
test utterances' audio in memory at once would cost 1-2GB, unnecessary
when only sampling a few hundred.
"""

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

_MUCS_TEST_DIR = Path(__file__).resolve().parents[1] / "datasets" / "openSLR" / "test"


@dataclass
class UtteranceMeta:
    utt_id: str
    recording_id: str
    start: float
    end: float
    reference_text: str


def list_test_utterances() -> list[UtteranceMeta]:
    """All utterance metadata (no audio) for utterances whose recording
    WAV file actually exists on disk."""
    segments = {}
    with open(_MUCS_TEST_DIR / "transcripts" / "segments", encoding="utf-8") as f:
        for line in f:
            utt_id, recording_id, start, end = line.split()
            segments[utt_id] = (recording_id, float(start), float(end))

    texts = {}
    with open(_MUCS_TEST_DIR / "transcripts" / "text", encoding="utf-8") as f:
        for line in f:
            utt_id, _, text = line.strip().partition(" ")
            texts[utt_id] = text

    result = []
    for utt_id, (recording_id, start, end) in segments.items():
        if not (_MUCS_TEST_DIR / f"{recording_id}.wav").exists():
            continue
        result.append(UtteranceMeta(utt_id, recording_id, start, end, texts.get(utt_id, "")))
    return result


def load_utterance_audio(meta: UtteranceMeta):
    wav_path = _MUCS_TEST_DIR / f"{meta.recording_id}.wav"
    info = sf.info(str(wav_path))
    start_frame = int(meta.start * info.samplerate)
    num_frames = int((meta.end - meta.start) * info.samplerate)
    audio, _sr = sf.read(str(wav_path), start=start_frame, frames=num_frames, dtype="float32")
    return audio
