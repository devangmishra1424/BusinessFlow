"""Voice activity detection: trims silence off raw audio before it reaches
ASR. Wraps Silero VAD's ONNX model -- lighter than the torch/jit variant,
matching the blueprint's stated choice.
"""

from functools import lru_cache

import torch
from silero_vad import collect_chunks, get_speech_timestamps, load_silero_vad


@lru_cache(maxsize=1)
def _model():
    """Loads once per process, not once per call -- the model load itself
    is the expensive part, not running inference on it."""
    return load_silero_vad(onnx=True)


def detect_speech_segments(audio: torch.Tensor, sampling_rate: int = 16000) -> list[dict]:
    """Raw speech timestamps (start/end sample indices) in the given audio.
    Empty list means no speech was detected at all."""
    return get_speech_timestamps(audio, _model(), sampling_rate=sampling_rate)


def trim_to_speech(audio: torch.Tensor, sampling_rate: int = 16000) -> torch.Tensor:
    """Returns just the speech segments of the audio, concatenated, with
    silence dropped. Returns a zero-length tensor if no speech was found --
    callers should skip ASR entirely on that, not feed it empty audio."""
    segments = detect_speech_segments(audio, sampling_rate)
    if not segments:
        return torch.zeros(0, dtype=audio.dtype)
    return collect_chunks(segments, audio)