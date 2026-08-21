"""Loading raw audio files into the tensor format the rest of the audio
pipeline expects: mono, float32, normalized to [-1, 1].

Deliberately not using silero_vad's own read_audio() here -- on this
torchaudio version it requires an extra torchcodec backend (plus ffmpeg)
just to decode a WAV file, which is a lot of new dependency surface for
something soundfile already does directly.
"""

import soundfile as sf
import torch


def load_wav_as_tensor(path: str, expected_sampling_rate: int = 16000) -> torch.Tensor:
    """Loads a WAV file as a 1-D float32 tensor. Raises if the file's real
    sample rate doesn't match what's expected -- silently resampling would
    hide a mismatch that should be caught, not papered over."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # collapse to mono
    if sr != expected_sampling_rate:
        raise ValueError(
            f"{path} is {sr}Hz, expected {expected_sampling_rate}Hz -- "
            "resample it explicitly first, don't rely on this function to do it silently."
        )
    return torch.from_numpy(data)
