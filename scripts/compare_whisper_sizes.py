"""Compares faster-whisper 'small' vs 'base' on CPU int8: peak RSS after
loading + transcribing, and transcription latency. Each size runs as its
own subprocess so one model's memory footprint never contaminates the
other's measurement.

This only reports what's measurable against the synthetic SAPI fixtures
right now (RAM, latency). Real WER comparison needs the actual domain test
set (real recordings), which isn't ready yet -- don't read a transcript
match here as an accuracy result, only as "did it produce something."

Run: python scripts/compare_whisper_sizes.py
"""

import json
import subprocess
import sys
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

_WORKER = '''
import json, sys, time
import psutil
from businessflow.audio.asr import transcribe
from businessflow.audio.io import load_wav_as_tensor

model_size = sys.argv[1]
fixture_path = sys.argv[2]

process = psutil.Process()
rss_before = process.memory_info().rss

audio = load_wav_as_tensor(fixture_path)
start = time.perf_counter()
text = transcribe(audio, model_size=model_size)
latency_s = time.perf_counter() - start

rss_after = process.memory_info().rss

print(json.dumps({
    "model_size": model_size,
    "rss_before_mb": rss_before / 1e6,
    "rss_after_mb": rss_after / 1e6,
    "rss_delta_mb": (rss_after - rss_before) / 1e6,
    "latency_s": latency_s,
    "transcript": text,
}))
'''


def _run_one(model_size: str, fixture_path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _WORKER, model_size, str(fixture_path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    fixture = _FIXTURES / "sample_speech_en.wav"
    print(f"Comparing against: {fixture.name}\n")
    for model_size in ["small", "base"]:
        result = _run_one(model_size, fixture)
        print(f"--- {model_size} ---")
        print(f"  Process RSS after load+transcribe: {result['rss_after_mb']:.0f} MB")
        print(f"  RSS delta (model + inference only): {result['rss_delta_mb']:.0f} MB")
        print(f"  Transcription latency:              {result['latency_s']:.2f}s")
        print(f"  Transcript:                          {result['transcript']!r}")
        print()


if __name__ == "__main__":
    main()
