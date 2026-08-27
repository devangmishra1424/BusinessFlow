"""Real, measured RAM footprint of the full voice pipeline as it's
actually used in production (channels/telegram_bot.py): VAD -> ASR
("small", the real default transcribe() uses when no model_size is
passed) -> both TTS engines, all loaded into ONE process, matching what
the live bot process actually holds in memory simultaneously once it's
handled at least one voice-note round trip in each language.

Same subprocess-isolation methodology as compare_whisper_sizes.py --
each measurement (per stage here, per model size for the sibling script)
runs in a subprocess so nothing leaks between runs.

Run: python scripts/measure_full_pipeline_ram.py
"""

import json
import subprocess
import sys
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

_WORKER = '''
import json, sys
import psutil
import torch
from businessflow.audio.asr import transcribe
from businessflow.audio.io import load_wav_as_tensor
from businessflow.audio.tts import speak_english, speak_hindi
from businessflow.audio.vad import trim_to_speech
from businessflow.rag.retriever import DocumentRetriever

fixture_path = sys.argv[1]
process = psutil.Process()
stages = {}

stages["baseline"] = process.memory_info().rss

audio = load_wav_as_tensor(fixture_path)
trimmed = trim_to_speech(audio, sampling_rate=16000)
stages["after_vad"] = process.memory_info().rss

text = transcribe(trimmed, model_size="small")
stages["after_asr_small"] = process.memory_info().rss

speak_english("This is a test reply from the collections agent.")
stages["after_tts_english"] = process.memory_info().rss

speak_hindi("आपका EMI भुगतान लंबित है")
stages["after_tts_hindi"] = process.memory_info().rss

retriever = DocumentRetriever()
retriever.retrieve("what happens if I miss my EMI payment")
stages["after_rag_retriever"] = process.memory_info().rss

print(json.dumps({"stages_mb": {k: v / 1e6 for k, v in stages.items()}, "transcript": text}))
'''


def main():
    fixture = _FIXTURES / "sample_speech_en.wav"
    result = subprocess.run(
        [sys.executable, "-c", _WORKER, str(fixture)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout.strip().splitlines()[-1])
    stages = data["stages_mb"]

    print("Real, measured process RSS -- full voice pipeline, one process, cumulative:\n")
    prev = 0.0
    for label, rss in stages.items():
        delta = rss - prev
        print(f"  {label:<20} RSS={rss:>7.0f} MB   (+{delta:.0f} MB this stage)")
        prev = rss
    print(f"\nTotal process RSS with everything loaded: {prev:.0f} MB")


if __name__ == "__main__":
    main()
