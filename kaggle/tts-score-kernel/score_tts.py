"""Read-back for eval/tts_benchmark.py: Whisper large-v3 transcribes every successfully synthesized clip of the clarity
categories; the repo computes the word and character errors from these transcripts (so the metric code is tested there).
Input: the dataset mishradevang14/businessflow-tts-benchmark-audio (manifest.json + one folder of wav files per engine).
Output: /kaggle/working/transcripts.json, {relative wav path: transcript}. Greedy decoding: same audio, same transcript."""
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import pipeline

CLARITY = {"hi_plain", "hi_numbers", "hi_acronyms", "hi_long"}
# Kaggle mounts datasets at varying depths (/kaggle/input/<slug> or /kaggle/input/datasets/<user>/<slug>): search for the manifest
found = sorted(Path("/kaggle/input").rglob("manifest.json"))
if not found:
    raise SystemExit("manifest.json not found anywhere under /kaggle/input: " + ", ".join(str(p) for p in sorted(Path("/kaggle/input").rglob("*"))[:20]))
root = found[0].parent
print("dataset root:", root, flush=True)
clips = [c for c in json.loads((root / "manifest.json").read_text(encoding="utf-8"))["clips"] if c.get("ok") and c["category"] in CLARITY]
print(len(clips), "clips to transcribe", flush=True)

asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0 if torch.cuda.is_available() else -1,
               torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
out, t0 = {}, time.time()
for i, c in enumerate(clips):
    audio, sr = sf.read(str(root / c["path"].replace("\\", "/")), dtype="float32")  # manifests written on Windows hold backslashes
    out[c["path"]] = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs={"language": "hindi", "task": "transcribe"})["text"].strip()
    if i % 25 == 0:
        print(f"{i}/{len(clips)} ({time.time() - t0:.0f}s)", flush=True)
Path("/kaggle/working/transcripts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote transcripts for", len(out), "clips", flush=True)
