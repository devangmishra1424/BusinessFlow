"""Runs the full MUCS 2021 test split (3,136 utterances) through
faster-whisper "base" on GPU and reports WER + script-consistency rate --
the full-corpus, fast version of eval/wer_benchmark.py, which runs a
smaller sample locally on CPU int8.

Logic (WER normalization, script classification, Kaldi-format loading) is
duplicated here rather than imported, since this runs standalone on
Kaggle without access to the local project package.

IMPORTANT CAVEAT, printed in the output on purpose: this measures GPU
float32 performance (float16 was tried first and rejected -- ctranslate2
raised "target device or backend do not support efficient float16
computation" on whatever GPU Kaggle assigned this run), not the CPU int8
configuration the model actually deploys on. It's the fast, full-coverage
number, not the deployment-representative one -- that comes from the
smaller local CPU sample.
"""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper", "jiwer"], check=True)

import jiwer
import soundfile as sf
from faster_whisper import WhisperModel

DATASET_DIR = Path("/kaggle/input/datasets/mishradevang14/mucs-2021-hindi-english-test")
OUTPUT_PATH = Path("/kaggle/working/mucs_test_full_base.json")

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")

_NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def dominant_script(text: str) -> str:
    d = len(_DEVANAGARI_RE.findall(text))
    l = len(_LATIN_RE.findall(text))
    if d == 0 and l == 0:
        return "none"
    if d > l * 2:
        return "devanagari"
    if l > d * 2:
        return "latin"
    return "mixed"


def load_test_utterances():
    transcripts_dir = DATASET_DIR / "transcripts"

    segments = {}
    with open(transcripts_dir / "segments", encoding="utf-8") as f:
        for line in f:
            utt_id, recording_id, start, end = line.split()
            segments[utt_id] = (recording_id, float(start), float(end))

    texts = {}
    with open(transcripts_dir / "text", encoding="utf-8") as f:
        for line in f:
            utt_id, _, text = line.strip().partition(" ")
            texts[utt_id] = text

    by_recording = {}
    for utt_id, (recording_id, start, end) in segments.items():
        reference = texts.get(utt_id, "")
        if not reference.strip():
            continue
        by_recording.setdefault(recording_id, []).append((utt_id, start, end, reference))
    return by_recording


def main():
    print("loading model (base, GPU, float32)...", flush=True)
    model = WhisperModel("base", device="cuda", compute_type="float32")

    by_recording = load_test_utterances()
    total = sum(len(v) for v in by_recording.values())
    print(f"{total} utterances across {len(by_recording)} recordings", flush=True)

    references = []
    hypotheses = []
    done = 0

    for recording_id, utts in by_recording.items():
        wav_path = DATASET_DIR / f"{recording_id}.wav"
        if not wav_path.exists():
            print(f"missing wav for recording {recording_id}, skipping its {len(utts)} utterances", flush=True)
            continue
        info = sf.info(str(wav_path))
        for utt_id, start, end, reference in utts:
            start_frame = int(start * info.samplerate)
            num_frames = int((end - start) * info.samplerate)
            audio, _sr = sf.read(str(wav_path), start=start_frame, frames=num_frames, dtype="float32")

            segs, _info = model.transcribe(audio, language="hi")
            hypothesis = " ".join(s.text.strip() for s in segs)

            references.append(reference)
            hypotheses.append(hypothesis)
            done += 1
            if done % 200 == 0:
                print(f"{done}/{total} utterances done...", flush=True)

    wer = jiwer.wer(references, hypotheses, reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)
    hyp_scripts = Counter(dominant_script(h) for h in hypotheses)
    n = len(references)

    result = {
        "model_size": "base",
        "device": "cuda",
        "compute_type": "float32",
        "note": "GPU float32, not the deployed CPU int8 config -- the fast full-corpus number, not the deployment-representative one",
        "utterances_scored": n,
        "wer": wer,
        "script_consistency_rate": hyp_scripts["devanagari"] / n if n else 0.0,
        "hypothesis_script_distribution": dict(hyp_scripts),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
