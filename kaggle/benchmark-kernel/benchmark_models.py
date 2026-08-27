"""Compares three ASR models against a real, 1000-utterance MUCS
Hindi-English sample plus the English regression set: pretrained
whisper-base (the untuned reference point), pretrained whisper-medium,
and our own r=32 LoRA fine-tune of whisper-base. large-v3 was dropped --
see the results= block below for why.

The original plan was compute_type="int8" on GPU -- matching real
deployment's exact precision (CPU/int8), unlike the training kernel's
own run_benchmark(), which runs float32 and (per a real, already-
confirmed finding elsewhere in this project) can disagree with int8-CPU
results, not just be slower/faster. That plan doesn't work here: Kaggle's
free GPU is a P100 (Pascal), confirmed live via a real
"Requested int8 compute type, but the target device or backend do not
support efficient int8 computation" error -- this hardware can't do
efficient int8 at all. Falls back to float32, same as the training
kernel's own benchmark. That makes these numbers a DIRECTIONAL signal
(is a bigger model even meaningfully better) rather than a precise
int8-CPU deployment number -- a real int8/CPU confirmation is still
needed on whichever model this points toward before actually committing
to it.

Same benchmark methodology (WER normalization, dominant_script,
_DECODE_KWARGS) as train_whisper_full.py's own run_benchmark(), just at
n=1000 instead of 300, and never doing any actual fine-tuning here --
this kernel only ever runs inference.

Run: pushed via the Kaggle API, same as the training kernel.
"""

import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "transformers", "ctranslate2", "faster-whisper", "jiwer", "soundfile",
     "torch<2.8"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao", "torchvision", "torchaudio"],
    check=True,
)

import jiwer
import soundfile as sf

DATASET_DIR = Path("/kaggle/input/datasets/mishradevang14/training-set")
TEST_DIR = Path("/kaggle/input/datasets/mishradevang14/mucs-2021-hindi-english-test")
FINETUNED_DIR = Path("/kaggle/input/datasets/mishradevang14/businessflow-whisper-finetuned-r32")
WORK_DIR = Path("/kaggle/working")

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_NORMALIZE = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
_DECODE_KWARGS = dict(repetition_penalty=1.3, no_repeat_ngram_size=3, condition_on_previous_text=False)


def dominant_script(text):
    d, l = len(_DEVANAGARI_RE.findall(text)), len(_LATIN_RE.findall(text))
    if d == 0 and l == 0:
        return "none"
    if d > l * 2:
        return "devanagari"
    if l > d * 2:
        return "latin"
    return "mixed"


def _resolve_english_regression_dir():
    """Same flat-root-zip handling as train_whisper_full.py's
    _resolve_assets_dir -- english_regression may arrive as a pre-made
    zip rather than a pre-extracted directory."""
    direct = DATASET_DIR / "english_regression"
    if direct.is_dir() and any(direct.iterdir()):
        return direct
    extracted = WORK_DIR / "english_regression"
    zip_path = DATASET_DIR / "english_regression.zip"
    if zip_path.exists() and not extracted.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted)
    return extracted


ENGLISH_DIR = _resolve_english_regression_dir()


def load_mucs_test_sample(n=1000, seed=42):
    segments, texts = {}, {}
    with open(TEST_DIR / "transcripts" / "segments", encoding="utf-8") as f:
        for line in f:
            utt_id, rec_id, start, end = line.split()
            segments[utt_id] = (rec_id, float(start), float(end))
    with open(TEST_DIR / "transcripts" / "text", encoding="utf-8") as f:
        for line in f:
            utt_id, _, text = line.strip().partition(" ")
            texts[utt_id] = text
    pool = [
        (uid, r, s, e, texts[uid])
        for uid, (r, s, e) in segments.items()
        if texts.get(uid, "").strip() and (TEST_DIR / f"{r}.wav").exists()
    ]
    return random.Random(seed).sample(pool, min(n, len(pool)))


def load_english_regression():
    return json.loads((ENGLISH_DIR / "manifest.json").read_text(encoding="utf-8"))


test_sample = load_mucs_test_sample(1000)
english_manifest = load_english_regression()
print(f"loaded {len(test_sample)} MUCS test utterances, {len(english_manifest)} English regression utterances")


def convert_to_ctranslate2(hf_model_dir, output_dir):
    subprocess.run([
        "ct2-transformers-converter", "--model", str(hf_model_dir),
        "--output_dir", str(output_dir), "--quantization", "float32", "--force",
    ], check=True)


def run_benchmark(ct2_model_dir, label):
    from faster_whisper import WhisperModel
    # int8 was the plan (see module docstring), but Kaggle's free GPU is
    # a P100 (Pascal) -- confirmed live it can't do efficient int8 at all
    # ("ValueError: Requested int8 compute type, but the target device or
    # backend do not support efficient int8 computation"), so this falls
    # back to float32, same as the training kernel's own run_benchmark().
    # These numbers are therefore a DIRECTIONAL signal (is a bigger model
    # even meaningfully better), not a precise int8-CPU deployment number
    # -- a real int8/CPU confirmation is still needed before actually
    # committing to whichever model this points toward.
    model = WhisperModel(str(ct2_model_dir), device="cuda", compute_type="float32")

    refs, hyps = [], []
    for i, (uid, rec_id, start, end, text) in enumerate(test_sample):
        info = sf.info(str(TEST_DIR / f"{rec_id}.wav"))
        audio, sr = sf.read(
            str(TEST_DIR / f"{rec_id}.wav"),
            start=int(start * info.samplerate), frames=int((end - start) * info.samplerate),
            dtype="float32",
        )
        segs, _ = model.transcribe(audio, language="hi", **_DECODE_KWARGS)
        refs.append(text)
        hyps.append(" ".join(s.text.strip() for s in segs))
        if (i + 1) % 100 == 0:
            print(f"  [{label}] {i + 1}/{len(test_sample)} MUCS utterances done...")

    mucs_wer = jiwer.wer(refs, hyps, reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)
    script_dist = Counter(dominant_script(h) for h in hyps)
    script_rate = script_dist["devanagari"] / len(hyps)

    en_refs, en_hyps = [], []
    for entry in english_manifest:
        audio, sr = sf.read(str(ENGLISH_DIR / entry["wav"]), dtype="float32")
        segs, _ = model.transcribe(audio, language="en", **_DECODE_KWARGS)
        en_refs.append(entry["reference"])
        en_hyps.append(" ".join(s.text.strip() for s in segs))
    en_wer = jiwer.wer(en_refs, en_hyps, reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)

    result = {
        "label": label, "mucs_wer": mucs_wer, "script_consistency": script_rate,
        "english_wer": en_wer, "script_distribution": dict(script_dist),
        "n_mucs": len(refs), "n_english": len(en_refs),
    }
    print(json.dumps(result, indent=2))
    del model
    return result


results = {}

# large-v3 dropped: already confirmed impractical for this project's real
# deployment (CPU/int8) -- measured live at ~1,812s (30 min) for a single
# utterance on CPU, versus whisper-base's ~5s. Its quality ceiling isn't
# actionable without a GPU-hosting decision nobody's made, so it's not
# worth spending Kaggle GPU time benchmarking. base (untuned) is added
# instead, as the real reference point for what our own fine-tuning
# actually buys over the same-sized, same-latency stock model.
print("\n=== whisper-base (pretrained, untuned) ===")
convert_to_ctranslate2("openai/whisper-base", WORK_DIR / "ct2_base")
results["base"] = run_benchmark(WORK_DIR / "ct2_base", "base_pretrained")

print("\n=== whisper-medium (pretrained) ===")
convert_to_ctranslate2("openai/whisper-medium", WORK_DIR / "ct2_medium")
results["medium"] = run_benchmark(WORK_DIR / "ct2_medium", "medium_pretrained")

print("\n=== whisper-base r=32 LoRA fine-tune (ours) ===")
results["finetuned_r32"] = run_benchmark(FINETUNED_DIR, "finetuned_r32")

(WORK_DIR / "comparison_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n=== SUMMARY ===")
print(f"{'model':<20}{'MUCS WER':<12}{'script cons.':<14}{'English WER':<12}")
for key, r in results.items():
    print(f"{key:<20}{r['mucs_wer']*100:>6.1f}%     {r['script_consistency']*100:>6.1f}%       {r['english_wer']*100:>6.1f}%")
print("\ndone -- download comparison_results.json from the Output tab")
