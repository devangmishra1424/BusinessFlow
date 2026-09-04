"""Continues the r=32 fine-tune (see kaggle/training-kernel-r32-round2/
train_r32_round2.py for the base recipe -- LoRA config, dependency
pinning, benchmark harness, all unchanged and not re-derived here) with
ONE new addition: real, domain-matched, genuinely code-switched
financial/collections dialogue, TTS-synthesized from datasets/
colloquial-hinglish-conversations (see kaggle/prepare_domain_codeswitch_
tts.py's own docstring for the full sourcing rationale and honest
caveats -- unknown source-data provenance, synthetic-audio risk, etc).

Deliberately built from r32 (round 1), NOT round 2 + splicing: that
combined run was tried and came back flat-to-worse (MUCS WER barely
moved, script consistency regressed) -- see this project's own real
results for that experiment. Stacking an unproven new technique on top
of an already-unproven one would make it impossible to tell which
change did what if this ALSO doesn't help. This run tests exactly one
new variable: does real, domain-matched code-switched audio (even if
synthetic) help, isolated from the splicing technique that didn't.

The real corpus reference below (mishradevang14/training-set) already
reflects round 2's own real-data expansion (56,143 utterances: 52,825
MUCS + full 3,318 HiACC-adult, not round 1's original 55,147) -- that
part of round 2 is a straightforward addition of more REAL data, not a
generative augmentation trick, so there's no real reason to revert it
just to isolate splicing specifically. What's NOT included here is the
splicing technique itself.

Run: python train_r32_domain_tts.py (via Kaggle kernel push)
"""

import json
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "transformers", "peft", "ctranslate2", "faster-whisper", "jiwer", "soundfile",
     "torch<2.8"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao", "torchvision", "torchaudio"],
    check=True,
)

import jiwer
import peft
import soundfile as sf
import torch
import transformers

print(f"transformers={transformers.__version__} peft={peft.__version__} torch={torch.__version__}")

BASE_MODEL_DIR = Path("/kaggle/input/datasets/mishradevang14/businessflow-whisper-finetuned-r32-hf")
DATASET_DIR = Path("/kaggle/input/datasets/mishradevang14/training-set")
DOMAIN_DIR = Path("/kaggle/input/datasets/mishradevang14/domain-codeswitch-tts")
TEST_DIR = Path("/kaggle/input/datasets/mishradevang14/mucs-2021-hindi-english-test")
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


def _resolve_assets_dir():
    audio_dir = DATASET_DIR / "audio"
    if audio_dir.is_dir() and any(audio_dir.iterdir()):
        return DATASET_DIR
    extracted = WORK_DIR / "training_assets"
    extracted.mkdir(parents=True, exist_ok=True)
    for name in ("audio", "english_regression"):
        zip_path = DATASET_DIR / f"{name}.zip"
        target = extracted / name
        if zip_path.exists() and not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
    manifest_src = DATASET_DIR / "manifest.json"
    if manifest_src.exists():
        shutil.copyfile(manifest_src, extracted / "manifest.json")
    return extracted


def _resolve_flat_zip_dir(source_dir: Path, work_subdir: str):
    """Same shape as _resolve_assets_dir but for a dataset shipped as one
    flat-root audio.zip + a top-level manifest.json (no english_regression
    subfolder) -- domain-codeswitch-tts and splicing-fuel both look like
    this; see kaggle/upload_domain_codeswitch_tts.py."""
    audio_dir = source_dir / "audio"
    if audio_dir.is_dir() and any(audio_dir.iterdir()):
        return source_dir
    extracted = WORK_DIR / work_subdir
    target = extracted / "audio"
    zip_path = source_dir / "audio.zip"
    if zip_path.exists() and not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    manifest_src = source_dir / "manifest.json"
    if manifest_src.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_src, extracted / "manifest.json")
    return extracted


ASSETS_DIR = _resolve_assets_dir()
DOMAIN_ASSETS_DIR = _resolve_flat_zip_dir(DOMAIN_DIR, "domain_codeswitch_tts_assets")
print(f"training assets resolved to: {ASSETS_DIR}")
print(f"domain code-switch TTS assets resolved to: {DOMAIN_ASSETS_DIR}")


def load_mucs_test_sample(n=300, seed=42):
    segments, texts = {}, {}
    with open(TEST_DIR / "transcripts" / "segments", encoding="utf-8") as f:
        for line in f:
            utt_id, rec_id, start, end = line.split()
            segments[utt_id] = (rec_id, float(start), float(end))
    with open(TEST_DIR / "transcripts" / "text", encoding="utf-8") as f:
        for line in f:
            utt_id, _, text = line.strip().partition(" ")
            texts[utt_id] = text
    pool = [(uid, r, s, e, texts[uid]) for uid, (r, s, e) in segments.items() if texts.get(uid, "").strip() and (TEST_DIR / f"{r}.wav").exists()]
    return random.Random(seed).sample(pool, min(n, len(pool)))


def load_english_regression():
    return json.loads((ASSETS_DIR / "english_regression" / "manifest.json").read_text(encoding="utf-8"))


test_sample = load_mucs_test_sample(300)
english_manifest = load_english_regression()
print(f"loaded {len(test_sample)} MUCS test utterances, {len(english_manifest)} English regression utterances")


def convert_to_ctranslate2(hf_model_dir, output_dir):
    subprocess.run([
        "ct2-transformers-converter", "--model", str(hf_model_dir),
        "--output_dir", str(output_dir), "--quantization", "float32", "--force",
    ], check=True)


def run_benchmark(ct2_model_dir, label):
    from faster_whisper import WhisperModel
    model = WhisperModel(str(ct2_model_dir), device="cuda", compute_type="float32")

    refs, hyps = [], []
    for uid, rec_id, start, end, text in test_sample:
        info = sf.info(str(TEST_DIR / f"{rec_id}.wav"))
        audio, sr = sf.read(str(TEST_DIR / f"{rec_id}.wav"), start=int(start * info.samplerate), frames=int((end - start) * info.samplerate), dtype="float32")
        segs, _ = model.transcribe(audio, language="hi", **_DECODE_KWARGS)
        refs.append(text)
        hyps.append(" ".join(s.text.strip() for s in segs))
    mucs_wer = jiwer.wer(refs, hyps, reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)
    script_dist = Counter(dominant_script(h) for h in hyps)
    script_rate = script_dist["devanagari"] / len(hyps)

    en_refs, en_hyps = [], []
    for entry in english_manifest:
        audio, sr = sf.read(str(ASSETS_DIR / "english_regression" / entry["wav"]), dtype="float32")
        segs, _ = model.transcribe(audio, language="en", **_DECODE_KWARGS)
        en_refs.append(entry["reference"])
        en_hyps.append(" ".join(s.text.strip() for s in segs))
    en_wer = jiwer.wer(en_refs, en_hyps, reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)

    result = {"label": label, "mucs_wer": mucs_wer, "script_consistency": script_rate, "english_wer": en_wer, "script_distribution": dict(script_dist)}
    print(json.dumps(result, indent=2))
    del model
    return result


# "before" here is our existing r=32 fine-tune, unmodified -- the real
# baseline this run needs to beat to be worth keeping.
convert_to_ctranslate2(str(BASE_MODEL_DIR), WORK_DIR / "ct2_before")
before_result = run_benchmark(WORK_DIR / "ct2_before", "before_r32_round1")

from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained(str(BASE_MODEL_DIR))
model = WhisperForConditionalGeneration.from_pretrained(str(BASE_MODEL_DIR))
model.config.use_cache = False
_MAX_TARGET_POSITIONS = model.config.max_target_positions

# Same r=32/alpha=64 as every prior round -- a fresh adapter on top of
# the already-fine-tuned weights (merge_and_unload collapsed the
# original adapter into the base, so there's no adapter left to resume).
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

manifest = json.loads((ASSETS_DIR / "manifest.json").read_text(encoding="utf-8"))
domain_manifest = json.loads((DOMAIN_ASSETS_DIR / "manifest.json").read_text(encoding="utf-8"))
print(f"training on {len(manifest)} real utterances + {len(domain_manifest)} domain-matched TTS utterances "
      f"({len(domain_manifest) / (len(manifest) + len(domain_manifest)):.1%} of the epoch)")


class SpeechDataset(torch.utils.data.Dataset):
    """Two real, static sources concatenated into one dataset -- unlike
    round2-splice's on-the-fly synthetic pairing, every example here is
    a complete, real (or TTS-real) utterance with its own real label, so
    there's nothing to randomize per access."""

    def __init__(self, sources, tokenizer):
        self.entries = [(audio_dir / e["wav"], e["reference"]) for manifest, audio_dir in sources for e in manifest]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        wav_path, reference = self.entries[idx]
        audio, _sr = sf.read(str(wav_path), dtype="float32")
        labels = self.tokenizer(reference, truncation=True, max_length=_MAX_TARGET_POSITIONS).input_ids
        return {"audio": audio, "labels": labels}


train_dataset = SpeechDataset(
    [(manifest, ASSETS_DIR / "audio"), (domain_manifest, DOMAIN_ASSETS_DIR / "audio")],
    processor.tokenizer,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        audio = [f["audio"] for f in features]
        batch = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

checkpoint_dir = WORK_DIR / "training_checkpoints"
training_args = Seq2SeqTrainingArguments(
    output_dir=str(checkpoint_dir),
    per_device_train_batch_size=8,
    dataloader_num_workers=2,
    learning_rate=1e-3,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    report_to=[],
    remove_unused_columns=False,
)

from transformers.trainer_utils import get_last_checkpoint

trainer = Seq2SeqTrainer(args=training_args, model=model, train_dataset=train_dataset, data_collator=data_collator)

resume_from = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
if resume_from:
    print(f"resuming from checkpoint: {resume_from}")
trainer.train(resume_from_checkpoint=resume_from)

merged = model.merge_and_unload()
merged.save_pretrained(WORK_DIR / "merged_finetuned")
processor.save_pretrained(WORK_DIR / "merged_finetuned")
shutil.make_archive(str(WORK_DIR / "businessflow-whisper-r32-domain-tts-hf"), "zip", str(WORK_DIR / "merged_finetuned"))

convert_to_ctranslate2(WORK_DIR / "merged_finetuned", WORK_DIR / "ct2_after")
after_result = run_benchmark(WORK_DIR / "ct2_after", "after_r32_domain_tts")

print(f"{'metric':<25}{'before (round 1)':<18}{'after (domain TTS)':<20}")
print(f"{'MUCS WER':<25}{before_result['mucs_wer']*100:>6.1f}%{'':<11}{after_result['mucs_wer']*100:>6.1f}%")
print(f"{'Script consistency':<25}{before_result['script_consistency']*100:>6.1f}%{'':<11}{after_result['script_consistency']*100:>6.1f}%")
print(f"{'English WER':<25}{before_result['english_wer']*100:>6.1f}%{'':<11}{after_result['english_wer']*100:>6.1f}%")

shutil.make_archive(str(WORK_DIR / "businessflow-whisper-r32-domain-tts"), "zip", str(WORK_DIR / "ct2_after"))
print("done -- download businessflow-whisper-r32-domain-tts.zip (CTranslate2) and "
      "businessflow-whisper-r32-domain-tts-hf.zip (merged HF) from the Output tab")
