"""Full (not LoRA) fine-tune of openai/whisper-medium on the same
Hindi-English code-switching corpus as the base-model LoRA runs, now
expanded with the 996 previously-unused HiACC/Corpus-adult utterances
(56,143 total, up from 55,147). "As high grade as possible" was the
brief for this run: full fine-tuning removes LoRA's rank ceiling
entirely, at two real costs the base-model LoRA runs didn't have to
manage -- a much bigger optimizer memory footprint (gradients +
Adam state for all 769M params, not ~1-4M via a LoRA adapter), and a
learning rate that has to be found for real, not reused from the LoRA
recipe (LoRA's effective update magnitude via alpha/r scaling has no
correspondence to a raw full-fine-tune LR -- reusing LoRA's 1e-3
here would very likely diverge and wreck the model, not just be
suboptimal).

Same dependency fixes as the base-model kernel (torch<2.8 for P100/
sm_60, torchao/torchvision/torchaudio uninstalled, label truncation
against the model's own max_target_positions) -- see that kernel's own
comments for why each of these exists; not re-derived here.

Structure:
  1. LR-finder pilot -- a few hundred steps each, on a small subset, at
     a handful of candidate LRs, comparing loss trajectories. Cheap
     relative to a full 3-epoch run, and avoids committing hours of GPU
     time to a rate that turns out to destabilize training.
  2. The real, full 3-epoch run on the complete corpus, at whichever LR
     the pilot actually favored -- with gradient_checkpointing and a
     smaller per-device batch (compensated by gradient_accumulation_steps
     to keep the same effective batch size as the LoRA runs) to fit the
     much larger full-fine-tune memory footprint on the P100's 16GB.
"""

import json
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
import soundfile as sf
import torch
import transformers

print(f"transformers={transformers.__version__} torch={torch.__version__}")

BASE_MODEL = "openai/whisper-medium"
DATASET_DIR = Path("/kaggle/input/datasets/mishradevang14/training-set")
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


ASSETS_DIR = _resolve_assets_dir()
print(f"training assets resolved to: {ASSETS_DIR}")

import random


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


convert_to_ctranslate2(BASE_MODEL, WORK_DIR / "ct2_before")
before_result = run_benchmark(WORK_DIR / "ct2_before", "before")

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained(BASE_MODEL)
manifest = json.loads((ASSETS_DIR / "manifest.json").read_text(encoding="utf-8"))
print(f"training on {len(manifest)} utterances")


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, audio_dir, tokenizer, max_target_positions):
        self.manifest = manifest
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.max_target_positions = max_target_positions

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        audio, _sr = sf.read(str(self.audio_dir / entry["wav"]), dtype="float32")
        labels = self.tokenizer(entry["reference"], truncation=True, max_length=self.max_target_positions).input_ids
        return {"audio": audio, "labels": labels}


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

# ---- LR-finder pilot ----
# A short, cheap probe before committing hours of GPU time to the full
# run. Each candidate gets a FRESH model load -- reusing one model
# object across candidates would let an earlier candidate's real weight
# updates bleed into the next one's test, invalidating the comparison.
_PILOT_CANDIDATES = [1e-5, 3e-5, 1e-4]
_PILOT_STEPS = 200
_PILOT_SAMPLE_SIZE = 2000

pilot_manifest = random.Random(42).sample(manifest, min(_PILOT_SAMPLE_SIZE, len(manifest)))

pilot_results = {}
for lr in _PILOT_CANDIDATES:
    print(f"\n=== LR pilot: {lr} ===")
    pilot_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    pilot_model.config.use_cache = False
    max_target_positions = pilot_model.config.max_target_positions
    pilot_dataset = SpeechDataset(pilot_manifest, ASSETS_DIR / "audio", processor.tokenizer, max_target_positions)

    pilot_args = Seq2SeqTrainingArguments(
        output_dir=str(WORK_DIR / f"pilot_{lr}"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=lr,
        # warmup_ratio dropped -- confirmed live (same bug hit once
        # already, at the very start of this project) that Kaggle's -U
        # transformers resolves to a version (5.15.1) whose
        # Seq2SeqTrainingArguments doesn't accept it.
        weight_decay=0.01,
        max_steps=_PILOT_STEPS,
        fp16=True,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    pilot_trainer = Seq2SeqTrainer(args=pilot_args, model=pilot_model, train_dataset=pilot_dataset, data_collator=data_collator)
    pilot_trainer.train()

    # Mean loss over the second half of the pilot run -- the first half
    # is dominated by the same sharp initial drop every LR shows
    # regardless of whether it's actually a good rate; the second half
    # is where a too-high LR would already be visibly unstable/diverging
    # and a too-low one would still be barely moving.
    losses = [h["loss"] for h in pilot_trainer.state.log_history if "loss" in h]
    second_half = losses[len(losses) // 2:] or losses
    mean_loss = sum(second_half) / len(second_half)
    pilot_results[lr] = mean_loss
    print(f"LR {lr}: mean loss (second half of pilot) = {mean_loss:.4f}")

    del pilot_model, pilot_trainer
    torch.cuda.empty_cache()

best_lr = min(pilot_results, key=pilot_results.get)
print(f"\npilot results: {pilot_results}")
print(f"selected learning_rate = {best_lr}")

# ---- Full run ----
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
model.config.use_cache = False
_MAX_TARGET_POSITIONS = model.config.max_target_positions

train_dataset = SpeechDataset(manifest, ASSETS_DIR / "audio", processor.tokenizer, _MAX_TARGET_POSITIONS)

checkpoint_dir = WORK_DIR / "training_checkpoints"
training_args = Seq2SeqTrainingArguments(
    output_dir=str(checkpoint_dir),
    # per_device_train_batch_size=2 (not the LoRA runs' 8) with
    # gradient_accumulation_steps=4 keeps the same effective batch size
    # (8) while cutting the activation memory a single forward/backward
    # pass has to hold -- full fine-tuning already spends far more VRAM
    # than LoRA on gradients + Adam state for all 769M params, so this
    # is the real, necessary tradeoff to still fit the P100's 16GB.
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    learning_rate=best_lr,
    # warmup_ratio dropped -- same version-incompatibility as the pilot's
    # own training_args above.
    weight_decay=0.01,
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

# No LoRA adapter here -- model IS the full fine-tuned model already,
# unlike the LoRA kernel's merge_and_unload() step.
model.save_pretrained(WORK_DIR / "merged_finetuned")
processor.save_pretrained(WORK_DIR / "merged_finetuned")
shutil.make_archive(str(WORK_DIR / "businessflow-whisper-medium-finetuned-hf"), "zip", str(WORK_DIR / "merged_finetuned"))

convert_to_ctranslate2(WORK_DIR / "merged_finetuned", WORK_DIR / "ct2_after")
after_result = run_benchmark(WORK_DIR / "ct2_after", "after")

print(f"{'metric':<25}{'before':<15}{'after':<15}")
print(f"{'MUCS WER':<25}{before_result['mucs_wer']*100:>6.1f}%{'':<8}{after_result['mucs_wer']*100:>6.1f}%")
print(f"{'Script consistency':<25}{before_result['script_consistency']*100:>6.1f}%{'':<8}{after_result['script_consistency']*100:>6.1f}%")
print(f"{'English WER':<25}{before_result['english_wer']*100:>6.1f}%{'':<8}{after_result['english_wer']*100:>6.1f}%")

shutil.make_archive(str(WORK_DIR / "businessflow-whisper-medium-finetuned"), "zip", str(WORK_DIR / "ct2_after"))
print("done -- download businessflow-whisper-medium-finetuned.zip (CTranslate2) and businessflow-whisper-medium-finetuned-hf.zip (merged HF) from the Output tab")
