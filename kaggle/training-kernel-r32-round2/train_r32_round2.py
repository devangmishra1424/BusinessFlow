"""Round 2 of LoRA fine-tuning: starts from our existing r=32 fine-tune
(merged weights, not fresh openai/whisper-base) and continues training
on the expanded corpus -- 56,143 utterances, up from 55,147, now
including all 3,318 of Corpus/adult (HiACC) instead of just the 2,322
already used; the 996 new ones were confirmed genuinely new via real
MD5 hash matching against the already-included files, not filename
guessing. Corpus/children and openSLR were deliberately left out (a
different acoustic domain, and off-topic programming-tutorial content,
respectively) -- see the manifest.json this run's dataset actually
ships to confirm what's really in scope.

The merged r=32 model has no LoRA adapter attached any more (merge_and_
unload() collapsed it into the base weights) -- there's nothing to
"resume". This applies a FRESH r=32/alpha=64 adapter on top of those
already-fine-tuned weights instead, a standard, legitimate continual-
fine-tuning pattern, not a resume in the checkpoint sense.

Same dependency fixes and label-truncation handling as the other two
training kernels -- see their own comments for the history; not
re-derived here.
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
import peft
import soundfile as sf
import torch
import transformers

print(f"transformers={transformers.__version__} peft={peft.__version__} torch={torch.__version__}")

BASE_MODEL_DIR = Path("/kaggle/input/datasets/mishradevang14/businessflow-whisper-finetuned-r32-hf")
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


# "before" here is our existing r=32 fine-tune, unmodified -- the real
# baseline this round-2 run needs to beat to be worth keeping.
convert_to_ctranslate2(str(BASE_MODEL_DIR), WORK_DIR / "ct2_before")
before_result = run_benchmark(WORK_DIR / "ct2_before", "before_r32_round1")

from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained(str(BASE_MODEL_DIR))
model = WhisperForConditionalGeneration.from_pretrained(str(BASE_MODEL_DIR))
model.config.use_cache = False
_MAX_TARGET_POSITIONS = model.config.max_target_positions

# Same r=32/alpha=64 as round 1 -- a fresh adapter on top of the
# already-fine-tuned weights (merge_and_unload collapsed the original
# adapter into the base, so there's no round-1 adapter left to resume).
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

manifest = json.loads((ASSETS_DIR / "manifest.json").read_text(encoding="utf-8"))
print(f"training on {len(manifest)} utterances (round 2, expanded corpus)")


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, audio_dir, tokenizer):
        self.manifest = manifest
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        audio, _sr = sf.read(str(self.audio_dir / entry["wav"]), dtype="float32")
        labels = self.tokenizer(entry["reference"], truncation=True, max_length=_MAX_TARGET_POSITIONS).input_ids
        return {"audio": audio, "labels": labels}


train_dataset = SpeechDataset(manifest, ASSETS_DIR / "audio", processor.tokenizer)


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
shutil.make_archive(str(WORK_DIR / "businessflow-whisper-r32-round2-hf"), "zip", str(WORK_DIR / "merged_finetuned"))

convert_to_ctranslate2(WORK_DIR / "merged_finetuned", WORK_DIR / "ct2_after")
after_result = run_benchmark(WORK_DIR / "ct2_after", "after_r32_round2")

print(f"{'metric':<25}{'before (round 1)':<18}{'after (round 2)':<18}")
print(f"{'MUCS WER':<25}{before_result['mucs_wer']*100:>6.1f}%{'':<11}{after_result['mucs_wer']*100:>6.1f}%")
print(f"{'Script consistency':<25}{before_result['script_consistency']*100:>6.1f}%{'':<11}{after_result['script_consistency']*100:>6.1f}%")
print(f"{'English WER':<25}{before_result['english_wer']*100:>6.1f}%{'':<11}{after_result['english_wer']*100:>6.1f}%")

shutil.make_archive(str(WORK_DIR / "businessflow-whisper-r32-round2"), "zip", str(WORK_DIR / "ct2_after"))
print("done -- download businessflow-whisper-r32-round2.zip (CTranslate2) and businessflow-whisper-r32-round2-hf.zip (merged HF) from the Output tab")
