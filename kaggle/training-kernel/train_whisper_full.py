"""Full-corpus LoRA fine-tune of openai/whisper-base on the Hindi-English
code-switching data (55,147 utterances: 52,825 MUCS + 2,322 HiACC -- the
first run that produced businessflow-whisper-finetuned.zip used a curated
3,500-utterance sample; this is the same recipe run against the whole
corpus now that the sample-size constraint has been lifted).

Differences from the original recipe (notebookd3491b2506), and why:

- Audio is loaded lazily per-example in a plain torch Dataset instead of
  pre-computing the 30s-padded mel spectrogram for every example up front
  via datasets.Dataset.map(). At this scale that upfront step would
  materialize ~52GB of float32 features (55,147 * 80 * 3000 * 4 bytes)
  before training even starts. Loading raw audio and running the feature
  extractor per-batch in the collator keeps the on-disk footprint to the
  raw audio itself (~15GB) and starts training immediately.
- save_strategy="steps" (was "no") -- a 20k+ step run has real value in
  checkpointing that the original ~1.3k-step run didn't. resume_from_checkpoint
  is auto-detected on startup so a re-pushed kernel can continue a partial run.
- run_benchmark() now passes the same decoding parameters that are actually
  deployed (repetition_penalty=1.3, no_repeat_ngram_size=3,
  condition_on_previous_text=False -- see src/businessflow/audio/asr.py).
  The original benchmark used faster-whisper's defaults, which is also what
  caused the repetition-loop hallucination that made the first fine-tune look
  worse than base on WER; benchmarking under the same params as production
  actually uses gives an honest before/after number.
- Everything else (LoRA config, learning rate, base model, batch size) is
  unchanged from the proven recipe -- no evidence those need to change, so
  they don't.
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
     "transformers", "peft", "ctranslate2", "faster-whisper", "jiwer", "soundfile", "torchao"],
    check=True,
)

import jiwer
import peft
import soundfile as sf
import torch
import transformers
from transformers.trainer_utils import get_last_checkpoint

print(f"transformers={transformers.__version__} peft={peft.__version__} torch={torch.__version__}")

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
    """audio/ and english_regression/ were uploaded as pre-made flat-root zips
    (scripts/kaggle_prezip.py) rather than pre-extracted directories -- handle
    that, and the case where Kaggle has already unpacked them, rather than
    assuming one. Flat-root means each zip's entries have no directory prefix
    (e.g. mucs_00000.wav, not audio/mucs_00000.wav), so extraction has to
    target the named subdirectory explicitly, not the parent."""
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


convert_to_ctranslate2("openai/whisper-base", WORK_DIR / "ct2_before")
before_result = run_benchmark(WORK_DIR / "ct2_before", "before")

from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.config.use_cache = False

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

manifest = json.loads((ASSETS_DIR / "manifest.json").read_text(encoding="utf-8"))
print(f"training on {len(manifest)} utterances")


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
        labels = self.tokenizer(entry["reference"]).input_ids
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
    # Trainer's default column-pruning drops any dataset dict key that
    # isn't a recognized WhisperForConditionalGeneration.forward() arg --
    # "audio" isn't one ("input_features" is), so it gets stripped before
    # ever reaching this collator. remove_unused_columns=False is HF's own
    # documented fix for exactly this: a custom collator that needs raw
    # fields the model signature doesn't recognize (confirmed live: this
    # crashed with KeyError: 'audio' inside the collator without this).
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(args=training_args, model=model, train_dataset=train_dataset, data_collator=data_collator)

resume_from = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
if resume_from:
    print(f"resuming from checkpoint: {resume_from}")
trainer.train(resume_from_checkpoint=resume_from)

merged = model.merge_and_unload()
merged.save_pretrained(WORK_DIR / "merged_finetuned")
processor.save_pretrained(WORK_DIR / "merged_finetuned")
shutil.make_archive(str(WORK_DIR / "businessflow-whisper-finetuned-hf"), "zip", str(WORK_DIR / "merged_finetuned"))

convert_to_ctranslate2(WORK_DIR / "merged_finetuned", WORK_DIR / "ct2_after")
after_result = run_benchmark(WORK_DIR / "ct2_after", "after")

print(f"{'metric':<25}{'before':<15}{'after':<15}")
print(f"{'MUCS WER':<25}{before_result['mucs_wer']*100:>6.1f}%{'':<8}{after_result['mucs_wer']*100:>6.1f}%")
print(f"{'Script consistency':<25}{before_result['script_consistency']*100:>6.1f}%{'':<8}{after_result['script_consistency']*100:>6.1f}%")
print(f"{'English WER':<25}{before_result['english_wer']*100:>6.1f}%{'':<8}{after_result['english_wer']*100:>6.1f}%")

shutil.make_archive(str(WORK_DIR / "businessflow-whisper-finetuned"), "zip", str(WORK_DIR / "ct2_after"))
print("done -- download businessflow-whisper-finetuned.zip (CTranslate2) and businessflow-whisper-finetuned-hf.zip (merged HF, for re-conversion) from the Output tab")
