"""Round 2 fine-tune (see kaggle/training-kernel-r32-round2/train_r32_round2.py
for the corpus-expansion half of this recipe -- unchanged here) PLUS the
on-the-fly code-mix audio splicing technique from Biswas et al. (Oracle,
Interspeech 2025, "Adapting Whisper for low-resource Hindi-English
Code-Mix speech") -- randomly concatenating a monolingual Hindi utterance
with a monolingual English one during training, teaching the model to
handle a language switch mid-utterance without needing any additional
real code-switched audio. On the SAME MUCS dataset this project already
uses, that paper reports 3.7-25% relative MER improvement from splicing
alone (depending how much real data it's combined with), and their best
combined result (splicing + synthetic LLM/TTS data, M9) reached 31%
relative improvement using NO real in-domain audio at all.

Splicing fuel (kaggle/prepare_splicing_fuel.py's own docstring has the
full sourcing rationale and a LICENSE NOTE worth reading before using the
resulting model beyond a demo): 1,871 Hindi utterances from Gramvaani's
spontaneous TELEPHONE speech (OpenSLR SLR118) and 2,703 English
utterances from LibriSpeech dev-clean (OpenSLR SLR12) -- both genuinely
monolingual, neither overlapping with the real MUCS/HiACC training data.

Simplification versus the paper's literal description: the paper tags
each spliced example with an explicit language-specific prompt token
matching its first segment. This recipe does NOT do that -- the EXISTING
proven recipe (round 1 and round 2, both already deployed/staged) never
sets an explicit per-example language token for its own genuinely
code-switched MUCS/HiACC labels either; it lets the mixed-script TEXT
LABEL alone carry the language signal, exactly as a real code-switched
utterance already does. Doing the same for synthetic splices keeps this
one consistent with how the rest of this pipeline already represents
code-switching, rather than introducing a second, differently-tagged
convention only synthetic examples would use.

Splice volume: min(1871, 2703) = 1,871 synthetic virtual examples added
on top of the 56,143 real ones (~3.2% of the epoch) -- generated fresh
"on the fly" on every access (not pre-cached), so across 3 epochs the
same virtual index sees a different real Hindi+English pairing each
time, maximizing how many distinct language-switch points the model
actually sees from a fixed, modest fuel pool. This is a deliberately
modest augmentation fraction, not a replacement for the real corpus --
the paper's own best result comes from combining both, not choosing one.

Everything else (dependency pinning, corpus-expansion decisions, LoRA
config, benchmark harness) is exactly train_r32_round2.py's own, unchanged
recipe -- see that script's docstring for the history behind each of
those choices; not re-derived here.
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
import numpy as np
import peft
import soundfile as sf
import torch
import transformers

print(f"transformers={transformers.__version__} peft={peft.__version__} torch={torch.__version__}")

BASE_MODEL_DIR = Path("/kaggle/input/datasets/mishradevang14/businessflow-whisper-finetuned-r32-hf")
DATASET_DIR = Path("/kaggle/input/datasets/mishradevang14/training-set")
SPLICE_DIR = Path("/kaggle/input/datasets/mishradevang14/splicing-fuel")
TEST_DIR = Path("/kaggle/input/datasets/mishradevang14/mucs-2021-hindi-english-test")
WORK_DIR = Path("/kaggle/working")

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_NORMALIZE = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
_DECODE_KWARGS = dict(repetition_penalty=1.3, no_repeat_ngram_size=3, condition_on_previous_text=False)

_SAMPLE_RATE = 16000
_MAX_AUDIO_SAMPLES = 30 * _SAMPLE_RATE  # Whisper's own 30s input window


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


def _resolve_splice_dir():
    """Same shape as _resolve_assets_dir but for the splicing-fuel dataset:
    one flat-root audio.zip plus a top-level manifest.json (see
    kaggle/upload_splicing_fuel.py), no "english_regression" subfolder."""
    audio_dir = SPLICE_DIR / "audio"
    if audio_dir.is_dir() and any(audio_dir.iterdir()):
        return SPLICE_DIR
    extracted = WORK_DIR / "splicing_fuel_assets"
    target = extracted / "audio"
    zip_path = SPLICE_DIR / "audio.zip"
    if zip_path.exists() and not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    manifest_src = SPLICE_DIR / "manifest.json"
    if manifest_src.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_src, extracted / "manifest.json")
    return extracted


ASSETS_DIR = _resolve_assets_dir()
SPLICE_ASSETS_DIR = _resolve_splice_dir()
print(f"training assets resolved to: {ASSETS_DIR}")
print(f"splicing fuel resolved to: {SPLICE_ASSETS_DIR}")


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

# Same r=32/alpha=64 as round 1 -- a fresh adapter on top of the
# already-fine-tuned weights (merge_and_unload collapsed the original
# adapter into the base, so there's no round-1 adapter left to resume).
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

manifest = json.loads((ASSETS_DIR / "manifest.json").read_text(encoding="utf-8"))
print(f"training on {len(manifest)} real utterances (round 2, expanded corpus)")

splice_manifest = json.loads((SPLICE_ASSETS_DIR / "manifest.json").read_text(encoding="utf-8"))
splice_hi = [e for e in splice_manifest if e["language"] == "hi"]
splice_en = [e for e in splice_manifest if e["language"] == "en"]
NUM_SPLICE_PAIRS = min(len(splice_hi), len(splice_en))
print(f"splicing fuel: {len(splice_hi)} Hindi + {len(splice_en)} English monolingual utterances "
      f"-> {NUM_SPLICE_PAIRS} synthetic virtual examples/epoch ({NUM_SPLICE_PAIRS / (len(manifest) + NUM_SPLICE_PAIRS):.1%} of the epoch)")


class SpeechDataset(torch.utils.data.Dataset):
    """Indices [0, len(manifest)) are real MUCS/HiACC examples, unchanged
    from round 2. Indices beyond that are synthetic code-mix splices --
    a fresh random Hindi+English fuel pairing generated on EVERY access
    (not cached), so the same virtual index yields a different real
    pairing epoch to epoch. See this module's own docstring for why no
    explicit language-token tagging is applied to the spliced label."""

    def __init__(self, manifest, audio_dir, splice_hi, splice_en, splice_audio_dir, num_splice_pairs, tokenizer):
        self.manifest = manifest
        self.audio_dir = audio_dir
        self.splice_hi = splice_hi
        self.splice_en = splice_en
        self.splice_audio_dir = splice_audio_dir
        self.num_splice_pairs = num_splice_pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.manifest) + self.num_splice_pairs

    def _real_example(self, idx):
        entry = self.manifest[idx]
        audio, _sr = sf.read(str(self.audio_dir / entry["wav"]), dtype="float32")
        labels = self.tokenizer(entry["reference"], truncation=True, max_length=_MAX_TARGET_POSITIONS).input_ids
        return {"audio": audio, "labels": labels}

    def _spliced_example(self):
        hi_entry = random.choice(self.splice_hi)
        en_entry = random.choice(self.splice_en)
        hi_audio, _ = sf.read(str(self.splice_audio_dir / hi_entry["wav"]), dtype="float32")
        en_audio, _ = sf.read(str(self.splice_audio_dir / en_entry["wav"]), dtype="float32")
        # Random order -- whichever segment actually comes first in the
        # concatenated audio is also whichever comes first in the label.
        if random.random() < 0.5:
            first_audio, first_text = hi_audio, hi_entry["reference"]
            second_audio, second_text = en_audio, en_entry["reference"]
        else:
            first_audio, first_text = en_audio, en_entry["reference"]
            second_audio, second_text = hi_audio, hi_entry["reference"]
        audio = np.concatenate([first_audio, second_audio])[:_MAX_AUDIO_SAMPLES]
        labels = self.tokenizer(f"{first_text} {second_text}", truncation=True, max_length=_MAX_TARGET_POSITIONS).input_ids
        return {"audio": audio, "labels": labels}

    def __getitem__(self, idx):
        if idx < len(self.manifest):
            return self._real_example(idx)
        return self._spliced_example()


train_dataset = SpeechDataset(
    manifest, ASSETS_DIR / "audio",
    splice_hi, splice_en, SPLICE_ASSETS_DIR / "audio", NUM_SPLICE_PAIRS,
    processor.tokenizer,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        audio = [f["audio"] for f in features]
        batch = self.processor.feature_extractor(audio, sampling_rate=_SAMPLE_RATE, return_tensors="pt")
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
shutil.make_archive(str(WORK_DIR / "businessflow-whisper-r32-round2-splice-hf"), "zip", str(WORK_DIR / "merged_finetuned"))

convert_to_ctranslate2(WORK_DIR / "merged_finetuned", WORK_DIR / "ct2_after")
after_result = run_benchmark(WORK_DIR / "ct2_after", "after_r32_round2_splice")

print(f"{'metric':<25}{'before (round 1)':<18}{'after (round 2 + splice)':<25}")
print(f"{'MUCS WER':<25}{before_result['mucs_wer']*100:>6.1f}%{'':<11}{after_result['mucs_wer']*100:>6.1f}%")
print(f"{'Script consistency':<25}{before_result['script_consistency']*100:>6.1f}%{'':<11}{after_result['script_consistency']*100:>6.1f}%")
print(f"{'English WER':<25}{before_result['english_wer']*100:>6.1f}%{'':<11}{after_result['english_wer']*100:>6.1f}%")

shutil.make_archive(str(WORK_DIR / "businessflow-whisper-r32-round2-splice"), "zip", str(WORK_DIR / "ct2_after"))
print("done -- download businessflow-whisper-r32-round2-splice.zip (CTranslate2) and "
      "businessflow-whisper-r32-round2-splice-hf.zip (merged HF) from the Output tab")
