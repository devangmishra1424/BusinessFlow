"""Extracts individual utterance clips from MUCS train (cutting them out
of MUCS's long recordings, since it ships long-form audio + a segments
file, not individual utterance files) and HiACC adult train (already
individual clips, just copied over). Bundles the English regression
fixtures alongside, all in one folder meant to be uploaded as a Kaggle
dataset.

MUCS_SAMPLE_SIZE/HIACC_SAMPLE_SIZE default to the FULL pool of each
(52,825 MUCS + 2,322 HiACC as of this repo's copy) -- the first fine-
tune deliberately used a curated ~7% sample (2,500 + 1,000) to keep the
Kaggle upload small; this now pulls everything, since that sample size
was the actual constraint being loosened, not a hardcoded requirement.
Still individually cut, correctly-shaped clips either way, not a
9.7GB dump of MUCS's original long-form audio + segments file.

Run: python -m eval.extract_training_sample
"""

import json
import random
import shutil
from pathlib import Path

import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MUCS_TRAIN_DIR = _PROJECT_ROOT / "datasets" / "openSLR" / "train"
_HIACC_TRAIN_DIR = _PROJECT_ROOT / "datasets" / "Corpus" / "adult"
_ENGLISH_REGRESSION_DIR = _PROJECT_ROOT / "eval" / "fixtures" / "english_regression"
_OUTPUT_DIR = _PROJECT_ROOT / "kaggle" / "training_assets"

MUCS_SAMPLE_SIZE = 52_825
HIACC_SAMPLE_SIZE = 2_322
SEED = 42


def _load_mucs_train_utterances():
    segments = {}
    with open(_MUCS_TRAIN_DIR / "transcripts" / "segments", encoding="utf-8") as f:
        for line in f:
            utt_id, recording_id, start, end = line.split()
            segments[utt_id] = (recording_id, float(start), float(end))

    texts = {}
    with open(_MUCS_TRAIN_DIR / "transcripts" / "text", encoding="utf-8") as f:
        for line in f:
            utt_id, _, text = line.strip().partition(" ")
            texts[utt_id] = text

    result = []
    for utt_id, (recording_id, start, end) in segments.items():
        text = texts.get(utt_id, "")
        if not text.strip():
            continue
        if not (_MUCS_TRAIN_DIR / f"{recording_id}.wav").exists():
            continue
        result.append((utt_id, recording_id, start, end, text))
    return result


def _load_hiacc_train_utterances():
    transcription_file = _HIACC_TRAIN_DIR / "transcription" / "combined_output_changed_train_output.txt"
    result = []
    with open(transcription_file, encoding="utf-8") as f:
        for line in f:
            if "," not in line:
                continue
            filename, _, text = line.strip().partition(",")
            filename, text = filename.strip(), text.strip()
            if not text:
                continue
            if (_HIACC_TRAIN_DIR / "audio" / "train_split" / filename).exists():
                result.append((filename, text))
    return result


def main():
    rng = random.Random(SEED)
    out_audio_dir = _OUTPUT_DIR / "audio"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    mucs_utterances = _load_mucs_train_utterances()
    recordings_present = {u[1] for u in mucs_utterances}
    print(f"MUCS train pool: {len(mucs_utterances)} utterances across {len(recordings_present)} recordings")

    mucs_sample = rng.sample(mucs_utterances, min(MUCS_SAMPLE_SIZE, len(mucs_utterances)))
    recordings_used = set()
    for i, (utt_id, recording_id, start, end, text) in enumerate(mucs_sample):
        wav_path = _MUCS_TRAIN_DIR / f"{recording_id}.wav"
        info = sf.info(str(wav_path))
        start_frame = int(start * info.samplerate)
        num_frames = int((end - start) * info.samplerate)
        audio, sr = sf.read(str(wav_path), start=start_frame, frames=num_frames, dtype="float32")
        out_name = f"mucs_{i:05d}.wav"
        sf.write(str(out_audio_dir / out_name), audio, sr)
        manifest.append({"wav": out_name, "reference": text, "source": "mucs"})
        recordings_used.add(recording_id)
        if (i + 1) % 500 == 0:
            print(f"  MUCS: {i + 1}/{len(mucs_sample)} extracted")

    print(f"MUCS sample spans {len(recordings_used)} of {len(recordings_present)} distinct recordings")

    hiacc_utterances = _load_hiacc_train_utterances()
    print(f"HiACC train pool: {len(hiacc_utterances)} utterances")
    hiacc_sample = rng.sample(hiacc_utterances, min(HIACC_SAMPLE_SIZE, len(hiacc_utterances)))
    for i, (filename, text) in enumerate(hiacc_sample):
        out_name = f"hiacc_{i:05d}.wav"
        shutil.copyfile(_HIACC_TRAIN_DIR / "audio" / "train_split" / filename, out_audio_dir / out_name)
        manifest.append({"wav": out_name, "reference": text, "source": "hiacc"})

    shutil.copytree(_ENGLISH_REGRESSION_DIR, _OUTPUT_DIR / "english_regression", dirs_exist_ok=True)

    (_OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ntotal training utterances: {len(manifest)}")
    print(f"output written to: {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
