"""One-off: adds the 996 Corpus/adult (HiACC) utterances not already in
the training set -- confirmed via real MD5 hash matching (not filename
guessing) that the existing 2,322 hiacc_*.wav entries are an exact match
for Corpus/adult's train split; the test (664) + val (332) splits were
never included. Copies their audio into kaggle/training_assets/audio/
continuing the existing hiacc_ numbering, and appends matching entries
to manifest.json.

Run: python scripts/add_hiacc_adult_remainder.py
"""

import json
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ADULT_DIR = _PROJECT_ROOT / "datasets" / "Corpus" / "adult"
_ASSETS_DIR = _PROJECT_ROOT / "kaggle" / "training_assets"

_SPLITS = [
    ("test_split", "combined_output_changed_test_output.txt"),
    ("val_split", "combined_output_changed_val_output.txt"),
]


def _parse_transcript(path: Path) -> dict[str, str]:
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        wav_name, _, text = line.partition(",")
        mapping[wav_name.strip()] = text.strip()
    return mapping


def main():
    manifest = json.loads((_ASSETS_DIR / "manifest.json").read_text(encoding="utf-8"))
    existing_count = len(manifest)
    next_index = sum(1 for m in manifest if m["wav"].startswith("hiacc_"))

    added = 0
    skipped_no_text = 0
    skipped_no_audio = 0

    for split_dir, transcript_file in _SPLITS:
        transcripts = _parse_transcript(_ADULT_DIR / "transcription" / transcript_file)
        audio_dir = _ADULT_DIR / "audio" / split_dir
        for wav_name, text in transcripts.items():
            if not text:
                skipped_no_text += 1
                continue
            src_path = audio_dir / wav_name
            if not src_path.exists():
                skipped_no_audio += 1
                continue
            new_name = f"hiacc_{next_index:05d}.wav"
            shutil.copyfile(src_path, _ASSETS_DIR / "audio" / new_name)
            manifest.append({"wav": new_name, "reference": text, "source": "hiacc"})
            next_index += 1
            added += 1

    (_ASSETS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"existing manifest entries: {existing_count}")
    print(f"added: {added}")
    print(f"skipped (no text): {skipped_no_text}")
    print(f"skipped (audio file missing): {skipped_no_audio}")
    print(f"new manifest total: {len(manifest)}")


if __name__ == "__main__":
    main()
