"""Builds the monolingual "splicing fuel" dataset for the code-mix audio
splicing technique (Biswas et al., Oracle, Interspeech 2025) -- randomly
concatenating a monolingual Hindi utterance with a monolingual English one
during training teaches the model to handle a language switch mid-utterance,
without needing any additional real code-switched audio at all.

Sources, both real speech, deliberately NOT MUCS/HiACC (already the actual
code-switched training corpus -- fuel here means monolingual only):

  - Hindi: Gramvaani's "1111 Hours Hindi ASR Challenge" dev set (OpenSLR
    SLR118, GV_Dev_5h -- https://openslr.org/118/), SPONTANEOUS TELEPHONE
    speech. Deliberately chosen over a scripted/read corpus: a separate
    finding (Vividh-ASR, arXiv:2605.13087) documents that fine-tuning on
    scripted speech degrades performance on spontaneous speech, which
    matches this project's own real production failure (a garbled
    transcription of natural, spontaneous Hindi-English speech) -- MUCS
    itself is scripted lecture speech, so this splicing fuel is a chance to
    add real spontaneous-register Hindi into the mix, not just more of the
    same register.
  - English: LibriSpeech dev-clean (OpenSLR SLR12), standard read speech --
    matches the paper's own use of Common Voice for this role; audiobook
    narration reads no worse for this purpose than Common Voice's read
    sentences, and needs no gated-dataset request (Common Voice, Kathbath,
    and Shrutilipi were all tried first and are all gated on Hugging Face,
    requiring a human to click "agree" on each dataset's own page before any
    token can access it -- not something this script can do on its own).

LICENSE NOTE, read before using the resulting model for anything beyond a
demo/prototype: Gramvaani's data is licensed for free ACADEMIC use only --
"permission for any commercial use of the data should be sought by writing
to contact@gramvaani.org" (openslr.org/118's own license text). LibriSpeech
is CC-BY-4.0, no such restriction. A model fine-tuned including the Gramvaani
slice inherits that same academic-only restriction until that permission is
sought -- this is flagged explicitly in kaggle/splicing_fuel_assets/
dataset-metadata.json's license field (set to "other", not a standard
open license) rather than glossed over.

Output: kaggle/splicing_fuel_assets/audio/{hi,en}_NNNNN.wav (mono, 16kHz --
resampled from Gramvaani's native 8kHz telephone rate; Whisper's feature
extractor expects 16kHz and does not resample itself) and manifest.json
({"wav", "reference", "language"} per entry).

Run: python kaggle/prepare_splicing_fuel.py
"""

import json
import re
from pathlib import Path

import soundfile as sf
import torchaudio

_ROOT = Path(__file__).resolve().parent
_GRAMVAANI_DIR = _ROOT.parent / "datasets" / "monolingual_splice_fuel" / "gramvaani" / "GV_Dev_5h"
_LIBRISPEECH_DIR = _ROOT.parent / "datasets" / "monolingual_splice_fuel" / "librispeech" / "LibriSpeech" / "dev-clean"
_OUTPUT_DIR = _ROOT / "splicing_fuel_assets"
_TARGET_SR = 16000

# Gramvaani's crowdsourced transcripts mark a genuinely inaudible stretch
# with this literal token -- an entry containing it has an incomplete/
# unreliable reference, not real training signal, so it's skipped rather
# than trained on as if it were a real transcript.
_INAUDIBLE_MARKER = "<inaudible>"


def _load_gramvaani_entries() -> list[dict]:
    scp = {}
    with open(_GRAMVAANI_DIR / "mp3.scp", encoding="utf-8") as f:
        for line in f:
            utt_id, rel_path = line.strip().split("\t")
            scp[utt_id] = rel_path

    entries = []
    with open(_GRAMVAANI_DIR / "text", encoding="utf-8") as f:
        for line in f:
            utt_id, _, text = line.strip().partition(" ")
            text = text.strip()
            if not text or _INAUDIBLE_MARKER in text or utt_id not in scp:
                continue
            entries.append({"utt_id": utt_id, "audio_path": _GRAMVAANI_DIR / scp[utt_id], "reference": text, "language": "hi"})
    return entries


_LIBRISPEECH_TRANS_LINE = re.compile(r"^(\S+) (.+)$")


def _load_librispeech_entries() -> list[dict]:
    entries = []
    for trans_file in sorted(_LIBRISPEECH_DIR.glob("*/*/*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                match = _LIBRISPEECH_TRANS_LINE.match(line.strip())
                if not match:
                    continue
                utt_id, text = match.groups()
                # LibriSpeech's own transcripts are ALL CAPS by convention --
                # lowercased here so spliced-in English doesn't read as a
                # jarring change in register next to the naturally-cased
                # Hindi/English text elsewhere in the real training corpus.
                entries.append({
                    "utt_id": utt_id, "audio_path": chapter_dir / f"{utt_id}.flac",
                    "reference": text.strip().lower(), "language": "en",
                })
    return entries


def _write_resampled_wav(audio_path: Path, out_path: Path) -> bool:
    """Returns False (and writes nothing) if the file can't be decoded --
    a handful of corrupt entries in a real, unvetted crowdsourced corpus
    must not take down the whole prep run."""
    try:
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception:
        return False
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _TARGET_SR:
        import torch
        audio = torchaudio.functional.resample(torch.from_numpy(audio), orig_freq=sr, new_freq=_TARGET_SR).numpy()
    sf.write(str(out_path), audio, _TARGET_SR)
    return True


def main() -> None:
    audio_dir = _OUTPUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for prefix, loader in (("hi", _load_gramvaani_entries), ("en", _load_librispeech_entries)):
        entries = loader()
        print(f"{prefix}: {len(entries)} candidate utterances")
        written = 0
        for i, entry in enumerate(entries):
            wav_name = f"{prefix}_{i:05d}.wav"
            if _write_resampled_wav(entry["audio_path"], audio_dir / wav_name):
                manifest.append({"wav": wav_name, "reference": entry["reference"], "language": entry["language"]})
                written += 1
        print(f"{prefix}: {written}/{len(entries)} written (rest failed to decode)")

    (_OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (_OUTPUT_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "BusinessFlow Splicing Fuel Hi-En",
        "id": "mishradevang14/splicing-fuel",
        # "other", not a standard open license -- see this script's own
        # module docstring for why (Gramvaani's academic-use-only term).
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} total utterances written to {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
