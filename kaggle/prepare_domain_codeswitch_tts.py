"""Synthesizes real audio for datasets/colloquial-hinglish-conversations'
genuinely code-switched turns -- financial/collections-domain dialogue
(e.g. "personal loan application", "pre-approved offer", "processing
fees"), unlike MUCS (scripted lecture speech) or HiACC (general
conversation). This is the second technique from the ASR-improvement
research (Biswas et al., Oracle, Interspeech 2025) -- LLM-generated
code-switched text + TTS-synthesized audio -- except the code-switched
text doesn't need generating: it already exists, real, human-shaped, and
already on-topic for this exact bot.

Reuses this project's OWN existing TTS (audio/tts.py's speak_english/
speak_hindi), not a new dependency -- Piper and MMS-TTS are each
monolingual, so a code-switched SENTENCE ("मुझे बस 2 लाख चाहिए, for a
family function") is split into per-script WORD RUNS first (unlike
prepare_splicing_fuel.py's whole-utterance pairing, since these switches
happen mid-sentence, not between sentences), each run synthesized with
whichever engine matches its script, then concatenated -- the seams
between engines don't need to sound natural (nothing plays this audio to
a real person; it exists only to teach the model the mixed vocabulary),
so no attempt is made to disguise them beyond a short gap.

HONEST CAVEATS, read before training on this:
  - Provenance/license of the source parquet file is UNKNOWN -- no
    README or dataset card shipped with it locally. Fine for a
    demo/prototype; worth tracking down before this goes into a model
    used commercially, same discipline applied to Gramvaani's license in
    prepare_splicing_fuel.py.
  - Synthetic audio is inherently cleaner than real speech (no
    background noise, no real breathing/hesitation) -- overtraining on
    it risks teaching the model to expect "TTS-clean" input. This
    dataset is a few thousand utterances added to a much larger real
    corpus, not a replacement for one.
  - A bare digit with no rupee symbol (e.g. "2" in "2 लाख") isn't caught
    by audio/verbalizer.py's verbalize() (scoped to ISO dates and ₹/Rs.
    amounts specifically, matching production's own real behavior) --
    MMS-TTS may mispronounce a small number of these. Not specially
    handled here: production doesn't specially handle it either, and
    this is a small, known, accepted rate of noise in an otherwise real
    corpus, not a new gap invented for this dataset.

Run: python kaggle/prepare_domain_codeswitch_tts.py [--limit N]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio

from businessflow.audio.tts import speak_english, speak_hindi
from businessflow.audio.verbalizer import verbalize

_SOURCE_PARQUET = Path(__file__).resolve().parents[1] / "datasets" / "colloquial-hinglish-conversations" / "train.parquet"
_OUTPUT_DIR = Path(__file__).resolve().parent / "domain_codeswitch_tts_assets"
_TARGET_SR = 16000
_SEGMENT_GAP_SECONDS = 0.05  # short -- these are adjacent words in one sentence, not separate sentences

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
# Splits on whitespace but keeps punctuation attached to its word (so TTS
# still hears "day." as one unit, not "day" then "." alone).
_WORD_RE = re.compile(r"\S+")


def _is_genuinely_code_switched(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text)) and bool(_LATIN_RE.search(text))


def _classify_word(word: str) -> str:
    has_dev, has_lat = bool(_DEVANAGARI_RE.search(word)), bool(_LATIN_RE.search(word))
    if has_dev and not has_lat:
        return "hi"
    if has_lat and not has_dev:
        return "en"
    return "neutral"  # pure digits/punctuation -- attaches to whichever run it's inside


def _segment_by_script(text: str) -> list[tuple[str, str]]:
    """Groups words into contiguous (script, run_text) pairs. A neutral
    (no-letters) word joins the CURRENT run rather than starting its
    own -- a lone number or punctuation mark has no script of its own to
    force a new TTS call over."""
    words = _WORD_RE.findall(text)
    runs: list[tuple[str, list[str]]] = []
    for word in words:
        script = _classify_word(word)
        if script == "neutral" and runs:
            runs[-1][1].append(word)
            continue
        if script == "neutral":
            script = "en"  # text starts with a neutral token -- default to English, arbitrary but rare
        if runs and runs[-1][0] == script:
            runs[-1][1].append(word)
        else:
            runs.append((script, [word]))
    return [(script, " ".join(ws)) for script, ws in runs]


def _synthesize_run(script: str, text: str) -> torch.Tensor:
    verbalized = verbalize(text, "hi" if script == "hi" else "en")
    speech = speak_hindi(verbalized) if script == "hi" else speak_english(verbalized)
    audio = speech.audio
    if speech.sample_rate != _TARGET_SR:
        audio = torchaudio.functional.resample(audio, orig_freq=speech.sample_rate, new_freq=_TARGET_SR)
    return audio


def synthesize_utterance(text: str) -> torch.Tensor:
    runs = _segment_by_script(text)
    gap = torch.zeros(int(_SEGMENT_GAP_SECONDS * _TARGET_SR))
    parts = []
    for i, (script, run_text) in enumerate(runs):
        if i > 0:
            parts.append(gap)
        parts.append(_synthesize_run(script, run_text))
    return torch.cat(parts)


def main(limit: int | None = None) -> None:
    df = pd.read_parquet(_SOURCE_PARQUET)
    candidates = []
    for msgs in df["messages"]:
        for m in msgs:
            text = m["content"].strip()
            if text and _is_genuinely_code_switched(text):
                candidates.append(text)
    # Same text can legitimately repeat across different conversations
    # (a scripted opening line, say) -- dedup so the corpus isn't padded
    # with exact duplicates, which would just overweight one phrasing.
    candidates = list(dict.fromkeys(candidates))
    print(f"{len(candidates)} unique, genuinely code-switched utterances found")
    if limit:
        candidates = candidates[:limit]

    audio_dir = _OUTPUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = 0
    for i, text in enumerate(candidates):
        wav_name = f"domain_cs_{i:05d}.wav"
        try:
            audio = synthesize_utterance(text)
        except Exception as e:
            failures += 1
            print(f"  [{i}] synthesis failed, skipping: {e!r}")
            continue
        sf.write(str(audio_dir / wav_name), audio.numpy(), _TARGET_SR)
        manifest.append({"wav": wav_name, "reference": text, "source": "domain_codeswitch_tts"})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(candidates)} synthesized ({failures} failures so far)")

    (_OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (_OUTPUT_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "BusinessFlow Domain CodeSwitch TTS",
        "id": "mishradevang14/domain-codeswitch-tts",
        # "other" -- source parquet's own license/provenance is unknown,
        # see this script's own module docstring.
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} utterances written to {_OUTPUT_DIR} ({failures} synthesis failures)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap the number of utterances (for a quick test run)")
    args = parser.parse_args()
    main(limit=args.limit)
