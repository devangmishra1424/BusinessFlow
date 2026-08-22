"""Runs the ASR model against a sample of MUCS's held-out test split and
reports two separate, named metrics -- the "before fine-tuning" baseline
now, and the exact same script re-run against the fine-tuned weights
later:

  - WER: the strict, deployment-realistic number. A wrong-script
    hypothesis counts as wrong here, on purpose -- Hindi TTS downstream
    needs Devanagari input, so a transliterated transcript would break
    the pipeline just as much as a wrong-word one would.
  - script consistency rate: % of hypotheses that came out Devanagari
    (matching what the reference transcripts, and the downstream
    pipeline, actually expect) vs Latin transliteration vs mixed/none.
    Kept separate from WER rather than folded into a "normalized" score,
    since normalizing script away would hide a real, distinct failure
    mode rather than measure it.

(A Devanagari-primed initial_prompt was tried as a fix for the script
issue and rejected -- it forced Devanagari output but caused content
hallucination/prompt-echoing on the "base" model instead. Not used here.)

The full test split (3,136 utterances) takes ~5-6 hours at this model's
observed CPU throughput -- impractical to run in one sitting. Instead this
randomly samples N utterances (fixed seed, so it's reproducible) spread
across all 30 test recordings, not just whichever recording happens to
sort first.

Deliberately run locally, CPU int8 -- not Kaggle's GPU. The fine-tune
trains on Kaggle, but the benchmark needs to measure performance in the
same configuration the model actually gets deployed in; running "before"
on CPU and "after" on GPU would measure a hardware/precision difference,
not the fine-tune's real effect.

Run from the project root: python -m eval.wer_benchmark [--sample-size 400] [--seed 42] [--model-size base]
(module mode, not a plain script -- it imports its sibling mucs_loader.py
as part of the eval package)
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import jiwer

from businessflow.audio.asr import transcribe
from eval.mucs_loader import list_test_utterances, load_utterance_audio
from eval.script_metrics import dominant_script

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

_NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    # jiwer.wer() needs the transform to reduce all the way to
    # list[list[str]] (tokenized words), not just a cleaned string --
    # without this it rejects the transform outright.
    jiwer.ReduceToListOfListOfWords(),
])


def main(model_size: str = "base", sample_size: int = 400, seed: int = 42):
    all_utterances = [u for u in list_test_utterances() if u.reference_text.strip()]

    rng = random.Random(seed)
    sample = rng.sample(all_utterances, min(sample_size, len(all_utterances)))
    recordings_spanned = len({u.recording_id for u in sample})
    print(f"sampled {len(sample)} utterances spanning {recordings_spanned} of 30 test recordings (seed={seed})")

    references = []
    hypotheses = []
    for i, meta in enumerate(sample):
        audio = load_utterance_audio(meta)
        hypothesis = transcribe(audio, model_size=model_size, language="hi")
        references.append(meta.reference_text)
        hypotheses.append(hypothesis)

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(sample)} utterances done...")

    wer = jiwer.wer(references, hypotheses, reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)

    hypothesis_scripts = Counter(dominant_script(h) for h in hypotheses)
    reference_scripts = Counter(dominant_script(r) for r in references)
    n = len(references)
    script_consistency_rate = hypothesis_scripts["devanagari"] / n if n else 0.0

    result = {
        "model_size": model_size,
        "sample_size": n,
        "seed": seed,
        "recordings_spanned": recordings_spanned,
        "wer": wer,
        "script_consistency_rate": script_consistency_rate,
        "hypothesis_script_distribution": dict(hypothesis_scripts),
        "reference_script_distribution": dict(reference_scripts),
    }

    print(f"\nmodel_size={model_size}")
    print(f"utterances scored: {n}")
    print(f"WER: {wer:.4f}")
    print(f"script consistency rate (hypotheses that are Devanagari): {script_consistency_rate:.4f}")
    print(f"hypothesis script distribution: {dict(hypothesis_scripts)}")

    _RESULTS_DIR.mkdir(exist_ok=True)
    # model_size is sometimes a local filesystem path (a fine-tuned model
    # directory, not a built-in size name like "base") -- embedding its
    # separators straight into the filename would try to write through
    # nonexistent nested directories instead of naming the result file.
    safe_label = model_size.replace("/", "_").replace("\\", "_")
    out_path = _RESULTS_DIR / f"mucs_test_{safe_label}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-size", default="base")
    args = parser.parse_args()
    main(model_size=args.model_size, sample_size=args.sample_size, seed=args.seed)
