"""Checks whether fine-tuning on Hindi-English code-switched data made
plain English performance *worse* -- a real risk of narrow-domain
fine-tuning (catastrophic forgetting), not just "did the target domain
improve." Run this before and after fine-tuning, same as
eval/wer_benchmark.py, and compare.

Run from the project root: python -m eval.english_regression_benchmark [--model-size base]
"""

import argparse
import json
from pathlib import Path

import jiwer

from businessflow.audio.asr import transcribe
from businessflow.audio.io import load_wav_as_tensor

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "english_regression"
_RESULTS_DIR = Path(__file__).parent / "results"

_NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def main(model_size: str = "base"):
    manifest = json.loads((_FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8"))

    references = []
    hypotheses = []
    for entry in manifest:
        audio = load_wav_as_tensor(str(_FIXTURES_DIR / entry["wav"]))
        hypothesis = transcribe(audio, model_size=model_size, language="en")
        references.append(entry["reference"])
        hypotheses.append(hypothesis)

    wer = jiwer.wer(references, hypotheses, reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)

    result = {"model_size": model_size, "utterances_scored": len(references), "wer": wer}
    print(json.dumps(result, indent=2))

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / f"english_regression_{model_size}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved to {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="base")
    args = parser.parse_args()
    main(model_size=args.model_size)
