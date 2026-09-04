"""One-off sweep over faster-whisper's decode-time parameters against the
real MUCS test set -- CPU int8, matching the deployed configuration
exactly (a production failure, a badly garbled real-world code-switched
transcription, is what prompted this).

Compares a handful of TARGETED configurations against current production
(repetition_penalty=1.3, no_repeat_ngram_size=3,
condition_on_previous_text=False -- see audio/asr.py's own docstring for
why those three are already set), not a full grid search -- CPU inference
is slow enough (audio/asr.py's own docstring: ~5-6 hours for the full
3,136-utterance test split) that a grid search isn't a realistic use of
time. Each config below targets one specific hypothesis about the
garbled/hallucinated-output failure mode:

  - wider_beam: more search per step (beam_size/best_of 5->10) might
    disambiguate a genuinely ambiguous code-switched acoustic signal
    better than the default search width.
  - stricter_fallback: compression_ratio_threshold/log_prob_threshold
    are exactly the mechanism faster-whisper already has for detecting a
    degenerate hypothesis and retrying at a different sampling
    temperature -- tightening them (closer to 0) should make it retry
    MORE readily instead of accepting a bad first-pass hypothesis.
  - wider_beam_stricter_fallback: both together.

Reuses audio/asr.py's own cached model loader (_model) and default model
id (_DEFAULT_MODEL_SIZE) directly rather than transcribe()'s public
signature -- transcribe() deliberately does NOT expose these as
overridable parameters (its three hardcoded values are a curated,
already-tested default, not a general-purpose knob), and this script's
entire purpose is to test alternate values for exactly those hardcoded
constants. A one-off diagnostic reaching into that internal is the same
pattern scripts/compare_whisper_sizes.py already uses to compare across
model sizes.

Run: python -m eval.decode_params_sweep [--sample-size 30] [--seed 42]
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import jiwer

from businessflow.audio.asr import _DEFAULT_MODEL_SIZE, _model
from eval.mucs_loader import list_test_utterances, load_utterance_audio
from eval.script_metrics import dominant_script

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

_NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

_BASE_KWARGS = dict(repetition_penalty=1.3, no_repeat_ngram_size=3, condition_on_previous_text=False)

_CONFIGS = {
    "current_production": {**_BASE_KWARGS},
    "wider_beam": {**_BASE_KWARGS, "beam_size": 10, "best_of": 10},
    "stricter_fallback": {**_BASE_KWARGS, "compression_ratio_threshold": 2.0, "log_prob_threshold": -0.5},
    "wider_beam_stricter_fallback": {**_BASE_KWARGS, "beam_size": 10, "best_of": 10, "compression_ratio_threshold": 2.0, "log_prob_threshold": -0.5},
}


def _transcribe_with(audio, decode_kwargs: dict) -> str:
    segments, _info = _model(_DEFAULT_MODEL_SIZE).transcribe(audio, language="hi", **decode_kwargs)
    return " ".join(segment.text.strip() for segment in segments)


def main(sample_size: int = 30, seed: int = 42):
    all_utterances = [u for u in list_test_utterances() if u.reference_text.strip()]
    rng = random.Random(seed)
    sample = rng.sample(all_utterances, min(sample_size, len(all_utterances)))
    print(f"sampled {len(sample)} utterances (seed={seed})")

    # Loaded once, reused across every config -- decoding the same WAV
    # segment from disk N times per config would waste real CPU time.
    loaded = [(u, load_utterance_audio(u)) for u in sample]

    results = {}
    for name, decode_kwargs in _CONFIGS.items():
        start = time.time()
        refs, hyps = [], []
        for meta, audio in loaded:
            hyps.append(_transcribe_with(audio, decode_kwargs))
            refs.append(meta.reference_text)
        elapsed = time.time() - start

        wer = jiwer.wer(refs, hyps, reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)
        script_dist = Counter(dominant_script(h) for h in hyps)
        script_rate = script_dist["devanagari"] / len(hyps)

        results[name] = {
            "decode_kwargs": decode_kwargs, "wer": wer, "script_consistency_rate": script_rate,
            "script_distribution": dict(script_dist), "elapsed_seconds": round(elapsed, 1),
        }
        print(f"{name:<32} WER={wer:.4f}  script_consistency={script_rate:.4f}  ({elapsed:.1f}s for {len(sample)} utterances)")

    print("\n=== vs current_production ===")
    baseline = results["current_production"]
    for name, r in results.items():
        if name == "current_production":
            continue
        wer_delta = r["wer"] - baseline["wer"]
        script_delta = r["script_consistency_rate"] - baseline["script_consistency_rate"]
        print(f"  {name}: WER {wer_delta:+.4f}, script_consistency {script_delta:+.4f}")

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / "decode_params_sweep.json"
    out_path.write_text(json.dumps({"sample_size": len(sample), "seed": seed, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(sample_size=args.sample_size, seed=args.seed)
