"""Measures whether synthesized speech actually sounds natural/correct --
never measured before this (wer_benchmark.py measures ASR accuracy, the
other direction entirely).

Two real, concrete checks per scenario, not a vibes-based listen:

  - An automated, reference-free naturalness/intelligibility proxy via
    torchaudio's pretrained SQUIM_OBJECTIVE model (predicts STOI, PESQ,
    SI-SDR from the waveform alone -- no human rating, no reference clip
    needed). Already a dependency of this project (torchaudio is used
    for resampling elsewhere) -- no new heavy model to install. Trained
    on English speech only, so the Hindi scores below are read as a
    RELATIVE regression signal (this run vs the last one), never as an
    absolute or cross-language naturalness ranking -- SQUIM was never
    shown Hindi audio during training, and treating its Hindi output as
    gospel would be dishonest.
  - A verbalizer regression check: for every scenario whose raw text
    contains an ISO date or rupee amount (the two patterns
    audio/verbalizer.py rewrites into words), confirm verbalize() has
    actually removed that pattern before TTS ever sees it -- re-running
    verbalizer.py's OWN regexes against its own output, the same ones
    that would silently let a number reach TTS unconverted if this
    regressed (see verbalizer.py's own docstring for the exact live bug
    this class of check would have caught).

Also flags a scenario whose synthesized duration is implausible for its
text length (near-zero, or wildly long) -- a cheap sanity check for a
genuinely different failure mode (a crash returning empty audio, a
runaway loop), not a naturalness measure.

No human-rating harness here -- that's a real, separate, periodic
process this can't replace. This script's job is catching a REGRESSION
between human-rating passes, via the exact same record_run_history/
print_regression_delta convention every other eval in this project
already uses (see eval/tool_scoring.py) -- and wired into
scripts/run_eval_monitor.py's nightly suite the same way.

Run from the project root: python -m eval.voice_naturalness_benchmark
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE

from businessflow.audio.tts import speak_english, speak_hindi
from businessflow.audio.verbalizer import has_unverbalized_pattern, verbalize
from eval.tool_scoring import print_regression_delta, record_run_history

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

_SQUIM_SAMPLE_RATE = 16000  # the only rate SQUIM_OBJECTIVE accepts

# Loose, deliberately generous bounds -- this is a crash/runaway-loop
# sanity check, not a naturalness measure. 0.15s/char covers this
# project's slowest real reply comfortably; 0.3s floor just rules out
# "came back basically silent".
_MIN_PLAUSIBLE_SECONDS = 0.3
_MAX_SECONDS_PER_CHAR = 0.5


@dataclass
class Scenario:
    scenario_id: str
    language: str  # "en" | "hi"
    text: str  # as the agent would actually say it, BEFORE verbalize()


SCENARIOS = [
    Scenario(
        "plain_multi_sentence_en", "en",
        "Your EMI is due soon. Please pay before the due date to avoid a late fee.",
    ),
    Scenario(
        "date_and_amount_en", "en",
        "Your payment of ₹12,500 is due on 2026-09-15.",
    ),
    Scenario(
        "single_sentence_en", "en",
        "How can I help you with your account today?",
    ),
    Scenario(
        "plain_multi_sentence_hi", "hi",
        "आपकी EMI जल्द देय है। कृपया समय पर भुगतान करें।",
    ),
    Scenario(
        "date_and_amount_hi", "hi",
        "आपकी ₹12,500 की किस्त 2026-09-15 को देय है।",
    ),
    Scenario(
        "single_sentence_hi", "hi",
        "मैं आपकी क्या मदद कर सकता हूँ?",
    ),
]


def _is_duration_plausible(duration_seconds: float, text_length: int) -> bool:
    """A crash/runaway-loop sanity check, not a naturalness measure -- see
    _MIN_PLAUSIBLE_SECONDS/_MAX_SECONDS_PER_CHAR's own comment for how
    generous these bounds deliberately are."""
    return _MIN_PLAUSIBLE_SECONDS <= duration_seconds <= _MAX_SECONDS_PER_CHAR * max(text_length, 1)


def run_scenario(scenario: Scenario, model: torch.nn.Module) -> dict:
    verbalized_text = verbalize(scenario.text, scenario.language)
    speech = speak_hindi(verbalized_text) if scenario.language == "hi" else speak_english(verbalized_text)

    verbalizer_left_a_stray_pattern = has_unverbalized_pattern(verbalized_text)

    audio = speech.audio
    if speech.sample_rate != _SQUIM_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, orig_freq=speech.sample_rate, new_freq=_SQUIM_SAMPLE_RATE)
    with torch.no_grad():
        stoi, pesq, sisdr = model(audio.unsqueeze(0))

    duration_seconds = len(speech.audio) / speech.sample_rate
    duration_plausible = _is_duration_plausible(duration_seconds, len(scenario.text))

    return {
        "scenario_id": scenario.scenario_id,
        "language": scenario.language,
        "text": scenario.text,
        "verbalized_text": verbalized_text,
        "duration_seconds": round(duration_seconds, 3),
        "duration_plausible": duration_plausible,
        "verbalizer_left_a_stray_pattern": verbalizer_left_a_stray_pattern,
        "stoi": round(stoi.item(), 4),
        "pesq": round(pesq.item(), 4),
        "sisdr": round(sisdr.item(), 4),
    }


def _mean(key: str, rows: list[dict]) -> float | None:
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None


def main():
    model = SQUIM_OBJECTIVE.get_model()
    results = [run_scenario(s, model) for s in SCENARIOS]

    en_results = [r for r in results if r["language"] == "en"]
    hi_results = [r for r in results if r["language"] == "hi"]
    verbalizer_failures = sum(1 for r in results if r["verbalizer_left_a_stray_pattern"])
    duration_implausible_count = sum(1 for r in results if not r["duration_plausible"])

    summary = {
        "scenario_count": len(results),
        "verbalizer_failures": verbalizer_failures,
        "duration_implausible_count": duration_implausible_count,
        "en_mean_stoi": _mean("stoi", en_results),
        "en_mean_pesq": _mean("pesq", en_results),
        "en_mean_sisdr": _mean("sisdr", en_results),
        "hi_mean_stoi": _mean("stoi", hi_results),
        "hi_mean_pesq": _mean("pesq", hi_results),
        "hi_mean_sisdr": _mean("sisdr", hi_results),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    for r in results:
        flags = []
        if r["verbalizer_left_a_stray_pattern"]:
            flags.append("VERBALIZER FAILED")
        if not r["duration_plausible"]:
            flags.append("IMPLAUSIBLE DURATION")
        flag_text = f"  <-- {', '.join(flags)}" if flags else ""
        print(f"  [{r['language']}] {r['scenario_id']}: stoi={r['stoi']} pesq={r['pesq']} sisdr={r['sisdr']}{flag_text}")

    print("\n=== vs previous run (SQUIM scores, for trend visibility only -- see below) ===")
    previous = record_run_history("voice_naturalness_benchmark", summary, _RESULTS_DIR)
    # Printed for a human to eyeball the trend, but deliberately NOT fed
    # into regressed_metrics below: two back-to-back runs of this exact
    # script, no code change at all, moved en_mean_sisdr by -2.27 (real
    # measurement, see this module's own dev notes) -- Piper and MMS-TTS
    # are both genuinely stochastic (confirmed live: identical input text
    # produces different sample counts run to run), so a naive delta
    # threshold on these continuous scores would flag "REGRESSION" most
    # nights from pure model sampling noise, not a real regression. That
    # would make the nightly Telegram alert cry wolf and get ignored --
    # worse than not alerting on this signal at all. A real fix (e.g.
    # averaging several synthesis repeats per scenario, the same fix
    # latency_benchmark.py already applies to real Groq latency noise)
    # is future work; until then this stays a human-reviewed trend line.
    print_regression_delta(
        previous, summary,
        metrics=("en_mean_stoi", "en_mean_pesq", "en_mean_sisdr", "hi_mean_stoi", "hi_mean_pesq", "hi_mean_sisdr"),
    )

    # verbalizer coverage and duration sanity are deterministic correctness
    # checks, not noisy continuous scores -- zero false-positive risk from
    # model stochasticity, so these (and only these) drive the automated
    # regressed_metrics signal scripts/run_eval_monitor.py alerts on.
    regressed_metrics = []
    if verbalizer_failures:
        regressed_metrics.append("verbalizer_failures")
    if duration_implausible_count:
        regressed_metrics.append("duration_implausible_count")
    summary["regressed_metrics"] = regressed_metrics

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / "voice_naturalness_benchmark.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
