"""How long each stage of the voice path takes on the machine it is run on: voice-activity trimming, speech recognition, then the
spoken reply (English and Hindi) and its Opus encoding.

This is a per-stage timing of the LOCAL stages only. It does not include the language model call (a network round trip to Groq
and the dominant cost of a turn: see eval/latency_benchmark.py and eval/results/injection_benchmark.md for turns with the model),
the database, or the network between a borrower and the server. Read it as "where the local compute goes", never as
"the latency of a voice turn".

It says which machine produced it. The deployed VM has 2 vCPUs and these stages are CPU-bound, so a number measured on a
laptop with more cores is faster than production; the ratio between stages is what transfers, not the absolute values.

Method: the fixture clip (tests/fixtures/sample_speech_en.wav, about 5 s of English speech) is run through each stage after two
warm-up runs, N timed runs each, median and 95th percentile reported. "Real-time factor" is stage time divided by the length of the
audio it handled (recognition) or produced (speech): below 1 is faster than real time.

Run from the repo root:  python -m eval.voice_stage_latency [--runs 10]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "eval" / "results"
CLIP = REPO / "tests" / "fixtures" / "sample_speech_en.wav"
ASR_MODEL = REPO / "data" / "models" / "businessflow-whisper-finetuned"
EN_REPLY = "Your next EMI of twelve thousand five hundred rupees is due on the seventh. Would you like me to send a payment link?"
HI_REPLY = "आपकी अगली किस्त बारह हज़ार पाँच सौ रुपये की है और सात तारीख को देय है। क्या मैं भुगतान का लिंक भेजूँ?"


def _timed(fn, runs: int, warmups: int = 2) -> tuple[list[float], object]:
    result = None
    for _ in range(warmups):
        result = fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def _summary(times: list[float], media_seconds: float | None = None) -> dict:
    out = {"p50_ms": round(statistics.median(times) * 1000), "p95_ms": round(float(np.percentile(times, 95)) * 1000), "runs": len(times)}
    if media_seconds:
        out["real_time_factor"] = round(statistics.median(times) / media_seconds, 3)
        out["media_seconds"] = round(media_seconds, 2)
    return out


def run(runs: int) -> dict:
    from businessflow.audio.asr import transcribe
    from businessflow.audio.tts import encode_ogg_opus, speak_english, speak_hindi
    from businessflow.audio.vad import trim_to_speech
    from businessflow.audio.verbalizer import verbalize

    data, rate = sf.read(str(CLIP), dtype="float32")
    audio = torch.from_numpy(data if data.ndim == 1 else data.mean(axis=1))
    clip_seconds = len(audio) / rate

    stages = {}
    t, trimmed = _timed(lambda: trim_to_speech(audio, sampling_rate=rate), runs)
    stages["voice-activity trimming (Silero)"] = _summary(t, clip_seconds)
    trimmed_seconds = len(trimmed) / rate

    model = str(ASR_MODEL) if ASR_MODEL.exists() else "small"
    t, text = _timed(lambda: transcribe(trimmed, model_size=model, language="en"), runs)
    stages[f"speech recognition (Whisper CTranslate2 int8, {'fine-tuned' if ASR_MODEL.exists() else 'small'})"] = _summary(t, trimmed_seconds)

    t, speech = _timed(lambda: speak_english(verbalize(EN_REPLY, "en")), runs)
    stages["speech synthesis, English (Piper)"] = _summary(t, len(speech.audio) / speech.sample_rate)
    t, speech_hi = _timed(lambda: speak_hindi(verbalize(HI_REPLY, "hi")), runs)
    stages["speech synthesis, Hindi (Piper hi_IN)"] = _summary(t, len(speech_hi.audio) / speech_hi.sample_rate)
    t, _ = _timed(lambda: encode_ogg_opus(speech), runs)
    stages["Opus encoding of the English reply"] = _summary(t)

    local_total = sum(v["p50_ms"] for k, v in stages.items() if not k.startswith("speech synthesis, Hindi"))
    return {
        "date": datetime.now().date().isoformat(),
        "machine": {"cpu": platform.processor() or platform.machine(), "logical_cores": os.cpu_count(), "torch_threads": torch.get_num_threads(), "python": platform.python_version(), "os": platform.platform()},
        "input_clip_seconds": round(clip_seconds, 2), "transcript_of_clip": text,
        "stages": stages,
        "sum_of_local_stage_medians_ms_english_path": local_total,
        "excluded": "the language-model call, the database, and the network",
    }


def markdown(summary: dict) -> str:
    m = summary["machine"]
    lines = ["# Voice path: time per local stage", "",
             "Read the header of `eval/voice_stage_latency.py`. **This is the dev machine, not the deployed VM, and it leaves out the language model call, "
             "the database and the network.**", "",
             f"Machine: {m['cpu']}, {m['logical_cores']} logical cores, torch threads {m['torch_threads']}, Python {m['python']}. Date: {summary['date']}.", "",
             "| stage | median ms | 95th pct ms | real-time factor | audio seconds | runs |", "|---|---|---|---|---|---|"]
    for name, s in summary["stages"].items():
        lines.append(f"| {name} | {s['p50_ms']} | {s['p95_ms']} | {s.get('real_time_factor', '-')} | {s.get('media_seconds', '-')} | {s['runs']} |")
    lines += ["", f"Sum of the medians on the English path (trim + recognise + synthesise + encode): {summary['sum_of_local_stage_medians_ms_english_path']} ms. "
              f"Transcript of the clip: {summary['transcript_of_clip']!r}", ""]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    summary = run(args.runs)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "voice_stage_latency.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(summary)
    (RESULTS / "voice_stage_latency.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
