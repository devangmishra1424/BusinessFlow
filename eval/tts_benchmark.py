"""Benchmark of the Hindi text-to-speech path: clarity (word and character errors), robustness, consistency,
speed and memory, for the engines that matter:

  new_piper     what this branch ships: Piper's hi_IN voice (TTS_HINDI_ENGINE=piper), new verbalizer, script-run splitting
  new_mms       the same code with Meta MMS first (TTS_HINDI_ENGINE=mms): isolates the engine from the other changes
  old_deployed  the code that was live before 2026-10-10 (BASELINE_REV: MMS only, old verbalizer, no script-run splitting)

    python -m eval.tts_benchmark generate          # synthesize every sentence (several times for the consistency
                                                   #   figures), measure the audio, write wav files + manifest.json
    python -m eval.tts_benchmark memory <engine>   # resident memory of one engine in a fresh process (run per engine)
    python -m eval.tts_benchmark report            # merge manifest + transcripts.json + memory probes into the tables

Clarity is measured by a read-back: another model (Whisper large-v3, run on a Kaggle GPU by kaggle/tts-score-kernel)
transcribes each clip, and the transcript is compared with the text that SHOULD have been spoken. The reference is the
current verbalizer's output for every engine, so an engine that drops a number or a word pays for it. This is a measure
of intelligibility. It is not a measure of how natural or how Indian-accented a voice sounds, and it is one judge model.

PRE-REGISTERED PASS CRITERIA (written before any result was seen; the report prints each as PASS/FAIL):
  C1 robustness      new_piper: no exception and no empty/near-empty clip on any sentence in any repeat.
  C2 clarity         over hi_plain+hi_numbers+hi_acronyms+hi_long: mean character error of new_piper <= new_mms (point
                     estimate), and the 95% CI upper bound of the paired difference (new_piper - new_mms) < +0.03.
  C3 consistency     mean duration variation across repeats (coefficient of variation) of new_piper <= 1.25 x new_mms; the
                     share of sentences whose repeats transcribe identically is not lower than new_mms's by more than 0.10.
  C4 audio hygiene   new_piper: no clipped clip (peak >= 0.999), no clip whose Opus-decoded peak exceeds 1.0 (what a listener's
                     player receives), and a standard deviation of loudness across clips <= 3.0 dB. (The first run failed the
                     peak test on every Piper clip: Piper normalises to exactly 1.0 and 7 of 12 clips decoded above 1.0 after
                     Opus. The product was fixed with -1 dBFS headroom; the Opus check was then added; the criterion was not relaxed.
                     12 clips proved too few: a later CI run caught one at 1.021, and 470 clips showed about 3% clip on a
                     48 kHz player, so encode_ogg_opus now limits to -2 dBFS before Opus -- see _OPUS_PEAK_LIMIT in tts.py.)
  C5 resources       new_piper: speed (real-time factor) <= new_mms; memory growth after the first synthesis <= new_mms.
  C6 leakage         digits left in the text sent to the Hindi voice: 0 for the new verbalizer, and fewer silently dropped
                     tokens (digit runs + Latin words) than the old verbalizer.
  C7 regression      new_piper character error is not worse than old_deployed's on hi_plain (the sentences nothing changed for).
The merge decision is the conjunction. Any FAIL is reported and the change is held.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

from eval.stats import bootstrap_ci, coefficient_of_variation, paired_ci  # noqa: F401  (re-exported for callers and tests)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "eval" / "fixtures" / "tts_benchmark_sentences.json"
RESULTS = REPO / "eval" / "results"
AUDIO_DIR = RESULTS / "tts_benchmark_audio"  # not committed (see .gitignore)
BASELINE_REV = "6d5773d"  # main before PR #3 (the Hindi robustness fix) was deployed

CLARITY_CATEGORIES = ("hi_plain", "hi_numbers", "hi_acronyms", "hi_long")
ENGINES = {
    "new_piper": {"tts": "new", "env": {"TTS_HINDI_ENGINE": "piper"}, "verbalizer": "new", "reps": 3},
    "new_mms": {"tts": "new", "env": {"TTS_HINDI_ENGINE": "mms"}, "verbalizer": "new", "reps": 3},
    "old_deployed": {"tts": "old", "env": {}, "verbalizer": "old", "reps": 1},
}


# --------------------------------------------------------------------------------------------------
# Pure helpers (tested in tests/test_tts_benchmark.py without any model)
# --------------------------------------------------------------------------------------------------
def norm_hi(text: str) -> str:
    """Spelling-insensitive form for comparing Hindi: no punctuation, chandrabindu = anusvara, no nukta, lower-case Latin."""
    t = unicodedata.normalize("NFD", text.lower())
    t = t.replace("ँ", "ं")  # chandrabindu -> anusvara (पाँच / पांच)
    t = t.replace("़", "")  # nukta (ज़ -> ज)
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[।॥.,!?;:\"'’()\-–—/%₹]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) if audio.size else 0.0
    return -120.0 if rms <= 0 else 20.0 * math.log10(rms)


def edge_silence_ms(audio: np.ndarray, sr: int, threshold_dbfs: float = -45.0, win_ms: int = 10) -> tuple[float, float]:
    """Milliseconds of near-silence at the start and at the end (windows below threshold_dbfs)."""
    win = max(1, int(sr * win_ms / 1000))
    n = len(audio) // win
    if n == 0:
        return 0.0, 0.0
    loud = [rms_dbfs(audio[i * win:(i + 1) * win]) > threshold_dbfs for i in range(n)]
    if not any(loud):
        return n * win_ms, n * win_ms
    first, last = loud.index(True), n - 1 - loud[::-1].index(True)
    return first * win_ms, (n - 1 - last) * win_ms


# --------------------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------------------
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_modules():
    """The tts.py and verbalizer.py of BASELINE_REV, exactly as they were, loaded under other names. Placed three
    directories below the repo root because tts.py finds its Piper voice relative to its own location."""
    dest = REPO / ".bench_old" / "businessflow" / "audio"
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("tts.py", "verbalizer.py"):
        content = subprocess.check_output(["git", "show", f"{BASELINE_REV}:src/businessflow/audio/{f}"], cwd=REPO)
        (dest / f"old_{f}").write_bytes(content)
    return _load_module("old_tts", dest / "old_tts.py"), _load_module("old_verbalizer", dest / "old_verbalizer.py")


def _sentences() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["sentences"]


def cmd_generate(args) -> None:
    import soundfile as sf
    import torch

    torch.set_num_threads(2)  # the VM has 2 vCPUs
    sys.path.insert(0, str(REPO / "src"))
    from businessflow.audio import tts as new_tts, verbalizer as new_verbalizer

    old_tts, old_verbalizer = _baseline_modules()
    modules = {"new": (new_tts, new_verbalizer), "old": (old_tts, old_verbalizer)}
    sentences = _sentences()
    only = set(args.engines.split(",")) if args.engines else set(ENGINES)
    manifest_path = AUDIO_DIR / "manifest.json"
    clips = json.loads(manifest_path.read_text(encoding="utf-8"))["clips"] if (manifest_path.exists() and args.engines) else []
    clips = [c for c in clips if c["engine"] not in only]
    load_times = json.loads(manifest_path.read_text(encoding="utf-8")).get("load_s", {}) if manifest_path.exists() else {}

    for engine, cfg in ENGINES.items():
        if engine not in only:
            continue
        tts, _ = modules[cfg["tts"]]
        verbalizer = modules[cfg["verbalizer"]][1]
        for k, v in cfg["env"].items():
            os.environ[k] = v
        out_dir = AUDIO_DIR / engine
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        tts.speak_hindi("आपकी किस्त देय है।")  # warm-up: loads the model, so synthesis timings below exclude it
        load_times[engine] = round(time.perf_counter() - t0, 1)
        print(f"[{engine}] warm-up/load {load_times[engine]}s", flush=True)
        for s in sentences:
            reference = norm_hi(new_verbalizer.verbalize(s["text"], "hi"))  # what SHOULD be heard, whatever the engine
            spoken = verbalizer.verbalize(s["text"], "hi")  # what this engine's own pipeline actually sends to its voice
            for rep in range(cfg["reps"]):
                row = {"engine": engine, "id": s["id"], "category": s["category"], "rep": rep, "text": s["text"],
                       "reference": reference, "spoken_text": spoken, "ok": False}
                t1 = time.perf_counter()
                try:
                    speech = tts.speak_hindi(spoken)
                    synth_s = time.perf_counter() - t1
                    audio = speech.audio.numpy().astype(np.float32)
                    sr = int(speech.sample_rate)
                    dur = len(audio) / sr
                    lead, trail = edge_silence_ms(audio, sr)
                    decoded, _ = sf.read(io.BytesIO(tts.encode_ogg_opus(speech)), dtype="float32")  # float decode is not clipped to +-1
                    opus_peak = float(np.max(np.abs(decoded))) if decoded.size else 0.0
                    path = out_dir / f"{s['id']}_r{rep}.wav"
                    sf.write(str(path), audio, sr, subtype="PCM_16")
                    row.update(ok=dur >= 0.3, empty=dur < 0.3, path=path.relative_to(AUDIO_DIR).as_posix(), duration_s=round(dur, 3),
                               synth_s=round(synth_s, 3), rtf=round(synth_s / dur, 3) if dur else None, sample_rate=sr,
                               rms_dbfs=round(rms_dbfs(audio), 2), peak=round(float(np.max(np.abs(audio))), 4),
                               lead_ms=lead, trail_ms=trail, clipped=bool(np.max(np.abs(audio)) >= 0.999), opus_peak=round(opus_peak, 4), opus_over=opus_peak > 1.0)
                except Exception as e:  # a failure is a RESULT of this benchmark: recorded, never skipped
                    row.update(error=f"{type(e).__name__}: {str(e)[:160]}")
                clips.append(row)
            print(f"[{engine}] {s['id']} done", flush=True)
        # persist after each engine so a crash does not lose finished ones
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"baseline_rev": BASELINE_REV, "load_s": load_times, "clips": clips}, ensure_ascii=False, indent=1), encoding="utf-8")

    ok = sum(c["ok"] for c in clips)
    print(f"wrote {manifest_path}: {len(clips)} clips, {ok} ok, {len(clips) - ok} failed or empty")


# --------------------------------------------------------------------------------------------------
# memory (one engine in a fresh process)
# --------------------------------------------------------------------------------------------------
def cmd_memory(args) -> None:
    import psutil

    cfg = ENGINES[args.engine]
    proc = psutil.Process()
    sys.path.insert(0, str(REPO / "src"))
    import torch  # noqa: F401  (imported before the baseline, as the bot has it loaded anyway)

    if cfg["tts"] == "new":
        from businessflow.audio import tts
    else:
        tts, _ = _baseline_modules()
    for k, v in cfg["env"].items():
        os.environ[k] = v
    rss_base = proc.memory_info().rss
    t0 = time.perf_counter()
    tts.speak_hindi("आपकी किस्त देय है।")
    load_s = time.perf_counter() - t0
    rss_after = proc.memory_info().rss
    print(json.dumps({"engine": args.engine, "rss_base_mb": round(rss_base / 1e6), "rss_after_first_synth_mb": round(rss_after / 1e6),
                      "growth_mb": round((rss_after - rss_base) / 1e6), "first_call_s": round(load_s, 1)}))


# --------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------
def _leakage(sentences: list[dict]) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from businessflow.audio import verbalizer as new_verbalizer

    _, old_verbalizer = _baseline_modules()
    out = {}
    for name, v in (("old", old_verbalizer), ("new", new_verbalizer)):
        digits = latin = 0
        for s in sentences:
            if s["category"] == "hinglish_roman":
                continue
            text = v.verbalize(s["text"], "hi")
            digits += len(re.findall(r"\d+", text))
            latin += len(re.findall(r"[A-Za-z]+", text))
        out[name] = {"digit_runs_left": digits, "latin_words_left": latin}
    return out


def cmd_report(args) -> None:
    import jiwer

    sys.path.insert(0, str(REPO / "src"))
    from businessflow.audio.verbalizer import verbalize

    manifest = json.loads((AUDIO_DIR / "manifest.json").read_text(encoding="utf-8"))
    clips = manifest["clips"]
    transcripts = json.loads((AUDIO_DIR / "transcripts.json").read_text(encoding="utf-8")) if (AUDIO_DIR / "transcripts.json").exists() else {}
    memory = json.loads((RESULTS / "tts_benchmark_memory.json").read_text(encoding="utf-8")) if (RESULTS / "tts_benchmark_memory.json").exists() else {}

    for c in clips:  # clarity per clip
        t = transcripts.get(c.get("path", ""))
        if c["ok"] and t is not None and c["category"] in CLARITY_CATEGORIES:
            hyp = norm_hi(verbalize(t, "hi"))
            c["heard"], c["hyp"] = t, hyp
            c["cer"] = jiwer.cer(c["reference"], hyp)
            c["wer"] = jiwer.wer(c["reference"], hyp)

    summary: dict = {"baseline_rev": manifest["baseline_rev"], "engines": {}}
    per_sentence_cer: dict[str, dict[str, float]] = {}
    for engine in ENGINES:
        rows = [c for c in clips if c["engine"] == engine]
        if not rows:
            continue
        ok_rows = [c for c in rows if c["ok"]]
        e: dict = {"clips": len(rows), "failed": sum(1 for c in rows if not c["ok"] and not c.get("empty")),
                   "empty": sum(1 for c in rows if c.get("empty")), "errors": sorted({c["error"] for c in rows if c.get("error")})[:5],
                   "load_s": manifest["load_s"].get(engine)}
        by_cat: dict = {}
        for cat in (*CLARITY_CATEGORIES, "hinglish_roman"):
            cat_rows = [c for c in rows if c["category"] == cat]
            if not cat_rows:
                continue
            entry = {"clips": len(cat_rows), "ok": sum(c["ok"] for c in cat_rows)}
            scored = [c for c in cat_rows if "cer" in c and c["rep"] == 0]
            if scored:
                entry["cer"] = dict(zip(("mean", "lo", "hi"), (round(x, 4) for x in bootstrap_ci([c["cer"] for c in scored]))))
                entry["wer"] = dict(zip(("mean", "lo", "hi"), (round(x, 4) for x in bootstrap_ci([c["wer"] for c in scored]))))
            by_cat[cat] = entry
        e["by_category"] = by_cat
        pooled = [c for c in ok_rows if "cer" in c and c["rep"] == 0 and c["category"] in CLARITY_CATEGORIES]
        if pooled:
            e["pooled_cer"] = dict(zip(("mean", "lo", "hi"), (round(x, 4) for x in bootstrap_ci([c["cer"] for c in pooled]))))
            e["pooled_wer"] = dict(zip(("mean", "lo", "hi"), (round(x, 4) for x in bootstrap_ci([c["wer"] for c in pooled]))))
            per_sentence_cer[engine] = {c["id"]: c["cer"] for c in pooled}
            per_sentence_cer[engine + "|plain"] = {c["id"]: c["cer"] for c in pooled if c["category"] == "hi_plain"}
        # consistency across repeats
        by_id: dict[str, list[dict]] = {}
        for c in ok_rows:
            by_id.setdefault(c["id"], []).append(c)
        cvs = [coefficient_of_variation([c["duration_s"] for c in g]) for g in by_id.values() if len(g) >= 2]
        same = [len({c.get("hyp") for c in g if c.get("hyp") is not None}) == 1 for g in by_id.values() if len(g) >= 2 and all("hyp" in c for c in g)]
        e["duration_cv_mean"] = round(statistics.fmean(cvs), 4) if cvs else None
        e["identical_transcript_share"] = round(sum(same) / len(same), 3) if same else None
        e["rms_dbfs_std"] = round(statistics.pstdev([c["rms_dbfs"] for c in ok_rows]), 2) if ok_rows else None
        e["clipped_clips"] = sum(c.get("clipped", False) for c in ok_rows)
        e["opus_over_clips"] = sum(c.get("opus_over", False) for c in ok_rows)
        e["lead_ms_mean"] = round(statistics.fmean([c["lead_ms"] for c in ok_rows]), 1) if ok_rows else None
        e["trail_ms_mean"] = round(statistics.fmean([c["trail_ms"] for c in ok_rows]), 1) if ok_rows else None
        e["rtf_mean"] = round(statistics.fmean([c["rtf"] for c in ok_rows if c.get("rtf")]), 3) if ok_rows else None
        e["sample_rates"] = sorted({c["sample_rate"] for c in ok_rows})
        e["memory"] = memory.get(engine)
        summary["engines"][engine] = e

    if "new_piper" in per_sentence_cer and "new_mms" in per_sentence_cer:
        d = paired_ci(per_sentence_cer["new_piper"], per_sentence_cer["new_mms"])
        summary["paired_piper_minus_mms"] = {"mean": round(d[0], 4), "lo": round(d[1], 4), "hi": round(d[2], 4), "piper_better": d[3], "tie": d[4], "piper_worse": d[5]}
    summary["leakage"] = _leakage(_sentences())

    # ---- pre-registered criteria
    en = summary["engines"]
    npz, nms, old = en.get("new_piper"), en.get("new_mms"), en.get("old_deployed")
    crit: list[tuple[str, bool | None, str]] = []
    if npz:
        crit.append(("C1 robustness", npz["failed"] == 0 and npz["empty"] == 0, f"{npz['failed']} failed, {npz['empty']} empty of {npz['clips']}"))
        if "paired_piper_minus_mms" in summary and npz.get("pooled_cer") and nms and nms.get("pooled_cer"):
            p = summary["paired_piper_minus_mms"]
            crit.append(("C2 clarity", npz["pooled_cer"]["mean"] <= nms["pooled_cer"]["mean"] and p["hi"] < 0.03,
                         f"CER piper {npz['pooled_cer']['mean']} vs mms {nms['pooled_cer']['mean']}; paired diff {p['mean']} [{p['lo']}, {p['hi']}]"))
        if nms and npz["duration_cv_mean"] is not None and nms["duration_cv_mean"] is not None:
            ok_cv = npz["duration_cv_mean"] <= 1.25 * nms["duration_cv_mean"]
            ok_same = (npz["identical_transcript_share"] is None or nms["identical_transcript_share"] is None
                       or npz["identical_transcript_share"] >= nms["identical_transcript_share"] - 0.10)
            crit.append(("C3 consistency", ok_cv and ok_same, f"duration CV piper {npz['duration_cv_mean']} vs mms {nms['duration_cv_mean']}; identical transcripts {npz['identical_transcript_share']} vs {nms['identical_transcript_share']}"))
        crit.append(("C4 audio hygiene", npz["clipped_clips"] == 0 and npz["opus_over_clips"] == 0 and (npz["rms_dbfs_std"] or 0) <= 3.0,
                     f"{npz['clipped_clips']} clipped, {npz['opus_over_clips']} above 1.0 after Opus; loudness std {npz['rms_dbfs_std']} dB"))
        if nms:
            mem_p = (npz.get("memory") or {}).get("growth_mb")
            mem_m = (nms.get("memory") or {}).get("growth_mb")
            ok_speed = npz["rtf_mean"] <= nms["rtf_mean"]
            ok_mem = None if mem_p is None or mem_m is None else mem_p <= mem_m
            crit.append(("C5 resources", ok_speed and (ok_mem is not False), f"RTF piper {npz['rtf_mean']} vs mms {nms['rtf_mean']}; memory growth piper {mem_p} MB vs mms {mem_m} MB"))
        lk = summary["leakage"]
        dropped_old = lk["old"]["digit_runs_left"] + lk["old"]["latin_words_left"]
        crit.append(("C6 leakage", lk["new"]["digit_runs_left"] == 0 and lk["new"]["digit_runs_left"] < dropped_old,
                     f"digit runs left new {lk['new']['digit_runs_left']} (old {lk['old']['digit_runs_left']}); Latin words left new {lk['new']['latin_words_left']} (old {lk['old']['latin_words_left']}, dropped by the old pipeline)"))
        if old and "new_piper|plain" in per_sentence_cer and "old_deployed|plain" in per_sentence_cer:
            a = statistics.fmean(per_sentence_cer["new_piper|plain"].values())
            b = statistics.fmean(per_sentence_cer["old_deployed|plain"].values())
            crit.append(("C7 regression (hi_plain)", a <= b, f"CER piper {a:.4f} vs old deployed {b:.4f}"))
    summary["criteria"] = [{"name": n, "passed": p, "detail": d} for n, p, d in crit]
    summary["all_passed"] = all(p for _, p, _ in crit) if crit and all(p is not None for _, p, _ in crit) else None

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tts_benchmark.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "tts_benchmark.md").write_text(_markdown(summary), encoding="utf-8")
    print(_markdown(summary))


def _markdown(s: dict) -> str:
    L = ["# Hindi TTS benchmark", "", f"Baseline (old_deployed) = `{s['baseline_rev']}`. Clarity = read-back by Whisper large-v3 against the intended spoken text (CER/WER, lower is better); "
         "it measures intelligibility, not naturalness or accent.", "", "## Clarity (character error rate, 95% bootstrap CI over sentences, first repeat)", "",
         "| engine | hi_plain | hi_numbers | hi_acronyms | hi_long | pooled |", "|---|---|---|---|---|---|"]
    def cell(d):
        return "n/a" if not d else f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    for name, e in s["engines"].items():
        bc = e["by_category"]
        L.append(f"| {name} | " + " | ".join(cell(bc.get(c, {}).get("cer")) for c in CLARITY_CATEGORIES) + f" | {cell(e.get('pooled_cer'))} |")
    if "paired_piper_minus_mms" in s:
        p = s["paired_piper_minus_mms"]
        L += ["", f"Paired per-sentence difference, new_piper minus new_mms: **{p['mean']}** [{p['lo']}, {p['hi']}] (Piper better on {p['piper_better']}, tied on {p['tie']}, worse on {p['piper_worse']} sentences)."]
    L += ["", "## Robustness and the Roman-script case", "", "| engine | clips | failed | empty | hinglish_roman ok | example error |", "|---|---:|---:|---:|---|---|"]
    for name, e in s["engines"].items():
        r = e["by_category"].get("hinglish_roman", {})
        L.append(f"| {name} | {e['clips']} | {e['failed']} | {e['empty']} | {r.get('ok', 'n/a')}/{r.get('clips', 'n/a')} | {(e['errors'] or [''])[0][:70]} |")
    L += ["", "## Consistency, audio hygiene, speed and memory", "", "| engine | duration CV | identical transcripts | loudness std dB | peak >= 0.999 | above 1.0 after Opus | lead / trail ms | RTF | sample rates | memory growth MB | load s |", "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|"]
    for name, e in s["engines"].items():
        mem = (e.get("memory") or {}).get("growth_mb", "n/a")
        L.append(f"| {name} | {e['duration_cv_mean']} | {e['identical_transcript_share']} | {e['rms_dbfs_std']} | {e['clipped_clips']} | {e.get('opus_over_clips', 'n/a')} | {e['lead_ms_mean']} / {e['trail_ms_mean']} | {e['rtf_mean']} | {e['sample_rates']} | {mem} | {e['load_s']} |")
    lk = s["leakage"]
    L += ["", "## Verbalizer leakage (text sent to the Hindi voice, Roman-script sentences excluded)", "",
          f"- old: {lk['old']['digit_runs_left']} digit runs and {lk['old']['latin_words_left']} Latin words left (the old pipeline silently dropped all of them)",
          f"- new: {lk['new']['digit_runs_left']} digit runs and {lk['new']['latin_words_left']} Latin words left (Latin words are now spoken by the English voice)", "",
          "## Pre-registered criteria", ""]
    for c in s["criteria"]:
        L.append(f"- **{c['name']}**: {'PASS' if c['passed'] else ('FAIL' if c['passed'] is False else 'not evaluable')}: {c['detail']}")
    L += ["", f"**All criteria passed: {s['all_passed']}**"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--engines", help="comma-separated subset (default all); existing clips of the others are kept")
    g.set_defaults(fn=cmd_generate)
    m = sub.add_parser("memory")
    m.add_argument("engine", choices=list(ENGINES))
    m.set_defaults(fn=cmd_memory)
    r = sub.add_parser("report")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
