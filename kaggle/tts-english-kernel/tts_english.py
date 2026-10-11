"""English voices for BusinessFlow: is there a less mechanical one than Piper's en_US-lessac, that still runs on CPU?

Same method and limits as ../tts-comparison-kernel/tts_compare.py: every TTS engine runs on CPU with 2 threads (so the
real-time factor means something for the 2-vCPU VM), Whisper large-v3 transcribes each clip (WER/CER = intelligibility,
NOT naturalness or accent: a person has to listen to the wav files). An engine that fails is recorded, not skipped.
"""

import gc, json, re, subprocess, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./out")
AUDIO = OUT / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)


def sh(cmd: str, required: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0 and required:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")


sh("pip install -q piper-tts==1.7.0 jiwer psutil soundfile huggingface_hub uv")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import jiwer  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

SENTENCES = [
    ("Hello, your next instalment is five thousand rupees, due on the fifteenth of October.", True),
    ("Your payment has gone through successfully. Thank you very much.", False),
    ("If you disagree with this amount, please tell me what happened and I will send it for review.", False),
    ("Your loan is fully repaid. An agent will contact you about the closure certificate.", False),
    ("A late fee applies after the grace period, but nothing is charged within it.", False),
]
results = {"engines": {}, "failures": {}}
proc = psutil.Process()
torch.set_num_threads(2)


def clip_row(name, i, text, has_numbers, audio, sr, synth_s):
    d = AUDIO / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"en_{i}.wav"
    sf.write(str(path), audio.astype(np.float32), sr)
    dur = len(audio) / sr
    return {"lang": "en", "n": i, "text": text, "has_numbers": has_numbers, "duration_s": round(dur, 3), "synth_s": round(synth_s, 3),
            "rtf": round(synth_s / dur, 3) if dur else None, "sample_rate": sr, "path": str(path)}


def piper_engine(name: str, repo_dir: str, voice: str, quality: str) -> None:
    print(f"\n=== {name}", flush=True)
    try:
        from piper import PiperVoice
        files = [f for f in list_repo_files("rhasspy/piper-voices") if f.startswith(f"{repo_dir}/{voice}/{quality}/") and f.endswith(".onnx")]
        if not files:
            raise RuntimeError(f"no {repo_dir}/{voice}/{quality} .onnx in rhasspy/piper-voices")
        rss0, t0 = proc.memory_info().rss, time.time()
        path = hf_hub_download("rhasspy/piper-voices", files[0])
        hf_hub_download("rhasspy/piper-voices", files[0] + ".json")
        v = PiperVoice.load(path)
        load_s = time.time() - t0
        clips = []
        for i, (text, hn) in enumerate(SENTENCES):
            t1 = time.time()
            chunks = list(v.synthesize(text))
            audio = np.concatenate([c.audio_float_array for c in chunks])
            clips.append(clip_row(name, i, text, hn, audio, chunks[0].sample_rate, time.time() - t1))
        results["engines"][name] = {"load_s": round(load_s, 1), "rss_growth_mb": round((proc.memory_info().rss - rss0) / 1e6), "clips": clips}
    except Exception:
        results["failures"][name] = traceback.format_exc()
        print("FAILED", name, results["failures"][name], flush=True)
    gc.collect()


KOKORO_GEN = r"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np, psutil, soundfile as sf, torch
torch.set_num_threads(2)
spec = json.load(open(sys.argv[1])); audio_dir = Path(spec["audio_dir"]); proc = psutil.Process()
from kokoro import KPipeline
engines, failures = {}, {}
rss0 = proc.memory_info().rss; t0 = time.time()
try:
    try:
        pipe = KPipeline(lang_code=spec["lang_code"], repo_id="hexgrad/Kokoro-82M", device="cpu")
    except TypeError:
        pipe = KPipeline(lang_code=spec["lang_code"], repo_id="hexgrad/Kokoro-82M")
    load_s = time.time() - t0; growth = (proc.memory_info().rss - rss0) / 1e6
except Exception:
    failures["kokoro_load"] = traceback.format_exc(); print("FAILED", failures["kokoro_load"], flush=True)
    json.dump({"engines": engines, "failures": failures}, open(sys.argv[2], "w"), ensure_ascii=False); sys.exit(0)
for voice in spec["voices"]:
    name = "kokoro_en_" + voice
    try:
        list(pipe("ok", voice=voice))  # warm-up: fetches the voice file
        clips = []
        for i, (text, hn) in enumerate(spec["sentences"]):
            t1 = time.time()
            parts = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(text, voice=voice)]
            audio = np.concatenate(parts); synth_s = time.time() - t1
            d = audio_dir / name; d.mkdir(parents=True, exist_ok=True)
            path = d / f"en_{i}.wav"; sf.write(str(path), audio.astype(np.float32), 24000)
            dur = len(audio) / 24000
            clips.append({"lang": "en", "n": i, "text": text, "has_numbers": hn, "duration_s": round(dur, 3), "synth_s": round(synth_s, 3),
                          "rtf": round(synth_s / dur, 3) if dur else None, "sample_rate": 24000, "path": str(path)})
        engines[name] = {"load_s": round(load_s, 1), "rss_growth_mb": round(growth), "clips": clips}
        print("ok", name, flush=True)
    except Exception:
        failures[name] = traceback.format_exc(); print("FAILED", name, failures[name], flush=True)
json.dump({"engines": engines, "failures": failures}, open(sys.argv[2], "w"), ensure_ascii=False)
"""

# today's voice, for the same-run reference, then Piper alternatives
piper_engine("piper_en_US-lessac-medium (current)", "en/en_US", "lessac", "medium")
piper_engine("piper_en_US-ryan-high", "en/en_US", "ryan", "high")
piper_engine("piper_en_US-amy-medium", "en/en_US", "amy", "medium")
piper_engine("piper_en_GB-alba-medium", "en/en_GB", "alba", "medium")

# Kokoro (82M, Apache-2.0) in its own Python 3.12 environment (it needs <3.13; this kernel runs 3.13), with the spaCy
# English model its English text front end needs (its absence is what failed the first attempt)
kokoro_voices_all = sorted(f.split("/")[-1][:-3] for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/") and f.endswith(".pt"))
wanted = [v for v in ("af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "bf_emma", "bm_george") if v in kokoro_voices_all]
print("Kokoro English voices tried:", wanted, flush=True)
Path("/tmp/kokoro_gen.py").write_text(KOKORO_GEN, encoding="utf-8")
Path("/tmp/kokoro_spec.json").write_text(json.dumps({"audio_dir": str(AUDIO), "lang_code": "a", "voices": wanted, "sentences": SENTENCES}), encoding="utf-8")
sh("uv run --python 3.12 --no-project --with 'kokoro>=0.9.2' --with soundfile --with numpy --with psutil --with spacy "
   "--with 'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl' "
   "python /tmp/kokoro_gen.py /tmp/kokoro_spec.json /tmp/kokoro_out.json", required=False)
out = Path("/tmp/kokoro_out.json")
if out.exists():
    got = json.loads(out.read_text(encoding="utf-8"))
    results["engines"].update(got["engines"])
    results["failures"].update(got["failures"])
else:
    results["failures"]["kokoro"] = "the Kokoro environment failed before writing results (see the log above)"

if not results["engines"]:
    (OUT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    raise SystemExit("no engine produced audio: " + ", ".join(results["failures"]))

# Whisper round trip (GPU)
from transformers import pipeline  # noqa: E402

device = 0 if torch.cuda.is_available() else -1
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device, torch_dtype=torch.float16 if device == 0 else torch.float32)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.,!?;:\"'()\-]", " ", t.lower())).strip()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


for name, eng in results["engines"].items():
    for clip in eng["clips"]:
        audio, sr = sf.read(clip["path"], dtype="float32")
        heard = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs={"language": "english", "task": "transcribe"})["text"]
        clip["heard"] = heard.strip()
        clip["wer"] = round(jiwer.wer(norm(clip["text"]), norm(heard)), 3)
        clip["cer"] = round(jiwer.cer(norm(clip["text"]), norm(heard)), 3)
    plain = [c for c in eng["clips"] if not c["has_numbers"]]
    eng["summary"] = {"wer": mean([c["wer"] for c in plain]), "cer": mean([c["cer"] for c in plain]), "rtf_cpu": mean([c["rtf"] for c in eng["clips"]]),
                      "load_s": eng["load_s"], "rss_growth_mb": eng["rss_growth_mb"], "sample_rate": eng["clips"][0]["sample_rate"]}

md = ["# English voices (TTS on CPU, 2 threads; Whisper large-v3 round trip, number-free sentences)", "",
      "| engine | WER | CER | RTF | load s | +MB | Hz |", "|---|---:|---:|---:|---:|---:|---:|"]
for name, eng in results["engines"].items():
    s = eng["summary"]
    md.append(f"| {name} | {s['wer']} | {s['cer']} | {s['rtf_cpu']} | {s['load_s']} | {s['rss_growth_mb']} | {s['sample_rate']} |")
if results["failures"]:
    md += ["", "## Failed", ""] + [f"- **{k}**: `{v.strip().splitlines()[-1]}`" for k, v in results["failures"].items()]
(OUT / "SUMMARY.md").write_text("\n".join(md))
(OUT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
print("\n".join(md), flush=True)
