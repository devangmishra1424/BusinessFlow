"""Compare the text-to-speech engines BusinessFlow could use, on the replies it actually speaks.

Run as a Kaggle script kernel (internet on, GPU on only for the ASR round-trip; every TTS engine runs on
CPU so the real-time factor means something for the 2-vCPU VM). Writes to /kaggle/working:
  results.json   per-clip and per-engine numbers
  SUMMARY.md     the table
  audio/<engine>/<lang>_<n>.wav   so a person can listen

What it measures, and what it does NOT:
  - intelligibility: each clip is transcribed by Whisper large-v3 and compared with the text that was
    spoken (WER / CER). Whisper hearing the words is a proxy for a listener understanding them.
  - signal quality: torchaudio SQUIM objective (reference-free STOI / PESQ / SI-SDR estimates). Trained on
    English speech, so for Hindi it is a rough relative signal only.
  - speed and size: real-time factor on CPU, load time, resident-memory growth after loading.
  - It does NOT measure how natural or how Indian-accented a voice sounds. That needs a person listening;
    the wav files are there for that. No accent classifier is used (no Indian-English candidate here is
    ungated, so there is nothing for it to rank).

An engine that fails is a RESULT: its error is recorded and printed, not skipped silently. The run exits
non-zero if no engine produced any audio.
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./out")
AUDIO = OUT / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)


def sh(cmd: str, required: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0 and required:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    if r.returncode != 0:
        print(f"(optional step failed with {r.returncode}; continuing)", flush=True)


# Pinned to what the project itself runs (requirements.txt / the venv) so the numbers describe the real engines.
sh("pip install -q piper-tts==1.7.0 jiwer psutil soundfile huggingface_hub")
sh("pip install -q uv")  # Kokoro requires Python < 3.13 and this kernel runs 3.13: it gets its own 3.12 environment via uv
sh("apt-get install -y -q espeak-ng", required=False)  # Kokoro's Hindi phonemizer; kokoro also bundles a loader

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
import jiwer  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

# ---------------------------------------------------------------------------------------------
# The test set: replies of the kind the agent speaks. Sentence 1 of each language has numbers (Whisper
# writes digits where the text has words, which counts as an error for every engine alike), so the
# summary uses the number-free sentences; the numeric one is still synthesized and reported.
# ---------------------------------------------------------------------------------------------
SENTENCES = {
    "hi": [
        ("नमस्ते, आपकी अगली किस्त पाँच हज़ार रुपये की है और इसकी अंतिम तारीख पंद्रह अक्टूबर है।", True),
        ("आपका भुगतान सफलतापूर्वक हो गया है। आपका बहुत धन्यवाद।", False),
        ("अगर आपको इस रकम पर आपत्ति है, तो कृपया बताइए कि क्या हुआ, मैं इसे जाँच के लिए भेज दूँगा।", False),
        ("आपका ऋण पूरी तरह चुकाया जा चुका है। समापन प्रमाणपत्र के लिए एक एजेंट आपसे संपर्क करेगा।", False),
        ("देर से भुगतान करने पर विलंब शुल्क लगता है, लेकिन छूट अवधि के भीतर कोई शुल्क नहीं है।", False),
    ],
    "en": [
        ("Hello, your next instalment is five thousand rupees, due on the fifteenth of October.", True),
        ("Your payment has gone through successfully. Thank you very much.", False),
        ("If you disagree with this amount, please tell me what happened and I will send it for review.", False),
        ("Your loan is fully repaid. An agent will contact you about the closure certificate.", False),
        ("A late fee applies after the grace period, but nothing is charged within it.", False),
    ],
}

results: dict = {"engines": {}, "failures": {}, "sentences": SENTENCES}
proc = psutil.Process()


def save_wav(engine: str, lang: str, i: int, audio: np.ndarray, sr: int) -> str:
    d = AUDIO / engine
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{lang}_{i}.wav"
    sf.write(str(path), audio.astype(np.float32), sr)
    return str(path)


def run_engine(name: str, loader, speak, langs: tuple[str, ...]) -> None:
    """loader() -> handle; speak(handle, text, lang) -> (float32 mono audio, sample_rate)."""
    print(f"\n=== engine: {name}", flush=True)
    rss0 = proc.memory_info().rss
    t0 = time.time()
    try:
        handle = loader()
        load_s = time.time() - t0
        rss_growth_mb = (proc.memory_info().rss - rss0) / 1e6
        clips = []
        for lang in langs:
            for i, (text, has_numbers) in enumerate(SENTENCES[lang]):
                t1 = time.time()
                audio, sr = speak(handle, text, lang)
                synth_s = time.time() - t1
                dur = len(audio) / sr
                clips.append({
                    "lang": lang, "n": i, "text": text, "has_numbers": has_numbers,
                    "duration_s": round(dur, 3), "synth_s": round(synth_s, 3), "rtf": round(synth_s / dur, 3) if dur else None,
                    "sample_rate": sr, "path": save_wav(name, lang, i, audio, sr),
                })
        results["engines"][name] = {"load_s": round(load_s, 1), "rss_growth_mb": round(rss_growth_mb), "clips": clips}
        print(f"    ok: {len(clips)} clips, load {load_s:.1f}s, +{rss_growth_mb:.0f} MB", flush=True)
        del handle
    except Exception:
        results["failures"][name] = traceback.format_exc()
        print(f"    FAILED: {name}\n{results['failures'][name]}", flush=True)
    gc.collect()


# ---------------------------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------------------------
def mms_loader():
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
    model = VitsModel.from_pretrained("facebook/mms-tts-hin").eval()
    return tok, model


def mms_speak(h, text, lang):
    tok, model = h
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        wav = model(**ids).waveform[0].numpy()
    return wav, model.config.sampling_rate


def piper_loader(repo_dir: str, name: str):
    from piper import PiperVoice
    files = [f for f in list_repo_files("rhasspy/piper-voices") if f.startswith(f"{repo_dir}/{name}/")]
    onnx = [f for f in files if f.endswith(".onnx") and "medium" in f] or [f for f in files if f.endswith(".onnx")]
    if not onnx:
        raise RuntimeError(f"no .onnx for {repo_dir}/{name} in rhasspy/piper-voices (found {files})")
    model_path = hf_hub_download("rhasspy/piper-voices", onnx[0])
    hf_hub_download("rhasspy/piper-voices", onnx[0] + ".json")
    return PiperVoice.load(model_path)


def piper_speak(voice, text, lang):
    chunks = list(voice.synthesize(text))
    return np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate


KOKORO_GEN = r"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np, psutil, soundfile as sf, torch
torch.set_num_threads(2)
spec = json.load(open(sys.argv[1]))
audio_dir = Path(spec["audio_dir"]); proc = psutil.Process()
from kokoro import KPipeline
engines, failures, pipes = {}, {}, {}
for job in spec["jobs"]:
    name = job["engine"]
    try:
        rss0 = proc.memory_info().rss; t0 = time.time()
        if job["lang_code"] not in pipes:
            try:
                pipes[job["lang_code"]] = KPipeline(lang_code=job["lang_code"], repo_id="hexgrad/Kokoro-82M", device="cpu")
            except TypeError:
                pipes[job["lang_code"]] = KPipeline(lang_code=job["lang_code"], repo_id="hexgrad/Kokoro-82M")
        pipe = pipes[job["lang_code"]]
        list(pipe("ok", voice=job["voice"]))  # warm-up: fetches this voice file, so the timed clips do not include a download
        load_s = time.time() - t0; growth = (proc.memory_info().rss - rss0) / 1e6
        clips = []
        for i, (text, has_numbers) in enumerate(job["sentences"]):
            t1 = time.time()
            parts = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(text, voice=job["voice"])]
            audio = np.concatenate(parts); synth_s = time.time() - t1
            d = audio_dir / name; d.mkdir(parents=True, exist_ok=True)
            path = d / f"{job['lang']}_{i}.wav"; sf.write(str(path), audio.astype(np.float32), 24000)
            dur = len(audio) / 24000
            clips.append({"lang": job["lang"], "n": i, "text": text, "has_numbers": has_numbers, "duration_s": round(dur, 3),
                          "synth_s": round(synth_s, 3), "rtf": round(synth_s / dur, 3) if dur else None, "sample_rate": 24000, "path": str(path)})
        engines[name] = {"load_s": round(load_s, 1), "rss_growth_mb": round(growth), "clips": clips}
        print("ok", name, flush=True)
    except Exception:
        failures[name] = traceback.format_exc(); print("FAILED", name, failures[name], flush=True)
json.dump({"engines": engines, "failures": failures}, open(sys.argv[2], "w"), ensure_ascii=False)
"""


def run_kokoro(jobs: list[dict]) -> None:
    """Kokoro runs in a separate Python 3.12 environment (see the pip/uv note above); its clips are merged in."""
    Path("/tmp/kokoro_gen.py").write_text(KOKORO_GEN, encoding="utf-8")
    Path("/tmp/kokoro_spec.json").write_text(json.dumps({"audio_dir": str(AUDIO), "jobs": jobs}, ensure_ascii=False), encoding="utf-8")
    sh("uv run --python 3.12 --no-project --with 'kokoro>=0.9.2' --with soundfile --with numpy --with psutil "
       "python /tmp/kokoro_gen.py /tmp/kokoro_spec.json /tmp/kokoro_out.json", required=False)
    out = Path("/tmp/kokoro_out.json")
    if not out.exists():
        for j in jobs:
            results["failures"][j["engine"]] = "Kokoro environment could not be created or crashed before writing results (see the log above)"
        return
    got = json.loads(out.read_text(encoding="utf-8"))
    results["engines"].update(got["engines"])
    results["failures"].update(got["failures"])


INDIC_GEN = r"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np, soundfile as sf, torch
spec = json.load(open(sys.argv[1])); audio_dir = Path(spec["audio_dir"])
token = os.environ["HF_TOKEN"]  # never printed
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"
engines, failures = {}, {}
try:
    t0 = time.time()
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts", token=token).to(device)
    tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", token=token)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    load_s = time.time() - t0
    sr = model.config.sampling_rate
except Exception:
    failures["indic_parler"] = traceback.format_exc(); print("FAILED to load", failures["indic_parler"], flush=True)
    json.dump({"engines": engines, "failures": failures}, open(sys.argv[2], "w"), ensure_ascii=False); sys.exit(0)

def speak(text, description):
    d = desc_tok(description, return_tensors="pt").to(device)
    p = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        g = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
    return g.cpu().numpy().squeeze().astype(np.float32)

speak("ok", spec["jobs"][0]["description"])  # warm-up
for job in spec["jobs"]:
    name = job["engine"]
    try:
        clips = []
        for i, (text, has_numbers) in enumerate(job["sentences"]):
            t1 = time.time(); audio = speak(text, job["description"])
            if device.startswith("cuda"): torch.cuda.synchronize()
            synth_s = time.time() - t1; dur = len(audio) / sr
            d = audio_dir / name; d.mkdir(parents=True, exist_ok=True)
            path = d / f"{job['lang']}_{i}.wav"; sf.write(str(path), audio, sr)
            clips.append({"lang": job["lang"], "n": i, "text": text, "has_numbers": has_numbers, "duration_s": round(dur, 3),
                          "synth_s": round(synth_s, 3), "rtf": round(synth_s / dur, 3) if dur else None, "sample_rate": sr, "path": str(path)})
        engines[name] = {"load_s": round(load_s, 1), "rss_growth_mb": None, "device": "gpu" if device.startswith("cuda") else "cpu", "clips": clips}
        print("ok", name, flush=True)
    except Exception:
        failures[name] = traceback.format_exc(); print("FAILED", name, failures[name], flush=True)
json.dump({"engines": engines, "failures": failures}, open(sys.argv[2], "w"), ensure_ascii=False)
"""

ACCENT_ID = r"""
import json, sys, traceback
import numpy as np, soundfile as sf, torch, torchaudio
items = json.load(open(sys.argv[1]))
out = {}
try:
    from speechbrain.inference.interfaces import foreign_class
    clf = foreign_class(source="Jzuluaga/accent-id-commonaccent_xlsr-en-english", pymodule_file="custom_interface.py",
                        classname="CustomEncoderWav2vec2Classifier", savedir="/tmp/accent_id")
    ind2lab = clf.hparams.label_encoder.ind2lab
    idx = {v: k for k, v in ind2lab.items()}
    for it in items:
        audio, sr = sf.read(it["path"], dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        wav = torchaudio.functional.resample(torch.from_numpy(audio)[None], sr, 16000)
        sf.write("/tmp/_accent_clip.wav", wav[0].numpy(), 16000)
        out_prob, score, index, text_lab = clf.classify_file("/tmp/_accent_clip.wav")
        lp = out_prob.squeeze()
        probs = lp.exp() if abs(float(lp.exp().sum()) - 1.0) < 1e-2 else torch.softmax(lp, dim=-1)
        top = torch.topk(probs, 3)
        out[it["path"]] = {"top": [(ind2lab[int(i)], round(float(v), 3)) for v, i in zip(top.values, top.indices)],
                           "p_indian": round(float(probs[idx["indian"]]), 3), "p_us": round(float(probs[idx["us"]]), 3)}
except Exception:
    out["_error"] = traceback.format_exc()
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False)
"""


def run_indic_parler(jobs: list[dict]) -> None:
    """Indic Parler-TTS is gated on Hugging Face: it needs the account owner's token, kept as a Kaggle secret named HF_TOKEN."""
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:  # a missing secret (or not running on Kaggle) is a recorded result, not a crash
        for j in jobs:
            results["failures"][j["engine"]] = f"HF_TOKEN secret unavailable ({type(e).__name__}): add it under Add-ons > Secrets and attach it to this notebook"
        return
    Path("/tmp/indic_gen.py").write_text(INDIC_GEN, encoding="utf-8")
    Path("/tmp/indic_spec.json").write_text(json.dumps({"audio_dir": str(AUDIO), "jobs": jobs}, ensure_ascii=False), encoding="utf-8")
    sh("uv run --python 3.12 --no-project --with 'parler-tts @ git+https://github.com/huggingface/parler-tts.git' --with soundfile --with numpy "
       "python /tmp/indic_gen.py /tmp/indic_spec.json /tmp/indic_out.json", required=False)
    out = Path("/tmp/indic_out.json")
    if not out.exists():
        for j in jobs:
            results["failures"][j["engine"]] = "Indic Parler environment could not be created or crashed before writing results (see the log above)"
        return
    got = json.loads(out.read_text(encoding="utf-8"))
    results["engines"].update(got["engines"])
    results["failures"].update(got["failures"])


def run_accent_id() -> dict:
    items = [{"path": c["path"]} for e in results["engines"].values() for c in e["clips"] if c["lang"] == "en"]
    if not items:
        return {}
    Path("/tmp/accent_id.py").write_text(ACCENT_ID, encoding="utf-8")
    Path("/tmp/accent_in.json").write_text(json.dumps(items), encoding="utf-8")
    sh("uv run --python 3.12 --no-project --with speechbrain --with torch --with torchaudio --with soundfile --with numpy "
       "python /tmp/accent_id.py /tmp/accent_in.json /tmp/accent_out.json", required=False)
    out = Path("/tmp/accent_out.json")
    return json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"_error": "accent-id environment failed (see the log above)"}


torch.set_num_threads(2)  # the VM has 2 vCPUs: time the engines the way the VM would run them

# the engine the bot uses today
run_engine("mms_hindi (current)", mms_loader, mms_speak, ("hi",))
run_engine("piper_en_US-lessac (current)", lambda: piper_loader("en/en_US", "lessac"), piper_speak, ("en",))

# Piper's own Hindi voices
hindi_voices = sorted({f.split("/")[2] for f in list_repo_files("rhasspy/piper-voices") if f.startswith("hi/hi_IN/")})
print("Piper hi_IN voices:", hindi_voices, flush=True)
for v in hindi_voices:
    run_engine(f"piper_hi_{v}", (lambda v=v: piper_loader("hi/hi_IN", v)), piper_speak, ("hi",))

# Kokoro (82M, Apache-2.0): its Hindi voices start with "h", its US-English ones with "a"
kokoro_files = [f for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/") and f.endswith(".pt")]
kokoro_voices = sorted(f.split("/")[-1][:-3] for f in kokoro_files)
print("Kokoro voices:", kokoro_voices, flush=True)
jobs = [{"engine": f"kokoro_hi_{v}", "lang_code": "h", "voice": v, "lang": "hi", "sentences": SENTENCES["hi"]} for v in kokoro_voices if v.startswith("h")]
if jobs:
    run_kokoro(jobs)

# Indic Parler-TTS (AI4Bharat, Apache-2.0, 0.9B): Indian-accented English and Hindi. Runs on the GPU here: it is far too heavy for the 2-vCPU VM.
_TAIL = " clear and friendly voice, at a moderate pace, with a very close recording that has no background noise."
indic_jobs = [
    {"engine": "indic_parler_hi_divya", "lang": "hi", "sentences": SENTENCES["hi"], "description": "Divya speaks in a" + _TAIL},
    {"engine": "indic_parler_hi_rohit", "lang": "hi", "sentences": SENTENCES["hi"], "description": "Rohit speaks in a" + _TAIL},
    {"engine": "indic_parler_en_mary", "lang": "en", "sentences": SENTENCES["en"], "description": "Mary speaks in a" + _TAIL},
    {"engine": "indic_parler_en_thoma", "lang": "en", "sentences": SENTENCES["en"], "description": "Thoma speaks in a" + _TAIL},
]
run_indic_parler(indic_jobs)

if not results["engines"]:
    (OUT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    raise SystemExit("no engine produced any audio: " + ", ".join(results["failures"]))

accent = run_accent_id()
if "_error" in accent:
    results["failures"]["accent_id"] = accent.pop("_error")

# ---------------------------------------------------------------------------------------------
# Scoring: Whisper round trip (GPU) and SQUIM (CPU)
# ---------------------------------------------------------------------------------------------
from transformers import pipeline  # noqa: E402

device = 0 if torch.cuda.is_available() else -1
print(f"\n=== scoring (whisper on {'GPU' if device == 0 else 'CPU'})", flush=True)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device,
               torch_dtype=torch.float16 if device == 0 else torch.float32)
squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()


def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[।.,!?;:\"'’()\-–—]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


for name, eng in results["engines"].items():
    for clip in eng["clips"]:
        audio, sr = sf.read(clip["path"], dtype="float32")
        out = asr({"raw": audio, "sampling_rate": sr},
                  generate_kwargs={"language": "hindi" if clip["lang"] == "hi" else "english", "task": "transcribe"})
        heard, ref = norm(out["text"]), norm(clip["text"])
        clip["heard"] = out["text"].strip()
        clip["wer"] = round(jiwer.wer(ref, heard), 3)
        clip["cer"] = round(jiwer.cer(ref, heard), 3)
        wav16 = torchaudio.functional.resample(torch.from_numpy(audio)[None], sr, 16000)
        with torch.no_grad():
            stoi, pesq, si_sdr = squim(wav16)
        clip["stoi"], clip["pesq"], clip["si_sdr"] = (round(float(x), 3) for x in (stoi, pesq, si_sdr))

# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------
def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


rows = []
for name, eng in results["engines"].items():
    plain = [c for c in eng["clips"] if not c["has_numbers"]]
    eng["summary"] = {
        "lang": eng["clips"][0]["lang"],
        "wer_number_free": mean([c["wer"] for c in plain]), "cer_number_free": mean([c["cer"] for c in plain]),
        "stoi": mean([c["stoi"] for c in eng["clips"]]), "pesq": mean([c["pesq"] for c in eng["clips"]]),
        "si_sdr": mean([c["si_sdr"] for c in eng["clips"]]), "rtf_cpu": mean([c["rtf"] for c in eng["clips"]]),
        "load_s": eng["load_s"], "rss_growth_mb": eng["rss_growth_mb"],
        "sample_rate": eng["clips"][0]["sample_rate"], "device": eng.get("device", "cpu"),
    }
    en = [accent[c["path"]] for c in eng["clips"] if c["path"] in accent]
    if en:
        eng["summary"]["accent_p_indian"] = mean([a["p_indian"] for a in en])
        eng["summary"]["accent_p_us"] = mean([a["p_us"] for a in en])
        tops = [a["top"][0][0] for a in en]
        eng["summary"]["accent_top"] = max(set(tops), key=tops.count)
    rows.append((eng["clips"][0]["lang"], name, eng["summary"]))

md = ["# TTS comparison (TTS on CPU with 2 threads unless the dev column says gpu; Whisper large-v3 round trip)", "",
      "| lang | engine | dev | WER | CER | STOI | PESQ | SI-SDR | RTF | load s | +MB | Hz | accent (top / P indian / P us) |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for lang, name, s in sorted(rows):
    acc = f"{s['accent_top']} / {s['accent_p_indian']} / {s['accent_p_us']}" if "accent_top" in s else ""
    md.append(f"| {lang} | {name} | {s['device']} | {s['wer_number_free']} | {s['cer_number_free']} | {s['stoi']} | {s['pesq']} | {s['si_sdr']} | {s['rtf_cpu']} | {s['load_s']} | {s['rss_growth_mb']} | {s['sample_rate']} | {acc} |")
if results["failures"]:
    md += ["", "## Engines that failed", ""] + [f"- **{k}**: `{v.strip().splitlines()[-1]}`" for k, v in results["failures"].items()]
(OUT / "SUMMARY.md").write_text("\n".join(md))
(OUT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
print("\n".join(md), flush=True)
