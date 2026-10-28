# Voice service

Speech-to-text (faster-whisper, int8, CPU) and text-to-speech (Piper, English and Hindi) behind one small HTTP API,
run on its own Azure VM (`voice-vm`, resource group `voice-rg`, Central India, 2 vCPU / 4 GB) so BusinessFlow and
VeriBook do not each hold the models in memory. Public at `https://voicevm.duckdns.org` (Caddy terminates TLS and
proxies to uvicorn on 127.0.0.1:8000). Every request needs an `X-API-Key` header; each app has its own key
(`VOICE_API_KEYS` in `/etc/voice.env` on the VM is a comma-separated list, so a key can be revoked on its own).

| Endpoint | Does |
|---|---|
| `GET /health` | models loaded, which Whisper model is live (`asr_model`) |
| `POST /v1/stt` | multipart `file` (any audio PyAV reads, up to 10 MB / 60 s), optional `language` (`en`/`hi`), optional `vad_filter`. Returns `text`, `language`, `audio_seconds`, `elapsed_s`, `avg_logprob`, `no_speech_prob` |
| `POST /v1/synth` | JSON `{text, voice: en\|hi}` (up to 2000 chars). Returns Piper's own output: `sample_rate` and `chunks`, one base64 float32 array per sentence. BusinessFlow joins, mixes and level-limits them itself |
| `POST /v1/tts` | JSON `{text, lang}`. Returns a finished WAV |

Status codes: 400 bad input, 401 bad key, 413 too large or too long, 422 nothing speakable, 429 busy (one job at a
time, 4 may wait), 503 no key configured.

Transcription uses the same decode settings as `businessflow/audio/asr.py` (`DECODE_KWARGS`, pinned by
`tests/test_voice_service.py`). `WHISPER_DIR` in `/etc/voice.env` picks the model: `/opt/voice/models/whisper-ft` is
BusinessFlow's fine-tuned Hindi-English model (copied from `data/models/businessflow-whisper-finetuned`, with the
base model's `tokenizer.json`, which the fine-tune folder lacks); delete the line to go back to `whisper-base`.

## Deploying a change

The VM has no inbound SSH (the network rule is opened only for a copy and closed again). Updates go through Azure
Run Command: fetch `app.py` from this repo pinned to a commit, check its sha256, install, restart, and check
`/health`. See the commands in the PR that added this directory.

## How BusinessFlow uses it

`VOICE_SERVICE_URL` and `VOICE_SERVICE_API_KEY` in BusinessFlow's `.env` (see `.env.example`). With them set,
`audio/asr.py` and `audio/tts.py` send the Whisper decode and the Piper synthesis here; everything else (silence
trimming, script mixing, sentence gaps, level limiting, Opus encoding) stays in BusinessFlow. If the service cannot
answer, BusinessFlow logs an ERROR saying why and uses its local models until the service is back.
