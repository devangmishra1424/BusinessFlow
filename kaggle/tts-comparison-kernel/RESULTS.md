# TTS comparison (CPU, 2 threads; Whisper large-v3 round trip)

| lang | engine | WER | CER | STOI | PESQ | SI-SDR | RTF | load s | +MB | Hz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | piper_en_US-lessac (current) | 0.018 | 0.007 | 0.999 | 4.167 | 30.746 | 0.058 | 6.6 | 21 | 22050 |
| hi | kokoro_hi_hf_alpha | 0.25 | 0.055 | 0.998 | 3.919 | 29.617 | 0.533 | 9.3 | 708 | 24000 |
| hi | kokoro_hi_hf_beta | 0.193 | 0.04 | 0.997 | 4.001 | 25.96 | 0.48 | 2.4 | 69 | 24000 |
| hi | kokoro_hi_hm_omega | 0.22 | 0.055 | 0.998 | 3.813 | 28.427 | 0.488 | 2.3 | 14 | 24000 |
| hi | kokoro_hi_hm_psi | 0.193 | 0.049 | 0.997 | 3.778 | 28.402 | 0.482 | 2.2 | 14 | 24000 |
| hi | mms_hindi (current) | 0.306 | 0.108 | 0.995 | 3.744 | 29.015 | 0.295 | 28.0 | 418 | 16000 |
| hi | piper_hi_pratham | 0.26 | 0.065 | 0.999 | 4.007 | 28.014 | 0.057 | 7.2 | 0 | 22050 |
| hi | piper_hi_priyamvada | 0.265 | 0.071 | 0.997 | 3.986 | 28.885 | 0.056 | 11.9 | 0 | 22050 |
| hi | piper_hi_rohan | 0.26 | 0.07 | 0.999 | 4.09 | 28.0 | 0.091 | 6.0 | 0 | 22050 |

## Engines that failed

- **kokoro_en_af_heart**: `OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.`
- **kokoro_en_am_michael**: `OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.`