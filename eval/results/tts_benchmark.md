# Hindi TTS benchmark

Baseline (old_deployed) = `6d5773d`. Clarity = read-back by Whisper large-v3 against the intended spoken text (CER/WER, lower is better); it measures intelligibility, not naturalness or accent.

## Clarity (character error rate, 95% bootstrap CI over sentences, first repeat)

| engine | hi_plain | hi_numbers | hi_acronyms | hi_long | pooled |
|---|---|---|---|---|---|
| new_piper | 0.070 [0.055, 0.086] | 0.055 [0.031, 0.082] | 0.106 [0.057, 0.171] | 0.063 [0.041, 0.079] | 0.073 [0.056, 0.094] |
| new_mms | 0.077 [0.053, 0.103] | 0.133 [0.088, 0.185] | 0.205 [0.124, 0.285] | 0.086 [0.056, 0.115] | 0.128 [0.098, 0.162] |
| old_deployed | 0.082 [0.047, 0.116] | 0.191 [0.118, 0.273] | 0.257 [0.201, 0.307] | 0.106 [0.077, 0.124] | 0.163 [0.128, 0.201] |

Paired per-sentence difference, new_piper minus new_mms: **-0.0547** [-0.0921, -0.0195] (Piper better on 22, tied on 3, worse on 7 sentences).

## Robustness and the Roman-script case

| engine | clips | failed | empty | hinglish_roman ok | example error |
|---|---:|---:|---:|---|---|
| new_piper | 120 | 0 | 0 | 24/24 |  |
| new_mms | 120 | 0 | 0 | 24/24 |  |
| old_deployed | 40 | 8 | 0 | 0/8 | RuntimeError: narrow(): length must be non-negative. |

## Consistency, audio hygiene, speed and memory

| engine | duration CV | identical transcripts | loudness std dB | peak >= 0.999 | above 1.0 after Opus | lead / trail ms | RTF | sample rates | memory growth MB | load s |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| new_piper | 0.0238 | 0.062 | 1.18 | 0 | 0 | 51.6 / 126.2 | 0.104 | [22050] | 138 | 6.6 |
| new_mms | 0.0458 | 0.031 | 0.76 | 0 | 0 | 248.0 / 331.5 | 0.741 | [16000, 22050] | 175 | 4.9 |
| old_deployed | None | None | 0.84 | 0 | 0 | 333.4 / 393.8 | 0.879 | [16000] | 175 | 6.4 |

## Verbalizer leakage (text sent to the Hindi voice, Roman-script sentences excluded)

- old: 5 digit runs and 8 Latin words left (the old pipeline silently dropped all of them)
- new: 0 digit runs and 6 Latin words left (Latin words are now spoken by the English voice)

## Pre-registered criteria

- **C1 robustness**: PASS: 0 failed, 0 empty of 120
- **C2 clarity**: PASS: CER piper 0.0731 vs mms 0.1278; paired diff -0.0547 [-0.0921, -0.0195]
- **C3 consistency**: PASS: duration CV piper 0.0238 vs mms 0.0458; identical transcripts 0.062 vs 0.031
- **C4 audio hygiene**: PASS: 0 clipped, 0 above 1.0 after Opus; loudness std 1.18 dB
- **C5 resources**: PASS: RTF piper 0.104 vs mms 0.741; memory growth piper 138 MB vs mms 175 MB
- **C6 leakage**: PASS: digit runs left new 0 (old 5); Latin words left new 6 (old 8, dropped by the old pipeline)
- **C7 regression (hi_plain)**: PASS: CER piper 0.0695 vs old deployed 0.0818

**All criteria passed: True**
