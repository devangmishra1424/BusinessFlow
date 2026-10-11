# Hindi TTS benchmark

Baseline (old_deployed) = `6d5773d`. Clarity = read-back by Whisper large-v3 against the intended spoken text (CER/WER, lower is better); it measures intelligibility, not naturalness or accent.

## Clarity (character error rate, 95% bootstrap CI over sentences, first repeat)

| engine | hi_plain | hi_numbers | hi_acronyms | hi_long | pooled |
|---|---|---|---|---|---|
| new_piper | 0.050 [0.028, 0.076] | 0.132 [0.046, 0.278] | 0.125 [0.057, 0.200] | 0.062 [0.035, 0.082] | 0.096 [0.057, 0.151] |
| new_mms | 0.063 [0.038, 0.088] | 0.161 [0.096, 0.245] | 0.237 [0.161, 0.320] | 0.095 [0.067, 0.135] | 0.141 [0.105, 0.184] |
| old_deployed | 0.077 [0.051, 0.101] | 0.185 [0.128, 0.241] | 0.231 [0.180, 0.278] | 0.108 [0.078, 0.138] | 0.153 [0.123, 0.185] |

Paired per-sentence difference, new_piper minus new_mms: **-0.0451** [-0.0815, -0.0085] (Piper better on 23, tied on 2, worse on 7 sentences).

## Robustness and the Roman-script case

| engine | clips | failed | empty | hinglish_roman ok | example error |
|---|---:|---:|---:|---|---|
| new_piper | 120 | 0 | 0 | 24/24 |  |
| new_mms | 120 | 0 | 0 | 24/24 |  |
| old_deployed | 40 | 8 | 0 | 0/8 | RuntimeError: narrow(): length must be non-negative. |

## Consistency, audio hygiene, speed and memory

| engine | duration CV | identical transcripts | loudness std dB | clipped | lead / trail ms | RTF | sample rates | memory growth MB | load s |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| new_piper | 0.0234 | 0.094 | 1.24 | 120 | 51.3 / 119.3 | 0.1 | [22050] | 138 | 6.4 |
| new_mms | 0.041 | 0.031 | 0.9 | 42 | 251.9 / 316.0 | 0.728 | [16000, 22050] | 175 | 4.9 |
| old_deployed | None | None | 0.82 | 0 | 335.3 / 340.0 | 0.983 | [16000] | 175 | 6.0 |

## Verbalizer leakage (text sent to the Hindi voice, Roman-script sentences excluded)

- old: 5 digit runs and 8 Latin words left (the old pipeline silently dropped all of them)
- new: 0 digit runs and 6 Latin words left (Latin words are now spoken by the English voice)

## Pre-registered criteria

- **C1 robustness**: PASS: 0 failed, 0 empty of 120
- **C2 clarity**: PASS: CER piper 0.0959 vs mms 0.141; paired diff -0.0451 [-0.0815, -0.0085]
- **C3 consistency**: PASS: duration CV piper 0.0234 vs mms 0.041; identical transcripts 0.094 vs 0.031
- **C4 audio hygiene**: FAIL: 120 clipped; loudness std 1.24 dB
- **C5 resources**: PASS: RTF piper 0.1 vs mms 0.728; memory growth piper 138 MB vs mms 175 MB
- **C6 leakage**: PASS: digit runs left new 0 (old 5); Latin words left new 6 (old 8, dropped by the old pipeline)
- **C7 regression (hi_plain)**: PASS: CER piper 0.0502 vs old deployed 0.0766

**All criteria passed: False**
