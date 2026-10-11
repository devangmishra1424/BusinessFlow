# Voice path: time per local stage

Read the header of `eval/voice_stage_latency.py`. **This is the dev machine, not the deployed VM, and it leaves out the language model call, the database and the network.**

Machine: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, 12 logical cores, torch threads 4, Python 3.12.10. Date: 2026-10-11.

| stage | median ms | 95th pct ms | real-time factor | audio seconds | runs |
|---|---|---|---|---|---|
| voice-activity trimming (Silero) | 37 | 38 | 0.007 | 5.07 | 8 |
| speech recognition (Whisper CTranslate2 int8, fine-tuned) | 845 | 874 | 0.197 | 4.28 | 8 |
| speech synthesis, English (Piper) | 315 | 340 | 0.05 | 6.31 | 8 |
| speech synthesis, Hindi (Piper hi_IN) | 394 | 416 | 0.048 | 8.18 | 8 |
| Opus encoding of the English reply | 91 | 92 | - | - | 8 |

Sum of the medians on the English path (trim + recognise + synthesise + encode): 1288 ms. Transcript of the clip: 'Your payment of 12,500 rupees is 3 days past 2'
