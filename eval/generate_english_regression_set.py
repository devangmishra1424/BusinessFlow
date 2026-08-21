"""One-time generation of a small English regression test set, via Windows
SAPI TTS -- same technique as the project's other synthesized fixtures.
Ground truth is exact, since we author the text ourselves; the trade-off
is synthetic-voice audio, not real human speech. That's an honest
limitation for a lightweight "did fine-tuning make English worse" check,
not a substitute for a real published English benchmark.

Run: python -m eval.generate_english_regression_set
(requires Windows -- uses PowerShell's System.Speech)
"""

import subprocess
from pathlib import Path

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "english_regression"

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your appointment for tomorrow afternoon.",
    "The weather today is sunny with a slight chance of rain.",
    "Your account balance has been updated successfully.",
    "Can you send me the report by end of day.",
    "The train to the city center departs every fifteen minutes.",
    "I would like to schedule a meeting next week.",
    "The library closes at nine in the evening on weekdays.",
    "Our customer support team is available around the clock.",
    "The new policy will take effect starting next month.",
]

_POWERSHELL_TEMPLATE = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile("{wav_path}", $fmt)
$synth.Speak("{text}")
$synth.SetOutputToNull()
"""


def main():
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, sentence in enumerate(SENTENCES):
        wav_path = _FIXTURES_DIR / f"{i:02d}.wav"
        script = _POWERSHELL_TEMPLATE.format(wav_path=str(wav_path).replace("\\", "\\\\"), text=sentence.replace('"', ""))
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
        manifest.append({"wav": wav_path.name, "reference": sentence})
        print(f"generated {wav_path.name}: {sentence}")

    import json
    (_FIXTURES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {len(manifest)} utterances + manifest.json")


if __name__ == "__main__":
    main()
