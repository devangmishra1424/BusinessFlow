"""The voice/text input-output shell: VAD -> ASR -> Agent -> Verbalizer ->
TTS, and the text-only shortcut that skips straight to the Agent. Any
channel (browser, WhatsApp, whatever comes later) calls into this rather
than wiring VAD/ASR/TTS itself.

The Agent call is now the real tool-calling loop (agent.loop) -- it can
actually check payment status, look up policy, propose restructuring, and
escalate, grounding its replies in real tool results instead of just
talking. Each call here is still a single, fresh turn (no memory across
calls yet) -- multi-turn session memory is the next layer to build on top.
"""

from dataclasses import dataclass

import torch

from businessflow.agent.loop import run_turn, start_conversation
from businessflow.audio.asr import transcribe
from businessflow.audio.tts import Speech, speak_english, speak_hindi
from businessflow.audio.vad import trim_to_speech
from businessflow.audio.verbalizer import verbalize


@dataclass
class RoundTripResult:
    transcript: str | None  # None for the text-only path, since there's nothing to transcribe
    reply_text: str
    speech: Speech


def _speak(text: str, language: str) -> Speech:
    return speak_hindi(text) if language == "hi" else speak_english(text)


def voice_roundtrip(
    audio: torch.Tensor, sampling_rate: int = 16000, language: str = "en", account_id: str | None = None
) -> RoundTripResult:
    """audio in, speech out. language picks the ASR hint and which TTS
    engine speaks the reply -- it does not yet auto-detect code-switching.
    account_id tells the agent which borrower it's speaking with, the same
    way a real deployment would identify the caller by phone number."""
    trimmed = trim_to_speech(audio, sampling_rate=sampling_rate)
    if trimmed.numel() == 0:
        raise ValueError("no speech detected in the given audio")

    transcript = transcribe(trimmed, language=language)
    conversation = start_conversation(language=language, account_id=account_id)
    conversation.append({"role": "user", "content": transcript})
    _conversation, reply_text = run_turn(conversation)

    speech = _speak(verbalize(reply_text, language), language)
    return RoundTripResult(transcript=transcript, reply_text=reply_text, speech=speech)


def text_roundtrip(user_message: str, language: str = "en", account_id: str | None = None) -> RoundTripResult:
    """text in, speech out -- skips VAD/ASR entirely."""
    conversation = start_conversation(language=language, account_id=account_id)
    conversation.append({"role": "user", "content": user_message})
    _conversation, reply_text = run_turn(conversation)

    speech = _speak(verbalize(reply_text, language), language)
    return RoundTripResult(transcript=None, reply_text=reply_text, speech=speech)
