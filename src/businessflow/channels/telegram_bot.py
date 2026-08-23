"""Telegram bot channel -- text AND voice notes, replying in the same
modality the user used (text in -> text out, voice in -> voice out). This
is the second borrower-facing channel (after browser_api.py's text-only
HTTP API), so it plugs into the exact same agent/loop.py entry points
(verify_and_start_conversation, start_conversation, run_turn_with_memory)
rather than re-deriving verification or multi-turn session handling.
pipeline.py's voice_roundtrip/text_roundtrip are NOT used here -- those
call agent.loop directly with a fresh, unverified, single-turn
conversation each time, which is wrong for a channel that needs account
verification and a persistent multi-turn session. This module instead
calls the same lower-level pieces pipeline.py composes (trim_to_speech,
transcribe, verbalize, speak_hindi/speak_english) itself, wired through
session-aware routing.

Long polling (Application.run_polling()), not a webhook: this project has
no public HTTPS hosting yet (see README's known gaps), and a webhook
needs one. Long polling needs nothing but outbound HTTPS to Telegram's
servers, which any dev machine already has.

Session state (_sessions, keyed by Telegram chat_id) is held in-process,
in memory -- the same tradeoff browser_api.py's _conversations makes:
restarting this process loses every conversation currently in flight.
Cross-session memory (the recap a returning borrower's NEXT conversation
gets) is unaffected, since that's persisted to Postgres via
memory/conversation_memory.py independently of this in-memory state.

Voice notes arrive from Telegram as raw OGG/Opus bytes. Rather than add
ffmpeg (not on this machine's PATH, confirmed via `where ffmpeg`) or
pydub (which would silently need ffmpeg for anything beyond raw WAV),
this module decodes and re-encodes OGG/Opus directly via soundfile --
the same principle audio/io.py already follows for WAV (soundfile over
torchaudio's file-decode backend, to avoid a codec dependency), just
extended to the one extra container/codec pair this channel needs.
Telegram voice notes are commonly 48kHz; since Telegram (not the
borrower) controls that rate, this module resamples to 16kHz via
torchaudio.functional.resample (a pure tensor op -- no codec/ffmpeg
backend involved) before VAD/ASR, rather than raising the way
audio/io.py's load_wav_as_tensor deliberately does for file loads, where
a rate mismatch signals an actual problem worth catching, not a format
Telegram itself dictates.

Run: python -m businessflow.channels.telegram_bot
(needs TELEGRAM_BOT_TOKEN in .env -- get one from @BotFather on Telegram)
"""

import io
import os
import re

import groq
import soundfile as sf
import torch
import torchaudio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from businessflow.agent.loop import (
    AccessDeniedError,
    AccountLockedError,
    run_turn_with_memory,
    start_conversation,
    verify_and_start_conversation,
)
from businessflow.audio.asr import transcribe
from businessflow.audio.tts import speak_english, speak_hindi
from businessflow.audio.vad import trim_to_speech
from businessflow.audio.verbalizer import verbalize

# Matches agent/client.py's convention: load .env at import time, then
# read os.environ directly wherever a value is needed (main(), below,
# still checks TELEGRAM_BOT_TOKEN explicitly and raises if it's missing).
load_dotenv()

_MAX_VOICE_NOTE_SECONDS = 120  # an explicit, bounded cap on ASR compute -- not an unbounded wait on a huge file

_sessions: dict[int, dict] = {}  # chat_id -> {"account_id": str | None, "language": str, "messages": list[dict]}
_language_choice: dict[int, str] = {}  # chat_id -> "en" | "hi", set via /hindi or /english before a session exists

_CREDENTIALS_PATTERN = re.compile(r"^(\S+)\s+(\d{6})$")


def _looks_like_credentials(text: str) -> bool:
    """True if text is shaped like "<account_id> <6-digit access key>"
    (e.g. "BF-1001 482913") -- shared by both the text and voice paths
    below to decide whether an unauthenticated message is a verification
    attempt."""
    return bool(_CREDENTIALS_PATTERN.match(text.strip()))


def handle_incoming_message(chat_id: int, text: str) -> str:
    """All routing logic for a plain-text turn, decoupled from
    python-telegram-bot's types so it's testable with plain (int, str)
    args. Mirrors browser_api.py's two-endpoint split (POST /conversations
    to verify+create, POST .../messages to talk) but collapsed into one
    call, since Telegram gives us one incoming message at a time rather
    than a client that calls two separate endpoints."""
    session = _sessions.get(chat_id)

    if session is None:
        if _looks_like_credentials(text):
            match = _CREDENTIALS_PATTERN.match(text.strip())
            account_id, access_key = match.group(1), match.group(2)
            language = _language_choice.get(chat_id, "en")
            try:
                conversation = verify_and_start_conversation(language, account_id, access_key)
            except AccessDeniedError:
                return f"That access key doesn't match account {account_id} -- please check and send both again."
            except AccountLockedError as e:
                return str(e)
            # Verification itself is not a turn -- run_turn_with_memory is
            # deliberately not called here, matching browser_api.py's
            # separate POST /conversations, which also doesn't run a turn.
            _sessions[chat_id] = {"account_id": account_id, "language": language, "messages": conversation}
            return f"Verified -- I've pulled up account {account_id}. What can I help you with?"

        # Anonymous/general chat: create the session, then treat this
        # message as the real first turn by falling through to the
        # shared existing-session logic below.
        language = _language_choice.get(chat_id, "en")
        conversation = start_conversation(language, account_id=None)
        _sessions[chat_id] = {"account_id": None, "language": language, "messages": conversation}
        session = _sessions[chat_id]

    session["messages"].append({"role": "user", "content": text})
    try:
        updated_conversation, reply = run_turn_with_memory(session["messages"], session["account_id"])
    except groq.RateLimitError as e:
        session["messages"].pop()  # don't leave a user message with no reply appended
        return f"I'm getting rate-limited by the LLM provider right now -- please try again shortly. ({e.message})"
    except groq.APIStatusError as e:
        session["messages"].pop()
        return f"The LLM provider had an error on its end -- please try again. ({e.message})"
    session["messages"] = updated_conversation
    return reply


def _decode_and_transcribe_voice_note(raw_bytes: bytes, language: str | None) -> str | None:
    """Given raw OGG/Opus bytes (as downloaded from a Telegram voice
    note), decodes via soundfile, collapses to mono, resamples to 16kHz
    if the native rate differs, trims to speech via VAD, and transcribes.
    Returns None if VAD found no speech at all -- callers must not feed
    that to ASR, per trim_to_speech's own contract -- or if the actual
    decoded audio exceeds _MAX_VOICE_NOTE_SECONDS, regardless of what
    duration the caller was told to expect (see _MAX_VOICE_NOTE_SECONDS).

    Takes plain bytes rather than a telegram.File, so this needs no
    network and no python-telegram-bot types -- the one part of the
    voice path a unit test can exercise directly, independent of a
    running bot."""
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # collapse to mono, same convention as audio/io.py

    # Re-check against the real decoded length, not just the caller-supplied
    # duration_seconds (Telegram's voice.duration metadata) -- a spoofed or
    # malformed container could under-report its length and otherwise run
    # full VAD/ASR on an unbounded amount of audio, bypassing the one cap
    # this pipeline relies on to bound compute.
    actual_duration_seconds = len(data) / sr
    if actual_duration_seconds > _MAX_VOICE_NOTE_SECONDS:
        return None

    audio = torch.from_numpy(data)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)

    trimmed = trim_to_speech(audio, sampling_rate=16000)
    if trimmed.numel() == 0:
        return None
    return transcribe(trimmed, language=language)


async def handle_incoming_voice(
    chat_id: int,
    telegram_file,
    duration_seconds: int,
    language_hint: str | None,
) -> tuple[str | None, str]:
    """Downloads a Telegram voice note, transcribes it, and routes the
    transcript through handle_incoming_message -- except when the
    transcript looks like account credentials and there's no session yet
    (see the security note below), in which case handle_incoming_message
    is deliberately never called.

    telegram_file needs only an async download_as_bytearray() method
    (telegram.File satisfies this, so does a plain test double) -- kept
    separate from _decode_and_transcribe_voice_note so the network-free
    part of this path stays independently testable.

    Returns (transcript, reply_text). transcript is None whenever there's
    no real reply to speak back as voice -- the note was rejected for
    length, or VAD found no speech -- so callers should send reply_text
    as plain text and skip TTS in that case.
    """
    if duration_seconds > _MAX_VOICE_NOTE_SECONDS:
        return None, (
            f"That voice note is too long ({duration_seconds}s) -- "
            f"please keep it under {_MAX_VOICE_NOTE_SECONDS} seconds."
        )

    raw_bytes = bytes(await telegram_file.download_as_bytearray())
    transcript = _decode_and_transcribe_voice_note(raw_bytes, language_hint)
    if not transcript or not transcript.strip():
        return None, "Sorry, I couldn't hear anything in that voice note -- please try again."

    # SECURITY: an account_id + 6-digit access key spoken aloud and
    # misheard by ASR (even one digit) would burn one of the account's
    # limited AccountLockedError attempts through no fault of the user's.
    # So a credential-shaped transcript from a brand-new session is never
    # sent into verification -- ask for it as text instead, and don't
    # call handle_incoming_message at all (no failed-attempt side effect).
    if chat_id not in _sessions and _looks_like_credentials(transcript):
        return transcript, (
            "That sounded like your account ID and access key -- to avoid a "
            'misheard digit locking your account, please TYPE them instead, '
            'e.g. "BF-1001 482913".'
        )

    reply = handle_incoming_message(chat_id, transcript)
    return transcript, reply


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        'Send your account ID and 6-digit access key together as TEXT (e.g. "BF-1001 482913") '
        "to discuss your loan, or just start talking -- text or voice -- for a general question."
    )


async def on_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    _sessions.pop(chat_id, None)
    _language_choice.pop(chat_id, None)
    await update.message.reply_text("Conversation reset.")


async def on_hindi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _language_choice[update.effective_chat.id] = "hi"
    await update.message.reply_text("Hindi set for this conversation.")


async def on_english(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _language_choice[update.effective_chat.id] = "en"
    await update.message.reply_text("English set for this conversation.")


async def on_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    reply = handle_incoming_message(chat_id, update.message.text)
    await update.message.reply_text(reply)


def _transcript_echo(transcript: str) -> str:
    """The "You said: ..." confirmation text sent back before speaking the
    reply. SECURITY: never echoes a credential-shaped transcript verbatim
    into the chat transcript -- this applies regardless of whether chat_id
    already has a session (handle_incoming_voice's own credential-redirect
    only covers the no-session case), since the same echo would otherwise
    put the account_id + 6-digit access key into Telegram's chat history
    either way. Factored out from on_voice_message so this specific
    redaction behavior is directly testable without a fake Update."""
    if _looks_like_credentials(transcript):
        return "You said: [account details -- redacted]"
    return f"You said: {transcript}"


async def on_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    voice = update.message.voice
    telegram_file = await voice.get_file()
    language_hint = _sessions.get(chat_id, {}).get("language") or _language_choice.get(chat_id, "en")

    transcript, reply_text = await handle_incoming_voice(chat_id, telegram_file, voice.duration, language_hint)

    if transcript is None:
        # Too-long rejection or VAD found no speech -- nothing to speak back as voice.
        await update.message.reply_text(reply_text)
        return

    await update.message.reply_text(_transcript_echo(transcript))

    language = _sessions.get(chat_id, {}).get("language") or _language_choice.get(chat_id, "en")
    speech = speak_hindi(verbalize(reply_text)) if language == "hi" else speak_english(verbalize(reply_text))
    buf = io.BytesIO()
    sf.write(buf, speech.audio.numpy(), speech.sample_rate, format="OGG", subtype="OPUS")
    await update.message.reply_voice(voice=buf.getvalue())


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set -- copy .env.example to .env and fill it in")

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", on_start))
    application.add_handler(CommandHandler("reset", on_reset))
    application.add_handler(CommandHandler("hindi", on_hindi))
    application.add_handler(CommandHandler("english", on_english))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_message))
    application.add_handler(MessageHandler(filters.VOICE, on_voice_message))

    application.run_polling()


if __name__ == "__main__":
    main()
