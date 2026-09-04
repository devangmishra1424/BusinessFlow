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

import asyncio
import io
import logging
import os

import groq
import soundfile as sf
import torch
import torchaudio
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from businessflow.accounts import store
from businessflow.agent.loop import (
    AccessDeniedError,
    AccountLockedError,
    run_turn_with_memory,
    start_conversation,
    start_conversation_with_recap,
    update_conversation_language,
    verify_and_start_conversation,
)
from businessflow.audio.asr import transcribe
from businessflow.audio.tts import encode_ogg_opus, speak_english, speak_hindi
from businessflow.audio.vad import trim_to_speech
from businessflow.audio.verbalizer import verbalize
from businessflow.channels.credentials import looks_like_credentials, parse_credentials, parse_telegram_start_payload
from businessflow.tools.account_tools import flag_dispute, get_payment_history, get_payment_status
from businessflow.tools.escalation_tools import escalate_to_human, request_closure_certificate
from businessflow.tools.payment_tools import generate_payment_link

# Matches agent/client.py's convention: load .env at import time, then
# read os.environ directly wherever a value is needed (main(), below,
# still checks TELEGRAM_BOT_TOKEN explicitly and raises if it's missing).
load_dotenv()

logger = logging.getLogger(__name__)

_MAX_VOICE_NOTE_SECONDS = 120  # an explicit, bounded cap on ASR compute -- not an unbounded wait on a huge file

_sessions: dict[int, dict] = {}  # chat_id -> {"account_id": str | None, "language": str, "messages": list[dict]}
_language_choice: dict[int, str] = {}  # chat_id -> "en" | "hi", set via /hindi or /english before a session exists
_voice_preference: dict[int, bool] = {}  # chat_id -> True if text replies should also be spoken, via /voice

# One lock per chat_id, serializing every call into handle_incoming_message
# for that chat. Without this, a voice turn (STT + LLM + TTS -- easily
# several seconds) still in flight when the user's next message (text or
# another voice note) arrives races on the same session["messages"] list:
# both turns read/append/reassign it concurrently in separate worker
# threads (see the to_thread calls below), and whichever finishes last
# silently overwrites the other's turn -- the real cause of a borrower's
# question appearing to vanish or get answered out of order. No await
# happens between the dict lookup and the assignment in
# _get_session_lock, so two coroutines can never race to create two
# different locks for the same brand-new chat_id.
_session_locks: dict[int, asyncio.Lock] = {}


def _get_session_lock(chat_id: int) -> asyncio.Lock:
    lock = _session_locks.get(chat_id)
    if lock is None:
        lock = _session_locks[chat_id] = asyncio.Lock()
    return lock


def handle_incoming_message(chat_id: int, text: str) -> str:
    """All routing logic for a plain-text turn, decoupled from
    python-telegram-bot's types so it's testable with plain (int, str)
    args. Mirrors browser_api.py's two-endpoint split (POST /conversations
    to verify+create, POST .../messages to talk) but collapsed into one
    call, since Telegram gives us one incoming message at a time rather
    than a client that calls two separate endpoints.

    Checks the credentials pattern whenever there's no VERIFIED account
    yet (session is None, or an anonymous session already exists) --  not
    just when session is None outright. Found live: a borrower who sent a
    non-credential message first (falling into anonymous chat) could never
    verify afterward, since the old code only ever checked the pattern
    before any session existed. A later "BF-1001 482913" just got forwarded
    to the LLM as plain text, which tried to use the whole string as a
    literal account_id and failed. Verifying from here on replaces the
    anonymous session outright rather than trying to splice its history
    into a freshly-authenticated one -- the anonymous system prompt never
    knew an account_id existed, so there's nothing worth preserving.
    Switching AWAY from an already-verified account still needs /reset,
    unchanged -- that's a different, deliberate tradeoff, not this bug."""
    session = _sessions.get(chat_id)
    welcome_back_prefix = ""  # only set (once) by the rehydration branch below

    # Found live: an already-verified borrower who (redundantly) re-sent
    # "BF-1007 892160" got it forwarded straight to the LLM as an ordinary
    # message below -- with no idea it was ever a credentials pair, the
    # model pattern-matched a bare 6-digit number in a financial
    # conversation and hallucinated a payment intent ("would you like to
    # make a payment of ₹892,160?"). Caught here, before it ever reaches
    # run_turn_with_memory, same as the real verification branch below.
    if session is not None and session.get("account_id") is not None and looks_like_credentials(text):
        return f"You're already verified as account {session['account_id']} -- what can I help you with?"

    if session is None or session.get("account_id") is None:
        if looks_like_credentials(text):
            account_id, access_key = parse_credentials(text)
            language = session["language"] if session else _language_choice.get(chat_id, "en")
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
            # Durable, unlike _sessions above -- so a decision made later on
            # the ops dashboard (approving/rejecting a restructuring
            # request) can reach this borrower even after this in-memory
            # session is long gone. Last-verified-chat-wins, same as the
            # column's comment in schema.sql.
            store.set_telegram_chat_id(account_id, chat_id)
            return f"Verified -- I've pulled up account {account_id}. What can I help you with?"

        if session is None:
            # A fresh process (a deploy restart, or -- found live -- an
            # OOM kill) means _sessions has nothing for this chat_id even
            # though it may have verified before the restart -- the
            # in-memory session that fact lived in is gone, but the
            # durable telegram_chat_id -> account_id mapping
            # (store.set_telegram_chat_id, written once at real
            # verification time) survives it. Rehydrating from that turns
            # "please verify again" into "welcome back" for exactly the
            # failure mode a real OOM kill caused live, mid-conversation,
            # without re-asking for the access key. Not a new trust
            # boundary -- see get_account_by_telegram_chat_id's own
            # docstring for why. Falls back to a genuinely anonymous
            # session, unchanged, if this chat_id was never verified.
            rehydrated = store.get_account_by_telegram_chat_id(chat_id)
            if rehydrated is not None:
                # NOT rehydrated.language_preference -- that account field
                # is a 3-way business preference ("hi"|"en"|"hinglish",
                # see accounts/models.py), but the live turn-based
                # conversation only ever supports "en"/"hi" runtime codes
                # (agent/client.py's own _LANGUAGE_INSTRUCTIONS has no
                # "hinglish" entry -- confirmed live via a real CI failure:
                # KeyError('hinglish') the first time this path actually
                # ran against a real "hinglish"-preference seeded account).
                # The real verification branch above never reads this
                # account field either, for the same reason -- match it.
                language = _language_choice.get(chat_id, "en")
                conversation = start_conversation_with_recap(language, rehydrated.account_id)
                _sessions[chat_id] = {"account_id": rehydrated.account_id, "language": language, "messages": conversation}
                welcome_back_prefix = f"Welcome back -- I've resumed account {rehydrated.account_id}.\n\n"
            else:
                language = _language_choice.get(chat_id, "en")
                conversation = start_conversation(language, account_id=None)
                _sessions[chat_id] = {"account_id": None, "language": language, "messages": conversation}
            session = _sessions[chat_id]
        # else: an anonymous session already exists and this message just
        # didn't look like credentials -- keep using it as-is below.

    session["messages"].append({"role": "user", "content": text})
    try:
        updated_conversation, reply = run_turn_with_memory(session["messages"], session["account_id"])
    except groq.RateLimitError as e:
        # Deliberately not including e.message in the reply -- it's Groq's raw
        # error body (org ID, exact token counts, an "upgrade to Dev
        # Tier" link), never meant for a borrower to see. Logged instead,
        # same as browser_api.py's equivalent path already does at the
        # HTTP layer (its frontend just never renders the raw detail).
        session["messages"].pop()  # don't leave a user message with no reply appended
        logger.warning("Groq rate limit hit for chat_id=%s: %s", chat_id, e)
        return "I'm getting rate-limited by the LLM provider right now -- please try again shortly."
    except groq.APIStatusError as e:
        session["messages"].pop()
        logger.warning("Groq API error for chat_id=%s: %s", chat_id, e)
        return "The LLM provider had an error on its end -- please try again."
    except groq.APIConnectionError as e:
        # A real gap found live: APIConnectionError (network drop, DNS
        # failure) and its subclass APITimeoutError are siblings of
        # APIStatusError, not subclasses of it -- neither except clause
        # above ever caught them. Left uncaught, this propagated all the
        # way past python-telegram-bot's own dispatcher (main() registers
        # no add_error_handler), so the borrower got total silence, and
        # the pop() cleanup never ran, stranding the just-appended user
        # message with no paired reply for every later turn to inherit.
        session["messages"].pop()
        logger.warning("Groq connection error for chat_id=%s: %s", chat_id, e)
        return "I'm having trouble reaching the LLM provider right now -- please try again shortly."
    session["messages"] = updated_conversation
    return welcome_back_prefix + reply


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
    if chat_id not in _sessions and looks_like_credentials(transcript):
        return transcript, (
            "That sounded like your account ID and access key -- to avoid a "
            'misheard digit locking your account, please TYPE them instead, '
            'e.g. "BF-1001 482913".'
        )

    # to_thread: handle_incoming_message ultimately calls agent.loop.run_turn,
    # which does asyncio.run(...) internally -- fine from a genuinely
    # synchronous caller, but this coroutine is already running on
    # python-telegram-bot's own event loop, and asyncio.run() refuses to
    # start a second one inside an already-running loop. Running the whole
    # synchronous call in a worker thread sidesteps that, without needing to
    # add an async version of the agent loop.
    #
    # The lock only wraps this call, not the transcription above -- two
    # voice notes' transcriptions can run concurrently (no shared state),
    # but only one turn per chat may ever mutate that chat's session at a
    # time (see _get_session_lock).
    async with _get_session_lock(chat_id):
        reply = await asyncio.to_thread(handle_incoming_message, chat_id, transcript)
    return transcript, reply


async def _handle_start_payload(chat_id: int, payload: str) -> str | None:
    """Verifies a real t.me/<bot>?start=<payload> deep-link tap (ops/api.py's
    telegram_invite_link, generated at account creation or a key reset) --
    reuses handle_incoming_message's own verification path wholesale (no
    separate logic to keep in sync), so a wrong/expired key fails exactly
    the same way typed credentials would, and its "Verified..." reply never
    echoes the key back into the chat either way.

    Returns the reply text once verification was actually attempted; None
    if payload isn't shaped like one this app itself produced, so on_start
    falls back to its generic help text instead of silently eating a plain
    /start with an unrelated argument. Factored out from on_start so this
    is directly testable without a fake Update/Context, same reasoning as
    _transcript_echo/_decode_and_transcribe_voice_note above."""
    parsed = parse_telegram_start_payload(payload)
    if parsed is None:
        return None
    account_id, access_key = parsed
    async with _get_session_lock(chat_id):
        return await asyncio.to_thread(handle_incoming_message, chat_id, f"{account_id} {access_key}")


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        reply = await _handle_start_payload(update.effective_chat.id, context.args[0])
        if reply is not None:
            await update.message.reply_text(reply)
            return

    await update.message.reply_text(
        'Send your account ID and 6-digit access key together as TEXT (e.g. "BF-1001 482913") '
        "to discuss your loan, or just start talking -- text or voice -- for a general question. "
        "Send /menu any time to see everything I can do."
    )


async def on_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    _sessions.pop(chat_id, None)
    _language_choice.pop(chat_id, None)
    await update.message.reply_text("Conversation reset.")


def _set_language(chat_id: int, language: str) -> None:
    # _language_choice alone only affects a FUTURE session (read once at
    # start_conversation/verify_and_start_conversation time). Found live
    # this was still genuinely broken even after previously updating
    # session["language"] here: that only affects OTHER things this file
    # reads it for (the voice-input language hint, which TTS engine a
    # spoken reply uses) -- the model itself only ever sees the language
    # instruction baked into conversation[0]'s system prompt, which this
    # never touched. So the bot would confirm "Hindi set for this
    # conversation" and then keep replying in whatever language the
    # conversation actually started in, while session["language"]
    # silently desynced from that and corrupted the voice-input/output
    # paths for a borrower still speaking/expecting the original
    # language. update_conversation_language actually rewrites
    # conversation[0] to match.
    _language_choice[chat_id] = language
    session = _sessions.get(chat_id)
    if session is not None:
        session["language"] = language
        update_conversation_language(session["messages"], language, session["account_id"])


async def on_hindi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _set_language(update.effective_chat.id, "hi")
    await update.message.reply_text("Hindi set for this conversation.")


async def on_english(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _set_language(update.effective_chat.id, "en")
    await update.message.reply_text("English set for this conversation.")


_LANGUAGE_KEYBOARD = InlineKeyboardMarkup(
    [[InlineKeyboardButton("English", callback_data="lang:en"), InlineKeyboardButton("हिन्दी", callback_data="lang:hi")]]
)


async def on_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Choose your language:", reply_markup=_LANGUAGE_KEYBOARD)


_NOT_VERIFIED_MESSAGE = (
    'This needs a verified account first -- send your account ID and 6-digit '
    'access key together as text (e.g. "BF-1001 482913").'
)


def _verified_account_id(chat_id: int) -> str | None:
    session = _sessions.get(chat_id)
    return session["account_id"] if session else None


def _log_tool_call(account_id: str, tool: str, arguments: dict, result: dict) -> None:
    # A slash command calls these tools directly, bypassing agent.loop's
    # LLM round trip entirely (no ambiguity to resolve for a deterministic
    # lookup/action) -- but it's just as real a tool call as one the LLM
    # makes, so it gets logged the same way, keeping the ops dashboard's
    # and conversation_memory's account history complete either way.
    store.log_event(account_id, "tool_called", {"tool": tool, "arguments": arguments, "result": result})


async def _run_status(chat_id: int) -> str:
    account_id = _verified_account_id(chat_id)
    if account_id is None:
        return _NOT_VERIFIED_MESSAGE
    result = get_payment_status(account_id)
    _log_tool_call(account_id, "get_payment_status", {"account_id": account_id}, result)
    lines = [
        f"{result['borrower_name']} -- account {result['account_id']}",
        f"EMI: ₹{result['emi_amount']:,.2f}, next due {result['emi_due_date']}",
        f"{result['months_remaining']} of {result['tenure_months']} months remaining",
        f"Outstanding (approx): ₹{result['outstanding_balance_approx']:,.2f}",
    ]
    if result["days_past_due"] > 0:
        due_line = f"{result['days_past_due']} days past due"
        if result["late_fee_applicable"]:
            due_line += f" -- a late fee of ₹{result['late_fee_amount']:,.2f} applies"
        lines.append(due_line)
    if result["dispute_open"]:
        lines.append("A dispute is currently open on this account.")
    return "\n".join(lines)


async def _run_history(chat_id: int) -> str:
    account_id = _verified_account_id(chat_id)
    if account_id is None:
        return _NOT_VERIFIED_MESSAGE
    result = get_payment_history(account_id)
    _log_tool_call(account_id, "get_payment_history", {"account_id": account_id}, result)
    records = result["payment_history"]
    if not records:
        return "No payment history on record yet."
    lines = ["Your recent payments:"]
    lines += [f"₹{r['amount']:,.2f} on {r['date']} -- {'on time' if r['on_time'] else 'late'}" for r in records]
    return "\n".join(lines)


async def _run_pay(chat_id: int, args: list[str] | None) -> str:
    account_id = _verified_account_id(chat_id)
    if account_id is None:
        return _NOT_VERIFIED_MESSAGE
    if not args:
        return "Usage: /pay <amount> -- e.g. /pay 5000"
    try:
        amount = float(args[0])
    except ValueError:
        return "That doesn't look like a number -- usage: /pay <amount>, e.g. /pay 5000"
    if amount <= 0:
        return "Amount must be greater than zero."
    result = generate_payment_link(account_id, amount)
    _log_tool_call(account_id, "generate_payment_link", {"account_id": account_id, "amount": amount}, result)
    return (
        f"Here's your payment link for ₹{amount:,.2f}:\n{result['payment_link']}\n\n"
        "(This is a demo link -- it doesn't move real money.)"
    )


async def _run_dispute(chat_id: int, args: list[str] | None) -> str:
    account_id = _verified_account_id(chat_id)
    if account_id is None:
        return _NOT_VERIFIED_MESSAGE
    if not args:
        return "Usage: /dispute <what happened> -- e.g. /dispute I already paid this via UPI on the 3rd"
    reason = " ".join(args)
    result = flag_dispute(account_id, reason)
    _log_tool_call(account_id, "flag_dispute", {"account_id": account_id, "reason": reason}, result)
    if result["already_open"]:
        return "You already have a dispute open on this account -- a human will review it."
    return (
        "Got it -- I've flagged this account for review. No further automated collection "
        "action will be taken until a human looks into it."
    )


async def _run_agent(chat_id: int, args: list[str] | None) -> str:
    account_id = _verified_account_id(chat_id)
    if account_id is None:
        return _NOT_VERIFIED_MESSAGE
    reason = " ".join(args) if args else "Borrower requested a human agent directly via /agent"
    result = escalate_to_human(account_id, reason)
    _log_tool_call(account_id, "escalate_to_human", {"account_id": account_id, "reason": reason}, result)
    return f"I've forwarded your request to a human agent (reference {result['escalation_id']}). They'll get back to you soon."


async def _run_closure(chat_id: int) -> str:
    account_id = _verified_account_id(chat_id)
    if account_id is None:
        return _NOT_VERIFIED_MESSAGE
    result = request_closure_certificate(account_id)
    _log_tool_call(account_id, "request_closure_certificate", {"account_id": account_id}, result)
    if result["eligible"]:
        return (
            f"Your loan shows as fully repaid. I've forwarded a request for your closure certificate "
            f"to a human (reference {result['escalation_id']}) -- they'll issue the actual document."
        )
    return (
        f"Your loan isn't fully repaid yet -- {result['months_remaining']} months remaining. "
        "A closure certificate can only be issued once the loan is fully paid off."
    )


async def on_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(await _run_status(update.effective_chat.id))


async def on_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(await _run_history(update.effective_chat.id))


async def on_pay(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(await _run_pay(update.effective_chat.id, context.args))


async def on_dispute(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(await _run_dispute(update.effective_chat.id, context.args))


async def on_agent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(await _run_agent(update.effective_chat.id, context.args))


async def on_closure(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(await _run_closure(update.effective_chat.id))


def _voice_toggle_text(chat_id: int) -> str:
    turning_on = not _voice_preference.get(chat_id, False)
    _voice_preference[chat_id] = turning_on
    return (
        "Voice replies are now ON -- I'll speak my replies as well as typing them."
        if turning_on
        else "Voice replies are now OFF."
    )


async def on_voice_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(_voice_toggle_text(update.effective_chat.id))


# (command, button label) -- also doubles as the /menu keyboard's contents.
# Not every borrower knows slash commands are a thing at all; tappable
# buttons are the accessible path in for exactly that person.
_MENU_ITEMS = [
    ("status", "📊 Check my status"),
    ("history", "📜 Payment history"),
    ("pay", "💳 Get a payment link"),
    ("dispute", "⚠️ Flag a dispute"),
    ("agent", "🧑 Talk to a human"),
    ("closure", "📄 Closure certificate"),
    ("language", "🌐 Change language"),
    ("voice", "🔊 Toggle voice replies"),
]


async def on_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=f"menu:{cmd}")] for cmd, label in _MENU_ITEMS])
    await update.message.reply_text("What would you like to do?", reply_markup=keyboard)


async def on_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    data = query.data

    if data.startswith("lang:"):
        language = data.split(":", 1)[1]
        _set_language(chat_id, language)
        await query.message.reply_text("Hindi set for this conversation." if language == "hi" else "English set for this conversation.")
        return

    if not data.startswith("menu:"):
        return
    cmd = data.split(":", 1)[1]
    # /pay and /dispute need a free-text argument a button tap can't
    # supply -- rather than build a whole "now reply with the amount"
    # follow-up flow, the button just shows the exact command to type,
    # same as a borrower who typed /pay with no argument would see.
    if cmd == "status":
        text = await _run_status(chat_id)
    elif cmd == "history":
        text = await _run_history(chat_id)
    elif cmd == "agent":
        text = await _run_agent(chat_id, None)
    elif cmd == "closure":
        text = await _run_closure(chat_id)
    elif cmd == "pay":
        text = "Send /pay <amount> to get a payment link -- e.g. /pay 5000"
    elif cmd == "dispute":
        text = "Send /dispute <what happened> -- e.g. /dispute I already paid this via UPI on the 3rd"
    elif cmd == "voice":
        text = _voice_toggle_text(chat_id)
    elif cmd == "language":
        await query.message.reply_text("Choose your language:", reply_markup=_LANGUAGE_KEYBOARD)
        return
    else:
        text = "Unknown option."
    await query.message.reply_text(text)


async def _send_spoken_reply(update: Update, chat_id: int, reply: str) -> None:
    # Best-effort: the text reply already succeeded and was already sent
    # by the caller -- a TTS failure here shouldn't take that back, just
    # skip the bonus spoken version. Broad except is deliberate (VAD/ASR/
    # TTS/network can all fail in ways worth treating identically here),
    # but never silent -- logged with the real exception either way.
    language = _sessions.get(chat_id, {}).get("language") or _language_choice.get(chat_id, "en")
    try:
        speech = speak_hindi(verbalize(reply, language)) if language == "hi" else speak_english(verbalize(reply, language))
        voice_bytes = encode_ogg_opus(speech)
    except Exception:
        logger.warning("Voice-reply synthesis failed for chat_id=%s", chat_id, exc_info=True)
        return
    await update.message.reply_voice(voice=voice_bytes)


async def _keep_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Telegram's own 'typing...' indicator only lasts ~5s per send, but a
    real turn takes 40-55s end to end (measured, see eval/results/
    latency_benchmark.json) -- found live that a borrower got total
    silence for that whole window, easily read mid-demo as the bot having
    hung or ignored the message. Resent every 4s (comfortably inside
    Telegram's own ~5s expiry) until the surrounding task is cancelled
    once the real reply is ready. Best-effort: a failure here should
    never take down the actual turn, just stop showing the indicator."""
    while True:
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            logger.warning("send_chat_action failed for chat_id=%s", chat_id, exc_info=True)
            return
        await asyncio.sleep(4)


async def on_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    typing_task = asyncio.create_task(_keep_typing(context, chat_id))
    try:
        # See handle_incoming_voice's comment on the same to_thread call --
        # handle_incoming_message eventually calls agent.loop.run_turn, which
        # asyncio.run()s internally, and that can't happen from inside a
        # coroutine already running on this event loop. The lock serializes
        # this against any voice turn for the same chat that might still be
        # in flight -- see _get_session_lock.
        async with _get_session_lock(chat_id):
            reply = await asyncio.to_thread(handle_incoming_message, chat_id, update.message.text)
    finally:
        typing_task.cancel()
    await update.message.reply_text(reply)
    if _voice_preference.get(chat_id):
        await _send_spoken_reply(update, chat_id, reply)


def _transcript_echo(transcript: str) -> str:
    """The "You said: ..." confirmation text sent back before speaking the
    reply. SECURITY: never echoes a credential-shaped transcript verbatim
    into the chat transcript -- this applies regardless of whether chat_id
    already has a session (handle_incoming_voice's own credential-redirect
    only covers the no-session case), since the same echo would otherwise
    put the account_id + 6-digit access key into Telegram's chat history
    either way. Factored out from on_voice_message so this specific
    redaction behavior is directly testable without a fake Update."""
    if looks_like_credentials(transcript):
        return "You said: [account details -- redacted]"
    return f"You said: {transcript}"


async def on_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    voice = update.message.voice
    telegram_file = await voice.get_file()
    language_hint = _sessions.get(chat_id, {}).get("language") or _language_choice.get(chat_id, "en")

    typing_task = asyncio.create_task(_keep_typing(context, chat_id))
    try:
        transcript, reply_text = await handle_incoming_voice(chat_id, telegram_file, voice.duration, language_hint)
    finally:
        typing_task.cancel()

    if transcript is None:
        # Too-long rejection or VAD found no speech -- nothing to speak back as voice.
        await update.message.reply_text(reply_text)
        return

    await update.message.reply_text(_transcript_echo(transcript))
    # Found live: the actual answer was only ever synthesized as speech
    # here, never sent as text -- a borrower who wanted to glance at (or
    # copy) a figure or link had to play an audio file for it, and a TTS
    # failure had no text fallback at all, unlike _send_spoken_reply's
    # already-existing best-effort handling on the text-input path.
    # Voice in still gets voice out (matches the modality), but the text
    # answer is no longer voice-only.
    await update.message.reply_text(reply_text)
    await _send_spoken_reply(update, chat_id, reply_text)


async def _register_commands(application: Application) -> None:
    # Telegram's own "/" autocomplete menu reads this list -- real,
    # native discoverability for a borrower who doesn't know a command
    # exists at all, at zero extra UI cost.
    await application.bot.set_my_commands(
        [
            ("start", "Begin or verify your account"),
            ("status", "Check your EMI, due date, and balance"),
            ("history", "See your recent payments"),
            ("pay", "Get a payment link -- /pay <amount>"),
            ("dispute", "Flag a dispute -- /dispute <what happened>"),
            ("agent", "Talk to a human"),
            ("closure", "Request a loan closure certificate"),
            ("language", "Change your language"),
            ("voice", "Toggle spoken replies on/off"),
            ("menu", "Show all options as buttons"),
            ("reset", "Start over"),
        ]
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Registered via application.add_error_handler -- the backstop for any
    exception a handler above didn't specifically catch (a real Postgres/
    Supabase connectivity blip during verification, e.g., or any other
    genuinely unexpected failure). Found live: with no error handler
    registered at all, python-telegram-bot's own default behavior for an
    unhandled handler exception is to log it internally and silently drop
    the update -- the borrower got zero indication anything failed, and
    had no reason to send /reset since nothing ever told them to. Logged
    here with the real traceback (never swallowed), and the borrower gets
    at least a generic acknowledgment instead of total silence."""
    logger.error("Unhandled exception while processing an update: %s", update, exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("Something went wrong on my end -- please try again in a moment.")


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set -- copy .env.example to .env and fill it in")

    application = Application.builder().token(token).post_init(_register_commands).build()
    application.add_handler(CommandHandler("start", on_start))
    application.add_handler(CommandHandler("reset", on_reset))
    application.add_handler(CommandHandler("hindi", on_hindi))
    application.add_handler(CommandHandler("english", on_english))
    application.add_handler(CommandHandler("language", on_language))
    application.add_handler(CommandHandler("status", on_status))
    application.add_handler(CommandHandler("history", on_history))
    application.add_handler(CommandHandler("pay", on_pay))
    application.add_handler(CommandHandler("dispute", on_dispute))
    application.add_handler(CommandHandler("agent", on_agent))
    application.add_handler(CommandHandler("closure", on_closure))
    application.add_handler(CommandHandler("voice", on_voice_toggle))
    application.add_handler(CommandHandler("menu", on_menu))
    application.add_handler(CallbackQueryHandler(on_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_message))
    application.add_handler(MessageHandler(filters.VOICE, on_voice_message))
    application.add_error_handler(on_error)

    application.run_polling()


if __name__ == "__main__":
    main()
