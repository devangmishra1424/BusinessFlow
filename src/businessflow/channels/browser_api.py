"""HTTP API for a browser-based text+voice chat channel -- the backend a
frontend can call (built separately, per the plan: backend first, then
UI/UX).

Conversation state is held in-process, in memory, keyed by a generated
conversation_id -- restarting this server loses any conversation still
in flight. Cross-session memory (the recap a returning borrower's NEXT
conversation gets) is unaffected, since that's persisted to Postgres via
memory/conversation_memory.py independently of this in-memory state.

Voice: the frontend records via MediaRecorder and uploads to
POST .../messages/voice. Chrome/Edge's MediaRecorder only ever emits
WebM/Opus, which soundfile/libsndfile (this project's only audio codec
dependency -- see telegram_bot.py's own note on avoiding ffmpeg/pydub)
cannot read at all -- unlike Telegram's OGG/Opus voice notes, which it
handles natively. Rather than add a codec dependency or an ffmpeg install
to the VM, the frontend decodes the recording via the browser's own
AudioContext (which can always decode whatever the browser itself just
encoded) and re-encodes it to a plain WAV blob before uploading -- see
static/app.js's audioBufferToWav. That keeps the server-side decode path
(_decode_and_transcribe_voice, below) identical to telegram_bot.py's,
since soundfile reads WAV natively.

TTS is on-demand per reply (POST .../speech), not generated for every
assistant message automatically -- the frontend only calls it when a
borrower taps the speaker icon on a specific bubble they want to hear,
so synthesis compute is spent only on replies someone actually wants
spoken.

Run: uvicorn businessflow.channels.browser_api:app --reload
(defaults to port 8000). The ops dashboard API (ops/api.py) is a
separate FastAPI app on port 8001 -- run both side by side, they don't
share state or a port.
"""

import io
import logging
import threading
import uuid
from datetime import date, datetime
from pathlib import Path

import groq
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from businessflow.accounts import store
from businessflow.accounts.documents import list_documents_for_account, resolve_document_path
from businessflow.accounts.models import Account
from businessflow.agent.loop import (
    AccessDeniedError,
    AccountLockedError,
    extract_new_tool_calls,
    index_of_last_user_message,
    run_turn_with_memory,
    start_conversation,
    verify_and_start_conversation,
)
from businessflow.audio.asr import transcribe
from businessflow.audio.tts import encode_ogg_opus, speak_english, speak_hindi
from businessflow.audio.vad import trim_to_speech
from businessflow.audio.verbalizer import verbalize
from businessflow.channels.credentials import looks_like_credentials, parse_credentials
from businessflow.ops.flags import Flag, compute_flags
from businessflow.rate_limit import RateLimiter
from businessflow.tools.account_tools import flag_dispute, get_payment_status
from businessflow.tools.escalation_tools import escalate_to_human
from businessflow.tools.payment_tools import generate_payment_link

_MAX_VOICE_NOTE_SECONDS = 120  # same bound as telegram_bot.py -- an explicit cap on ASR compute, not an unbounded wait

logger = logging.getLogger(__name__)

app = FastAPI(title="BusinessFlow Chat API")

# The borrower-facing chat's static frontend -- mounted at the bottom of
# this file, after every API route, same reasoning as ops/api.py's own
# static mount: an exact-path route like POST /conversations is always
# matched first, so the mount only ever serves /static/styles.css etc.
_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Dev-only: lets a locally-served frontend on any port call this API
# without a CORS error. Tighten to a specific origin before any real
# deployment -- this is intentionally permissive for local frontend work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_conversations: dict[str, dict] = {}

# One lock per conversation_id, serializing every call into
# _process_text_turn for that conversation -- the same race
# telegram_bot.py's _session_locks fixes (see that module's comment):
# without it, a text turn and a voice turn for the same conversation_id
# still in flight at once could both read/append/reassign
# session["messages"] concurrently in separate thread-pool workers (every
# endpoint below is a sync def, which FastAPI runs in a thread pool, not
# on the event loop), and whichever finishes last would silently
# overwrite the other's turn. setdefault is a single atomic dict
# operation in CPython, so two threads can never race to create two
# different locks for the same brand-new conversation_id.
_conversation_locks: dict[str, threading.Lock] = {}


def _get_conversation_lock(conversation_id: str) -> threading.Lock:
    return _conversation_locks.setdefault(conversation_id, threading.Lock())


class StartConversationRequest(BaseModel):
    account_id: str | None = None
    access_key: str | None = None  # required if account_id is given
    language: str = "en"


class StartConversationResponse(BaseModel):
    conversation_id: str
    account_id: str | None
    language: str


class SendMessageRequest(BaseModel):
    message: str


class ToolCallInfo(BaseModel):
    tool: str
    arguments: dict


class SendMessageResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCallInfo]
    # Set only on the turn where an anonymous session just verified via
    # credentials typed into the chat (see send_message_endpoint) -- lets
    # the frontend update its own account-bound UI (the header chip)
    # without parsing the reply text to detect what just happened.
    verified_account_id: str | None = None
    # Set only by the voice endpoint -- what ASR actually heard, so the
    # frontend can show it as the borrower's own chat bubble instead of
    # silently acting on audio the borrower never sees transcribed. None
    # for a plain text turn (nothing was transcribed).
    transcript: str | None = None


class AccountSnapshotOut(BaseModel):
    """Exactly the fields get_payment_status(account_id) returns -- see
    tools/account_tools.py's docstring for what each one means and its
    real caveats (outstanding_balance_approx is a simplification,
    interest_rate_pct is None until an agreement is parsed, etc)."""

    account_id: str
    borrower_name: str
    business_name: str
    principal_amount: float
    emi_amount: float
    emi_due_date: str
    days_past_due: int
    tenure_months: int
    months_remaining: int
    outstanding_balance_approx: float
    interest_rate_pct: float | None
    nach_mandate_active: bool
    late_fee_applicable: bool
    late_fee_amount: float | None
    dispute_open: bool
    risk_tier: str
    broken_promise_count: int
    pending_emi_credit: float


class TimelineEntryOut(BaseModel):
    date: str
    amount: float
    # "paid-on-time" | "paid-late" | "overdue" | "upcoming" | "extra-applied"
    # | "extra-unapplied" -- the last two are an off-cycle payment that
    # didn't retire a month on its own (see accounts.store.record_payment).
    status: str
    label: str


class DashboardEscalationOut(BaseModel):
    escalation_id: str
    reason: str
    status: str
    created_at: datetime
    resolved_at: datetime | None


class DocumentOut(BaseModel):
    filename: str
    size_bytes: int
    uploaded_at: datetime


class WarningOut(BaseModel):
    label: str  # "overdue" | "disputed" | "broken_promises" -- drives which action(s) the UI offers, never shown directly
    text: str


class MessageOut(BaseModel):
    message: str
    delivered_via_telegram: bool
    created_at: datetime


class DashboardResponse(BaseModel):
    account: AccountSnapshotOut
    timeline: list[TimelineEntryOut]
    warnings: list[WarningOut]
    escalations: list[DashboardEscalationOut]
    documents: list[DocumentOut]
    messages: list[MessageOut]


class QuickActionDisputeRequest(BaseModel):
    reason: str


class QuickActionAgentRequest(BaseModel):
    reason: str | None = None


class QuickActionPaymentLinkRequest(BaseModel):
    amount: float = Field(gt=0)


def _require_verified_account(conversation_id: str) -> str:
    """Shared gate for every dashboard/document/quick-action endpoint below.
    Deliberately two different failure codes, not one: a conversation_id
    that doesn't exist at all (typo'd, expired, made up) is a 404, same
    convention as send_message_endpoint's own 404 above; a conversation_id
    that IS real but hasn't verified into an account yet is a 403 -- the id
    itself is valid, but nothing has proven the caller owns any account, so
    there's nothing here for them to see. A borrower must never be able to
    reach another account's data by guessing or reusing a conversation_id,
    which is exactly what collapsing these two cases into one response
    (or, worse, trusting an unverified session's account_id) would risk."""
    session = _conversations.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no conversation found for id={conversation_id!r}")
    if session["account_id"] is None:
        raise HTTPException(
            status_code=403,
            detail="no verified account for this conversation -- verify with your account ID and access key first",
        )
    return session["account_id"]


def _log_quick_action(account_id: str, tool: str, arguments: dict, result: dict) -> None:
    # Mirrors telegram_bot.py's _log_tool_call: a quick-action button calls
    # these tools directly, bypassing agent.loop's LLM round trip entirely
    # (no ambiguity to resolve for a deterministic action) -- but it's just
    # as real a tool call as one the LLM makes, so it's logged the same way,
    # keeping the ops dashboard's and conversation_memory's account history
    # complete regardless of which path triggered the call.
    store.log_event(account_id, "tool_called", {"tool": tool, "arguments": arguments, "result": result})


# add_one_month lives in accounts.store now -- record_payment (the real,
# money-moving version of "advance one EMI cycle") needs the exact same
# calendar arithmetic this timeline projection does, so there's one
# shared implementation instead of two copies that could quietly drift
# apart. See its docstring for the JS-parity note this used to carry here.


def _build_emi_timeline(account: Account, days_past_due: int) -> list[dict]:
    """Server-side port of ops/static/app.js's buildEmiTimeline (see that
    function's own comment for the reasoning): real payment_history entries
    first, in chronological order (already sorted that way -- store.py's
    _load_payment_history loads "order by payment_date"), then a monthly
    projection forward from account.emi_due_date for account.months_remaining
    occurrences at account.emi_amount each. The first projected occurrence is
    flagged "overdue" (instead of "upcoming") exactly when days_past_due > 0,
    same condition the JS version checks."""
    def _entry(r):
        base_label = "Paid on time" if r.on_time else "Paid late"
        if r.kind == "extra_unapplied":
            return {"date": r.date.isoformat(), "amount": r.amount, "status": "extra-unapplied", "label": "Extra payment (not applied)"}
        if r.kind == "extra_applied":
            return {"date": r.date.isoformat(), "amount": r.amount, "status": "extra-applied", "label": "Extra payment (credited to next EMI)"}
        if r.kind == "overpayment_applied":
            return {"date": r.date.isoformat(), "amount": r.amount, "status": "paid-on-time" if r.on_time else "paid-late", "label": f"{base_label} + extra credited"}
        return {"date": r.date.isoformat(), "amount": r.amount, "status": "paid-on-time" if r.on_time else "paid-late", "label": base_label}

    past = [_entry(r) for r in account.payment_history]

    upcoming = []
    if account.months_remaining > 0:
        cursor = account.emi_due_date
        for i in range(account.months_remaining):
            is_next_due = i == 0
            overdue = is_next_due and days_past_due > 0
            # Only the very next installment can carry a credit from an
            # earlier off-cycle payment (see record_payment) -- every
            # projection after that is a plain, full emi_amount.
            amount_due = round(account.emi_amount - account.pending_emi_credit, 2) if is_next_due else account.emi_amount
            label = f"Overdue -- {days_past_due}d past due" if overdue else "Scheduled"
            if is_next_due and account.pending_emi_credit > 0.01:
                label += f" (₹{account.pending_emi_credit:,.2f} credited)"
            upcoming.append(
                {
                    "date": cursor.isoformat(),
                    "amount": amount_due,
                    "status": "overdue" if overdue else "upcoming",
                    "label": label,
                }
            )
            cursor = store.add_one_month(cursor)

    return past + upcoming


# DELIBERATE REFRAMING -- read before "simplifying" this back to raw flag
# text: ops/flags.py's Flag.reason strings are written for STAFF (e.g.
# "20 days past due (beyond the 3-day grace period)", "has an open,
# unresolved dispute"). Those exact strings are correct and useful on the
# ops dashboard, and wrong here -- a borrower reading their own dashboard
# needs a plain, first-person, non-accusatory sentence, not an internal
# policy citation. This function is the one place that translation happens;
# it is not a duplicate of compute_flags, it is what compute_flags' output
# has to go through before it's fit for a borrower to read. Every number in
# the output (days overdue, late fee amount, broken-promise count) is real,
# taken from payment_status (itself get_payment_status's own real result) or
# the flag -- never fabricated or estimated here.
def _build_warnings(flags: list[Flag], payment_status: dict) -> list[dict]:
    """Returns {label, text} per warning, not just text -- label is what
    the dashboard's own UI uses to decide which action(s) a warning can
    offer (see channels/static/app.js's WARNING_ACTIONS): "overdue" and
    "broken_promises" can be paid off or contested, "disputed" is already
    an active claim under review and offers neither. The label is never
    shown to the borrower directly -- only text is, same reframing rule
    as before."""
    warnings = []
    for flag in flags:
        if flag.label == "overdue":
            days = payment_status["days_past_due"]
            if payment_status["late_fee_applicable"]:
                text = (
                    f"Your EMI is {days} days overdue -- pay soon to avoid a late fee of "
                    f"₹{payment_status['late_fee_amount']:,.2f}."
                )
            else:
                text = f"Your EMI is {days} days overdue -- pay soon."
        elif flag.label == "disputed":
            text = "You have an open dispute -- our team is reviewing it."
        elif flag.label == "broken_promises":
            text = f"You have {payment_status['broken_promise_count']} missed payment promises on record."
        else:
            # Defensive only: no flag label besides the three above exists
            # today (see ops/flags.py's compute_flags). Falling back to the
            # raw, staff-toned reason for an unrecognized future label is a
            # real warning a borrower should still see, wrong tone and all --
            # strictly better than silently dropping it.
            text = flag.reason
        warnings.append({"label": flag.label, "text": text})
    return warnings


@app.get("/health")
def health():
    return {"status": "ok"}


# AccessDeniedError's own AccountLockedError (repeated wrong keys against
# ONE account_id) doesn't cover trying a handful of DIFFERENT account_ids,
# one guess each, from the same IP -- this closes that specific gap. Only
# actual failures count as hits, same reasoning as ops/api.py's own
# brute-force limiter: a borrower who gets it right immediately (the
# overwhelming normal case) must never be throttled by this.
_credential_brute_force_limiter = RateLimiter(max_requests=8, window_seconds=300)


@app.post("/conversations", response_model=StartConversationResponse)
def start_conversation_endpoint(request: Request, req: StartConversationRequest):
    if req.language not in ("en", "hi"):
        raise HTTPException(status_code=400, detail="language must be 'en' or 'hi'")

    if req.account_id:
        if not req.access_key:
            raise HTTPException(status_code=401, detail=f"access_key is required to talk about account {req.account_id}")
        try:
            conversation = verify_and_start_conversation(req.language, req.account_id, req.access_key)
        except AccessDeniedError:
            client_ip = request.client.host if request.client else "unknown"
            if not _credential_brute_force_limiter.check(client_ip):
                raise HTTPException(
                    status_code=429, detail="Too many verification attempts from this location -- please wait a few minutes."
                ) from None
            raise HTTPException(status_code=401, detail=f"wrong access key for account {req.account_id}") from None
        except AccountLockedError as e:
            raise HTTPException(status_code=429, detail=str(e)) from None
    else:
        conversation = start_conversation(language=req.language, account_id=None)

    conversation_id = str(uuid.uuid4())
    _conversations[conversation_id] = {
        "account_id": req.account_id,
        "language": req.language,
        "messages": conversation,
    }
    return StartConversationResponse(conversation_id=conversation_id, account_id=req.account_id, language=req.language)


def _process_text_turn(conversation_id: str, session: dict, text: str, client_ip: str) -> SendMessageResponse:
    """Shared turn-processing logic for both the text (send_message_endpoint)
    and voice (send_voice_message_endpoint) paths below: verification
    mid-conversation, run_turn_with_memory, and tool-call extraction. text
    is either the raw typed message or a voice transcript -- identical
    handling either way, same as telegram_bot.py's handle_incoming_message
    is shared by its text and voice paths. Everything that reads or
    mutates session["messages"]/session["account_id"] happens under this
    conversation's lock (see _get_conversation_lock's comment for the race
    this closes)."""
    with _get_conversation_lock(conversation_id):
        # Found live (Telegram, same underlying gap here): an already-
        # verified session that gets a redundant "<account_id> <key>"
        # message forwarded it straight to the LLM as ordinary text, with
        # no idea it was ever a credentials pair -- the model pattern-
        # matched the bare 6-digit number in a financial conversation and
        # hallucinated a payment intent. Short-circuit before it ever
        # reaches run_turn_with_memory, same as the verification branch below.
        if session["account_id"] is not None and looks_like_credentials(text):
            return SendMessageResponse(
                reply=f"You're already verified as account {session['account_id']} -- what can I help you with?",
                tool_calls=[],
            )

        # An anonymous session can still verify mid-conversation by sending
        # "<account_id> <6-digit key>" as a plain message -- the same pattern
        # telegram_bot.py already handles (see channels/credentials.py). Found
        # live: a borrower typing their real account_id + access_key straight
        # into an anonymous browser chat had both values forwarded to the LLM
        # as free text, which then passed the ACCESS KEY as account_id to a
        # tool and crashed it (get_payment_status raised "no account found for
        # account_id='<the key>'"). Checked here, before this message ever
        # reaches run_turn_with_memory, exactly like Telegram's guard.
        if session["account_id"] is None and looks_like_credentials(text):
            account_id, access_key = parse_credentials(text)
            try:
                conversation = verify_and_start_conversation(session["language"], account_id, access_key)
            except AccessDeniedError:
                # Same brute-force limiter /conversations uses -- a wrong
                # guess typed mid-chat is exactly as real an attempt as one
                # sent at conversation start, and must count toward the
                # same per-IP budget, not reset it for free.
                if not _credential_brute_force_limiter.check(client_ip):
                    return SendMessageResponse(
                        reply="Too many verification attempts from this location -- please wait a few minutes.",
                        tool_calls=[],
                    )
                return SendMessageResponse(
                    reply=f"That access key doesn't match account {account_id} -- please check and send both again.",
                    tool_calls=[],
                )
            except AccountLockedError as e:
                return SendMessageResponse(reply=str(e), tool_calls=[])
            # Verification itself is not a turn -- run_turn_with_memory is
            # deliberately not called for this message, matching
            # telegram_bot.py's handle_incoming_message. The anonymous
            # session's history is discarded outright: its system prompt
            # never knew an account_id existed, so there's nothing in it
            # worth carrying into the newly-verified conversation.
            session["account_id"] = account_id
            session["messages"] = conversation
            return SendMessageResponse(
                reply=f"Verified -- I've pulled up account {account_id}. What can I help you with?",
                tool_calls=[],
                verified_account_id=account_id,
            )

        session["messages"].append({"role": "user", "content": text})
        try:
            updated_conversation, reply = run_turn_with_memory(session["messages"], session["account_id"])
        except groq.RateLimitError as e:
            # Something the frontend can actually act on (show "try again in a
            # bit", maybe auto-retry) -- not the same as a real bug, so it
            # shouldn't come back as an opaque 500.
            session["messages"].pop()  # don't leave a user message with no reply appended
            raise HTTPException(status_code=503, detail=f"Groq rate limit reached: {e.message}") from e
        except groq.APIStatusError as e:
            session["messages"].pop()
            raise HTTPException(status_code=502, detail=f"upstream LLM provider error: {e.message}") from e
        except groq.APIConnectionError as e:
            # A real gap found live: APIConnectionError (network drop, DNS
            # failure) and its subclass APITimeoutError are siblings of
            # APIStatusError, not subclasses of it -- groq's own
            # _exceptions.py confirms this -- so neither except clause
            # above ever caught them. Left uncaught, this both returned an
            # unhandled 500 AND skipped the pop() cleanup, stranding the
            # just-appended user message with no paired reply for every
            # later turn to inherit.
            session["messages"].pop()
            raise HTTPException(status_code=502, detail=f"could not reach the LLM provider: {e}") from e
        session["messages"] = updated_conversation

        # Computed from the RETURNED conversation, not a pre-call
        # len(session["messages"]) snapshot -- that snapshot goes stale
        # the moment _run_turn_async trims older turns off the front
        # (see agent/loop.py's _trim_to_recent_turns), silently
        # mis-slicing this into the wrong turn's messages.
        turn_start = index_of_last_user_message(updated_conversation)
        tool_calls = extract_new_tool_calls(updated_conversation, turn_start)
        return SendMessageResponse(
            reply=reply,
            tool_calls=[ToolCallInfo(tool=name, arguments=args) for name, args in tool_calls],
        )


@app.post("/conversations/{conversation_id}/messages", response_model=SendMessageResponse)
def send_message_endpoint(request: Request, conversation_id: str, req: SendMessageRequest):
    session = _conversations.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no conversation found for id={conversation_id!r}")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    client_ip = request.client.host if request.client else "unknown"
    return _process_text_turn(conversation_id, session, req.message, client_ip)


def _decode_and_transcribe_voice(raw_bytes: bytes, language: str | None) -> str | None:
    """Same decode/resample/VAD/transcribe pipeline as telegram_bot.py's
    _decode_and_transcribe_voice_note (see that function's docstring for
    the full reasoning) -- duplicated rather than imported across channel
    modules, matching this codebase's existing convention of each channel
    wiring the lower-level audio primitives itself (see this file's and
    telegram_bot.py's module docstrings). soundfile auto-detects the
    container from the bytes themselves, so this handles the WAV this
    channel actually uploads (see send_voice_message_endpoint) exactly as
    it would OGG/Opus. Returns None if VAD found no speech, or if the
    decoded audio exceeds _MAX_VOICE_NOTE_SECONDS -- callers must not feed
    either case to ASR."""
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # collapse to mono, same convention as audio/io.py

    actual_duration_seconds = len(data) / sr
    if actual_duration_seconds > _MAX_VOICE_NOTE_SECONDS:
        return None

    audio_tensor = torch.from_numpy(data)
    if sr != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sr, new_freq=16000)

    trimmed = trim_to_speech(audio_tensor, sampling_rate=16000)
    if trimmed.numel() == 0:
        return None
    return transcribe(trimmed, language=language)


def _transcript_echo(transcript: str) -> str:
    """What's safe to send back to the frontend as the borrower's own
    displayed transcript. SECURITY: mirrors telegram_bot.py's own
    _transcript_echo exactly (see that function's docstring) -- a
    credential-shaped transcript must never be echoed back verbatim, since
    the frontend renders this field as the borrower's own chat bubble, and
    that would put the account_id + 6-digit access key in the browser's
    visible chat history and in this response's JSON body. Applies
    unconditionally, not just for an unverified session, same as
    Telegram's version."""
    if looks_like_credentials(transcript):
        return "[account details -- redacted]"
    return transcript


@app.post("/conversations/{conversation_id}/messages/voice", response_model=SendMessageResponse)
def send_voice_message_endpoint(request: Request, conversation_id: str, audio: UploadFile = File(...)):
    """Browser equivalent of telegram_bot.py's handle_incoming_voice: the
    frontend records via MediaRecorder, re-encodes to WAV client-side (see
    this file's module docstring for why), and uploads it here. Same
    credential-safety guard as Telegram: a spoken account_id+key that ASR
    mis-hears must never be forwarded into verification, since a wrong
    digit would burn one of the account's limited AccountLockedError
    attempts through no fault of the borrower's -- so a credential-shaped
    transcript from an unverified session is rejected before it ever
    reaches _process_text_turn, asking the borrower to type it instead."""
    session = _conversations.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no conversation found for id={conversation_id!r}")

    raw_bytes = audio.file.read()
    transcript = _decode_and_transcribe_voice(raw_bytes, session["language"])
    if not transcript or not transcript.strip():
        return SendMessageResponse(
            reply="Sorry, I couldn't make out that recording -- please make sure it's under 2 minutes and try again.",
            tool_calls=[],
        )

    if session["account_id"] is None and looks_like_credentials(transcript):
        return SendMessageResponse(
            reply=(
                "That sounded like your account ID and access key -- to avoid a "
                'misheard digit locking your account, please TYPE them instead, '
                'e.g. "BF-1001 482913".'
            ),
            tool_calls=[],
            transcript=_transcript_echo(transcript),
        )

    client_ip = request.client.host if request.client else "unknown"
    result = _process_text_turn(conversation_id, session, transcript, client_ip)
    result.transcript = _transcript_echo(transcript)
    return result


class SpeechRequest(BaseModel):
    text: str


@app.post("/conversations/{conversation_id}/speech")
def speech_endpoint(conversation_id: str, req: SpeechRequest):
    """On-demand TTS for one reply -- called only when the borrower taps
    the speaker icon on a specific bubble (see this file's module
    docstring), not generated automatically for every assistant message.
    Reuses this conversation's own language setting rather than accepting
    one from the client, so playback always matches what the conversation
    has been using throughout. Returns raw OGG/Opus bytes, same encoding
    telegram_bot.py's TTS replies already use -- every major browser can
    play that back natively, even though (per this file's module
    docstring) Chrome/Edge cannot *record* into that container."""
    session = _conversations.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no conversation found for id={conversation_id!r}")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    language = session["language"]
    # Found live: this had no error handling at all, unlike telegram_bot.py's
    # _send_spoken_reply -- which wraps the identical speak_hindi/speak_english/
    # encode_ogg_opus calls in a broad except for exactly this reason (VAD/
    # TTS/network can all fail in ways worth treating identically here).
    # Matching that same deliberate, already-established pattern rather than
    # a narrower catch: an unhandled failure here was a bare 500 with nothing
    # logged server-side to explain it.
    try:
        speech = speak_hindi(verbalize(req.text, language)) if language == "hi" else speak_english(verbalize(req.text, language))
        audio_bytes = encode_ogg_opus(speech)
    except Exception as e:
        logger.warning("TTS synthesis failed for conversation_id=%s: %s", conversation_id, e, exc_info=True)
        raise HTTPException(status_code=502, detail="text-to-speech is temporarily unavailable -- please try again") from e
    return Response(content=audio_bytes, media_type="audio/ogg")


@app.get("/conversations/{conversation_id}/dashboard", response_model=DashboardResponse)
def get_dashboard_endpoint(conversation_id: str):
    """The main aggregation endpoint the borrower dashboard screen (a
    separate frontend, landed after verification instead of going straight
    into chat) loads on open: account snapshot, EMI timeline (real past
    payments + computed upcoming EMIs), borrower-toned warnings, this
    account's own escalation history, and its uploaded documents. Every
    piece is real data for THIS conversation's verified account only --
    _require_verified_account is what makes that a guarantee, not a
    convention callers have to remember."""
    account_id = _require_verified_account(conversation_id)

    payment_status = get_payment_status(account_id)
    account = store.get_account_or_raise(account_id)
    flags = compute_flags(account)
    escalations = store.get_escalations_for_account(account_id)
    documents = list_documents_for_account(account_id)

    return DashboardResponse(
        account=AccountSnapshotOut(**payment_status),
        timeline=[TimelineEntryOut(**entry) for entry in _build_emi_timeline(account, payment_status["days_past_due"])],
        warnings=[WarningOut(**w) for w in _build_warnings(flags, payment_status)],
        escalations=[
            DashboardEscalationOut(
                escalation_id=e.escalation_id, reason=e.reason, status=e.status,
                created_at=e.created_at, resolved_at=e.resolved_at,
            )
            for e in escalations
        ],
        documents=[DocumentOut(**d) for d in documents],
        # A clarification request is a real message from ops about this
        # account's flags -- previously visible to the borrower ONLY if
        # Telegram was linked and delivery succeeded; otherwise it was
        # silently lost to them (still logged, but nowhere they'd ever
        # see it). Surfacing the same real history here means it's never
        # lost, regardless of Telegram.
        messages=[MessageOut(**m) for m in store.get_clarification_requests(account_id)],
    )


@app.get("/conversations/{conversation_id}/documents/{filename}")
def download_document_endpoint(conversation_id: str, filename: str):
    """Read-only, conversation-scoped download of one of THIS account's own
    uploaded documents -- the borrower-facing equivalent of ops/api.py's
    staff-only document download, backed by the same data/documents/
    {account_id}/ files. resolve_document_path (accounts/documents.py)
    is what actually enforces "only this account, nothing else reachable
    via a crafted filename" -- a miss there and a genuinely nonexistent
    filename come back as the exact same 404, so a caller can never tell
    the difference between "wrong filename" and "that belongs to another
    account" (guessing/probing must not leak which)."""
    account_id = _require_verified_account(conversation_id)
    path = resolve_document_path(account_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no document found for filename={filename!r}")
    return FileResponse(path, filename=path.name)


@app.post("/conversations/{conversation_id}/quick-actions/dispute")
def quick_action_dispute_endpoint(conversation_id: str, req: QuickActionDisputeRequest) -> dict:
    account_id = _require_verified_account(conversation_id)
    if not req.reason.strip():
        raise HTTPException(status_code=400, detail="reason must not be empty")
    result = flag_dispute(account_id, req.reason)
    _log_quick_action(account_id, "flag_dispute", {"account_id": account_id, "reason": req.reason}, result)
    return result


_DEFAULT_AGENT_REASON = "Borrower requested a human agent from the dashboard"


@app.post("/conversations/{conversation_id}/quick-actions/agent")
def quick_action_agent_endpoint(conversation_id: str, req: QuickActionAgentRequest) -> dict:
    account_id = _require_verified_account(conversation_id)
    reason = req.reason or _DEFAULT_AGENT_REASON
    result = escalate_to_human(account_id, reason)
    _log_quick_action(account_id, "escalate_to_human", {"account_id": account_id, "reason": reason}, result)
    return result


@app.post("/conversations/{conversation_id}/quick-actions/payment-link")
def quick_action_payment_link_endpoint(conversation_id: str, req: QuickActionPaymentLinkRequest) -> dict:
    account_id = _require_verified_account(conversation_id)
    result = generate_payment_link(account_id, req.amount)
    _log_quick_action(
        account_id, "generate_payment_link", {"account_id": account_id, "amount": req.amount}, result
    )
    return result


class PaymentTokenInfoOut(BaseModel):
    account_id: str
    amount: float
    business_name: str
    borrower_name: str
    status: str  # "pending" | "used" | "expired"
    # What's actually due THIS cycle (emi_amount minus any pending credit)
    # -- the confirm page compares this against amount to decide whether
    # to ask "apply this toward your next EMI?" before the borrower even
    # clicks confirm. None for a used/expired token (no live account math
    # worth showing at that point).
    emi_amount_due: float | None = None


class PaymentConfirmRequest(BaseModel):
    # Only meaningful when amount < emi_amount_due; see store.record_payment's
    # docstring for the full decision table. Left None for a normal payment
    # that fully covers what's due -- store.record_payment never asks for
    # it in that case.
    apply_extra_to_next: bool | None = None


class PaymentConfirmOut(BaseModel):
    amount: float
    kind: str  # "regular" | "extra_unapplied" | "extra_applied" | "overpayment_applied"
    months_remaining: int
    next_emi_due_date: str
    pending_emi_credit: float


@app.get("/pay/{token}/info", response_model=PaymentTokenInfoOut)
def payment_token_info_endpoint(token: str):
    """Read-only -- what the confirm PAGE calls on load to render "Confirm
    ₹X for [business]" before anything is actually redeemed. A genuinely
    unknown token is the only 404 case here; an expired or already-used
    one still returns its real status (see store.get_payment_token_info)
    so the page can tell the borrower WHY it can't be paid, not just that
    it can't."""
    info = store.get_payment_token_info(token)
    if info is None:
        raise HTTPException(status_code=404, detail=f"no payment link found for token={token!r}")
    if info["status"] == "pending":
        account = store.get_account_or_raise(info["account_id"])
        info["emi_amount_due"] = round(account.emi_amount - account.pending_emi_credit, 2)
    return PaymentTokenInfoOut(**info)


# Unlike the two limiters above, this one is NOT scoped to failures only --
# a payment token is a single guessable string with no separate password,
# so unlike an account_id+key pair, volume itself is the risk (someone
# scanning many token guesses from one IP looking for a live one). A real
# borrower only ever calls this once per link, so a generous per-IP volume
# cap here costs normal use nothing.
_payment_confirm_rate_limiter = RateLimiter(max_requests=20, window_seconds=300)


@app.post("/pay/{token}/confirm", response_model=PaymentConfirmOut)
def payment_confirm_endpoint(request: Request, token: str, req: PaymentConfirmRequest | None = None):
    """The only endpoint that can actually move an account forward from a
    payment link -- store.redeem_payment_token re-checks used_at/
    expires_at itself rather than trusting this endpoint already did (a
    double-submit from a slow network or an impatient double-tap must
    never record two payments for one token). req is optional so a plain
    POST with no body (a full payment, matching what's due) still works --
    the frontend only ever sends apply_extra_to_next when it actually has
    an answer to send."""
    client_ip = request.client.host if request.client else "unknown"
    if not _payment_confirm_rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Too many payment attempts from this location -- please wait a few minutes.")
    apply_extra_to_next = req.apply_extra_to_next if req is not None else None
    try:
        result = store.redeem_payment_token(token, apply_extra_to_next=apply_extra_to_next)
    except store.PaymentTokenNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except store.PaymentTokenAlreadyUsedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except store.PaymentTokenExpiredError as e:
        raise HTTPException(status_code=410, detail=str(e)) from e
    except store.ExtraPaymentDecisionRequiredError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    store.log_event(
        result["account_id"], "tool_called",
        {"tool": "record_payment", "arguments": {"token": token, "apply_extra_to_next": apply_extra_to_next}, "result": result},
    )
    return PaymentConfirmOut(
        amount=result["amount"],
        kind=result["kind"],
        months_remaining=result["months_remaining"],
        next_emi_due_date=result["next_emi_due_date"],
        pending_emi_credit=result["pending_emi_credit"],
    )


@app.get("/pay/{token}", include_in_schema=False)
def payment_page(token: str):
    """Serves the standalone confirm-payment page shell -- deliberately
    NOT redeeming anything on this GET (a payment link is exactly the
    kind of URL a chat client, a Telegram link preview, or a bot might
    pre-fetch; a GET must never have that side effect). The page's own
    JS calls /pay/{token}/info to render, then POSTs /pay/{token}/confirm
    only once the borrower actually clicks Confirm."""
    return FileResponse(_STATIC_DIR / "pay.html")


@app.get("/", include_in_schema=False)
def chat_ui():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
