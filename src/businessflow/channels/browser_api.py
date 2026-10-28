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
spoken. With the page's "Naturalise tone" option on, it asks for the
prepared words instead (POST .../speech-text) and speaks them with a
natural voice from the borrower's own browser, falling back to the
server voice if that voice never starts.

Run: uvicorn businessflow.channels.browser_api:app --reload
(defaults to port 8000). The ops dashboard API (ops/api.py) is a
separate FastAPI app on port 8001 -- run both side by side, they don't
share state or a port.
"""

import asyncio
import hmac
import io
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

import groq
import psycopg
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from businessflow.accounts import store
from businessflow.accounts import cases
from businessflow.accounts.db import check_database
from businessflow.accounts.dues import allocate_payment, charges_due
from businessflow.accounts.escalation_kinds import (
    CALLBACK, CLOSURE, FRAUD, REPLY_REASON_PREFIX, RESTRUCTURING, SYSTEM, classify_escalation,
)
from businessflow.accounts.policy import GRACE_PERIOD_DAYS, PAYMENT_CLAIM_REVIEW_WORKING_DAYS
from businessflow.accounts.documents import list_documents_for_account, resolve_document_path
from businessflow.accounts.models import Account
from businessflow.agent.loop import (
    AccessDeniedError,
    AccountLockedError,
    extract_new_tool_calls,
    index_of_last_user_message,
    run_turn_with_memory,
    start_conversation,
    start_conversation_with_recap,
    verify_and_start_conversation,
)
from businessflow.audio import remote
from businessflow.audio.asr import transcribe
from businessflow.audio.tts import encode_ogg_opus, prefetch_voices, speak_english, speak_hindi
from businessflow.audio.vad import trim_to_speech
from businessflow.audio.verbalizer import verbalize
from businessflow.channels.abilities import abilities_for, intro
from businessflow.channels.credentials import looks_like_credentials, parse_credentials
from businessflow.observability import trace
from businessflow.ops.flags import Flag, compute_flags
from businessflow.rate_limit import RateLimiter
from businessflow.tools import validation
from businessflow.tools.account_tools import flag_dispute, get_payment_status
from businessflow.tools.escalation_tools import escalate_to_human, request_closure_certificate
from businessflow.tools.payment_tools import generate_payment_link
from businessflow.tools.policy_tools import warm_up_in_background as warm_up_policy_search
from businessflow.tools.request_tools import request_fee_waiver

_MAX_VOICE_NOTE_SECONDS = 120  # same bound as telegram_bot.py -- an explicit cap on ASR compute, not an unbounded wait

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # The Hindi voice is fetched before the first request, so the first Hindi "speak" tap does not download 63 MB.
    await asyncio.to_thread(prefetch_voices)
    # The policy search's models load here, on a thread of their own, so the first policy question after a restart or a
    # deploy does not spend ~20 s loading them (tools/policy_tools.py). The service takes requests meanwhile.
    warm_up_policy_search()
    yield


app = FastAPI(title="BusinessFlow Chat API", lifespan=_lifespan)

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
    if len(_conversation_locks) > 2000:
        for k in list(_conversation_locks.keys())[:1000]:
            _conversation_locks.pop(k, None)
    return _conversation_locks.setdefault(conversation_id, threading.Lock())


class StartConversationRequest(BaseModel):
    account_id: str | None = None
    access_key: str | None = None  # required if account_id is given, unless demo_open on a demo deployment
    language: str = "en"
    # The public demo's one-click sign-in (GET /demo/accounts lists every account): no key asked. Refused unless the
    # deployment is a demo -- see _demo_open_access.
    demo_open: bool = False


def _demo_open_access() -> bool:
    """On for the public demo, where every account is a demo account and the page opens any of them in one click;
    a real lender's deployment sets DEMO_OPEN_ACCESS=0, and then every account needs its access key."""
    return os.environ.get("DEMO_OPEN_ACCESS", "1") != "0"


class DemoAccountOut(BaseModel):
    account_id: str
    borrower_name: str
    business_name: str
    loan_type: str = ""  # a borrower with several loans appears once per loan; this says which


@app.get("/demo/accounts", response_model=list[DemoAccountOut])
def demo_accounts_endpoint():
    """Every account, for the demo's sign-in list -- one opened from ops or a loan-book import shows up by itself.
    404 when the deployment is not a demo."""
    if not _demo_open_access():
        raise HTTPException(status_code=404, detail="not available on this deployment")
    return [DemoAccountOut(account_id=a.account_id, borrower_name=a.borrower_name, business_name=a.business_name,
                           loan_type=a.loan_type) for a in store.list_accounts()]


class StartConversationResponse(BaseModel):
    conversation_id: str
    account_id: str | None
    language: str


class SendMessageRequest(BaseModel):
    message: str


class ToolCallInfo(BaseModel):
    tool: str
    arguments: dict


class ToolTiming(BaseModel):
    tool: str
    seconds: float
    outcome: str  # "ok", "error", or "blocked" (another account, or an unverified chat)


class AnswerTiming(BaseModel):
    """What the chat shows small under a reply (observability/trace.py's answer_timing): server time from the request
    arriving to the reply being ready. It does not include the network to the browser or speaking the reply aloud."""
    seconds: float
    listen_seconds: float | None = None  # speech-to-text, on a voice turn only
    model_seconds: float  # every model call of the turn together
    model_calls: int
    tools: list[ToolTiming]


class SendMessageResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCallInfo]
    # Set when the turn went through the model; None for a reply written without it (verification, a refused
    # recording), which has nothing to time.
    timing: AnswerTiming | None = None
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
    # Each charge still owed (a late fee), with the EMI it is for -- see accounts/dues.py.
    charges: list[dict] = []
    # What the borrower owes right now (EMI less credit, plus every late fee still owed)
    # -- the one number the Pay button, reminders and Telegram agree on.
    amount_due_now: float
    dispute_open: bool
    risk_tier: str
    broken_promise_count: int
    pending_emi_credit: float
    # The borrower's current (or most recent) promise, judged by its own words -- accounts/promises.py plan_view.
    promise: dict | None = None
    # Late fees a person on the team waived in the last 60 days, with their reason (tools/account_tools.py).
    recently_waived: list[dict] = []


class TimelineEntryOut(BaseModel):
    date: str
    amount: float
    # "paid-on-time" | "paid-late" | "overdue" | "upcoming" | "extra-applied"
    # | "extra-unapplied" -- the last two are an off-cycle payment that
    # didn't retire a month on its own (see accounts.store.record_payment).
    status: str
    label: str


class ThreadMessageOut(BaseModel):
    author: str  # "borrower" (shown as "You") | "ops" (shown as "Our team")
    message: str
    created_at: datetime


class RequestOut(BaseModel):
    """One thing the borrower asked for or raised, in the borrower's own
    terms, as one conversation: what they said, every message about it
    since, and how it ended (accounts/cases.py). Never the raw escalation
    reason: that is staff-facing text (and for system-raised tickets -- a
    guardrail block, a chronic-delinquency notice -- it was previously
    printed verbatim on the borrower's screen).
    """

    id: str  # the case reference ("D-12", "ESC-0012") -- what a note is posted against
    kind: str  # restructuring | closure | callback | request | fraud | dispute | payment_claim
    title: str
    detail: str | None = None
    status: str  # open | done | declined (css hook)
    status_label: str
    outcome: str | None = None  # the human's own words on how it ended, when they gave any
    created_at: datetime
    resolved_at: datetime | None = None
    thread: list[ThreadMessageOut] = []
    can_reply: bool = False  # open: the borrower can add a note


class DocumentOut(BaseModel):
    filename: str
    size_bytes: int
    uploaded_at: datetime


class WarningOut(BaseModel):
    label: str  # "overdue" | "fee_due" | "disputed" | "broken_promises" -- drives which action(s) the UI offers, never shown directly
    text: str


class MessageOut(BaseModel):
    """Something we sent that isn't about one request (a reminder, a general
    message from our team); a message about a request is in its thread."""

    message: str
    delivered_via_telegram: bool
    created_at: datetime
    kind: str = "message"  # message | decision | dispute | fee | reminder
    id: int | None = None
    replied_case_ref: str | None = None  # the request the borrower's reply opened, if they replied


class DashboardResponse(BaseModel):
    account: AccountSnapshotOut
    timeline: list[TimelineEntryOut]
    warnings: list[WarningOut]
    requests: list[RequestOut]
    documents: list[DocumentOut]
    messages: list[MessageOut]


class QuickActionDisputeRequest(BaseModel):
    reason: str


class QuickActionPaymentClaimRequest(BaseModel):
    # When and how they paid, in their own words (amount, date, UPI/UTR reference) -- what ops checks against.
    details: str


class CaseNoteRequest(BaseModel):
    message: str


class QuickActionAgentRequest(BaseModel):
    reason: str | None = None


class QuickActionPaymentLinkRequest(BaseModel):
    amount: float = Field(gt=0)


class QuickActionFeeWaiverRequest(BaseModel):
    reason: str  # why they ask, in their own words -- what the person deciding reads


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
        if r.kind == "principal_prepayment_reduce_emi":
            return {"date": r.date.isoformat(), "amount": r.amount, "status": "paid-on-time" if r.on_time else "paid-late", "label": f"{base_label} + extra paid off the loan (EMI lowered)"}
        if r.kind == "charge":
            return {"date": r.date.isoformat(), "amount": r.amount, "status": "paid-on-time", "label": "Late fee paid"}
        if r.charges_paid > 0.005 and r.kind == "regular":
            return {"date": r.date.isoformat(), "amount": r.amount, "status": "paid-on-time" if r.on_time else "paid-late",
                    "label": f"{base_label} (includes ₹{r.charges_paid:,.0f} late fee)"}
        if r.kind == "principal_prepayment_reduce_tenure":
            return {"date": r.date.isoformat(), "amount": r.amount, "status": "paid-on-time" if r.on_time else "paid-late", "label": f"{base_label} + extra paid off the loan (loan shortened)"}
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
            if not overdue:
                label = "Scheduled"
            elif days_past_due > GRACE_PERIOD_DAYS:
                label = f"Overdue — {days_past_due}d past due"
            else:
                # Past the due date but inside the grace period: no late fee
                # and no warning banner yet, so don't paint it as "Overdue".
                label = f"Due {days_past_due}d ago — within the grace period"
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
def _borrower_date(iso: str) -> str:
    """'2026-10-03' -> '3 Oct 2026', the way the dashboard writes dates."""
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%b')} {d.year}"


def _build_warnings(flags: list[Flag], payment_status: dict, open_claim_kind: str | None = None) -> list[dict]:
    """Returns {label, text} per warning, not just text -- label is what
    the dashboard's own UI uses to decide which action(s) a warning can
    offer (see channels/static/app.js's WARNING_ACTIONS): "overdue" and
    "broken_promises" can be paid off or contested, "disputed" is already
    an active claim under review and offers neither. The label is never
    shown to the borrower directly -- only text is, same reframing rule
    as before."""
    warnings = []
    # A late fee left after the EMI was paid: the account is no longer overdue, so no overdue warning would mention it,
    # and the fee would drop out of sight while still owed. Said once, with the EMI it is for.
    fees = payment_status.get("charges") or []
    if fees and not any(f.label == "overdue" for f in flags):
        total = sum(c["amount_still_owed"] for c in fees)
        for_dates = ", ".join(_borrower_date(c["for_emi_due_date"]) for c in fees)
        flags = [Flag(label="fee_due", reason=f"A late fee of ₹{total:,.0f} (for your {for_dates} EMI) is still due."), *flags]
    for flag in flags:
        if flag.label == "overdue":
            days = payment_status["days_past_due"]
            if payment_status["late_fee_applicable"]:
                # The fee ALREADY applies at this point (late_fee_applicable
                # is days_past_due > the grace period, the same condition
                # that raises this flag) -- so don't say "pay soon to avoid"
                # a fee the borrower has already been charged.
                text = (
                    f"Your EMI is {days} days overdue, and a late fee of "
                    f"₹{payment_status['late_fee_amount']:,.0f} has been added."
                )
            else:
                text = f"Your EMI is {days} days overdue."
        elif flag.label == "fee_due":
            text = flag.reason
        elif flag.label == "disputed" and open_claim_kind == cases.PAYMENT_CLAIM:
            text = (
                f"We're checking the payment you reported, usually within {PAYMENT_CLAIM_REVIEW_WORKING_DAYS} "
                "working days. Reminders are paused until then."
            )
        elif flag.label == "disputed":
            text = "You have an open dispute — our team is reviewing it."
        elif flag.label == "broken_promises":
            n = payment_status["broken_promise_count"]
            text = (
                f"{n} earlier payment promises weren't completed — talk to us and we'll "
                "agree a plan that works for you."
            )
        else:
            # Defensive only: no flag label besides the three above exists
            # today (see ops/flags.py's compute_flags). Falling back to the
            # raw, staff-toned reason for an unrecognized future label is a
            # real warning a borrower should still see, wrong tone and all --
            # strictly better than silently dropping it.
            text = flag.reason
        warnings.append({"label": flag.label, "text": text})
    return warnings


def _request_title(kind: str, proposed_changes: dict | None) -> tuple[str, str | None]:
    if kind == RESTRUCTURING:
        detail = None
        if proposed_changes and proposed_changes.get("type") == "extend_tenure":
            detail = (
                f"Extend by {proposed_changes['extra_months']} month(s) · new EMI "
                f"₹{float(proposed_changes['new_emi_amount']):,.2f} over "
                f"{proposed_changes['new_months_remaining']} months"
            )
        return "Request for more time on your EMI", detail
    if kind == CLOSURE:
        return "Loan closure certificate", None
    if kind == CALLBACK:
        return "Call-back request", None
    if kind == FRAUD:
        return "Your concern about this loan", None
    return "Request to our team", None


def _thread(threads: dict[str, list[dict]], case_ref: str) -> list[dict]:
    return [
        {"author": m["author"], "message": m["message"], "created_at": m["created_at"]}
        for m in threads.get(case_ref, [])
    ]


def _build_requests(escalations: list, disputes: list[dict], threads: dict[str, list[dict]] | None = None) -> list[dict]:
    """The borrower's own view of what they've asked for -- escalations the
    borrower (or the agent on their behalf) raised, plus their disputes --
    newest first. Tickets the system or staff raised (guardrail blocks,
    chronic-delinquency notices, broken-promise patterns, call-log nudges)
    are internal and never shown: before this, the borrower read
    "Guardrail blocked a reply: amount(s) not from any real tool result ... [12500.0]"
    under "Your requests", and their actual dispute wasn't listed at all.
    Statuses are in borrower terms: an Approve on a plain hand-off just means
    staff closed it, so it reads "Closed", not "Approved".

    threads: {case_ref: messages} (store.get_case_threads) -- each request
    carries its own messages, so our reply sits under what it answers."""
    threads = threads or {}
    items = []
    for e in escalations:
        kind = classify_escalation(e.reason, e.proposed_changes)
        if kind == SYSTEM:
            continue
        title, detail = _request_title(kind, e.proposed_changes)
        if (e.reason or "").startswith(REPLY_REASON_PREFIX):  # their answer to a message from us: show what they said
            title, detail = "Your reply to our message", e.reason[len(REPLY_REASON_PREFIX):].removeprefix(" of ")[:400]
        if e.status == "queued_for_human":
            status, label = "open", "In progress"
        elif e.status == "rejected":
            status, label = "declined", "Not approved"
        else:
            status, label = "done", ("Approved" if kind in (RESTRUCTURING, CLOSURE) else "Closed")
        items.append({
            "id": e.escalation_id, "kind": kind, "title": title, "detail": detail,
            "status": status, "status_label": label, "outcome": e.resolution_reason,
            "created_at": e.created_at, "resolved_at": e.resolved_at,
            "thread": _thread(threads, e.escalation_id), "can_reply": e.status == "queued_for_human",
        })
    for d in disputes:
        is_open = d["status"] == "open"
        kind = d.get("kind") or cases.DISPUTE
        case_ref = cases.dispute_case_ref(d["id"]) if d.get("id") is not None else f"dispute-{d['opened_at'].isoformat()}"
        if kind == cases.PAYMENT_CLAIM:
            title, open_label, closed_label = "Payment you reported", "Being checked", "Closed"
        else:
            title, open_label, closed_label = "Your question about a charge", "Under review", "Resolved"
        items.append({
            "id": case_ref, "kind": kind, "title": title,
            "detail": (d["reason"] or "")[:200] or None,
            "status": "open" if is_open else "done",
            "status_label": open_label if is_open else closed_label,
            "outcome": d.get("resolution_note"),
            "created_at": d["opened_at"], "resolved_at": d.get("resolved_at"),
            "thread": _thread(threads, case_ref), "can_reply": is_open and d.get("id") is not None,
        })
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items


@app.get("/health")
def health():
    # 503 (not 200) when the database is unreachable: a health check that
    # only proves the process is alive kept saying "ok" while every real
    # request took ~36 s to fail. The body names the failing dependency
    # but never the error text, which can carry the host/user.
    try:
        check_database()
    except (psycopg.Error, RuntimeError) as exc:
        logger.error("health check failed: database unreachable (%s) -- check the Supabase project isn't paused and DATABASE_URL is right", type(exc).__name__)
        return JSONResponse(status_code=503, content={"status": "degraded", "database": "unreachable"})
    body = {"status": "ok", "database": "ok"}
    if remote.configured():
        # Still 200: a voice service that is down costs speed (the local models answer), not the service. "local-fallback"
        # is how a monitor sees that it is down; the reason is in the log (audio/remote.py), not in this public body.
        body["voice"] = "local-fallback" if remote.status()["paused"] else "remote"
    return body


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

    if req.account_id and req.demo_open and not req.access_key:
        if not _demo_open_access():
            raise HTTPException(status_code=401, detail=f"access_key is required to talk about account {req.account_id}")
        if store.get_account(req.account_id) is None:
            raise HTTPException(status_code=404, detail=f"no account {req.account_id}")
        conversation = start_conversation_with_recap(req.language, req.account_id)
        store.log_event(req.account_id, "demo_open_sign_in", {})
    elif req.account_id:
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
    if len(_conversations) > 2000:
        for k in list(_conversations.keys())[:1000]:
            _conversations.pop(k, None)
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
            msg = f"Groq rate limit reached: {e.message}"
            session["messages"].append({"role": "assistant", "content": "I am currently rate limited, please try again shortly."})
            raise HTTPException(status_code=503, detail=msg) from e
        except groq.APIStatusError as e:
            msg = f"upstream LLM provider error: {e.message}"
            session["messages"].append({"role": "assistant", "content": "I encountered an API error, please try again."})
            raise HTTPException(status_code=502, detail=msg) from e
        except groq.APIConnectionError as e:
            msg = f"could not reach the LLM provider: {e}"
            session["messages"].append({"role": "assistant", "content": "I am having trouble connecting to the network, please try again."})
            raise HTTPException(status_code=502, detail=msg) from e
        session["messages"] = updated_conversation

        # Computed from the RETURNED conversation, not a pre-call
        # len(session["messages"]) snapshot -- that snapshot goes stale
        # the moment _run_turn_async trims older turns off the front
        # (see agent/loop.py's _trim_to_recent_turns), silently
        # mis-slicing this into the wrong turn's messages.
        turn_start = index_of_last_user_message(updated_conversation)
        tool_calls = extract_new_tool_calls(updated_conversation, turn_start)
        timing = trace.answer_timing()
        return SendMessageResponse(
            reply=reply,
            tool_calls=[ToolCallInfo(tool=name, arguments=args) for name, args in tool_calls],
            timing=AnswerTiming(**timing) if timing else None,
        )


@app.post("/conversations/{conversation_id}/messages", response_model=SendMessageResponse)
def send_message_endpoint(request: Request, conversation_id: str, req: SendMessageRequest):
    session = _conversations.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no conversation found for id={conversation_id!r}")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    client_ip = request.client.host if request.client else "unknown"
    with trace.turn("web_text", account_id=session["account_id"]):
        return _process_text_turn(conversation_id, session, req.message, client_ip)


def _decode_and_transcribe_voice(
    raw_bytes: bytes, language: str | None, max_seconds: float = _MAX_VOICE_NOTE_SECONDS,
) -> str | None:
    """Same decode/resample/VAD/transcribe pipeline as telegram_bot.py's
    _decode_and_transcribe_voice_note (see that function's docstring for
    the full reasoning) -- duplicated rather than imported across channel
    modules, matching this codebase's existing convention of each channel
    wiring the lower-level audio primitives itself (see this file's and
    telegram_bot.py's module docstrings). soundfile auto-detects the
    container from the bytes themselves, so this handles the WAV this
    channel actually uploads (see send_voice_message_endpoint) exactly as
    it would OGG/Opus. Returns None if VAD found no speech, or if the
    decoded audio exceeds max_seconds -- callers must not feed either case
    to ASR."""
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # collapse to mono, same convention as audio/io.py

    actual_duration_seconds = len(data) / sr
    if actual_duration_seconds > max_seconds:
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
    """One voice turn, traced as one chain (observability/trace.py): speech-to-text, the agent, its tools and checks."""
    session = _conversations.get(conversation_id)
    with trace.turn("web_voice", account_id=session["account_id"] if session else None):
        return _voice_turn(request, conversation_id, audio)


def _voice_turn(request: Request, conversation_id: str, audio: UploadFile):
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
    with trace.turn("web_speak", account_id=session["account_id"]):
        try:
            speech = speak_hindi(verbalize(req.text, language)) if language == "hi" else speak_english(verbalize(req.text, language))
            audio_bytes = encode_ogg_opus(speech)
        except Exception as e:
            logger.warning("TTS synthesis failed for conversation_id=%s: %s", conversation_id, e, exc_info=True)
            raise HTTPException(status_code=502, detail="text-to-speech is temporarily unavailable -- please try again") from e
    return Response(content=audio_bytes, media_type="audio/ogg")


class SpeechTextRequest(BaseModel):
    # A reply is a few sentences. The speech endpoint above takes any size because making audio is its own natural limit;
    # this one only rewrites text, so it needs an explicit cap instead.
    text: str = Field(min_length=1, max_length=4000)


class SpeechTextResponse(BaseModel):
    text: str
    language: str


@app.post("/conversations/{conversation_id}/speech-text", response_model=SpeechTextResponse)
def speech_text_endpoint(conversation_id: str, req: SpeechTextRequest):
    """The words /speech would have spoken, as text instead of audio: the same verbalize() step (account ids spelled out,
    rupee amounts in lakh/crore or Devanagari, phone numbers in groups, ...) in this conversation's own language. The chat
    page's "Naturalise tone" option reads this with a natural-sounding voice from the borrower's own browser, so that voice
    says the same thing the server voice would, and no audio is synthesised here (which also makes this the cheap call).
    Same session rules as /speech: the language is the conversation's, never one the client supplies."""
    session = _conversations.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no conversation found for id={conversation_id!r}")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    language = session["language"]
    return SpeechTextResponse(text=verbalize(req.text, language), language=language)


# --- Speech for the ops dashboard ---------------------------------------------------------------------------------
# The ops dashboard (ops/api.py, a separate service on the same VM) answers spoken questions, and it borrows this
# process's speech models instead of loading its own: on a 3.8 GB VM with an OOM history, a second Whisper costs about
# 125 MB (measured: +106 MB at load, +19 MB on first use) and a second Piper voice about 140 MB, while this process
# already loads both, for borrowers' voice messages and speaker taps. The ops service sends its audio and text here
# over the loopback interface (ops/speech_client.py).
#
# Not for the internet. Both gates must pass: the ops key (OPS_API_KEY, the one secret that gates the whole ops API), and
# no X-Forwarded-For header. Caddy adds that header to every request it proxies and a client cannot take it off, so a
# request that came in through the public site is refused before its key is even looked at -- which is also why there
# is no brute-force limiter here: nothing outside this machine reaches the key check. The ops service calls
# 127.0.0.1 directly, not through Caddy, so its requests never carry one.
_OPS_VOICE_MAX_SECONDS = 30  # a question to a dashboard, not a voice note: 30 s bounds the ASR work one request can ask for
_OPS_VOICE_MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # 30 s of 16-bit mono WAV at 48 kHz is ~2.9 MB
_OPS_SPEECH_MAX_CHARS = 400


def _require_ops_voice_caller(request: Request) -> None:
    if "x-forwarded-for" in request.headers:
        raise HTTPException(status_code=403, detail="internal endpoint -- not reachable through the public site")
    expected_key = os.environ.get("OPS_API_KEY")
    if not expected_key:
        raise RuntimeError("OPS_API_KEY is not set -- copy .env.example to .env and fill it in")
    if not hmac.compare_digest(request.headers.get("x-api-key", ""), expected_key):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key header")


class OpsTranscriptOut(BaseModel):
    # None: no speech in the recording, or it ran past _OPS_VOICE_MAX_SECONDS (the same None the borrower path gets)
    transcript: str | None


@app.post("/internal/ops-voice/transcribe", response_model=OpsTranscriptOut, include_in_schema=False)
async def ops_voice_transcribe_endpoint(request: Request):
    """Speech to text for one spoken question to the ops dashboard: the request body is the WAV itself (the ops page
    converts its recording the same way the chat page does, see static/app.js's audioBufferToWav). English: v1 of ops
    voice takes questions in English only. The decode and Whisper run off the event loop, like every other blocking
    call in this process."""
    _require_ops_voice_caller(request)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > _OPS_VOICE_MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="recording too large")
    try:
        transcript = await asyncio.to_thread(
            _decode_and_transcribe_voice, bytes(raw), "en", max_seconds=_OPS_VOICE_MAX_SECONDS,
        )
    except sf.LibsndfileError as e:
        raise HTTPException(status_code=400, detail="couldn't read that recording as audio") from e
    return OpsTranscriptOut(transcript=transcript.strip() if transcript and transcript.strip() else None)


class OpsSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=_OPS_SPEECH_MAX_CHARS)


@app.post("/internal/ops-voice/speak", include_in_schema=False)
def ops_voice_speak_endpoint(request: Request, req: OpsSpeechRequest):
    """The ops dashboard's spoken answer: the same verbalize -> Piper (English) -> Opus steps as /speech above, so a
    rupee amount or an account ID is said the way it is for a borrower."""
    _require_ops_voice_caller(request)
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    try:
        audio_bytes = encode_ogg_opus(speak_english(verbalize(req.text, "en")))
    except Exception as e:
        # Same deliberately broad catch as speech_endpoint above, for the same reason (VAD/TTS/encoding fail in many
        # ways that all mean the same thing to the caller), and logged with the stack so the cause is never lost.
        logger.warning("ops voice: TTS synthesis failed (%d chars): %s", len(req.text), e, exc_info=True)
        raise HTTPException(status_code=502, detail="text-to-speech is temporarily unavailable") from e
    return Response(content=audio_bytes, media_type="audio/ogg")


class AbilityOut(BaseModel):
    id: str
    label: str
    # Exactly one is set: `prompt` is sent as the borrower's next message; `reply` is shown as it stands (no assistant turn).
    prompt: str | None = None
    reply: str | None = None


class AbilitiesResponse(BaseModel):
    language: str
    verified: bool
    intro: str
    abilities: list[AbilityOut]


@app.get("/conversations/{conversation_id}/abilities", response_model=AbilitiesResponse)
def abilities_endpoint(conversation_id: str):
    """The "what I can help with" buttons for this chat (channels/abilities.py, the same list Telegram shows). Decided by the
    session, never by the client: the language is the conversation's, and an unverified chat is only offered what it can
    actually do, so it never sees a button the assistant would have to refuse. Verification can happen mid-chat, so the page
    asks again after it does."""
    session = _conversations.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no conversation found for id={conversation_id!r}")
    language = session["language"]
    return AbilitiesResponse(
        language=language,
        verified=session["account_id"] is not None,
        intro=intro(language),
        abilities=abilities_for(language, session["account_id"] is not None),
    )


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
    disputes = store.get_disputes_for_account(account_id)
    threads = store.get_case_threads(account_id)
    documents = list_documents_for_account(account_id)
    open_claim_kind = next((d["kind"] for d in disputes if d["status"] == "open"), None)

    return DashboardResponse(
        account=AccountSnapshotOut(**payment_status),
        timeline=[TimelineEntryOut(**entry) for entry in _build_emi_timeline(account, payment_status["days_past_due"])],
        warnings=[WarningOut(**w) for w in _build_warnings(flags, payment_status, open_claim_kind)],
        requests=[RequestOut(**r) for r in _build_requests(escalations, disputes, threads)],
        documents=[DocumentOut(**d) for d in documents],
        # Everything sent to this borrower -- staff messages, restructuring
        # decisions, dispute outcomes and reminders -- previously visible to
        # them ONLY if Telegram was linked and delivery succeeded (and even
        # then, only clarification requests were ever shown here). The page
        # promises nothing sent to them is only visible on Telegram; this is
        # what makes that true, and makes this page a real channel for a
        # borrower who never linked Telegram. A message about one request is
        # in that request's thread instead (see store.get_borrower_messages).
        messages=[MessageOut(**m) for m in store.get_borrower_messages(account_id)],
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


@app.post("/conversations/{conversation_id}/quick-actions/payment-claim")
def quick_action_payment_claim_endpoint(conversation_id: str, req: QuickActionPaymentClaimRequest) -> dict:
    """"I've already paid": opens a payment claim (accounts/cases.py) for the
    team to check against the bank record, or adds these details to the one
    already open. Found live: this was a dispute ("Something looks wrong"),
    and ops couldn't tell a payment to look for from a disagreement."""
    account_id = _require_verified_account(conversation_id)
    if not req.details.strip():
        raise HTTPException(status_code=400, detail="tell us when and how you paid")
    result = flag_dispute(account_id, req.details, kind=cases.PAYMENT_CLAIM)
    _log_quick_action(
        account_id, "flag_dispute", {"account_id": account_id, "reason": req.details, "kind": cases.PAYMENT_CLAIM}, result
    )
    return result


# A borrower's notes on their own cases are stored and shown to ops, so a
# runaway client can't fill the log: a few a minute is plenty for a person.
_case_note_limiter = RateLimiter(max_requests=10, window_seconds=300)


@app.post("/conversations/{conversation_id}/cases/{case_ref}/notes")
def add_case_note_endpoint(conversation_id: str, case_ref: str, req: CaseNoteRequest) -> dict:
    """The borrower adds to one of their own open requests (a payment
    reference, an answer to our question): it appears in that request's
    thread on their dashboard and on the ops dashboard. Only for a case on
    this conversation's verified account, and only while it is open -- a
    closed one says so (409) instead of taking a note nobody will read."""
    account_id = _require_verified_account(conversation_id)
    try:
        message = validation.free_text(req.message, "message")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    case = store.get_case(account_id, case_ref)
    if case is None or case["kind"] == SYSTEM:
        raise HTTPException(status_code=404, detail=f"no request {case_ref!r} on this account")
    if not case["open"]:
        raise HTTPException(status_code=409, detail="this request is closed -- use Get help to raise something new")
    if not _case_note_limiter.check(account_id):
        raise HTTPException(status_code=429, detail="too many notes in a short time -- please wait a few minutes")
    store.add_case_message(account_id, case_ref, "borrower", message)
    return {"case_ref": case_ref, "author": "borrower", "message": message}


class MessageReplyRequest(BaseModel):
    message: str


@app.post("/conversations/{conversation_id}/messages-from-us/{message_id}/reply")
def reply_to_message_endpoint(conversation_id: str, message_id: int, req: MessageReplyRequest) -> dict:
    """The borrower answers a message from our team: it becomes a request in the ops Inbox (quoting what it answers),
    or joins that request's thread if they replied before (accounts/store.py reply_to_team_message). 404 for a
    message that is not a team message on this conversation's account."""
    account_id = _require_verified_account(conversation_id)
    try:
        text = validation.free_text(req.message, "message")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not _case_note_limiter.check(account_id):
        raise HTTPException(status_code=429, detail="too many messages in a short time -- please wait a few minutes")
    try:
        return store.reply_to_team_message(account_id, message_id, text, via="web")
    except store.MessageNotReplyableError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/conversations/{conversation_id}/quick-actions/fee-waiver")
def quick_action_fee_waiver_endpoint(conversation_id: str, req: QuickActionFeeWaiverRequest) -> dict:
    """"Remove a late fee": the same request the assistant's request_fee_waiver sends -- to a person, with the reason;
    nothing is waived here. {no_fee_owed: true} when there is no late fee to ask about."""
    account_id = _require_verified_account(conversation_id)
    try:
        result = request_fee_waiver(account_id, req.reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="tell us why you are asking") from e
    _log_quick_action(account_id, "request_fee_waiver", {"account_id": account_id, "reason": req.reason}, result)
    return result


@app.post("/conversations/{conversation_id}/quick-actions/closure-certificate")
def quick_action_closure_certificate_endpoint(conversation_id: str) -> dict:
    """"Closure certificate": the assistant's request_closure_certificate from a button. eligible=false (with the EMIs
    still left) while the loan is not fully repaid; otherwise a request in the ops Inbox for a person to issue it."""
    account_id = _require_verified_account(conversation_id)
    result = request_closure_certificate(account_id)
    _log_quick_action(account_id, "request_closure_certificate", {"account_id": account_id}, result)
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


class ChargePartOut(BaseModel):
    label: str
    amount: float


class PaymentBreakdownOut(BaseModel):
    """What one payment covers, in the order it is applied: the EMI (all of it, or part: short_of_emi > 0), late fees,
    then anything extra -- which the borrower is asked how to use."""
    to_emi: float
    emi_retired: bool
    short_of_emi: float
    to_charges: list[ChargePartOut]
    extra: float


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
    # Every late fee still owed (0 when none). A link for exactly
    # emi_amount_due + late_fee_due is a plain payment of what's owed -- the
    # pay page must not treat the fee as an "extra" to be applied somewhere.
    late_fee_due: float | None = None
    # How this payment will be applied (accounts/dues.py's allocate_payment): None for a used/expired link.
    breakdown: PaymentBreakdownOut | None = None



class PaymentConfirmRequest(BaseModel):
    apply_extra_to_next: bool | None = None
    payment_scheme: str | None = None  # "credit_next_emi" | "reduce_emi" | "reduce_tenure"


class PaymentConfirmOut(BaseModel):
    amount: float
    kind: str
    months_remaining: int
    next_emi_due_date: str
    pending_emi_credit: float
    principal_amount: float | None = None
    emi_amount: float | None = None
    late_fee_paid: float = 0.0  # >0 when the amount included the late fee (not a prepayment)
    # True when a late fee charged for this EMI was removed because the payment was dated within the grace period.
    late_fee_waived_paid_in_grace: bool = False


@app.get("/pay/{token}/info", response_model=PaymentTokenInfoOut)
def payment_token_info_endpoint(token: str):
    info = store.get_payment_token_info(token)
    if info is None:
        raise HTTPException(status_code=404, detail=f"no payment link found for token={token!r}")
    if info["status"] == "pending":
        today = store.current_date()
        store.assess_late_fees(today, info["account_id"])
        account = store.get_account_or_raise(info["account_id"])
        info["emi_amount_due"] = round(account.emi_amount - account.pending_emi_credit, 2)
        info["late_fee_due"] = charges_due(account)
        # The same split record_payment will apply (accounts/dues.py's allocate_payment), so the page says exactly what
        # this payment covers before anyone clicks Confirm.
        alloc = allocate_payment(account, float(info["amount"]), today)
        by_id = {c.id: c for c in account.charges}
        info["breakdown"] = PaymentBreakdownOut(
            to_emi=alloc.to_emi, emi_retired=alloc.emi_retired, short_of_emi=alloc.short_of_emi, extra=alloc.extra,
            to_charges=[
                ChargePartOut(label=f"Late fee for the {_borrower_date(by_id[cid].for_due_date.isoformat())} EMI", amount=amt)
                for cid, amt in alloc.to_charges
            ],
        )
    return PaymentTokenInfoOut(**info)


_payment_confirm_rate_limiter = RateLimiter(max_requests=20, window_seconds=300)


@app.post("/pay/{token}/confirm", response_model=PaymentConfirmOut)
def payment_confirm_endpoint(request: Request, token: str, req: PaymentConfirmRequest | None = None):
    client_ip = request.client.host if request.client else "unknown"
    if not _payment_confirm_rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Too many payment attempts from this location -- please wait a few minutes.")
    apply_extra_to_next = req.apply_extra_to_next if req is not None else None
    payment_scheme = req.payment_scheme if req is not None else None
    try:
        result = store.redeem_payment_token(token, apply_extra_to_next=apply_extra_to_next, payment_scheme=payment_scheme)
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
        {"tool": "record_payment", "arguments": {"token": token, "apply_extra_to_next": apply_extra_to_next, "payment_scheme": payment_scheme}, "result": result},
    )
    return PaymentConfirmOut(
        amount=result["amount"],
        kind=result["kind"],
        months_remaining=result["months_remaining"],
        next_emi_due_date=result["next_emi_due_date"],
        pending_emi_credit=result["pending_emi_credit"],
        principal_amount=result.get("principal_amount"),
        emi_amount=result.get("emi_amount"),
        late_fee_paid=result.get("late_fee_paid", 0.0),
        late_fee_waived_paid_in_grace=result.get("late_fee_waived_paid_in_grace", False),
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
    # The page names its scripts with a version (app.js?v=N); a browser that kept an old copy of this HTML kept loading
    # the old script against the new API after a deploy (found live: the Quality page read "undefined"). Always ask.
    return FileResponse(_STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
