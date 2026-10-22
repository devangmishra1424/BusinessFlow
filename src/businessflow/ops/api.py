"""HTTP API for the internal business/ops dashboard -- a separate app
from the borrower-facing chat channel (channels/browser_api.py), since
this serves a different audience (the lender's own staff) with
different data (every account at once, not one borrower's own -- which
is exactly why this needs its own gate: browser_api.py's per-account
access key can't protect a "list every account" endpoint).

Gated by a single shared secret (OPS_API_KEY in .env), sent as an
X-API-Key header -- a real, if simple, access control appropriate for a
handful of internal staff, not per-user accounts. Don't expose this
publicly as-is; a real deployment would want per-operator auth.

Run: uvicorn businessflow.ops.api:app --reload --port 8001
(the borrower-facing chat API, channels/browser_api.py, runs on the
default port 8000 -- run both side by side, they're independent apps).
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import groq
import psycopg
from docling.exceptions import ConversionError
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from businessflow.accounts import cases, store
from businessflow.accounts.db import check_database, get_connection
from businessflow.accounts.dues import amount_due_now
from businessflow.accounts.promises import Part as PromisePart
from businessflow.accounts.promises import terms_text as promise_terms_text
from businessflow.accounts.promises import validate_parts as validate_promise_parts
from businessflow.tools import validation
from businessflow.accounts.escalation_kinds import (
    BROKEN_PROMISE_PATTERN_PREFIX,
    CALLBACK,
    CHRONIC_DELINQUENCY_REASON,
    CLOSURE,
    FRAUD,
    GUARDRAIL_REASON_PREFIX,
    OPS_CALL_NUDGE_PREFIX,
    REQUEST,
    RESTRUCTURING,
    SYSTEM,
    URGENCY_RANK,
    classify_escalation,
)
from businessflow.accounts.models import Account
from businessflow.channels.credentials import build_telegram_start_payload
from businessflow.observability import metrics
from businessflow.ops import speech_client, voice, voice_reply
from businessflow.ops.flags import Flag, compute_flags
from businessflow.outbound import send as notify
from businessflow.outbound.compose import draft_clarification_message
from businessflow.outbound.risk_engine import calculate_account_risk
from businessflow.outbound.run import run_daily_pass
from businessflow.rag.extraction import extract_loan_terms
from businessflow.rate_limit import RateLimiter
from businessflow.rag.ingest import SuspiciousDocumentError, UnreadableDocumentError, extract_document_text, ingest_document
from businessflow.tools import validation
from businessflow.tools.escalation_tools import escalate_to_human

logger = logging.getLogger(__name__)

app = FastAPI(title="BusinessFlow Ops API")

# Where ops-uploaded per-account documents (signed loan agreements, KYC,
# etc.) live on disk -- the permanent source file ingest_document() parses,
# analogous to data/kb/ holding the general policy docs' source files.
_DOCUMENTS_DIR = Path(__file__).resolve().parents[3] / "data" / "documents"

# The ops dashboard's static frontend (index.html/styles.css/app.js) --
# mounted below, after every API route, so an exact-path route like
# GET /accounts is always matched before the static mount's catch-all.
_STATIC_DIR = Path(__file__).resolve().parent / "static"

# What ingest.py's Docling call actually supports (see its docstring) --
# rejecting anything else here with a clear 400 beats letting
# DocumentConverter fail obscurely deep inside ingest_document.
_ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".md"}

# A real, deliberate bound: an unbounded upload accepted into memory/disk is
# a real (if modest) DoS surface. Sized for actual documents this system
# plausibly needs to ingest -- real regulatory circulars and scanned
# multi-page agreements can exceed a 20MB cap -- while still being a real,
# bounded limit rather than no limit at all. Enforced by counting bytes
# actually read, not by trusting the client-supplied Content-Length header.
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024

# The real E.164 shape (models.py documents every account's phone_number
# as E.164 but never enforced it -- every existing value came from a
# hand-authored seed script, not user input). A new account opened
# through this API is the first real input boundary for this field, so
# it's validated here rather than left to whatever an ops person typed.
_E164_PATTERN = re.compile(r"^\+[1-9]\d{6,14}$")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_header = APIKeyHeader(name="X-API-Key")


def _telegram_invite_link(account_id: str, access_key: str) -> str | None:
    """A real, tappable t.me/<bot>?start=<payload> link that auto-verifies
    the instant a borrower taps it (see telegram_bot.py's on_start /
    _handle_start_payload) -- closes a real gap: the account-creation
    success screen used to just tell the OPERATOR to manually relay the
    key "over chat/Telegram," with no automated send and no deep link at
    all, so a borrower had to find the bot cold and type the key in by
    hand. None (not a broken link) when TELEGRAM_BOT_USERNAME isn't
    configured -- there's no sensible default bot to point at."""
    username = os.environ.get("TELEGRAM_BOT_USERNAME")
    if not username:
        return None
    payload = build_telegram_start_payload(account_id, access_key)
    return f"https://t.me/{username}?start={payload}"


# Uploads are streamed into a staging directory first, never straight over the file already on disk. A re-upload that is
# rejected (too big, unparseable, unreadable scan) or that fails during ingestion therefore leaves the previous version
# in place, on disk AND in the index; before this, the old file had already been overwritten (or truncated, for a size
# rejection), so the document the chunks came from was gone while the chunks stayed. Kept outside the per-account
# folders so the documents list never shows a half-finished upload.
_STAGING_DIR = _DOCUMENTS_DIR / ".staging"


def _install_upload(incoming: Path, saved_path: Path, staging: Path) -> Path | None:
    """Moves a fully-received upload into place, setting any previous version aside. Returns where the previous version
    was set aside (None if there was none), for _rollback_upload."""
    previous = None
    if saved_path.exists():
        previous = staging / "previous" / saved_path.name
        previous.parent.mkdir()
        os.replace(saved_path, previous)
    os.replace(incoming, saved_path)
    return previous


def _rollback_upload(saved_path: Path, previous: Path | None, account_dir: Path) -> None:
    """Undoes _install_upload: the previous version (if any) goes back, otherwise the new file is removed and the
    account folder too, if this was its first-ever upload attempt and it is now empty."""
    saved_path.unlink(missing_ok=True)
    if previous is not None:
        os.replace(previous, saved_path)
        return
    try:
        account_dir.rmdir()
    except OSError:
        pass  # the account already has other documents in it, so the folder stays


# Only a WRONG key counts as a hit -- a legitimate operator's dashboard
# auto-refreshes every 30s across several endpoints, which a limiter
# keyed on every request (right or wrong) would eventually throttle for
# real, normal use. This is specifically a brute-force guard on the one
# shared secret gating this entire portfolio's data -- there's no
# per-operator account to lock the way browser_api.py's borrower flow
# already can (see AccountLockedError), so this is the whole mitigation.
_ops_key_brute_force_limiter = RateLimiter(max_requests=5, window_seconds=300)


def require_api_key(request: Request, provided_key: str = Security(_api_key_header)) -> None:
    expected_key = os.environ.get("OPS_API_KEY")
    if not expected_key:
        raise RuntimeError("OPS_API_KEY is not set -- copy .env.example to .env and fill it in")
    if provided_key != expected_key:
        client_ip = request.client.host if request.client else "unknown"
        if not _ops_key_brute_force_limiter.check(client_ip):
            raise HTTPException(status_code=429, detail="Too many invalid API key attempts -- try again later.")
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key header")


class FlagOut(BaseModel):
    label: str
    reason: str


class RiskFactorOut(BaseModel):
    label: str
    points: float


class RiskOut(BaseModel):
    """The one risk (outbound/risk_engine.py): a 0-100 score from the account's own behaviour plus its onboarding
    rating, the tier it maps to, and every factor with its points -- the same number the reminder tone uses."""
    tier: str  # 'low' | 'medium' | 'high' | 'critical'
    score: float
    factors: list[RiskFactorOut]


class AccountSummaryOut(BaseModel):
    account_id: str
    borrower_name: str
    business_name: str
    loan_type: str
    principal_amount: float
    emi_amount: float
    days_past_due: int
    # The tier everyone sees: computed (risk below), not the stored onboarding rating, which is onboarding_risk_tier.
    risk_tier: str
    flags: list[FlagOut]
    # False means every automated reminder for this account has been
    # silently logged-but-undelivered (see outbound/send.py's
    # _deliver_and_log) since it verified nowhere -- surfaced here so ops
    # can see and re-target a borrower who was onboarded but never
    # actually connected, instead of it being invisible until someone
    # notices a due date came and went with no response.
    telegram_linked: bool
    # What the borrower owes right now (accounts/dues.py) -- the figure a payment claim's "Confirm payment" starts from.
    amount_due_now: float = 0.0
    # The rating given at onboarding (accounts.risk_tier): editable, and one input to `risk`.
    onboarding_risk_tier: str = "low"
    risk: RiskOut | None = None


class PaymentRecordOut(BaseModel):
    date: date
    amount: float
    on_time: bool
    charges_paid: float = 0.0  # how much of it went to a late fee
    kind: str  # "regular" | "extra_unapplied" | "extra_applied" | "overpayment_applied"


class PromiseOut(BaseModel):
    made_on: date
    promised_date: date
    promised_amount: float
    kept: bool | None
    # One part of a promise plan (accounts/promises.py): which plan, which part, the terms as confirmed, how it was
    # made, and its verdict -- 'kept' | 'broken' | 'void', None while open.
    plan_id: str | None = None
    part_no: int = 1
    parts_total: int = 1
    terms_text: str | None = None
    made_via: str | None = None
    resolution: str | None = None


class PromisePartIn(BaseModel):
    date: date
    amount: float = Field(gt=0)


class DisputeOut(BaseModel):
    id: int | None = None
    kind: str = "dispute"  # 'dispute' | 'payment_claim' (accounts/cases.py)
    reason: str
    status: str  # 'open' | 'resolved'
    opened_at: datetime
    resolved_at: datetime | None
    resolution_note: str | None = None


class MetricsOut(BaseModel):
    since_hours: float
    event_counts: dict[str, int]
    escalation_rate: float


class EscalationOut(BaseModel):
    escalation_id: str
    account_id: str
    reason: str
    status: str
    created_at: datetime
    resolved_at: datetime | None
    # Structured proposed terms (see tools/escalation_tools.py's
    # propose_restructuring) -- None for every other escalation kind.
    proposed_changes: dict | None = None
    resolution_reason: str | None = None
    # restructuring | closure | callback | fraud | request | system (see
    # accounts/escalation_kinds.py) -- display/notification wording and queue
    # triage only; best-effort, never used for anything that touches an account.
    kind: str = "request"
    # Set only on the response to approve/reject: True = reached over Telegram,
    # False = logged but the borrower has no linked Telegram, None = nothing
    # was sent (system-raised tickets never message the borrower).
    borrower_notified: bool | None = None


class EscalationRejectIn(BaseModel):
    # Optional, ops-entered explanation shown back to the borrower.
    reason: str | None = None


class RecordPaymentIn(BaseModel):
    amount: float
    # Defaults to today (store.record_payment's own default) if omitted --
    # only given explicitly for a payment ops is entering after the fact
    # (e.g. an NEFT credit confirmed a day later than it actually landed).
    payment_date: date | None = None
    # Only meaningful when amount is less than what's actually due this
    # cycle -- see store.record_payment's own docstring for the full
    # decision table. Left None for a normal/full payment, which never
    # needs it.
    apply_extra_to_next: bool | None = None

    @field_validator("amount")
    @classmethod
    def _validate_positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be a positive amount")
        return v


class RecordPaymentOut(BaseModel):
    account_id: str
    amount: float
    payment_date: date
    on_time: bool
    kind: str  # "regular" | "extra_unapplied" | "extra_applied" | "overpayment_applied"
    months_remaining: int
    next_emi_due_date: date
    pending_emi_credit: float
    late_fee_paid: float = 0.0  # >0 when the amount included the late fee (not a prepayment)
    charges_paid: float = 0.0  # how much of this payment went to charges (accounts/dues.py's allocate_payment)
    # A late fee charged for this EMI was removed: the payment is dated within the grace period.
    late_fee_waived_paid_in_grace: bool = False


class ChargeOut(BaseModel):
    """A charge on the account (schema.sql's charges table): today only the late fee."""
    id: int
    kind: str
    for_due_date: date
    amount: float
    amount_paid: float
    outstanding: float
    status: str  # "due" | "paid" | "waived"
    assessed_on: date
    paid_on: date | None = None
    waived_on: date | None = None
    waived_by: str | None = None
    waived_reason: str | None = None


class WaiveChargeIn(BaseModel):
    """Who is waiving it and why -- both required: a waiver is a recorded decision, and the reason is sent to the
    borrower. case_ref (optional) puts the borrower's message in that case's thread (e.g. the dispute about the fee)."""
    waived_by: str = Field(min_length=2, max_length=80)
    reason: str = Field(min_length=3, max_length=500)
    case_ref: str | None = Field(default=None, max_length=40)


class WaiveChargeOut(BaseModel):
    charge: ChargeOut
    borrower_notified: bool


def _charge_out(c) -> ChargeOut:
    return ChargeOut(
        id=c.id, kind=c.kind, for_due_date=c.for_due_date, amount=c.amount, amount_paid=c.amount_paid,
        outstanding=c.outstanding, status=c.status, assessed_on=c.assessed_on, paid_on=c.paid_on,
        waived_on=c.waived_on, waived_by=c.waived_by, waived_reason=c.waived_reason,
    )


class ResolveDisputeIn(BaseModel):
    # Ops-entered free text on how/why this was resolved -- optional for a
    # dispute, since not every resolution needs one, but never lost when
    # given. Required to close a payment claim without recording a payment:
    # the borrower is owed the reason.
    resolution_note: str | None = None
    # The case the operator is looking at (the case card always sends it): it
    # must still be the open one, see store.resolve_dispute.
    dispute_id: int | None = None


class ResolveDisputeOut(BaseModel):
    account_id: str
    dispute_id: int
    kind: str = "dispute"
    resolution_note: str | None = None
    borrower_notified: bool | None = None  # True = reached over Telegram; False = logged only (visible on their dashboard)


class LogPromiseIn(BaseModel):
    promised_date: date
    promised_amount: float
    # Later parts of the same promise, if the borrower promised in parts ("10,000 by the 25th, the rest by the 2nd").
    more_parts: list[PromisePartIn] = Field(default_factory=list, max_length=3)
    # Defaults to today if omitted -- an operator logging a promise made
    # during a call happening right now, the common case.
    made_on: date | None = None

    @field_validator("promised_amount")
    @classmethod
    def _validate_positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("promised_amount must be a positive amount")
        return v


class LogPromiseOut(BaseModel):
    account_id: str
    made_on: date
    promised_date: date
    promised_amount: float
    created: bool  # False if this exact promise was already logged (see store.add_promise_plan)
    plan_id: str | None = None
    terms: str | None = None
    replaced_earlier_promise: bool = False


class CallLogIn(BaseModel):
    outcome: Literal["reached", "no_answer", "voicemail", "wrong_number"]
    note: str | None = None


class CallLogEntryOut(BaseModel):
    outcome: str
    note: str | None
    created_at: datetime


class DocumentUploadOut(BaseModel):
    account_id: str
    document_type: str
    filename: str
    chunks_stored: int
    # True only if a real, non-null interest_rate_pct was written to the
    # account row this call -- lets ops see at a glance whether structured
    # extraction actually found something, without a separate account query.
    interest_rate_extracted: bool


class DocumentOut(BaseModel):
    filename: str
    size_bytes: int
    uploaded_at: datetime


class ClarificationDraftIn(BaseModel):
    operator_note: str


class ClarificationDraftOut(BaseModel):
    draft: str


class ClarificationRequestIn(BaseModel):
    # The final message text an operator has reviewed and wants sent --
    # possibly the LLM-drafted wording, possibly hand-edited or written
    # from scratch. Never auto-sent from a draft without this explicit call.
    message: str


class ClarificationRequestOut(BaseModel):
    message: str
    delivered_via_telegram: bool
    created_at: datetime
    # True if sent before this account's most recent "mark all resolved"
    # checkpoint -- see store.get_clarification_requests's own docstring
    # for why this is a watermark, not per-message state.
    resolved: bool


class EscalateToHumanIn(BaseModel):
    reason: str


class RecentPaymentOut(BaseModel):
    amount: float
    payment_date: date
    recorded_at: datetime


class CaseMessageOut(BaseModel):
    author: str  # "borrower" | "ops"
    message: str
    created_at: datetime
    delivered_via_telegram: bool = False


class CaseOut(BaseModel):
    """One case (accounts/cases.py) as ops works it: what the borrower said,
    every message about it since, and how it ended -- the same thing the
    borrower sees as one thread on their dashboard."""

    case_ref: str  # "D-12" | "ESC-0012"
    source: str  # "dispute" | "escalation"
    kind: str  # payment_claim | dispute | fraud | request | restructuring | callback | closure | system
    title: str
    account_id: str
    borrower_name: str | None = None
    open: bool
    status: str  # open | resolved | approved | rejected
    opened_at: datetime
    closed_at: datetime | None = None
    borrower_words: str  # the dispute's reason, or the escalation's reason (written by the agent for the borrower)
    proposed_changes: dict | None = None
    outcome: str | None = None  # ops' note on how it ended
    thread: list[CaseMessageOut] = []
    can_reply: bool = True  # False for a system notice: the borrower never sees it, so a reply would go nowhere
    # Payment claims only: payments already recorded on the account in the last few days, shown on the claim so a
    # payment that is already in is never confirmed a second time.
    recent_payments: list[RecentPaymentOut] = []


class CaseReplyIn(BaseModel):
    message: str


class CaseReplyOut(CaseMessageOut):
    case_ref: str


class ConfirmPaymentIn(BaseModel):
    amount: float
    # The day the borrower actually paid (decides on time or late) -- defaults to today if omitted.
    payment_date: date | None = None
    # Same as RecordPaymentIn: only needed when the amount is less than what is due this cycle.
    apply_extra_to_next: bool | None = None
    note: str | None = None  # optional, added to the message the borrower gets
    # Set only after the 409 "possible_duplicate" answer, when ops has checked it is a second, separate payment.
    record_even_if_duplicate: bool = False

    @field_validator("amount")
    @classmethod
    def _validate_positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be a positive amount")
        return v


class ConfirmPaymentOut(BaseModel):
    case_ref: str
    payment: RecordPaymentOut
    borrower_notified: bool


class AccountMessageOut(BaseModel):
    """Something sent to the borrower that isn't about one case (a reminder, a
    general message from ops) -- the profile's timeline lists these."""

    message: str
    kind: str  # message | decision | dispute | reminder
    delivered_via_telegram: bool
    created_at: datetime


class AccountCreateIn(BaseModel):
    borrower_name: str
    business_name: str
    phone_number: str
    language_preference: Literal["hi", "en", "hinglish"]
    loan_type: str
    principal_amount: float
    emi_amount: float
    tenure_months: int
    emi_due_date: date
    nach_mandate_active: bool = True
    risk_tier: Literal["low", "medium", "high"] = "low"
    # Only set when opened via the ops UI's EMI calculator (see
    # store.set_interest_rate_pct, already used elsewhere for the exact
    # same column via document-extracted rates) -- a manually-entered
    # account has no rate to derive, so this stays None for those.
    interest_rate_pct: float | None = None

    @field_validator("borrower_name", "business_name", "loan_type")
    @classmethod
    def _validate_name_text(cls, v: str, info) -> str:
        return validation.short_name(v, info.field_name)

    @field_validator("phone_number")
    @classmethod
    def _validate_e164(cls, v: str) -> str:
        if not _E164_PATTERN.match(v):
            raise ValueError(f"phone_number must be E.164 (e.g. +919812345001), got {v!r}")
        return v

    @field_validator("principal_amount", "emi_amount")
    @classmethod
    def _validate_positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be a positive amount")
        return v

    @field_validator("tenure_months")
    @classmethod
    def _validate_positive_tenure(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("tenure_months must be positive")
        return v


class AccountCreateOut(BaseModel):
    account: AccountSummaryOut
    # Handed back exactly once, at creation -- see accounts.store.create_
    # account's docstring for why this isn't recoverable any other way.
    access_key: str
    # None if TELEGRAM_BOT_USERNAME isn't configured -- see
    # _telegram_invite_link's own docstring.
    telegram_invite_link: str | None = None


class ResetAccessKeyOut(BaseModel):
    account_id: str
    access_key: str
    telegram_invite_link: str | None = None


class AccountUpdateIn(BaseModel):
    # Fields legitimately correctable after an account already exists --
    # not principal/EMI/tenure, which real payment/restructuring flows own.
    phone_number: str | None = None
    language_preference: Literal["hi", "en", "hinglish"] | None = None
    risk_tier: Literal["low", "medium", "high"] | None = None

    @field_validator("phone_number")
    @classmethod
    def _validate_e164(cls, v: str | None) -> str | None:
        if v is not None and not _E164_PATTERN.match(v):
            raise ValueError(f"phone_number must be E.164 (e.g. +919812345001), got {v!r}")
        return v


class TranscriptEntryOut(BaseModel):
    event_type: str  # "user_message" | "assistant_message" | "tool_called"
    content: str | None = None  # user_message / assistant_message
    tool: str | None = None  # tool_called
    arguments: dict | None = None  # tool_called
    created_at: datetime


class RunOutboundIn(BaseModel):
    # None = every account currently due -- matches
    # outbound.run.run_daily_pass's own account_ids contract.
    account_ids: list[str] | None = None


class RunOutboundOut(BaseModel):
    promises_resolved: int
    escalated_for_broken_promises: int
    reminders_sent: list[dict]


class AccountDetailOut(AccountSummaryOut):
    # Every charge assessed on the account (late fees), oldest first, with what happened to each.
    charges: list[ChargeOut] = []
    phone_number: str
    language_preference: str
    tenure_months: int
    months_remaining: int
    emi_due_date: date
    nach_mandate_active: bool
    dispute_open: bool
    pending_emi_credit: float
    payment_history: list[PaymentRecordOut]
    promises: list[PromiseOut]
    disputes: list[DisputeOut]
    escalations: list[EscalationOut]
    clarification_requests: list[ClarificationRequestOut]
    call_log: list[CallLogEntryOut]
    # Every case on the account, open first (each with its thread), and everything else sent to the borrower -- the
    # profile's Cases tab. disputes / escalations / clarification_requests above stay for API callers.
    cases: list[CaseOut] = []
    messages: list[AccountMessageOut] = []


# Triage order for the inbox, lowest first: escalation_kinds.URGENCY_RANK, with a payment claim or dispute next to an
# ordinary request (reminders are paused on the account while either is open, so neither should sit at the bottom).
_CASE_URGENCY = {**URGENCY_RANK, cases.PAYMENT_CLAIM: 1, cases.DISPUTE: 1}
_IST = timezone(timedelta(hours=5, minutes=30), "IST")

_CASE_TITLES = {
    cases.PAYMENT_CLAIM: "Says they already paid",
    cases.DISPUTE: "Dispute",
    FRAUD: "Identity / fraud claim",
    RESTRUCTURING: "Asks for more time",
    CALLBACK: "Asks to talk to a person",
    CLOSURE: "Closure certificate",
    REQUEST: "Request",
}
# A system notice says which pattern raised it, so the inbox can tell a broken-promise pattern from a guardrail block.
_SYSTEM_TITLES = (
    (BROKEN_PROMISE_PATTERN_PREFIX, "Broken promises"),
    (CHRONIC_DELINQUENCY_REASON, "Chronically overdue"),
    (GUARDRAIL_REASON_PREFIX, "Agent reply blocked"),
    (OPS_CALL_NUDGE_PREFIX, "Couldn't reach by phone"),
)
# What a message from ops is about, in the borrower's words -- Telegram has no thread to show it in.
_CASE_ABOUT = {
    cases.PAYMENT_CLAIM: "the payment you reported",
    cases.DISPUTE: "the issue you raised",
    FRAUD: "your concern about this loan",
    RESTRUCTURING: "your request for more time",
    CALLBACK: "your request to talk to us",
    CLOSURE: "your closure certificate",
    REQUEST: "your request",
}


def _case_title(kind: str, reason: str) -> str:
    if kind == SYSTEM:
        return next((title for prefix, title in _SYSTEM_TITLES if (reason or "").startswith(prefix)), "System notice")
    return _CASE_TITLES.get(kind, "Request")


def _thread_out(thread: list[dict]) -> list[CaseMessageOut]:
    return [CaseMessageOut(**{k: m[k] for k in ("author", "message", "created_at", "delivered_via_telegram")}) for m in thread]


def _dispute_case(
    d: dict, account_id: str, borrower_name: str | None, threads: dict[str, list[dict]], recent: dict[str, list[dict]] | None = None,
) -> CaseOut:
    case_ref = cases.dispute_case_ref(d["id"])
    is_open = d["status"] == "open"
    claim_payments = (recent or {}).get(account_id, []) if d["kind"] == cases.PAYMENT_CLAIM and is_open else []
    return CaseOut(
        recent_payments=[RecentPaymentOut(**{k: p[k] for k in ("amount", "payment_date", "recorded_at")}) for p in claim_payments],
        case_ref=case_ref, source="dispute", kind=d["kind"], title=_case_title(d["kind"], d["reason"]),
        account_id=account_id, borrower_name=borrower_name, open=is_open, status="open" if is_open else "resolved",
        opened_at=d["opened_at"], closed_at=d.get("resolved_at"), borrower_words=d["reason"],
        outcome=d.get("resolution_note"), thread=_thread_out(threads.get(case_ref, [])),
    )


def _escalation_case(e, borrower_name: str | None, threads: dict[str, list[dict]]) -> CaseOut:
    kind = classify_escalation(e.reason, e.proposed_changes)
    is_open = e.resolved_at is None
    return CaseOut(
        case_ref=e.escalation_id, source="escalation", kind=kind, title=_case_title(kind, e.reason),
        account_id=e.account_id, borrower_name=borrower_name, open=is_open,
        status="open" if is_open else e.status, opened_at=e.created_at, closed_at=e.resolved_at,
        borrower_words=e.reason, proposed_changes=e.proposed_changes, outcome=e.resolution_reason,
        thread=_thread_out(threads.get(e.escalation_id, [])), can_reply=kind != SYSTEM,
    )


def _sorted_cases(items: list[CaseOut]) -> list[CaseOut]:
    """Open cases first, most urgent then oldest; then closed ones, most recently closed first."""
    open_cases = sorted((c for c in items if c.open), key=lambda c: (_CASE_URGENCY.get(c.kind, 1), c.opened_at))
    closed = sorted((c for c in items if not c.open), key=lambda c: c.closed_at or c.opened_at, reverse=True)
    return open_cases + closed


def _summarize(account: Account, flags: list[Flag]) -> AccountSummaryOut:
    profile = calculate_account_risk(account, store.current_date())
    return AccountSummaryOut(
        account_id=account.account_id,
        borrower_name=account.borrower_name,
        business_name=account.business_name,
        loan_type=account.loan_type,
        principal_amount=account.principal_amount,
        emi_amount=account.emi_amount,
        days_past_due=account.days_past_due(store.current_date()),
        risk_tier=profile.tier.value,
        flags=[FlagOut(label=f.label, reason=f.reason) for f in flags],
        telegram_linked=account.telegram_chat_id is not None,
        amount_due_now=amount_due_now(account, store.current_date()),
        onboarding_risk_tier=account.risk_tier,
        risk=RiskOut(tier=profile.tier.value, score=profile.score,
                     factors=[RiskFactorOut(**f) for f in profile.breakdown]),
    )


@app.get("/health")
def health():
    # Same contract as the chat API's /health (channels/browser_api.py): 503
    # when the database is unreachable, never the error text.
    try:
        check_database()
    except (psycopg.Error, RuntimeError) as exc:
        logger.error("health check failed: database unreachable (%s) -- check the Supabase project isn't paused and DATABASE_URL is right", type(exc).__name__)
        return JSONResponse(status_code=503, content={"status": "degraded", "database": "unreachable"})
    return {"status": "ok", "database": "ok"}


@app.get("/accounts", response_model=list[AccountSummaryOut], dependencies=[Depends(require_api_key)])
def list_accounts(flag: str | None = None):
    """The portfolio overview. Pass ?flag=overdue|disputed|broken_promises
    to see only accounts carrying that flag -- the queue an ops person
    would actually work from, not the full unfiltered list every time."""
    # Three queries for the accounts and one for the dispute reasons, however many accounts there are (store.list_accounts
    # explains why per-account reads were too slow: 12 s for 10 accounts).
    store.assess_late_fees()  # one statement: every fee due today is on the books before anything is shown
    accounts = store.list_accounts()
    reasons = store.latest_open_dispute_reasons([a.account_id for a in accounts if a.dispute_open])
    summaries = [_summarize(a, compute_flags(a, open_dispute_reasons=reasons)) for a in accounts]
    if flag is not None:
        summaries = [s for s in summaries if any(f.label == flag for f in s.flags)]
    return summaries


@app.post("/accounts", response_model=AccountCreateOut, status_code=201, dependencies=[Depends(require_api_key)])
def create_account(body: AccountCreateIn):
    """Opens a brand-new loan account -- the ops-side equivalent of a real
    sign-up flow this system doesn't otherwise have (see accounts.store.
    create_account). A freshly created account has no payment history, no
    promises, and no flags yet -- it enters the portfolio exactly the way
    a real newly-issued loan would, and only accumulates real activity
    from here."""
    account, access_key = store.create_account(
        borrower_name=body.borrower_name,
        business_name=body.business_name,
        phone_number=body.phone_number,
        language_preference=body.language_preference,
        loan_type=body.loan_type,
        principal_amount=body.principal_amount,
        emi_amount=body.emi_amount,
        tenure_months=body.tenure_months,
        emi_due_date=body.emi_due_date,
        nach_mandate_active=body.nach_mandate_active,
        risk_tier=body.risk_tier,
    )
    if body.interest_rate_pct is not None:
        store.set_interest_rate_pct(account.account_id, body.interest_rate_pct)
        account = store.get_account_or_raise(account.account_id)
    return AccountCreateOut(
        account=_summarize(account, compute_flags(account)),
        access_key=access_key,
        telegram_invite_link=_telegram_invite_link(account.account_id, access_key),
    )


@app.post(
    "/accounts/{account_id}/reset-access-key",
    response_model=ResetAccessKeyOut,
    dependencies=[Depends(require_api_key)],
)
def reset_account_access_key(account_id: str):
    """Mints a fresh access key (and, if configured, a fresh Telegram
    invite link) for an account that already exists -- the only way to
    recover a lost key or get a deep link for an account opened before
    that feature existed, since store.create_account only ever hands the
    key back once. The OLD key stops working immediately."""
    if store.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
    access_key = store.reset_access_key(account_id)
    return ResetAccessKeyOut(
        account_id=account_id, access_key=access_key,
        telegram_invite_link=_telegram_invite_link(account_id, access_key),
    )


@app.get("/accounts/{account_id}", response_model=AccountDetailOut, dependencies=[Depends(require_api_key)])
def get_account(account_id: str):
    # Every read for this page on one connection: about 8 reads, each of which used to check out (and ping) its own
    # pooled connection, took 5 s on the live server (see store.py, the note above _payment_record).
    with get_connection().transaction() as conn:
        store.assess_late_fees(store.current_date(), account_id, ex=conn)
        store.resolve_promises(store.current_date(), account_id, ex=conn)
        account = store.get_account(account_id, ex=conn)
        if account is None:
            raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
        reasons = {account_id: store.get_latest_open_dispute_reason(account_id, ex=conn)} if account.dispute_open else {}
        summary = _summarize(account, compute_flags(account, open_dispute_reasons=reasons))
        escalations = store.get_escalations_for_account(account_id, ex=conn)
        disputes = store.get_disputes_for_account(account_id, ex=conn)
        clarification_requests = store.get_clarification_requests(account_id, ex=conn)
        call_log = store.get_call_log(account_id, ex=conn)
        threads = store.get_case_threads(account_id, ex=conn)
        messages = store.get_borrower_messages(account_id, ex=conn)
        recent = store.recent_payments([account_id], ex=conn) if account.dispute_open else {}
    case_list = _sorted_cases(
        [_dispute_case(d, account_id, account.borrower_name, threads, recent) for d in disputes]
        + [_escalation_case(e, account.borrower_name, threads) for e in escalations]
    )
    return AccountDetailOut(
        **summary.model_dump(),
        charges=[_charge_out(c) for c in account.charges],
        phone_number=account.phone_number,
        language_preference=account.language_preference,
        tenure_months=account.tenure_months,
        months_remaining=account.months_remaining,
        emi_due_date=account.emi_due_date,
        nach_mandate_active=account.nach_mandate_active,
        dispute_open=account.dispute_open,
        pending_emi_credit=account.pending_emi_credit,
        payment_history=[
            PaymentRecordOut(date=p.date, amount=p.amount, on_time=p.on_time, kind=p.kind, charges_paid=p.charges_paid)
            for p in account.payment_history
        ],
        promises=[
            PromiseOut(made_on=p.made_on, promised_date=p.promised_date, promised_amount=p.promised_amount, kept=p.kept,
                       plan_id=p.plan_id, part_no=p.part_no, parts_total=p.parts_total, terms_text=p.terms_text,
                       made_via=p.made_via, resolution=p.resolution)
            for p in account.promises
        ],
        escalations=[
            EscalationOut(
                escalation_id=e.escalation_id, account_id=e.account_id, reason=e.reason, status=e.status,
                created_at=e.created_at, resolved_at=e.resolved_at,
                proposed_changes=e.proposed_changes, resolution_reason=e.resolution_reason,
            )
            for e in escalations
        ],
        disputes=[DisputeOut(**d) for d in disputes],
        clarification_requests=[ClarificationRequestOut(**c) for c in clarification_requests],
        call_log=[CallLogEntryOut(**c) for c in call_log],
        cases=case_list,
        messages=[AccountMessageOut(**m) for m in messages],
    )


@app.patch("/accounts/{account_id}", response_model=AccountSummaryOut, dependencies=[Depends(require_api_key)])
def update_account(account_id: str, body: AccountUpdateIn):
    """Fixes a typo'd phone number or reclassifies risk tier -- found live:
    there was no way to correct either after an account was created at
    all. Only the fields actually given are touched; a wrong phone number
    today means the borrower is genuinely uncontactable with no UI fix,
    which this closes."""
    if store.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="at least one field must be given")
    store.update_account_fields(account_id, **fields)
    account = store.get_account_or_raise(account_id)
    return _summarize(account, compute_flags(account))


@app.get(
    "/accounts/{account_id}/conversation",
    response_model=list[TranscriptEntryOut],
    dependencies=[Depends(require_api_key)],
)
def get_account_conversation(account_id: str):
    """What the AI agent actually said to this borrower, and what it did
    on their behalf -- not just messages ops itself sent. Found live: this
    was the one real gap between "monitor" and "watch numbers change" --
    staff could see payment history and flags but never the conversation
    that produced them."""
    if not store.account_exists(account_id):
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
    entries = store.get_conversation_transcript(account_id)
    return [
        TranscriptEntryOut(
            event_type=e["event_type"],
            content=e["details"].get("content"),
            tool=e["details"].get("tool"),
            arguments=e["details"].get("arguments"),
            created_at=e["created_at"],
        )
        for e in entries
    ]


@app.post("/outbound/run", response_model=RunOutboundOut, dependencies=[Depends(require_api_key)])
def trigger_outbound_run(body: RunOutboundIn):
    """Staff-triggered version of scripts/run_outbound_pass.py -- turns
    what used to be a developer-only terminal command into something ops
    can actually press, optionally scoped to a specific set of accounts
    (e.g. the currently filtered/selected ones) instead of always
    everyone. run_daily_pass's own idempotency (reminder_sent event-log
    dedup, see outbound/run.py) makes pressing this twice for the same
    account on the same day a safe no-op, not a double-send."""
    result = run_daily_pass(body.account_ids)
    return RunOutboundOut(
        promises_resolved=len(result["promises_resolved"]),
        escalated_for_broken_promises=len(result["escalated_for_broken_promises"]),
        reminders_sent=result["reminders_sent"],
    )


@app.post(
    "/accounts/{account_id}/documents",
    response_model=DocumentUploadOut,
    dependencies=[Depends(require_api_key)],
)
async def upload_account_document(
    account_id: str, file: UploadFile = File(...), document_type: str = Form(...)
):
    """Ops uploads one document for a specific borrower (a signed loan
    agreement, KYC, etc.) -- saved to disk under data/documents/{account_id}/
    and ingested into the same RAG pipeline scripts/seed_kb.py uses for the
    general policy KB (ingest_document), scoped via account_id so it's only
    ever retrievable for this borrower (see retriever.py's allowed_scopes,
    already tested in test_retriever.py).

    Visibility: the document is in the database the instant this call
    returns, and the borrower-facing processes (channels/browser_api.py,
    channels/telegram_bot.py) pick it up within
    policy_tools._FINGERPRINT_CHECK_INTERVAL_SECONDS (15 s): each one
    compares a cheap fingerprint of the chunk table against the one its
    snapshot was built under and rebuilds when they differ. Not instant,
    and no pub/sub; a bounded, checked delay.

    Fails with 422 when the file has no readable text (a blank page, a
    scan too poor for OCR, a Hindi/Devanagari scan -- the bundled OCR
    reads English), and leaves any previous version of the same filename
    exactly as it was.
    """
    account = store.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")

    # Only the basename -- a client-supplied filename is untrusted input,
    # and stripping any directory components keeps the save path confined
    # to data/documents/{account_id}/ instead of wherever "../../x" points.
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in _ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported file extension {extension!r} -- must be one of "
                f"{sorted(_ALLOWED_DOCUMENT_EXTENSIONS)}"
            ),
        )

    account_dir = _DOCUMENTS_DIR / account_id
    # Overwrite-by-filename is deliberate: re-uploading a corrected version
    # of the same document is the expected case, and ingest_document()
    # already handles safe re-ingestion at the same file_path.
    saved_path = account_dir / filename

    staging = _STAGING_DIR / uuid.uuid4().hex
    staging.mkdir(parents=True)
    incoming = staging / filename
    try:
        size = 0
        with open(incoming, "wb") as out:
            while chunk := await file.read(_UPLOAD_READ_CHUNK_BYTES):
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit",
                    )
                out.write(chunk)

        account_dir.mkdir(parents=True, exist_ok=True)
        previous = _install_upload(incoming, saved_path, staging)
        try:
            # Found live: this ran directly on the event loop despite the
            # route being `async def` -- ingest_document does real CPU-bound
            # work (docling parsing, local embedding), and the extraction
            # step below makes a genuinely blocking, synchronous Groq HTTP
            # call. Either one running in-loop blocks EVERY other request
            # this single server process is handling (this app has no other
            # worker to fall back to) until it returns -- confirmed live: a
            # slow/rate-limited extraction call froze the entire ops API,
            # including plain GETs, until the process was restarted.
            # asyncio.to_thread moves the blocking work off the event loop
            # without changing ingest_document/extract_loan_terms themselves.
            chunks_stored = await asyncio.to_thread(
                ingest_document, str(saved_path), document_type=document_type, account_id=account_id
            )
        except (ConversionError, UnreadableDocumentError, SuspiciousDocumentError) as e:
            # ConversionError: the extension passed our allow-list check, but the bytes behind it aren't a real,
            # parseable document of that type (corrupt file, wrong content under this extension). Docling raises its
            # own error for exactly this -- caught specifically, not via a bare `except Exception`, so a genuinely
            # unexpected failure still propagates as a 500 instead of being misreported as "bad input".
            # UnreadableDocumentError: it parsed, but nothing usable came out (a blank page, a scan the OCR cannot
            # read, Hindi script). SuspiciousDocumentError: it contains text written to instruct an AI (indirect prompt
            # injection). In every case ingest_document() wrote nothing to the index, and the previous version of this
            # document, if any, is put back.
            _rollback_upload(saved_path, previous, account_dir)
            detail = f"could not parse this file as {extension}: {e}" if isinstance(e, ConversionError) else str(e)
            raise HTTPException(status_code=422, detail=detail) from e
        except Exception:
            # Ingestion is one transaction, so the index still holds the previous version; keep the disk in step.
            _rollback_upload(saved_path, previous, account_dir)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # best effort: only a leftover staging folder if this fails

    # Structured extraction only applies to a signed loan agreement --
    # there's no interest rate to find in a KYC document, say. Best-effort
    # on top of an ingestion that already succeeded above: the upload's
    # real contract ("the document is in RAG") is already satisfied by
    # this point, regardless of whether this extra step finds anything.
    interest_rate_extracted = False
    if document_type == "loan_agreement":
        try:
            # Same reasoning as the ingest_document call above -- this
            # makes a real, synchronous Groq HTTP call that can block the
            # whole server process for as long as Groq takes to respond
            # (including a slow rate-limit retry), not just this request.
            document_text = await asyncio.to_thread(extract_document_text, str(saved_path))
            terms = await asyncio.to_thread(extract_loan_terms, document_text)
            rate = terms.get("interest_rate_pct")
            if rate is not None:
                store.set_interest_rate_pct(account_id, rate)
                interest_rate_extracted = True
        except Exception:
            # Deliberately broad, with the reason spelled out: this can
            # fail from a genuine Groq API error (e.g. a rate limit), a
            # docling parse failure re-reading the same file, or anything
            # else in this best-effort enhancement layer -- none of which
            # may fail an upload whose real work (RAG ingestion) already
            # succeeded above. Logged loudly (exc_info) so it's visible to
            # ops, not silently lost -- just not raised to the caller.
            logger.warning(
                "upload_account_document: interest-rate extraction failed for "
                "account_id=%r, filename=%r -- document is already ingested "
                "into RAG regardless; continuing without a structured rate.",
                account_id, filename, exc_info=True,
            )

    return DocumentUploadOut(
        account_id=account_id,
        document_type=document_type,
        filename=filename,
        chunks_stored=chunks_stored,
        interest_rate_extracted=interest_rate_extracted,
    )


@app.get(
    "/accounts/{account_id}/documents",
    response_model=list[DocumentOut],
    dependencies=[Depends(require_api_key)],
)
def list_account_documents(account_id: str):
    """Every document ops has uploaded for this borrower via
    upload_account_document -- the read half of that endpoint's write,
    newest first so the most recently handled document is what an
    operator sees first."""
    if not store.account_exists(account_id):
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")

    account_dir = _DOCUMENTS_DIR / account_id
    if not account_dir.is_dir():
        # A real account that just hasn't had anything uploaded for it yet
        # -- not a missing account, so an empty list rather than a 404.
        # account_dir only ever comes into existence via upload_account_
        # document's own mkdir(parents=True, exist_ok=True), on its first
        # successful upload.
        return []

    documents = []
    for path in account_dir.iterdir():
        if not path.is_file():
            continue
        stat = path.stat()
        documents.append(
            DocumentOut(
                filename=path.name,
                size_bytes=stat.st_size,
                uploaded_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    documents.sort(key=lambda d: d.uploaded_at, reverse=True)
    return documents


@app.get(
    "/accounts/{account_id}/documents/{filename}",
    dependencies=[Depends(require_api_key)],
)
def download_account_document(account_id: str, filename: str):
    """Serves one previously-uploaded document for download -- the other
    read half of upload_account_document's write.

    SECURITY: filename is untrusted client input, same as the upload
    endpoint's own file.filename. That endpoint gets away with just
    Path(...).name because it only ever WRITES into a directory it just
    mkdir'd itself -- there's nothing outside account_dir for a stripped
    name to collide with. This endpoint instead reads from a directory
    that may already contain whatever the filesystem has in it, so
    stripping to .name alone isn't quite enough: Path("..").name is
    literally ".." (unlike "../../x", whose .name is "x"), so a bare ".."
    survives that strip and would resolve to the account's *parent*
    directory. The real guarantee here is the resolve()+is_relative_to()
    check below -- confining the resolved path to inside account_dir
    strictly, regardless of what .name did or didn't strip.
    """
    account = store.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")

    account_dir = (_DOCUMENTS_DIR / account_id).resolve()
    requested_name = Path(filename).name
    file_path = (account_dir / requested_name).resolve()
    if not file_path.is_relative_to(account_dir) or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"no document found for filename={filename!r}")

    return FileResponse(file_path, filename=requested_name)


@app.post(
    "/accounts/{account_id}/clarification-requests/draft",
    response_model=ClarificationDraftOut,
    dependencies=[Depends(require_api_key)],
)
def draft_clarification(account_id: str, body: ClarificationDraftIn):
    """A wording suggestion only -- grounded in the account's real,
    current flags plus the operator's own note (see outbound/compose.py).
    Nothing is sent or logged here; the operator reviews/edits the draft
    and calls POST .../clarification-requests separately to actually send."""
    account = store.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")

    flags = compute_flags(account)
    draft = draft_clarification_message(
        borrower_name=account.borrower_name,
        business_name=account.business_name,
        flag_reasons=[f.reason for f in flags],
        operator_note=body.operator_note,
    )
    return ClarificationDraftOut(draft=draft)


@app.post(
    "/accounts/{account_id}/clarification-requests",
    response_model=ClarificationRequestOut,
    dependencies=[Depends(require_api_key)],
)
async def send_clarification_request(account_id: str, body: ClarificationRequestIn):
    """The real send: message is whatever the operator has already
    reviewed (LLM-drafted, hand-edited, or written from scratch) --
    delivered over Telegram if the account has a linked chat, logged
    either way (see outbound/send.py's notify_clarification_request)."""
    account = store.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")

    await notify.notify_clarification_request(account_id, body.message)
    # Re-fetch the just-logged event for its authoritative created_at
    # rather than trust client-side time, same pattern as approve/reject.
    latest = store.get_clarification_requests(account_id)[0]
    return ClarificationRequestOut(**latest)


class BulkClarificationIn(BaseModel):
    account_ids: list[str]
    message: str


class BulkClarificationOut(BaseModel):
    sent_to: list[str]
    delivered_via_telegram: list[str]
    not_found: list[str]


# An explicit, bounded cap on a single bulk send -- an ops person picking a
# whole filtered view (e.g. "everyone overdue") is a real, expected use, but
# an unbounded fan-out of real Telegram sends from one click is a real (if
# modest) outage/spam surface, not something worth allowing unbounded.
_MAX_BULK_CLARIFICATION_ACCOUNTS = 50


@app.post(
    "/clarification-requests/bulk", response_model=BulkClarificationOut, dependencies=[Depends(require_api_key)]
)
async def send_bulk_clarification(body: BulkClarificationIn):
    """Sends ONE already-written message, verbatim, to every account in
    account_ids -- deliberately not an LLM draft per account (that's
    grounded in a single account's own flags, see draft_clarification;
    fanning that out to N accounts would mean N real Groq calls for one
    click, adding real cost/latency and, per this project's own
    experience, real rate-limit risk). An operator wanting personalized
    wording for one specific account still has the single-account flow.

    Continues past an unknown account_id rather than failing the whole
    batch -- a stale/typo'd id in a large selection shouldn't block
    delivery to everyone else; it comes back in not_found instead."""
    if not body.account_ids:
        raise HTTPException(status_code=400, detail="account_ids must not be empty")
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    if len(body.account_ids) > _MAX_BULK_CLARIFICATION_ACCOUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"at most {_MAX_BULK_CLARIFICATION_ACCOUNTS} accounts per bulk send, got {len(body.account_ids)}",
        )

    sent_to: list[str] = []
    delivered_via_telegram: list[str] = []
    not_found: list[str] = []
    for account_id in body.account_ids:
        if store.get_account(account_id) is None:
            not_found.append(account_id)
            continue
        delivered = await notify.notify_clarification_request(account_id, body.message)
        sent_to.append(account_id)
        if delivered:
            delivered_via_telegram.append(account_id)

    return BulkClarificationOut(sent_to=sent_to, delivered_via_telegram=delivered_via_telegram, not_found=not_found)


@app.post(
    "/accounts/{account_id}/clarification-requests/mark-resolved",
    response_model=list[ClarificationRequestOut],
    dependencies=[Depends(require_api_key)],
)
def mark_clarifications_resolved(account_id: str):
    """Closes the loop on a thread of clarification requests -- found
    live: sending one had no way to ever come back as "handled." A
    borrower's reply is visible in the Conversation tab, but there was no
    way for ops to say "I've seen it, this is dealt with" and have that
    stick anywhere. See store.mark_clarifications_resolved's own docstring
    for why this is a whole-thread checkpoint, not per-message state."""
    if store.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
    store.mark_clarifications_resolved(account_id)
    return [ClarificationRequestOut(**c) for c in store.get_clarification_requests(account_id)]


@app.post("/accounts/{account_id}/escalate", response_model=EscalationOut, dependencies=[Depends(require_api_key)])
def escalate_account(account_id: str, body: EscalateToHumanIn):
    """Ops-side hand-off to a human -- found live: this action existed
    for a BORROWER (browser_api.py's own quick-actions/agent endpoint) but
    nowhere for staff themselves. An operator noticing a worrying pattern
    (repeated failed call attempts, something that doesn't fit any
    existing flag) previously had no way to open a real escalation
    without asking the borrower to request one first."""
    if store.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
    result = escalate_to_human(account_id, body.reason)
    store.log_event(account_id, "tool_called", {"tool": "escalate_to_human", "arguments": {"account_id": account_id, "reason": body.reason}, "result": result})
    escalation = store.get_escalation(result["escalation_id"])
    return _escalation_out(escalation)


@app.post(
    "/accounts/{account_id}/payments", response_model=RecordPaymentOut, dependencies=[Depends(require_api_key)]
)
def record_account_payment(account_id: str, body: RecordPaymentIn):
    """Records a payment ops took directly -- over the phone, UPI, cash,
    or an NEFT credit the automated payment-link flow never saw. Found
    live: store.record_payment (the one place a payment actually lands,
    with correct reducing-balance/overpayment/partial-payment semantics)
    was only ever reachable through a borrower's own single-use payment
    link -- an operator who takes a payment any other way had no door into
    the system at all, meaning the single most common real action a
    collections operator takes couldn't be done from the ops dashboard."""
    if store.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
    try:
        result = store.record_payment(
            account_id, body.amount, payment_date=body.payment_date, apply_extra_to_next=body.apply_extra_to_next
        )
    except store.ExtraPaymentDecisionRequiredError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    store.log_event(account_id, "tool_called", {"tool": "record_payment_ops", "arguments": body.model_dump(mode="json"), "result": result})
    return RecordPaymentOut(**result)


# ============================================================
# Bank statement (cash application): accounts/bank_statement.py reads and matches, accounts/store.py posts.
# ============================================================

_MAX_STATEMENT_BYTES = 2 * 1024 * 1024


class BankLineOut(BaseModel):
    id: int
    statement: str
    line_no: int
    line_date: date
    amount: float
    narration: str
    reference: str
    status: str  # 'posted' | 'suggested' | 'unmatched' | 'ignored'
    account_id: str | None
    match_reason: str | None
    payment_id: int | None
    dispute_id: int | None  # the payment claim it settled
    decided_by: str | None
    decided_note: str | None
    decided_at: datetime | None


class StatementImportOut(BaseModel):
    statement: str
    posted: int
    suggested: int
    unmatched: int
    already_uploaded: int
    debits_skipped: int
    rejected_rows: list[dict]  # {row, reason}: rows that could not be read, said, not dropped
    lines: list[BankLineOut]


class PostBankLineIn(BaseModel):
    account_id: str
    decided_by: str = Field(min_length=2, max_length=80)
    record_even_if_duplicate: bool = False


class IgnoreBankLineIn(BaseModel):
    decided_by: str = Field(min_length=2, max_length=80)
    note: str = Field(min_length=3, max_length=300)


@app.post("/bank-statements", response_model=StatementImportOut, dependencies=[Depends(require_api_key)])
async def upload_bank_statement(file: UploadFile = File(...)):
    """Ops upload the lender's bank statement (a CSV export). Every credit is saved and matched: a line that names one
    account by a strong identifier (account number, the reference the borrower gave on a payment claim, their phone
    number) is posted at once and closes that claim, and the borrower is told in the claim's thread; the rest wait,
    pre-filled with the best guess and why. Uploading the same statement again posts nothing twice. 400 for a file
    that is not a readable statement."""
    from businessflow.accounts.bank_statement import StatementError, parse_statement

    raw = await file.read(_MAX_STATEMENT_BYTES + 1)
    if len(raw) > _MAX_STATEMENT_BYTES:
        raise HTTPException(status_code=400, detail="the statement is larger than 2 MB -- upload a shorter date range")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail="the file is not UTF-8 text -- export the statement as CSV") from e
    name = Path(file.filename or "statement.csv").name[:120]
    try:
        parsed = parse_statement(text)
    except StatementError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    result = store.import_bank_statement(name, parsed)
    for line in result["lines"]:
        if line["status"] == "posted" and line["dispute_id"]:
            await notify.notify_case_message(
                line["account_id"], cases.dispute_case_ref(line["dispute_id"]),
                f"We found your payment of ₹{line['amount']:,.0f} on {line['line_date'].day} "
                f"{line['line_date'].strftime('%b')} in our bank statement. It is now recorded on your loan, "
                "and this claim is closed.",
                about="your payment claim", reply_in_chat=False,
            )
    return StatementImportOut(
        statement=name, posted=result["posted"], suggested=result["suggested"], unmatched=result["unmatched"],
        already_uploaded=result["already_uploaded"], debits_skipped=parsed.debits_skipped,
        rejected_rows=parsed.rejected[:50], lines=[BankLineOut(**line) for line in result["lines"]],
    )


@app.get("/bank-lines", response_model=list[BankLineOut], dependencies=[Depends(require_api_key)])
def list_bank_lines(status: Literal["open", "done"] = "open"):
    """'open': lines waiting for a person, oldest first. 'done': the most recently posted or ignored, newest first."""
    return [BankLineOut(**line) for line in store.list_bank_lines(status)]


@app.post("/bank-lines/{line_id}/post", response_model=BankLineOut, dependencies=[Depends(require_api_key)])
def post_bank_line(line_id: int, body: PostBankLineIn):
    """Posts a waiting line to an account as a payment dated the day the bank received it. 409 if it was posted or
    ignored already; 409 with code possible_duplicate when the same amount is already on the account this week
    (send record_even_if_duplicate to post it anyway); 404 for an unknown line or account."""
    try:
        result = store.post_bank_line(line_id, body.account_id, body.decided_by.strip(), body.record_even_if_duplicate)
    except store.BankLineAlreadyDecidedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except store.PossibleDuplicatePaymentError as e:
        first = e.existing[0]
        raise HTTPException(status_code=409, detail={
            "code": "possible_duplicate",
            "message": f"₹{first['amount']:,.0f} paid on {first['payment_date'].isoformat()} is already on this account. "
                       "Post anyway only if this is a different payment.",
            "existing": [{**x, "payment_date": x["payment_date"].isoformat(), "recorded_at": str(x["recorded_at"])} for x in e.existing],
        }) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return BankLineOut(**result["line"])


@app.post("/bank-lines/{line_id}/ignore", response_model=BankLineOut, dependencies=[Depends(require_api_key)])
def ignore_bank_line(line_id: int, body: IgnoreBankLineIn):
    """Marks a waiting line as not a loan payment, with who and why. 409 if already decided, 404 if unknown."""
    try:
        line = store.ignore_bank_line(line_id, body.decided_by.strip(), body.note.strip())
    except store.BankLineAlreadyDecidedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return BankLineOut(**line)


@app.post(
    "/accounts/{account_id}/charges/{charge_id}/waive",
    response_model=WaiveChargeOut,
    dependencies=[Depends(require_api_key)],
)
async def waive_account_charge(account_id: str, charge_id: int, body: WaiveChargeIn):
    """Waives what is still owed on a charge (a late fee): who and why are required and kept on the charge, and the
    borrower is told, with the reason. 404 for a charge that is not this account's; 409 if it is already paid or
    waived. A partly paid fee keeps what was paid; the rest is waived."""
    waived_by, reason = body.waived_by.strip(), body.reason.strip()
    if len(waived_by) < 2 or len(reason) < 3:
        raise HTTPException(status_code=400, detail="say who is waiving it and why -- the reason is sent to the borrower")
    try:
        charge = store.waive_charge(account_id, charge_id, waived_by, reason)
    except store.ChargeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except store.ChargeNotOpenError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    store.log_event(account_id, "tool_called", {"tool": "waive_charge", "arguments": {"charge_id": charge_id, "waived_by": waived_by, "reason": reason}, "result": {"status": charge.status}})
    due = charge.for_due_date
    message = (
        f"Our team has removed the late fee of ₹{charge.amount - charge.amount_paid:,.0f} for your "
        f"{due.day} {due.strftime('%b')} {due.year} EMI. Reason: {reason}"
    )
    notified = await notify.notify_charge_waived(account_id, message, case_ref=body.case_ref)
    return WaiveChargeOut(charge=_charge_out(charge), borrower_notified=notified)


@app.post(
    "/accounts/{account_id}/disputes/resolve",
    response_model=ResolveDisputeOut,
    dependencies=[Depends(require_api_key)],
)
async def resolve_account_dispute(account_id: str, body: ResolveDisputeIn | None = None):
    """Closes this account's currently open dispute. Found live: there was
    no way, anywhere in this codebase (ops UI or store layer), to ever
    turn a dispute back off -- once flagged, an account stayed permanently
    'disputed' (and kept showing up in the Disputed filter/portfolio
    count) even after an operator actually resolved the underlying issue.
    body is optional so a plain POST with no reasoning still works --
    resolution_note is worth capturing when an operator has one, never
    required."""
    if store.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
    resolution_note = ((body.resolution_note or "").strip() or None) if body is not None else None
    dispute_id = body.dispute_id if body is not None else None
    open_case = next((d for d in store.get_disputes_for_account(account_id) if d["status"] == "open"), None)
    if open_case is not None and open_case["kind"] == cases.PAYMENT_CLAIM and not resolution_note:
        # Closing a payment claim without recording the payment: the borrower is told why, so a note is required.
        raise HTTPException(status_code=400, detail="say why the payment couldn't be confirmed -- the borrower is sent this note")
    try:
        result = store.resolve_dispute(account_id, resolution_note=resolution_note, dispute_id=dispute_id)
    except store.NoOpenDisputeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    store.log_event(account_id, "tool_called", {"tool": "resolve_dispute", "arguments": {"account_id": account_id, "resolution_note": resolution_note, "dispute_id": dispute_id}, "result": result})
    # Tell the borrower. Resolving used to be silent: the freeze lifted and the
    # banner vanished, but nothing said how it ended. Delivered over Telegram
    # when linked, and always shown in that case's thread on their dashboard.
    message = _case_closed_message(result["kind"], resolution_note)
    notified = await notify.notify_dispute_resolved(account_id, message, case_ref=cases.dispute_case_ref(result["dispute_id"]))
    return ResolveDisputeOut(**result, borrower_notified=notified)


def _case_closed_message(kind: str, resolution_note: str | None) -> str:
    """What the borrower is sent when ops closes their dispute or payment claim (also shown, ahead, on a voice
    confirmation card -- ops/voice_reply.py)."""
    if kind == cases.PAYMENT_CLAIM:
        # Neutral: closing a claim without recording a payment can mean "not found" or "already recorded" (found
        # live: a claim filed for a payment that had landed minutes earlier). The note says which.
        message = "Update on the payment you reported: our team has checked it and closed it."
    else:
        message = "Update on your dispute: our team has reviewed it and closed it."
    if resolution_note:
        message += f" Note from our team: {resolution_note}"
    return message


@app.post("/accounts/{account_id}/promises", response_model=LogPromiseOut, dependencies=[Depends(require_api_key)])
def log_account_promise(account_id: str, body: LogPromiseIn):
    """Logs a promise-to-pay an operator negotiated directly (e.g. on a
    phone call). Found live: store.add_promise was only ever called from
    the LLM agent's own borrower-chat tool -- a promise an operator
    negotiated by phone had no way into the system at all, so it never
    counted toward broken_promise_count() (which real policy -- the
    mandatory-escalation threshold -- depends on) and was invisible to
    whichever operator opened this account next."""
    if store.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
    made_on = body.made_on or store.current_date()
    account = store.get_account_or_raise(account_id)
    raw = [PromisePart(body.promised_date, body.promised_amount)] + [PromisePart(p.date, p.amount) for p in body.more_parts]
    try:
        parts = validate_promise_parts(raw, made_on, ceiling=validation.max_payable(account))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    result = store.add_promise_plan(account_id, parts, made_on, made_via="ops")
    return LogPromiseOut(
        account_id=account_id, made_on=made_on, promised_date=body.promised_date,
        promised_amount=body.promised_amount, created=not result["already_logged"], plan_id=result["plan_id"],
        terms=promise_terms_text(parts), replaced_earlier_promise=result["replaced_plan_id"] is not None,
    )


@app.post("/accounts/{account_id}/call-log", response_model=CallLogEntryOut, dependencies=[Depends(require_api_key)])
def log_account_call(account_id: str, body: CallLogIn):
    """Records that ops attempted to reach this borrower by phone, and
    what happened -- so a second operator opening the same profile later
    the same day can see a call was already attempted, instead of the
    phone number being inert text with no record of contact ever attached
    to it."""
    if store.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")
    store.log_call(account_id, body.outcome, body.note)
    latest = store.get_call_log(account_id)[0]
    return CallLogEntryOut(**latest)


@app.get("/escalations", response_model=list[EscalationOut], dependencies=[Depends(require_api_key)])
def list_open_escalations():
    """The human-handoff queue: every escalation still waiting on a
    person, oldest first."""
    return [_escalation_out(e) for e in store.list_open_escalations()]


@app.get("/inbox", response_model=list[CaseOut], dependencies=[Depends(require_api_key)])
def get_inbox():
    """Everything open that a person has to deal with, in one list: payment
    claims, disputes, requests for a person or for more time, closure
    certificates, fraud claims and system notices (broken promises, chronic
    delinquency) -- most urgent first, then oldest, each with the borrower's
    own words and the messages since. Before, escalations had their own queue
    and disputes had none (an operator found a dispute only by opening the
    account), and a borrower's "I already paid" waited in neither."""
    with get_connection().transaction() as conn:
        escalations = store.list_open_escalations(ex=conn)
        disputes = store.list_open_disputes(ex=conn)
        names = store.borrower_names(sorted({e.account_id for e in escalations}), ex=conn)
        refs = [e.escalation_id for e in escalations] + [cases.dispute_case_ref(d["id"]) for d in disputes]
        threads = store.get_threads_for_cases(refs, ex=conn)
        recent = store.recent_payments(sorted({d["account_id"] for d in disputes if d["kind"] == cases.PAYMENT_CLAIM}), ex=conn)
    items = [_escalation_case(e, names.get(e.account_id), threads) for e in escalations]
    items += [_dispute_case(d, d["account_id"], d["borrower_name"], threads, recent) for d in disputes]
    return _sorted_cases(items)


def _case_or_404(account_id: str, case_ref: str) -> dict:
    case = store.get_case(account_id, case_ref)
    if case is None:
        raise HTTPException(status_code=404, detail=f"no case {case_ref!r} on account_id={account_id!r}")
    return case


@app.post(
    "/accounts/{account_id}/cases/{case_ref}/reply", response_model=CaseReplyOut, dependencies=[Depends(require_api_key)]
)
async def reply_to_case(account_id: str, case_ref: str, body: CaseReplyIn):
    """A message from ops about one case: it appears in that case's thread on
    the borrower's dashboard and on this one, and goes over Telegram when
    linked (saying what it is about). Found live: ops could only send a
    general "clarification request", unconnected to the dispute or request
    it answered, and the borrower read it in a different panel from their
    request. Works on a closed case too (a follow-up), not on a system
    notice, which the borrower never sees."""
    try:
        message = validation.free_text(body.message, "message")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    case = _case_or_404(account_id, case_ref)
    if case["kind"] == SYSTEM:
        raise HTTPException(
            status_code=400,
            detail="a system notice isn't shown to the borrower -- send a general message from the profile instead",
        )
    notified = await notify.notify_case_message(
        account_id, case_ref, message, about=_CASE_ABOUT.get(case["kind"], "your request"),
        reply_in_chat=case["source"] == "dispute",
    )
    return CaseReplyOut(
        case_ref=case_ref, author="ops", message=message, created_at=datetime.now(timezone.utc),
        delivered_via_telegram=notified,
    )


@app.post(
    "/accounts/{account_id}/cases/{case_ref}/confirm-payment",
    response_model=ConfirmPaymentOut,
    dependencies=[Depends(require_api_key)],
)
async def confirm_payment_claim(account_id: str, case_ref: str, body: ConfirmPaymentIn):
    """The one right action on a payment claim: ops found the payment, so it is
    recorded (store.record_payment's rules) and the claim closed, together,
    and the borrower is told in the claim's thread. A second click, or a
    colleague who got there first, gets 409 and records nothing twice."""
    case = _case_or_404(account_id, case_ref)
    if case["kind"] != cases.PAYMENT_CLAIM:
        raise HTTPException(status_code=400, detail=f"{case_ref} is not a payment claim")
    note = (body.note or "").strip() or None
    _, dispute_id = cases.parse_case_ref(case_ref)
    try:
        result = store.confirm_payment_claim(
            account_id, dispute_id, body.amount, payment_date=body.payment_date,
            apply_extra_to_next=body.apply_extra_to_next, resolution_note=note or "Payment confirmed and recorded.",
            record_even_if_duplicate=body.record_even_if_duplicate,
        )
    except store.ExtraPaymentDecisionRequiredError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except store.NoOpenDisputeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except store.PossibleDuplicatePaymentError as e:
        # Structured, so the case card can show what is already on record and offer "record it anyway".
        first = e.existing[0]
        raise HTTPException(status_code=409, detail={
            "code": "possible_duplicate",
            "message": (
                f"A payment of ₹{first['amount']:,.2f} (paid {first['payment_date']:%d %b}) was already recorded on this account "
                f"{first['recorded_at'].astimezone(_IST):%d %b, %H:%M} IST. If it's the payment this claim is about, close the "
                "claim instead; record this only if it's a second, separate payment."
            ),
            "existing": [
                {"amount": p["amount"], "payment_date": p["payment_date"].isoformat(), "recorded_at": p["recorded_at"].isoformat()}
                for p in e.existing
            ],
        }) from e
    payment = result["payment"]
    store.log_event(account_id, "tool_called", {"tool": "confirm_payment_claim", "arguments": {"case_ref": case_ref, **body.model_dump(mode="json")}, "result": result})
    message = _claim_confirmed_message(payment["amount"], date.fromisoformat(payment["payment_date"]), note)
    notified = await notify.notify_dispute_resolved(account_id, message, case_ref=case_ref)
    return ConfirmPaymentOut(case_ref=case_ref, payment=RecordPaymentOut(**payment), borrower_notified=notified)


def _claim_confirmed_message(amount: float, paid_on: date, note: str | None) -> str:
    """What the borrower is sent when ops confirms their payment claim (also shown, ahead, on a voice confirmation
    card -- ops/voice_reply.py)."""
    message = f"Good news: we've found your payment of ₹{amount:,.2f} from {paid_on:%d %b %Y} and recorded it on your account."
    if note:
        message += f" Note from our team: {note}"
    return message


def _escalation_out(escalation, borrower_notified: bool | None = None) -> EscalationOut:
    return EscalationOut(
        escalation_id=escalation.escalation_id, account_id=escalation.account_id,
        reason=escalation.reason, status=escalation.status,
        created_at=escalation.created_at, resolved_at=escalation.resolved_at,
        proposed_changes=escalation.proposed_changes, resolution_reason=escalation.resolution_reason,
        kind=classify_escalation(escalation.reason, escalation.proposed_changes),
        borrower_notified=borrower_notified,
    )


def _approval_message(kind: str, result: dict, account: Account | None) -> str | None:
    """What the borrower is told when staff approve/close their ticket, by
    kind. Only a real restructuring says "approved ... new EMI": before this,
    approving ANY ticket sent "Good news -- your recent request has been
    approved", including to a borrower who had filed a complaint (live: a
    grievance got exactly that). A system-raised ticket (guardrail block,
    chronic-delinquency notice) has no borrower request behind it, so it
    sends nothing at all."""
    if kind == SYSTEM:
        return None
    if kind == RESTRUCTURING and "new_months_remaining" in result and "new_emi_amount" in result:
        message = (
            f"Good news — your request for more time has been approved. Your loan now has "
            f"{result['new_months_remaining']} months remaining, with a new EMI of "
            f"₹{result['new_emi_amount']:,.2f}."
        )
        if account is not None and account.days_past_due(store.current_date()) > 0:
            message += " Your overdue instalment is still due — please pay it, or talk to us if you need help."
        return message
    if kind == CLOSURE:
        return "Your loan closure request has been approved by our team."
    if kind == CALLBACK:
        return "Our team has followed up on your call-back request."
    return "Our team has reviewed your request and closed it. If anything is still unresolved, reply here and we'll pick it up."


def _rejection_message(kind: str, reason: str | None) -> str | None:
    if kind == SYSTEM:
        return None
    message = (
        "Your request for more time could not be approved."
        if kind == RESTRUCTURING
        else "Your recent request could not be approved."
    )
    if reason:
        message += f" Reason: {reason}"
    if kind == RESTRUCTURING:
        message += " You can still pay what's due, or talk to us about other options."
    return message


@app.post(
    "/escalations/{escalation_id}/approve", response_model=EscalationOut, dependencies=[Depends(require_api_key)]
)
async def approve_escalation(escalation_id: str):
    """A human clicking Approve: if this escalation is a structured
    restructuring request, applies its real proposed_changes to the
    account (see accounts/store.py's approve_restructuring -- the only
    place in this system that actually commits one); most escalations
    aren't one of those (an open dispute, a broken-promise pattern, or
    the agent just being unsure -- see approve_restructuring's own
    docstring), so this just closes those out with no account change.
    Either way, if the borrower has a linked Telegram chat, tells them."""
    # Read the ticket first: what the borrower is told depends on what it is.
    ticket = store.get_escalation(escalation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"No escalation found for escalation_id={escalation_id!r}")
    kind = classify_escalation(ticket.reason, ticket.proposed_changes)
    try:
        result = store.approve_restructuring(escalation_id)
    except store.EscalationNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except store.EscalationAlreadyResolvedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        # An unrecognized proposed_changes shape -- a real, if unlikely,
        # data problem worth surfacing as a clean 400, not the unhandled
        # 500 this used to be.
        raise HTTPException(status_code=400, detail=str(e)) from e

    message = _approval_message(kind, result, store.get_account(result["account_id"]))
    notified = None
    if message is not None:
        notified = await notify.notify_restructuring_decision(
            result["account_id"], approved=True, message=message, case_ref=escalation_id
        )

    escalation = store.get_escalation(escalation_id)
    return _escalation_out(escalation, borrower_notified=notified)


@app.post(
    "/escalations/{escalation_id}/reject", response_model=EscalationOut, dependencies=[Depends(require_api_key)]
)
async def reject_escalation(escalation_id: str, body: EscalationRejectIn):
    """A human clicking Reject: marks the escalation rejected (the
    account itself is never touched -- nothing was ever applied to it)
    and, if the borrower has a linked Telegram chat, tells them, including
    the optional reason if ops entered one."""
    ticket = store.get_escalation(escalation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"No escalation found for escalation_id={escalation_id!r}")
    kind = classify_escalation(ticket.reason, ticket.proposed_changes)
    try:
        result = store.reject_restructuring(escalation_id, body.reason)
    except store.EscalationNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except store.EscalationAlreadyResolvedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    message = _rejection_message(kind, result["reason"])
    notified = None
    if message is not None:
        notified = await notify.notify_restructuring_decision(
            result["account_id"], approved=False, message=message, case_ref=escalation_id
        )

    escalation = store.get_escalation(escalation_id)
    return _escalation_out(escalation, borrower_notified=notified)


@app.get("/portfolio/health", dependencies=[Depends(require_api_key)])
def get_portfolio_health():
    """Collection health (accounts/health.py): days-past-due buckets, collection efficiency, promises kept and cure
    rate, plus containment from the event log -- each with the numbers behind it and how it was worked out. Late fees
    are assessed first so the buckets' amounts match what every screen shows as owed."""
    from businessflow.accounts.health import portfolio_health

    store.assess_late_fees()
    store.resolve_promises()
    as_of = store.current_date()
    health = portfolio_health(store.list_accounts(), as_of)
    turns, handed = metrics.escalation_counts(datetime.now(timezone.utc) - timedelta(days=30))
    health["containment"] = {
        "value": round(1 - handed / turns, 4) if turns else None, "turns": turns, "handed_to_person": handed,
        "window_days": 30,
        "how": "Of the assistant's tool-calling turns with borrowers in the last 30 days, the share it handled without "
               "handing the borrower to a person.",
    }
    return health


# ============================================================
# The living demo: anyone trying the public demo can put the demo borrowers back to their starting story.
# ============================================================

_DEMO_RESET_COOLDOWN = timedelta(minutes=5)
_REPO_ROOT = Path(__file__).resolve().parents[3]


class DemoStatusOut(BaseModel):
    enabled: bool
    last_reset_at: datetime | None


class DemoResetIn(BaseModel):
    confirm: Literal["reset"]  # typed, so a stray request cannot wipe what someone is in the middle of


def _demo_reset_enabled() -> bool:
    """On for this demo deployment; a real lender's deployment sets DEMO_RESET_ENABLED=0, and the endpoint is gone."""
    return os.environ.get("DEMO_RESET_ENABLED", "1") != "0"


def _last_demo_reset() -> datetime | None:
    row = get_connection().execute(
        "select created_at from events where event_type = 'demo_reset' order by created_at desc limit 1"
    ).fetchone()
    return row["created_at"] if row else None


@app.get("/demo", response_model=DemoStatusOut, dependencies=[Depends(require_api_key)])
def demo_status():
    return DemoStatusOut(enabled=_demo_reset_enabled(), last_reset_at=_last_demo_reset())


@app.post("/demo/reset", response_model=DemoStatusOut, dependencies=[Depends(require_api_key)])
def reset_demo(body: DemoResetIn):
    """Puts the demo borrowers back to their starting story (scripts/seed_accounts.py, dated from today): payments,
    promises, fees, cases, messages and bank lines on those accounts are replaced. Accounts anyone opened themselves
    are left alone. Once per _DEMO_RESET_COOLDOWN at most, so two people trying the demo at once do not keep wiping
    each other; 404 when the deployment has it off.

    The seed runs as its own process from the repository root (python -m scripts.seed_accounts), the way it is always
    run: whether `scripts` is importable from inside this server depends on how the server was started. Bounded by a
    timeout; a seed that fails is a 500 with its last error line, and nothing is half-reset (it is one transaction)."""
    if not _demo_reset_enabled():
        raise HTTPException(status_code=404, detail="demo reset is off on this deployment")
    last = _last_demo_reset()
    if last is not None and datetime.now(timezone.utc) - last < _DEMO_RESET_COOLDOWN:
        minutes = max(1, int((datetime.now(timezone.utc) - last).total_seconds() // 60))
        raise HTTPException(status_code=429, detail=f"the demo was reset {minutes} minute(s) ago -- try again in a few minutes")
    try:
        subprocess.run([sys.executable, "-m", "scripts.seed_accounts"], cwd=_REPO_ROOT, check=True, capture_output=True,
                       text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        last_line = (e.stderr or "").strip().splitlines()[-1:] or ["no error output"]
        logger.error("demo reset failed: %s", last_line[0])
        raise HTTPException(status_code=500, detail=f"the demo could not be reset: {last_line[0]}") from e
    except subprocess.TimeoutExpired as e:
        logger.error("demo reset timed out after 120 s")
        raise HTTPException(status_code=500, detail="the demo reset took too long and was stopped") from e
    store.log_event(None, "demo_reset", {})
    logger.info("demo reset: the demo accounts are back to their starting story")
    return DemoStatusOut(enabled=True, last_reset_at=_last_demo_reset())


# ============================================================
# Loan-book import: a lender's existing loans from a CSV export (ops/loan_book.py reads and checks every row).
# ============================================================

_MAX_LOAN_BOOK_BYTES = 2 * 1024 * 1024


class LoanBookRowOut(BaseModel):
    row: int
    borrower_name: str | None = None
    external_ref: str | None = None
    errors: list[str] = []
    account_id: str | None = None  # set when created, or when the loan was already on the books
    access_key: str | None = None  # set only when created -- shown once, never stored in clear
    telegram_invite_link: str | None = None


class LoanBookOut(BaseModel):
    dry_run: bool
    imported: bool  # False whenever any row has an error: nothing is written then
    rows: int
    ready: int  # rows that would be (or were) opened
    already_on_books: int
    with_errors: int
    details: list[LoanBookRowOut]


@app.post("/loan-book", response_model=LoanBookOut, dependencies=[Depends(require_api_key)])
async def import_loan_book(file: UploadFile = File(...), dry_run: bool = Form(True)):
    """Imports a loan book (CSV): every row checked first (ops/loan_book.py). dry_run=true (the default) only says
    what would happen. With dry_run=false the loans are opened -- all of them, or none if any row has an error; a loan
    whose reference is already on the books is skipped, never opened twice. Each new account's access key is in the
    response, once (the database keeps only its hash). 400 for a file that is not a loan book at all."""
    from businessflow.ops.loan_book import LoanBookError, parse_loan_book

    raw = await file.read(_MAX_LOAN_BOOK_BYTES + 1)
    if len(raw) > _MAX_LOAN_BOOK_BYTES:
        raise HTTPException(status_code=400, detail="the loan book is larger than 2 MB -- split it")
    try:
        rows = parse_loan_book(raw.decode("utf-8-sig"))
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail="the file is not UTF-8 text -- export it as CSV") from e
    except LoanBookError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    existing = store.accounts_by_external_ref([r.values["external_ref"] for r in rows if r.values.get("external_ref")])
    details = []
    for r in rows:
        ref = r.values.get("external_ref")
        details.append(LoanBookRowOut(row=r.row, borrower_name=r.values.get("borrower_name"), external_ref=ref,
                                      errors=r.errors, account_id=existing.get(ref) if ref else None))
    with_errors = sum(1 for d in details if d.errors)
    to_open = [(r, d) for r, d in zip(rows, details) if not d.errors and d.account_id is None]
    imported = not dry_run and with_errors == 0
    if imported:
        for r, d in to_open:
            v = r.values
            account, key = store.create_account(
                v["borrower_name"], v["business_name"], v["phone_number"], v["language_preference"], v["loan_type"],
                v["principal_amount"], v["emi_amount"], v["tenure_months"], v["emi_due_date"], v["nach_mandate_active"],
                v["risk_tier"], months_remaining=v["months_remaining"], external_ref=v["external_ref"],
            )
            d.account_id, d.access_key = account.account_id, key
            d.telegram_invite_link = _telegram_invite_link(account.account_id, key)
        logger.info("loan book %r: %d opened, %d already on the books", file.filename, len(to_open),
                    sum(1 for d in details if d.account_id and not d.access_key))
    return LoanBookOut(
        dry_run=dry_run, imported=imported, rows=len(rows), ready=len(to_open),
        already_on_books=sum(1 for d in details if d.account_id and not d.access_key and not d.errors),
        with_errors=with_errors, details=details,
    )


_EVAL_RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval" / "results"
# The nightly suites whose last result the Quality tab shows, with what each checks, in plain words.
_NIGHTLY_SUITES = {
    "red_team": "attacks on the assistant handled safely",
    "tool_calling_benchmark": "the right action taken for a clear request",
    "realistic_conversation_benchmark": "messy multi-turn conversations handled right",
    "reasoning_accuracy": "replies stating the account's facts correctly",
    "latency_benchmark": "how long a reply takes",
}


def _read_json(path: Path) -> dict | None:
    """A results file, or None when it does not exist yet. A file that exists but is not valid JSON is a real fault
    (a half-written run) and raises."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/quality", dependencies=[Depends(require_api_key)])
def get_quality():
    """What the evals say, for ops (eval/conversation_lab.py and the nightly suites on this server's disk): the
    conversation lab's labelled cases -- each with the policy line it is labelled from, whether it passed, and whether
    the model's side was recorded or scripted -- any prompt variant's lab run beside it, and each nightly suite's last
    result. Nothing here runs a model; it reads what the runs saved."""
    # A run on this server writes to live/ (git-ignored); the committed copy is shown only until one has.
    live_dir = _EVAL_RESULTS_DIR / "live"
    lab_paths = sorted(live_dir.glob("conversation_lab*.json")) or sorted(_EVAL_RESULTS_DIR.glob("conversation_lab*.json"))
    labs = [data for data in (_read_json(p) for p in lab_paths) if data]
    nightly = []
    for name, what in _NIGHTLY_SUITES.items():
        history = _EVAL_RESULTS_DIR / f"{name}_history.jsonl"
        last = None
        if history.exists():
            lines = [line for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
            last = json.loads(lines[-1]) if lines else None
        nightly.append({"suite": name, "checks": what, "last_run": last})
    from businessflow.observability import trace_report

    return {
        "labs": labs, "nightly": nightly,
        # Per AI component and chain (observability/trace_report.py): live, from the last 7 days of real turns, and
        # from the last nightly run's turns.
        "components_live": trace_report.collect(days=7),
        "components_nightly": _read_json(live_dir / "trace_report.json"),
    }


@app.get("/metrics", response_model=MetricsOut, dependencies=[Depends(require_api_key)])
def get_metrics(since_hours: float = 24.0):
    """Operational signal for the ops dashboard, built on
    observability/metrics.py (event counts by type, escalation rate) --
    that module previously had no caller anywhere in the app."""
    since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    return MetricsOut(
        since_hours=since_hours,
        event_counts=metrics.event_counts_since(since),
        escalation_rate=metrics.escalation_rate(since),
    )


# --- Voice -----------------------------------------------------------------------------------------------------------
# Ask the dashboard a question, or tell it to do something, by voice or typing: ops/voice.py works out what was meant
# (one language-model call choosing from a fixed menu), ops/voice_reply.py answers from the data below or builds a
# confirmation card. No endpoint in this section writes anything: an action happens only when the operator clicks
# Confirm on its card, which calls the ordinary endpoint above that the dashboard's own form for it uses.
#
# Speech goes through the chat service's Whisper and Piper (ops/speech_client.py), never loaded in this process.

# Each question is a model call plus database reads, and each recording is Whisper time on a 2-vCPU VM the borrowers'
# voice messages share -- and the ops key is a single shared secret -- so both are capped per client.
_voice_limiter = RateLimiter(max_requests=12, window_seconds=60)
_speech_limiter = RateLimiter(max_requests=30, window_seconds=60)
_MAX_VOICE_UPLOAD_BYTES = 4 * 1024 * 1024  # channels/browser_api.py's _OPS_VOICE_MAX_UPLOAD_BYTES: 30 s of WAV fits


class VoiceWorkflowIn(BaseModel):
    """The guided run in progress on the page, if any (ops/voice_workflows.py): which one, and where in it."""

    kind: Literal["call_list", "inbox"]
    position: int = Field(ge=1, le=500)
    total: int = Field(ge=1, le=500)
    case_ref: str | None = Field(default=None, max_length=40)  # the inbox's current case


class VoiceScreenIn(BaseModel):
    """What the operator is looking at, so "this account" and "her" mean the profile that is open, and "next" or
    "approve it" mean the item of the run in progress."""

    view: Literal["overview", "inbox"] = "overview"
    account_id: str | None = Field(default=None, max_length=40)
    workflow: VoiceWorkflowIn | None = None


class VoiceAskTextIn(BaseModel):
    text: str = Field(min_length=1, max_length=voice.MAX_REQUEST_CHARS)
    screen: VoiceScreenIn = VoiceScreenIn()


class VoicePlanIn(BaseModel):
    """A plan the page already has -- one of the "did you mean" choices of an earlier reply -- run without the model."""

    tool: str
    arguments: dict = {}
    heard: str = Field(default="", max_length=voice.MAX_REQUEST_CHARS)
    screen: VoiceScreenIn = VoiceScreenIn()


class VoiceSpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=400)


class _VoiceData:
    """ops/voice_reply.py's DashboardData, served by this module's own endpoint functions -- the same data, the same
    shape, the dashboard draws its cards from. Remembers what it read for the length of one request."""

    def __init__(self):
        self._accounts: list[dict] | None = None
        self._details: dict[str, dict | None] = {}

    def accounts(self) -> list[dict]:
        if self._accounts is None:
            self._accounts = [a.model_dump() for a in list_accounts()]
        return self._accounts

    def account(self, account_id: str) -> dict | None:
        if account_id not in self._details:
            try:
                self._details[account_id] = get_account(account_id).model_dump()
            except HTTPException as e:
                if e.status_code != 404:
                    raise
                self._details[account_id] = None
        return self._details[account_id]

    def inbox(self) -> list[dict]:
        return [c.model_dump() for c in get_inbox()]

    def metrics(self, since_hours: float) -> dict:
        return get_metrics(since_hours).model_dump()

    def documents(self, account_id: str) -> list[dict]:
        return [d.model_dump() for d in list_account_documents(account_id)]

    def conversation(self, account_id: str) -> list[dict]:
        return [e.model_dump() for e in get_account_conversation(account_id)]

    def decision_message(self, case: dict, account: dict, approved: bool, reason: str | None) -> str | None:
        if approved:
            # The restructuring terms the approval applies are the proposal's own (store.approve_restructuring).
            return _approval_message(case["kind"], case.get("proposed_changes") or {}, store.get_account(account["account_id"]))
        return _rejection_message(case["kind"], reason)

    def claim_confirmed_message(self, amount: float, paid_on: date, note: str | None) -> str:
        return _claim_confirmed_message(amount, paid_on, note)

    def case_closed_message(self, kind: str, note: str | None) -> str:
        return _case_closed_message(kind, note)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_voice_limit(limiter: RateLimiter, request: Request) -> None:
    if not limiter.check(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many voice requests in a minute -- wait a moment and try again.")


def _describe_screen(screen: VoiceScreenIn) -> str:
    where = "the Inbox" if screen.view == "inbox" else "the Portfolio"
    if screen.account_id:
        name = store.borrower_names([screen.account_id]).get(screen.account_id)
        if name:
            where = f"the profile of {name} ({screen.account_id})"
    run = screen.workflow
    if run:
        label = "the call list (a call outcome is log_call on this account)" if run.kind == "call_list" else (
            f"working through the inbox, on case {run.case_ref}")
        where = f"{label}, item {run.position} of {run.total}: {where}"
    return where


def _voice_respond(plan, heard: str, screen: VoiceScreenIn) -> voice_reply.VoiceReplyOut:
    run = screen.workflow
    return voice_reply.respond(
        plan, heard, _VoiceData(), screen.account_id, store.current_date(),
        screen_case=run.case_ref if run else None, workflow=run.kind if run else None,
    )


def _voice_answer(text: str, screen: VoiceScreenIn, source: str) -> voice_reply.VoiceReplyOut:
    """Understand text (a transcript or a typed request) and answer it. Logs the ability chosen and the time taken --
    never the words, which name borrowers."""
    started = time.monotonic()
    try:
        plan = voice.understand(text, store.current_date(), _describe_screen(screen))
    except voice.NotUnderstood as e:
        logger.warning("ops voice: model's choice unusable (%s) -- a steady stream of these means the menu or prompt needs work", e)
        return voice_reply.VoiceReplyOut(
            heard=text, kind="clarify", say="Sorry, I didn't understand that. Ask “what can you do?” to hear what I can.",
        )
    except groq.RateLimitError as e:
        logger.error("ops voice: every Groq key is rate-limited for the voice model -- voice is down until a bucket refills")
        raise HTTPException(
            status_code=503, detail="Voice is out of its AI allowance for now -- use the dashboard's buttons, or try again later.",
        ) from e
    except (groq.APIConnectionError, groq.APIStatusError) as e:
        logger.error("ops voice: Groq call failed (%s)", type(e).__name__)
        raise HTTPException(status_code=502, detail="Couldn't reach the AI service -- try again in a moment.") from e
    understood = time.monotonic()
    reply = _voice_respond(plan, text, screen)
    logger.info(
        "ops voice: source=%s ability=%s reply=%s model_ms=%d total_ms=%d",
        source, plan.tool, reply.kind, (understood - started) * 1000, (time.monotonic() - started) * 1000,
    )
    return reply


@app.post("/voice/ask", response_model=voice_reply.VoiceReplyOut, dependencies=[Depends(require_api_key)])
def voice_ask(request: Request, audio: UploadFile = File(...), screen: str = Form("{}")):
    """A spoken request: the page's recording, as WAV (converted in the browser, like the chat page's), plus what is on
    screen (JSON). Transcribed by the chat service, then answered like a typed one."""
    _check_voice_limit(_voice_limiter, request)
    try:
        screen_in = VoiceScreenIn.model_validate_json(screen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="screen must be a JSON object like {\"view\": \"overview\"}") from e
    raw = audio.file.read(_MAX_VOICE_UPLOAD_BYTES + 1)
    if len(raw) > _MAX_VOICE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Recording too long -- keep a question under 30 seconds.")
    started = time.monotonic()
    try:
        transcript = speech_client.transcribe(raw)
    except speech_client.UnreadableRecording as e:
        raise HTTPException(status_code=400, detail="Couldn't read that recording -- please try again.") from e
    except speech_client.SpeechServiceUnavailable as e:
        raise HTTPException(status_code=503, detail="Voice is unavailable right now (the speech service isn't answering). Typing still works.") from e
    logger.info("ops voice: transcribed %d bytes in %d ms", len(raw), (time.monotonic() - started) * 1000)
    if not transcript:
        return voice_reply.VoiceReplyOut(
            kind="clarify", say="I didn't hear anything. Hold the mic closer and keep the question under 30 seconds.",
        )
    return _voice_answer(transcript, screen_in, "voice")


@app.post("/voice/ask-text", response_model=voice_reply.VoiceReplyOut, dependencies=[Depends(require_api_key)])
def voice_ask_text(request: Request, body: VoiceAskTextIn):
    """The same as /voice/ask for a typed request -- for when speaking isn't possible."""
    _check_voice_limit(_voice_limiter, request)
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return _voice_answer(body.text.strip(), body.screen, "text")


@app.post("/voice/plan", response_model=voice_reply.VoiceReplyOut, dependencies=[Depends(require_api_key)])
def voice_plan(request: Request, body: VoicePlanIn):
    """Runs a plan the page sends back (a "did you mean" choice), checked against the same menu as the model's own
    choices. heard is the original request, which a figure on a resulting card must still be found in."""
    _check_voice_limit(_voice_limiter, request)
    try:
        plan = voice.parse_plan(body.tool, body.arguments)
    except voice.NotUnderstood as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _voice_respond(plan, body.heard, body.screen)


@app.post("/voice/speak", dependencies=[Depends(require_api_key)])
def voice_speak(request: Request, body: VoiceSpeakIn):
    """A reply, spoken: OGG/Opus from the chat service's Piper voice."""
    _check_voice_limit(_speech_limiter, request)
    try:
        audio_bytes = speech_client.speak(body.text)
    except (speech_client.SpeechServiceUnavailable, speech_client.UnreadableRecording) as e:
        raise HTTPException(status_code=503, detail="Spoken replies are unavailable right now; the text is on screen.") from e
    return Response(content=audio_bytes, media_type="audio/ogg")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(_STATIC_DIR / "index.html")


# Registered last so every API route above (exact paths like /accounts,
# /metrics, /escalations/...) is matched first -- this only serves
# /static/styles.css, /static/app.js, etc.
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
