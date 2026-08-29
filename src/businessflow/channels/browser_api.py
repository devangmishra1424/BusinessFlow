"""HTTP API for a browser-based text chat channel -- the backend a
frontend can call (built separately, per the plan: backend first, then
UI/UX). Text only for now, matching the project's "text primary, voice
later" decision.

Conversation state is held in-process, in memory, keyed by a generated
conversation_id -- restarting this server loses any conversation still
in flight. Cross-session memory (the recap a returning borrower's NEXT
conversation gets) is unaffected, since that's persisted to Postgres via
memory/conversation_memory.py independently of this in-memory state.

Run: uvicorn businessflow.channels.browser_api:app --reload
(defaults to port 8000). The ops dashboard API (ops/api.py) is a
separate FastAPI app on port 8001 -- run both side by side, they don't
share state or a port.
"""

import uuid
from datetime import date, datetime
from pathlib import Path

import groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from businessflow.accounts import store
from businessflow.accounts.documents import list_documents_for_account, resolve_document_path
from businessflow.accounts.models import Account
from businessflow.agent.loop import (
    AccessDeniedError,
    AccountLockedError,
    extract_new_tool_calls,
    run_turn_with_memory,
    start_conversation,
    verify_and_start_conversation,
)
from businessflow.channels.credentials import looks_like_credentials, parse_credentials
from businessflow.ops.flags import Flag, compute_flags
from businessflow.tools.account_tools import flag_dispute, get_payment_status
from businessflow.tools.escalation_tools import escalate_to_human
from businessflow.tools.payment_tools import generate_payment_link

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


class TimelineEntryOut(BaseModel):
    date: str
    amount: float
    status: str  # "paid-on-time" | "paid-late" | "overdue" | "upcoming"
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
    past = [
        {
            "date": r.date.isoformat(),
            "amount": r.amount,
            "status": "paid-on-time" if r.on_time else "paid-late",
            "label": "Paid on time" if r.on_time else "Paid late",
        }
        for r in account.payment_history
    ]

    upcoming = []
    if account.months_remaining > 0:
        cursor = account.emi_due_date
        for i in range(account.months_remaining):
            is_next_due = i == 0
            overdue = is_next_due and days_past_due > 0
            upcoming.append(
                {
                    "date": cursor.isoformat(),
                    "amount": account.emi_amount,
                    "status": "overdue" if overdue else "upcoming",
                    "label": f"Overdue -- {days_past_due}d past due" if overdue else "Scheduled",
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


@app.post("/conversations", response_model=StartConversationResponse)
def start_conversation_endpoint(req: StartConversationRequest):
    if req.language not in ("en", "hi"):
        raise HTTPException(status_code=400, detail="language must be 'en' or 'hi'")

    if req.account_id:
        if not req.access_key:
            raise HTTPException(status_code=401, detail=f"access_key is required to talk about account {req.account_id}")
        try:
            conversation = verify_and_start_conversation(req.language, req.account_id, req.access_key)
        except AccessDeniedError:
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


@app.post("/conversations/{conversation_id}/messages", response_model=SendMessageResponse)
def send_message_endpoint(conversation_id: str, req: SendMessageRequest):
    session = _conversations.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no conversation found for id={conversation_id!r}")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    # An anonymous session can still verify mid-conversation by sending
    # "<account_id> <6-digit key>" as a plain message -- the same pattern
    # telegram_bot.py already handles (see channels/credentials.py). Found
    # live: a borrower typing their real account_id + access_key straight
    # into an anonymous browser chat had both values forwarded to the LLM
    # as free text, which then passed the ACCESS KEY as account_id to a
    # tool and crashed it (get_payment_status raised "no account found for
    # account_id='<the key>'"). Checked here, before this message ever
    # reaches run_turn_with_memory, exactly like Telegram's guard.
    if session["account_id"] is None and looks_like_credentials(req.message):
        account_id, access_key = parse_credentials(req.message)
        try:
            conversation = verify_and_start_conversation(session["language"], account_id, access_key)
        except AccessDeniedError:
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

    turn_start = len(session["messages"])
    session["messages"].append({"role": "user", "content": req.message})
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
    session["messages"] = updated_conversation

    tool_calls = extract_new_tool_calls(updated_conversation, turn_start)
    return SendMessageResponse(
        reply=reply,
        tool_calls=[ToolCallInfo(tool=name, arguments=args) for name, args in tool_calls],
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


class PaymentConfirmOut(BaseModel):
    amount: float
    months_remaining: int
    next_emi_due_date: str


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
    return PaymentTokenInfoOut(**info)


@app.post("/pay/{token}/confirm", response_model=PaymentConfirmOut)
def payment_confirm_endpoint(token: str):
    """The only endpoint that can actually move an account forward from a
    payment link -- store.redeem_payment_token re-checks used_at/
    expires_at itself rather than trusting this endpoint already did (a
    double-submit from a slow network or an impatient double-tap must
    never record two payments for one token)."""
    try:
        result = store.redeem_payment_token(token)
    except store.PaymentTokenNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except store.PaymentTokenAlreadyUsedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except store.PaymentTokenExpiredError as e:
        raise HTTPException(status_code=410, detail=str(e)) from e
    store.log_event(
        result["account_id"], "tool_called",
        {"tool": "record_payment", "arguments": {"token": token}, "result": result},
    )
    return PaymentConfirmOut(
        amount=result["amount"], months_remaining=result["months_remaining"], next_emi_due_date=result["next_emi_due_date"]
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
