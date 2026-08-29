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

import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from docling.exceptions import ConversionError
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from businessflow.accounts import store
from businessflow.accounts.models import Account
from businessflow.observability import metrics
from businessflow.ops.flags import Flag, compute_flags
from businessflow.outbound import send as notify
from businessflow.outbound.compose import draft_clarification_message
from businessflow.rag.extraction import extract_loan_terms
from businessflow.rag.ingest import extract_document_text, ingest_document

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


def _cleanup_rejected_upload(saved_path: Path, account_dir: Path) -> None:
    """Removes a partially-saved upload and, if now empty, the account
    directory that was created to hold it -- the shared cleanup for every
    upload_account_document rejection path that runs *after* the file has
    already been written to disk (the size cap, and an unparseable
    document), so this logic lives in exactly one place instead of being
    duplicated slightly differently at each call site.

    account_dir is assumed freshly mkdir'd (exist_ok=True) for this
    upload -- if this was the account's first-ever upload attempt and it
    got rejected, don't leave an empty directory behind. rmdir() only
    succeeds on an empty directory, so this is a no-op (not an error)
    when the account already has other real documents saved.
    """
    saved_path.unlink(missing_ok=True)
    try:
        account_dir.rmdir()
    except OSError:
        pass


def require_api_key(provided_key: str = Security(_api_key_header)) -> None:
    expected_key = os.environ.get("OPS_API_KEY")
    if not expected_key:
        raise RuntimeError("OPS_API_KEY is not set -- copy .env.example to .env and fill it in")
    if provided_key != expected_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key header")


class FlagOut(BaseModel):
    label: str
    reason: str


class AccountSummaryOut(BaseModel):
    account_id: str
    borrower_name: str
    business_name: str
    loan_type: str
    principal_amount: float
    emi_amount: float
    days_past_due: int
    risk_tier: str
    flags: list[FlagOut]


class PaymentRecordOut(BaseModel):
    date: date
    amount: float
    on_time: bool


class PromiseOut(BaseModel):
    made_on: date
    promised_date: date
    promised_amount: float
    kept: bool | None


class DisputeOut(BaseModel):
    reason: str
    status: str  # 'open' | 'resolved'
    opened_at: datetime
    resolved_at: datetime | None


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


class EscalationRejectIn(BaseModel):
    # Optional, ops-entered explanation shown back to the borrower.
    reason: str | None = None


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


class AccountDetailOut(AccountSummaryOut):
    phone_number: str
    language_preference: str
    tenure_months: int
    months_remaining: int
    emi_due_date: date
    nach_mandate_active: bool
    dispute_open: bool
    payment_history: list[PaymentRecordOut]
    promises: list[PromiseOut]
    disputes: list[DisputeOut]
    escalations: list[EscalationOut]
    clarification_requests: list[ClarificationRequestOut]


def _summarize(account: Account, flags: list[Flag]) -> AccountSummaryOut:
    return AccountSummaryOut(
        account_id=account.account_id,
        borrower_name=account.borrower_name,
        business_name=account.business_name,
        loan_type=account.loan_type,
        principal_amount=account.principal_amount,
        emi_amount=account.emi_amount,
        days_past_due=account.days_past_due(store.current_date()),
        risk_tier=account.risk_tier,
        flags=[FlagOut(label=f.label, reason=f.reason) for f in flags],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/accounts", response_model=list[AccountSummaryOut], dependencies=[Depends(require_api_key)])
def list_accounts(flag: str | None = None):
    """The portfolio overview. Pass ?flag=overdue|disputed|broken_promises
    to see only accounts carrying that flag -- the queue an ops person
    would actually work from, not the full unfiltered list every time."""
    summaries = [_summarize(a, compute_flags(a)) for a in store.list_accounts()]
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
    return AccountCreateOut(account=_summarize(account, compute_flags(account)), access_key=access_key)


@app.get("/accounts/{account_id}", response_model=AccountDetailOut, dependencies=[Depends(require_api_key)])
def get_account(account_id: str):
    account = store.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account found for account_id={account_id!r}")

    summary = _summarize(account, compute_flags(account))
    escalations = store.get_escalations_for_account(account_id)
    return AccountDetailOut(
        **summary.model_dump(),
        phone_number=account.phone_number,
        language_preference=account.language_preference,
        tenure_months=account.tenure_months,
        months_remaining=account.months_remaining,
        emi_due_date=account.emi_due_date,
        nach_mandate_active=account.nach_mandate_active,
        dispute_open=account.dispute_open,
        payment_history=[
            PaymentRecordOut(date=p.date, amount=p.amount, on_time=p.on_time) for p in account.payment_history
        ],
        promises=[
            PromiseOut(made_on=p.made_on, promised_date=p.promised_date, promised_amount=p.promised_amount, kept=p.kept)
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
        disputes=[DisputeOut(**d) for d in store.get_disputes_for_account(account_id)],
        clarification_requests=[
            ClarificationRequestOut(**c) for c in store.get_clarification_requests(account_id)
        ],
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

    Known limitation, not solved here: businessflow.tools.policy_tools.
    _retriever() caches its DocumentRetriever PER PROCESS, refreshing it
    only on a time-based poll (see _REFRESH_INTERVAL_SECONDS there). This
    document is in the persistent Chroma store the instant this call
    returns, but a currently-running borrower-facing process (channels/
    browser_api.py, channels/telegram_bot.py) may not see it until that
    process's own cached snapshot next refreshes -- bounded to that
    interval, not "until restart", but not immediate either. There's no
    pub/sub or other cross-process cache invalidation here; that would be
    real new infra and is out of scope for this endpoint.
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
    account_dir.mkdir(parents=True, exist_ok=True)
    # Overwrite-by-filename is deliberate: re-uploading a corrected version
    # of the same document is the expected case, and ingest_document()
    # already handles safe re-ingestion at the same file_path.
    saved_path = account_dir / filename

    size = 0
    try:
        with open(saved_path, "wb") as out:
            while chunk := await file.read(_UPLOAD_READ_CHUNK_BYTES):
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit",
                    )
                out.write(chunk)
    except HTTPException:
        _cleanup_rejected_upload(saved_path, account_dir)
        raise

    try:
        chunks_stored = ingest_document(str(saved_path), document_type=document_type, account_id=account_id)
    except ConversionError as e:
        # The extension passed our allow-list check, but the bytes behind
        # it aren't a real, parseable document of that type (corrupt file,
        # wrong content masquerading under this extension, etc). Docling
        # raises its own ConversionError for exactly this -- caught
        # specifically, not via a bare `except Exception`, so a genuinely
        # unexpected failure still propagates as a 500 instead of being
        # misreported as "bad input". ingest_document() converts the file
        # before it ever touches the collection (both the delete-existing-
        # chunks call and the upsert come after), so nothing was written
        # to the vector store either -- the only cleanup needed is the
        # same saved-file/empty-dir removal the size-cap rejection above
        # already does.
        _cleanup_rejected_upload(saved_path, account_dir)
        raise HTTPException(
            status_code=422,
            detail=f"could not parse this file as {extension}: {e}",
        ) from e

    # Structured extraction only applies to a signed loan agreement --
    # there's no interest rate to find in a KYC document, say. Best-effort
    # on top of an ingestion that already succeeded above: the upload's
    # real contract ("the document is in RAG") is already satisfied by
    # this point, regardless of whether this extra step finds anything.
    interest_rate_extracted = False
    if document_type == "loan_agreement":
        try:
            document_text = extract_document_text(str(saved_path))
            terms = extract_loan_terms(document_text)
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
    account = store.get_account(account_id)
    if account is None:
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


@app.get("/escalations", response_model=list[EscalationOut], dependencies=[Depends(require_api_key)])
def list_open_escalations():
    """The human-handoff queue: every escalation still waiting on a
    person, oldest first."""
    return [_escalation_out(e) for e in store.list_open_escalations()]


def _escalation_out(escalation) -> EscalationOut:
    return EscalationOut(
        escalation_id=escalation.escalation_id, account_id=escalation.account_id,
        reason=escalation.reason, status=escalation.status,
        created_at=escalation.created_at, resolved_at=escalation.resolved_at,
        proposed_changes=escalation.proposed_changes, resolution_reason=escalation.resolution_reason,
    )


@app.post(
    "/escalations/{escalation_id}/approve", response_model=EscalationOut, dependencies=[Depends(require_api_key)]
)
async def approve_escalation(escalation_id: str):
    """A human clicking Approve on a structured restructuring request:
    applies the real proposed_changes to the account (see accounts/
    store.py's approve_restructuring -- the only place in this system
    that actually commits one) and, if the borrower has a linked
    Telegram chat, sends them the real new terms."""
    try:
        result = store.approve_restructuring(escalation_id)
    except store.EscalationNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except store.EscalationAlreadyResolvedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    message = (
        f"Good news -- your recent request has been approved. Your loan now has "
        f"{result['new_months_remaining']} months remaining, with a new EMI of "
        f"₹{result['new_emi_amount']:,.2f}."
    )
    await notify.notify_restructuring_decision(result["account_id"], approved=True, message=message)

    escalation = store.get_escalation(escalation_id)
    return _escalation_out(escalation)


@app.post(
    "/escalations/{escalation_id}/reject", response_model=EscalationOut, dependencies=[Depends(require_api_key)]
)
async def reject_escalation(escalation_id: str, body: EscalationRejectIn):
    """A human clicking Reject: marks the escalation rejected (the
    account itself is never touched -- nothing was ever applied to it)
    and, if the borrower has a linked Telegram chat, tells them, including
    the optional reason if ops entered one."""
    try:
        result = store.reject_restructuring(escalation_id, body.reason)
    except store.EscalationNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except store.EscalationAlreadyResolvedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    message = "Your recent request could not be approved."
    if result["reason"]:
        message += f" Reason: {result['reason']}"
    await notify.notify_restructuring_decision(result["account_id"], approved=False, message=message)

    escalation = store.get_escalation(escalation_id)
    return _escalation_out(escalation)


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


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(_STATIC_DIR / "index.html")


# Registered last so every API route above (exact paths like /accounts,
# /metrics, /escalations/...) is matched first -- this only serves
# /static/styles.css, /static/app.js, etc.
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
