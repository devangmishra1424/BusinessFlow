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

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from businessflow.accounts import store
from businessflow.accounts.models import Account
from businessflow.observability import metrics
from businessflow.ops.flags import Flag, compute_flags
from businessflow.rag.ingest import ingest_document

app = FastAPI(title="BusinessFlow Ops API")

# Where ops-uploaded per-account documents (signed loan agreements, KYC,
# etc.) live on disk -- the permanent source file ingest_document() parses,
# analogous to data/kb/ holding the general policy docs' source files.
_DOCUMENTS_DIR = Path(__file__).resolve().parents[3] / "data" / "documents"

# What ingest.py's Docling call actually supports (see its docstring) --
# rejecting anything else here with a clear 400 beats letting
# DocumentConverter fail obscurely deep inside ingest_document.
_ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".md"}

# A real, deliberate bound: an unbounded upload accepted into memory/disk is
# a real (if modest) DoS surface. Enforced by counting bytes actually read,
# not by trusting the client-supplied Content-Length header.
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_header = APIKeyHeader(name="X-API-Key")


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


class MetricsOut(BaseModel):
    since_hours: float
    event_counts: dict[str, int]
    escalation_rate: float


class EscalationOut(BaseModel):
    escalation_id: str
    reason: str
    status: str
    created_at: datetime
    resolved_at: datetime | None


class DocumentUploadOut(BaseModel):
    account_id: str
    document_type: str
    filename: str
    chunks_stored: int


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
    escalations: list[EscalationOut]


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
                escalation_id=e.escalation_id, reason=e.reason, status=e.status,
                created_at=e.created_at, resolved_at=e.resolved_at,
            )
            for e in escalations
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
    _retriever() is @lru_cache(maxsize=1)'d PER PROCESS. This document is
    in the persistent Chroma store the instant this call returns, but a
    currently-running borrower-facing process (channels/browser_api.py,
    channels/telegram_bot.py) won't see it until THAT process restarts --
    its cached DocumentRetriever snapshot isn't rebuilt automatically.
    There's no pub/sub or other cross-process cache invalidation here; that
    would be real new infra and is out of scope for this endpoint.
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
        saved_path.unlink(missing_ok=True)
        # account_dir was just created above (mkdir) for this upload -- if
        # this was the account's first-ever upload attempt and it got
        # rejected, don't leave an empty directory behind. rmdir() only
        # succeeds on an empty directory, so this is a no-op (not an
        # error) when the account already has other real documents saved.
        try:
            account_dir.rmdir()
        except OSError:
            pass
        raise

    chunks_stored = ingest_document(str(saved_path), document_type=document_type, account_id=account_id)

    return DocumentUploadOut(
        account_id=account_id, document_type=document_type, filename=filename, chunks_stored=chunks_stored
    )


@app.get("/escalations", response_model=list[EscalationOut], dependencies=[Depends(require_api_key)])
def list_open_escalations():
    """The human-handoff queue: every escalation still waiting on a
    person, oldest first."""
    return [
        EscalationOut(
            escalation_id=e.escalation_id, reason=e.reason, status=e.status,
            created_at=e.created_at, resolved_at=e.resolved_at,
        )
        for e in store.list_open_escalations()
    ]


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
