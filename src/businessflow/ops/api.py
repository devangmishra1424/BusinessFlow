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

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from businessflow.accounts import store
from businessflow.accounts.models import Account
from businessflow.observability import metrics
from businessflow.ops.flags import Flag, compute_flags

app = FastAPI(title="BusinessFlow Ops API")

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
