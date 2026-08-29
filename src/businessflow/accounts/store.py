"""Account store, backed by real Postgres (Supabase) tables instead of the
in-memory dict this started as. Public function signatures are unchanged
from the mock -- the 8 tools that call these never needed to change for
this swap, only the internals did.

current_date() returns the real calendar date. The demo accounts'
days-past-due/grace-period story stays intact regardless of when this
runs because scripts/seed_accounts.py seeds every date relative to
date.today() at seed time, not a fixed literal -- so re-seeding (e.g.
the morning of a demo) keeps this function's real, live "today" in sync
with the story, instead of the two silently drifting apart the longer a
frozen anchor sat unseeded.
"""

import calendar
import json
import logging
import secrets
from datetime import date, datetime, timedelta, timezone

import psycopg

from businessflow.accounts.db import get_connection
from businessflow.accounts.models import Account, Escalation, PaymentRecord, PromiseToPay
from businessflow.accounts.policy import PAYMENT_TOKEN_TTL_HOURS

logger = logging.getLogger(__name__)


def _load_payment_history(account_id: str) -> list[PaymentRecord]:
    rows = get_connection().execute(
        "select payment_date, amount, on_time from payment_history where account_id = %s order by payment_date",
        (account_id,),
    ).fetchall()
    return [PaymentRecord(date=r["payment_date"], amount=float(r["amount"]), on_time=r["on_time"]) for r in rows]


def _load_promises(account_id: str) -> list[PromiseToPay]:
    rows = get_connection().execute(
        "select made_on, promised_date, promised_amount, kept from promises where account_id = %s order by promised_date",
        (account_id,),
    ).fetchall()
    return [
        PromiseToPay(made_on=r["made_on"], promised_date=r["promised_date"], promised_amount=float(r["promised_amount"]), kept=r["kept"])
        for r in rows
    ]


def _row_to_account(row: dict) -> Account:
    return Account(
        account_id=row["account_id"],
        borrower_name=row["borrower_name"],
        business_name=row["business_name"],
        phone_number=row["phone_number"],
        language_preference=row["language_preference"],
        loan_type=row["loan_type"],
        principal_amount=float(row["principal_amount"]),
        emi_amount=float(row["emi_amount"]),
        tenure_months=row["tenure_months"],
        months_remaining=row["months_remaining"],
        emi_due_date=row["emi_due_date"],
        nach_mandate_active=row["nach_mandate_active"],
        dispute_open=row["dispute_open"],
        risk_tier=row["risk_tier"],
        interest_rate_pct=float(row["interest_rate_pct"]) if row["interest_rate_pct"] is not None else None,
        telegram_chat_id=row["telegram_chat_id"],
        payment_history=_load_payment_history(row["account_id"]),
        promises=_load_promises(row["account_id"]),
    )


def get_account(account_id: str) -> Account | None:
    row = get_connection().execute("select * from accounts where account_id = %s", (account_id,)).fetchone()
    return _row_to_account(row) if row else None


def get_account_or_raise(account_id: str) -> Account:
    account = get_account(account_id)
    if account is None:
        raise ValueError(f"No account found for account_id={account_id!r}")
    return account


def list_accounts() -> list[Account]:
    rows = get_connection().execute("select * from accounts order by account_id").fetchall()
    return [_row_to_account(r) for r in rows]


def get_account_by_phone(phone_number: str) -> Account | None:
    row = get_connection().execute("select * from accounts where phone_number = %s", (phone_number,)).fetchone()
    return _row_to_account(row) if row else None


def _next_account_id() -> str:
    """BF-1001, BF-1002, ... -- the next id after whatever's already the
    highest numbered account. Computed in Python from the existing rows
    rather than a real Postgres sequence (like escalation_seq) -- ops
    account creation is a rare, deliberate, one-at-a-time human action,
    not a hot concurrent path, so the small window between this read and
    the insert below racing another creation isn't worth a schema change
    to close. Starts at BF-1001 if the table is somehow empty."""
    row = get_connection().execute(
        "select account_id from accounts where account_id ~ '^BF-[0-9]+$' order by (substring(account_id from 4))::int desc limit 1"
    ).fetchone()
    if row is None:
        return "BF-1001"
    next_number = int(row["account_id"].removeprefix("BF-")) + 1
    return f"BF-{next_number:04d}"


def create_account(
    borrower_name: str, business_name: str, phone_number: str, language_preference: str,
    loan_type: str, principal_amount: float, emi_amount: float, tenure_months: int,
    emi_due_date: date, nach_mandate_active: bool, risk_tier: str,
) -> tuple[Account, str]:
    """Opens a brand-new loan account -- months_remaining starts equal to
    tenure_months (nothing paid down yet) and dispute_open starts False;
    every other real-world field (payment_history, promises, flags) only
    accumulates from here through the account's normal real activity, the
    same way a real newly-issued loan would. Returns (account, access_key):
    access_key is a fresh random 6-digit PIN (this system's stand-in for a
    real sign-up flow, same as every seeded demo account's) -- it isn't a
    field on Account itself (see verify_account_key, which queries the
    column directly), so this is the only place it's ever handed back;
    the caller (ops/api.py) is responsible for showing it to whoever's
    opening this account."""
    account_id = _next_account_id()
    access_key = f"{secrets.randbelow(1_000_000):06d}"
    get_connection().execute(
        """
        insert into accounts (
            account_id, borrower_name, business_name, phone_number, language_preference,
            loan_type, principal_amount, emi_amount, tenure_months, months_remaining,
            emi_due_date, nach_mandate_active, dispute_open, risk_tier, access_key
        ) values (
            %(account_id)s, %(borrower_name)s, %(business_name)s, %(phone_number)s, %(language_preference)s,
            %(loan_type)s, %(principal_amount)s, %(emi_amount)s, %(tenure_months)s, %(tenure_months)s,
            %(emi_due_date)s, %(nach_mandate_active)s, false, %(risk_tier)s, %(access_key)s
        )
        """,
        {
            "account_id": account_id, "borrower_name": borrower_name, "business_name": business_name,
            "phone_number": phone_number, "language_preference": language_preference, "loan_type": loan_type,
            "principal_amount": principal_amount, "emi_amount": emi_amount, "tenure_months": tenure_months,
            "emi_due_date": emi_due_date, "nach_mandate_active": nach_mandate_active, "risk_tier": risk_tier,
            "access_key": access_key,
        },
    )
    return get_account_or_raise(account_id), access_key


def current_date() -> date:
    """The single seam tools call through for 'today' -- the real
    calendar date, in UTC specifically, not date.today()'s local
    timezone. Real, found-live reason: outbound/run.py's
    _already_sent_today() combines this date with tzinfo=UTC to build a
    "since midnight" cutoff compared against events.created_at (a real
    Postgres timestamptz, itself UTC) -- on a host running any timezone
    ahead of UTC (confirmed live in IST, UTC+5:30, during the ~5.5-hour
    window after local midnight but before UTC midnight), date.today()
    already reads as tomorrow while UTC is still today, producing a
    since-midnight cutoff that's hours in the future relative to any
    real event -- so the "already sent today" check could never find a
    match and would resend the same reminder every time it's called.
    See scripts/seed_accounts.py for why the demo data stays consistent
    with this rather than pinning it to a fixed date."""
    return datetime.now(timezone.utc).date()


def add_promise(account_id: str, made_on: date, promised_date: date, promised_amount: float) -> bool:
    """Returns True if a new promise row was inserted, False if an
    identical one (same account, made today, same promised date/amount)
    already existed -- a duplicate tool call (a retry, or the model
    calling this twice in one turn) logs the promise once, not twice.

    This is a plain check-then-act, not a DB-enforced constraint -- fine
    for the realistic threat here (one conversation's sequential tool
    calls), not a guarantee against genuinely concurrent writers racing
    on the exact same promise. A unique index would be the fuller fix
    if that ever becomes a real risk."""
    existing = get_connection().execute(
        "select id from promises where account_id = %s and made_on = %s "
        "and promised_date = %s and promised_amount = %s",
        (account_id, made_on, promised_date, promised_amount),
    ).fetchone()
    if existing:
        return False
    get_connection().execute(
        "insert into promises (account_id, made_on, promised_date, promised_amount) values (%s, %s, %s, %s)",
        (account_id, made_on, promised_date, promised_amount),
    )
    return True


def open_dispute(account_id: str, reason: str) -> bool:
    """Returns True if this call actually opened a new dispute, False if
    the account already had one open -- flagging an already-disputed
    account is a no-op, not a second dispute-log entry. Same
    check-then-act caveat as add_promise above."""
    conn = get_connection()
    row = conn.execute("select dispute_open from accounts where account_id = %s", (account_id,)).fetchone()
    if row and row["dispute_open"]:
        return False
    conn.execute("update accounts set dispute_open = true, updated_at = now() where account_id = %s", (account_id,))
    conn.execute("insert into disputes (account_id, reason) values (%s, %s)", (account_id, reason))
    return True


def get_disputes_for_account(account_id: str) -> list[dict]:
    """Every dispute ever opened on this account, newest first -- the
    real reason text a borrower (or a warning's "Contest" action) gave,
    which flags.py's own "disputed" flag never carries (it's a fixed
    generic string, not this). Ops has had no way to see WHY an account
    is disputed without a direct database query until this existed."""
    rows = get_connection().execute(
        "select reason, status, opened_at, resolved_at from disputes where account_id = %s order by opened_at desc",
        (account_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_latest_open_dispute_reason(account_id: str) -> str | None:
    """The single most recent still-open dispute's real reason -- what
    flags.py's compute_flags uses instead of a generic "has an open,
    unresolved dispute" string, so an ops operator glancing at the flags
    list sees the actual claim, not just that one exists."""
    row = get_connection().execute(
        "select reason from disputes where account_id = %s and status = 'open' order by opened_at desc limit 1",
        (account_id,),
    ).fetchone()
    return row["reason"] if row else None


def set_interest_rate_pct(account_id: str, interest_rate_pct: float) -> None:
    """Writes a real, extracted interest rate onto the account row --
    called once, right after a loan_agreement upload's structured
    extraction pass (rag/extraction.py's extract_loan_terms) finds a
    real, non-null rate. Never called with None: "no rate extracted"
    just leaves the column at its existing value (NULL until a real
    agreement is uploaded and parsed) rather than writing a null here."""
    get_connection().execute(
        "update accounts set interest_rate_pct = %s, updated_at = now() where account_id = %s",
        (interest_rate_pct, account_id),
    )


def create_escalation(account_id: str, reason: str, proposed_changes: dict | None = None) -> str:
    """Idempotent against an exact-duplicate call: if an unresolved
    escalation with this same account_id + reason already exists,
    returns its existing escalation_id instead of opening a second
    ticket for the same thing. A genuinely different reason still opens
    a new escalation -- this only collapses true repeats.

    proposed_changes is not part of that dedup check -- callers that pass
    it (see tools/escalation_tools.py's propose_restructuring) build reason
    text that already embeds the real proposed numbers, so two genuinely
    different proposals never share a reason string in the first place."""
    conn = get_connection()
    existing = conn.execute(
        "select escalation_id from escalations where account_id = %s and reason = %s and resolved_at is null",
        (account_id, reason),
    ).fetchone()
    if existing:
        return existing["escalation_id"]
    seq_number = conn.execute("select nextval('escalation_seq')").fetchone()["nextval"]
    escalation_id = f"ESC-{seq_number:04d}"
    conn.execute(
        "insert into escalations (escalation_id, account_id, reason, proposed_changes) values (%s, %s, %s, %s)",
        (
            escalation_id, account_id, reason,
            json.dumps(proposed_changes, ensure_ascii=False, default=str) if proposed_changes is not None else None,
        ),
    )
    return escalation_id


def _row_to_escalation(row: dict) -> Escalation:
    return Escalation(
        escalation_id=row["escalation_id"],
        account_id=row["account_id"],
        reason=row["reason"],
        status=row["status"],
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
        proposed_changes=row.get("proposed_changes"),
        resolution_reason=row.get("resolution_reason"),
    )


class EscalationNotFoundError(ValueError):
    pass


class EscalationAlreadyResolvedError(ValueError):
    pass


def approve_restructuring(escalation_id: str) -> dict:
    """The only place in this system that actually commits a
    restructuring: applies an escalation's proposed_changes to the real
    account row, then marks the escalation resolved. Re-checks
    resolved_at rather than blindly re-applying, so a double-click or a
    retried request can't double-apply the same change -- raises instead
    of silently no-op'ing a second time, since silently returning
    "success" on a stale retry could make an operator think a second,
    different approval landed when nothing happened.

    Known limitation, not solved here: this applies proposed_changes
    exactly as they were computed at proposal time. If the account's real
    state changed in between (a payment posted, a dispute opened), this
    does not re-validate against the CURRENT state -- a real deployment
    handling money for real would want that; this demo doesn't."""
    conn = get_connection()
    row = conn.execute(
        "select account_id, proposed_changes, resolved_at from escalations where escalation_id = %s",
        (escalation_id,),
    ).fetchone()
    if row is None:
        raise EscalationNotFoundError(f"No escalation found for escalation_id={escalation_id!r}")
    if row["resolved_at"] is not None:
        raise EscalationAlreadyResolvedError(f"escalation_id={escalation_id!r} is already resolved")

    changes = row["proposed_changes"]
    if not changes:
        raise ValueError(f"escalation_id={escalation_id!r} has no proposed_changes to apply")

    account_id = row["account_id"]
    if changes.get("type") == "extend_tenure":
        conn.execute(
            "update accounts set months_remaining = %s, emi_amount = %s, updated_at = now() where account_id = %s",
            (changes["new_months_remaining"], changes["new_emi_amount"], account_id),
        )
    else:
        raise ValueError(f"escalation_id={escalation_id!r} has an unknown proposed_changes type {changes.get('type')!r}")

    conn.execute(
        "update escalations set status = 'approved', resolved_at = now() where escalation_id = %s",
        (escalation_id,),
    )
    return {"escalation_id": escalation_id, "account_id": account_id, **changes}


def reject_restructuring(escalation_id: str, reason: str | None) -> dict:
    """Marks the escalation rejected -- never touches the account row,
    since nothing was ever applied to it in the first place. reason is
    optional, ops-entered free text shown back to the borrower (see
    outbound/send.py's notify_restructuring_decision)."""
    conn = get_connection()
    row = conn.execute(
        "select account_id, resolved_at from escalations where escalation_id = %s",
        (escalation_id,),
    ).fetchone()
    if row is None:
        raise EscalationNotFoundError(f"No escalation found for escalation_id={escalation_id!r}")
    if row["resolved_at"] is not None:
        raise EscalationAlreadyResolvedError(f"escalation_id={escalation_id!r} is already resolved")

    conn.execute(
        "update escalations set status = 'rejected', resolved_at = now(), resolution_reason = %s where escalation_id = %s",
        (reason, escalation_id),
    )
    return {"escalation_id": escalation_id, "account_id": row["account_id"], "reason": reason}


def set_telegram_chat_id(account_id: str, chat_id: int) -> None:
    """Records the Telegram chat this account most recently verified
    from -- called once, right after a successful credential check in
    channels/telegram_bot.py's handle_incoming_message. Last-verified-
    chat-wins by design (see the column's comment in schema.sql)."""
    get_connection().execute(
        "update accounts set telegram_chat_id = %s, updated_at = now() where account_id = %s",
        (chat_id, account_id),
    )


def record_payment(account_id: str, amount: float, payment_date: date | None = None) -> dict:
    """The one place a payment actually LANDS: every other payment_history
    row in this system is seed data, never a live event this app itself
    produced. Called only from redeem_payment_token below -- never
    reachable directly from a borrower-facing tool, since a payment must
    always be gated by a real, single-use token (see that function's own
    docstring for why).

    Real reducing-balance EMI semantics, kept deliberately simple to match
    this project's existing "one flat EMI cuts one month" model (same
    simplification calculate_hypothetical and get_payment_status's
    outstanding_balance_approx already rely on): one payment always
    retires exactly one month, regardless of the amount actually paid.
    months_remaining floors at 0 rather than going negative on a stray
    extra payment against an already-closed loan."""
    if payment_date is None:
        payment_date = current_date()
    conn = get_connection()
    account = get_account_or_raise(account_id)

    on_time = payment_date <= account.emi_due_date
    conn.execute(
        "insert into payment_history (account_id, payment_date, amount, on_time) values (%s, %s, %s, %s)",
        (account_id, payment_date, amount, on_time),
    )
    new_months_remaining = max(0, account.months_remaining - 1)
    next_due_date = add_one_month(account.emi_due_date)
    conn.execute(
        "update accounts set months_remaining = %s, emi_due_date = %s, updated_at = now() where account_id = %s",
        (new_months_remaining, next_due_date, account_id),
    )
    return {
        "account_id": account_id,
        "amount": amount,
        "payment_date": payment_date.isoformat(),
        "on_time": on_time,
        "months_remaining": new_months_remaining,
        "next_emi_due_date": next_due_date.isoformat(),
    }


def add_one_month(d: date) -> date:
    """Calendar-correct month rollover (Jan 31 + 1mo = Feb 28, not a spill
    into March) -- deliberately NOT "add 30 days," which drifts the due
    date earlier every few cycles. The one shared implementation: this
    used to be duplicated privately in channels/browser_api.py for its
    projected-EMI-timeline display math (a DIFFERENT use from
    record_payment's real one here, but the same calendar arithmetic) --
    that copy now imports this one instead of keeping its own."""
    if d.month == 12:
        year, month = d.year + 1, 1
    else:
        year, month = d.year, d.month + 1
    last_day_of_month = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day_of_month))


class PaymentTokenNotFoundError(ValueError):
    pass


class PaymentTokenExpiredError(ValueError):
    pass


class PaymentTokenAlreadyUsedError(ValueError):
    pass


def create_payment_token(account_id: str, amount: float) -> str:
    """Mints a real, single-use, expiring link for one specific payment --
    the account_id and amount are baked into this server-side row, never
    trusted from the URL itself, so a borrower can only ever pay exactly
    what this token was minted for for their own account (see
    redeem_payment_token). token_urlsafe(24) gives ~144 bits of entropy --
    not realistically guessable, the same class of unpredictability a
    real payment provider's own one-time link would rely on."""
    token = secrets.token_urlsafe(24)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=PAYMENT_TOKEN_TTL_HOURS)
    get_connection().execute(
        "insert into payment_tokens (token, account_id, amount, expires_at) values (%s, %s, %s, %s)",
        (token, account_id, amount, expires_at),
    )
    return token


def get_payment_token_info(token: str) -> dict | None:
    """Read-only lookup for the confirmation PAGE itself (before the
    borrower has clicked Confirm) -- returns the real account/amount/
    status so the page can render "Confirm ₹X for [business]" without
    ever redeeming anything. None only for a token that never existed at
    all; an expired or already-used token still returns its real status
    rather than None, so the page can say WHY it can't be paid instead of
    a generic "not found\" for every case alike."""
    row = get_connection().execute(
        """
        select pt.account_id, pt.amount, pt.expires_at, pt.used_at, a.business_name, a.borrower_name
        from payment_tokens pt join accounts a on a.account_id = pt.account_id
        where pt.token = %s
        """,
        (token,),
    ).fetchone()
    if row is None:
        return None
    now = datetime.now(timezone.utc)
    if row["used_at"] is not None:
        status = "used"
    elif row["expires_at"] < now:
        status = "expired"
    else:
        status = "pending"
    return {
        "account_id": row["account_id"],
        "amount": float(row["amount"]),
        "business_name": row["business_name"],
        "borrower_name": row["borrower_name"],
        "status": status,
    }


def redeem_payment_token(token: str) -> dict:
    """The only path by which a "pay now" link can actually move the
    account forward -- confirming re-checks used_at/expires_at rather
    than trusting the caller already checked via get_payment_token_info
    (that read can go stale between page-load and the borrower's click),
    so a double-submit or a replayed request can never record two
    payments for one token."""
    conn = get_connection()
    row = conn.execute(
        "select account_id, amount, expires_at, used_at from payment_tokens where token = %s",
        (token,),
    ).fetchone()
    if row is None:
        raise PaymentTokenNotFoundError(f"no payment token found for token={token!r}")
    if row["used_at"] is not None:
        raise PaymentTokenAlreadyUsedError("this payment link has already been used")
    if row["expires_at"] < datetime.now(timezone.utc):
        raise PaymentTokenExpiredError("this payment link has expired")

    conn.execute("update payment_tokens set used_at = now() where token = %s", (token,))
    return record_payment(row["account_id"], float(row["amount"]))


_ESCALATION_COLUMNS = "escalation_id, account_id, reason, status, created_at, resolved_at, proposed_changes, resolution_reason"


def get_escalations_for_account(account_id: str) -> list[Escalation]:
    rows = get_connection().execute(
        f"select {_ESCALATION_COLUMNS} from escalations where account_id = %s order by created_at desc",
        (account_id,),
    ).fetchall()
    return [_row_to_escalation(r) for r in rows]


def get_escalation(escalation_id: str) -> Escalation | None:
    row = get_connection().execute(
        f"select {_ESCALATION_COLUMNS} from escalations where escalation_id = %s",
        (escalation_id,),
    ).fetchone()
    return _row_to_escalation(row) if row else None


def list_open_escalations() -> list[Escalation]:
    """The ops queue: every escalation still waiting on a human, oldest
    first -- the order a human should actually work through them in."""
    rows = get_connection().execute(
        f"select {_ESCALATION_COLUMNS} from escalations where status = 'queued_for_human' order by created_at",
    ).fetchall()
    return [_row_to_escalation(r) for r in rows]


def count_recent_events(account_id: str, event_type: str, since: datetime) -> int:
    row = get_connection().execute(
        "select count(*) as n from events where account_id = %s and event_type = %s and created_at >= %s",
        (account_id, event_type, since),
    ).fetchone()
    return row["n"]


def has_recent_event_with_detail(account_id: str, event_type: str, since: datetime, detail_key: str, detail_value) -> bool:
    """True if a matching event already exists -- the idempotency check
    the proactive outbound pass uses to avoid re-sending the same
    reminder kind to the same account twice in one day."""
    row = get_connection().execute(
        "select 1 from events where account_id = %s and event_type = %s and created_at >= %s "
        "and details->>%s = %s limit 1",
        (account_id, event_type, since, detail_key, str(detail_value)),
    ).fetchone()
    return row is not None


def get_clarification_requests(account_id: str) -> list[dict]:
    """Every clarification-request message ever sent to this account,
    most recent first -- the ops dashboard's communication-history
    thread for that borrower (see outbound/send.py's
    notify_clarification_request, which is what actually writes these
    events)."""
    rows = get_connection().execute(
        "select details, created_at from events where account_id = %s and event_type = %s order by created_at desc",
        (account_id, "clarification_request_sent"),
    ).fetchall()
    return [
        {
            "message": r["details"]["message"],
            "delivered_via_telegram": r["details"]["delivered_via_telegram"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def verify_account_key(account_id: str, access_key: str) -> bool:
    """Checks a borrower-supplied key against the account's fixed PIN
    (assigned at seed time -- there's no real sign-up flow here). Used
    once, at conversation start, to gate access to that account's data;
    never returns the key itself, only whether it matched."""
    row = get_connection().execute(
        "select access_key from accounts where account_id = %s", (account_id,)
    ).fetchone()
    return row is not None and row["access_key"] == access_key


def log_event(account_id: str | None, event_type: str, details: dict) -> None:
    """account_id is a foreign key -- a value that doesn't correspond to
    a real account (e.g. a hallucinated account_id the model passed to
    a tool that doesn't itself validate it, like check_policy) would
    otherwise crash this insert outright. A logging side-effect failing
    must never take down a tool call that already succeeded (or a
    failure event trying to record why a DIFFERENT call failed), so this
    falls back to logging with no specific borrower instead of losing
    the event entirely -- found live, when exactly this happened during
    eval/reasoning_accuracy.py's general-Q&A scenario."""
    payload = json.dumps(details, ensure_ascii=False, default=str)
    try:
        get_connection().execute(
            "insert into events (account_id, event_type, details) values (%s, %s, %s)",
            (account_id, event_type, payload),
        )
    except psycopg.errors.ForeignKeyViolation:
        logger.warning(
            "log_event: account_id=%r doesn't exist -- logging event_type=%r with no specific borrower instead",
            account_id, event_type,
        )
        get_connection().execute(
            "insert into events (account_id, event_type, details) values (%s, %s, %s)",
            (None, event_type, payload),
        )
