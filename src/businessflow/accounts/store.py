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

from businessflow.accounts import cases
from businessflow.accounts.db import get_connection
from businessflow.accounts.dues import late_fee_due
from businessflow.accounts.escalation_kinds import classify_escalation
from businessflow.accounts.models import Account, Escalation, PaymentRecord, PromiseToPay
from businessflow.accounts.policy import PAYMENT_TOKEN_TTL_HOURS, PROMISE_TOLERANCE_DAYS

logger = logging.getLogger(__name__)


# Every read below takes an optional `ex`: an open connection (from get_connection().transaction()) to run on instead
# of checking out a pooled one. Found live: the database is far from the server (the VM talks to Sydney), and each
# pooled checkout pings the connection before the query, so every separate read costs several round trips. The ops
# portfolio of 10 accounts took 12 s and one account's page 5 s, purely from those round trips. Reads that belong to
# one screen now share one connection.

def _payment_record(r: dict) -> PaymentRecord:
    return PaymentRecord(date=r["payment_date"], amount=float(r["amount"]), on_time=r["on_time"], kind=r["kind"])


def _promise(r: dict) -> PromiseToPay:
    return PromiseToPay(made_on=r["made_on"], promised_date=r["promised_date"], promised_amount=float(r["promised_amount"]), kept=r["kept"])


def _load_payment_history(account_id: str, ex=None) -> list[PaymentRecord]:
    rows = (ex or get_connection()).execute(
        "select payment_date, amount, on_time, kind from payment_history where account_id = %s order by payment_date",
        (account_id,),
    ).fetchall()
    return [_payment_record(r) for r in rows]


def _load_promises(account_id: str, ex=None) -> list[PromiseToPay]:
    rows = (ex or get_connection()).execute(
        "select made_on, promised_date, promised_amount, kept from promises where account_id = %s order by promised_date",
        (account_id,),
    ).fetchall()
    return [_promise(r) for r in rows]


def _row_to_account(row: dict, with_history: bool = True, ex=None) -> Account:
    """with_history=False skips the payment-history and promise lookups: they take
    their own pooled connections, which a caller holding a transaction must not do
    (see _lock_account) -- unless that caller passes its connection as `ex`."""
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
        pending_emi_credit=float(row["pending_emi_credit"]),
        interest_rate_pct=float(row["interest_rate_pct"]) if row["interest_rate_pct"] is not None else None,
        telegram_chat_id=row["telegram_chat_id"],
        payment_history=_load_payment_history(row["account_id"], ex) if with_history else [],
        promises=_load_promises(row["account_id"], ex) if with_history else [],
    )


def get_account(account_id: str, ex=None) -> Account | None:
    row = (ex or get_connection()).execute("select * from accounts where account_id = %s", (account_id,)).fetchone()
    return _row_to_account(row, ex=ex) if row else None


def account_exists(account_id: str) -> bool:
    """One small query, for an endpoint that only needs to 404 on an unknown id (loading the whole account for that
    costs two more round trips)."""
    return get_connection().execute("select 1 from accounts where account_id = %s", (account_id,)).fetchone() is not None


def get_account_or_raise(account_id: str) -> Account:
    account = get_account(account_id)
    if account is None:
        raise ValueError(f"No account found for account_id={account_id!r}")
    return account


def list_accounts() -> list[Account]:
    """Every account with its payment history and promises, in three queries on one connection (not two more per
    account: see the note above _payment_record). The three reads share one transaction, so they see one consistent
    state of the books."""
    with get_connection().transaction() as conn:
        rows = conn.execute("select * from accounts order by account_id").fetchall()
        ids = [r["account_id"] for r in rows]
        history_rows = conn.execute(
            "select account_id, payment_date, amount, on_time, kind from payment_history "
            "where account_id = any(%s) order by account_id, payment_date",
            (ids,),
        ).fetchall()
        promise_rows = conn.execute(
            "select account_id, made_on, promised_date, promised_amount, kept from promises "
            "where account_id = any(%s) order by account_id, promised_date",
            (ids,),
        ).fetchall()
    history: dict[str, list[PaymentRecord]] = {i: [] for i in ids}
    for r in history_rows:
        history[r["account_id"]].append(_payment_record(r))
    promises: dict[str, list[PromiseToPay]] = {i: [] for i in ids}
    for r in promise_rows:
        promises[r["account_id"]].append(_promise(r))
    accounts = []
    for r in rows:
        account = _row_to_account(r, with_history=False)
        account.payment_history = history[r["account_id"]]
        account.promises = promises[r["account_id"]]
        accounts.append(account)
    return accounts


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


def reset_access_key(account_id: str) -> str:
    """Mints a fresh 6-digit access key for an EXISTING account, replacing
    whatever it had -- the old key stops working the instant this commits.
    Same generation as create_account's own access_key (secrets.randbelow,
    zero-padded to 6 digits), just against a row that already exists.

    Real gap this closes: create_account only ever hands the key back
    ONCE, at creation (see its own docstring) -- there was previously no
    way to recover a lost key, or to get a fresh Telegram deep-link
    (build_telegram_start_payload) for an account opened before that
    feature existed, without a direct database query."""
    if get_account(account_id) is None:
        raise ValueError(f"No account found for account_id={account_id!r}")
    access_key = f"{secrets.randbelow(1_000_000):06d}"
    get_connection().execute(
        "update accounts set access_key = %s, updated_at = now() where account_id = %s",
        (access_key, account_id),
    )
    return access_key


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


def resolve_matured_promises() -> list[dict]:
    """Evaluates every promise-to-pay whose evaluation window has passed
    (promised_date + PROMISE_TOLERANCE_DAYS is before today) and still has
    kept=null, against real payment_history, and writes the real verdict.

    Found live (this codebase's own README names it explicitly, under
    "Known gaps"): promises.kept was only ever written once, by
    scripts/seed_accounts.py at seed time -- no code path anywhere in this
    app ever evaluated a promise made after that and flipped it true/false.
    That silently froze Account.broken_promise_count() (used by ops/
    flags.py's broken_promises flag and the mandatory-restructuring-
    escalation policy) at whatever the seed data happened to show,
    regardless of what actually happened afterward. Called from outbound/
    run.py's daily pass -- this is now a real, live evaluation, not a
    one-time seed artifact.

    A promise counts as kept if a real payment for at least the promised
    amount landed on or after it was made and on or before the tolerance
    deadline -- a single matching payment_history row, not a sum across
    several partial payments (this project's existing "simple, explainable
    rule over black-box scoring" style -- see SETTLEMENT_DISCOUNT_PCT's own
    comment in policy.py for the same tradeoff made elsewhere). Returns
    every {id, account_id, kept} row this call just resolved, so callers
    (run.py) can react to newly-broken promises without a second query."""
    rows = get_connection().execute(
        """
        update promises p
        set kept = exists (
            select 1 from payment_history ph
            where ph.account_id = p.account_id
              and ph.payment_date >= p.made_on
              and ph.payment_date <= p.promised_date + %(tolerance)s
              and ph.amount >= p.promised_amount
        )
        where p.kept is null
          and p.promised_date + %(tolerance)s < %(today)s
        returning p.id, p.account_id, p.kept
        """,
        {"tolerance": PROMISE_TOLERANCE_DAYS, "today": current_date()},
    ).fetchall()
    return [dict(r) for r in rows]


def open_dispute(account_id: str, reason: str, kind: str = cases.DISPUTE) -> dict:
    """Opens a dispute, or a payment claim (kind="payment_claim": the
    borrower says they already paid), and pauses automated reminders
    (accounts.dispute_open) until a person closes it.

    At most one is open at a time: if one already is, nothing new opens
    and the open one is returned -- the caller adds the borrower's new
    words to it as a message (see tools/account_tools.py's flag_dispute),
    rather than dropping them as this used to.

    Returns {"opened": bool, "dispute_id": int, "kind": str, "reason": str}
    -- all of the case now open (the existing one when opened is
    False). The account row is locked for the check and the insert, so two
    claims arriving together can't both open one."""
    if kind not in cases.DISPUTE_KINDS:
        raise ValueError(f"kind must be one of {cases.DISPUTE_KINDS}, got {kind!r}")
    with get_connection().transaction() as tx:
        account = tx.execute("select dispute_open from accounts where account_id = %s for update", (account_id,)).fetchone()
        if account is None:
            raise ValueError(f"No account found for account_id={account_id!r}")
        if account["dispute_open"]:
            existing = tx.execute(
                "select id, kind, reason from disputes where account_id = %s and status = 'open' order by opened_at desc limit 1",
                (account_id,),
            ).fetchone()
            if existing is not None:
                return {"opened": False, "dispute_id": existing["id"], "kind": existing["kind"], "reason": existing["reason"]}
            # dispute_open without an open row: the flag is stale (set before disputes rows existed). Open a real
            # one below rather than reporting a case that doesn't exist.
            logger.warning("open_dispute: account_id=%s had dispute_open set but no open dispute row; opening one", account_id)
        tx.execute("update accounts set dispute_open = true, updated_at = now() where account_id = %s", (account_id,))
        row = tx.execute(
            "insert into disputes (account_id, reason, kind) values (%s, %s, %s) returning id",
            (account_id, reason, kind),
        ).fetchone()
        return {"opened": True, "dispute_id": row["id"], "kind": kind, "reason": reason}


def get_disputes_for_account(account_id: str, ex=None) -> list[dict]:
    """Every dispute and payment claim ever opened on this account, newest
    first -- the real reason text a borrower gave, which flags.py's own
    "disputed" flag never carries (it's a fixed generic string, not this).
    Ops has had no way to see WHY an account is disputed without a direct
    database query until this existed."""
    rows = (ex or get_connection()).execute(
        "select id, kind, reason, status, opened_at, resolved_at, resolution_note from disputes "
        "where account_id = %s order by opened_at desc",
        (account_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _flag_reason(kind: str, reason: str) -> str:
    # A payment claim says what to check, not just that something is disputed.
    return f"Says they already paid: {reason}" if kind == cases.PAYMENT_CLAIM else reason


def get_latest_open_dispute_reason(account_id: str, ex=None) -> str | None:
    """The single most recent still-open dispute's real reason -- what
    flags.py's compute_flags uses instead of a generic "has an open,
    unresolved dispute" string, so an ops operator glancing at the flags
    list sees the actual claim, not just that one exists."""
    row = (ex or get_connection()).execute(
        "select kind, reason from disputes where account_id = %s and status = 'open' order by opened_at desc limit 1",
        (account_id,),
    ).fetchone()
    return _flag_reason(row["kind"], row["reason"]) if row else None


def latest_open_dispute_reasons(account_ids: list[str]) -> dict[str, str]:
    """get_latest_open_dispute_reason for many accounts in one query: {account_id: reason}, only for accounts with an
    open dispute. For the portfolio list, which would otherwise ask once per disputed account."""
    if not account_ids:
        return {}
    rows = get_connection().execute(
        "select distinct on (account_id) account_id, kind, reason from disputes "
        "where account_id = any(%s) and status = 'open' order by account_id, opened_at desc",
        (list(account_ids),),
    ).fetchall()
    return {r["account_id"]: _flag_reason(r["kind"], r["reason"]) for r in rows}


_MAX_OPEN_DISPUTES = 200  # an explicit bound for the ops inbox, far above any real queue here


def list_open_disputes(ex=None) -> list[dict]:
    """Every open dispute and payment claim across all accounts, oldest
    first, with the borrower's name -- the ops inbox's half that isn't an
    escalation. Bounded."""
    rows = (ex or get_connection()).execute(
        # The same columns as get_disputes_for_account (plus the name), so one builder serves both.
        "select d.id, d.account_id, d.kind, d.reason, d.status, d.opened_at, d.resolved_at, d.resolution_note, "
        "a.borrower_name from disputes d "
        "join accounts a on a.account_id = d.account_id where d.status = 'open' order by d.opened_at limit %s",
        (_MAX_OPEN_DISPUTES,),
    ).fetchall()
    return [dict(r) for r in rows]


class NoOpenDisputeError(ValueError):
    pass


def get_dispute(dispute_id: int, ex=None) -> dict | None:
    row = (ex or get_connection()).execute(
        "select id, account_id, kind, reason, status, opened_at, resolved_at, resolution_note from disputes where id = %s",
        (dispute_id,),
    ).fetchone()
    return dict(row) if row else None


def confirm_payment_claim(
    account_id: str,
    dispute_id: int,
    amount: float,
    payment_date: date | None = None,
    apply_extra_to_next: bool | None = None,
    resolution_note: str | None = None,
) -> dict:
    """Ops found the payment a borrower reported: records it (record_payment's
    rules, see there) and closes the claim, in one transaction -- the payment
    can't land without the claim closing, or the claim close without the
    payment. Raises NoOpenDisputeError when that claim isn't open any more (a
    second click, or a colleague got there first), so the payment is never
    recorded twice, and ValueError when the case isn't a payment claim on this
    account. Returns {"payment": <record_payment's dict>, "dispute_id": ...}."""
    with get_connection().transaction() as tx:
        _lock_account(tx, account_id)  # account first, as everywhere else, then the claim
        claim = tx.execute(
            "select id, kind, status from disputes where id = %s and account_id = %s for update",
            (dispute_id, account_id),
        ).fetchone()
        if claim is None:
            raise ValueError(f"no case D-{dispute_id} on account_id={account_id!r}")
        if claim["kind"] != cases.PAYMENT_CLAIM:
            raise ValueError(f"case D-{dispute_id} is a {claim['kind']}, not a payment claim")
        if claim["status"] != "open":
            raise NoOpenDisputeError(f"case D-{dispute_id} is already closed")
        payment = _record_payment_in(tx, account_id, amount, payment_date, apply_extra_to_next, None)
        tx.execute(
            "update disputes set status = 'resolved', resolved_at = now(), resolution_note = %s where id = %s",
            (resolution_note, dispute_id),
        )
        tx.execute("update accounts set dispute_open = false, updated_at = now() where account_id = %s", (account_id,))
    return {"payment": payment, "dispute_id": dispute_id, "kind": cases.PAYMENT_CLAIM, "resolution_note": resolution_note}


def resolve_dispute(account_id: str, resolution_note: str | None = None, dispute_id: int | None = None) -> dict:
    """Closes this account's currently open dispute -- open_dispute's own
    docstring guarantees there's ever at most one open at a time (a second
    open_dispute call while one's already open is a no-op), so this
    resolves that single row: disputes.status='resolved'/resolved_at=now(),
    and accounts.dispute_open=false so ops/flags.py's compute_flags stops
    firing the 'disputed' flag and the borrower's own dashboard stops
    showing it as an active claim.

    Found live: there was no resolve/close counterpart to open_dispute
    anywhere in this codebase at all -- once opened, a dispute (and the
    flag it drives) could never be turned off again by any code path,
    ops included, regardless of what an operator actually did about it.
    resolution_note closes a second gap found right after: resolving one
    only ever recorded THAT it happened, never the operator's own
    reasoning (confirmed with the borrower, late fee waived, etc.) -- so a
    later reader could see a dispute was closed but never why.

    Raises rather than silently no-op'ing when there's nothing open to
    resolve -- an operator clicking "Resolve" on an account that's
    already clean almost certainly means they're looking at stale data.

    dispute_id, when given (the ops case card always sends it), must be
    the open one: an operator closing case D-12 must not close a different
    case that opened since they loaded the page."""
    with get_connection().transaction() as tx:
        # The account row first, then the dispute: the same order as open_dispute and confirm_payment_claim, so two of
        # them on one account wait for each other instead of deadlocking.
        tx.execute("select 1 from accounts where account_id = %s for update", (account_id,))
        row = tx.execute(
            "select id, kind from disputes where account_id = %s and status = 'open' order by opened_at desc limit 1 for update",
            (account_id,),
        ).fetchone()
        if row is None or (dispute_id is not None and row["id"] != dispute_id):
            which = f"case D-{dispute_id}" if dispute_id is not None else "a dispute"
            raise NoOpenDisputeError(f"account_id={account_id!r} has no open {which} to resolve")

        tx.execute(
            "update disputes set status = 'resolved', resolved_at = now(), resolution_note = %s where id = %s",
            (resolution_note, row["id"]),
        )
        tx.execute("update accounts set dispute_open = false, updated_at = now() where account_id = %s", (account_id,))
    return {"account_id": account_id, "dispute_id": row["id"], "kind": row["kind"], "resolution_note": resolution_note}


_CALL_LOG_OUTCOMES = ("reached", "no_answer", "voicemail", "wrong_number")


def log_call(account_id: str, outcome: str, note: str | None) -> None:
    """Records that ops actually attempted to reach this borrower by
    phone, and what happened -- reuses log_event (the same generic
    audit-log primitive every other real activity in this project already
    goes through, e.g. clarification requests, quick actions) rather than
    a new table. Found live: the phone number shown on every account's
    detail panel was plain, inert text -- nothing on the account ever
    recorded that a call was attempted, so a second operator opening the
    same profile the same day had no way to know a colleague had already
    called, risking duplicate or conflicting outreach on exactly the
    flagged accounts where coordination matters most."""
    if outcome not in _CALL_LOG_OUTCOMES:
        raise ValueError(f"outcome must be one of {_CALL_LOG_OUTCOMES}, got {outcome!r}")
    log_event(account_id, "manual_call_logged", {"outcome": outcome, "note": note})


def get_call_log(account_id: str, ex=None) -> list[dict]:
    """Every logged call attempt for this account, most recent first --
    the read half of log_call, same pattern as get_clarification_requests."""
    rows = (ex or get_connection()).execute(
        "select details, created_at from events where account_id = %s and event_type = %s order by created_at desc",
        (account_id, "manual_call_logged"),
    ).fetchall()
    return [
        {"outcome": r["details"]["outcome"], "note": r["details"].get("note"), "created_at": r["created_at"]}
        for r in rows
    ]


def update_account_fields(account_id: str, **fields) -> None:
    """Partial update -- only the given fields are touched; anything
    omitted keeps its current value. Callers control the keyword names
    (this is never handed raw user input as field names, only from
    ops/api.py's own fixed, Pydantic-validated AccountUpdateIn), so the
    generated SET clause is never attacker-controlled -- values are still
    always bound as real query parameters, never string-interpolated.

    Found live: there was no way to fix a typo'd phone number or
    reclassify risk tier once an account existed -- create_account but no
    update counterpart, no PATCH route anywhere in ops/api.py."""
    if not fields:
        return
    if get_account(account_id) is None:
        raise ValueError(f"No account found for account_id={account_id!r}")
    set_clause = ", ".join(f"{key} = %({key})s" for key in fields)
    get_connection().execute(
        f"update accounts set {set_clause}, updated_at = now() where account_id = %(account_id)s",
        {**fields, "account_id": account_id},
    )


_TRANSCRIPT_EVENT_TYPES = ("user_message", "assistant_message", "tool_called")
_MAX_TRANSCRIPT_EVENTS = 200  # an explicit bound on an unbounded query, not a UI nicety


def get_conversation_transcript(account_id: str) -> list[dict]:
    """Every real user/assistant turn and tool call this account's AI
    conversations have ever produced, oldest first -- the full picture
    behind memory/conversation_memory.py's own short recap, which reads
    this exact same events data but only the last few, for seeding a new
    conversation rather than for a human to read.

    Found live: ops previously had zero visibility into what the AI agent
    actually SAID to a borrower -- staff could see payment history and
    flags but not the conversation that produced them, only messages ops
    itself sent via clarification requests."""
    rows = get_connection().execute(
        "select event_type, details, created_at from events "
        "where account_id = %s and event_type = any(%s) "
        "order by created_at desc limit %s",
        (account_id, list(_TRANSCRIPT_EVENT_TYPES), _MAX_TRANSCRIPT_EVENTS),
    ).fetchall()
    return [
        {"event_type": r["event_type"], "details": r["details"], "created_at": r["created_at"]}
        for r in reversed(rows)  # oldest first -- matches how it actually happened
    ]


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
    account row (if it has any -- see below), then marks the escalation
    resolved. Re-checks resolved_at rather than blindly re-applying, so a
    double-click or a retried request can't double-apply the same change
    -- raises instead of silently no-op'ing a second time, since silently
    returning "success" on a stale retry could make an operator think a
    second, different approval landed when nothing happened.

    Most escalations are NOT a structured restructuring proposal --
    escalate_to_human (an open dispute, repeated broken promises, or the
    agent just being unsure) creates one with proposed_changes=None; only
    propose_restructuring computes real terms. Approving one of those
    used to raise ValueError here -- an unhandled 500 in the ops
    dashboard the instant anyone clicked Approve on the vast majority of
    real escalations, which are exactly this kind. There's nothing to
    mechanically apply for those, so approving one now just closes it out
    (status='approved', account untouched) -- the same "I've handled this,
    close it" an operator already means by clicking Approve on a plain
    hand-off, typically after sending a clarification request or handling
    it some other real way outside this one mechanical apply-terms step.

    Known limitation, not solved here: for a real extend_tenure proposal,
    this applies proposed_changes exactly as they were computed at
    proposal time. If the account's real state changed in between (a
    payment posted, a dispute opened), this does not re-validate against
    the CURRENT state -- a real deployment handling money for real would
    want that; this demo doesn't."""
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
    account_id = row["account_id"]
    if changes:
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
    return {"escalation_id": escalation_id, "account_id": account_id, **(changes or {})}


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


def get_account_by_telegram_chat_id(chat_id: int) -> Account | None:
    """The reverse of set_telegram_chat_id -- lets telegram_bot.py
    recognize "this chat already proved it owns an account" WITHOUT the
    in-memory session that fact used to live in exclusively. Needed
    because that in-memory session (_sessions, keyed by chat_id) is wiped
    on every process restart (a deploy, or -- found live -- an OOM kill),
    while this column survives one. Not a new trust boundary: this
    mapping is only ever written once a real access-key check already
    passed (see set_telegram_chat_id's own docstring), and the ops
    dashboard already sends real restructuring-decision notifications to
    whatever chat_id it currently holds with no further re-verification --
    this just extends that same, already-accepted trust to resuming a
    text conversation instead of only outbound pushes."""
    row = get_connection().execute("select * from accounts where telegram_chat_id = %s", (chat_id,)).fetchone()
    return _row_to_account(row) if row else None


class ExtraPaymentDecisionRequiredError(ValueError):
    """Raised when amount doesn't cover what's actually due this cycle and
    the caller didn't say whether to credit it toward the next cycle. See
    record_payment's docstring for the full decision table; the caller
    (payment_confirm_endpoint) is expected to have already asked the
    borrower before ever reaching this point, using the emi_amount_due
    figure /pay/{token}/info returns for exactly that purpose."""


def record_payment(
    account_id: str,
    amount: float,
    payment_date: date | None = None,
    apply_extra_to_next: bool | None = None,
    payment_scheme: str | None = None,
) -> dict:
    """The one place a payment actually LANDS: records regular EMI payments
    as well as off-cycle prepayments.

    What's due THIS cycle is emi_amount minus pending_emi_credit.

    - amount covers it: the cycle retires -- one month off, due date rolls
      forward. Anything above what was due is then either
        * exactly the late fee (accounts/dues.py) -- the borrower paid
          "EMI + fee", the figure the dashboard and reminders quote. That is
          NOT a prepayment: it is recorded as a regular payment with nothing
          credited (late_fee_paid says so), or
        * a genuine extra, applied per payment_scheme:
          1. credit_next_emi (default): pending_emi_credit, so next cycle's
             EMI is smaller.
          2. reduce_emi: against the remaining balance, lowering the EMI for
             the months that are left.
          3. reduce_tenure: against the remaining balance, removing as many
             whole EMIs from the end as the extra pays for; a remainder
             smaller than one EMI stays as credit rather than disappearing.
      Both schemes measure "remaining balance" the way the rest of the
      product does (emi_amount x months left, see get_payment_status and
      calculate_hypothetical), *after* this cycle's EMI is retired -- so the
      EMI just paid is never also counted as prepayment, and never lost.
    - amount falls short: a partial payment toward THIS cycle. The cycle is
      not retired (months_remaining/emi_due_date untouched) and
      apply_extra_to_next must be True (credit it -- what is still due this
      cycle is emi_amount - credit) or False (record only, no effect on the
      schedule); None raises ExtraPaymentDecisionRequiredError.

    Atomic, and serialised per account: the payment row and the account
    update commit together, and the account row is locked (FOR UPDATE) from
    the read to the commit. Before, they were separate autocommit statements
    after an unlocked read, so two payments on one account at the same moment
    (two payment links, or a link and an ops entry) both started from the same
    balance and one update was lost.
    """
    with get_connection().transaction() as tx:
        return _record_payment_in(tx, account_id, amount, payment_date, apply_extra_to_next, payment_scheme)


def _lock_account(conn, account_id: str) -> Account:
    """The account, read inside the caller's open transaction and locked
    until it commits, so payments on one account apply one after another.

    Without payment history or promises: payment arithmetic never uses them,
    and loading them would check a SECOND connection out of the pool while this
    one is held. With the pool capped at 5, a burst of simultaneous confirms
    each holding one connection could then wait on each other for good."""
    row = conn.execute("select * from accounts where account_id = %s for update", (account_id,)).fetchone()
    if row is None:
        raise ValueError(f"No account found for account_id={account_id!r}")
    return _row_to_account(row, with_history=False)


def _record_payment_in(
    conn,
    account_id: str,
    amount: float,
    payment_date: date | None,
    apply_extra_to_next: bool | None,
    payment_scheme: str | None,
) -> dict:
    """record_payment's body, run on the caller's open transaction (see
    record_payment for the rules). redeem_payment_token uses it directly so
    the token and the payment commit, or roll back, together."""
    if payment_date is None:
        payment_date = current_date()
    account = _lock_account(conn, account_id)

    due_this_cycle = round(account.emi_amount - account.pending_emi_credit, 2)
    on_time = payment_date <= account.emi_due_date
    new_principal = account.principal_amount
    new_emi_amount = account.emi_amount
    late_fee_paid = 0.0

    if amount + 0.01 >= due_this_cycle:
        excess = round(amount - due_this_cycle, 2)
        months_after = max(0, account.months_remaining - 1)
        new_months_remaining = months_after
        next_due_date = add_one_month(account.emi_due_date)

        fee = late_fee_due(account, payment_date)
        if fee and excess > 0.01 and abs(excess - fee) <= 0.01:
            late_fee_paid = fee
            excess = 0.0

        outstanding_after = round(account.emi_amount * months_after, 2)
        kind = "regular"
        new_credit = 0.0
        if excess > 0.01:
            kind = "overpayment_applied"
            new_credit = excess
            if payment_scheme == "reduce_emi" and months_after > 0:
                applied = min(excess, outstanding_after)
                kind = "principal_prepayment_reduce_emi"
                new_principal = max(0.0, round(account.principal_amount - applied, 2))
                new_emi_amount = max(1.0, round((outstanding_after - applied) / months_after, 2))
                new_credit = round(excess - applied, 2)
            elif payment_scheme == "reduce_tenure" and months_after > 0:
                whole_emis = min(int(excess // account.emi_amount), months_after)
                if whole_emis > 0:
                    applied = round(whole_emis * account.emi_amount, 2)
                    kind = "principal_prepayment_reduce_tenure"
                    new_months_remaining = months_after - whole_emis
                    new_principal = max(0.0, round(account.principal_amount - applied, 2))
                    new_credit = round(excess - applied, 2)
    else:
        if apply_extra_to_next is None:
            raise ExtraPaymentDecisionRequiredError(
                f"amount {amount} is less than the {due_this_cycle} due this cycle for account_id={account_id!r} -- "
                "apply_extra_to_next must be True or False"
            )
        kind = "extra_applied" if apply_extra_to_next else "extra_unapplied"
        new_months_remaining = account.months_remaining
        next_due_date = account.emi_due_date
        new_credit = round(account.pending_emi_credit + amount, 2) if apply_extra_to_next else account.pending_emi_credit

    conn.execute(
        "insert into payment_history (account_id, payment_date, amount, on_time, kind) values (%s, %s, %s, %s, %s)",
        (account_id, payment_date, amount, on_time, kind),
    )
    conn.execute(
        "update accounts set principal_amount = %s, emi_amount = %s, months_remaining = %s, emi_due_date = %s, pending_emi_credit = %s, updated_at = now() where account_id = %s",
        (new_principal, new_emi_amount, new_months_remaining, next_due_date, new_credit, account_id),
    )
    return {
        "account_id": account_id,
        "amount": amount,
        "payment_date": payment_date.isoformat(),
        "on_time": on_time,
        "kind": kind,
        "principal_amount": new_principal,
        "emi_amount": new_emi_amount,
        "months_remaining": new_months_remaining,
        "next_emi_due_date": next_due_date.isoformat(),
        "pending_emi_credit": new_credit,
        "late_fee_paid": late_fee_paid,
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


def redeem_payment_token(
    token: str,
    apply_extra_to_next: bool | None = None,
    payment_scheme: str | None = None,
) -> dict:
    """Redeems a payment token and records the payment according to the selected scheme.

    One transaction, with the token row locked (SELECT ... FOR UPDATE) from
    the check to the commit. A second confirm of the same link -- a double
    tap, or a retry while the first is still running -- waits for the first
    to finish, then finds the link used and is refused. Before, the check and
    "mark used" were separate autocommit statements with no condition on
    used_at, so two confirms at the same moment could both pass the check and
    record the payment twice.

    Marking the link used and recording the payment commit together: if the
    payment cannot be recorded (a short payment with no decision, a database
    error), nothing is written and the link stays usable, rather than being
    burned with nothing paid."""
    with get_connection().transaction() as tx:
        row = tx.execute(
            "select account_id, amount, expires_at, used_at from payment_tokens where token = %s for update",
            (token,),
        ).fetchone()
        if row is None:
            raise PaymentTokenNotFoundError(f"no payment token found for token={token!r}")
        if row["used_at"] is not None:
            raise PaymentTokenAlreadyUsedError("this payment link has already been used")
        if row["expires_at"] < datetime.now(timezone.utc):
            raise PaymentTokenExpiredError("this payment link has expired")

        tx.execute("update payment_tokens set used_at = now() where token = %s", (token,))
        # A short payment with no decision raises ExtraPaymentDecisionRequiredError
        # in here, which rolls back the line above too: the link stays usable.
        return _record_payment_in(
            tx, row["account_id"], float(row["amount"]), None, apply_extra_to_next, payment_scheme,
        )


_ESCALATION_COLUMNS = "escalation_id, account_id, reason, status, created_at, resolved_at, proposed_changes, resolution_reason"


def get_escalations_for_account(account_id: str, ex=None) -> list[Escalation]:
    rows = (ex or get_connection()).execute(
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


def list_open_escalations(ex=None) -> list[Escalation]:
    """The ops queue: every escalation still waiting on a human, oldest
    first -- the order a human should actually work through them in."""
    rows = (ex or get_connection()).execute(
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


def get_clarification_requests(account_id: str, ex=None) -> list[dict]:
    """Every clarification-request message ever sent to this account,
    most recent first -- the ops dashboard's communication-history
    thread for that borrower (see outbound/send.py's
    notify_clarification_request, which is what actually writes these
    events). resolved is True for anything sent before the account's most
    recent mark_clarifications_resolved() checkpoint (see that function's
    own docstring for why this is a watermark, not per-message state) --
    False, including when nothing has ever been marked resolved at all."""
    conn = ex or get_connection()
    rows = conn.execute(
        "select details, created_at from events where account_id = %s and event_type = %s order by created_at desc",
        (account_id, "clarification_request_sent"),
    ).fetchall()
    watermark_row = conn.execute(
        "select created_at from events where account_id = %s and event_type = %s order by created_at desc limit 1",
        (account_id, "clarification_thread_resolved"),
    ).fetchone()
    watermark = watermark_row["created_at"] if watermark_row else None
    return [
        {
            "message": r["details"]["message"],
            "delivered_via_telegram": r["details"]["delivered_via_telegram"],
            "created_at": r["created_at"],
            "resolved": watermark is not None and r["created_at"] <= watermark,
        }
        for r in rows
    ]


# Everything the system or staff has SENT to a borrower, by the event type
# outbound/send.py's _deliver_and_log logs it under. The borrower's dashboard
# reads this (not just clarification requests) so what it says -- that nothing
# sent to them is only visible on Telegram -- is actually true: a web-only
# borrower previously never saw a restructuring decision or a reminder at all.
_BORROWER_MESSAGE_KINDS = {
    "clarification_request_sent": "message",
    "restructuring_decision_notified": "decision",
    "dispute_resolution_notified": "dispute",
    "reminder_sent": "reminder",
}
_MAX_BORROWER_MESSAGES = 50  # an explicit bound on an otherwise unbounded history


def get_borrower_messages(account_id: str, ex=None) -> list[dict]:
    """Messages sent to this borrower that don't belong to a case, newest
    first, bounded. Each is {message, delivered_via_telegram, created_at,
    kind} with kind one of message | decision | dispute | reminder.

    A message about a case (it carries a case_ref, see accounts/cases.py)
    is shown in that case's own thread instead -- listing it here too was
    how the borrower ended up reading our reply in one panel and their
    request in another."""
    rows = (ex or get_connection()).execute(
        "select event_type, details, created_at from events where account_id = %s and event_type = any(%s) "
        "and details->>'case_ref' is null order by created_at desc limit %s",
        (account_id, list(_BORROWER_MESSAGE_KINDS), _MAX_BORROWER_MESSAGES),
    ).fetchall()
    return [
        {
            "message": r["details"].get("message", ""),
            "delivered_via_telegram": bool(r["details"].get("delivered_via_telegram", False)),
            "created_at": r["created_at"],
            "kind": _BORROWER_MESSAGE_KINDS[r["event_type"]],
        }
        for r in rows
        if r["details"].get("message")
    ]


_MAX_THREAD_EVENTS = 500  # an explicit bound: a few hundred messages is far beyond any real case history here


def add_case_message(account_id: str, case_ref: str, author: str, message: str, delivered_via_telegram: bool = False) -> None:
    """Appends one message to a case's thread. The caller has already checked
    the case belongs to this account (case_belongs_to_account); a message ops
    sends over Telegram goes through outbound/send.py's notify_case_message,
    which logs it with the same shape."""
    if author not in cases.AUTHORS:
        raise ValueError(f"author must be one of {cases.AUTHORS}, got {author!r}")
    log_event(account_id, cases.CASE_MESSAGE_EVENT, {
        "case_ref": case_ref, "author": author, "message": message, "delivered_via_telegram": delivered_via_telegram,
    })


def get_case_threads(account_id: str, ex=None) -> dict[str, list[dict]]:
    """{case_ref: [message, ...]} for every case on this account, each thread
    oldest first (see accounts/cases.py). Bounded to the newest
    _MAX_THREAD_EVENTS messages."""
    rows = (ex or get_connection()).execute(
        "select event_type, details, created_at from events where account_id = %s and event_type = any(%s) "
        "and details->>'case_ref' is not null order by created_at desc limit %s",
        (account_id, list(cases.THREAD_EVENT_TYPES), _MAX_THREAD_EVENTS),
    ).fetchall()
    return cases.group_threads([dict(r) for r in rows])


def get_threads_for_cases(case_refs: list[str], ex=None) -> dict[str, list[dict]]:
    """get_case_threads for given cases across accounts, in one query -- the
    ops inbox, which shows each open case's latest messages."""
    if not case_refs:
        return {}
    rows = (ex or get_connection()).execute(
        "select event_type, details, created_at from events where event_type = any(%s) "
        "and details->>'case_ref' = any(%s) order by created_at desc limit %s",
        (list(cases.THREAD_EVENT_TYPES), list(case_refs), _MAX_THREAD_EVENTS),
    ).fetchall()
    return cases.group_threads([dict(r) for r in rows])


def borrower_names(account_ids: list[str], ex=None) -> dict[str, str]:
    """{account_id: borrower_name} for the given accounts, in one query."""
    if not account_ids:
        return {}
    rows = (ex or get_connection()).execute(
        "select account_id, borrower_name from accounts where account_id = any(%s)", (list(account_ids),)
    ).fetchall()
    return {r["account_id"]: r["borrower_name"] for r in rows}


def changes_since_last_status_check(account_id: str) -> dict:
    """What changed on this account after the agent last looked it up (its latest get_payment_status call): payments
    recorded (through a payment link or by ops) and restructurings approved. {"payments": n, "restructurings": n,
    "since": <that call's time>}; both 0 when nothing changed, or when it was never looked up.

    Found live: a borrower paid through the link the agent had just sent, then asked "do I have any dues right now?"
    -- and was told "21 days past due, ₹35,500 due today", from the status the agent had looked up before the payment."""
    row = get_connection().execute(
        """
        with last_check as (
            select max(created_at) as at from events
            where account_id = %(a)s and event_type = 'tool_called' and details->>'tool' = 'get_payment_status'
        )
        select (select at from last_check) as since,
               (select count(*) from payment_history, last_check
                 where account_id = %(a)s and created_at > last_check.at) as payments,
               (select count(*) from escalations, last_check
                 where account_id = %(a)s and status = 'approved' and proposed_changes is not null
                   and resolved_at > last_check.at) as restructurings
        """,
        {"a": account_id},
    ).fetchone()
    return {"since": row["since"], "payments": row["payments"], "restructurings": row["restructurings"]}


def get_case(account_id: str, case_ref: str, ex=None) -> dict | None:
    """The case case_ref names, when it is a case on THIS account, else None
    -- the check every endpoint taking a case_ref from a request makes before
    acting, so a reference to another account's case is simply not found.

    {case_ref, source ("dispute" | "escalation"), kind (a dispute kind, or an
    escalation kind from escalation_kinds.classify_escalation), open (bool),
    reason}."""
    parsed = cases.parse_case_ref(case_ref)
    if parsed is None:
        return None
    source, key = parsed
    conn = ex or get_connection()
    if source == "dispute":
        row = conn.execute(
            "select kind, status, reason from disputes where id = %s and account_id = %s", (key, account_id)
        ).fetchone()
        if row is None:
            return None
        return {"case_ref": case_ref, "source": source, "kind": row["kind"], "open": row["status"] == "open", "reason": row["reason"]}
    row = conn.execute(
        "select reason, proposed_changes, resolved_at from escalations where escalation_id = %s and account_id = %s",
        (key, account_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "case_ref": case_ref, "source": source, "kind": classify_escalation(row["reason"], row["proposed_changes"]),
        "open": row["resolved_at"] is None, "reason": row["reason"],
    }


def mark_clarifications_resolved(account_id: str) -> None:
    """Marks every clarification request sent to this account SO FAR as
    resolved -- a checkpoint, not per-message state: clarification_request_
    sent events (see notify_clarification_request) have no id exposed
    anywhere in this app to attach per-row status to, and in practice a
    whole thread of flags tends to get addressed together on one call, not
    message by message. Logging a fresh event here (rather than mutating
    past ones) matches this project's existing append-only convention for
    the events table -- a corrected/resolved state is a new fact, not an
    edit to history."""
    log_event(account_id, "clarification_thread_resolved", {})


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
