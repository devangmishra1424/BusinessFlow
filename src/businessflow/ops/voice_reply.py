"""What the ops dashboard says and shows for one understood request (ops/voice.py's Plan): an answer written by
this code from the dashboard's own data, a confirmation card for an action, or a question back ("which Khan?",
"how much did they pay?") when something needed is missing.

Every figure in an answer comes from the same data the dashboard's cards are drawn from, fetched through
DashboardData (ops/api.py implements it with its own endpoints' functions), so what is said always matches the
screen. Nothing here writes: a confirmation card carries the exact request (method, path, body) the dashboard's own
form for that action sends, and only the operator's Confirm click sends it -- through the endpoint that already
validates it, guards against duplicates and tells the borrower.
"""

import difflib
import re
from datetime import date, datetime, timezone
from typing import Literal, Protocol

from pydantic import BaseModel

from businessflow.accounts import cases
from businessflow.ops import spoken_numbers
from businessflow.ops.voice import Plan

_MAX_GROUP_MESSAGE = 50  # ops/api.py's _MAX_BULK_CLARIFICATION_ACCOUNTS: the bulk endpoint refuses more
_SOON_ENOUGH_DAYS = 3  # the inbox's own "stale" line: open more than 72 h

GROUP_LABELS = {
    "all": "accounts", "overdue": "overdue accounts", "disputed": "accounts with an open dispute",
    "broken_promises": "accounts with broken promises", "not_on_telegram": "accounts not linked to Telegram",
}
CASE_KIND_LABELS = {  # mirrors ops/static/app.js's KIND_LABELS, in the plural where counted
    "payment_claim": ("payment claim", "payment claims"), "dispute": ("dispute", "disputes"),
    "fraud": ("fraud claim", "fraud claims"), "request": ("request", "requests"),
    "restructuring": ("request for more time", "requests for more time"), "callback": ("call-back", "call-backs"),
    "closure": ("closure request", "closure requests"), "system": ("system notice", "system notices"),
}
_TOPIC_TABS = {
    "summary": "overview", "dues": "overview", "loan": "overview", "contact": "overview", "payments": "payments",
    "promises": "cases", "calls": "cases", "cases": "cases", "conversation": "cases", "documents": "documents",
}
_SORTS = {"most_overdue": "urgency", "highest_risk": "risk", "account_id": "id"}
_WINDOW_HOURS = {"24h": 24, "3d": 72, "7d": 168, "30d": 720}
_WINDOW_WORDS = {"24h": "24 hours", "3d": "3 days", "7d": "7 days", "30d": "30 days"}
_LANGUAGES = {"hi": "Hindi", "en": "English", "hinglish": "Hinglish"}
_CALL_OUTCOMES = {"reached": "reached them", "no_answer": "no answer", "voicemail": "left a voicemail", "wrong_number": "wrong number"}
_HONORIFICS = {"mr", "mrs", "ms", "miss", "shri", "smt", "sri", "ji", "sir", "madam", "sahab", "saheb", "account", "of"}
_ACCOUNT_ID = re.compile(r"\bb\.?\s*f\.?\s*[-\s]?\s*(\d{4,})\b", re.IGNORECASE)
_CURRENT = {"", "current", "this", "this one", "this account", "her", "him", "them", "they", "he", "she", "open", "it"}


class DashboardData(Protocol):
    """The dashboard's own data, as its endpoints serve it (plain dicts of their response models)."""

    def accounts(self) -> list[dict]: ...  # GET /accounts
    def account(self, account_id: str) -> dict | None: ...  # GET /accounts/{id}; None when there is no such account
    def inbox(self) -> list[dict]: ...  # GET /inbox
    def metrics(self, since_hours: float) -> dict: ...  # GET /metrics
    def documents(self, account_id: str) -> list[dict]: ...
    def conversation(self, account_id: str) -> list[dict]: ...
    # What the borrower is sent when the endpoint behind a card runs -- the endpoint's own wording, so a card can't
    # promise one message while another is sent.
    def decision_message(self, case: dict, account: dict, approved: bool, reason: str | None) -> str | None: ...
    def claim_confirmed_message(self, amount: float, paid_on: date, note: str | None) -> str: ...
    def case_closed_message(self, kind: str, note: str | None) -> str: ...


class ScreenOut(BaseModel):
    """Where the page goes. Every field is optional: only what is set changes."""

    view: Literal["overview", "inbox"] | None = None
    filter: str | None = None  # all | overdue | disputed | broken_promises | not_on_telegram
    sort: Literal["urgency", "risk", "id"] | None = None
    search: str | None = None
    since_hours: int | None = None
    account_id: str | None = None
    tab: Literal["overview", "cases", "payments", "documents"] | None = None
    open_form: Literal["new_account"] | None = None


class DetailOut(BaseModel):
    label: str
    value: str


class ProposalOut(BaseModel):
    action: str
    title: str
    details: list[DetailOut]
    borrower_told: str | None = None  # what the borrower is sent, word for word (with a note, before it)
    warnings: list[str] = []
    method: Literal["POST", "PATCH"]
    path: str
    body: dict | None = None
    editable: str | None = None  # a body field (a message or note) the operator may reword before confirming
    editable_label: str | None = None
    editable_required: bool = False  # Confirm stays off while the editable field is blank
    credit_choice: bool = False  # the amount is short of this cycle's EMI: Confirm needs "credit it toward the next EMI?"


class ChoiceOut(BaseModel):
    """One answer to a "did you mean" question: clicking it runs this plan as-is, without the model."""

    label: str
    tool: str
    arguments: dict


class VoiceReplyOut(BaseModel):
    heard: str | None = None
    kind: Literal["answer", "proposal", "clarify"]
    say: str
    screen: ScreenOut | None = None
    proposal: ProposalOut | None = None
    choices: list[ChoiceOut] = []


# --- wording --------------------------------------------------------------------------------------------------------

def inr(amount: float, whole: bool = False) -> str:
    """₹ with Indian digit grouping: ₹12,500 / ₹1,20,000 / ₹13,250.50. whole=True rounds to the rupee, the way the
    dashboard's cards show a total (fmtInr in ops/static/app.js)."""
    paise_total = round(amount) * 100 if whole else round(amount * 100)
    whole, paise = divmod(abs(paise_total), 100)
    digits = str(whole)
    if len(digits) > 3:
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", digits[:-3])
        digits = f"{head},{digits[-3:]}"
    sign = "-" if paise_total < 0 else ""
    return f"{sign}₹{digits}" + (f".{paise:02d}" if paise else "")


def _n(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _day(d: date, today: date) -> str:
    delta = (d - today).days
    if delta == 0:
        return "today"
    if delta == -1:
        return "yesterday"
    if delta == 1:
        return "tomorrow"
    return f"{d:%a} {d.day} {d:%b}" + ("" if d.year == today.year else f" {d.year}")


def _age(opened_at: datetime, now: datetime) -> str:
    hours = max(0.0, (now - opened_at).total_seconds() / 3600)
    if hours < 1:
        return "under an hour"
    if hours < 48:
        return _n(int(hours), "hour")
    return _n(int(hours // 24), "day")


def _who(account: dict) -> str:
    return f"{account['borrower_name']} ({account['account_id']})"


def _flags(account: dict) -> set[str]:
    return {f["label"] for f in account.get("flags", [])}


def _answer(say: str, screen: ScreenOut | None = None) -> VoiceReplyOut:
    return VoiceReplyOut(kind="answer", say=say, screen=screen)


def _clarify(say: str, choices: list[ChoiceOut] | None = None, screen: ScreenOut | None = None) -> VoiceReplyOut:
    return VoiceReplyOut(kind="clarify", say=say, choices=choices or [], screen=screen)


# --- which account --------------------------------------------------------------------------------------------------

def _name_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in _HONORIFICS]


def _name_score(ref: str, account: dict) -> float:
    """How well what was said matches the borrower's or business's name, 0..1. Token by token, so "Fatima" matches
    "Fatima Khan" fully and a misheard "Fatma" still scores high."""
    said = _name_tokens(ref)
    if not said:
        return 0.0
    best = 0.0
    for name in (account["borrower_name"], account["business_name"]):
        tokens = _name_tokens(name)
        if not tokens:
            continue
        per_token = [max(difflib.SequenceMatcher(None, s, t).ratio() for t in tokens) for s in said]
        whole = difflib.SequenceMatcher(None, " ".join(said), " ".join(tokens)).ratio()
        best = max(best, sum(per_token) / len(per_token), whole)
    return best


class _Unresolved(Exception):
    def __init__(self, reply: VoiceReplyOut):
        self.reply = reply


def resolve_account(ref: str | None, plan: Plan, data: DashboardData, screen_account_id: str | None) -> dict:
    """The account a request is about, as GET /accounts/{id} returns it. Raises _Unresolved with the question to ask
    when it is ambiguous, missing or not found -- never guesses between two close names."""
    text = (ref or "").strip()
    if text.lower() in _CURRENT:
        if screen_account_id and (account := data.account(screen_account_id)) is not None:
            return account
        raise _Unresolved(_clarify("Which account? Say the borrower's name or account ID."))
    if match := _ACCOUNT_ID.search(text):
        account_id = f"BF-{match.group(1)}"
        if (account := data.account(account_id)) is not None:
            return account
        raise _Unresolved(_clarify(f"I couldn't find account {account_id}."))
    scored = sorted(((_name_score(text, a), a) for a in data.accounts()), key=lambda pair: -pair[0])
    if scored and scored[0][0] >= 0.85 and (len(scored) == 1 or scored[1][0] < scored[0][0] - 0.1):
        return data.account(scored[0][1]["account_id"]) or _not_found(text)
    close = [a for score, a in scored if score >= 0.6][:3]
    if not close:
        _not_found(text)
    raise _Unresolved(_clarify(
        "Which account did you mean?",
        [ChoiceOut(label=_who(a), tool=plan.tool, arguments={**plan.args.model_dump(mode="json"), "account": a["account_id"]})
         for a in close],
    ))


def _not_found(text: str):
    raise _Unresolved(_clarify(f"I couldn't find an account for “{text}”. Try the account ID, like BF-1003."))


# --- questions ------------------------------------------------------------------------------------------------------

def _overdue_rupees(accounts: list[dict]) -> float:
    # The Overdue card's own figure (ops/static/app.js renderKpis): the EMIs of the overdue accounts.
    return sum(a["emi_amount"] for a in accounts if "overdue" in _flags(a))


def _sorted(accounts: list[dict], sort: str) -> list[dict]:
    """ops/static/app.js's _accountSortComparator, so the first name said is the first card shown."""
    if sort == "highest_risk":
        rank = {"high": 0, "medium": 1, "low": 2}
        return sorted(accounts, key=lambda a: (rank.get(a["risk_tier"], 3), -a["days_past_due"]))
    if sort == "account_id":
        return sorted(accounts, key=lambda a: a["account_id"])
    return sorted(accounts, key=lambda a: (
        "overdue" not in _flags(a), -a["days_past_due"] if "overdue" in _flags(a) else 0, a["account_id"],
    ))


def _risk_mix(accounts: list[dict]) -> str:
    counts = {tier: sum(a["risk_tier"] == tier for a in accounts) for tier in ("high", "medium", "low")}
    return f"{counts['high']} high, {counts['medium']} medium and {counts['low']} low risk"


def _portfolio_overview(data: DashboardData) -> VoiceReplyOut:
    accounts, inbox = data.accounts(), data.inbox()
    screen = ScreenOut(view="overview", filter="all")
    if not accounts:
        return _answer("No accounts on the books yet.", screen)
    flagged = sum(bool(a["flags"]) for a in accounts)
    overdue = [a for a in accounts if "overdue" in _flags(a)]
    parts = [f"{_n(len(accounts), 'account')} on the books, {flagged} needing attention."]
    parts.append(
        f"{len(overdue)} overdue, with {inr(_overdue_rupees(accounts), whole=True)} in EMIs at risk." if overdue else "Nothing is overdue."
    )
    parts.append(f"{_n(len(inbox), 'open case')} waiting on a person." if inbox else "The inbox is clear.")
    parts.append(f"Risk mix: {_risk_mix(accounts)}.")
    return _answer(" ".join(parts), screen)


def _show_accounts(args, data: DashboardData) -> VoiceReplyOut:
    accounts = data.accounts()
    group = args.group
    chosen = [a for a in accounts if group == "all" or (
        not a["telegram_linked"] if group == "not_on_telegram" else group in _flags(a)
    )]
    if args.search:
        q = args.search.lower()
        chosen = [a for a in chosen if q in a["borrower_name"].lower() or q in a["business_name"].lower() or q in a["account_id"].lower()]
    chosen = _sorted(chosen, args.sort)
    screen = ScreenOut(view="overview", filter=group, sort=_SORTS[args.sort], search=args.search or "")
    label = GROUP_LABELS[group]
    matching = f" matching “{args.search}”" if args.search else ""
    if not chosen:
        return _answer("Nothing is overdue." if group == "overdue" and not args.search else f"No {label}{matching}.", screen)
    parts = [f"{len(chosen)} {label if len(chosen) != 1 else label.replace('accounts', 'account', 1)}{matching}."]
    if group == "overdue":
        parts[0] = parts[0][:-1] + f", with {inr(_overdue_rupees(chosen), whole=True)} in EMIs at risk."
    top = chosen[:3]
    if args.sort == "most_overdue" and any("overdue" in _flags(a) for a in top):
        named = [f"{a['borrower_name']}, {_n(a['days_past_due'], 'day')}" for a in top if "overdue" in _flags(a)]
        parts.append(("Most overdue: " if len(named) > 1 else "Most overdue is ") + "; ".join(named) + ".")
    elif args.sort == "highest_risk":
        parts.append(f"Risk mix: {_risk_mix(chosen)}. Highest: {_list([a['borrower_name'] + ' (' + a['risk_tier'] + ')' for a in top])}.")
    else:
        parts.append(f"{'First' if len(chosen) > 3 else 'They are'}: {_list([a['borrower_name'] for a in top])}.")
    return _answer(" ".join(parts), screen)


def _show_inbox(args, data: DashboardData, now: datetime) -> VoiceReplyOut:
    inbox = [c for c in data.inbox() if args.kind == "any" or c["kind"] == args.kind]
    screen = ScreenOut(view="inbox")
    if args.kind != "any":
        singular, plural = CASE_KIND_LABELS[args.kind]
        if not inbox:
            return _answer(f"No open {plural}.", screen)
        named = [f"{c.get('borrower_name') or c['account_id']}, open {_age(c['opened_at'], now)}" for c in inbox[:3]]
        return _answer(f"{_n(len(inbox), 'open ' + singular, 'open ' + plural)}: {'; '.join(named)}.", screen)
    if not inbox:
        return _answer("The inbox is clear: nothing is waiting on a person.", screen)
    by_kind: dict[str, int] = {}
    for c in inbox:
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    counted = [_n(n, *CASE_KIND_LABELS.get(kind, (kind, kind + "s"))) for kind, n in by_kind.items()]
    oldest = min(inbox, key=lambda c: c["opened_at"])
    stale = sum((now - c["opened_at"]).total_seconds() > _SOON_ENOUGH_DAYS * 86400 for c in inbox)
    parts = [f"{_n(len(inbox), 'open case')}: {_list(counted)}."]
    parts.append(f"Most urgent: {inbox[0].get('borrower_name') or inbox[0]['account_id']}, {inbox[0]['title'].lower()}.")
    if oldest is not inbox[0]:
        parts.append(f"Oldest: {oldest.get('borrower_name') or oldest['account_id']}, open {_age(oldest['opened_at'], now)}.")
    if stale:
        parts.append(f"{_n(stale, 'case has', 'cases have')} waited more than 3 days.")
    return _answer(" ".join(parts), screen)


def _escalation_rate(args, data: DashboardData) -> VoiceReplyOut:
    hours = _WINDOW_HOURS[args.window]
    metrics = data.metrics(hours)
    calls = metrics["event_counts"].get("tool_called", 0)
    screen = ScreenOut(view="overview", since_hours=hours)
    if not calls:
        return _answer(f"No AI assistant activity in the last {_WINDOW_WORDS[args.window]}.", screen)
    rate = round(metrics["escalation_rate"] * 100)
    return _answer(
        f"In the last {_WINDOW_WORDS[args.window]}, {rate}% of the AI assistant's {_n(calls, 'tool call')} "
        "were handed to a person.", screen,
    )


def _account_details(args, account: dict, data: DashboardData, today: date) -> VoiceReplyOut:
    a = account
    screen = ScreenOut(account_id=a["account_id"], tab=_TOPIC_TABS[args.topic])
    head = f"{a['borrower_name']}, {a['account_id']}:"
    open_cases = [c for c in a.get("cases", []) if c["open"]]
    topic = args.topic
    if topic == "summary":
        status = (f"{_n(a['days_past_due'], 'day')} past due" if a["days_past_due"] > 0
                  else f"up to date, next EMI {_day(date.fromisoformat(str(a['emi_due_date'])), today)}")
        parts = [status, f"{inr(a['amount_due_now'])} due now" if a["amount_due_now"] > 0 else "nothing due now"]
        if open_cases:
            parts.append(_n(len(open_cases), "open case"))
        worries = [f["label"].replace("_", " ") for f in a["flags"] if f["label"] != "overdue"]
        if worries:
            parts.append("flagged " + _list(worries))
        return _answer(f"{head} {_list(parts)}.", screen)
    if topic == "dues":
        this_cycle = round(a["emi_amount"] - a["pending_emi_credit"], 2)
        fee = round(a["amount_due_now"] - this_cycle, 2) if a["amount_due_now"] > this_cycle else 0
        due = f"{inr(a['amount_due_now'])} due now" + (f", including a {inr(fee)} late fee" if fee > 0 else "")
        if a["amount_due_now"] <= 0:
            due = "nothing due now"
        when = date.fromisoformat(str(a["emi_due_date"]))
        late = f", {_n(a['days_past_due'], 'day')} past due" if a["days_past_due"] > 0 else ""
        return _answer(f"{head} {due}. EMI {inr(a['emi_amount'])}, due {_day(when, today)}{late}.", screen)
    if topic == "payments":
        history = a.get("payment_history", [])
        if not history:
            return _answer(f"{head} no payments on record yet.", screen)
        last = history[-1]
        on_time = sum(p["on_time"] for p in history)
        return _answer(
            f"{head} {_n(len(history), 'payment')} on record, {on_time} on time. Last: {inr(last['amount'])}, "
            f"{_day(date.fromisoformat(str(last['date'])), today)}.", screen,
        )
    if topic == "promises":
        promises = a.get("promises", [])
        if not promises:
            return _answer(f"{head} no promises to pay on record.", screen)
        kept = sum(p["kept"] is True for p in promises)
        broken = sum(p["kept"] is False for p in promises)
        pending = len(promises) - kept - broken
        latest = max(promises, key=lambda p: str(p["made_on"]))
        return _answer(
            f"{head} {_n(len(promises), 'promise')}: {kept} kept, {broken} broken, {pending} still open. Latest: "
            f"{inr(latest['promised_amount'])} by {_day(date.fromisoformat(str(latest['promised_date'])), today)}.", screen,
        )
    if topic == "cases":
        if not open_cases:
            return _answer(f"{head} no open cases.", screen)
        return _answer(f"{head} {_n(len(open_cases), 'open case')}: {_list([c['title'] for c in open_cases])}.", screen)
    if topic == "calls":
        calls = a.get("call_log", [])
        if not calls:
            return _answer(f"{head} no calls logged.", screen)
        last = calls[0]  # newest first (store.get_call_log)
        when = _day(last["created_at"].date() if isinstance(last["created_at"], datetime) else date.fromisoformat(str(last["created_at"])[:10]), today)
        return _answer(f"{head} {_n(len(calls), 'call')} logged. Last, {when}: {_CALL_OUTCOMES.get(last['outcome'], last['outcome'])}.", screen)
    if topic == "contact":
        telegram = "linked to Telegram" if a["telegram_linked"] else "not linked to Telegram"
        return _answer(f"{head} phone {a['phone_number']}, prefers {_LANGUAGES.get(a['language_preference'], a['language_preference'])}, {telegram}.", screen)
    if topic == "loan":
        nach = "NACH mandate active" if a["nach_mandate_active"] else "no active NACH mandate"
        return _answer(
            f"{head} {a['loan_type']}, {inr(a['principal_amount'])} principal, EMI {inr(a['emi_amount'])}, "
            f"{a['months_remaining']} of {a['tenure_months']} months left, {nach}.", screen,
        )
    if topic == "documents":
        documents = data.documents(a["account_id"])
        if not documents:
            return _answer(f"{head} no documents uploaded.", screen)
        return _answer(f"{head} {_n(len(documents), 'document')}: {_list([d['filename'] for d in documents[:3]])}.", screen)
    entries = data.conversation(a["account_id"])  # "conversation"
    said = [e for e in entries if e["event_type"] in ("user_message", "assistant_message")]
    if not said:
        return _answer(f"{head} no conversation with the AI assistant yet.", screen)
    last = said[-1]["created_at"]
    last_day = last.date() if isinstance(last, datetime) else date.fromisoformat(str(last)[:10])
    return _answer(f"{head} {_n(len(said), 'message')} with the AI assistant, the last {_day(last_day, today)}.", screen)


_HELP = (
    "Ask me about the portfolio, the inbox, the escalation rate or any borrower: what they owe, payments, promises, "
    "cases, calls or documents. I can also record a payment, confirm a payment claim, close a case, log a promise or "
    "a call, message a borrower or a group, approve or reject a request, hand an account to a person, send reminders "
    "or fix contact details -- and I always ask you to confirm before changing anything."
)


# --- actions --------------------------------------------------------------------------------------------------------

def _proposal(say: str, proposal: ProposalOut, screen: ScreenOut | None) -> VoiceReplyOut:
    prompt = " First choose whether to credit it toward the next EMI, then press Confirm." if proposal.credit_choice else " Press Confirm to go ahead."
    return VoiceReplyOut(kind="proposal", say=say + prompt, proposal=proposal, screen=screen)


def _amount(args, account: dict, heard: str, what: str) -> float:
    """The amount to act on: what's due now (from the database) when they said "the full amount", else a figure they
    actually said. Raises _Unresolved otherwise -- an amount nobody said is never put on a card."""
    if args.full_amount_due:
        if account["amount_due_now"] <= 0:
            raise _Unresolved(_clarify(f"{account['borrower_name']} has nothing due right now. Say the amount instead."))
        return round(account["amount_due_now"], 2)
    if args.amount is None:
        raise _Unresolved(_clarify(f"How much {what}? Say the amount, or “the full amount due”."))
    if not spoken_numbers.was_said(args.amount, heard):
        raise _Unresolved(_clarify(
            f"I didn't catch the amount clearly ({inr(args.amount)} isn't in what I heard). Please say it again."
        ))
    return round(args.amount, 2)


def _check_drafted(text: str | None, heard: str, what: str) -> None:
    if not text:
        return
    missing = spoken_numbers.unsaid_figures(text, heard)
    if missing:
        figures = ", ".join(f"{m:g}" for m in missing)
        raise _Unresolved(_clarify(
            f"The {what} I drafted has a figure you didn't say ({figures}), so I haven't used it. Please say it again "
            "with the exact figures."
        ))


def _open_cases(account: dict, *, source: str | None = None, kinds: set[str] | None = None) -> list[dict]:
    return [
        c for c in account.get("cases", [])
        if c["open"] and (source is None or c["source"] == source) and (kinds is None or c["kind"] in kinds)
    ]


def _pick_case(open_cases: list[dict], args, plan: Plan, account: dict, none_say: str) -> dict:
    if args.case:
        for c in open_cases:
            if c["case_ref"].lower() == args.case.lower():
                return c
    if not open_cases:
        raise _Unresolved(_clarify(none_say))
    if len(open_cases) == 1:
        return open_cases[0]
    raise _Unresolved(_clarify(
        f"{account['borrower_name']} has {len(open_cases)} open cases. Which one?",
        [ChoiceOut(label=f"{c['title']} ({c['case_ref']})", tool=plan.tool,
                   arguments={**plan.args.model_dump(mode="json"), "account": account["account_id"], "case": c["case_ref"]})
         for c in open_cases],
        ScreenOut(account_id=account["account_id"], tab="cases"),
    ))


def _payment_date(paid_on: date | None, today: date) -> date:
    paid_on = paid_on or today
    if paid_on > today:
        raise _Unresolved(_clarify(f"A payment can't be dated {_day(paid_on, today)} -- that's in the future. When was it paid?"))
    return paid_on


def _short_of_cycle(amount: float, account: dict) -> bool:
    # store.record_payment's own rule: short of emi_amount - pending_emi_credit needs the credit decision
    return amount < round(account["emi_amount"] - account["pending_emi_credit"], 2) - 0.005


def _recent_same_amount(account: dict, amount: float, today: date) -> list[str]:
    warnings = []
    for p in account.get("payment_history", []):
        paid = date.fromisoformat(str(p["date"]))
        if abs(p["amount"] - amount) < 0.005 and (today - paid).days <= 7:
            warnings.append(f"A payment of {inr(amount)} was already recorded {_day(paid, today)}. Make sure this is a second, separate payment.")
    return warnings


def _record_payment(args, account, heard, today) -> VoiceReplyOut:
    amount = _amount(args, account, heard, "did they pay")
    paid_on = _payment_date(args.paid_on, today)
    body = {"amount": amount, "payment_date": paid_on.isoformat()}
    short = _short_of_cycle(amount, account)
    if short and args.credit_toward_next is not None:
        body["apply_extra_to_next"] = args.credit_toward_next
    warnings = _recent_same_amount(account, amount, today)
    if _open_cases(account, kinds={cases.PAYMENT_CLAIM}):
        warnings.append(
            f"{account['borrower_name']} has an open payment claim. If this is that payment, cancel and say “confirm her "
            "payment claim” instead: that records it and closes the claim, and tells her."
        )
    return _proposal(
        f"Record a payment of {inr(amount)} from {_who(account)}, paid {_day(paid_on, today)}?",
        ProposalOut(
            action="record_payment", title="Record a payment",
            details=[DetailOut(label="Borrower", value=_who(account)), DetailOut(label="Amount", value=inr(amount)),
                     DetailOut(label="Paid", value=_day(paid_on, today)),
                     DetailOut(label="Due now (for reference)", value=inr(account["amount_due_now"]))],
            warnings=warnings,
            method="POST", path=f"/accounts/{account['account_id']}/payments", body=body,
            credit_choice=short and args.credit_toward_next is None,
        ),
        ScreenOut(account_id=account["account_id"], tab="payments"),
    )


def _confirm_payment_claim(args, plan, account, heard, today, data) -> VoiceReplyOut:
    claims = _open_cases(account, kinds={cases.PAYMENT_CLAIM})
    if not claims:
        # Found in eval/ops_voice_benchmark.py: "she paid the full amount today" was read as confirming a claim the
        # borrower never filed. With no claim to confirm, the payment is what was meant -- and the note the model
        # drafted for a claim (nobody dictated it) goes with the claim.
        return _record_payment(args, account, heard, today)
    claim = _pick_case(claims, args, plan, account, "")
    amount = _amount(args, account, heard, "was the payment")
    paid_on = _payment_date(args.paid_on, today)
    _check_drafted(args.note, heard, "note")
    body = {"amount": amount, "payment_date": paid_on.isoformat(), "note": args.note}
    short = _short_of_cycle(amount, account)
    if short and args.credit_toward_next is not None:
        body["apply_extra_to_next"] = args.credit_toward_next
    warnings = [
        f"{inr(p['amount'])} was already recorded on this account (paid {_day(date.fromisoformat(str(p['payment_date'])), today)}). "
        "If that's this payment, close the claim instead."
        for p in claim.get("recent_payments", [])
    ]
    return _proposal(
        f"Confirm {account['borrower_name']}'s payment claim: record {inr(amount)}, paid {_day(paid_on, today)}, and close it?",
        ProposalOut(
            action="confirm_payment_claim", title="Confirm a payment claim",
            details=[DetailOut(label="Borrower", value=_who(account)), DetailOut(label="Claim", value=claim["case_ref"]),
                     DetailOut(label="Their words", value=claim["borrower_words"][:200]),
                     DetailOut(label="Amount", value=inr(amount)), DetailOut(label="Paid", value=_day(paid_on, today))],
            borrower_told=data.claim_confirmed_message(amount, paid_on, None),
            warnings=warnings, method="POST",
            path=f"/accounts/{account['account_id']}/cases/{claim['case_ref']}/confirm-payment", body=body,
            editable="note", editable_label="Note to the borrower (optional, added to the message)",
            credit_choice=short and args.credit_toward_next is None,
        ),
        ScreenOut(account_id=account["account_id"], tab="cases"),
    )


def _close_case(args, plan, account, heard, data) -> VoiceReplyOut:
    case = _pick_case(_open_cases(account, source="dispute"), args, plan, account,
                      f"{account['borrower_name']} has no open dispute or payment claim.")
    _check_drafted(args.note, heard, "note")
    if case["kind"] == cases.PAYMENT_CLAIM and not args.note:
        raise _Unresolved(_clarify(
            "Closing a payment claim without recording it needs a reason -- the borrower is sent it. Why couldn't the "
            "payment be confirmed?", screen=ScreenOut(account_id=account["account_id"], tab="cases"),
        ))
    _, dispute_id = cases.parse_case_ref(case["case_ref"])
    what = "payment claim" if case["kind"] == cases.PAYMENT_CLAIM else "dispute"
    return _proposal(
        f"Close {account['borrower_name']}'s {what} without recording a payment?",
        ProposalOut(
            action="close_case", title=f"Close a {what}",
            details=[DetailOut(label="Borrower", value=_who(account)), DetailOut(label="Case", value=case["case_ref"]),
                     DetailOut(label="Their words", value=case["borrower_words"][:200])],
            borrower_told=data.case_closed_message(case["kind"], None),
            method="POST", path=f"/accounts/{account['account_id']}/disputes/resolve",
            body={"resolution_note": args.note, "dispute_id": dispute_id},
            editable="resolution_note", editable_label="Note to the borrower" + ("" if case["kind"] == cases.PAYMENT_CLAIM else " (optional)"),
            editable_required=case["kind"] == cases.PAYMENT_CLAIM,
        ),
        ScreenOut(account_id=account["account_id"], tab="cases"),
    )


def _log_promise(args, account, heard, today) -> VoiceReplyOut:
    amount = _amount(args, account, heard, "did they promise")
    if args.promised_date is None:
        raise _Unresolved(_clarify("By what date did they promise to pay?"))
    if args.promised_date < today:
        raise _Unresolved(_clarify(f"{_day(args.promised_date, today)} has already passed. By what date did they promise to pay?"))
    return _proposal(
        f"Log a promise from {_who(account)} to pay {inr(amount)} by {_day(args.promised_date, today)}?",
        ProposalOut(
            action="log_promise", title="Log a promise to pay",
            details=[DetailOut(label="Borrower", value=_who(account)), DetailOut(label="Amount", value=inr(amount)),
                     DetailOut(label="By", value=_day(args.promised_date, today))],
            method="POST", path=f"/accounts/{account['account_id']}/promises",
            body={"promised_date": args.promised_date.isoformat(), "promised_amount": amount},
        ),
        ScreenOut(account_id=account["account_id"], tab="cases"),
    )


def _log_call(args, account, heard) -> VoiceReplyOut:
    _check_drafted(args.note, heard, "note")
    outcome = _CALL_OUTCOMES[args.outcome]
    return _proposal(
        f"Log a call to {_who(account)}: {outcome}?",
        ProposalOut(
            action="log_call", title="Log a call",
            details=[DetailOut(label="Borrower", value=_who(account)), DetailOut(label="Outcome", value=outcome)],
            method="POST", path=f"/accounts/{account['account_id']}/call-log",
            body={"outcome": args.outcome, "note": args.note}, editable="note", editable_label="Note (optional)",
        ),
        ScreenOut(account_id=account["account_id"], tab="cases"),
    )


def _delivery(account: dict) -> str:
    return "Telegram and their dashboard" if account["telegram_linked"] else "their dashboard only (no Telegram linked)"


def _message_borrower(args, account, heard) -> VoiceReplyOut:
    _check_drafted(args.message, heard, "message")
    return _proposal(
        f"Send {account['borrower_name']} this message?",
        ProposalOut(
            action="message_borrower", title="Message a borrower",
            details=[DetailOut(label="To", value=_who(account)), DetailOut(label="Goes to", value=_delivery(account))],
            borrower_told=args.message, method="POST",
            path=f"/accounts/{account['account_id']}/clarification-requests", body={"message": args.message},
            editable="message", editable_label="Message", editable_required=True,
        ),
        ScreenOut(account_id=account["account_id"], tab="cases"),
    )


def _message_group(args, data, heard) -> VoiceReplyOut:
    _check_drafted(args.message, heard, "message")
    group = [a for a in _sorted(data.accounts(), "most_overdue") if (
        not a["telegram_linked"] if args.group == "not_on_telegram" else args.group in _flags(a)
    )]
    label = GROUP_LABELS[args.group]
    screen = ScreenOut(view="overview", filter=args.group)
    if not group:
        return _clarify(f"There are no {label} to message.", screen=screen)
    if len(group) > _MAX_GROUP_MESSAGE:
        return _clarify(f"That's {len(group)} {label}; one message can go to at most {_MAX_GROUP_MESSAGE}.", screen=screen)
    names = _list([a["borrower_name"] for a in group[:5]]) + (f" and {len(group) - 5} more" if len(group) > 5 else "")
    return _proposal(
        f"Send this message to all {len(group)} {label}?",
        ProposalOut(
            action="message_group", title=f"Message {len(group)} {label}",
            details=[DetailOut(label="To", value=names)], borrower_told=args.message, method="POST",
            path="/clarification-requests/bulk",
            body={"account_ids": [a["account_id"] for a in group], "message": args.message},
            editable="message", editable_label="Message", editable_required=True,
        ),
        screen,
    )


def _reply_to_case(args, plan, account, heard) -> VoiceReplyOut:
    case = _pick_case([c for c in _open_cases(account) if c["can_reply"]], args, plan, account,
                      f"{account['borrower_name']} has no open case to reply on. Say “send {account['borrower_name']} a message” instead.")
    _check_drafted(args.message, heard, "reply")
    return _proposal(
        f"Reply on {account['borrower_name']}'s case {case['case_ref']}?",
        ProposalOut(
            action="reply_to_case", title="Reply on a case",
            details=[DetailOut(label="Borrower", value=_who(account)), DetailOut(label="Case", value=f"{case['title']} ({case['case_ref']})"),
                     DetailOut(label="Goes to", value=_delivery(account))],
            borrower_told=args.message, method="POST",
            path=f"/accounts/{account['account_id']}/cases/{case['case_ref']}/reply", body={"message": args.message},
            editable="message", editable_label="Reply", editable_required=True,
        ),
        ScreenOut(account_id=account["account_id"], tab="cases"),
    )


def _decide_request(args, plan, account, heard, data, approved: bool) -> VoiceReplyOut:
    case = _pick_case(_open_cases(account, source="escalation"), args, plan, account,
                      f"{account['borrower_name']} has no open request to {'approve' if approved else 'reject'}.")
    reason = None if approved else args.reason
    _check_drafted(reason, heard, "reason")
    verb = "Approve" if approved else "Reject"
    details = [DetailOut(label="Borrower", value=_who(account)), DetailOut(label="Request", value=f"{case['title']} ({case['case_ref']})")]
    terms = case.get("proposed_changes") or {}
    if terms.get("type") == "extend_tenure":
        details.append(DetailOut(label="Terms", value=f"+{terms['extra_months']} months, new EMI {inr(float(terms['new_emi_amount']))} over {terms['new_months_remaining']} months"))
    told = data.decision_message(case, account, approved, None)
    return _proposal(
        f"{verb} {account['borrower_name']}'s {case['title'].lower()}?",
        ProposalOut(
            action="approve_request" if approved else "reject_request", title=f"{verb} a request", details=details,
            borrower_told=told, warnings=[] if told else ["This is an internal notice: the borrower is not told anything."],
            method="POST", path=f"/escalations/{case['case_ref']}/{'approve' if approved else 'reject'}",
            body=None if approved else {"reason": reason},
            editable=None if approved else "reason", editable_label=None if approved else "Reason for the borrower (optional)",
        ),
        ScreenOut(account_id=account["account_id"], tab="cases"),
    )


def _hand_to_person(args, account, heard) -> VoiceReplyOut:
    _check_drafted(args.reason, heard, "reason")
    return _proposal(
        f"Open a case for a person on the team to handle {_who(account)}?",
        ProposalOut(
            action="hand_to_person", title="Hand to a person",
            details=[DetailOut(label="Borrower", value=_who(account))], method="POST",
            path=f"/accounts/{account['account_id']}/escalate", body={"reason": args.reason},
            editable="reason", editable_label="Reason (for the team)", editable_required=True,
        ),
        ScreenOut(account_id=account["account_id"], tab="cases"),
    )


def _send_reminders(args, plan, data, heard, screen_account_id) -> VoiceReplyOut:
    if args.account:
        account = resolve_account(args.account, plan, data, screen_account_id)
        ids, who = [account["account_id"]], _who(account)
        screen = ScreenOut(account_id=account["account_id"])
    else:
        group = args.group or "all"
        if group == "all":
            ids, who = None, "every account that is due"
        else:
            ids = [a["account_id"] for a in data.accounts() if group in _flags(a)]
            if not ids:
                return _clarify(f"There are no {GROUP_LABELS[group]}.", screen=ScreenOut(view="overview", filter=group))
            who = f"the {len(ids)} {GROUP_LABELS[group]}"
        screen = ScreenOut(view="overview", filter=group)
    return _proposal(
        f"Send today's reminders to {who}?",
        ProposalOut(
            action="send_reminders", title="Send reminders now", details=[DetailOut(label="To", value=who)],
            borrower_told="The usual automated payment reminder.",
            warnings=["Only accounts actually due get one, and nobody gets a second reminder today."],
            method="POST", path="/outbound/run", body={"account_ids": ids},
        ),
        screen,
    )


def _update_account(args, account, heard) -> VoiceReplyOut:
    body, details = {}, [DetailOut(label="Account", value=_who(account))]
    if args.phone_number:
        digits = re.sub(r"\D", "", args.phone_number)
        if len(digits) == 10:
            digits = "91" + digits
        if not (10 <= len(digits) <= 15) or digits[-10:] not in spoken_numbers.spoken_digits(heard):
            raise _Unresolved(_clarify("I didn't catch the new phone number clearly. Please say all the digits again."))
        body["phone_number"] = f"+{digits}"
        details.append(DetailOut(label="Phone", value=f"{account['phone_number']} → +{digits}"))
    if args.language:
        body["language_preference"] = args.language
        details.append(DetailOut(label="Language", value=f"{_LANGUAGES.get(account['language_preference'])} → {_LANGUAGES[args.language]}"))
    if args.risk_tier:
        body["risk_tier"] = args.risk_tier
        details.append(DetailOut(label="Risk tier", value=f"{account['risk_tier']} → {args.risk_tier}"))
    if not body:
        raise _Unresolved(_clarify("What should change: the phone number, the language, or the risk tier?"))
    return _proposal(
        f"Update {_who(account)}?",
        ProposalOut(action="update_account", title="Update an account", details=details, method="PATCH",
                    path=f"/accounts/{account['account_id']}", body=body),
        ScreenOut(account_id=account["account_id"], tab="overview"),
    )


def _mark_messages_handled(account) -> VoiceReplyOut:
    unhandled = [c for c in account.get("clarification_requests", []) if not c["resolved"]]
    if not unhandled:
        return _clarify(f"{account['borrower_name']} has no unhandled messages from the team.",
                        screen=ScreenOut(account_id=account["account_id"], tab="cases"))
    return _proposal(
        f"Mark {account['borrower_name']}'s {_n(len(unhandled), 'message')} from the team as handled?",
        ProposalOut(
            action="mark_messages_handled", title="Mark messages handled",
            details=[DetailOut(label="Borrower", value=_who(account)), DetailOut(label="Messages", value=str(len(unhandled)))],
            method="POST", path=f"/accounts/{account['account_id']}/clarification-requests/mark-resolved",
        ),
        ScreenOut(account_id=account["account_id"], tab="cases"),
    )


# --- entry ----------------------------------------------------------------------------------------------------------

def respond(plan: Plan, heard: str, data: DashboardData, screen_account_id: str | None, today: date, now: datetime | None = None) -> VoiceReplyOut:
    """The reply to one understood request. heard is what the operator said (or typed): an amount, phone number or
    figure in a drafted message must be found in it."""
    now = now or datetime.now(timezone.utc)
    args, tool = plan.args, plan.tool
    try:
        if tool == "portfolio_overview":
            reply = _portfolio_overview(data)
        elif tool == "show_accounts":
            reply = _show_accounts(args, data)
        elif tool == "show_inbox":
            reply = _show_inbox(args, data, now)
        elif tool == "escalation_rate":
            reply = _escalation_rate(args, data)
        elif tool == "open_page":
            reply = {
                "portfolio": _answer("Here's the portfolio.", ScreenOut(view="overview", filter="all")),
                "inbox": _answer("Here's the inbox.", ScreenOut(view="inbox")),
                "new_account": _answer("Here's the new-account form. Accounts are opened from the form, not by voice.",
                                       ScreenOut(view="overview", open_form="new_account")),
            }[args.page]
        elif tool == "help":
            reply = _answer(_HELP)
        elif tool == "cannot_help":
            reply = _clarify("Sorry, I can't do that from here. Ask “what can you do?” to hear what I can.")
        elif tool == "message_group":
            reply = _message_group(args, data, heard)
        elif tool == "send_reminders":
            reply = _send_reminders(args, plan, data, heard, screen_account_id)
        else:
            account = resolve_account(args.account, plan, data, screen_account_id)
            reply = {
                "account_details": lambda: _account_details(args, account, data, today),
                "record_payment": lambda: _record_payment(args, account, heard, today),
                "confirm_payment_claim": lambda: _confirm_payment_claim(args, plan, account, heard, today, data),
                "close_case": lambda: _close_case(args, plan, account, heard, data),
                "log_promise": lambda: _log_promise(args, account, heard, today),
                "log_call": lambda: _log_call(args, account, heard),
                "message_borrower": lambda: _message_borrower(args, account, heard),
                "reply_to_case": lambda: _reply_to_case(args, plan, account, heard),
                "approve_request": lambda: _decide_request(args, plan, account, heard, data, approved=True),
                "reject_request": lambda: _decide_request(args, plan, account, heard, data, approved=False),
                "hand_to_person": lambda: _hand_to_person(args, account, heard),
                "update_account": lambda: _update_account(args, account, heard),
                "mark_messages_handled": lambda: _mark_messages_handled(account),
            }[tool]()
    except _Unresolved as e:
        reply = e.reply
    reply.heard = heard
    return reply
