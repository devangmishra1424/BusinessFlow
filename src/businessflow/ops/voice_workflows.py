"""Guided work through the ops voice channel: a start-of-day briefing, and two runs through a queue one item at a time
-- the call list (borrowers to phone) and the inbox (open cases) -- so an operator can work by voice instead of
hunting through the dashboard: hear the next borrower or case, say what happened ("no answer"; "reached, she'll pay
8,000 by Friday"; "approve it"), confirm the card, and the page moves on.

Like the rest of the voice channel this only reads. Each step is an answer built from the dashboard's own data
(voice_reply.DashboardData) with the right next actions offered for that item; every action is still a confirmation
card from voice_reply.py, sent only by the operator's Confirm click. The queue is decided once, when the run starts
(voice_reply.WorkflowOut); each step reads its item fresh, so a case a colleague closed in the meantime is passed
over rather than acted on twice.

The page keeps the run (which item, how many done), because a run is one operator's, in one browser tab, and needs
nothing on the server between steps.
"""

from datetime import date, datetime, timedelta, timezone

from businessflow.accounts import cases
from businessflow.ops.voice import Plan
from businessflow.ops.voice_reply import (
    _CALL_OUTCOMES,
    _SOON_ENOUGH_DAYS,
    CASE_KIND_LABELS,
    DashboardData,
    RowOut,
    ScreenOut,
    SuggestionOut,
    VoiceReplyOut,
    WorkflowItemOut,
    WorkflowOut,
    _age,
    _answer,
    _clarify,
    _day,
    _flags,
    _list,
    _n,
    _open_promises,
    _overdue_rupees,
    _phone_row,
    _sorted,
    _when,
    inr,
)

TOOLS = {"daily_briefing", "start_call_list", "start_inbox", "workflow_control", "call_step", "case_step"}

_IST = timezone(timedelta(hours=5, minutes=30))
# The first reply ops sends to a payment claim, word for word as the case card's own "Ask for the reference" button
# (ops/static/app.js ASK_REFERENCE_TEXT), so the borrower gets the same message whichever way it was sent.
ASK_REFERENCE_TEXT = (
    "Thanks for letting us know. Could you share the payment reference (UTR or transaction ID), with the date and "
    "the amount? We'll match it against the bank record."
)
_SKIP = SuggestionOut(label="Skip", tool="workflow_control", arguments={"command": "skip"})
_STOP = SuggestionOut(label="Stop", tool="workflow_control", arguments={"command": "stop"})


def _greeting(now: datetime) -> str:
    hour = now.astimezone(_IST).hour
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"


def _call_queue(accounts: list[dict], group: str) -> tuple[list[dict], list[dict]]:
    """Who to phone, most overdue first, and who is left out: an account with an open dispute or payment claim is not
    chased for payment while it is open (its reminders are paused for the same reason), so it is not on the list."""
    in_group = [a for a in _sorted(accounts, "most_overdue") if group in _flags(a)]
    return [a for a in in_group if "disputed" not in _flags(a)], [a for a in in_group if "disputed" in _flags(a)]


# --- the briefing ---------------------------------------------------------------------------------------------------

def _daily_briefing(data: DashboardData, now: datetime) -> VoiceReplyOut:
    accounts, inbox = data.accounts(), data.inbox()
    to_call, held = _call_queue(accounts, "overdue")
    claims = [c for c in inbox if c["kind"] == cases.PAYMENT_CLAIM]
    stale = [c for c in inbox if (now - c["opened_at"]).total_seconds() > _SOON_ENOUGH_DAYS * 86400]
    broken = [a for a in accounts if "broken_promises" in _flags(a)]
    metrics = data.metrics(24)
    tool_calls = metrics["event_counts"].get("tool_called", 0)

    work = []
    if to_call:
        work.append(f"{_n(len(to_call), 'borrower')} to call, with {inr(_overdue_rupees(to_call), whole=True)} overdue")
    if inbox:
        work.append(f"{_n(len(inbox), 'open case')} in the inbox" + (f", {len(stale)} waiting over 3 days" if stale else ""))
    if not work:
        say = f"{_greeting(now)}. Nothing is overdue and the inbox is clear."
    else:
        todo = _list(work)
        say = f"{_greeting(now)}. {todo[0].upper()}{todo[1:]}. Where do you want to start?"

    by_kind: dict[str, int] = {}
    for c in inbox:
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    rows = [
        RowOut(label="Calls to make", value=str(len(to_call)),
               detail=(f"{inr(_overdue_rupees(to_call), whole=True)} overdue" if to_call else "nothing overdue")
               + (f" · {_n(len(held), 'disputed account')} left out" if held else ""),
               tone="danger" if to_call else "ok", tool="start_call_list", arguments={"group": "overdue"}),
        RowOut(label="Inbox", value=str(len(inbox)),
               detail=_list([_n(n, *CASE_KIND_LABELS.get(k, (k, k + "s"))) for k, n in by_kind.items()]) or "clear",
               tone="warn" if inbox else "ok", tool="start_inbox", arguments={"kind": "any"}),
    ]
    if claims:
        rows.append(RowOut(label="Payment claims to check", value=str(len(claims)), detail="the borrower says they paid",
                           tone="warn", tool="start_inbox", arguments={"kind": cases.PAYMENT_CLAIM}))
    if stale:
        rows.append(RowOut(label="Waiting over 3 days", value=str(len(stale)), detail=_list([c.get("borrower_name") or c["account_id"] for c in stale[:3]]),
                           tone="danger", tool="show_inbox", arguments={}))
    if broken:
        # The list, not a call list: an account with broken promises and an open dispute is shown but not chased.
        rows.append(RowOut(label="Broken promises", value=str(len(broken)), detail=_list([a["borrower_name"] for a in broken[:3]]),
                           tone="warn", tool="show_accounts", arguments={"group": "broken_promises"}))
    rows.append(RowOut(label="Escalation rate, 24 hours",
                       value=f"{round(metrics['escalation_rate'] * 100)}%" if tool_calls else "no activity",
                       detail=f"of {_n(tool_calls, 'AI tool call')}" if tool_calls else None,
                       tool="escalation_rate", arguments={"window": "24h"}))

    reply = _answer(say, ScreenOut(view="overview", filter="all"))
    reply.title = "Today"
    reply.rows = rows
    if to_call:
        reply.suggestions.append(SuggestionOut(label="Start the call list", tool="start_call_list", arguments={"group": "overdue"}))
    if inbox:
        reply.suggestions.append(SuggestionOut(label="Work the inbox", tool="start_inbox", arguments={"kind": "any"}))
    if claims:
        reply.suggestions.append(SuggestionOut(label="Review payment claims", tool="start_inbox", arguments={"kind": cases.PAYMENT_CLAIM}))
    return reply


# --- the call list --------------------------------------------------------------------------------------------------

def _start_call_list(args, data: DashboardData, today: date) -> VoiceReplyOut:
    queue, held = _call_queue(data.accounts(), args.group)
    left_out = f" {_n(len(held), 'account')} with an open dispute or payment claim left out: those aren't chased while it's open." if held else ""
    if not queue:
        nobody = "Nothing is overdue." if args.group == "overdue" else "No accounts with broken promises."
        return _answer(f"No one to call. {nobody}{left_out}", ScreenOut(view="overview", filter=args.group))
    first = data.account(queue[0]["account_id"])
    reply = _call_step(first, today) if first else _gone(queue[0]["account_id"])
    reply.say = f"Call list: {_n(len(queue), 'borrower')}, most overdue first.{left_out} First, " + reply.say
    reply.workflow = WorkflowOut(
        kind="call_list", title="Call list" if args.group == "overdue" else "Call list: broken promises",
        step_tool="call_step",
        items=[WorkflowItemOut(key=a["account_id"], label=a["borrower_name"], arguments={"account": a["account_id"]}) for a in queue],
    )
    return reply


def _call_step(a: dict, today: date) -> VoiceReplyOut:
    calls = a.get("call_log", [])
    called_today = bool(calls) and _when(calls[0]["created_at"]) == today
    parts = [f"{a['borrower_name']}, {a['business_name']}:"]
    if a["days_past_due"] > 0:
        parts.append(f"{_n(a['days_past_due'], 'day')} overdue, {inr(a['amount_due_now'])} due.")
    else:
        parts.append("no longer overdue, nothing to chase.")
    if called_today:
        parts.append(f"Already called today: {_CALL_OUTCOMES.get(calls[0]['outcome'], calls[0]['outcome'])}.")
    pending = _open_promises(a, today)
    if pending:
        parts.append(f"They promised {inr(pending[0]['promised_amount'])} by {_day(_when(pending[0]['promised_date']), today)}.")
    if a["language_preference"] != "en":
        parts.append(f"Prefers {'Hindi' if a['language_preference'] == 'hi' else 'Hinglish'}.")
    parts.append("Tell me how the call goes.")

    rows = [_phone_row(a),
            RowOut(label="Due now", value=inr(a["amount_due_now"]) if a["amount_due_now"] > 0 else "nothing",
                   detail=f"EMI {inr(a['emi_amount'])}" + (f" · {_n(a['days_past_due'], 'day')} overdue" if a["days_past_due"] > 0 else ""),
                   tone="danger" if a["days_past_due"] > 0 else "ok")]
    if calls:
        rows.append(RowOut(label="Last call", value=_CALL_OUTCOMES.get(calls[0]["outcome"], calls[0]["outcome"]),
                           detail=_day(_when(calls[0]["created_at"]), today) + (f" · {calls[0]['note']}" if calls[0].get("note") else ""),
                           tone="warn" if called_today else None))
    broken = sum(p["kept"] is False for p in a.get("promises", []))
    if pending or broken:
        rows.append(RowOut(label="Promises", value=f"{inr(pending[0]['promised_amount'])} open" if pending else "none open",
                           detail=f"{broken} broken before" if broken else None, tone="warn" if broken else None))
    open_cases = [c for c in a.get("cases", []) if c["open"]]
    if open_cases:
        rows.append(RowOut(label="Open cases", value=str(len(open_cases)), detail=_list([c["title"] for c in open_cases]),
                           tone="warn", account_id=a["account_id"], tab="cases"))

    account = {"account": a["account_id"]}
    suggestions = [
        SuggestionOut(label="No answer", tool="log_call", arguments={**account, "outcome": "no_answer"}),
        SuggestionOut(label="Left a voicemail", tool="log_call", arguments={**account, "outcome": "voicemail"}),
        SuggestionOut(label="Reached, promised…", prompt="reached them, they promised ₹ by "),
        SuggestionOut(label="Reached, no promise", tool="log_call", arguments={**account, "outcome": "reached"}),
        SuggestionOut(label="Wrong number", tool="log_call", arguments={**account, "outcome": "wrong_number"}),
        _SKIP, _STOP,
    ]
    if called_today or a["days_past_due"] <= 0:
        suggestions.insert(0, suggestions.pop(suggestions.index(_SKIP)))
    reply = _answer(" ".join(parts), ScreenOut(account_id=a["account_id"], tab="cases"))
    reply.title = f"Call {a['borrower_name']}"
    reply.rows = rows
    reply.suggestions = suggestions
    return reply


# --- the inbox ------------------------------------------------------------------------------------------------------

def _start_inbox(args, data: DashboardData, now: datetime) -> VoiceReplyOut:
    queue = [c for c in data.inbox() if args.kind == "any" or c["kind"] == args.kind]  # already most urgent first
    if not queue:
        plural = CASE_KIND_LABELS[args.kind][1] if args.kind != "any" else "cases"
        return _answer(f"Nothing to work through: no open {plural}.", ScreenOut(view="inbox"))
    first = queue[0]
    reply = _case_step(data.account(first["account_id"]), first["case_ref"], now)
    what = CASE_KIND_LABELS[args.kind][1] if args.kind != "any" else "open cases"
    reply.say = f"{len(queue)} {what if len(queue) != 1 else what[:-1]}, most urgent first. First, " + reply.say
    reply.workflow = WorkflowOut(
        kind="inbox", title="Inbox" if args.kind == "any" else f"Inbox: {CASE_KIND_LABELS[args.kind][1]}", step_tool="case_step",
        items=[WorkflowItemOut(key=c["case_ref"], label=f"{c.get('borrower_name') or c['account_id']}: {c['title']}",
                               arguments={"account": c["account_id"], "case": c["case_ref"]}) for c in queue],
    )
    return reply


_CASE_PROMPTS = {
    cases.PAYMENT_CLAIM: "Confirm it, ask for the payment reference, or close it?",
    cases.DISPUTE: "Reply, or close the dispute?",
    "restructuring": "Approve or reject?",
    "callback": "Call them, then close it.",
    "fraud": "Hand it to a person, or reply?",
    "system": "Close the notice once it's dealt with.",
}


def _case_suggestions(case: dict, account_id: str) -> list[SuggestionOut]:
    target = {"account": account_id, "case": case["case_ref"]}
    kind = case["kind"]
    if kind == cases.PAYMENT_CLAIM:
        options = [
            SuggestionOut(label="Confirm the payment…", prompt="confirm their payment claim of ₹ paid on "),
            SuggestionOut(label="Ask for the reference", tool="reply_to_case", arguments={**target, "message": ASK_REFERENCE_TEXT}),
            SuggestionOut(label="Close: not found…", prompt="close their payment claim because "),
        ]
    elif kind == cases.DISPUTE:
        options = [SuggestionOut(label="Reply…", prompt="reply to them: "),
                   SuggestionOut(label="Close the dispute…", prompt="close the dispute: ")]
    elif kind == "restructuring":
        options = [SuggestionOut(label="Approve", tool="approve_request", arguments=target),
                   SuggestionOut(label="Reject…", prompt="reject it because ")]
    elif kind == "callback":
        options = [SuggestionOut(label="Log the call…", prompt="called them, "),
                   SuggestionOut(label="Close it", tool="approve_request", arguments=target)]
    elif kind == "system":
        options = [SuggestionOut(label="Close the notice", tool="approve_request", arguments=target)]
    else:  # closure, fraud, request
        options = [SuggestionOut(label="Approve / close", tool="approve_request", arguments=target),
                   SuggestionOut(label="Reply…", prompt="reply to them: "),
                   SuggestionOut(label="Hand to a person…", prompt="hand this to a person: ")]
    return [*options, _SKIP, _STOP]


def _case_step(a: dict | None, case_ref: str, now: datetime) -> VoiceReplyOut:
    case = next((c for c in (a or {}).get("cases", []) if c["case_ref"] == case_ref), None)
    if a is None or case is None or not case["open"]:
        # Closed by a colleague (or by this operator elsewhere) since the run started: never act on it twice.
        reply = _answer(f"{case_ref} is already closed. Moving on.")
        reply.workflow_command = "skip"
        return reply
    who = a["borrower_name"]
    words = case["borrower_words"].strip()
    parts = [f"{case['title']} from {who}, open {_age(case['opened_at'], now)}."]
    if case["kind"] != "system":
        parts.append(f"They said: “{words[:160]}{'…' if len(words) > 160 else ''}”")
    recent = case.get("recent_payments") or []
    if case["kind"] == cases.PAYMENT_CLAIM:
        parts.append(f"A payment of {inr(recent[0]['amount'])} is already recorded, so it may be this one."
                     if recent else "No payment recorded on the account in the last few days.")
    terms = case.get("proposed_changes") or {}
    if terms.get("type") == "extend_tenure":
        parts.append(f"They asked for {terms['extra_months']} more months: new EMI {inr(float(terms['new_emi_amount']))} "
                     f"over {terms['new_months_remaining']} months.")
    parts.append(_CASE_PROMPTS.get(case["kind"], "Approve, reply, or hand it to a person?"))

    rows = [RowOut(label="Their words" if case["kind"] != "system" else "Notice", detail=words[:400] or None,
                   value=case["case_ref"]),
            RowOut(label="Open for", value=_age(case["opened_at"], now),
                   tone="danger" if (now - case["opened_at"]).total_seconds() > _SOON_ENOUGH_DAYS * 86400 else "warn")]
    for p in recent[:3]:
        rows.append(RowOut(label="Already recorded", value=inr(p["amount"]), detail=f"paid {_when(p['payment_date']):%d %b}", tone="warn"))
    thread = case.get("thread") or []
    if thread:
        last = thread[-1]
        rows.append(RowOut(label="Last message", value="from them" if last["author"] == "borrower" else "from us",
                           detail=last["message"][:200]))
    rows.append(RowOut(label="Due now", value=inr(a["amount_due_now"]) if a["amount_due_now"] > 0 else "nothing",
                       detail=f"{_n(a['days_past_due'], 'day')} overdue" if a["days_past_due"] > 0 else None,
                       tone="danger" if a["days_past_due"] > 0 else None))
    rows.append(_phone_row(a))

    reply = _answer(" ".join(parts), ScreenOut(account_id=a["account_id"], tab="cases"))
    reply.title = f"{CASE_KIND_LABELS.get(case['kind'], (case['kind'],))[0].capitalize()} · {who}"
    reply.rows = rows
    reply.suggestions = _case_suggestions(case, a["account_id"])
    return reply


def _gone(account_id: str) -> VoiceReplyOut:
    reply = _answer(f"{account_id} isn't there any more. Moving on.")
    reply.workflow_command = "skip"
    return reply


# --- inside a run ---------------------------------------------------------------------------------------------------

_COMMAND_WORDS = {"next": "Next.", "skip": "Skipping this one.", "previous": "Going back.", "stop": "Stopped.", "repeat": "Again:"}


def _workflow_control(args, workflow: str | None) -> VoiceReplyOut:
    if not workflow:
        reply = _clarify("Nothing is running to move through. Say “start my day”, “call list” or “work the inbox” to begin.")
        reply.suggestions = [
            SuggestionOut(label="Start my day", tool="daily_briefing", arguments={}),
            SuggestionOut(label="Call list", tool="start_call_list", arguments={}),
            SuggestionOut(label="Work the inbox", tool="start_inbox", arguments={}),
        ]
        return reply
    reply = _answer(_COMMAND_WORDS[args.command])
    reply.workflow_command = args.command
    return reply


def respond(plan: Plan, data: DashboardData, today: date, now: datetime, workflow: str | None) -> VoiceReplyOut:
    """voice_reply.respond's branch for the guided tools (TOOLS)."""
    args = plan.args
    if plan.tool == "daily_briefing":
        return _daily_briefing(data, now)
    if plan.tool == "start_call_list":
        return _start_call_list(args, data, today)
    if plan.tool == "start_inbox":
        return _start_inbox(args, data, now)
    if plan.tool == "workflow_control":
        return _workflow_control(args, workflow)
    if plan.tool == "call_step":
        account = data.account(args.account)
        return _call_step(account, today) if account else _gone(args.account)
    return _case_step(data.account(args.account), args.case, now)  # case_step
