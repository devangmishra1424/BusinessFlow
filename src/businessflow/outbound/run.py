"""On-demand orchestrator: decide -> compose -> send for every account
that needs a reminder today, plus resolve_promises() (evaluating matured
promises-to-pay against real payment history) and run_daily_pass(), which
runs both together -- the one real entrypoint the scheduler/cron scripts
call. run_daily_outbound_pass is idempotent against being run twice in one
day for the same account+kind -- checks the events log for an existing
reminder_sent event of that kind today before sending again, reusing
the events table rather than adding a new dedup table (same approach
already used everywhere else in this project that logs activity).

No scheduler in THIS module -- it stays a plain, on-demand function, same
as blueprint §13's own explicit "no scheduler" choice for the report
feature. scripts/run_outbound_scheduler.py is the real trigger: a
standalone, long-running process that calls run_daily_pass() once a day.
Point a real OS/cloud cron at scripts/run_outbound_pass.py's main()
instead once real hosting exists -- this module itself doesn't change
either way.
"""

from datetime import datetime, time, timezone

from businessflow.accounts import store
from businessflow.accounts.dues import reminder_amounts
from businessflow.accounts.escalation_kinds import BROKEN_PROMISE_PATTERN_PREFIX, CHRONIC_DELINQUENCY_REASON
from businessflow.accounts.policy import BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION, MANDATORY_ESCALATION_DAYS_PAST_DUE
from businessflow.outbound.compose import compose_message
from businessflow.outbound.decide import decide_reminders
from businessflow.outbound.send import send_reminder
from businessflow.tools.escalation_tools import escalate_to_human
from businessflow.tools.payment_tools import generate_payment_link

# Fixed, with no day count baked in -- create_escalation's own
# account_id+reason dedup (see accounts/store.py) relies on this exact
# string staying stable across days, so an account that's still over the
# threshold tomorrow reuses today's still-open escalation instead of
# opening a second one. A NEW escalation with this same reason only opens
# again once a human has actually resolved the last one and the account is
# STILL over threshold afterward -- which is correct, not a bug.
_CHRONIC_DELINQUENCY_REASON = CHRONIC_DELINQUENCY_REASON  # shared with accounts/escalation_kinds.py so the classifier can't drift


def _already_sent_today(account_id: str, kind: str) -> bool:
    since_midnight = datetime.combine(store.current_date(), time.min, tzinfo=timezone.utc)
    return store.has_recent_event_with_detail(account_id, "reminder_sent", since_midnight, "kind", kind)


from businessflow.outbound.risk_engine import calculate_account_risk


def run_daily_outbound_pass(account_ids: list[str] | None = None) -> list[dict]:
    sent = []
    for reminder in decide_reminders(account_ids):
        if _already_sent_today(reminder.account_id, reminder.kind):
            continue
        # A follow_up reminder alone fires identically forever once an
        # account clears GRACE_PERIOD_DAYS, with no ceiling -- found live,
        # this never itself escalates to a human no matter how delinquent
        # an account gets. Escalating doesn't replace sending the reminder
        # (the borrower should still hear from the reminder itself); it
        # adds a human into the loop once daily nagging alone clearly isn't
        # working.
        if reminder.kind == "follow_up" and reminder.days >= MANDATORY_ESCALATION_DAYS_PAST_DUE:
            escalate_to_human(reminder.account_id, _CHRONIC_DELINQUENCY_REASON)
        account = store.get_account_or_raise(reminder.account_id)
        # What the reminder quotes and the link is minted for: EMI still due
        # this cycle (less any credit) plus the late fee for a follow_up --
        # see accounts/dues.py. Previously the bare emi_amount, which ignored
        # credit and the fee and sent borrowers to a pay page that called the
        # difference an "extra payment".
        _emi, _late_fee, amount_due = reminder_amounts(account, reminder.kind)
        if _emi <= 0:
            # This cycle's EMI is already covered by credit from an earlier
            # extra payment -- there is nothing to remind them to pay.
            continue
        risk_profile = calculate_account_risk(account)
        message = compose_message(account, reminder, risk_profile=risk_profile)
        # A real, single-use payment link on every reminder kind -- even a
        # heads_up borrower paying a few days early, or a follow_up
        # borrower already past the grace period, both benefit from "pay
        # now" being one tap away just as much as someone reminded on the
        # exact due date. generate_payment_link mints a fresh token per
        # reminder (accounts.store.create_payment_token) -- never reused
        # across sends, so an old reminder's link can't outlive this one.
        link = generate_payment_link(reminder.account_id, amount_due)
        delivered = send_reminder(reminder.account_id, reminder.kind, message, link["payment_link"], amount_due)
        sent.append({
            "account_id": reminder.account_id,
            "kind": reminder.kind,
            "days": reminder.days,
            "risk_tier": risk_profile.tier.value,
            "recommended_tone": risk_profile.recommended_tone.value,
            "risk_score": risk_profile.score,
            "message": message,
            "delivered_via_telegram": delivered,
        })
    return sent


def resolve_promises() -> dict:
    """Evaluates every matured promise-to-pay against real payment history
    (accounts.store.resolve_matured_promises -- see its own docstring for
    the real gap this closes), then escalates any account whose broken-
    promise count just crossed BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION.

    Checked with == , not >=: this fires the escalation exactly once, at
    the moment an account crosses the threshold -- an account already well
    past it (a human already saw the first escalation) doesn't get a new
    one every single day this runs, only the actual crossing does."""
    resolved = store.resolve_matured_promises()
    newly_broken_account_ids = {r["account_id"] for r in resolved if r["kept"] is False}

    escalated = []
    for account_id in newly_broken_account_ids:
        account = store.get_account_or_raise(account_id)
        new_breaks = len([r for r in resolved if r["account_id"] == account_id and r["kept"] is False])
        prev_count = account.broken_promise_count() - new_breaks
        if prev_count < BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION <= account.broken_promise_count():
            result = escalate_to_human(
                account_id,
                f"{BROKEN_PROMISE_PATTERN_PREFIX} -- {account.broken_promise_count()} broken promises on record",
            )
            escalated.append(result)

    return {"resolved": resolved, "escalated": escalated}


def run_daily_pass(account_ids: list[str] | None = None) -> dict:
    """The one real entrypoint scripts/run_outbound_scheduler.py and
    scripts/run_outbound_pass.py call. Promises are resolved first, before
    today's reminders go out -- a promise broken just now should already
    count toward broken_promise_count() for anything reminders/flags
    compute later in this same pass, not stale data from before today's
    evaluation."""
    promises_result = resolve_promises()
    sent = run_daily_outbound_pass(account_ids)
    return {
        "promises_resolved": promises_result["resolved"],
        "escalated_for_broken_promises": promises_result["escalated"],
        "reminders_sent": sent,
    }
