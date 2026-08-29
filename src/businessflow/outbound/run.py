"""On-demand orchestrator: decide -> compose -> send for every account
that needs a reminder today. Idempotent against being run twice in one
day for the same account+kind -- checks the events log for an existing
reminder_sent event of that kind today before sending again, reusing
the events table rather than adding a new dedup table (same approach
already used everywhere else in this project that logs activity).

No scheduler in THIS module -- it stays a plain, on-demand function, same
as blueprint §13's own explicit "no scheduler" choice for the report
feature. scripts/run_outbound_scheduler.py is the real trigger: a
standalone, long-running process that calls run_daily_outbound_pass()
once a day. Point a real OS/cloud cron at scripts/run_outbound_pass.py's
main() instead once real hosting exists -- this function itself doesn't
change either way.
"""

from datetime import datetime, time, timezone

from businessflow.accounts import store
from businessflow.outbound.compose import compose_message
from businessflow.outbound.decide import decide_reminders
from businessflow.outbound.send import send_reminder
from businessflow.tools.payment_tools import generate_payment_link


def _already_sent_today(account_id: str, kind: str) -> bool:
    since_midnight = datetime.combine(store.current_date(), time.min, tzinfo=timezone.utc)
    return store.has_recent_event_with_detail(account_id, "reminder_sent", since_midnight, "kind", kind)


def run_daily_outbound_pass(account_ids: list[str] | None = None) -> list[dict]:
    sent = []
    for reminder in decide_reminders(account_ids):
        if _already_sent_today(reminder.account_id, reminder.kind):
            continue
        account = store.get_account_or_raise(reminder.account_id)
        message = compose_message(account, reminder)
        # A real, single-use payment link on every reminder kind -- even a
        # heads_up borrower paying a few days early, or a follow_up
        # borrower already past the grace period, both benefit from "pay
        # now" being one tap away just as much as someone reminded on the
        # exact due date. generate_payment_link mints a fresh token per
        # reminder (accounts.store.create_payment_token) -- never reused
        # across sends, so an old reminder's link can't outlive this one.
        link = generate_payment_link(reminder.account_id, account.emi_amount)
        send_reminder(reminder.account_id, reminder.kind, message, link["payment_link"], account.emi_amount)
        sent.append({"account_id": reminder.account_id, "kind": reminder.kind, "days": reminder.days, "message": message})
    return sent
