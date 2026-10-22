"""Collection health: the handful of numbers a collections head reads the book by, each with how it was worked out.

- Days-past-due buckets: how many accounts are current, 1-30, 31-60, 61-90 and 90+ days past due, with what each
  bucket owes now and what is still to come on its loans.
- Collection efficiency (last 30 days): money received / (money received + what is past due now). Of what fell due,
  how much came in; an EMI not yet due is not counted as owed.
- Promises kept (last 90 days): promises (plans, not parts) whose outcome fell in the window, kept / (kept + broken).
  A promise replaced by a newer one is neither.
- Cure rate (last 90 days): accounts charged a late fee in the window (they went past the grace period) that are now
  not past due at all.
- Containment comes from the event log (observability/metrics.py), so ops/api.py adds it; it is not computed here.

Every rate is None when nothing is in its denominator -- "no data" is not 0% or 100%.

Pure: accounts in (accounts/store.py list_accounts, with payment history, promises and charges), numbers out.
"""

from datetime import date, timedelta

from businessflow.accounts.dues import amount_due_now
from businessflow.accounts.models import Account

BUCKETS = (("current", "Current", 0, 0), ("1_30", "1-30 days", 1, 30), ("31_60", "31-60 days", 31, 60),
           ("61_90", "61-90 days", 61, 90), ("90_plus", "90+ days", 91, None))
COLLECTION_WINDOW_DAYS = 30
OUTCOME_WINDOW_DAYS = 90


def _rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator > 0 else None


def dpd_buckets(accounts: list[Account], as_of: date) -> list[dict]:
    """Every live account (months_remaining > 0) in exactly one bucket."""
    out = []
    for key, label, lo, hi in BUCKETS:
        members = [a for a in accounts if a.months_remaining > 0
                   and lo <= a.days_past_due(as_of) and (hi is None or a.days_past_due(as_of) <= hi)]
        out.append({
            "key": key, "label": label, "accounts": len(members),
            "account_ids": [a.account_id for a in members],
            "owed_now": round(sum(amount_due_now(a) for a in members), 2),
            "book": round(sum(a.emi_amount * a.months_remaining for a in members), 2),
        })
    return out


def collection_efficiency(accounts: list[Account], as_of: date) -> dict:
    since = as_of - timedelta(days=COLLECTION_WINDOW_DAYS)
    collected = round(sum(p.amount for a in accounts for p in a.payment_history if since < p.date <= as_of), 2)
    owed = round(sum(amount_due_now(a) for a in accounts if a.days_past_due(as_of) > 0), 2)
    return {
        "value": _rate(collected, collected + owed), "collected": collected, "still_owed": owed,
        "window_days": COLLECTION_WINDOW_DAYS,
        "how": f"Money received in the last {COLLECTION_WINDOW_DAYS} days, out of that plus everything past due now.",
    }


def _plan_outcomes(account: Account) -> list[tuple[str, date]]:
    """(outcome, the date it was settled) for each promise plan on the account that has an outcome."""
    plans: dict[str, list] = {}
    for i, p in enumerate(account.promises):
        plans.setdefault(p.plan_id or f"row-{i}", []).append(p)
    out = []
    for parts in plans.values():
        if any(p.kept is False for p in parts):
            broken = min(p.promised_date for p in parts if p.kept is False)
            out.append(("broken", broken))
        elif all(p.kept is True for p in parts):
            out.append(("kept", max(p.promised_date for p in parts)))
    return out


def promises_kept(accounts: list[Account], as_of: date) -> dict:
    since = as_of - timedelta(days=OUTCOME_WINDOW_DAYS)
    outcomes = [o for a in accounts for o, d in _plan_outcomes(a) if since < d <= as_of]
    kept, broken = outcomes.count("kept"), outcomes.count("broken")
    return {
        "value": _rate(kept, kept + broken), "kept": kept, "broken": broken, "window_days": OUTCOME_WINDOW_DAYS,
        "how": f"Promises whose last date (or the date they were broken) fell in the last {OUTCOME_WINDOW_DAYS} days: "
               "kept, out of kept plus broken. A promise replaced by a newer one is not counted.",
    }


def cure_rate(accounts: list[Account], as_of: date) -> dict:
    since = as_of - timedelta(days=OUTCOME_WINDOW_DAYS)
    went_late = [a for a in accounts if any(c.kind == "late_fee" and since < c.assessed_on <= as_of for c in a.charges)]
    cured = [a for a in went_late if a.days_past_due(as_of) == 0]
    return {
        "value": _rate(len(cured), len(went_late)), "cured": len(cured), "went_late": len(went_late),
        "cured_ids": [a.account_id for a in cured], "window_days": OUTCOME_WINDOW_DAYS,
        "how": f"Accounts that went past the grace period in the last {OUTCOME_WINDOW_DAYS} days (charged a late fee) "
               "and are now not past due at all.",
    }


def portfolio_health(accounts: list[Account], as_of: date) -> dict:
    return {
        "as_of": as_of.isoformat(),
        "dpd_buckets": dpd_buckets(accounts, as_of),
        "collection_efficiency": collection_efficiency(accounts, as_of),
        "promises_kept": promises_kept(accounts, as_of),
        "cure_rate": cure_rate(accounts, as_of),
    }
