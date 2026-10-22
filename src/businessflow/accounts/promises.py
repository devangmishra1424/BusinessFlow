"""A promise to pay, judged by its own words.

The owner's rule (2026-10-22): when a borrower promises, they say exactly what -- one payment or parts, each an
amount by a date -- and if the words of the promise are broken, the promise is broken. So:

- A plan is one or more parts. Part k is kept once the money paid since the promise was made reaches the total of
  parts 1..k, on or before part k's date. Paying in pieces is fine; paying late is not.
- A part whose date has passed (the next day) without that total is broken, and every later open part of the same
  plan is void: the plan was broken at that part. One broken plan counts once, however many parts it had.
- There is no extra window after the date: the window used to live only in a policy document, never in what the
  borrower said.

Pure: no I/O, no clock of its own (callers pass as_of). accounts/store.py reads and writes the rows.
"""

from dataclasses import dataclass
from datetime import date

MAX_PARTS = 4
MAX_DAYS_AHEAD = 90
_CENT = 0.005


@dataclass(frozen=True)
class Part:
    date: date
    amount: float


@dataclass(frozen=True)
class PartState:
    """One part's verdict as of a date: 'kept', 'broken', 'void' (a later part of a broken plan), or None (open)."""
    part_no: int
    resolution: str | None
    paid_toward: float  # how much of THIS part the payments so far cover (0..amount)


def validate_parts(parts: list[Part], today: date, ceiling: float) -> list[Part]:
    """Checks a plan the way the borrower must have stated it: 1 to MAX_PARTS parts, each a positive amount by a date
    from today to MAX_DAYS_AHEAD days ahead, dates strictly increasing, and the total no more than `ceiling` (what the
    account could owe at most). Raises ValueError saying what is wrong; returns the parts, amounts rounded to paise."""
    if not 1 <= len(parts) <= MAX_PARTS:
        raise ValueError(f"a promise has 1 to {MAX_PARTS} parts, got {len(parts)}")
    previous = None
    for i, p in enumerate(parts, 1):
        if not (p.amount > 0):
            raise ValueError(f"part {i}: the amount must be more than zero")
        if p.date < today:
            raise ValueError(f"part {i}: {p.date.isoformat()} is in the past")
        if (p.date - today).days > MAX_DAYS_AHEAD:
            raise ValueError(f"part {i}: {p.date.isoformat()} is more than {MAX_DAYS_AHEAD} days away")
        if previous is not None and p.date <= previous:
            raise ValueError(f"part {i}: each part's date must come after the one before it")
        previous = p.date
    total = round(sum(p.amount for p in parts), 2)
    if total > ceiling + 0.01:
        raise ValueError(f"the promise totals {total:,.2f}, more than this account could owe ({ceiling:,.2f})")
    return [Part(date=p.date, amount=round(p.amount, 2)) for p in parts]


def terms_text(parts: list[Part]) -> str:
    """The terms as they are read back to the borrower and kept on the record: "₹10,000 by 25 Oct 2026, then ₹12,500 by
    2 Nov 2026". Dates are written out in full so nothing about which month or year is left to guess."""
    def one(p: Part) -> str:
        return f"₹{p.amount:,.0f} by {p.date.day} {p.date.strftime('%b')} {p.date.year}"
    return ", then ".join(one(p) for p in parts)


def evaluate_plan(parts: list[Part], made_on: date, payments: list[tuple[date, float]], as_of: date) -> list[PartState]:
    """Each part's verdict as of as_of. payments are (payment_date, amount) for the account; only those on or after
    made_on count toward the plan (money paid before the promise was not promised money)."""
    counted = sorted((d, a) for d, a in payments if d >= made_on)
    states: list[PartState] = []
    required = 0.0
    broken = False
    for i, part in enumerate(parts, 1):
        before = required
        required = round(required + part.amount, 2)
        paid_by_date = sum(a for d, a in counted if d <= part.date)
        paid_so_far = sum(a for d, a in counted if d <= min(as_of, part.date))
        toward = round(max(0.0, min(part.amount, paid_so_far - before)), 2)
        if broken:
            states.append(PartState(i, "void", toward))
        elif paid_so_far + _CENT >= required:
            states.append(PartState(i, "kept", part.amount))
        elif part.date < as_of:
            # The date has passed: whatever reached it by then is all that will.
            states.append(PartState(i, "kept" if paid_by_date + _CENT >= required else "broken", toward))
            broken = states[-1].resolution == "broken"
        else:
            states.append(PartState(i, None, toward))
    return states


def plan_status(states: list[PartState]) -> str:
    """'kept' (every part kept), 'broken' (a part was broken), or 'on_track' (still open, nothing broken)."""
    if any(s.resolution == "broken" for s in states):
        return "broken"
    if states and all(s.resolution == "kept" for s in states):
        return "kept"
    return "on_track"


def plan_view(parts, payments: list[tuple[date, float]], as_of: date) -> dict | None:
    """One plan, as every screen and the assistant describe it: its terms, status, each part with how much of it is
    paid, and the next part still owed (date, amount left), or None. `parts` are one plan's PromiseToPay rows in any
    order (accounts/models.py); payments are (payment_date, amount). None for no parts."""
    if not parts:
        return None
    parts = sorted(parts, key=lambda p: p.part_no)
    states = evaluate_plan([Part(p.promised_date, p.promised_amount) for p in parts], parts[0].made_on, payments, as_of)
    # A recorded verdict wins over a recomputed one (a part voided by a newer promise stays void).
    rows = []
    for p, st in zip(parts, states):
        status = p.resolution or st.resolution or "open"
        rows.append({"part_no": p.part_no, "date": p.promised_date.isoformat(), "amount": p.promised_amount,
                     "paid_toward": st.paid_toward, "status": status})
    nxt = next((r for r in rows if r["status"] == "open"), None)
    statuses = {r["status"] for r in rows}
    status = "broken" if "broken" in statuses else "kept" if statuses == {"kept"} else "replaced" if "open" not in statuses and "void" in statuses else "on_track"
    return {
        "plan_id": parts[0].plan_id, "terms": parts[0].terms_text, "made_on": parts[0].made_on.isoformat(),
        "made_via": parts[0].made_via, "status": status, "parts": rows,
        "next_part": {"date": nxt["date"], "amount_left": round(nxt["amount"] - nxt["paid_toward"], 2)} if nxt else None,
    }


def promise_view_for(account, as_of: date) -> dict | None:
    """plan_view of the account's open promise plan, or of its most recent plan if none is open; None if it never
    promised. Uses the account's own payment history, so no I/O beyond what loaded the account."""
    plan = account.open_plan()
    if not plan and account.promises:
        latest = max(account.promises, key=lambda p: (p.made_on, p.id or 0))
        plan = [p for p in account.promises if p.plan_id == latest.plan_id] if latest.plan_id else [latest]
    payments = [(r.date, r.amount) for r in account.payment_history]
    return plan_view(plan, payments, as_of)
