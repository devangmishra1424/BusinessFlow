"""Data model for the mock loan-servicing system of record.

This stands in for a real core banking / LMS backend. Nothing here talks to
money — principal/EMI amounts are illustrative figures on synthetic accounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class PaymentRecord:
    date: date
    amount: float
    on_time: bool
    # "regular" (matched that cycle's EMI, business as usual), "extra_unapplied"
    # (an off-cycle amount the borrower chose NOT to apply to the schedule --
    # recorded, but months_remaining/emi_due_date never moved), "extra_applied"
    # (an off-cycle amount the borrower chose to credit toward the next EMI),
    # or "overpayment_applied" (paid more than the EMI due that cycle; the
    # excess was automatically credited toward the next EMI, no confirmation
    # needed since overpaying is never ambiguous). See
    # accounts/store.py's record_payment for where this is decided.
    kind: str = "regular"


@dataclass
class PromiseToPay:
    made_on: date
    promised_date: date
    promised_amount: float
    # None until the promised_date has passed and is checked against payment_history.
    kept: bool | None = None


@dataclass
class Escalation:
    escalation_id: str
    account_id: str
    reason: str
    status: str  # "queued_for_human" | "approved" | "rejected"
    created_at: datetime
    resolved_at: datetime | None = None
    # Structured terms a human can apply with one click (see
    # tools/escalation_tools.py's propose_restructuring) -- None for every
    # other escalation kind, which is just a free-form human hand-off.
    proposed_changes: dict | None = None
    # Optional, ops-entered explanation shown back to the borrower on
    # rejection. None is a valid, expected value, not a missing field.
    resolution_reason: str | None = None


@dataclass
class Account:
    account_id: str
    borrower_name: str
    business_name: str
    phone_number: str  # E.164
    language_preference: str  # "hi" | "en" | "hinglish"

    loan_type: str
    principal_amount: float
    emi_amount: float
    tenure_months: int
    months_remaining: int
    emi_due_date: date  # due date of the current, unpaid EMI cycle

    nach_mandate_active: bool
    dispute_open: bool
    risk_tier: str  # "low" | "medium" | "high"

    # A running credit from a prior off-cycle extra payment the borrower
    # chose to apply, or the excess from an overpayment -- subtracted from
    # emi_amount to get what's actually due THIS cycle (see record_payment).
    # Zero for the overwhelming majority of accounts, which never make an
    # off-cycle payment at all.
    pending_emi_credit: float = 0.0

    # Extracted from an uploaded, signed loan agreement (see
    # rag/extraction.py's extract_loan_terms) -- None for most accounts
    # until their agreement is uploaded and successfully parsed. None
    # means "not extracted yet," not zero.
    interest_rate_pct: float | None = None

    # The Telegram chat_id this account last verified from -- None for
    # accounts never verified over Telegram. See outbound/send.py's real
    # delivery path for a restructuring approval/rejection notification.
    telegram_chat_id: int | None = None

    payment_history: list[PaymentRecord] = field(default_factory=list)
    promises: list[PromiseToPay] = field(default_factory=list)

    def days_past_due(self, as_of: date) -> int:
        delta = (as_of - self.emi_due_date).days
        return max(delta, 0)

    def broken_promise_count(self) -> int:
        return sum(1 for p in self.promises if p.kept is False)
