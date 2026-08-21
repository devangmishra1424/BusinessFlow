"""Data model for the mock loan-servicing system of record.

This stands in for a real core banking / LMS backend. Nothing here talks to
money — principal/EMI amounts are illustrative figures on synthetic accounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class PaymentRecord:
    date: date
    amount: float
    on_time: bool


@dataclass
class PromiseToPay:
    made_on: date
    promised_date: date
    promised_amount: float
    # None until the promised_date has passed and is checked against payment_history.
    kept: bool | None = None


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

    payment_history: list[PaymentRecord] = field(default_factory=list)
    promises: list[PromiseToPay] = field(default_factory=list)

    def days_past_due(self, as_of: date) -> int:
        delta = (as_of - self.emi_due_date).days
        return max(delta, 0)

    def broken_promise_count(self) -> int:
        return sum(1 for p in self.promises if p.kept is False)
