"""Adaptive Outreach Intelligence & Multi-Factor Risk Scoring Engine.

Evaluates an account's risk level dynamically based on multiple quantitative factors:
1. Days Past Due (DPD)
2. Broken promises count
3. Historical payment punctuality
4. Account disputes and pending credits

Maps the overall risk score (0 to 100) into a RiskTier and selects an optimal
ToneStrategy for proactive outbound engagement.

This is the one risk the product shows (2026-10-24): the ops dashboard's tier, its sort, the reminder tone and the
voice assistant all read it. The tier stored on the account (accounts.risk_tier) is the rating given at onboarding --
one input here among the account's own behaviour, not a second, competing answer. Every point added or taken away is
listed in `breakdown` with its reason, so the tier is never a black box.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from businessflow.accounts.models import Account
from businessflow.accounts.policy import GRACE_PERIOD_DAYS
from businessflow.accounts.store import current_date


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToneStrategy(str, Enum):
    EMPATHETIC = "empathetic"
    NEUTRAL_REMINDER = "neutral_reminder"
    FIRM_URGENT = "firm_urgent"
    FINAL_NOTICE = "final_notice"


@dataclass
class RiskProfile:
    tier: RiskTier
    score: float
    recommended_tone: ToneStrategy
    factors: list[str]
    # Each factor with the points it added (or took away), in the order applied: [{"label", "points"}].
    breakdown: list[dict] = field(default_factory=list)


def calculate_account_risk(account: Account, as_of: date | None = None) -> RiskProfile:
    """Calculates multi-factor risk score (0-100) and produces an adaptive RiskProfile."""
    if as_of is None:
        # The same UTC "today" every other days-past-due calculation uses
        # (store.current_date), not the host's local date.
        as_of = current_date()

    dpd = account.days_past_due(as_of)
    broken_promises = account.broken_promise_count()
    score = 0.0
    factors = []
    breakdown: list[dict] = []

    def add(points: float, label: str | None) -> None:
        nonlocal score
        score += points
        if label:
            factors.append(label)
            breakdown.append({"label": label, "points": points})

    # 1. Days Past Due (0 - 50 points)
    if dpd == 0:
        add(5.0, None)
        breakdown.append({"label": "Not past due", "points": 5.0})
    elif dpd <= 5:
        add(20.0, f"{dpd} days past due (grace period)")
    elif dpd <= 15:
        add(35.0, f"{dpd} days past due")
    elif dpd <= 30:
        add(50.0, f"{dpd} days past due (moderate delinquency)")
    else:
        add(65.0, f"{dpd} days past due (severe delinquency)")

    # 2. Broken Promises (+15 per broken promise, up to +45)
    if broken_promises > 0:
        add(min(broken_promises * 15.0, 45.0), f"{broken_promises} broken promise(s) on record")

    # 3. Payment Punctuality History (+15 pts if recent payments late)
    if account.payment_history:
        recent_late = [p for p in account.payment_history[-3:] if not p.on_time]
        if recent_late:
            add(15.0, f"{len(recent_late)} late payment(s) in recent history")

    # 4. Dispute status (+10 pts if dispute open)
    if account.dispute_open:
        add(10.0, "Active dispute flagged on account")

    # 5. The rating given at onboarding (accounts.risk_tier): a prior, worth less than anything the account has done.
    onboarding = {"high": 10.0, "medium": 5.0}.get(getattr(account, "risk_tier", "low"), 0.0)
    if onboarding:
        add(onboarding, f"Rated {account.risk_tier} risk at onboarding")

    # 6. Mitigations: credit already paid ahead, and a promise still on track (the borrower said what they will pay).
    if account.pending_emi_credit > 0:
        add(-10.0, f"Pending credit of {account.pending_emi_credit:,.0f} rupees available")
    if any(p.is_open for p in getattr(account, "promises", []) if hasattr(p, "is_open")):
        add(-10.0, "A promise to pay is open and on track")

    # Clamp score between 0 and 100
    score = min(100.0, max(0.0, score))

    # Tier and Tone Assignment
    #
    # A broken promise already adds to the score above. It only FORCES a
    # tier when the borrower is currently past the grace period: broken
    # promises are a lifetime count (nothing ever expires them), so letting
    # them force "final notice" on their own sent a firm/critical-toned
    # message to a borrower who was fully current -- even as a pre-due
    # heads-up -- for something they did months ago. And a single broken
    # promise never forces anything: accounts/policy.py's own rule is that
    # one slip is normal and a pattern (BROKEN_PROMISES_BEFORE_MANDATORY_
    # ESCALATION) is what matters.
    currently_delinquent = dpd > GRACE_PERIOD_DAYS
    if score >= 75.0 or (broken_promises >= 2 and currently_delinquent):
        tier = RiskTier.CRITICAL
        tone = ToneStrategy.FINAL_NOTICE
    elif score >= 50.0:
        tier = RiskTier.HIGH
        tone = ToneStrategy.FIRM_URGENT
    elif score >= 25.0:
        tier = RiskTier.MEDIUM
        tone = ToneStrategy.NEUTRAL_REMINDER
    else:
        tier = RiskTier.LOW
        tone = ToneStrategy.EMPATHETIC

    return RiskProfile(
        tier=tier,
        score=score,
        recommended_tone=tone,
        factors=factors,
        breakdown=breakdown,
    )
