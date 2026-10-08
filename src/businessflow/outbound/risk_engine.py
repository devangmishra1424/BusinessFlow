"""Adaptive Outreach Intelligence & Multi-Factor Risk Scoring Engine.

Evaluates an account's risk level dynamically based on multiple quantitative factors:
1. Days Past Due (DPD)
2. Broken promises count
3. Historical payment punctuality
4. Account disputes and pending credits

Maps the overall risk score (0 to 100) into a RiskTier and selects an optimal
ToneStrategy for proactive outbound engagement.
"""

from dataclasses import dataclass
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

    # 1. Days Past Due (0 - 50 points)
    if dpd == 0:
        score += 5.0
    elif dpd <= 5:
        score += 20.0
        factors.append(f"{dpd} days past due (grace period)")
    elif dpd <= 15:
        score += 35.0
        factors.append(f"{dpd} days past due")
    elif dpd <= 30:
        score += 50.0
        factors.append(f"{dpd} days past due (moderate delinquency)")
    else:
        score += 65.0
        factors.append(f"{dpd} days past due (severe delinquency)")

    # 2. Broken Promises (+15 per broken promise, up to +45)
    if broken_promises > 0:
        add_pts = min(broken_promises * 15.0, 45.0)
        score += add_pts
        factors.append(f"{broken_promises} broken promise(s) on record")

    # 3. Payment Punctuality History (+15 pts if recent payments late)
    if account.payment_history:
        recent_late = [p for p in account.payment_history[-3:] if not p.on_time]
        if recent_late:
            score += 15.0
            factors.append(f"{len(recent_late)} late payment(s) in recent history")
    
    # 4. Dispute status (+10 pts if dispute open)
    if account.dispute_open:
        score += 10.0
        factors.append("Active dispute flagged on account")

    # 5. Pending credit mitigation (-10 pts if account has positive credit)
    if account.pending_emi_credit > 0:
        score = max(0.0, score - 10.0)
        factors.append(f"Pending credit of {account.pending_emi_credit:,.0f} rupees available")

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
    )
