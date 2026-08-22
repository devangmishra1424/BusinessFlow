"""Stage 4 (Accuracy check): mechanically verifies every account_id and
day-count mentioned in the written report actually traces back to the
gathered facts -- the same "ground before it's shown" idea as
guardrail/grounding.py's chat-reply check, applied to a report instead
of a conversational reply.

Scoped deliberately to account_ids and days-past-due-style day-counts
(the two things this report's example claims actually are), not every
possible number -- a report legitimately mentioning an unrelated
constant (e.g. quoting a policy's grace-period days) would false-
positive here, the same imperfect-but-real trade-off grounding.py
documents for derived arithmetic.
"""

import re

_ACCOUNT_ID_RE = re.compile(r"\bBF-\d+\b")
_DAYS_RE = re.compile(r"(\d+)\s*days?\b", re.IGNORECASE)


class ReportAccuracyFailure:
    def __init__(self, unknown_account_ids: set[str], unmatched_day_counts: set[int]):
        self.unknown_account_ids = unknown_account_ids
        self.unmatched_day_counts = unmatched_day_counts

    def __bool__(self) -> bool:
        return bool(self.unknown_account_ids or self.unmatched_day_counts)

    def describe(self) -> str:
        parts = []
        if self.unknown_account_ids:
            parts.append(f"mentions account(s) not in the gathered data: {sorted(self.unknown_account_ids)}")
        if self.unmatched_day_counts:
            parts.append(
                f"states a day-count not matching any gathered account's days_past_due: {sorted(self.unmatched_day_counts)}"
            )
        return "; ".join(parts)


def check_report_accuracy(report_text: str, account_facts: list[dict]) -> ReportAccuracyFailure:
    known_ids = {a["account_id"] for a in account_facts}
    known_days = {a["days_past_due"] for a in account_facts}

    mentioned_ids = set(_ACCOUNT_ID_RE.findall(report_text))
    mentioned_days = {int(m) for m in _DAYS_RE.findall(report_text)}

    return ReportAccuracyFailure(mentioned_ids - known_ids, mentioned_days - known_days)
