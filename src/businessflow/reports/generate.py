"""Orchestrates the full Gather -> Analyze -> Write -> Accuracy-check
pipeline (blueprint's own §13) -- on-demand only, no scheduler, matching
that section's explicit "sidesteps needing the host awake on a timer"
reasoning.

On an accuracy-check failure, retries Write with feedback about what
was wrong, bounded to _MAX_WRITE_RETRIES -- if it's still ungrounded
after that many tries, the report comes back explicitly flagged
unverified rather than silently shipping something that failed its own
check.
"""

from dataclasses import dataclass

from businessflow.reports.analyze import analyze_accounts_needing_attention
from businessflow.reports.gather import gather_account_facts
from businessflow.reports.verify import check_report_accuracy
from businessflow.reports.write import write_report

_MAX_WRITE_RETRIES = 2


@dataclass
class Report:
    query: str
    text: str
    verified: bool
    analysis: dict
    retries_used: int


def generate_report(query: str, account_ids: list[str] | None = None) -> Report:
    account_facts = gather_account_facts(account_ids)
    analysis = analyze_accounts_needing_attention(account_facts)

    feedback = None
    text = ""
    for attempt in range(_MAX_WRITE_RETRIES + 1):
        text = write_report(query, analysis, feedback=feedback)
        failure = check_report_accuracy(text, account_facts)
        if not failure:
            return Report(query=query, text=text, verified=True, analysis=analysis, retries_used=attempt)
        feedback = failure.describe()

    return Report(query=query, text=text, verified=False, analysis=analysis, retries_used=_MAX_WRITE_RETRIES)
