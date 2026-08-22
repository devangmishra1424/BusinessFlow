"""Stage 2 (Analyze): finds the pattern in the gathered facts -- which
accounts actually need attention, and why, ranked by real severity. No
AI here either -- a simple, explainable ordering (more distinct flag
types first, then more days overdue), not an opaque scored risk model,
same reasoning as ops/flags.py itself.
"""

# Order matters here: a dispute or a broken-promise pattern is a bigger
# deal than being merely overdue, so an account carrying more of these
# flag types ranks first regardless of exact day counts.
_SEVERITY_FLAGS = ["disputed", "broken_promises", "overdue"]


def analyze_accounts_needing_attention(account_facts: list[dict]) -> dict:
    flagged = [a for a in account_facts if a["flags"]]

    def _severity_key(account: dict) -> tuple:
        labels = {f["label"] for f in account["flags"]}
        return (-sum(1 for s in _SEVERITY_FLAGS if s in labels), -account["days_past_due"])

    flagged_sorted = sorted(flagged, key=_severity_key)

    return {
        "total_accounts": len(account_facts),
        "accounts_needing_attention": len(flagged_sorted),
        "flagged_accounts": flagged_sorted,
    }
