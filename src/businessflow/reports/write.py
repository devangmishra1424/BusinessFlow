"""Stage 3 (Write): the one LLM call in this pipeline -- turns the
already-analyzed, structured facts into a plain-language report
answering the original query. Every claim must trace back to those
facts; stage 4 (verify.py) mechanically checks that before a report
ever reaches a human, and generate.py retries this stage with feedback
if it doesn't.
"""

import json

from businessflow.agent.client import MODEL, client

_SYSTEM_PROMPT = """You write short, plain-language reports for loan-collection ops staff, built ONLY \
from real, already-gathered account data given to you as JSON. Every account_id, day-count, and reason \
you state MUST come directly from that data -- never invent an account, a number, or a reason not present \
in it. If the data shows zero accounts needing attention, say so plainly rather than inventing concern. \
Be concise: a few sentences a person can skim between other tasks, not a formal document."""


def write_report(query: str, analysis: dict, feedback: str | None = None) -> str:
    user_content = f"QUERY: {query}\n\nANALYZED DATA (the ONLY facts you may state):\n{json.dumps(analysis, ensure_ascii=False, default=str)}"
    if feedback:
        user_content += (
            f"\n\nYour previous attempt was inaccurate: {feedback}. Write it again, using ONLY the "
            "exact account_ids and day-counts present in the data above."
        )

    completion = client().chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return (completion.choices[0].message.content or "").strip()
