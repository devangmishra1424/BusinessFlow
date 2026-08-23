"""One-off: runs ONLY many_operations_same_account_en (real Groq/Postgres
calls) 5 times to check whether round 4 -- the exact case README's "known
gaps" section calls out as unresolved -- actually passes now that
guardrail/unverified_restructuring.py exists, or whether that guardrail
commit landed without anyone re-checking this specific scenario.

Run: python -m scripts.verify_restructuring_guardrail
"""

import json

from eval.realistic_conversation_benchmark import SCENARIOS, run_scenario

_GUARDRAIL_SAFE_REPLY = "Let me connect you with someone who can confirm those exact details before we go further."

scenario = next(s for s in SCENARIOS if s.scenario_id == "many_operations_same_account_en")

for i in range(5):
    result = run_scenario(scenario)
    round4 = result["turns"][3]
    reply_kind = (
        "guardrail_deflected" if round4["reply"] == _GUARDRAIL_SAFE_REPLY
        else "tool_called" if not round4["missed_required"]
        else "UNVERIFIED_CLAIM_LEAKED"
    )
    print(f"run {i + 1}: round4 passed={round4['passed']} reply_kind={reply_kind}")
    if reply_kind == "UNVERIFIED_CLAIM_LEAKED":
        print(json.dumps(round4, indent=2, ensure_ascii=False))
