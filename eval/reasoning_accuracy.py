"""Checks something the tool-calling benchmarks and the mechanical
Guardrail both miss: does a reply's STATED REASONING actually match the
real tool results, not just "did the right tool get called" (tool_
calling_benchmark.py / realistic_conversation_benchmark.py) or "does
every ₹ amount and URL trace back to something real" (guardrail/
grounding.py, which only checks presence, not whether a real number got
attached to the right claim).

Mirrors the blueprint's own report-generation "accuracy check" pattern
(gather real facts -> write -> mechanically/judgementally verify every
claim traces back) applied to a live conversational reply instead of a
report: an LLM judge is given the REAL tool call + result (ground
truth) and the agent's actual reply, and has to say whether the reply's
claims about that specific fact are accurate -- not vibes-based, scoped
to one concrete, checkable claim per scenario.

Every scenario runs against the real Groq API, real Postgres, and the
real RAG index -- no mocks, same requirements as tool_calling_benchmark.py.

Run from the project root: python -m eval.reasoning_accuracy
"""

import contextlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import groq

from businessflow.agent.client import MODEL, client
from businessflow.agent.loop import extract_tool_calls_with_results, run_turn, start_conversation
from eval.tool_scoring import print_regression_delta, record_run_history
from scripts.seed_accounts import main as _reseed_demo_accounts

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

_JUDGE_SYSTEM_PROMPT = """You are checking whether an AI collections agent's reply is FACTUALLY \
ACCURATE about one specific thing, given the REAL tool call results from that same turn (ground truth).

You will be given:
1. The real tool call(s) made this turn and their real results (ground truth -- trust this completely).
2. The specific claim to check.
3. The agent's actual reply to the borrower.

Judge ONLY the specific claim given -- ignore tone, phrasing, language, and anything else in the reply. \
The reply does not need to restate every field, only not be WRONG about the specific claim when it does \
address it. If the reply doesn't address the claim at all, that's INACCURATE (a real, checkable claim \
should be reflected in an on-topic reply), not something to give a pass on.

Respond in EXACTLY this format, nothing else:
VERDICT: ACCURATE or INACCURATE
REASON: one sentence, citing the specific real value and what the reply said instead if inaccurate"""


def _judge(tool_calls_with_results: list[dict], claim: str, reply: str) -> dict:
    ground_truth = json.dumps(tool_calls_with_results, indent=2, ensure_ascii=False)
    user_content = f"REAL TOOL RESULTS (ground truth):\n{ground_truth}\n\nCLAIM TO CHECK:\n{claim}\n\nAGENT'S REPLY:\n{reply}"
    try:
        completion = client().chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
    except groq.GroqError as e:
        return {"accurate": None, "reason": f"judge call failed: {e}", "raw": None}

    accurate = None
    reason = text
    for line in text.splitlines():
        if line.upper().startswith("VERDICT:"):
            accurate = "ACCURATE" in line.upper() and "INACCURATE" not in line.upper()
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return {"accurate": accurate, "reason": reason, "raw": text}


@dataclass
class Scenario:
    scenario_id: str
    account_id: str | None
    language: str
    user_message: str
    claim: str
    notes: str = ""


SCENARIOS = [
    Scenario(
        "dispute_block_reason_en", "BF-1003", "en",
        "Can you lower my monthly payment or extend my loan? I really need some breathing room here.",
        claim="BF-1003 has an open dispute AND 2+ broken promises-to-pay on record -- either fact alone "
              "blocks automated restructuring per policy. The reply's stated reason for any block/refusal "
              "must correctly reflect the account's real, actual blocking condition(s), not a vague or "
              "wrong reason (e.g. not blaming it on 'system limitations' or a made-up cooldown period).",
        notes="Tests whether the reasoning behind a block is accurate, not just whether a tool got called.",
    ),
    Scenario(
        "settlement_amount_accuracy_hi", "BF-1004", "hi",
        "Agar main abhi ek baar mein poora loan settle kar doon, toh kitna paisa dena hoga?",
        claim="The specific settlement amount (in ₹) that the reply states as what the borrower would need "
              "to pay must exactly match calculate_hypothetical's actual returned total for a one_time_settlement "
              "-- not a different number, not the plain outstanding balance without the discount applied.",
        notes="Guardrail already checks the number appears somewhere in tool results -- this checks it's "
              "attached to the RIGHT claim (the settlement total), not just present anywhere.",
    ),
    Scenario(
        "grace_period_policy_accuracy_en", None, "en",
        "How many days do I actually get after my due date before anything bad happens?",
        claim="The real grace period policy (from check_policy's retrieved KB text) is exactly 3 days, with "
              "no late fee inside that window. The reply must state 3 days -- not a different number "
              "(e.g. 5, 7, or a vague 'a few days'), and must not claim a fee applies inside the grace period.",
        notes="Tests whether a policy fact gets restated correctly, not fabricated or approximated.",
    ),
    Scenario(
        "partial_payment_minimum_accuracy_en", "BF-1004", "en",
        "I can only pay ₹15000 this month instead of the full amount, would that work?",
        claim="propose_partial_payment's real result states whether ₹15000 meets the policy minimum "
              "(70% of the standard EMI) for BF-1004, and if not, what the actual minimum acceptable amount "
              "is. The reply's stated reason/number must match that real result exactly, not a guessed or "
              "rounded figure.",
        notes="BF-1004's EMI is ₹28000 -- 70% is ₹19600, so ₹15000 is genuinely below the real minimum.",
    ),
    Scenario(
        "extend_tenure_new_emi_accuracy_en", "BF-1001", "en",
        "If I extended my loan by 2 more months, what would my new monthly payment actually be?",
        claim="The reply's stated new EMI amount after a 2-month extension must exactly match "
              "calculate_hypothetical's real returned new_emi_amount for BF-1001 with extra_months=2 -- "
              "not the original unchanged EMI, and not an invented figure.",
    ),
]


def run_scenario(scenario: Scenario) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _reseed_demo_accounts()

    conversation = start_conversation(language=scenario.language, account_id=scenario.account_id)
    turn_start = len(conversation)
    conversation.append({"role": "user", "content": scenario.user_message})
    conversation, reply = run_turn(conversation)

    tool_calls_with_results = extract_tool_calls_with_results(conversation, turn_start)
    verdict = _judge(tool_calls_with_results, scenario.claim, reply)

    return {
        "scenario_id": scenario.scenario_id,
        "account_id": scenario.account_id,
        "user_message": scenario.user_message,
        "claim": scenario.claim,
        "reply": reply,
        "tool_calls": tool_calls_with_results,
        **verdict,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    results = [run_scenario(s) for s in SCENARIOS]

    judged = [r for r in results if r["accurate"] is not None]  # excludes judge-call failures from the rate
    accurate_count = sum(1 for r in judged if r["accurate"])
    summary = {
        "scenario_count": len(results),
        "judged_count": len(judged),
        "judge_call_failures": len(results) - len(judged),
        "accurate_count": accurate_count,
        "reasoning_accuracy_rate": round(accurate_count / len(judged), 4) if judged else None,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    for r in results:
        if r["accurate"] is True:
            status = "ACCURATE"
        elif r["accurate"] is False:
            status = "INACCURATE"
        else:
            status = "JUDGE_FAILED"
        print(f"[{status}] {r['scenario_id']}")
        print(f"    reason: {r['reason']}")

    print("\n=== vs previous run ===")
    previous = record_run_history("reasoning_accuracy", summary, _RESULTS_DIR)
    print_regression_delta(previous, summary, metrics=("reasoning_accuracy_rate",))

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / "reasoning_accuracy.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
