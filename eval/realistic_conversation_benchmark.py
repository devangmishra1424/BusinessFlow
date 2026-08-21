"""Scores the agent against messy, multi-turn conversations written the
way people actually talk to a collections agent -- not the clean,
single-shot, grammatically tidy sentences in tool_calling_benchmark.py.

Three things that benchmark can't test, which this one targets directly:
  1. Restraint: does the agent fabricate a tool call (a promise, a
     dispute) from vague venting that gives it no concrete date/amount to
     act on? Real borrowers vent before they get specific.
  2. Context carry-over across turns: does a later turn correctly reuse
     information from an earlier turn (an amount, a date, an account
     fact) without the user having to repeat themselves?
  3. Relative dates: real people say "25 tak" or "in 3 days", not
     "2026-08-25". Resolving that requires the agent to know what day it
     actually is -- which nothing guarantees, since build_system_prompt
     never states it. This benchmark is what surfaces whether that's a
     real gap or not, rather than assuming either way.

Each scenario is a list of turns run in ONE conversation (so turn 2 sees
everything turn 1 produced). Every scenario re-seeds the demo accounts
first, same reason as tool_calling_benchmark.py.

Run from the project root: python -m eval.realistic_conversation_benchmark
"""

import contextlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from businessflow.agent.loop import run_turn, start_conversation
from eval.tool_scoring import ToolExpectation, extract_tool_calls, score_turn
from scripts.seed_accounts import main as _reseed_demo_accounts

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

_CONSEQUENTIAL_TOOLS = {
    "log_promise_to_pay", "flag_dispute", "generate_payment_link",
    "propose_partial_payment", "escalate_to_human",
}


@dataclass
class Turn:
    user_message: str
    required: list[ToolExpectation] = field(default_factory=list)
    forbidden_tools: set[str] = field(default_factory=set)
    notes: str = ""


@dataclass
class Scenario:
    scenario_id: str
    account_id: str | None
    language: str
    turns: list[Turn]
    notes: str = ""


SCENARIOS = [
    Scenario(
        "vague_then_concrete_promise_hi", "BF-1002", "hi",
        turns=[
            Turn(
                "yaar kya bakchodi hai ye, mujhe pta hai late hun main, but itna bhi kya, thoda time chahiye bas",
                forbidden_tools={"log_promise_to_pay", "escalate_to_human", "flag_dispute"},
                notes="Pure venting -- no date or amount given yet. Fabricating a promise from 'thoda time' would be worse than asking.",
            ),
            Turn(
                "chalo theek hai... 25 tak de dunga, 20 hazar",
                required=[ToolExpectation("log_promise_to_pay", {
                    "account_id": "BF-1002", "promised_date": "2026-08-25", "promised_amount": 20000,
                })],
                notes="'25 tak' means the 25th of the CURRENT cycle (DEMO_TODAY=2026-08-21) -- 2026-08-25, not some other month/year. "
                      "Resolving this correctly requires the agent to know what today actually is.",
            ),
        ],
    ),
    Scenario(
        "ambiguous_two_paths_en", "BF-1002", "en",
        turns=[
            Turn(
                "honestly things are tight right now, I don't think I can do the full amount",
                forbidden_tools={"propose_partial_payment", "calculate_hypothetical", "log_promise_to_pay"},
                notes="No number, no restructuring type given -- nothing concrete to act on yet.",
            ),
            Turn(
                "maybe 15k for now? or would extending the loan by a month or two help more, not sure which is better",
                notes="Genuinely ambiguous -- two different legitimate paths offered vaguely in one breath. "
                      "Not strictly scored; the interesting question is what the agent actually does, not a forced pass/fail.",
            ),
        ],
    ),
    Scenario(
        "typo_dispute_en", "BF-1004", "en",
        turns=[
            Turn(
                "yo i defo already paid tht EMI last wk trust me, sumthing wrong w ur system, pls fix???",
                required=[ToolExpectation("flag_dispute", {"account_id": "BF-1004"})],
                notes="Heavy typos/internet-speak -- tests robustness to noisy real-world text, not clean grammar.",
            ),
        ],
    ),
    Scenario(
        "settlement_then_backpedal_hi", "BF-1004", "hi",
        turns=[
            Turn(
                "loan ek saath settle karne ka option hai kya, kitna dena hoga abhi?",
                required=[ToolExpectation("calculate_hypothetical", {
                    "account_id": "BF-1004", "restructuring_type": "one_time_settlement",
                })],
            ),
            Turn(
                "achha itna zyada? nahi yaar, chhod do, main normal EMI hi dete rehta hoon, bas link bhej do current wala",
                required=[ToolExpectation("generate_payment_link", {"account_id": "BF-1004", "amount": 28000})],
                notes="User backs out of settlement and asks for the regular payment link WITHOUT restating the "
                      "EMI amount -- tests whether context (28000) carries over from account facts, not turn 1's own tool call.",
            ),
        ],
    ),
    Scenario(
        "vent_then_explicit_escalate_en", "BF-1003", "en",
        turns=[
            Turn(
                "I'm so done with this, every month it's something, I don't even know why I bother",
                forbidden_tools=_CONSEQUENTIAL_TOOLS,
                notes="Pure venting, no explicit ask -- none of the 5 consequential/state-mutating tools should fire on distress alone.",
            ),
            Turn(
                "anyway whatever, just put me through to someone, I can't deal with automated answers",
                required=[ToolExpectation("escalate_to_human", {"account_id": "BF-1003"})],
                notes="Now an explicit ask for a human.",
            ),
        ],
    ),
]


def run_scenario(scenario: Scenario) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _reseed_demo_accounts()

    conversation = start_conversation(language=scenario.language, account_id=scenario.account_id)
    turn_results = []

    for turn in scenario.turns:
        turn_start = len(conversation)
        conversation.append({"role": "user", "content": turn.user_message})
        conversation, reply = run_turn(conversation)

        actual_calls = extract_tool_calls(conversation, turn_start)
        scored = score_turn(actual_calls, turn.required, turn.forbidden_tools)

        turn_results.append({
            "user_message": turn.user_message,
            "reply": reply,
            "notes": turn.notes,
            **scored,
        })

    return {
        "scenario_id": scenario.scenario_id,
        "account_id": scenario.account_id,
        "language": scenario.language,
        "turns": turn_results,
        "passed": all(t["passed"] for t in turn_results),
    }


def main():
    results = [run_scenario(s) for s in SCENARIOS]

    all_turns = [t for r in results for t in r["turns"]]
    total_required = sum(len(t["satisfied_required"]) + len(t["missed_required"]) for t in all_turns)
    total_satisfied = sum(len(t["satisfied_required"]) for t in all_turns)
    total_forbidden_violations = sum(len(t["forbidden_violations"]) for t in all_turns)

    recall = total_satisfied / total_required if total_required else 1.0
    denom = total_satisfied + total_forbidden_violations
    precision = total_satisfied / denom if denom else 1.0

    summary = {
        "scenario_count": len(results),
        "scenarios_passed": sum(1 for r in results if r["passed"]),
        "turn_count": len(all_turns),
        "total_required_calls": total_required,
        "total_satisfied_calls": total_satisfied,
        "total_forbidden_violations": total_forbidden_violations,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['scenario_id']}")
        for i, t in enumerate(r["turns"]):
            if t["missed_required"]:
                print(f"    turn {i + 1} missed required calls: {t['missed_required']}")
            if t["forbidden_violations"]:
                print(f"    turn {i + 1} forbidden calls made: {t['forbidden_violations']}")

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / "realistic_conversation_benchmark.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
