"""Scores the agent's tool-calling behavior against a fixed set of
single-turn scenarios with known-correct expected outcomes -- the same
"before/after, scored, saved to a results file" discipline as
wer_benchmark.py, applied to the thing that was never measured: does the
agent call the right tool, with the right arguments, and not call tools
it shouldn't?

These are clean, well-formed sentences -- a baseline sanity check that
each tool works at all in the obvious case. For messy, human, multi-turn
conversations (typos, vague phrasing, changing your mind mid-negotiation),
see realistic_conversation_benchmark.py -- that's the harder, more
representative test.

Every scenario runs against the real Groq API, real Postgres (the seeded
demo accounts BF-1001..1004), and the real RAG index -- no mocks. This
requires GROQ_API_KEY, a reachable Postgres (DATABASE_URL), and an
already-seeded chroma_db, same as tests/test_agent_loop.py.

Run from the project root: python -m eval.tool_calling_benchmark
"""

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from businessflow.agent.loop import run_turn, start_conversation
from eval.tool_scoring import (
    ToolExpectation,
    aggregate_results,
    extract_tool_calls,
    print_regression_delta,
    record_run_history,
    score_turn,
)
from scripts.seed_accounts import main as _reseed_demo_accounts

_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class Scenario:
    scenario_id: str
    account_id: str | None
    language: str
    user_message: str
    required: list[ToolExpectation]
    forbidden_tools: set[str]  # "*" (as a member) means "all 8 tools"
    notes: str = ""


SCENARIOS = [
    Scenario(
        "balance_inquiry_en", "BF-1001", "en",
        "What's my current balance and how many days late am I?",
        required=[ToolExpectation("get_payment_status", {"account_id": "BF-1001"})],
        forbidden_tools={"escalate_to_human", "flag_dispute"},
        notes="Routine read-only status check -- shouldn't trigger anything consequential.",
    ),
    Scenario(
        "promise_to_pay_en", "BF-1001", "en",
        "I will pay ₹12500 by August 25 2026, I promise.",
        required=[ToolExpectation("log_promise_to_pay", {
            "account_id": "BF-1001", "promised_date": "2026-08-25", "promised_amount": 12500,
        })],
        forbidden_tools={"escalate_to_human", "flag_dispute"},
    ),
    Scenario(
        "dispute_hi", "BF-1004", "hi",
        "Maine yeh EMI pehle hi UPI se pay kar diya tha last week, lekin account mein nahi dikh raha. "
        "Main is charge ko dispute karna chahta hoon.",
        required=[ToolExpectation("flag_dispute", {"account_id": "BF-1004"})],
        forbidden_tools={"escalate_to_human"},
        notes="reason text is free-form -- only tool+account_id checked, not exact wording.",
    ),
    Scenario(
        "payment_link_en", "BF-1004", "en",
        "Can you send me a payment link for the full ₹28000 I owe this month?",
        required=[ToolExpectation("generate_payment_link", {"account_id": "BF-1004", "amount": 28000})],
        forbidden_tools={"escalate_to_human", "flag_dispute"},
    ),
    Scenario(
        "partial_payment_accept_hi", "BF-1002", "hi",
        "Main is mahine sirf 18000 rupaye de sakta hoon, poora amount nahi. Kya yeh chalega?",
        required=[ToolExpectation("propose_partial_payment", {"account_id": "BF-1002", "proposed_amount": 18000})],
        forbidden_tools={"escalate_to_human", "flag_dispute"},
        notes="18000 >= 70% of 22000 EMI -- policy-eligible, but the eval only checks the tool got called correctly, not the outcome.",
    ),
    Scenario(
        "partial_payment_reject_en", "BF-1004", "en",
        "I can only pay ₹15000 this month, is that okay?",
        required=[ToolExpectation("propose_partial_payment", {"account_id": "BF-1004", "proposed_amount": 15000})],
        forbidden_tools={"escalate_to_human", "flag_dispute"},
        notes="15000 < 70% of 28000 EMI -- policy-ineligible, but the agent should still call the tool to check rather than guess.",
    ),
    Scenario(
        "hypothetical_extend_en", "BF-1001", "en",
        "If I extended my loan by 2 more months, what would my new EMI be?",
        required=[ToolExpectation("calculate_hypothetical", {
            "account_id": "BF-1001", "restructuring_type": "extend_tenure", "extra_months": 2,
        })],
        forbidden_tools={"escalate_to_human", "flag_dispute"},
    ),
    Scenario(
        "hypothetical_settlement_hi", "BF-1004", "hi",
        "Agar main abhi ek baar mein poora loan settle kar doon, toh kitna paisa dena hoga?",
        required=[ToolExpectation("calculate_hypothetical", {
            "account_id": "BF-1004", "restructuring_type": "one_time_settlement",
        })],
        forbidden_tools={"escalate_to_human", "flag_dispute"},
    ),
    Scenario(
        "escalation_forced_en", "BF-1003", "en",
        "This dispute has been open forever and nobody's helping me -- I want to talk to a real person right now.",
        required=[ToolExpectation("escalate_to_human", {"account_id": "BF-1003"})],
        forbidden_tools=set(),
        notes="BF-1003 has an open dispute + 2 broken promises -- calculate_hypothetical is allowed as an extra, neutral step.",
    ),
    Scenario(
        "policy_question_en", None, "en",
        "What is your grace period policy before late fees kick in?",
        required=[ToolExpectation("check_policy", {})],
        forbidden_tools={"escalate_to_human", "flag_dispute", "log_promise_to_pay"},
    ),
    Scenario(
        "no_tool_greeting_en", None, "en",
        "Hi, who am I speaking with, and what languages can you help me in?",
        required=[],
        forbidden_tools={"*"},
        notes="Pure chit-chat -- no account context given, nothing to look up or act on.",
    ),
]


def run_scenario(scenario: Scenario) -> dict:
    # Re-seed before every scenario -- scenarios run against the real,
    # persistent Postgres store, so one scenario's side effect (e.g.
    # dispute_hi opening a dispute) would otherwise leak into any later
    # scenario reusing the same account and expecting a clean start. Bit us
    # on the first real run: 3 "failures" turned out to be BF-1004 correctly
    # reacting to a dispute a *different* scenario had opened on it.
    with contextlib.redirect_stdout(io.StringIO()):
        _reseed_demo_accounts()

    conversation = start_conversation(language=scenario.language, account_id=scenario.account_id)
    turn_start = len(conversation)
    conversation.append({"role": "user", "content": scenario.user_message})
    conversation, reply = run_turn(conversation)

    actual_calls = extract_tool_calls(conversation, turn_start)
    scored = score_turn(actual_calls, scenario.required, scenario.forbidden_tools)

    return {
        "scenario_id": scenario.scenario_id,
        "account_id": scenario.account_id,
        "language": scenario.language,
        "user_message": scenario.user_message,
        "reply": reply,
        **scored,
    }


def main():
    results = [run_scenario(s) for s in SCENARIOS]

    agg = aggregate_results(results)
    scenario_success_rate = sum(1 for r in results if r["passed"]) / len(results)
    summary = {
        "scenario_count": len(results),
        "scenarios_passed": sum(1 for r in results if r["passed"]),
        "scenario_success_rate": round(scenario_success_rate, 4),
        **agg,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['scenario_id']}")
        if r["missed_required"]:
            print(f"    missed required calls: {r['missed_required']}")
        if r["forbidden_violations"]:
            print(f"    forbidden calls made: {r['forbidden_violations']}")

    print("\n=== vs previous run ===")
    previous = record_run_history("tool_calling_benchmark", summary, _RESULTS_DIR)
    print_regression_delta(previous, summary, metrics=("precision", "recall", "scenario_success_rate"))

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / "tool_calling_benchmark.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
