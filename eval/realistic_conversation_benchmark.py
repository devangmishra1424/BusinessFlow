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

_CONSEQUENTIAL_TOOLS = {
    "log_promise_to_pay", "flag_dispute", "generate_payment_link",
    "propose_partial_payment", "escalate_to_human",
}

# The exact safe-deflection text agent/loop.py's _finalize_reply
# substitutes in on a Guardrail failure -- used to recognize when a
# Turn's accept_guardrail_intervention override actually applies.
_GUARDRAIL_SAFE_REPLY = "Let me connect you with someone who can confirm those exact details before we go further."


@dataclass
class Turn:
    user_message: str
    required: list[ToolExpectation] = field(default_factory=list)
    required_any: list[ToolExpectation] = field(default_factory=list)  # at least one must happen
    forbidden_tools: set[str] = field(default_factory=set)
    # The mechanical Guardrail (guardrail/unverified_restructuring.py)
    # correctly intervening -- rewriting an unverified claim to a safe
    # deflection -- counts as satisfying `required` too, not just the
    # tool call actually happening. Protecting the borrower from an
    # unverified claim is the real goal; a strict tool-call-only check
    # can't see that as a pass on its own.
    accept_guardrail_intervention: bool = False
    notes: str = ""


@dataclass
class Scenario:
    scenario_id: str
    account_id: str | None
    language: str
    turns: list[Turn]
    notes: str = ""


def _apply_guardrail_intervention_override(scored: dict, turn: Turn, reply: str) -> dict:
    if not turn.accept_guardrail_intervention or reply != _GUARDRAIL_SAFE_REPLY:
        return scored
    scored = dict(scored)
    scored["satisfied_required"] = scored["satisfied_required"] + scored["missed_required"]
    scored["satisfied_required_tools"] = scored["satisfied_required_tools"] + scored["missed_required_tools"]
    scored["missed_required"] = []
    scored["missed_required_tools"] = []
    scored["passed"] = not scored["forbidden_violations"]
    scored["guardrail_intervened"] = True
    return scored


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
    Scenario(
        "messy_early_payoff_question_en", "BF-1002", "en",
        turns=[
            Turn(
                "yo this loan thing is dragging on forever lol, if i just paid the WHOLE thing off right now "
                "and closed it out for good today, how much would that actually run me total",
                required=[ToolExpectation("calculate_hypothetical", {
                    "account_id": "BF-1002", "restructuring_type": "one_time_settlement",
                })],
                notes="Never says 'settlement' or 'restructuring' -- real users don't know the product's own "
                      "vocabulary for 'pay it all off early and close the loan'.",
            ),
            Turn(
                "wait so closing it early is actually cheaper than just riding out the rest of the EMIs? "
                "is that for real or nah, whats the catch",
                notes="Not strictly scored -- the real check here is manual: does the reply correctly state the "
                      "5% discount (SETTLEMENT_DISCOUNT_PCT) rather than inventing a different number or catch?",
            ),
        ],
    ),
    Scenario(
        "many_operations_same_account_en", "BF-1003", "en",
        notes="Multiple rounds, multiple DIFFERENT operations, all on the SAME account/data point within one "
              "session -- BF-1003 has a real open dispute + 2 broken promises seeded in, so several of these "
              "should come back policy-blocked, not just succeed. Tests session coherence across many operations, "
              "not just a single exchange.",
        turns=[
            Turn(
                "hey whats going on with my account, kya scene hai",
                required=[ToolExpectation("get_payment_status", {"account_id": "BF-1003"})],
                notes="Round 1: a plain status check to open the session.",
            ),
            Turn(
                "can you lower my monthly payment or stretch it out longer, im struggling here",
                required_any=[
                    ToolExpectation("calculate_hypothetical", {"account_id": "BF-1003"}),
                    ToolExpectation("escalate_to_human", {"account_id": "BF-1003"}),
                ],
                notes="Round 2: restructuring request on a dispute-blocked account. Either checking eligibility "
                      "(which will come back ineligible) or escalating directly given the already-known block "
                      "are both correct -- the prompt explicitly permits either. Requiring only one specific "
                      "tool here was a scoring bug, not an agent one: found live, the model correctly escalated "
                      "directly in 2 of 3 runs after the check_dispute_block_first prompt fix, and the eval was "
                      "marking that as a failure.",
            ),
            Turn(
                "ok forget that, whats even the deal with my dispute, when does that usually get sorted",
                required=[ToolExpectation("check_policy")],
                notes="Round 3: a policy question, switching topic entirely -- tests the session doesn't get "
                      "stuck on the previous (blocked) restructuring topic.",
            ),
            Turn(
                "fine whatever, can i at least pay like 20000 now instead of the full amount",
                required_any=[
                    ToolExpectation("propose_partial_payment", {"account_id": "BF-1003", "proposed_amount": 20000}),
                    ToolExpectation("escalate_to_human", {"account_id": "BF-1003"}),
                ],
                accept_guardrail_intervention=True,
                notes="Round 4: a THIRD different operation, with a concrete number this time -- also expected "
                      "to come back policy-blocked (same dispute). Found live, across several runs: the model "
                      "sometimes calls propose_partial_payment (gets the real blocked result), sometimes "
                      "escalates directly with the amount captured in the reason (also a real, verified action -- "
                      "guardrail/unverified_restructuring.py's own _VERIFYING_TOOLS already treats these two as "
                      "equally acceptable) -- requiring only one specific tool here was the same scoring rigidity "
                      "round 2 above already needed required_any for. accept_guardrail_intervention stays as a "
                      "further fallback for the rarer case where NEITHER tool gets called but the mechanical "
                      "Guardrail still catches and safely deflects an unverified claim.",
            ),
            Turn(
                "ugh ok, i can promise to pay 20k by the 28th then, will figure the rest out later",
                required=[ToolExpectation("log_promise_to_pay", {
                    "account_id": "BF-1003", "promised_amount": 20000, "promised_date": "2026-08-28",
                })],
                notes="Round 5: a promise-to-pay is NOT blocked by the dispute policy (only automated "
                      "restructuring/partial-payment are) -- this one should actually succeed.",
            ),
            Turn(
                "send me the link for that then",
                required=[ToolExpectation("generate_payment_link", {"account_id": "BF-1003", "amount": 20000})],
                notes="Round 6: 'that' must resolve to the 20000 just promised in round 5, not the full EMI or "
                      "the round-4 partial-payment amount (same number here, but for the right reason).",
            ),
            Turn(
                "you know what, this is too much back and forth, just get me an actual human",
                required=[ToolExpectation("escalate_to_human", {"account_id": "BF-1003"})],
                notes="Round 7: explicit escalation request after a long, winding session -- final operation.",
            ),
        ],
    ),
    Scenario(
        "compound_account_status_question_en", "BF-1001", "en",
        notes="Found live via the Telegram channel (not scripted here first): a single, natural message "
              "bundling several account facts at once got answered with NO tool call at all -- 'your loan "
              "balance is Rs 12,500... you have 0 months left' -- wrong (BF-1001's real months_remaining is "
              "14), while the EMI figure happened to be right by coincidence. get_payment_status now returns "
              "principal_amount/tenure_months/months_remaining/outstanding_balance_approx (it didn't expose "
              "any of them before), and the prompt has an explicit grounding instruction for account facts "
              "(agent/client.py's _GROUND_ACCOUNT_FACTS) -- this scenario is the regression test for the "
              "actual failure, not just the tool/prompt fix in isolation.",
        turns=[
            Turn(
                "how much is the loan amount in my banking and how many months of emi is left and what is "
                "the emi payment and interest",
                required=[ToolExpectation("get_payment_status", {"account_id": "BF-1001"})],
                notes="One compound question, one tool call -- get_payment_status alone answers the loan "
                      "amount, months remaining, and EMI parts. The interest part has no real answer (this "
                      "system tracks no interest rate field at all), so this only requires the tool call, not "
                      "a specific claim about interest -- eval/tool_scoring.py can't check 'did it correctly "
                      "decline to answer part of a question', only whether a real tool got called before "
                      "answering the parts that ARE real.",
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
        scored = score_turn(actual_calls, turn.required, turn.forbidden_tools, turn.required_any)
        scored = _apply_guardrail_intervention_override(scored, turn, reply)

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
    agg = aggregate_results(all_turns)
    summary = {
        "scenario_count": len(results),
        "scenarios_passed": sum(1 for r in results if r["passed"]),
        **agg,
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

    print("\n=== vs previous run ===")
    previous = record_run_history("realistic_conversation_benchmark", summary, _RESULTS_DIR)
    print_regression_delta(previous, summary)

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / "realistic_conversation_benchmark.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
