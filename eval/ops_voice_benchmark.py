"""Does the ops voice model pick the right ability, with the right details, from
how an operator actually talks? ops/voice.py's one model call, scored on fixed
requests with a known-correct answer each: questions, navigation, actions,
transcript-style mistakes ("fatma kahn"), requests that refer to the profile
on screen ("her number"), things it must refuse (resetting an access key, the
weather), and requests with no figure, where it must NOT supply one.

Scored on the model's choice only -- what voice.understand() returns -- not on
what voice_reply.py then does with it, which is deterministic and covered by
tests/test_ops_voice.py. In particular a wrong amount here would still be
caught downstream (it must appear in what was said), so "invented a figure" is
reported separately: it is the failure that check exists for.

Real Groq calls on OPS_VOICE_MODEL (default openai/gpt-oss-120b), one per
request, about 2,450 tokens each: roughly 95k tokens of that model's daily
allowance per run. No database. Requires GROQ_API_KEY.

Run from the project root: python -m eval.ops_voice_benchmark
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from businessflow.ops import voice
from eval.stats import wilson_ci

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
TODAY = date(2026, 10, 15)  # a Thursday; fixed so "yesterday" and "on Monday" have one right answer

_PORTFOLIO = "the Portfolio"
_FATIMA = "the profile of Fatima Khan (BF-1003)"
_ASHA = "the profile of Asha Verma (BF-1002)"
_RAVI = "the profile of Ravi Kumar (BF-1001)"


@dataclass
class Case:
    case_id: str
    said: str
    tools: set[str]  # any of these is a right choice
    expect: dict = field(default_factory=dict)  # argument -> expected value; see _arg_ok for the forms
    screen: str = _PORTFOLIO
    no_figure: bool = False  # the operator said no amount: the model must not supply one


PRESENT = object()  # the argument must be given (a message, a reason)
CURRENT = object()  # the account must be the one on screen: "current", or that account's own ID or name


CASES = [
    # questions
    Case("portfolio_count", "how many accounts do we have", {"portfolio_overview", "show_accounts"}),
    Case("portfolio_overview", "give me a quick overview of the portfolio", {"portfolio_overview"}),
    Case("who_overdue", "who is overdue", {"show_accounts"}, {"group": "overdue"}),
    Case("call_first", "who should i call first today", {"show_accounts"}, {"group": "overdue"}),
    Case("overdue_money", "how much money is stuck in overdue emis", {"show_accounts", "portfolio_overview"}),
    Case("disputes_open", "any disputes open right now", {"show_accounts", "show_inbox"}),
    Case("broken_promises", "which accounts have broken promises", {"show_accounts"}, {"group": "broken_promises"}),
    Case("high_risk", "show me the high risk accounts", {"show_accounts"}, {"sort": "highest_risk"}),
    Case("no_telegram", "who isn't on telegram yet", {"show_accounts"}, {"group": "not_on_telegram"}),
    Case("find_name", "find sharma", {"show_accounts", "account_details"}),
    Case("inbox", "what's in the inbox", {"show_inbox"}),
    Case("payment_claims", "any payment claims waiting", {"show_inbox"}, {"kind": "payment_claim"}),
    Case("rate_week", "what's the escalation rate this week", {"escalation_rate"}, {"window": "7d"}),
    Case("rate_month", "escalation rate for the last month", {"escalation_rate"}, {"window": "30d"}),
    Case("owes", "how much does fatima khan owe", {"account_details"}, {"account": "fatima", "topic": "dues"}),
    Case("last_paid", "when did ravi last pay", {"account_details"}, {"account": "ravi", "topic": "payments"}),
    Case("her_phone", "what's her phone number", {"account_details"}, {"account": CURRENT, "topic": "contact"}, _FATIMA),
    Case("open_by_id", "open BF 1002", {"account_details"}, {"account": "1002"}),
    # navigation, help, refusals
    Case("go_inbox", "take me to the inbox", {"open_page", "show_inbox"}),
    Case("new_account", "i want to open a new account", {"open_page"}, {"page": "new_account"}),
    Case("help", "what can you do", {"help"}),
    Case("weather", "what's the weather in mumbai", {"cannot_help"}),
    Case("reset_key", "reset fatima's access key", {"cannot_help"}),
    # actions
    Case("pay_words", "record twelve thousand five hundred from fatima khan paid yesterday", {"record_payment"},
         {"account": "fatima", "amount": 12500, "paid_on": date(2026, 10, 14)}),
    Case("pay_full", "she paid the full amount today by upi", {"record_payment"},
         {"account": CURRENT, "full_amount_due": True}, _FATIMA, no_figure=True),
    Case("pay_no_amount", "record a payment from ravi", {"record_payment"}, {"account": "ravi"}, no_figure=True),
    Case("pay_misheard", "record a payment for fatma kahn of five thousand", {"record_payment"},
         {"account": "fatma", "amount": 5000}),
    Case("promise", "log a promise from ravi to pay 8000 by the 25th", {"log_promise"},
         {"account": "ravi", "amount": 8000, "promised_date": date(2026, 10, 25)}),
    Case("call", "called imran no answer", {"log_call"}, {"account": "imran", "outcome": "no_answer"}),
    Case("message", "tell fatima we've received her documents", {"message_borrower"}, {"account": "fatima", "message": PRESENT}),
    Case("approve", "approve ravi's request for more time", {"approve_request"}, {"account": "ravi"}),
    Case("reject_here", "reject it, the documents don't match", {"reject_request"}, {"account": CURRENT, "reason": PRESENT}, _RAVI),
    Case("remind_overdue", "send reminders to everyone overdue", {"send_reminders"}, {"group": "overdue"}),
    Case("hand_over", "hand asha's account to a person, she sounds distressed", {"hand_to_person"},
         {"account": "asha", "reason": PRESENT}),
    Case("new_number", "change her number to 98123 45678", {"update_account"}, {"account": CURRENT, "phone_number": "9812345678"}, _ASHA),
    Case("confirm_claim", "confirm fatima's payment claim, 12500 paid on monday", {"confirm_payment_claim"},
         {"account": "fatima", "amount": 12500, "paid_on": date(2026, 10, 12)}),
    Case("close_dispute", "close fatima's dispute, it was a bank error that's now fixed", {"close_case"},
         {"account": "fatima", "note": PRESENT}),
    Case("mark_handled", "mark imran's messages as handled", {"mark_messages_handled"}, {"account": "imran"}),
]

_SCREEN_ACCOUNTS = {_FATIMA: ("bf-1003", "fatima"), _ASHA: ("bf-1002", "asha"), _RAVI: ("bf-1001", "ravi")}


def _arg_ok(case: Case, name: str, expected, got) -> bool:
    if expected is PRESENT:
        return bool(got and str(got).strip())
    if expected is CURRENT:
        text = str(got or "current").lower()
        return text in ("current", "this", "her", "him", "them") or any(k in text for k in _SCREEN_ACCOUNTS[case.screen])
    if name == "phone_number":
        return expected in "".join(ch for ch in str(got or "") if ch.isdigit())
    if name == "account":
        return expected in str(got or "").lower().replace(" ", "")
    return got == expected


def run() -> dict:
    results = []
    for case in CASES:
        started = time.monotonic()
        try:
            plan = voice.understand(case.said, TODAY, case.screen)
            tool, args = plan.tool, plan.args.model_dump(mode="python")
            error = None
        except voice.NotUnderstood as e:
            tool, args, error = None, {}, str(e)
        elapsed = time.monotonic() - started
        tool_ok = tool in case.tools
        wrong_args = [n for n, v in case.expect.items() if not _arg_ok(case, n, v, args.get(n))] if tool_ok else list(case.expect)
        invented = case.no_figure and args.get("amount") is not None
        results.append({
            "case_id": case.case_id, "said": case.said, "screen": case.screen, "expected_tools": sorted(case.tools),
            "tool": tool, "tool_ok": tool_ok, "wrong_args": wrong_args, "ok": tool_ok and not wrong_args and not invented,
            "invented_figure": invented, "seconds": round(elapsed, 2), "error": error,
            "arguments": {k: (v.isoformat() if isinstance(v, date) else v) for k, v in args.items() if v not in (None, False, "")},
        })
        print(f"{'ok ' if results[-1]['ok'] else 'BAD'} {case.case_id:<16} {tool} {results[-1]['arguments']} {wrong_args or ''}")

    n = len(results)
    full = sum(r["ok"] for r in results)
    tool_only = sum(r["tool_ok"] for r in results)
    full_rate, full_lo, full_hi = wilson_ci(full, n)
    tool_rate, tool_lo, tool_hi = wilson_ci(tool_only, n)
    seconds = sorted(r["seconds"] for r in results)
    summary = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": os.environ.get("OPS_VOICE_MODEL") or voice._DEFAULT_MODEL,
        "requests": n,
        "right_ability_and_details": {"count": full, "rate": round(full_rate, 3), "ci95": [round(full_lo, 3), round(full_hi, 3)]},
        "right_ability": {"count": tool_only, "rate": round(tool_rate, 3), "ci95": [round(tool_lo, 3), round(tool_hi, 3)]},
        "invented_a_figure": sum(r["invented_figure"] for r in results),
        "no_figure_requests": sum(c.no_figure for c in CASES),
        "seconds_median": seconds[n // 2], "seconds_max": seconds[-1],
        "results": results,
    }
    _RESULTS_DIR.mkdir(exist_ok=True)
    (_RESULTS_DIR / "ops_voice_benchmark.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nright ability and details: {full}/{n} ({full_rate:.0%}, 95% CI {full_lo:.0%}-{full_hi:.0%}); "
        f"right ability: {tool_only}/{n}; invented a figure: {summary['invented_a_figure']} of {summary['no_figure_requests']}; "
        f"median {summary['seconds_median']} s per request"
    )
    return summary


if __name__ == "__main__":
    run()
