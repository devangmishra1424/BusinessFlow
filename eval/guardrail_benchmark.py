"""How well the reply guardrail's number-and-link check (guardrail/grounding.py: check_grounding) separates a reply that states
a figure the conversation actually established from one that invented it.

This is the project's hallucination detector for the two kinds of claim that do real damage in a lending conversation: a rupee
amount and a payment link. A reply may state an amount only if it traces to a tool result or to what the borrower said, and a
link only if a tool produced it. Deterministic, no model, so it runs in CI and costs nothing.

How the cases are built. One conversation holds the facts (an EMI of 12,500, a balance of 175,000, an amount due of 13,000, a
late fee of 500, a payment link, and an amount the borrower says they can pay). FABRICATED replies state an amount or link that
is not in it, in every format the model really writes (rupee sign, Rs., INR, "rupees", lakh/crore, Devanagari digits, Indian
digit grouping), inside several sentence shapes. GROUNDED replies state amounts that are in it, in the same formats, including
the roundings people use. Both come from the same templates, so a format cannot be easy on one side only.

Limits, stated up front. The cases are generated from templates by the engineer who wrote the check, so they probe the formats
it was built for; a format nobody thought of is not in here. The check is literal by design -- it compares values, it does not do
arithmetic -- so a correct DERIVED figure (a 5% discount computed by the model) is blocked; the benchmark reports that
separately as the known trade-off it is, not as a miss. It checks amounts and links only: a confident wrong statement about a
date or a tenure is not in scope.

Run from the repo root:  python -m eval.guardrail_benchmark
"""

from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

from eval.stats import wilson_ci

RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"

LINK = "https://chat.example.test/pay/AbC123xyz"
CONVERSATION = [
    {"role": "system", "content": "You are a collections agent."},
    {"role": "user", "content": "What do I owe? I could manage 8000 this week."},
    {"role": "tool", "tool_call_id": "1", "content": json.dumps({
        "emi_amount": 12500.0, "outstanding_balance_approx": 175000.0, "amount_due_now": 13000.0, "late_fee_amount": 500.0,
        "principal_amount": 250000.0, "payment_link": LINK,
    })},
]

_DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def _indian_grouping(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def _formats(value: float) -> dict[str, str]:
    """The ways the model writes one amount."""
    whole = int(value)
    has_paise = abs(value - whole) > 1e-9
    plain = f"{value:,.2f}" if has_paise else f"{whole:,}"
    out = {
        "rupee sign": f"₹{plain}", "Rs.": f"Rs. {plain}", "Rs no dot": f"Rs {whole}", "INR": f"INR {plain}",
        "rupees after": f"{plain} rupees", "Indian grouping": f"₹{_indian_grouping(whole)}",
        "Hindi suffix": f"{plain} रुपये", "Devanagari digits": f"₹{plain}".translate(_DEVANAGARI_DIGITS),
    }
    if not has_paise and whole >= 100_000 and whole % 10_000 == 0:
        out["lakh"] = f"₹{whole / 100_000:g} lakh"
    return out


FRAMES = ["Your EMI is {x}.", "You owe {x} in total.", "I can offer a settlement of {x}.", "The late fee is {x}.", "Please pay {x} by Friday."]
# amounts that are NOT in the conversation / amounts that ARE
INVENTED = [7500.0, 9999.0, 18250.5, 45000.0, 64000.75, 85000.0, 150000.0, 267500.0, 333.0, 1200000.0]  # none within the check's 1-rupee rounding tolerance of an established figure
ESTABLISHED = [12500.0, 175000.0, 13000.0, 500.0, 8000.0, 250000.0]
INVENTED_LINKS = [
    "https://pay.example.com/abc123", "http://bit.ly/3xYzQ", "https://chat.example.test/pay/ZZZ999", "https://payments.example.com/loan/4471",
    "[Pay here](https://secure-pay.example.org/verma)", "<https://upi.example.net/pay?id=88>", "https://chat.example.test/pay/AbC123xyz0",
]
REAL_LINK_FORMS = [f"{LINK}", f"[Pay now]({LINK})", f"<{LINK}>", f"Use this link: {LINK}.", f"({LINK})", f"**{LINK}**"]


def _reply(frame: str, text: str) -> str:
    return frame.format(x=text)


def build_cases() -> dict[str, list[dict]]:
    fabricated, grounded = [], []
    for amount, frame in itertools.product(INVENTED, FRAMES):
        for fmt, text in _formats(amount).items():
            fabricated.append({"kind": "amount", "format": fmt, "reply": _reply(frame, text)})
    for link in INVENTED_LINKS:
        fabricated.append({"kind": "link", "format": "invented link", "reply": f"You can pay securely here: {link}"})
    for amount, frame in itertools.product(ESTABLISHED, FRAMES):
        for fmt, text in _formats(amount).items():
            grounded.append({"kind": "amount", "format": fmt, "reply": _reply(frame, text)})
    for form in REAL_LINK_FORMS:
        grounded.append({"kind": "link", "format": "real link", "reply": f"Here is your payment link {form}"})
    # the roundings people really use, and replies with nothing to ground
    for text in ("about ₹12,500", "₹12,500.40", "₹175,000.00", "roughly ₹13,000"):
        grounded.append({"kind": "amount", "format": "rounded", "reply": f"That comes to {text}."})
    for reply in ("Your payment has been noted. Is there anything else?", "I understand. Let me check that for you.", "The grace period is 3 days."):
        grounded.append({"kind": "no figure", "format": "no figure", "reply": reply})
    return {"fabricated": fabricated, "grounded": grounded}


# A correct figure the model derived itself: not in any tool result, so a literal check blocks it. Known, accepted trade-off
# (grounding.py's module docstring); reported separately because it is a design choice, not a failure to detect.
DERIVED = ["You would save ₹8,750 with a 5% discount.", "Over 14 months that is ₹182,000.", "After the discount you pay ₹166,250.",
           "That is ₹2,500 less than your EMI.", "Two EMIs together are ₹25,000."]


def run() -> dict:
    from businessflow.guardrail import grounding

    cases = build_cases()
    flagged = lambda reply: bool(grounding.check_grounding(reply, CONVERSATION))

    def rate(k, n):
        p, lo, hi = wilson_ci(k, n)
        return {"k": k, "n": n, "rate": round(p, 3), "lo": round(lo, 3), "hi": round(hi, 3)}

    fab = [(c, flagged(c["reply"])) for c in cases["fabricated"]]
    gro = [(c, flagged(c["reply"])) for c in cases["grounded"]]

    def by_format(rows, want_flagged):
        out = {}
        for fmt in sorted({c["format"] for c, _ in rows}):
            sub = [f for c, f in rows if c["format"] == fmt]
            out[fmt] = f"{sum(f == want_flagged for f in sub)}/{len(sub)}"
        return out

    missed = [c["reply"] for c, f in fab if not f]
    wrongly_blocked = [c["reply"] for c, f in gro if f]
    return {
        "date": datetime.now().date().isoformat(),
        "fabricated_caught": rate(sum(f for _, f in fab), len(fab)),
        "fabricated_caught_by_format": by_format(fab, True),
        "grounded_wrongly_blocked": rate(sum(f for _, f in gro), len(gro)),
        "grounded_by_format_passed": by_format(gro, False),
        "derived_but_correct_blocked": rate(sum(flagged(r) for r in DERIVED), len(DERIVED)),
        "missed_examples": missed[:12],
        "wrongly_blocked_examples": wrongly_blocked[:12],
    }


def markdown(summary: dict) -> str:
    fmt = lambda x: f"{x['k']}/{x['n']} ({x['rate']:.1%}, 95% CI {x['lo']:.1%}-{x['hi']:.1%})"
    lines = ["# Guardrail benchmark: invented figures and links", "",
             "Read the header of `eval/guardrail_benchmark.py` for how the cases are built and what they cannot show.", "",
             f"- Fabricated amounts and links caught: **{fmt(summary['fabricated_caught'])}**",
             f"- Grounded amounts and links wrongly blocked: **{fmt(summary['grounded_wrongly_blocked'])}**",
             f"- Correct figures the model derived itself (blocked by design; the check compares values, it does not do arithmetic): {fmt(summary['derived_but_correct_blocked'])}", "",
             "Fabricated caught, by how the amount was written:", "", "| format | caught |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["fabricated_caught_by_format"].items()]
    lines += ["", "Grounded passed, by how the amount was written:", "", "| format | passed |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["grounded_by_format_passed"].items()]
    if summary["missed_examples"]:
        lines += ["", "Fabricated replies that got through:", ""] + [f"- {r}" for r in summary["missed_examples"]]
    if summary["wrongly_blocked_examples"]:
        lines += ["", "Grounded replies that were blocked:", ""] + [f"- {r}" for r in summary["wrongly_blocked_examples"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    summary = run()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "guardrail_benchmark.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(summary)
    (RESULTS / "guardrail_benchmark.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
