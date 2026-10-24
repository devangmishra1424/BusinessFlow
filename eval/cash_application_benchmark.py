"""Cash application: does the bank-statement matcher (accounts/bank_statement.py match_line) put money on the right
borrower, and -- the number that matters -- never on the wrong one?

Labelled synthetic statement lines for every demo borrower (scripts/demo_profiles.py), in the shapes a real collections
account shows: a NACH credit that names the loan, a UPI credit that carries the phone, a borrower's own UTR from a payment
claim, a name and nothing else (truncated, initials, business only), a lookalike name that is NOT a borrower, a phone shared
by two loans, and the bank's own noise. No model, no database: the matcher is pure, so this runs in CI.

Metrics (the report prints them; tests/test_cash_application.py gates them):
- wrong_auto: lines the matcher would post without a person that went to the wrong account, or that belong to nobody /
  to one of several loans on a shared phone. Must be 0: an automatic wrong post is money credited to the wrong borrower.
- strong_recall: of the lines that carry a strong identifier for exactly one loan, how many were posted automatically.
- name_only_top1: of the name-only lines that name one borrower, how often the suggestion (a person confirms it) is right.
- noise_unmatched: bank noise and unknown names that were not matched to anyone.

Run: python -m eval.cash_application_benchmark
"""

from dataclasses import dataclass
from datetime import date

from businessflow.accounts.bank_statement import Candidate, StatementLine, claim_reference_tokens, match_line
from scripts.demo_profiles import LOANS


_TODAY = date(2026, 10, 23)


@dataclass(frozen=True)
class Case:
    kind: str  # nach | upi_phone | claim_ref | name_only | lookalike | shared_phone | noise
    narration: str
    amount: float
    truth: str | None  # the account that really paid; None = nobody (noise, a stranger) or several (a shared phone)
    reference: str = ""


def candidates() -> tuple[list[Candidate], dict[str, str]]:
    """One candidate per demo loan (ids BF-3001...), two of them with an open payment claim carrying a UTR."""
    ids = {loan.ref: f"BF-{3001 + i}" for i, loan in enumerate(LOANS)}
    claims = {LOANS[0].ref: ("412345678901", 11), LOANS[4].ref: ("UTR998877665544", 12)}
    out = []
    for loan in LOANS:
        refs = {claims[loan.ref][0]: claims[loan.ref][1]} if loan.ref in claims else {}
        out.append(Candidate(ids[loan.ref], loan.borrower, loan.business, loan.phone, 14_000.0, refs))
    return out, ids


def cases() -> list[Case]:
    cands, ids = candidates()
    phone_count: dict[str, int] = {}
    for c in cands:
        phone_count[c.phone_number] = phone_count.get(c.phone_number, 0) + 1
    out: list[Case] = []
    for loan, cand in zip(LOANS, cands):
        aid, name, phone10 = cand.account_id, loan.borrower.upper(), loan.phone[3:]
        out.append(Case("nach", f"ACH CR NACH {aid} {name} UMRN BFFN{aid[-4:]}", 14_000.0, aid))
        out.append(Case("nach", f"NACH-CR/{aid.replace('-', '')}/{loan.business.upper()}", 14_000.0, aid))  # "BF3001": spaced/dashless
        if phone_count[cand.phone_number] == 1:
            out.append(Case("upi_phone", f"UPI/{phone10}/{name}/YESB0000001/EMI", 14_000.0, aid))
            out.append(Case("upi_phone", f"IMPS/P2A/{phone10}@ybl/EMI PAYMENT", 9_800.0, aid))
        else:
            out.append(Case("shared_phone", f"UPI/{phone10}/{name}/YESB0000001/EMI", 14_000.0, None))
        first, last = loan.borrower.split()[0].upper(), loan.borrower.split()[-1].upper()
        first_biz = loan.business.split()[0].upper()
        if phone_count[cand.phone_number] == 1:  # a name that is only ever one borrower
            out.append(Case("name_only", f"UPI/{first} {last[0]}/OKAXIS/GPAY1022", 14_000.0, aid))
            out.append(Case("name_only", f"NEFT CR-{loan.business.upper()}", 6_000.0, aid))
            out.append(Case("name_only", f"UPI/{first_biz}/PAYTM/EMI", 14_000.0, aid))
    for ref, token in (("BFF/WC/2025/0418", "412345678901"), ("BFF/BL/2025/0291", "UTR998877665544")):
        out.append(Case("claim_ref", f"IMPS/P2A/{token}/SENDER", 14_000.0, ids[ref], reference=token))
    # A real person, not a borrower, whose name shares one word with a borrower: a suggestion is fine, an auto-post is not.
    for stranger in ("MEENA KUMARI", "SUNITA PATIL", "YADAV TRANSPORT", "IMRAN KHAN", "HARPREET KAUR", "LAKSHMI VILAS STORES",
                     "RAMESH CHANDRA", "GUPTA ELECTRICALS"):
        out.append(Case("lookalike", f"UPI/{stranger}/OKSBI/GPAY0212", 14_000.0, None))
    for narration, amount in (("INT.PD:30-09-2026", 1214.37), ("NEFT CR-SBIN0007321-KAVYA TRADERS-REFUND INV 2231", 4720.0),
                              ("CASH DEPOSIT BRANCH", 20_000.0), ("CHQ DEP 004411 CLG", 14_000.0),
                              ("UPI/9876543210/UNKNOWN PERSON/PAYTM", 14_000.0), ("NEFT CR-ICIC0001-ACME PVT LTD-INV 77", 88_000.0)):
        out.append(Case("noise", narration, amount, None))
    return out


def evaluate() -> dict:
    cands, _ = candidates()
    rows = []
    for n, case in enumerate(cases(), 1):
        line = StatementLine(n, _TODAY, case.amount, case.narration, case.reference)
        rows.append((case, match_line(line, cands)))
    wrong_auto = [(c.narration, m.account_id) for c, m in rows if m.status == "auto" and m.account_id != c.truth]
    strong = [(c, m) for c, m in rows if c.kind in ("nach", "upi_phone", "claim_ref")]
    name_only = [(c, m) for c, m in rows if c.kind == "name_only"]
    noise = [(c, m) for c, m in rows if c.kind == "noise"]
    return {
        "lines": len(rows),
        "wrong_auto": wrong_auto,
        "strong_recall": sum(m.status == "auto" and m.account_id == c.truth for c, m in strong) / len(strong),
        "strong_missed": [c.narration for c, m in strong if not (m.status == "auto" and m.account_id == c.truth)],
        "name_only_never_auto": all(m.status != "auto" for c, m in name_only),
        "name_only_top1": sum(m.account_id == c.truth for c, m in name_only) / len(name_only),
        "name_only_wrong": [(c.narration, m.account_id) for c, m in name_only if m.account_id != c.truth],
        "noise_unmatched": sum(m.status == "unmatched" for c, m in noise) / len(noise),
        "noise_matched": [(c.narration, m.account_id) for c, m in noise if m.status != "unmatched"],
        "lookalike_suggested": sum(m.status == "suggested" for c, m in rows if c.kind == "lookalike"),
        "lookalike_total": sum(c.kind == "lookalike" for c, m in rows),
    }


if __name__ == "__main__":
    r = evaluate()
    print(f"{r['lines']} labelled lines")
    print(f"wrong automatic posts: {len(r['wrong_auto'])}  {r['wrong_auto']}")
    print(f"strong-identifier recall: {r['strong_recall']:.0%}  missed {r['strong_missed']}")
    print(f"name-only never posted alone: {r['name_only_never_auto']}; suggestion right: {r['name_only_top1']:.0%} {r['name_only_wrong']}")
    print(f"noise left unmatched: {r['noise_unmatched']:.0%}  {r['noise_matched']}")
    print(f"lookalike strangers shown as a suggestion (a person says no): {r['lookalike_suggested']}/{r['lookalike_total']}")
