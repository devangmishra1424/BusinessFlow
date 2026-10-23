"""Builds the public demo's borrowers the way a lender's data would really get there: through the product's own front
doors, over HTTP, dated relative to the day it runs.

Nothing here writes a payment, a promise, a fee or a case into the database. It:

1. imports the lender's loan book (POST /loan-book) as it stood on the go-live day, 120 days ago, and opens the one
   loan disbursed since with the New account form (POST /accounts), without a rate;
2. uploads each loan's signed agreement and Key Fact Statement (POST /accounts/{id}/documents) -- read, checked and
   indexed by the ingestion pipeline; the new loan's interest rate is extracted from its agreement;
3. uploads the collections account's bank statement month by month (POST /bank-statements): NACH and UPI credits are
   matched and posted on the day the bank received them, which is what makes an EMI on time, late, or missing; the
   few the matcher cannot place are posted or set aside by a person (POST /bank-lines/{id}/post | ignore), as ops
   would that month;
4. records what the collections team did -- promises taken on calls (POST /accounts/{id}/promises, with the day they
   were made), the daily outbound pass (POST /outbound/run), call outcomes, a message to a borrower;
5. and what borrowers did themselves, from their dashboard (demo sign-in, then the Get help buttons): a payment claim,
   a dispute, a fee-waiver request, a closure-certificate request, a reply to our message.

A reset (run again, or the ops dashboard's Reset demo) keeps the demo loans, their ids and their documents: it clears
what happened on them (accounts/store.py clear_account_activity), puts each loan back to its go-live terms, and enters
the history again -- seconds, not the minutes and the model calls that parsing sixteen PDFs again would cost. Those, the
loan terms, and the 'demo_reset' event the ops dashboard shows are the only direct database writes. Run it from the repository root of the machine that
serves the apps (their uploaded files live there):

    python -m scripts.build_demo                      # against http://127.0.0.1:8001 (ops) and :8000 (chat)
    DEMO_OPS_URL=... DEMO_CHAT_URL=... python -m scripts.build_demo

Needs OPS_API_KEY and DATABASE_URL (the same .env the apps use). Takes a few minutes: every document is parsed.
"""

import csv
import hashlib
import io
import logging
import os
import shutil
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from scripts.build_demo_documents import file_stem
from scripts.demo_profiles import LOANS, Loan, add_months, by_ref, due_dates, emi

load_dotenv()

logger = logging.getLogger("build_demo")
_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "data" / "demo" / "documents"
_UPLOADED = _REPO / "data" / "documents"

GO_LIVE_DAYS_AGO = 120
STATEMENT_PREFIX = "HDFC-Collections-4417-"
OPS = "Priya Nair (Collections)"
_TIMEOUT = httpx.Timeout(30.0, read=300.0)  # a document upload parses a PDF: minutes on a small VM, never unbounded

MEENA, RAMESH_TL, RAMESH_EQ, SUNITA, IMRAN, LAKSHMI, HARPREET, RAMESH_WC = (
    "BFF/WC/2025/0418", "BFF/TL/2025/0137", "BFF/EQ/2026/0912", "BFF/WC/2025/0377",
    "BFF/BL/2025/0291", "BFF/WC/2025/0503", "BFF/CV/2025/0066", "BFF/WC/2025/0352",
)


class DemoBuildError(RuntimeError):
    pass


class Front:
    """The two apps' HTTP front doors. Every call is checked; a refusal stops the build with what was refused."""

    def __init__(self, ops_url: str, chat_url: str, ops_key: str):
        self.ops = httpx.Client(base_url=ops_url, headers={"X-API-Key": ops_key}, timeout=_TIMEOUT)
        self.chat = httpx.Client(base_url=chat_url, timeout=_TIMEOUT)
        self._conversations: dict[str, str] = {}

    @staticmethod
    def _ok(response: httpx.Response, what: str):
        if response.status_code >= 400:
            raise DemoBuildError(f"{what}: {response.status_code} {response.text[:400]}")
        return response.json() if response.content else None

    def op(self, method: str, path: str, what: str, **kwargs):
        return self._ok(self.ops.request(method, path, **kwargs), what)

    def borrower(self, account_id: str, method: str, path: str, what: str, **kwargs):
        """As the borrower, from their own dashboard (the demo's one-click sign-in)."""
        if account_id not in self._conversations:
            started = self._ok(self.chat.post("/conversations", json={"account_id": account_id, "language": "en",
                                                                       "demo_open": True}), f"sign in as {account_id}")
            self._conversations[account_id] = started["conversation_id"]
        cid = self._conversations[account_id]
        return self._ok(self.chat.request(method, path.format(cid=cid), **kwargs), what)


# --- the story's dates ---------------------------------------------------------------------------------------------

def _dates_between(loan: Loan, start: date, end: date) -> list[date]:
    """The loan's due dates d with start <= d <= end."""
    return [d for d in due_dates(loan) if start <= d <= end]


def _latest_due(loan: Loan, today: date) -> date:
    return max(d for d in due_dates(loan) if d <= today)


def _previous_due(loan: Loan, d: date, back: int = 1) -> date:
    dates = due_dates(loan)
    return dates[dates.index(d) - back]


# --- 1. the loan book ----------------------------------------------------------------------------------------------

def _loan_book_csv(go_live: date) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Loan No", "Borrower Name", "Business Name", "Mobile", "Language", "Product", "Sanctioned Amount",
                "EMI", "Tenure", "Months Remaining", "Next Due Date", "NACH", "Risk", "Interest Rate (% p.a.)"])
    for loan in LOANS:
        if loan.opened_here:
            continue
        dates = due_dates(loan)
        paid = [d for d in dates if d < go_live]
        nxt = next(d for d in dates if d >= go_live)
        w.writerow([loan.ref, loan.borrower, loan.business, loan.phone[3:], {"hi": "Hindi", "en": "English",
                    "hinglish": "Hinglish"}[loan.language], loan.product, f"{loan.principal:,.0f}", f"{emi(loan):.0f}",
                    loan.tenure, loan.tenure - len(paid), nxt.strftime("%d/%m/%Y"), "Yes", loan.risk.title(),
                    f"{loan.rate_pct:.2f}"])
    return out.getvalue()


def open_loans(front: Front, go_live: date) -> dict[str, str]:
    """Imports the loan book as of go-live, checked first (dry run) the way ops would; opens the loan disbursed since
    with the New account form. Returns {loan ref: account id}."""
    book = _loan_book_csv(go_live)
    files = {"file": (f"loan-book-{go_live.isoformat()}.csv", book, "text/csv")}
    check = front.op("POST", "/loan-book", "check the loan book", files=files, data={"dry_run": "true"})
    if check["with_errors"]:
        raise DemoBuildError(f"the loan book has errors: {[d for d in check['details'] if d['errors']]}")
    done = front.op("POST", "/loan-book", "import the loan book", files=files, data={"dry_run": "false"})
    ids = {d["external_ref"]: d["account_id"] for d in done["details"]}
    for loan in LOANS:
        if not loan.opened_here:
            continue
        created = front.op("POST", "/accounts", f"open {loan.ref}", json={
            "borrower_name": loan.borrower, "business_name": loan.business, "phone_number": loan.phone,
            "language_preference": loan.language, "loan_type": loan.product, "principal_amount": loan.principal,
            "emi_amount": emi(loan), "tenure_months": loan.tenure, "emi_due_date": loan.first_emi.isoformat(),
            "nach_mandate_active": True, "risk_tier": loan.risk, "external_ref": loan.ref,
        })
        ids[loan.ref] = created["account"]["account_id"]
    logger.info("opened %d loans: %s", len(ids), ids)
    return ids


# --- 2. documents ----------------------------------------------------------------------------------------------------

def upload_documents(front: Front, ids: dict[str, str]) -> None:
    """Each loan's agreement and KFS, unless the account already has them (a reset keeps documents)."""
    for loan in LOANS:
        account_id = ids[loan.ref]
        have = {d["filename"] for d in front.op("GET", f"/accounts/{account_id}/documents", f"documents of {account_id}")}
        for kind, document_type in (("Loan Agreement", "loan_agreement"), ("Key Fact Statement", "kfs")):
            path = _DOCS / f"{file_stem(loan)} {kind}.pdf"
            if path.name in have:
                continue
            with open(path, "rb") as fh:
                out = front.op("POST", f"/accounts/{account_id}/documents", f"upload {path.name}",
                               files={"file": (path.name, fh, "application/pdf")}, data={"document_type": document_type})
            logger.info("%s: %s -> %d chunks%s", account_id, path.name, out["chunks_stored"],
                        ", rate extracted" if out.get("interest_rate_extracted") else "")


# --- 3. the bank statements ------------------------------------------------------------------------------------------

def _utr(d: date, n: int) -> str:
    """A 12-digit bank reference, the same on every build for the same line."""
    digest = hashlib.sha256(f"{d.isoformat()}|{n}".encode()).hexdigest()
    return f"{d.strftime('%y%j')}{int(digest[:12], 16) % 10**7:07d}"


def _credits(ids: dict[str, str], go_live: date, today: date) -> list[dict]:
    """Every credit that reached the collections account since go-live, from each borrower's story. kind: nach (names
    the loan account, posts itself), upi_phone (names their phone, posts itself unless the phone has two loans),
    upi_name (only a name: waits for a person)."""
    out: list[dict] = []

    def nach(ref: str, d: date, amount: float | None = None, note: str = ""):
        loan = by_ref(ref)
        out.append({"date": d, "amount": amount or emi(loan), "ref": ref, "kind": "nach",
                    "narration": f"ACH CR NACH {ids[ref]} {loan.borrower.upper()} UMRN BFFN{ids[ref][-4:]}{loan.first_emi:%y%m}{note}"})

    def upi_phone(ref: str, d: date, amount: float):
        loan = by_ref(ref)
        out.append({"date": d, "amount": amount, "ref": ref, "kind": "upi_phone",
                    "narration": f"UPI/{loan.phone[3:]}/{loan.borrower.upper()}/YESB0000001/EMI {loan.ref}"})

    def upi_name(ref: str, d: date, amount: float, short: str):
        out.append({"date": d, "amount": amount, "ref": ref, "kind": "upi_name",
                    "narration": f"UPI/{short}/OKAXIS/GPAY{d:%d%m}"})

    def history(ref: str, start: date, end: date, skip=()):
        for d in _dates_between(by_ref(ref), start, end):
            if d < today and d not in skip:
                nach(ref, d)

    # Meena: the auto-debit, on the day, every month.
    history(MEENA, go_live, today)
    # Ramesh's term loan: on the day, except one month NACH bounced and he paid by UPI two days later (in grace).
    ramesh = _dates_between(by_ref(RAMESH_TL), go_live, today)
    history(RAMESH_TL, go_live, today, skip=(ramesh[1],))
    upi_phone(RAMESH_TL, ramesh[1] + timedelta(days=2), emi(by_ref(RAMESH_TL)))
    # Ramesh's equipment loan, opened with the New account form: its first EMIs, on the day.
    history(RAMESH_EQ, by_ref(RAMESH_EQ).first_emi, today)
    # Sunita: last month her NACH failed (she had closed that bank account); she paid the EMI by UPI on day 6 -- late,
    # so a fee -- and this month's EMI is still unpaid.
    sunita = by_ref(SUNITA)
    last = _latest_due(sunita, today)
    prev = _previous_due(sunita, last)
    history(SUNITA, go_live, prev - timedelta(days=1))
    upi_phone(SUNITA, prev + timedelta(days=6), emi(sunita))
    # Imran: two months ago paid on day 9, EMI and fee together; nothing since.
    imran = by_ref(IMRAN)
    p1 = _previous_due(imran, _latest_due(imran, today))
    p2 = _previous_due(imran, p1)
    history(IMRAN, go_live, p2 - timedelta(days=1))
    upi_phone(IMRAN, p2 + timedelta(days=9), emi(imran) + 500)
    # Lakshmi: NACH failed on this month's due date (a bank-side error); she paid by GPay the next day. The credit
    # carries only a shortened name, so it waits for a person.
    lakshmi = by_ref(LAKSHMI)
    l_last = _latest_due(lakshmi, today)
    history(LAKSHMI, go_live, l_last - timedelta(days=1))
    upi_name(LAKSHMI, min(l_last + timedelta(days=1), today), emi(lakshmi), "LAKSHMI N")
    # Harpreet: last month the first NACH debit failed at the bank; it went through on re-presentation on day 5 (EMI
    # only, so the ₹500 fee stays owed); this month's went through on the day.
    harpreet = by_ref(HARPREET)
    h_last = _latest_due(harpreet, today)
    h_prev = _previous_due(harpreet, h_last)
    history(HARPREET, go_live, h_last, skip=(h_prev,))
    nach(HARPREET, h_prev + timedelta(days=5), note=" RE-PRESENTED")
    # Ramesh's festival working-capital loan: every EMI on the day, to the last one -- repaid.
    history(RAMESH_WC, go_live, today)
    return sorted(out, key=lambda c: (c["date"], c["ref"]))


_NOISE = (  # credits on a collections account that are not loan EMIs, and the debits around them
    ("INT.PD:{prev_end:%d-%m-%Y}", 1214.37, "credit", "Interest from the bank on the account balance"),
    ("NEFT CR-SBIN0007321-KAVYA TRADERS-REFUND INV 2231", 4720.00, "credit", "A supplier's refund, not a borrower"),
    ("NEFT DR-HDFC0000060-BFF PAYROLL {d:%b} -SALARIES", 186500.00, "debit", ""),
)


def _statement_csv(month_start: date, month_end: date, credits: list[dict]) -> tuple[str, list[str]]:
    """A month of the collections account, in the bank's export layout. Returns (csv, the noise narrations)."""
    rows = []
    noise = []
    for c in credits:
        rows.append((c["date"], c["narration"], _utr(c["date"], len(rows)), 0.0, c["amount"]))
    prev_end = month_start - timedelta(days=1)
    for template, amount, side, _why in _NOISE:
        d = month_start + timedelta(days=(9 if side == "debit" else 2 if "INT" in template else 16))
        if d > month_end:
            continue
        narration = template.format(prev_end=prev_end, d=d)
        rows.append((d, narration, _utr(d, len(rows) + 50), amount if side == "debit" else 0.0,
                     amount if side == "credit" else 0.0))
        if side == "credit":
            noise.append(narration)
    rows.sort(key=lambda r: r[0])
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["HDFC BANK LTD", "", "", "", "", "", ""])
    w.writerow([f"Account: BusinessFlow Finance - Collections A/c ending 4417", "", "", "", "", "", ""])
    w.writerow([f"Statement from {month_start:%d/%m/%Y} to {month_end:%d/%m/%Y}", "", "", "", "", "", ""])
    w.writerow(["Date", "Narration", "Chq./Ref.No.", "Value Dt", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"])
    balance = 2_450_000.0
    for d, narration, ref, dr, cr in rows:
        balance += cr - dr
        w.writerow([d.strftime("%d/%m/%y"), narration, ref, d.strftime("%d/%m/%y"), f"{dr:.2f}" if dr else "",
                    f"{cr:.2f}" if cr else "", f"{balance:.2f}"])
    return out.getvalue(), noise


def upload_statements(front: Front, ids: dict[str, str], go_live: date, today: date) -> None:
    """Month by month, oldest first, as ops would have uploaded them. After each month, the credits that waited are
    decided the way that month's ops person decided them -- except in the current month, where they still wait."""
    credits = _credits(ids, go_live, today)
    month = go_live.replace(day=1)
    by_id = {v: k for k, v in ids.items()}
    while month <= today:
        end = min(add_months(month, 1) - timedelta(days=1), today)
        current = end == today
        in_month = [c for c in credits if month <= c["date"] <= end]
        text, _ = _statement_csv(max(month, go_live) if month < go_live else month, end, in_month)
        name = f"{STATEMENT_PREFIX}{month:%Y-%m}.csv"
        result = front.op("POST", "/bank-statements", f"upload {name}", files={"file": (name, text, "text/csv")})
        logger.info("%s: %d posted, %d with a guess, %d unmatched", name, result["posted"], result["suggested"], result["unmatched"])
        if not current:
            for line in result["lines"]:
                if line["status"] in ("posted", "ignored"):
                    continue
                owner = next((c for c in in_month if c["narration"] == line["narration"]), None)
                if owner is not None and owner["kind"] == "upi_name":
                    continue  # the story: a credit with only a name, still waiting for ops when the demo opens
                if owner is not None:  # a borrower's EMI the matcher could not place alone (a phone with two loans)
                    front.op("POST", f"/bank-lines/{line['id']}/post", f"post {line['narration']}",
                             json={"account_id": ids[owner["ref"]], "decided_by": OPS})
                else:
                    why = next(w for t, _a, _s, w in _NOISE if line["narration"].startswith(t.split("{")[0][:12]))
                    front.op("POST", f"/bank-lines/{line['id']}/ignore", f"set aside {line['narration']}",
                             json={"decided_by": OPS, "note": why})
        if by_id and month <= _previous_due(by_ref(SUNITA), _latest_due(by_ref(SUNITA), today)) <= end:
            # The month her NACH failed for good: the bank's return said the account was closed.
            front.op("PATCH", f"/accounts/{ids[SUNITA]}", "mark Sunita's NACH mandate not active",
                     json={"nach_mandate_active": False})
        month = add_months(month, 1)


# --- 4 and 5. what the team and the borrowers did ---------------------------------------------------------------------

def team_and_borrowers(front: Front, ids: dict[str, str], today: date) -> None:
    lakshmi = by_ref(LAKSHMI)
    l_last = _latest_due(lakshmi, today)
    paid_on = min(l_last + timedelta(days=1), today)
    front.borrower(ids[LAKSHMI], "POST", "/conversations/{cid}/quick-actions/payment-claim", "Lakshmi's payment claim",
                   json={"details": f"I paid ₹{emi(lakshmi):,.0f} by GPay on {paid_on:%d %b} because the auto-debit "
                                    f"failed. I don't have the UPI reference with me right now."})

    harpreet = by_ref(HARPREET)
    h_prev = _previous_due(harpreet, _latest_due(harpreet, today))
    front.borrower(ids[HARPREET], "POST", "/conversations/{cid}/quick-actions/dispute", "Harpreet's dispute",
                   json={"reason": f"Late fee of ₹500 for the {h_prev:%d %b} EMI is wrong. HDFC told me the first NACH "
                                   f"debit failed from their side (technical reason) -- the balance was there. They "
                                   f"debited it again on {h_prev + timedelta(days=5):%d %b}. Please remove the fee."})

    sunita = by_ref(SUNITA)
    s_prev = _previous_due(sunita, _latest_due(sunita, today))
    front.borrower(ids[SUNITA], "POST", "/conversations/{cid}/quick-actions/fee-waiver", "Sunita's fee-waiver request",
                   json={"reason": f"My NACH failed in {s_prev:%B} because I shifted my account from Bank of Baroda to "
                                   "SBI. I paid by UPI as soon as I came to know. First time I was late in a year."})
    front.op("POST", f"/accounts/{ids[SUNITA]}/call-log", "log the call with Sunita", json={
        "outcome": "reached",
        "note": "Explained the auto-debit fails because her Bank of Baroda account is closed; a new mandate is needed "
                "on her SBI account. She will pay this month's EMI by UPI.",
    })
    front.op("POST", f"/accounts/{ids[SUNITA]}/promises", "Sunita's promise", json={
        "promised_date": (today + timedelta(days=4)).isoformat(), "promised_amount": emi(sunita),
        "made_on": today.isoformat(),
    })

    front.borrower(ids[RAMESH_WC], "POST", "/conversations/{cid}/quick-actions/closure-certificate",
                   "Ramesh's closure certificate request for the repaid loan")

    # Imran: two promises taken on calls, each broken; the daily pass judges each and, at the second, hands him to a
    # person. Then today's call, our message, and his answer.
    imran = by_ref(IMRAN)
    p1 = _previous_due(imran, _latest_due(imran, today))
    demo_ids = sorted(ids.values())
    for made, due, amount in ((p1 + timedelta(days=5), p1 + timedelta(days=12), emi(imran)),
                              (p1 + timedelta(days=18), p1 + timedelta(days=25), 10_000.0)):
        front.op("POST", f"/accounts/{ids[IMRAN]}/promises", f"Imran's promise of {made}", json={
            "promised_date": due.isoformat(), "promised_amount": amount, "made_on": made.isoformat()})
        front.op("POST", "/outbound/run", "the daily outbound pass", json={"account_ids": demo_ids})
    front.op("POST", f"/accounts/{ids[IMRAN]}/call-log", "log today's call to Imran", json={
        "outcome": "no_answer", "note": "Phone switched off at 11:40 and again at 16:15."})
    front.op("POST", f"/accounts/{ids[IMRAN]}/clarification-requests", "message Imran", json={
        "message": f"Imran ji, the EMIs of {p1:%d %b} and {_latest_due(imran, today):%d %b} are unpaid and both "
                   "promises were missed. What amount can you realistically pay this week? Reply here and I will see "
                   "what we can arrange. -- Priya, BusinessFlow Finance"})
    dashboard = front.borrower(ids[IMRAN], "GET", "/conversations/{cid}/dashboard", "Imran's dashboard")
    ours = next(m for m in dashboard["messages"] if m.get("kind") == "message")
    front.borrower(ids[IMRAN], "POST", f"/conversations/{{cid}}/messages-from-us/{ours['id']}/reply", "Imran's reply",
                   json={"message": "Sorry for the delay. Since the flyover work closed our road, walk-in sales are "
                                    "down by half. I can pay ₹8,000 on Saturday and the rest of one EMI after the 20th, "
                                    "when two garages clear their dues."})


# --- the reset ---------------------------------------------------------------------------------------------------------

def _go_live_terms(loan: Loan, go_live: date) -> dict:
    """The loan as the books held it on go-live: what open_loans enters, and what a reset puts back."""
    dates = due_dates(loan)
    if loan.opened_here:
        months_remaining, next_due = loan.tenure, loan.first_emi
    else:
        months_remaining, next_due = loan.tenure - len([d for d in dates if d < go_live]), next(d for d in dates if d >= go_live)
    terms = {
        "borrower_name": loan.borrower, "business_name": loan.business, "phone_number": loan.phone,
        "language_preference": loan.language, "loan_type": loan.product, "principal_amount": loan.principal,
        "emi_amount": emi(loan), "tenure_months": loan.tenure, "months_remaining": months_remaining,
        "emi_due_date": next_due, "pending_emi_credit": 0, "nach_mandate_active": True, "dispute_open": False,
        "risk_tier": loan.risk,
    }
    if not loan.opened_here:
        terms["interest_rate_pct"] = loan.rate_pct  # the loan book's column; the new loan keeps its extracted rate
    return terms


def reset_previous_demo(go_live: date) -> dict[str, str] | None:
    """When every demo loan is already on the books: clears what happened on them and puts each back to its go-live
    terms, keeping ids and documents. Returns {ref: account id}, or None when there is no complete demo to reset (a
    partial one is removed, files too, and built again)."""
    from businessflow.accounts import store

    found = store.accounts_by_external_ref([loan.ref for loan in LOANS])
    if len(found) == len(LOANS):
        removed = store.clear_account_activity(list(found.values()), statement_prefix=STATEMENT_PREFIX)
        for loan in LOANS:
            store.update_account_fields(found[loan.ref], **_go_live_terms(loan, go_live))
        logger.info("reset the demo loans to %s: %s", go_live, removed)
        return found
    if found:
        store.remove_accounts(list(found.values()), statement_prefix=STATEMENT_PREFIX)
        for account_id in found.values():
            shutil.rmtree(_UPLOADED / account_id, ignore_errors=True)  # best effort: a leftover folder holds no live data
        logger.info("removed an incomplete demo: %s", found)
    return None


def build(ops_url: str, chat_url: str, ops_key: str, today: date | None = None) -> dict[str, str]:
    from businessflow.accounts import store

    today = today or datetime.now(timezone.utc).date()  # store.current_date(): the apps' "today" is UTC too
    go_live = today - timedelta(days=GO_LIVE_DAYS_AGO)
    front = Front(ops_url, chat_url, ops_key)
    ids = reset_previous_demo(go_live) or open_loans(front, go_live)
    upload_documents(front, ids)
    upload_statements(front, ids, go_live, today)
    team_and_borrowers(front, ids, today)
    store.log_event(None, "demo_reset", {"accounts": ids})
    logger.info("demo built for %s: %s", today, ids)
    return ids


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    key = os.environ.get("OPS_API_KEY")
    if not key:
        raise SystemExit("OPS_API_KEY is not set")
    ops_url = os.environ.get("DEMO_OPS_URL", "http://127.0.0.1:8001")
    chat_url = os.environ.get("DEMO_CHAT_URL", "http://127.0.0.1:8000")
    try:
        ids = build(ops_url, chat_url, key)
    except (DemoBuildError, httpx.HTTPError) as e:
        logger.error("the demo build stopped: %s", e)
        sys.exit(1)
    print({ref: ids[ref] for ref in ids})


if __name__ == "__main__":
    main()
