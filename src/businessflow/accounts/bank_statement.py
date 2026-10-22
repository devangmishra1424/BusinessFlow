"""Reading a bank statement and matching each credit to a borrower (cash application).

The lender's bank statement is where money actually arrives: a UPI or NEFT credit with a narration and a reference,
not always the borrower's own words. Each credit line is matched to an account by what it carries, strongest first:

- the account number in the narration ("BF-1002");
- a reference the borrower already gave us on an open payment claim (a UTR they typed in chat);
- the borrower's phone number (UPI narrations often carry it);
- the borrower's or business's name -- a hint only, names repeat and banks truncate them.

A line with exactly one account found by a strong identifier is posted without anyone touching it (the owner's rule:
make it automatic, don't hand people choices they need not make). Everything else is shown pre-filled with the best
guess and why, for one click. Debits are not ours to apply and are skipped.

Pure: no I/O. accounts/store.py saves lines and posts payments; ops/api.py is the trust boundary that hands this
module the uploaded text.
"""

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime

MAX_ROWS = 2000

_DATE_HEADERS = ("date", "txn date", "transaction date", "value date", "tran date", "posting date")
_NARRATION_HEADERS = ("narration", "description", "particulars", "remarks", "details", "transaction details")
_REFERENCE_HEADERS = ("reference", "ref no", "ref", "utr", "utr no", "chq/ref no", "chq/refno", "chq / ref no", "cheque/ref no",
                      "refno", "reference no")
_CREDIT_HEADERS = ("credit", "deposit", "deposit amt", "deposit amount", "credit amount", "cr", "credit amt")
_DEBIT_HEADERS = ("debit", "withdrawal", "withdrawal amt", "withdrawal amount", "debit amount", "dr", "debit amt")
_AMOUNT_HEADERS = ("amount", "amount (inr)", "txn amount")
_TYPE_HEADERS = ("type", "cr/dr", "dr/cr", "txn type")

_DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y", "%d-%b-%Y", "%d %b %Y", "%d-%b-%y", "%d.%m.%Y")
_ACCOUNT_ID_RE = re.compile(r"\bBF-?\s?(\d{3,6})\b", re.IGNORECASE)
_REFERENCE_TOKEN_RE = re.compile(r"\b[A-Z0-9]{8,22}\b")
_DIGITS_RE = re.compile(r"\d+")
_NAME_STOPWORDS = {"the", "and", "pvt", "ltd", "private", "limited", "co", "company", "enterprises", "traders", "store",
                   "stores", "shop", "mr", "mrs", "ms", "shri", "smt", "upi", "neft", "imps", "rtgs", "to", "from", "by"}


class StatementError(ValueError):
    """The file is not a statement this reader can use at all (no date or amount column, empty, too large)."""


@dataclass(frozen=True)
class StatementLine:
    line_no: int  # 1-based data row in the file
    line_date: date
    amount: float  # a credit, > 0
    narration: str
    reference: str

    @property
    def line_hash(self) -> str:
        """The same bank line, uploaded again (the same statement, or an overlapping one), hashes the same: date,
        amount, reference (or narration when there is none), plus how many identical lines came before it in the
        file -- two genuinely identical credits on one day stay two lines."""
        key = f"{self.line_date.isoformat()}|{self.amount:.2f}|{(self.reference or self.narration).strip().upper()}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class ParsedStatement:
    lines: list[StatementLine]
    debits_skipped: int = 0
    rejected: list[dict] = field(default_factory=list)  # {row, reason}

    def hashes(self) -> list[str]:
        """Each line's hash, numbered within the file so identical lines stay distinct (see StatementLine.line_hash)."""
        seen: dict[str, int] = {}
        out = []
        for line in self.lines:
            h = line.line_hash
            seen[h] = seen.get(h, 0) + 1
            out.append(h if seen[h] == 1 else f"{h}#{seen[h]}")
        return out


def _norm(header: str) -> str:
    """Lower-case, no full stops, single spaces: "Deposit Amt." and "Chq./Ref.No." read as "deposit amt", "chq/refno"."""
    return re.sub(r"\s+", " ", (header or "").replace(".", "").strip().lower())


def _find(headers: list[str], names: tuple[str, ...]) -> int | None:
    for i, h in enumerate(headers):
        if h in names:
            return i
    return None


def _money(raw: str) -> float | None:
    s = (raw or "").strip().replace(",", "").replace("₹", "").replace("INR", "").replace("Rs.", "").replace("Rs", "").strip()
    if not s or s in ("-", "--"):
        return None
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    return float(s)  # raises ValueError, caught per row


def _date(raw: str) -> date:
    s = (raw or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unreadable date {s!r}")


def parse_statement(text: str) -> ParsedStatement:
    """Reads a bank statement CSV export. Finds the header row (the first with a date column and an amount, credit or
    debit column -- banks put a few lines of account details above it), then keeps every credit. A row that cannot be
    read is listed in `rejected` with the reason rather than silently dropped. Raises StatementError when no usable
    header is found or the file has more than MAX_ROWS rows."""
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    if not rows:
        raise StatementError("the file is empty")
    header_at = None
    for i, row in enumerate(rows[:30]):
        headers = [_norm(h) for h in row]
        has_amount = any(h in _CREDIT_HEADERS + _AMOUNT_HEADERS for h in headers)
        if _find(headers, _DATE_HEADERS) is not None and has_amount:
            header_at = i
            break
    if header_at is None:
        raise StatementError("no header row with a date column and a credit or amount column was found in the first 30 rows")
    headers = [_norm(h) for h in rows[header_at]]
    data = rows[header_at + 1:]
    if len(data) > MAX_ROWS:
        raise StatementError(f"the statement has {len(data)} rows; upload at most {MAX_ROWS} at a time")

    col_date = _find(headers, _DATE_HEADERS)
    col_narr = _find(headers, _NARRATION_HEADERS)
    col_ref = _find(headers, _REFERENCE_HEADERS)
    col_cr = _find(headers, _CREDIT_HEADERS)
    col_dr = _find(headers, _DEBIT_HEADERS)
    col_amt = _find(headers, _AMOUNT_HEADERS)
    col_type = _find(headers, _TYPE_HEADERS)

    out = ParsedStatement(lines=[])
    for n, row in enumerate(data, 1):
        if not any(c.strip() for c in row):
            continue

        def cell(i):
            return row[i].strip() if i is not None and i < len(row) else ""

        try:
            if not cell(col_date):
                raise ValueError("no date")
            line_date = _date(cell(col_date))
            if col_cr is not None:
                credit = _money(cell(col_cr))
                if credit is None or credit <= 0:
                    if col_dr is not None and (_money(cell(col_dr)) or 0) > 0:
                        out.debits_skipped += 1
                        continue
                    raise ValueError("no credit amount")
                amount = credit
            else:
                amount = _money(cell(col_amt))
                if amount is None:
                    raise ValueError("no amount")
                kind = cell(col_type).lower()
                if amount < 0 or kind.startswith("d"):
                    out.debits_skipped += 1
                    continue
            if amount > 1e9:
                raise ValueError(f"amount {amount:,.2f} is not plausible")
        except ValueError as e:
            out.rejected.append({"row": n, "reason": str(e)})
            continue
        out.lines.append(StatementLine(n, line_date, round(amount, 2), cell(col_narr)[:500], cell(col_ref)[:100]))
    return out


# --- matching --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """What the matcher knows about one account: who it is, and references the borrower gave on an open claim."""
    account_id: str
    borrower_name: str
    business_name: str
    phone_number: str
    amount_due_now: float
    claim_refs: dict[str, int] = field(default_factory=dict)  # reference token -> dispute_id of the open claim


@dataclass(frozen=True)
class Match:
    status: str  # "auto" (post it), "suggested" (a person confirms), "unmatched"
    account_id: str | None
    reason: str
    dispute_id: int | None = None  # the open payment claim this line settles, when it matched one


def claim_reference_tokens(text: str) -> set[str]:
    """Reference-like tokens a borrower typed on a claim ("UTR 412345678901", "ref ABC123XYZ9"): 8-22 letters/digits
    with at least 6 digits, upper-cased. A plain word or a date never qualifies."""
    return {t for t in _REFERENCE_TOKEN_RE.findall((text or "").upper()) if sum(c.isdigit() for c in t) >= 6}


def _name_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", (name or "").lower()) if t not in _NAME_STOPWORDS}


def _phone_in(text: str, phone: str) -> bool:
    digits = "".join(_DIGITS_RE.findall(phone or ""))[-10:]
    return len(digits) == 10 and digits in "".join(_DIGITS_RE.findall(text))


def match_line(line: StatementLine, candidates: list[Candidate]) -> Match:
    """One line against every account. Strong identifiers (account number, a claim reference, the phone number)
    decide alone when exactly one account has one; two accounts with strong identifiers is a conflict for a person.
    A name is only ever a suggestion, said as such, with the amount noted when it equals what is due."""
    text = f"{line.narration} {line.reference}"
    upper = text.upper()
    ids_in_text = {f"BF-{m}" for m in _ACCOUNT_ID_RE.findall(text)}
    tokens = claim_reference_tokens(text)

    strong: dict[str, tuple[str, int | None]] = {}
    for c in candidates:
        if c.account_id.upper() in ids_in_text:
            strong.setdefault(c.account_id, (f"the narration names account {c.account_id}", None))
        hit = next((t for t in tokens if t in c.claim_refs), None)
        if hit:
            strong[c.account_id] = (f"reference {hit} is the one the borrower gave on their payment claim", c.claim_refs[hit])
        if c.account_id not in strong and _phone_in(upper, c.phone_number):
            strong[c.account_id] = ("the narration carries the borrower's phone number", None)

    if len(strong) == 1:
        account_id, (reason, dispute_id) = next(iter(strong.items()))
        return Match("auto", account_id, reason, dispute_id)
    if len(strong) > 1:
        first = sorted(strong)[0]
        return Match("suggested", first, f"more than one account matches ({', '.join(sorted(strong))}) -- check which one paid")

    words = _name_tokens(text)
    scored = []
    for c in candidates:
        borrower, business = _name_tokens(c.borrower_name), _name_tokens(c.business_name)
        hits = (words & borrower) | (words & business)
        if not hits:
            continue
        full = (borrower and borrower <= words) or (business and business <= words)
        amount_fits = abs(line.amount - c.amount_due_now) < 1
        scored.append((2 * bool(full) + len(hits) + 2 * amount_fits, c, hits, amount_fits))
    if not scored:
        return Match("unmatched", None, "nothing in the narration identifies a borrower")
    scored.sort(key=lambda s: (-s[0], s[1].account_id))
    score, c, hits, amount_fits = scored[0]
    tie = len(scored) > 1 and scored[1][0] == score
    reason = f"the name in the narration ({', '.join(sorted(hits))}) looks like {c.borrower_name} / {c.business_name}"
    if amount_fits:
        reason += f", and the amount equals what they owe now (₹{c.amount_due_now:,.0f})"
    if tie:
        reason += f" -- {scored[1][1].account_id} fits as well, check which one paid"
    return Match("suggested", c.account_id, reason)
