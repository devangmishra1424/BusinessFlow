"""Reading a loan book: a lender's existing loans, from a CSV export of their loan system, onto the books here.

A real lender does not open its loans one form at a time. This reads the export, checks every row the way the New
account form checks one (tools/validation.py's name and phone rules; positive amounts; a tenure that adds up), and
says, row by row, what is wrong. Nothing is written until every row is right -- a loan book half-imported is worse
than one not imported, because nobody knows which half.

Columns are matched by name, case and spacing ignored, with the usual export names accepted (see _COLUMNS). A loan
already imported (same loan reference) is skipped on a second import, never opened twice.

Pure apart from tools/validation.py's checks: accounts/store.py writes.
"""

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from businessflow.tools import validation

MAX_ROWS = 1000
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%d %b %Y", "%d/%m/%y")
_COLUMNS = {
    "external_ref": ("loan ref", "loan reference", "loan id", "loan no", "loan number", "loan account", "reference", "ref"),
    "borrower_name": ("borrower name", "borrower", "customer name", "name", "applicant"),
    "business_name": ("business name", "business", "firm", "firm name", "shop name", "entity"),
    "phone_number": ("phone", "phone number", "mobile", "mobile number", "contact"),
    "language_preference": ("language", "preferred language", "language preference"),
    "loan_type": ("loan type", "product", "scheme"),
    "principal_amount": ("principal", "principal amount", "sanctioned amount", "loan amount", "disbursed amount"),
    "emi_amount": ("emi", "emi amount", "instalment", "installment"),
    "tenure_months": ("tenure", "tenure months", "tenure (months)", "term months"),
    "months_remaining": ("months remaining", "remaining emis", "remaining tenure", "balance tenure", "emis left"),
    "emi_due_date": ("next due date", "next emi date", "due date", "emi due date"),
    "nach_mandate_active": ("nach", "nach active", "nach mandate", "auto debit", "mandate active"),
    "risk_tier": ("risk", "risk tier", "risk rating", "rating", "onboarding risk"),
}
_REQUIRED = ("borrower_name", "business_name", "phone_number", "principal_amount", "emi_amount", "tenure_months", "emi_due_date")
_LANGUAGES = {"en": "en", "english": "en", "hi": "hi", "hindi": "hi", "hinglish": "hinglish"}
_YES = {"yes", "y", "true", "1", "active"}
_NO = {"no", "n", "false", "0", "inactive", ""}


class LoanBookError(ValueError):
    """The file is not a loan book this reader can use at all (no header, a required column missing, too many rows)."""


@dataclass
class LoanRow:
    row: int  # 1-based data row
    values: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _norm(header: str) -> str:
    return re.sub(r"\s+", " ", (header or "").replace("_", " ").replace(".", "").strip().lower())


def _money(raw: str, name: str) -> float:
    value = float(raw.replace(",", "").replace("₹", "").replace("Rs", "").strip())
    if not value > 0:
        raise ValueError(f"{name} must be more than zero")
    return round(value, 2)


def _phone(raw: str) -> str:
    digits = re.sub(r"[^\d+]", "", raw)
    if re.fullmatch(r"\d{10}", digits):
        digits = "+91" + digits
    elif re.fullmatch(r"91\d{10}", digits):
        digits = "+" + digits
    if not re.fullmatch(r"\+[1-9]\d{7,14}", digits):
        raise ValueError(f"phone {raw!r} is not a phone number (10 digits, or +country code)")
    return digits


def _date(raw: str) -> date:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"date {raw!r} is not readable (use YYYY-MM-DD or DD/MM/YYYY)")


def parse_loan_book(text: str) -> list[LoanRow]:
    """Every data row, with its parsed values or what is wrong with it. Raises LoanBookError when the file as a whole
    cannot be used."""
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    if not rows:
        raise LoanBookError("the file is empty")
    headers = [_norm(h) for h in rows[0]]
    col = {}
    for key, names in _COLUMNS.items():
        for i, h in enumerate(headers):
            if h in names or h == key.replace("_", " "):
                col[key] = i
                break
    missing = [k for k in _REQUIRED if k not in col]
    if missing:
        raise LoanBookError("missing column(s): " + ", ".join(m.replace("_", " ") for m in missing)
                            + f" -- the header row has: {', '.join(h for h in headers if h)}")
    data = [r for r in rows[1:] if any(c.strip() for c in r)]
    if len(data) > MAX_ROWS:
        raise LoanBookError(f"{len(data)} loans; import at most {MAX_ROWS} at a time")

    out = []
    for n, r in enumerate(data, 1):
        def cell(key):
            i = col.get(key)
            return r[i].strip() if i is not None and i < len(r) else ""

        row = LoanRow(n)
        v = row.values
        checks = (
            ("borrower_name", lambda: validation.short_name(cell("borrower_name"), "borrower name")),
            ("business_name", lambda: validation.short_name(cell("business_name"), "business name")),
            ("loan_type", lambda: validation.short_name(cell("loan_type") or "Working Capital Loan", "loan type")),
            ("phone_number", lambda: _phone(cell("phone_number"))),
            ("principal_amount", lambda: _money(cell("principal_amount"), "principal")),
            ("emi_amount", lambda: _money(cell("emi_amount"), "EMI")),
            ("tenure_months", lambda: int(float(cell("tenure_months")))),
            ("emi_due_date", lambda: _date(cell("emi_due_date"))),
        )
        for key, parse in checks:
            try:
                v[key] = parse()
            except ValueError as e:
                row.errors.append(str(e) if str(e) else f"{key.replace('_', ' ')} is not readable")
        lang = cell("language_preference").lower()
        if lang and lang not in _LANGUAGES:
            row.errors.append(f"language {lang!r} is not en, hi or hinglish")
        v["language_preference"] = _LANGUAGES.get(lang, "en")
        nach = cell("nach_mandate_active").lower()
        if nach not in _YES | _NO:
            row.errors.append(f"NACH {nach!r} is not yes or no")
        v["nach_mandate_active"] = nach in _YES or (nach == "" and "nach_mandate_active" not in col)
        risk = cell("risk_tier").lower() or "low"
        if risk not in ("low", "medium", "high"):
            row.errors.append(f"risk {risk!r} is not low, medium or high")
        v["risk_tier"] = risk if risk in ("low", "medium", "high") else "low"
        ref = cell("external_ref")
        v["external_ref"] = ref[:60] or None
        tenure = v.get("tenure_months")
        if isinstance(tenure, int):
            if not 1 <= tenure <= 360:
                row.errors.append(f"tenure {tenure} months is not between 1 and 360")
            remaining = cell("months_remaining")
            try:
                v["months_remaining"] = int(float(remaining)) if remaining else tenure
                if not 1 <= v["months_remaining"] <= tenure:
                    row.errors.append(f"months remaining {v['months_remaining']} must be between 1 and the tenure ({tenure})")
            except ValueError:
                row.errors.append(f"months remaining {remaining!r} is not a number")
        if "emi_amount" in v and "principal_amount" in v and isinstance(tenure, int) and v["emi_amount"] * tenure < v["principal_amount"] * 0.9:
            row.errors.append(f"EMI ₹{v['emi_amount']:,.0f} × {tenure} months does not even repay the principal ₹{v['principal_amount']:,.0f}")
        out.append(row)

    refs = [r.values.get("external_ref") for r in out if r.values.get("external_ref")]
    for r in out:
        ref = r.values.get("external_ref")
        if ref and refs.count(ref) > 1:
            r.errors.append(f"loan reference {ref!r} appears more than once in the file")
    return out
