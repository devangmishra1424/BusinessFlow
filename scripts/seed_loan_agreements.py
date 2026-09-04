"""Seeds a real, realistic signed loan agreement (Markdown -- see note
below on why not PDF) for every one of the 9 demo accounts into the RAG
vector store, scoped by account_id, and sets accounts.interest_rate_pct
to match what each document actually states.

Why Markdown, not PDF: docling ingests Markdown natively and
deterministically (no OCR/text-layer-extraction ambiguity at all,
unlike a real PDF) -- this project has no PDF-*writing* library
installed (only pypdfium2, a reader, and Pillow, which can only
produce an image-based PDF requiring OCR, the opposite of what a
"real signed contract" document should be). Four accounts (BF-1005,
1006, 1007, 1009) already had real PDF agreements from an earlier
session; this script explicitly clears their old chunks and removes
the old files so the account has exactly one current document, not a
stale PDF sitting alongside a new Markdown one that accounts.documents
.list_documents_for_account (which lists disk files directly, not the
DB) would otherwise surface as a confusing duplicate.

Deliberately does NOT call rag/extraction.py's extract_loan_terms (that
makes a real Groq call per document) -- the interest rate embedded in
each generated document below is already known at generation time, so
accounts.interest_rate_pct is set directly to match it. Re-running the
real extraction pipeline later (e.g. via the ops dashboard's own
upload flow) against this exact same text would arrive at the same
number, since the phrasing here ("interest ... at the rate of X% per
annum") matches exactly what that pipeline is designed to parse.

All figures for late payment, prepayment discount, and restructuring
below are pulled directly from accounts/policy.py's real enforced
constants (GRACE_PERIOD_DAYS, LATE_FEE_FLAT_AMOUNT,
SETTLEMENT_DISCOUNT_PCT, MAX_RESTRUCTURING_EXTENSION_MONTHS) --
deliberately, so a document a borrower could ask about never states a
different number than what the agent's own tools actually enforce.

Run: python -m scripts.seed_loan_agreements
"""

import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from businessflow.accounts.db import get_connection  # noqa: E402
from businessflow.accounts.policy import (  # noqa: E402
    GRACE_PERIOD_DAYS,
    LATE_FEE_FLAT_AMOUNT,
    MAX_RESTRUCTURING_EXTENSION_MONTHS,
    SETTLEMENT_DISCOUNT_PCT,
)
from businessflow.rag.ingest import ingest_document  # noqa: E402
from businessflow.rag.store import delete_chunks_for_document  # noqa: E402

_DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "data" / "documents"

_LENDER_CITY = "Mumbai"
_LENDER_BLOCK = """**Sahyog Vyapar Finance**
Regd. Office: 402, Turret Business Bay, Andheri-Kurla Road, Mumbai 400059, Maharashtra
CIN: U65999MH2015PTC123456 | NBFC Regn. No.: N-13.09999 (illustrative, for demo purposes)
customercare@sahyogvyapar.example | 1800-XXXXXXX"""

# (account_id, borrower_name, business_name, address, phone, business_description,
#  loan_purpose, security_clause, interest_rate_pct, old_pdf_filename or None,
#  agreement_ref_suffix)
_ACCOUNTS = [
    (
        "BF-1001", "Priya Sharma", "Cotton Threads Boutique",
        "Shop No. 22, Lajpat Nagar Central Market, New Delhi 110024, Delhi",
        "+919812345001",
        "retail trading of readymade garments and textile furnishings",
        "meeting working capital requirements, including inventory purchase and day-to-day operational expenses",
        "unsecured", 14.5, None, "00301",
    ),
    (
        "BF-1002", "Arjun Mehta", "Mehta Hardware & Electricals",
        "Shop No. 7, Karol Bagh Main Market, New Delhi 110005, Delhi",
        "+919812345002",
        "retail and wholesale trading of hardware, electrical fittings, and allied products",
        "purchase of a commercial delivery vehicle and warehouse storage/racking equipment",
        "equipment:one (1) commercial delivery vehicle and warehouse storage/racking equipment",
        13.5, None, "00302",
    ),
    (
        "BF-1003", "Fatima Khan", "Khan Textiles Wholesale",
        "Plot No. 15, Chandni Chowk Textile Market, Delhi 110006, Delhi",
        "+919812345003",
        "wholesale trading of textiles and fabric",
        "working capital requirements, including bulk fabric procurement",
        "unsecured", 18.5, None, "00303",
    ),
    (
        "BF-1004", "Ravi Iyer", "Iyer Auto Spares",
        "Shop No. 3, Kashmere Gate Auto Market, Delhi 110006, Delhi",
        "+919812345004",
        "retail trading of automobile spare parts and accessories",
        "expansion of business premises and inventory into a second retail outlet",
        "unsecured", 13.75, None, "00304",
    ),
    (
        "BF-1005", "Sunita Patil", "Patil Dairy & Kirana Store",
        "Shop No. 14, Tilak Road Market, Kothrud, Pune 411038, Maharashtra",
        "+919845102234",
        "retail sale of dairy products and general kirana/grocery items",
        "working capital requirements, including stock replenishment",
        "unsecured", 15.5, "sunita_patil_loan_agreement.pdf", "00231",
    ),
    (
        "BF-1006", "Rajesh Kumar Yadav", "Yadav Welding & Fabrication Works",
        "Plot No. 45, Sachin Industrial Estate, Surat 394230, Gujarat",
        "+917710234567",
        "welding, metal fabrication, and allied engineering services",
        "purchase of an industrial welding machine and metal-cutting equipment",
        "equipment:one (1) industrial welding machine and metal-cutting equipment",
        13.75, "rajesh_kumar_yadav_loan_agreement.pdf", "00186",
    ),
    (
        "BF-1007", "Meera Nair", "Nair Spices & Provisions",
        "Shop No. 9, Broadway Spice Market, Kochi 682031, Kerala",
        "+919961123456",
        "wholesale and retail trading of spices, condiments, and provisions",
        "expansion of storage and distribution capacity",
        "unsecured", 12.25, "meera_nair_loan_agreement.pdf", "00147",
    ),
    (
        "BF-1008", "Harpreet Singh Sodhi", "Sodhi Auto Care Garage",
        "Plot No. 112, Industrial Area Phase 1, Ludhiana 141003, Punjab",
        "+918146234567",
        "automobile repair, servicing, and maintenance",
        "purchase of a hydraulic vehicle lift and automotive diagnostic equipment",
        "equipment:one (1) hydraulic vehicle lift and automotive diagnostic equipment",
        17.75, None, "00305",
    ),
    (
        "BF-1009", "Anjali Deshmukh", "Deshmukh Boutique & Tailoring",
        "Shop No. 18, Sitabuldi Main Road, Nagpur 440012, Maharashtra",
        "+919689234567",
        "retail tailoring and boutique garment sales",
        "working capital requirements, including fabric and material procurement",
        "unsecured", 14.0, "anjali_deshmukh_loan_agreement.pdf", "00268",
    ),
]


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _security_section(security: str, business_name: str) -> str:
    if security == "unsecured":
        return (
            "This Loan is unsecured and is extended by the Lender solely on the basis of the "
            "Borrower's creditworthiness and business track record, without any collateral security."
        )
    _, _, equipment_desc = security.partition(":")
    return (
        f"As security for due repayment of the Loan, the Borrower hereby hypothecates in favour "
        f"of the Lender the {equipment_desc} purchased using the proceeds of this Loan, until the "
        f"Loan is repaid in full. The Borrower shall keep the hypothecated asset insured and in "
        f"good working condition, and shall not sell, transfer, or further encumber it without the "
        f"Lender's prior written consent."
    )


def _build_document(
    account_id: str, borrower_name: str, business_name: str, address: str, phone: str,
    business_description: str, loan_purpose: str, security: str, rate: float, ref_suffix: str,
    principal: float, emi: float, tenure_months: int, months_remaining: int, loan_type: str,
    emi_due_date: date,
) -> str:
    # Approximate disbursal date: one month before the first EMI would
    # have fallen due, itself back-computed from how many EMIs have
    # already elapsed against the account's current emi_due_date -- kept
    # approximate on purpose (this document's job is stating the real
    # loan terms, not being a precise ledger; get_payment_status/
    # payment_history already own that).
    emis_elapsed = tenure_months - months_remaining
    first_emi_date = emi_due_date - timedelta(days=30 * emis_elapsed)
    agreement_date = first_emi_date - timedelta(days=30)

    principal_str = f"{principal:,.0f}"

    return f"""# Business Loan Agreement

{_LENDER_BLOCK}

---

(Governed by the Indian Contract Act, 1872 and the applicable RBI Master Directions for Non-Banking Financial Companies)

**Agreement Reference No.:** SVF/BL/2026/{ref_suffix}
**Date of Agreement:** {agreement_date.strftime("%d %B %Y")}

This Business Loan Agreement ("**Agreement**") is made and entered into at {_LENDER_CITY} on the date stated above,

**BETWEEN:**

Sahyog Vyapar Finance, a company incorporated under the Companies Act and registered as a Non-Banking Financial Company with the Reserve Bank of India, having its registered office at 402, Turret Business Bay, Andheri-Kurla Road, Mumbai 400059, Maharashtra (hereinafter referred to as the "**Lender**", which expression shall, unless repugnant to the context, include its successors and permitted assigns) of the ONE PART;

**AND:**

{borrower_name}, proprietor of {business_name}, having a place of business at {address}, and contactable at {phone} (hereinafter referred to as the "**Borrower**", which expression shall, unless repugnant to the context, include the Borrower's legal heirs, executors, and permitted assigns) of the OTHER PART.

The Lender and the Borrower are hereinafter collectively referred to as the "**Parties**" and individually as a "**Party**".

## Recitals

WHEREAS the Borrower carries on the business of {business_description} under the name and style of {business_name}, and has applied to the Lender for a {loan_type} for the purpose of {loan_purpose};

WHEREAS the Lender has agreed to extend the said loan to the Borrower on the terms and conditions recorded in this Agreement;

NOW THEREFORE, in consideration of the mutual covenants contained herein, the Parties agree as follows:

## 1. Loan Details

| Particular | Detail |
|---|---|
| Loan Type | {loan_type} |
| Sanctioned Principal Amount | Rs. {principal_str} |
| Tenure | {tenure_months} months |
| Number of Equated Monthly Instalments (EMIs) | {tenure_months} |
| Monthly EMI Amount | Rs. {emi:,.2f} |
| First EMI Due Date | {first_emi_date.strftime("%d %B %Y")} |
| Mode of Repayment | NACH / auto-debit mandate registered with the Borrower's bank |

## 2. Interest

The Loan shall carry interest at the rate of **{rate}% per annum**, calculated on a reducing balance basis on the outstanding principal, compounded monthly. The EMI stated above has been computed on this basis and includes both principal and interest components.

## 3. Security

{_security_section(security, business_name)}

## 4. Late Payment

In the event any EMI is not paid by its due date, the Borrower shall be liable to pay a late payment charge of Rs. {LATE_FEE_FLAT_AMOUNT} (flat) once the delay exceeds a grace period of {GRACE_PERIOD_DAYS} (three) days from the due date, without prejudice to the Lender's other rights under this Agreement.

## 5. Prepayment and Foreclosure

The Borrower may prepay or foreclose the Loan, in part or in full, at any time during the tenure. A one-time settlement of the entire outstanding principal, where offered by the Lender, shall be eligible for a flat discount of {SETTLEMENT_DISCOUNT_PCT * 100:.0f}% (percent) on the outstanding principal, subject to the Lender's then-current settlement policy and the Borrower's account being free of any unresolved dispute.

## 6. Restructuring

Where the Borrower is unable to service the Loan as originally scheduled, the Lender may, at its sole discretion and subject to its internal policy, permit an extension of the repayment tenure by up to {MAX_RESTRUCTURING_EXTENSION_MONTHS} (three) months. Any such restructuring requires the Lender's prior written approval and does not take effect merely upon being proposed or discussed.

## 7. Default and Grievance Redressal

A material or repeated default by the Borrower may result in the account being referred to the Lender's collections and recovery process, including escalation to a human representative. The Borrower may raise a grievance regarding the servicing of this Loan in accordance with the Lender's Grievance Redressal Policy, and, if unresolved, before the RBI Banking/NBFC Ombudsman.

## 8. Governing Law

This Agreement shall be governed by and construed in accordance with the laws of India, and the courts at {_LENDER_CITY} shall have exclusive jurisdiction over any dispute arising herefrom.

IN WITNESS WHEREOF, the Parties have executed this Agreement on the date first mentioned above.

**For Sahyog Vyapar Finance**
Authorised Signatory

**Borrower**
{borrower_name}
"""


def main() -> None:
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is not set -- copy .env.example to .env and fill it in")

    conn = get_connection()
    total_chunks = 0

    for (
        account_id, borrower_name, business_name, address, phone, business_description,
        loan_purpose, security, rate, old_pdf_filename, ref_suffix,
    ) in _ACCOUNTS:
        row = conn.execute(
            "select loan_type, principal_amount, emi_amount, tenure_months, months_remaining, emi_due_date "
            "from accounts where account_id = %s",
            (account_id,),
        ).fetchone()
        if row is None:
            print(f"skip {account_id}: no such account (run scripts.seed_accounts first)")
            continue

        account_dir = _DOCUMENTS_DIR / account_id
        account_dir.mkdir(parents=True, exist_ok=True)

        if old_pdf_filename:
            old_path = account_dir / old_pdf_filename
            delete_chunks_for_document(str(old_path))
            if old_path.exists():
                old_path.unlink()
                print(f"{account_id}: removed old {old_pdf_filename}, cleared its chunks")

        new_path = account_dir / f"{_slug(borrower_name)}_loan_agreement.md"
        content = _build_document(
            account_id, borrower_name, business_name, address, phone, business_description,
            loan_purpose, security, rate, ref_suffix,
            float(row["principal_amount"]), float(row["emi_amount"]), row["tenure_months"],
            row["months_remaining"], row["loan_type"], row["emi_due_date"],
        )
        new_path.write_text(content, encoding="utf-8")

        chunks_stored = ingest_document(str(new_path), document_type="loan_agreement", account_id=account_id)
        conn.execute("update accounts set interest_rate_pct = %s where account_id = %s", (rate, account_id))
        total_chunks += chunks_stored
        print(f"{account_id} ({borrower_name}): {chunks_stored} chunks, interest_rate_pct={rate}")

    print(f"\ndone -- {len(_ACCOUNTS)} accounts, {total_chunks} total new chunks stored")


if __name__ == "__main__":
    main()
