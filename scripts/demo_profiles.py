"""The demo lender's borrowers: six small businesses, eight loans (Ramesh Yadav has three: one repaid, two running), each with the terms of its signed agreement.

Only terms live here -- what was signed, and when. What has happened on each loan since (payments, a bounced NACH, a
promise, a dispute) is not data in this file: scripts/build_demo.py enters it through the product's own front doors,
dated relative to the day it runs, the way a lender's staff and borrowers would have. scripts/build_demo_documents.py
prints each loan's agreement and Key Fact Statement from these same terms, so the documents, the loan book and the
schedule can never disagree.

Phone numbers sit in the 98123-451xx block the test fixtures already use; personal identifiers on the documents (PAN,
Aadhaar, account numbers) are masked, as a lender's own copies show them.
"""

import calendar
import math
from dataclasses import dataclass, field
from datetime import date

LENDER = "BusinessFlow Finance"
COLLECTIONS_ACCOUNT = "HDFC Bank, Fort branch, Mumbai -- collections account ending 4417 (IFSC HDFC0000060)"


@dataclass(frozen=True)
class Loan:
    ref: str  # the lender's loan number (external_ref on the books)
    borrower: str
    business: str
    business_kind: str
    address: str
    phone: str
    language: str  # en | hi | hinglish, as the loan book records it
    product: str
    purpose: str
    principal: float
    rate_pct: float  # fixed, per annum, reducing balance
    tenure: int
    sanctioned_on: date
    disbursed_on: date
    first_emi: date
    nach_bank: str  # where the NACH mandate debits
    risk: str  # onboarding rating
    processing_fee_pct: float = 1.5
    security: str | None = None  # what is hypothecated, if anything
    opened_here: bool = False  # True: disbursed after the loan book was migrated, so opened with the New account form
    notes: tuple[str, ...] = field(default_factory=tuple)  # extra special conditions in the agreement


LOANS: tuple[Loan, ...] = (
    Loan(
        ref="BFF/WC/2025/0418", borrower="Meena Gupta", business="Gupta Saree Kendra",
        business_kind="retail shop selling Banarasi sarees and dress materials",
        address="D-47/112 Ramapura, Luxa Road, Varanasi, Uttar Pradesh 221010", phone="+919812345101", language="hi",
        product="Working Capital Loan", purpose="stock for the wedding season (sarees, dress materials)",
        principal=300_000, rate_pct=18.0, tenure=24, sanctioned_on=date(2025, 10, 28), disbursed_on=date(2025, 11, 3),
        first_emi=date(2025, 12, 5), nach_bank="State Bank of India, Varanasi Cantt branch (savings a/c ending 2291)",
        risk="low",
    ),
    Loan(
        ref="BFF/TL/2025/0137", borrower="Ramesh Yadav", business="Yadav Dairy & Sweets",
        business_kind="dairy and sweet shop with a small production unit",
        address="Shop 3, Kapoorthala Complex, Aliganj, Lucknow, Uttar Pradesh 226024", phone="+919812345102",
        language="hinglish", product="Business Term Loan", purpose="expansion of the sweets production unit (boiler, khoya machine, shop fit-out)",
        principal=500_000, rate_pct=17.0, tenure=36, sanctioned_on=date(2025, 4, 2), disbursed_on=date(2025, 4, 8),
        first_emi=date(2025, 5, 10), nach_bank="Punjab National Bank, Aliganj branch (current a/c ending 7730)",
        risk="low",
    ),
    Loan(
        ref="BFF/EQ/2026/0912", borrower="Ramesh Yadav", business="Yadav Dairy & Sweets",
        business_kind="dairy and sweet shop with a small production unit",
        address="Shop 3, Kapoorthala Complex, Aliganj, Lucknow, Uttar Pradesh 226024", phone="+919812345102",
        language="hinglish", product="Equipment Loan", purpose="a two-door visi-cooler and a refrigerated sweets display counter",
        principal=180_000, rate_pct=16.0, tenure=18, sanctioned_on=date(2026, 8, 14), disbursed_on=date(2026, 8, 20),
        first_emi=date(2026, 9, 15), nach_bank="Punjab National Bank, Aliganj branch (current a/c ending 7730)",
        risk="low", security="the visi-cooler and the sweets display counter bought with this loan (invoice 4471/26 of Shree Cooling Systems, Lucknow)",
        opened_here=True,
    ),
    Loan(
        ref="BFF/WC/2025/0377", borrower="Sunita Sharma", business="Sharma Boutique & Tailoring",
        business_kind="women's boutique with a tailoring unit of four machines",
        address="14, Shanti Path, Tilak Nagar, Jaipur, Rajasthan 302004", phone="+919812345103", language="hinglish",
        product="Working Capital Loan", purpose="fabric stock and two new industrial sewing machines",
        principal=200_000, rate_pct=20.0, tenure=24, sanctioned_on=date(2025, 9, 22), disbursed_on=date(2025, 9, 26),
        first_emi=date(2025, 11, 1), nach_bank="Bank of Baroda, Raja Park branch (savings a/c ending 5106)",
        risk="medium",
    ),
    Loan(
        ref="BFF/BL/2025/0291", borrower="Imran Qureshi", business="Qureshi Auto Spares",
        business_kind="wholesale and retail two-wheeler and car spare parts",
        address="5-2-176, Ranigunj, Secunderabad, Telangana 500003", phone="+919812345104", language="en",
        product="Business Loan", purpose="working capital and a second godown for bulk stock",
        principal=400_000, rate_pct=19.0, tenure=30, sanctioned_on=date(2025, 6, 6), disbursed_on=date(2025, 6, 12),
        first_emi=date(2025, 7, 12), nach_bank="ICICI Bank, Secunderabad branch (current a/c ending 0488)",
        risk="medium",
    ),
    Loan(
        ref="BFF/WC/2025/0503", borrower="Lakshmi Narayanan", business="Sri Lakshmi Provision Stores",
        business_kind="grocery and provision store with home delivery",
        address="22, Bharathi Park Road 4, Saibaba Colony, Coimbatore, Tamil Nadu 641011", phone="+919812345105",
        language="en", product="Working Capital Loan", purpose="bulk purchase of rice, pulses and oil before the festival season",
        principal=250_000, rate_pct=18.0, tenure=24, sanctioned_on=date(2025, 12, 15), disbursed_on=date(2025, 12, 19),
        first_emi=date(2026, 1, 20), nach_bank="Indian Bank, R.S. Puram branch (current a/c ending 3352)",
        risk="low",
    ),
    Loan(
        ref="BFF/CV/2025/0066", borrower="Harpreet Singh", business="Singh Roadlines",
        business_kind="goods transport (two light commercial trucks on Ludhiana-Delhi routes)",
        address="Plot 18, Transport Nagar, Ludhiana, Punjab 141003", phone="+919812345106", language="hinglish",
        product="Commercial Vehicle Loan", purpose="purchase of a 7-tonne light commercial truck",
        principal=850_000, rate_pct=15.5, tenure=36, sanctioned_on=date(2025, 2, 10), disbursed_on=date(2025, 2, 15),
        first_emi=date(2025, 3, 5), nach_bank="HDFC Bank, Ferozepur Road branch (current a/c ending 9064)",
        risk="medium", processing_fee_pct=1.0,
        security="the light commercial truck registration PB 10 HK 4471 (engine no. ending 88213, chassis no. ending 61907), hypothecated to the lender and recorded on the registration certificate",
        notes=("The Borrower shall keep the vehicle comprehensively insured, with the Lender named as loss payee, for the whole tenure.",),
    ),
    Loan(
        ref="BFF/WC/2025/0352", borrower="Ramesh Yadav", business="Yadav Dairy & Sweets",
        business_kind="dairy and sweet shop with a small production unit",
        address="Shop 3, Kapoorthala Complex, Aliganj, Lucknow, Uttar Pradesh 226024", phone="+919812345102",
        language="hinglish", product="Working Capital Loan", purpose="festival-season stock of khoya, dry fruits and gift packaging for Diwali 2025",
        principal=120_000, rate_pct=18.0, tenure=12, sanctioned_on=date(2025, 9, 2), disbursed_on=date(2025, 9, 8),
        first_emi=date(2025, 10, 10), nach_bank="Punjab National Bank, Aliganj branch (current a/c ending 7730)",
        risk="low",
    ),
)


def add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year, month = d.year + month // 12, month % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def emi(loan: Loan) -> float:
    """The EMI on the reducing balance, rounded up to the rupee (the last EMI absorbs the difference)."""
    r = loan.rate_pct / 1200
    raw = loan.principal * r * (1 + r) ** loan.tenure / ((1 + r) ** loan.tenure - 1)
    return float(math.ceil(round(raw, 6)))


def due_dates(loan: Loan) -> list[date]:
    return [add_months(loan.first_emi, k) for k in range(loan.tenure)]


def schedule(loan: Loan) -> list[dict]:
    """The full repayment schedule: per EMI, the date, opening principal, interest, principal repaid, the EMI and the
    closing principal. Interest is the monthly rate on the opening principal; the last EMI clears what is left."""
    r = loan.rate_pct / 1200
    instalment = emi(loan)
    balance = loan.principal
    rows = []
    for n, when in enumerate(due_dates(loan), 1):
        interest = round(balance * r, 2)
        amount = instalment if n < loan.tenure else round(balance + interest, 2)
        principal = round(amount - interest, 2)
        closing = round(balance - principal, 2)
        rows.append({"n": n, "date": when, "opening": round(balance, 2), "interest": interest, "principal": principal,
                     "emi": amount, "closing": max(closing, 0.0)})
        balance = closing
    return rows


def processing_fee(loan: Loan) -> tuple[float, float]:
    """(fee, GST at 18% on it)."""
    fee = round(loan.principal * loan.processing_fee_pct / 100, 2)
    return fee, round(fee * 0.18, 2)


def apr_pct(loan: Loan) -> float:
    """Annual Percentage Rate as the KFS states it: the yearly rate (monthly IRR x 12) at which the EMIs repay the
    amount actually received (principal less the processing fee and its GST)."""
    fee, gst = processing_fee(loan)
    received = loan.principal - fee - gst
    payments = [row["emi"] for row in schedule(loan)]
    lo, hi = 0.0, 0.1
    for _ in range(200):
        mid = (lo + hi) / 2
        pv = sum(p / (1 + mid) ** (k + 1) for k, p in enumerate(payments))
        lo, hi = (mid, hi) if pv > received else (lo, mid)
    return round(mid * 12 * 100, 2)


def by_ref(ref: str) -> Loan:
    return next(loan for loan in LOANS if loan.ref == ref)
