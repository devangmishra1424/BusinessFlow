"""Prints each demo loan's signed Loan Agreement and Key Fact Statement (KFS) to PDF, from the terms in
scripts/demo_profiles.py, into data/demo/documents/. The PDFs are committed: scripts/build_demo.py uploads them through
the ops dashboard's own document upload, where they are read, checked and indexed like any lender's document, and the
agreement's interest rate is extracted from its text.

Run (from the repo root, on a machine with Edge or Chrome): python -m scripts.build_demo_documents
"""

import shutil
import tempfile
from datetime import timedelta
from html import escape
from pathlib import Path

from scripts.build_policy_pdfs import _browser, print_pdf
from scripts.demo_profiles import COLLECTIONS_ACCOUNT, LENDER, LOANS, Loan, apr_pct, due_dates, emi, processing_fee, schedule

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "demo" / "documents"
_CSS = _REPO / "data" / "policies" / "source" / "policy.css"
_GRACE_DAYS = 3  # accounts/policy.py GRACE_PERIOD_DAYS -- stated in the contract the borrower signed
_LATE_FEE = 500  # accounts/policy.py LATE_FEE_FLAT_AMOUNT


def inr(amount: float, paise: bool = False) -> str:
    """Indian digit grouping: ₹3,00,000 / ₹14,977.50."""
    negative = amount < 0
    whole, frac = f"{abs(amount):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    text = f"₹{whole}" + (f".{frac}" if paise else "")
    return f"-{text}" if negative else text


def long_date(d) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def file_stem(loan: Loan) -> str:
    return loan.ref.replace("/", "-")


def _page(title: str, code: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>
<link rel="stylesheet" href="policy.css">
<style>@page {{ @bottom-left {{ content: "{escape(code)}"; font: 8pt "Segoe UI", Arial, sans-serif; color: #555; }} }}
table.sched td, table.sched th {{ padding: 1.5pt 4pt; font-size: 8.4pt; }} table.sched {{ break-inside: auto; }}
table.sched tr {{ break-inside: avoid; }}</style></head><body>{body}</body></html>"""


def _masthead(kind: str, loan: Loan) -> str:
    return f"""<div class="masthead"><div class="lender">{LENDER}<small>Demonstration lender for the BusinessFlow
collections software</small></div><div class="meta">{escape(kind)}<br>Loan No. {escape(loan.ref)}<br>
{escape(loan.borrower)} &middot; {escape(loan.business)}</div></div>"""


def _schedule_table(loan: Loan) -> str:
    rows = "".join(
        f"<tr><td class='num'>{r['n']}</td><td>{r['date'].strftime('%d-%b-%Y')}</td><td class='num'>{inr(r['opening'], True)}</td>"
        f"<td class='num'>{inr(r['interest'], True)}</td><td class='num'>{inr(r['principal'], True)}</td>"
        f"<td class='num'>{inr(r['emi'], True)}</td><td class='num'>{inr(r['closing'], True)}</td></tr>"
        for r in schedule(loan)
    )
    return ("<table class='sched'><tr><th class='num'>EMI no.</th><th>Due date</th><th class='num'>Opening principal</th>"
            "<th class='num'>Interest</th><th class='num'>Principal</th><th class='num'>EMI</th><th class='num'>Closing principal</th></tr>"
            f"{rows}</table>")


def _figures(loan: Loan) -> dict:
    rows = schedule(loan)
    fee, gst = processing_fee(loan)
    return {
        "emi": emi(loan), "last": rows[-1], "interest": round(sum(r["interest"] for r in rows), 2),
        "total": round(sum(r["emi"] for r in rows), 2), "fee": fee, "gst": gst,
        "received": round(loan.principal - fee - gst, 2), "apr": apr_pct(loan), "dates": due_dates(loan),
    }


def agreement_html(loan: Loan) -> str:
    f = _figures(loan)
    city = loan.address.split(",")[-2].strip().split(" ")[0] if "," in loan.address else ""
    secured = loan.security is not None
    vehicle = "truck" in (loan.security or "")
    security_section = ""
    vehicle_clause = (", issue a No Objection Certificate and a signed Form 35 so that the Borrower can have the "
                      "hypothecation entry removed from the registration certificate," if vehicle else ",")
    if secured:
        security_section = f"""<h2>10. Security</h2>
<p>10.1 As security for the Loan, the Borrower hypothecates to the Lender {escape(loan.security)} (the "Asset"), by way of
a first and exclusive charge, until every amount payable under this Agreement has been paid.</p>
<p>10.2 The Borrower shall keep the Asset in good working order at the business premises stated above, shall not sell,
lease, pledge or otherwise part with it without the Lender's written consent, and shall allow the Lender to inspect it
on reasonable notice during business hours.</p>
<p>10.3 Within 30 days of the Loan being fully repaid, the Lender shall release the hypothecation{vehicle_clause} and
return any original document deposited with it, as the Reserve Bank of India's
directions require; for a delay attributable to the Lender beyond 30 days, the Lender compensates the Borrower at
₹5,000 for each day of delay.</p>"""
    notes = "".join(f"<li>{escape(n)}</li>" for n in loan.notes)
    special = f"""<h2>17. Special conditions</h2><ul>{notes or '<li>None.</li>'}</ul>"""
    body = f"""{_masthead("Loan Agreement", loan)}
<h1>Loan Agreement &ndash; {escape(loan.product)}</h1>
<div class="summary"><p><b>Loan of {inr(loan.principal)}</b> at a fixed <b>{loan.rate_pct:.2f}% per annum</b> (reducing
balance), repayable in <b>{loan.tenure} monthly EMIs of {inr(f['emi'])}</b> due on the <b>{loan.first_emi.day}{_ordinal(loan.first_emi.day)} of each
month</b> from {long_date(loan.first_emi)} to {long_date(f['dates'][-1])}. Grace period {_GRACE_DAYS} days; late payment
charge {inr(_LATE_FEE)} flat per EMI after it; no prepayment or foreclosure charge.</p></div>

<p>This Loan Agreement (the "Agreement") is made at {escape(city or 'the Borrower\'s place of business')} on
{long_date(loan.sanctioned_on)} between <b>{LENDER}</b>, a non-banking financial company having its registered office at
Mumbai, Maharashtra (the "Lender"), and <b>{escape(loan.borrower)}</b>, sole proprietor of <b>{escape(loan.business)}</b>, a
{escape(loan.business_kind)}, carrying on business at {escape(loan.address)} (the "Borrower").</p>

<h2>1. Definitions</h2>
<p>In this Agreement: "EMI" means each equated monthly instalment set out in Schedule II; "Due Date" means the date on
which an EMI falls due under Schedule II; "Grace Period" means the {_GRACE_DAYS} calendar days immediately after a Due Date;
"Outstanding Balance" means the EMIs not yet paid together with any late payment charge not yet paid or waived;
"Policies" means the Lender's published policies BFF-POL-01 to BFF-POL-07, as amended from time to time with notice to
the Borrower.</p>

<h2>2. The loan</h2>
<p>2.1 The Lender agrees to lend and the Borrower agrees to borrow {inr(loan.principal)}
({_words(loan.principal)} only) (the "Loan") for {escape(loan.purpose)}, and for no other purpose.</p>
<p>2.2 The Loan was disbursed in one tranche on {long_date(loan.disbursed_on)} to the Borrower's account with
{escape(loan.nach_bank)}, net of the processing fee of {inr(f['fee'], True)} and GST of {inr(f['gst'], True)} on it,
so that {inr(f['received'], True)} was credited to the Borrower.</p>

<h2>3. Interest</h2>
<p>3.1 Interest is charged at the fixed rate of <b>{loan.rate_pct:.2f}% per annum</b> on the reducing balance of the Loan,
computed monthly and payable as part of each EMI. The rate does not change during the tenure.</p>
<p>3.2 The Annual Percentage Rate (APR) of the Loan, including the processing fee, is {f['apr']:.2f}% per annum, as stated in
the Key Fact Statement given to the Borrower before this Agreement was signed. The total interest payable over the tenure
is {inr(f['interest'], True)}.</p>

<h2>4. Repayment</h2>
<p>4.1 The Borrower shall repay the Loan with interest in {loan.tenure} EMIs of {inr(f['emi'])} each, the first on
{long_date(loan.first_emi)} and the last on {long_date(f['dates'][-1])}, on the {loan.first_emi.day}{_ordinal(loan.first_emi.day)} of each month, as set
out in Schedule II. The last EMI is {inr(f['last']['emi'], True)}, as it absorbs the rounding of the earlier EMIs.</p>
<p>4.2 The Borrower has registered a NACH mandate on {escape(loan.nach_bank)} for an amount not less than
the EMI, and shall keep that account funded on every Due Date. A Due Date falling on a bank holiday is presented on the
next working day.</p>
<p>4.3 If a NACH debit fails, the EMI remains payable. The Borrower may pay it through a payment link sent by the Lender
(valid for 72 hours), by UPI, or by NEFT, IMPS or RTGS to the Lender's {escape(COLLECTIONS_ACCOUNT)}, quoting the loan number
{escape(loan.ref)} in the remarks.</p>

<h2>5. Grace period and late payment charge</h2>
<p>5.1 An EMI received within the Grace Period is treated as paid on time for every purpose, and no charge applies.</p>
<p>5.2 An EMI remaining unpaid after the Grace Period attracts a flat late payment charge of {inr(_LATE_FEE)}, levied once for
that EMI. It is not levied per day, is not compounded, is not added to the principal and bears no interest. The Lender
levies no charge for a returned or dishonoured NACH debit; the Borrower's bank may levy its own.</p>
<p>5.3 Only an authorised officer of the Lender may waive a late payment charge, by a recorded decision with the reason,
which is communicated to the Borrower.</p>

<h2>6. Application of payments</h2>
<p>Every payment is applied first to the EMI that is due, then to any late payment charge outstanding, oldest first, and
any balance is credited towards the next EMI, unless the Borrower asks for it to reduce the EMI or the remaining number
of EMIs, as policy BFF-POL-02 provides.</p>

<h2>7. Prepayment and closure</h2>
<p>7.1 The Borrower may prepay all or part of the Outstanding Balance at any time without any prepayment or foreclosure
charge.</p>
<p>7.2 When the Outstanding Balance has been paid in full, the Lender shall, on request, issue a closure (no-dues)
certificate and report the Loan as closed to the credit information companies.</p>

<h2>8. Difficulty in paying</h2>
<p>If the Borrower expects difficulty in paying an EMI, the Borrower should inform the Lender as early as possible. The
Lender will consider, in accordance with policy BFF-POL-04 and subject to approval by its officer: a promise to pay in up
to four parts within 90 days; a part-payment of at least 70% of one EMI; an extension of the tenure by one to three
months, which spreads the remaining EMIs over the longer tenure without additional interest; or a one-time settlement of
the remaining EMIs less 5%, paid in one payment. Any such arrangement binds the Lender only once confirmed in writing.</p>

<h2>9. The Borrower's undertakings</h2>
<ul>
<li>To use the Loan only for the purpose in clause 2.1 and to keep proper books of account of the business.</li>
<li>To inform the Lender within 15 days of any change of address, mobile number, business name or bank account, and to
register a fresh NACH mandate before closing the account on which the existing mandate is registered.</li>
<li>Not to raise any other loan against the assets financed by this Loan without the Lender's consent.</li>
<li>To give true and complete information; the Borrower confirms that all information given in the application is true.</li>
</ul>
{security_section}

<h2>11. Events of default</h2>
<p>Each of the following is an event of default: (a) any EMI or other amount remaining unpaid for 90 days or more after its
Due Date; (b) any information given by the Borrower proving to have been false in a material respect; (c) the sale or
transfer of {'the Asset' if secured else 'the business'} without the Lender's consent; (d) the Borrower closing the business or
becoming insolvent.</p>

<h2>12. Consequences of default</h2>
<p>On an event of default the Lender may, after giving the Borrower 15 days' written notice and an opportunity to be heard,
declare the Outstanding Balance immediately payable{', and enforce its security over the Asset in accordance with law' if secured else ''}.
The Lender reports the repayment record of the Loan to the credit information companies as the law requires. The Lender
does not engage third-party recovery agents for this Loan; all follow-up is by its own staff and its automated
assistant, in accordance with its Fair Practices Code (policy BFF-POL-01).</p>

<h2>13. Communication</h2>
<p>The Lender communicates with the Borrower through its web assistant, through Telegram once the Borrower has linked it,
and by phone, in the Borrower's chosen language ({ {'en': 'English', 'hi': 'Hindi', 'hinglish': 'Hinglish'}[loan.language] }).
The assistant is automated and is identified as such; it cannot waive a charge, change the terms of this Agreement or
approve a restructuring. The Borrower may ask for a person at any time.</p>

<h2>14. Grievances</h2>
<p>The Borrower may raise any grievance through the assistant, Telegram or by phone; it is handled under policy BFF-POL-05,
with escalation to the Grievance Redressal Officer and the Principal Nodal Officer and, after 30 days or an unsatisfactory
reply, to the Reserve Bank &ndash; Integrated Ombudsman Scheme, 2021.</p>

<h2>15. Assignment</h2>
<p>The Lender may assign or securitise the Loan in accordance with the directions of the Reserve Bank of India, with
notice to the Borrower; the Borrower may not assign this Agreement.</p>

<h2>16. Governing law</h2>
<p>This Agreement is governed by the laws of India. The courts at Mumbai and at {escape(city or 'the Borrower\'s place of business')}
have jurisdiction.</p>
{special}

<h2 class="pagebreak">Schedule I &ndash; Loan details</h2>
<table>
<tr><th>Item</th><th>Detail</th></tr>
<tr><td>Loan number</td><td>{escape(loan.ref)}</td></tr>
<tr><td>Borrower</td><td>{escape(loan.borrower)} (PAN XXXXX{loan.phone[-4:]}X, Aadhaar XXXX XXXX {loan.phone[-4:]}), mobile {escape(loan.phone)}</td></tr>
<tr><td>Business</td><td>{escape(loan.business)} &ndash; {escape(loan.business_kind)}</td></tr>
<tr><td>Product</td><td>{escape(loan.product)}</td></tr>
<tr><td>Purpose</td><td>{escape(loan.purpose)}</td></tr>
<tr><td>Loan amount</td><td>{inr(loan.principal)}</td></tr>
<tr><td>Rate of interest</td><td>{loan.rate_pct:.2f}% per annum, fixed, reducing balance</td></tr>
<tr><td>APR</td><td>{f['apr']:.2f}% per annum</td></tr>
<tr><td>Tenure</td><td>{loan.tenure} months</td></tr>
<tr><td>EMI</td><td>{inr(f['emi'])} (last EMI {inr(f['last']['emi'], True)})</td></tr>
<tr><td>EMI due date</td><td>{loan.first_emi.day}{_ordinal(loan.first_emi.day)} of each month, {long_date(loan.first_emi)} to {long_date(f['dates'][-1])}</td></tr>
<tr><td>Processing fee</td><td>{loan.processing_fee_pct:.1f}% = {inr(f['fee'], True)} + GST {inr(f['gst'], True)}, deducted from the disbursement</td></tr>
<tr><td>Amount credited</td><td>{inr(f['received'], True)} on {long_date(loan.disbursed_on)}</td></tr>
<tr><td>Total amount payable</td><td>{inr(f['total'], True)} (principal {inr(loan.principal)} + interest {inr(f['interest'], True)})</td></tr>
<tr><td>Grace period</td><td>{_GRACE_DAYS} days after each due date</td></tr>
<tr><td>Late payment charge</td><td>{inr(_LATE_FEE)} flat per EMI unpaid after the grace period</td></tr>
<tr><td>Prepayment / foreclosure charge</td><td>Nil</td></tr>
<tr><td>Repayment mode</td><td>NACH mandate on {escape(loan.nach_bank)}</td></tr>
<tr><td>Security</td><td>{escape(loan.security) if secured else 'Unsecured'}</td></tr>
</table>

<h2>Schedule II &ndash; Repayment schedule</h2>
{_schedule_table(loan)}

<div class="sign"><div>For {LENDER}<br>Authorised signatory (Credit Operations)<br>Signed electronically on {long_date(loan.sanctioned_on)}</div>
<div>{escape(loan.borrower)}, Borrower<br>Signed by Aadhaar e-Sign on {long_date(loan.sanctioned_on)}<br>e-Sign reference ending {loan.phone[-6:]}</div></div>
"""
    return _page(f"Loan Agreement {loan.ref}", f"Loan Agreement {loan.ref}", body)


def kfs_html(loan: Loan) -> str:
    f = _figures(loan)
    secured = loan.security is not None
    body = f"""{_masthead("Key Fact Statement", loan)}
<h1>Key Fact Statement</h1>
<p>Date: {long_date(loan.sanctioned_on - timedelta(days=1))} &middot; Applicant: {escape(loan.borrower)}, {escape(loan.business)}.
This statement sets out the key terms of the loan offered to you, in the format the Reserve Bank of India prescribes, before
you sign the loan agreement.</p>

<h2>Part 1 &ndash; Interest rate and fees/charges</h2>
<table>
<tr><th>#</th><th>Item</th><th>Detail</th></tr>
<tr><td>1</td><td>Loan proposal / account number; type of loan</td><td>{escape(loan.ref)}; {escape(loan.product)}</td></tr>
<tr><td>2</td><td>Sanctioned loan amount</td><td>{inr(loan.principal)}</td></tr>
<tr><td>3</td><td>Disbursal schedule</td><td>100% upfront, in one tranche, on {long_date(loan.disbursed_on)}</td></tr>
<tr><td>4</td><td>Loan term</td><td>{loan.tenure} months</td></tr>
<tr><td>5</td><td>Instalment details</td><td>{loan.tenure} monthly EMIs of {inr(f['emi'])} (last {inr(f['last']['emi'], True)}); first EMI {long_date(loan.first_emi)}; due on the {loan.first_emi.day}{_ordinal(loan.first_emi.day)} of each month</td></tr>
<tr><td>6</td><td>Interest rate and type</td><td>{loan.rate_pct:.2f}% per annum, fixed, on the reducing balance</td></tr>
<tr><td>7</td><td>Floating-rate reset details</td><td>Not applicable (fixed rate)</td></tr>
<tr><td>8</td><td>Fees and charges payable to the lender</td><td>Processing fee {loan.processing_fee_pct:.1f}% = {inr(f['fee'], True)} plus GST {inr(f['gst'], True)}, one-time, deducted from the disbursement. No other fee.</td></tr>
<tr><td>8a</td><td>Payable to third parties through the lender</td><td>{'Comprehensive vehicle insurance with the lender as loss payee, arranged and paid by the borrower directly' if 'truck' in (loan.security or '') else 'None'}</td></tr>
<tr><td>9</td><td>Annual Percentage Rate (APR)</td><td><b>{f['apr']:.2f}%</b> per annum</td></tr>
<tr><td>10a</td><td>Penal charges for delayed payment</td><td>{inr(_LATE_FEE)} flat for each EMI unpaid after a grace period of {_GRACE_DAYS} days from its due date; once per EMI, not per day; no interest on it</td></tr>
<tr><td>10b</td><td>Other penal charges</td><td>None</td></tr>
<tr><td>10c</td><td>Foreclosure / prepayment charges</td><td>Nil</td></tr>
<tr><td>10d</td><td>Charges for a dishonoured NACH debit</td><td>Nil from the lender (your bank may charge you under its own tariff)</td></tr>
</table>

<h2>Part 2 &ndash; Other qualitative information</h2>
<table>
<tr><th>#</th><th>Item</th><th>Detail</th></tr>
<tr><td>1</td><td>Engagement of recovery agents</td><td>None. Follow-up is by the lender's own staff and its automated assistant (loan agreement clause 12).</td></tr>
<tr><td>2</td><td>Grievance redressal mechanism</td><td>Loan agreement clause 14 and policy BFF-POL-05: raise it in the chat, Telegram or by phone; escalation to the Grievance Redressal Officer, then the Principal Nodal Officer.</td></tr>
<tr><td>3</td><td>Grievance Redressal Officer</td><td>Grievance Redressal Officer, {LENDER}, Mumbai; current contact details are given on request through the chat or by our team.</td></tr>
<tr><td>4</td><td>Transfer or securitisation</td><td>The loan may be assigned or securitised in accordance with RBI directions, with notice to you.</td></tr>
<tr><td>5</td><td>Co-lending</td><td>Not applicable</td></tr>
<tr><td>6</td><td>Cooling-off / look-up period</td><td>3 days from disbursement: you may exit by repaying the principal and the proportionate interest, with no penalty.</td></tr>
<tr><td>7</td><td>Lending service provider</td><td>None; the loan is sourced and serviced by the lender directly.</td></tr>
<tr><td>8</td><td>Security</td><td>{escape(loan.security) if secured else 'None (unsecured loan)'}</td></tr>
</table>

<h2>Annex A &ndash; Computation of the APR</h2>
<table>
<tr><th>#</th><th>Parameter</th><th class="num">Value</th></tr>
<tr><td>1</td><td>Sanctioned loan amount</td><td class="num">{inr(loan.principal, True)}</td></tr>
<tr><td>2</td><td>Loan term</td><td class="num">{loan.tenure} months</td></tr>
<tr><td>3</td><td>Number of EMIs</td><td class="num">{loan.tenure}</td></tr>
<tr><td>4</td><td>EMI amount</td><td class="num">{inr(f['emi'], True)}</td></tr>
<tr><td>5</td><td>Interest rate</td><td class="num">{loan.rate_pct:.2f}% p.a.</td></tr>
<tr><td>6</td><td>Total interest over the term</td><td class="num">{inr(f['interest'], True)}</td></tr>
<tr><td>7</td><td>Fees and charges (processing fee + GST)</td><td class="num">{inr(f['fee'] + f['gst'], True)}</td></tr>
<tr><td>8</td><td>Net disbursed amount (1 &minus; 7)</td><td class="num">{inr(f['received'], True)}</td></tr>
<tr><td>9</td><td>Total amount to be paid (1 + 6)</td><td class="num">{inr(f['total'], True)}</td></tr>
<tr><td>10</td><td>APR (rate at which the EMIs repay the net disbursed amount)</td><td class="num">{f['apr']:.2f}%</td></tr>
<tr><td>11</td><td>Schedule of disbursement</td><td class="num">One tranche, {long_date(loan.disbursed_on)}</td></tr>
<tr><td>12</td><td>Due date of payment of instalment and interest</td><td class="num">{loan.first_emi.day}{_ordinal(loan.first_emi.day)} of each month</td></tr>
</table>

<h2>Annex B &ndash; Illustrative repayment schedule</h2>
{_schedule_table(loan)}
<p class="small">Acknowledged by the applicant on {long_date(loan.sanctioned_on - timedelta(days=1))}, before signing the loan agreement.</p>
"""
    return _page(f"Key Fact Statement {loan.ref}", f"KFS {loan.ref}", body)


def _ordinal(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


_ONES = ("", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
         "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen")
_TENS = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")


def _two(n: int) -> str:
    return _ONES[n] if n < 20 else (_TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else ""))


def _words(amount: float) -> str:
    """Rupees in words, Indian style (lakh, thousand)."""
    n = int(amount)
    parts = []
    for size, name in ((10_000_000, "Crore"), (100_000, "Lakh"), (1000, "Thousand"), (100, "Hundred")):
        if n >= size:
            parts.append(f"{_two(n // size)} {name}")
            n %= size
    if n:
        parts.append(_two(n))
    return "Rupees " + " ".join(parts)


def main() -> None:
    browser = _browser()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy(_CSS, work / "policy.css")
        for loan in LOANS:
            stem = file_stem(loan)
            for kind, html in (("Loan Agreement", agreement_html(loan)), ("Key Fact Statement", kfs_html(loan))):
                page = work / f"{stem} {kind}.html"
                page.write_text(html, encoding="utf-8")
                pdf = _OUT / f"{stem} {kind}.pdf"
                print_pdf(browser, page, pdf)
                print(f"{pdf.relative_to(_REPO)}: {pdf.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
