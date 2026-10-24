"""Retrieval on the library that is actually live: the eight policy PDFs (data/policies) and each demo borrower's own
loan agreement and Key Fact Statement (data/demo/documents), read from whatever database DATABASE_URL names.

eval/retrieval_benchmark.py scores the older markdown corpus and its fixtures; this scores the documents a borrower is
really answered from. No model call: with GROQ_API_KEY unset a Devanagari query is searched by the multilingual
embedding alone (the lower bound -- with a key it is translated first), which is how the Hindi gap shows up as a number.

A question is a hit at k when one of the top k passages is from an acceptable document (and, when given, under the
heading it should come from). Reported: Recall@2 (what check_policy hands the assistant), Recall@5, MRR, English against
Hindi, and the borrower-own-agreement questions on their own.

Run (needs the demo built in DATABASE_URL): python -m eval.library_benchmark
"""

from dataclasses import dataclass

POLICY = "data/documents/_library/"


@dataclass(frozen=True)
class Q:
    text: str
    sources: tuple[str, ...]  # any of these substrings in source_document is acceptable
    heading: str = ""  # when set, the passage's heading path must contain it
    lang: str = "en"


QUESTIONS = [
    Q("Is there a charge if my auto-debit bounces?", ("02-repayment", "06-nach", "08-borrower")),
    Q("How long is the grace period after the due date?", ("02-repayment", "08-borrower"), "grace"),
    Q("How much is the late fee and when does it start?", ("02-repayment", "08-borrower")),
    Q("Which gets paid first, the EMI or the late fee?", ("02-repayment", "08-borrower")),
    Q("Can I prepay my loan and is there a foreclosure charge?", ("02-repayment", "07-closure", "08-borrower")),
    Q("Can the late fee be waived and who decides?", ("02-repayment", "08-borrower"), "waiv"),
    Q("What do SMA-1 and 31-60 days past due mean?", ("02-repayment",), "Days past due"),
    Q("When am I reminded about my EMI?", ("03-reminders",), "eminder"),
    Q("How many parts can a promise to pay have and how far ahead?", ("03-reminders", "08-borrower")),
    Q("When is a promise to pay counted as broken?", ("03-reminders",), "judged"),
    Q("When is my account passed to a person on the team?", ("03-reminders",), "scalation"),
    Q("What is the minimum I can pay as a part-payment?", ("04-restructuring", "08-borrower"), "art-payment"),
    Q("How much does a tenure extension lower my EMI?", ("04-restructuring",), "tenure extension"),
    Q("How big is the one-time settlement discount?", ("04-restructuring",), "settlement"),
    Q("Why am I refused an automatic extension offer?", ("04-restructuring", "08-borrower")),
    Q("I already paid but it is not showing, what happens?", ("05-disputes", "08-borrower")),
    Q("What happens to reminders when I raise a dispute?", ("05-disputes", "08-borrower")),
    Q("How do I complain to the RBI Ombudsman?", ("05-disputes",), "Ombudsman"),
    Q("What if someone claims a loan was taken in my name?", ("05-disputes",), "fraud"),
    Q("Why would a NACH debit fail?", ("06-nach", "08-borrower"), "fails"),
    Q("How long is a payment link valid?", ("06-nach", "08-borrower")),
    Q("How do I register a new NACH mandate?", ("06-nach",), "mandate"),
    Q("How do I get my loan closure certificate?", ("07-closure", "08-borrower")),
    Q("What happens to the vehicle papers when I finish paying?", ("07-closure",), "vehicle"),
    Q("Can the lender call my relatives about my dues?", ("01-fair",), "conduct"),
    Q("What can the automated assistant not do?", ("01-fair", "08-borrower")),
    # Hindi: the same needs in the borrower's own language.
    Q("किस्त चूकने पर कितनी लेट फीस लगती है?", ("02-repayment", "08-borrower"), lang="hi"),
    Q("ग्रेस पीरियड कितने दिन का होता है?", ("02-repayment", "08-borrower"), lang="hi"),
    Q("लोन जल्दी चुकाने पर कोई चार्ज लगेगा?", ("02-repayment", "07-closure", "08-borrower"), lang="hi"),
    Q("वादा तोड़ने पर क्या होता है?", ("03-reminders", "08-borrower"), lang="hi"),
    Q("भुगतान का वादा कितने हिस्सों में कर सकते हैं?", ("03-reminders", "08-borrower"), lang="hi"),
    Q("मैंने पेमेंट कर दिया है पर दिख नहीं रहा", ("05-disputes", "08-borrower"), lang="hi"),
    Q("RBI ओम्बड्समैन से शिकायत कैसे करें?", ("05-disputes",), lang="hi"),
    Q("NACH ऑटो डेबिट फेल हो जाए तो क्या करें?", ("06-nach", "08-borrower"), lang="hi"),
    Q("किस्त में कम से कम कितना भुगतान कर सकता हूँ?", ("04-restructuring", "08-borrower"), lang="hi"),
    Q("लोन बंद होने का सर्टिफिकेट कैसे मिलेगा?", ("07-closure", "08-borrower"), lang="hi"),
]

# (borrower, loan type, question, the fact the answer must contain): a hit is a passage of THEIR OWN documents, among what
# check_policy offers, that contains the fact -- wherever in the agreement it sits (the EMI is in the summary, clause 4 and
# Schedule I alike).
OWN = [
    ("Harpreet Singh", "Commercial Vehicle Loan", "What did I pledge as security for this loan?", "PB 10 HK 4471"),
    ("Ramesh Yadav", "Equipment Loan", "What is the interest rate on this loan?", "16.00%"),
    ("Meena Gupta", "Working Capital Loan", "How many EMIs do I pay and how much is each?", "₹14,978"),
    ("Sunita Sharma", "Working Capital Loan", "Which bank account is my auto-debit on?", "Bank of Baroda"),
    ("Imran Qureshi", "Business Loan", "What is the processing fee on my loan?", "6,000.00"),
    ("Lakshmi Narayanan", "Working Capital Loan", "What was this loan taken for?", "bulk purchase of rice"),
]


def _rank(results: list[dict], q_sources: tuple[str, ...], heading: str) -> int | None:
    for i, r in enumerate(results, 1):
        doc = r["source_document"]
        if any(s in doc for s in q_sources) and heading.lower() in (r.get("headings") or "").lower():
            return i
    return None


def run() -> dict:
    from businessflow.accounts.db import get_connection
    from businessflow.rag.retriever import DocumentRetriever

    retriever = DocumentRetriever()
    rows = []
    for q in QUESTIONS:
        results = retriever.retrieve(q.text, top_k=5, candidate_pool=8)
        # A heading is a bonus the benchmark reports, not a requirement: a hit is the right document.
        rank = _rank(results, q.sources, "")
        rows.append({"kind": "policy-" + q.lang, "q": q.text, "rank": rank,
                     "heading_ok": _rank(results, q.sources, q.heading) is not None if q.heading else None})
    accounts = {(r["borrower_name"], r["loan_type"]): r["account_id"]
                for r in get_connection().execute("select account_id, borrower_name, loan_type from accounts").fetchall()}
    for borrower, loan_type, text, fact in OWN:
        account_id = accounts.get((borrower, loan_type))
        if account_id is None:
            rows.append({"kind": "own-agreement", "q": f"[{borrower} not in this database] {text}", "rank": None, "heading_ok": None})
            continue
        # What check_policy hands the model: the usual top 2 plus the borrower's two best agreement passages.
        offered = retriever.retrieve(text, top_k=2, candidate_pool=4, account_id=account_id) + retriever.retrieve(
            text, top_k=2, candidate_pool=8, account_id=account_id, document_type="loan_agreement")
        rank = next((i for i, r in enumerate(offered, 1) if r["account_id"] == account_id and fact in r["text"]), None)
        rows.append({"kind": "own-agreement", "q": f"[{account_id} {borrower}] {text}", "rank": rank, "heading_ok": None})
    return {"rows": rows}


def summarise(rows: list[dict]) -> dict:
    out = {}
    for kind in ("policy-en", "policy-hi", "own-agreement"):
        part = [r for r in rows if r["kind"] == kind]
        if not part:
            continue
        ranks = [r["rank"] for r in part]
        if kind == "own-agreement":  # the question is whether the fact is anywhere in the four passages handed to the model
            out[kind] = {"n": len(part), "offered": sum(x is not None for x in ranks) / len(part),
                         "missed": [r["q"] for r in part if r["rank"] is None]}
            continue
        out[kind] = {
            "n": len(part),
            "recall@2": sum(1 for x in ranks if x is not None and x <= 2) / len(part),
            "recall@5": sum(1 for x in ranks if x is not None and x <= 5) / len(part),
            "mrr": sum(1 / x for x in ranks if x) / len(part),
            "missed": [r["q"] for r in part if r["rank"] is None or r["rank"] > 2],
        }
    return out


if __name__ == "__main__":
    import logging

    logging.disable(logging.WARNING)
    from dotenv import load_dotenv

    load_dotenv()
    result = run()
    for kind, s in summarise(result["rows"]).items():
        if kind == "own-agreement":
            print(f"{kind:14} n={s['n']:2}  the answer is among what check_policy offers: {s['offered']:.0%}")
        else:
            print(f"{kind:14} n={s['n']:2}  recall@2={s['recall@2']:.0%}  recall@5={s['recall@5']:.0%}  MRR={s['mrr']:.2f}")
        for m in s["missed"]:
            print(f"      missed: {m}")
