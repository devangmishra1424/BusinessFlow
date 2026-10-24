"""Gates that need no model and run on every change: what must stay true however the documents, the retriever or the
matcher are edited. (The reports behind them: eval/cash_application_benchmark.py; the library benchmark is
eval/library_benchmark.py.)

- cash application: the bank matcher never posts a credit to the wrong borrower on its own;
- no leakage: a borrower's own documents are never offered to another borrower, over every pair;
- the policy documents say what the code enforces (accounts/policy.py, accounts/promises.py);
- every figure in a policy document survives being read as a PDF;
- the server can read a PDF at all (the libGL outage of 2026-10-24).
"""

import os
import re
from pathlib import Path

import pytest

from businessflow.accounts import policy
from businessflow.accounts.promises import MAX_DAYS_AHEAD, MAX_PARTS

REPO = Path(__file__).resolve().parents[1]
SOURCES = REPO / "data" / "policies" / "source"
needs_db = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs real Postgres (DATABASE_URL)")


# --- cash application -------------------------------------------------------------------------------------------------

def test_the_bank_matcher_never_posts_to_the_wrong_borrower_on_its_own():
    from eval.cash_application_benchmark import evaluate

    r = evaluate()
    assert r["wrong_auto"] == [], "an automatic post went to the wrong account (or to nobody)"
    assert r["strong_recall"] == 1.0, r["strong_missed"]  # a line that names the loan, phone or claim UTR posts itself
    assert r["name_only_never_auto"], "a name alone must only ever be a suggestion"
    assert r["noise_matched"] == [], "bank noise was matched to a borrower"


# --- the policy documents against the code ---------------------------------------------------------------------------

def _text(name: str) -> str:
    html = (SOURCES / name).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html).replace("&nbsp;", " ").replace("&ndash;", "-"))


def test_every_number_the_code_enforces_is_stated_by_the_policy_that_governs_it():
    p = policy
    expected = {  # document -> phrases built from the constants, so changing a constant fails until the document is updated
        "02-repayment-grace-late-charges.html": [f"{p.GRACE_PERIOD_DAYS}-day grace period", f"₹{p.LATE_FEE_FLAT_AMOUNT}"],
        "03-reminders-promises-escalation.html": [
            f"{MAX_PARTS} parts", f"{MAX_DAYS_AHEAD} days", f"{p.BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION} promises",
            f"{p.MANDATORY_ESCALATION_DAYS_PAST_DUE} or more days", f"₹{p.LATE_FEE_FLAT_AMOUNT}",
            f"{p.HEADS_UP_DAYS_BEFORE_DUE}, 2 and 1 days before"],
        "04-restructuring-part-payment-settlement.html": [
            f"{round(p.MIN_PARTIAL_PAYMENT_PCT * 100)}% of one EMI", f"{round(p.SETTLEMENT_DISCOUNT_PCT * 100)}%",
            f"1 to {p.MAX_RESTRUCTURING_EXTENSION_MONTHS} months"],
        "05-disputes-claims-grievances.html": [f"{p.PAYMENT_CLAIM_REVIEW_WORKING_DAYS} working days"],
        "06-nach-and-ways-to-pay.html": [f"{p.PAYMENT_TOKEN_TTL_HOURS} hours"],
    }
    for name, phrases in expected.items():
        text = _text(name)
        missing = [ph for ph in phrases if ph not in text]
        assert not missing, f"{name} does not state {missing} (code: accounts/policy.py)"


# --- every figure survives being read as a PDF -----------------------------------------------------------------------

_FIGURE = re.compile(r"₹\s?[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?%|\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b")


def _squash(s: str) -> str:
    return re.sub(r"[\s,]", "", s.replace("₹", ""))


@pytest.mark.parametrize("pdf", sorted((REPO / "data" / "policies").glob("*.pdf")), ids=lambda p: p.name[:14])
def test_every_figure_in_a_policy_document_survives_being_read_as_a_pdf(pdf):
    from businessflow.rag.ingest import parse_and_chunk

    body = _squash(" ".join(f"{r.headings} {r.text}" for r in parse_and_chunk(str(pdf))))
    source = _text(pdf.name.replace(".pdf", ".html"))
    lost = sorted({f for f in _FIGURE.findall(source) if _squash(f) not in body})
    assert not lost, f"{pdf.name}: figures in the source that did not come out of the PDF: {lost}"


# --- no leakage between borrowers ------------------------------------------------------------------------------------

@needs_db
def test_no_borrower_is_ever_offered_another_borrowers_own_documents(reseed_accounts):
    from businessflow.rag.ingest import ingest_document
    from businessflow.rag.store import delete_chunks_for_document
    from businessflow.tools import policy_tools

    accounts = ["BF-1001", "BF-1002", "BF-1003", "BF-1004"]
    marker = {a: f"ZQX{n}MARKER" for n, a in enumerate(accounts)}
    paths = {}
    try:
        for a in accounts:
            path = REPO / "data" / "documents" / a / "zz_leak_check.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# Security\n\nThe asset pledged by this borrower is registered {marker[a]} and nothing else.\n", encoding="utf-8")
            ingest_document(str(path), "loan_agreement", a)
            paths[a] = path
        policy_tools._cached_retriever = None
        for asker in accounts:
            for question in ("what did I pledge as security?", "what is the grace period?", marker[accounts[(accounts.index(asker) + 1) % 4]]):
                text = " ".join(r["text"] for r in policy_tools.check_policy(question, asker)["results"])
                others = [marker[o] for o in accounts if o != asker and marker[o] in text]
                assert not others, f"{asker} was offered {others} on {question!r}"
        own = " ".join(r["text"] for r in policy_tools.check_policy("what did I pledge as security?", "BF-1001")["results"])
        assert marker["BF-1001"] in own, "a borrower must be offered their own agreement"
        anonymous = " ".join(r["text"] for r in policy_tools.check_policy("what did I pledge as security?")["results"])
        assert not any(m in anonymous for m in marker.values()), "an unverified chat must see no borrower's documents"
    finally:
        for a, path in paths.items():
            delete_chunks_for_document(str(path))
            path.unlink(missing_ok=True)
            try:
                path.parent.rmdir()
            except OSError:
                pass  # the account has other documents
        policy_tools._cached_retriever = None


# --- the server can read a PDF ---------------------------------------------------------------------------------------

def test_the_pdf_reader_canary_says_ok_here_and_names_the_missing_library_when_it_is_not(monkeypatch):
    import importlib

    from businessflow.rag import ingest

    monkeypatch.setattr(ingest, "_pdf_check", None)
    assert ingest.pdf_reader_status() == "ok"

    real = importlib.import_module

    def without_libgl(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("libGL.so.1: cannot open shared object file: No such file or directory")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(ingest, "_pdf_check", None)
    monkeypatch.setattr(ingest.importlib, "import_module", without_libgl)
    status = ingest.pdf_reader_status()
    assert status.startswith("ImportError: libGL.so.1")
    assert ingest.pdf_reader_status() == status  # cached for a minute, not re-imported on every health check
