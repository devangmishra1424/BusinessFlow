"""The 2026-10-23 capabilities audit: what the product could do but no conversational surface could reach.

- the assistant's new tools (tools/request_tools.py): its requests and our replies, asking for a fee waiver, replying
  to our team -- against real Postgres;
- Telegram's /requests and /promise, and the note about other loans in a multi-loan chat;
- the ops voice assistant's new answers and cards (bank, collection health, quality, waive a fee, unlink Telegram,
  why an account is risky) -- with the same fake dashboard data as tests/test_ops_voice.py.
"""

import asyncio

import pytest

from businessflow.accounts import store
from businessflow.ops import voice_reply
from businessflow.ops.voice import parse_plan
from businessflow.tools import request_tools

# --- the assistant's tools ------------------------------------------------------------------------------------------


def test_my_requests_lists_theirs_with_our_last_word_and_leaves_out_system_tickets(reseed_accounts):
    claim = store.open_dispute("BF-1001", "Paid 12,500 by UPI on the 3rd", kind="payment_claim")
    ref = f"D-{claim['dispute_id']}"
    store.add_case_message("BF-1001", ref, "ops", "We are checking the bank record.")
    store.create_escalation("BF-1001", "Guardrail blocked a reply: an amount nobody gave")  # a system ticket
    store.log_event("BF-1001", "clarification_request_sent", {"message": "Why is the EMI late?", "delivered_via_telegram": False})

    result = request_tools.get_my_requests("BF-1001")

    [open_claim] = [r for r in result["open_requests"] if r["ref"] == ref]
    assert open_claim["what"] == "payment the borrower reported" and open_claim["last_from_our_team"] == "We are checking the bank record."
    assert not any("Guardrail" in r["their_words"] for r in result["open_requests"] + result["recently_closed"])
    assert result["messages_from_our_team"][0]["message"] == "Why is the EMI late?"


def test_a_fee_waiver_is_only_ever_a_request(reseed_accounts):
    assert request_tools.request_fee_waiver("BF-1001", "Hospital stay")["no_fee_owed"] is True  # BF-1001 is in grace

    store.assess_late_fees(account_id="BF-1002")
    result = request_tools.request_fee_waiver("BF-1002", "I was in hospital that week")
    assert result["requested"] is True and "₹500" in result["fees_owed"]
    assert store.get_account_or_raise("BF-1002").open_charges(), "the fee is still owed -- only a person waives it"
    mine = request_tools.get_my_requests("BF-1002")["open_requests"]
    assert any(r["ref"] == result["case_ref"] and r["what"] == "fee waiver request" for r in mine)


def test_a_reply_to_our_team_goes_to_the_team(reseed_accounts):
    assert request_tools.reply_to_team("BF-1001", "here is why")["no_team_message"] is True
    store.log_event("BF-1001", "clarification_request_sent", {"message": "Why is the EMI late?", "delivered_via_telegram": False})

    result = request_tools.reply_to_team("BF-1001", "Shop was shut for a week.")

    assert result["new_request"] is True and result["replying_to"] == "Why is the EMI late?"
    assert store.get_case("BF-1001", result["case_ref"])["open"]


# --- Telegram ---------------------------------------------------------------------------------------------------------

def test_telegram_requests_and_promise_commands(reseed_accounts):
    from businessflow.accounts.promises import Part
    from businessflow.channels import telegram_bot

    store.set_telegram_chat_id("BF-1001", 930001)
    telegram_bot._sessions.clear()
    assert asyncio.run(telegram_bot._run_requests(930001)) == "You have no requests with us, and no messages from our team."
    assert asyncio.run(telegram_bot._run_promise(930001)).startswith("No promise to pay is on record.")

    today = store.current_date()
    from datetime import timedelta
    store.add_promise_plan("BF-1001", [Part(today + timedelta(days=3), 5000), Part(today + timedelta(days=10), 7500)], today, "ops")
    text = asyncio.run(telegram_bot._run_promise(930001))
    assert "₹5,000 by" in text and "then ₹7,500" in text and "on track" in text and "/pay" in text


def test_a_multi_loan_chat_tells_the_assistant_about_the_other_loans(reseed_accounts):
    from businessflow.channels import telegram_bot

    store.set_telegram_chat_id("BF-1001", 930002)
    store.set_telegram_chat_id("BF-1002", 930002)
    conversation = telegram_bot._with_loans_note([{"role": "system", "content": "x"}], 930002, "BF-1002")
    assert "BF-1001" in conversation[-1]["content"] and "/loans" in conversation[-1]["content"]
    alone = telegram_bot._with_loans_note([], 930003, "BF-1003")
    assert alone == []


def test_the_new_abilities_are_offered_to_a_verified_borrower():
    from businessflow.channels.abilities import abilities_for

    ids = [a["id"] for a in abilities_for("en", verified=True)]
    assert {"promise", "requests", "fee_waiver"} <= set(ids) and ids[-1] == "more"


# --- ops voice ---------------------------------------------------------------------------------------------------------

from tests.test_ops_voice import NOW, TODAY, FakeData, _detail, _summary  # noqa: E402


class RichData(FakeData):
    def bank_lines(self):
        return [{"amount": 22000.0, "line_date": "2026-10-14", "status": "suggested", "account_id": "BF-1002",
                 "match_reason": "the name in the narration looks like Asha Verma", "narration": "IMPS ASHA"},
                {"amount": 41.2, "line_date": "2026-10-14", "status": "unmatched", "account_id": None,
                 "match_reason": "nothing identifies a borrower", "narration": "INTEREST"}]

    def health(self):
        how = "how it is worked out"
        return {"dpd_buckets": [{"key": "current", "label": "Current", "accounts": 2, "owed_now": 0},
                                {"key": "1_30", "label": "1-30 days", "accounts": 1, "owed_now": 12500},
                                {"key": "31_60", "label": "31-60 days", "accounts": 1, "owed_now": 13250},
                                {"key": "61_90", "label": "61-90 days", "accounts": 0, "owed_now": 0},
                                {"key": "90_plus", "label": "90+ days", "accounts": 0, "owed_now": 0}],
                "collection_efficiency": {"value": 0.5, "how": how}, "promises_kept": {"value": None, "how": how},
                "cure_rate": {"value": 0.25, "how": how}}

    def quality(self):
        return {"verdict": [{"component": "Model calls", "status": "slow", "why": "answers right, but slow"}],
                "gates": [{"name": "Every model reply went through the guardrail", "passed": True, "detail": "all checked"}]}


def _ask(tool, heard="", data=None, screen_account_id=None, **arguments):
    return voice_reply.respond(parse_plan(tool, arguments), heard, data or RichData(), screen_account_id, TODAY, NOW)


def test_voice_answers_about_the_bank_the_book_and_the_ai():
    bank = _ask("bank_status")
    assert bank.say.startswith("2 bank credits waiting: 1 with a best guess") and bank.screen.view == "bank"
    health = _ask("collection_health")
    assert "Collection efficiency 50%" in health.say and "promises kept no data" in health.say.lower()
    assert "1 31-60 days" in health.say
    quality = _ask("quality_summary")
    assert quality.say.startswith("No wrong answers from the AI") and quality.screen.view == "quality"
    assert _ask("open_page", page="bank").screen.view == "bank"


def _with_fee():
    a = _summary("BF-1003", "Fatima Khan", "Khan Kirana", dpd=45, flags=("overdue",), due_now=13250, telegram=True)
    detail = _detail(a, charges=[{"id": 77, "kind": "late_fee", "for_due_date": "2026-09-01", "amount": 500.0,
                                  "amount_paid": 0.0, "outstanding": 500.0, "status": "due", "assessed_on": "2026-09-05"}],
                     risk={"tier": "high", "score": 60, "factors": [{"label": "45 days past due", "points": 65.0},
                                                                     {"label": "Pending credit", "points": -10.0}]})
    return RichData(accounts=[a], details={"BF-1003": detail})


def test_voice_waives_a_fee_only_through_a_card_with_the_reason():
    reply = _ask("waive_fee", heard="waive Fatima's late fee because she was in hospital", data=_with_fee(),
                 account="Fatima", reason="She was in hospital")
    p = reply.proposal
    assert reply.kind == "proposal" and p.path == "/accounts/BF-1003/charges/77/waive"
    assert p.body == {"waived_by": "Ops (voice)", "reason": "She was in hospital"} and p.editable == "reason"
    none = _ask("waive_fee", heard="waive the fee", data=RichData(), account="BF-1002")
    assert none.kind == "clarify" and "no late fee" in none.say


def test_voice_unlinks_telegram_and_explains_risk():
    data = _with_fee()
    assert _ask("unlink_telegram", data=data, account="BF-1003").proposal.path == "/accounts/BF-1003/telegram/unlink"
    risk = _ask("account_details", data=data, account="BF-1003", topic="risk")
    assert "high risk, score 60 of 100" in risk.say and "45 days past due" in risk.say
    assert [r.value for r in risk.rows] == ["+65", "-10"]


@pytest.mark.parametrize("tool", ["bank_status", "collection_health", "quality_summary", "waive_fee", "unlink_telegram"])
def test_every_new_voice_command_is_on_the_menu_the_model_chooses_from(tool):
    from businessflow.ops.voice import MENU

    assert tool in {t["function"]["name"] for t in MENU}


def test_voice_opens_the_policy_library_and_accepts_any_page_as_the_current_screen():
    assert _ask("open_page", page="library").screen.view == "library"
    from fastapi.testclient import TestClient
    import os
    from businessflow.ops.api import app

    for view in ("bank", "quality", "library"):  # asking while one of these pages is open used to be refused (422)
        r = TestClient(app).post("/voice/plan", headers={"X-API-Key": os.environ["OPS_API_KEY"]},
                                 json={"tool": "open_page", "arguments": {"page": "bank"}, "heard": "open the bank", "screen": {"view": view}})
        assert r.status_code == 200, (view, r.text)


def test_a_borrowers_own_agreement_is_always_offered_beside_the_policy_excerpts(reseed_accounts):
    from pathlib import Path

    from businessflow.rag.ingest import ingest_document
    from businessflow.rag.store import delete_chunks_for_document
    from businessflow.tools import policy_tools

    path = Path(__file__).resolve().parents[1] / "data" / "documents" / "BF-1001" / "zz_agreement_check.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Security\n\nThe Borrower hypothecates the light commercial truck registered PB 10 HK 4471.\n", encoding="utf-8")
    try:
        ingest_document(str(path), "loan_agreement", "BF-1001")
        policy_tools._cached_retriever = None  # a fresh snapshot with the new document
        results = policy_tools.check_policy("what is the grace period?", "BF-1001")["results"]  # a general-policy question
        assert any("PB 10 HK 4471" in r["text"] for r in results), "the borrower's own agreement must be offered too"
        other = policy_tools.check_policy("what is the grace period?", "BF-1002")["results"]
        assert not any("PB 10 HK 4471" in r["text"] for r in other), "never another borrower's"
    finally:
        delete_chunks_for_document(str(path))
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass  # other documents live there
        policy_tools._cached_retriever = None
