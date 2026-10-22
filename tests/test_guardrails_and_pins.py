"""PR 5's hardening: Hindi/Hinglish claim checks (guardrail/grounding.py), the automated-reminder check and its fixed
fallback (outbound/compose.py), and hashed access keys (accounts/pins.py, accounts/store.py).
"""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from businessflow.accounts import pins
from businessflow.accounts.models import Account
from businessflow.guardrail import grounding
from businessflow.outbound import compose
from businessflow.outbound.decide import OutboundReminder

# --- Hindi and Hinglish claims ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "आपकी लेट फीस माफ़ कर दी गई है।",
    "हमने आपका जुर्माना हटा दिया है।",
    "Aapki late fee waive kar di gayi hai.",
])
def test_a_waiver_claimed_in_hindi_or_hinglish_is_caught(text):
    assert grounding.check_fabricated_waiver(text)


@pytest.mark.parametrize("text", [
    "लेट फीस माफ़ नहीं की जा सकती, लेकिन टीम देख सकती है।",
    "Late fee maaf nahi ho sakti.",
    "आपकी अगली EMI 12,500 रुपये है, 25 अक्टूबर तक।",
])
def test_a_refusal_or_a_plain_fact_in_hindi_is_not_a_claim(text):
    assert not grounding.check_fabricated_waiver(text)
    assert not grounding.check_unapplied_restructuring_claim(text)
    assert not grounding.check_fabricated_action(text)


def test_a_sent_document_and_an_applied_restructuring_are_caught_in_hindi():
    assert grounding.check_fabricated_action("मैंने आपको एग्रीमेंट की कॉपी भेज दी है।")
    assert grounding.check_fabricated_action("Maine agreement ki copy bhej di hai.")
    assert grounding.check_unapplied_restructuring_claim("आपकी EMI कम कर दी गई है।")
    assert grounding.check_unapplied_restructuring_claim("Aapka tenure badha diya gaya hai.")
    assert not grounding.check_unapplied_restructuring_claim("आपकी EMI कम हो सकती है अगर टीम मंज़ूर करे।")


def test_a_waiver_a_person_really_made_is_not_blocked():
    told = [{"role": "tool", "content": '{"recently_waived": [{"for_emi_due_date": "2026-10-05", "amount_waived": 500.0}]}'}]
    assert not grounding.check_fabricated_waiver("Your late fee has been waived by our team.", told)
    assert grounding.check_fabricated_waiver("Your late fee has been waived by our team.", [{"role": "tool", "content": '{"recently_waived": []}'}])


# --- automated reminders ---------------------------------------------------------------------------------------------

def _account(lang="en") -> Account:
    return Account(
        account_id="T-1", borrower_name="Priya Sharma", business_name="B", phone_number="+919800000000",
        language_preference=lang, loan_type="WC", principal_amount=100_000, emi_amount=12_500, tenure_months=12,
        months_remaining=6, emi_due_date=date(2026, 10, 25), nach_mandate_active=True, dispute_open=False, risk_tier="low",
    )


def test_a_reminder_may_only_state_the_figures_it_was_given():
    assert compose.reminder_problem("Your EMI of ₹12,500 is due on 25 Oct.", {12_500.0}) is None
    assert "₹15,000" in compose.reminder_problem("Pay ₹15,000 today.", {12_500.0})
    assert compose.reminder_problem("Your late fee has been waived, pay ₹12,500.", {12_500.0})
    assert compose.reminder_problem("", {12_500.0}) == "the draft is empty"


def _fake_groq(drafts):
    calls = []

    def create(**kwargs):
        calls.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=drafts[len(calls) - 1]))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_a_bad_draft_is_rewritten_once_then_replaced_by_the_fixed_wording(monkeypatch):
    client, calls = _fake_groq(["Pay ₹99,999 now.", "Pay ₹88,888 now."])
    monkeypatch.setattr(compose, "groq_client", lambda: client)
    monkeypatch.setattr(compose, "reminder_amounts", lambda account, kind: (12_500.0, 0.0, 12_500.0))
    monkeypatch.setattr(compose.store, "log_event", lambda *a, **k: None)

    text = compose.compose_message(_account(), OutboundReminder("T-1", "heads_up", 3))

    assert len(calls) == 2 and "cannot be sent" in calls[1][-1]["content"]
    assert text == "Hi Priya, your EMI of ₹12,500 is due on 25 Oct 2026."


def test_a_good_draft_goes_out_as_written(monkeypatch):
    client, calls = _fake_groq(["Hi Priya, your EMI of ₹12,500 is due in 3 days."])
    monkeypatch.setattr(compose, "groq_client", lambda: client)
    monkeypatch.setattr(compose, "reminder_amounts", lambda account, kind: (12_500.0, 0.0, 12_500.0))

    assert compose.compose_message(_account(), OutboundReminder("T-1", "heads_up", 3)) == "Hi Priya, your EMI of ₹12,500 is due in 3 days."
    assert len(calls) == 1


def test_the_fixed_wording_is_hindi_for_a_hindi_borrower():
    text = compose.fallback_reminder(_account("hi"), OutboundReminder("T-1", "follow_up", 9), 12_500, 500, 13_000, None)
    assert "₹500" in text and "₹13,000" in text and "लेट फीस" in text


# --- access keys -----------------------------------------------------------------------------------------------------

def test_a_pin_verifies_against_its_hash_and_nothing_else():
    stored = pins.hash_pin("482913")
    assert stored.startswith("scrypt$") and "482913" not in stored
    assert pins.verify_pin("482913", stored)
    assert not pins.verify_pin("482914", stored)
    assert not pins.verify_pin("482913", None) and not pins.verify_pin("482913", "md5$x$y") and not pins.verify_pin("482913", "junk")
    assert pins.hash_pin("482913") != stored  # salted


def test_the_database_keeps_only_the_hash_and_an_old_plain_key_is_upgraded_on_use(reseed_accounts):
    from businessflow.accounts import store

    conn = store.get_connection()
    row = conn.execute("select access_key, access_key_hash from accounts where account_id = 'BF-1001'").fetchone()
    assert row["access_key"] is None and row["access_key_hash"].startswith("scrypt$")
    assert store.verify_account_key("BF-1001", "482913") and not store.verify_account_key("BF-1001", "000000")

    # A key from before hashing: works once as it was, and is stored hashed from then on.
    conn.execute("update accounts set access_key = '123456', access_key_hash = null where account_id = 'BF-1001'")
    assert store.verify_account_key("BF-1001", "123456")
    row = conn.execute("select access_key, access_key_hash from accounts where account_id = 'BF-1001'").fetchone()
    assert row["access_key"] is None and pins.verify_pin("123456", row["access_key_hash"])


def test_a_reset_key_is_returned_once_and_stored_only_as_a_hash(reseed_accounts):
    from businessflow.accounts import store

    new_key = store.reset_access_key("BF-1002")
    row = store.get_connection().execute("select access_key, access_key_hash from accounts where account_id = 'BF-1002'").fetchone()
    assert row["access_key"] is None and pins.verify_pin(new_key, row["access_key_hash"])
    assert not store.verify_account_key("BF-1002", "716044")  # the old seeded key stopped working
