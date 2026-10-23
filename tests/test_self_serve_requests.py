"""The borrower dashboard's Get help buttons for what only the assistant could do before: asking to remove a late fee,
and asking for the closure certificate (channels/browser_api.py quick-actions/fee-waiver and /closure-certificate).

Real Postgres, the same tools the assistant calls (tools/request_tools.py, tools/escalation_tools.py).
"""

import os

import pytest
from fastapi.testclient import TestClient

from businessflow.accounts import store
from businessflow.accounts.db import get_connection
from businessflow.channels.browser_api import app

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs real Postgres (DATABASE_URL)")

client = TestClient(app)


def _open(account_id: str) -> str:
    r = client.post("/conversations", json={"language": "en", "account_id": account_id, "demo_open": True})
    assert r.status_code == 200, r.text
    return r.json()["conversation_id"]


def test_asking_to_remove_a_late_fee_sends_a_request_and_leaves_the_fee_owed(reseed_accounts):
    store.assess_late_fees(account_id="BF-1002")
    cid = _open("BF-1002")
    r = client.post(f"/conversations/{cid}/quick-actions/fee-waiver", json={"reason": "I was in hospital that week"})
    assert r.status_code == 200 and r.json()["requested"] is True
    assert store.get_account_or_raise("BF-1002").open_charges(), "only a person on the team waives a fee"
    case = store.get_case("BF-1002", r.json()["case_ref"])
    assert case["open"] and "Fee waiver request" in str(case)


def test_with_no_fee_owed_there_is_nothing_to_ask_and_a_reason_is_required(reseed_accounts):
    cid = _open("BF-1001")  # in its grace period: no fee
    assert client.post(f"/conversations/{cid}/quick-actions/fee-waiver", json={"reason": "please"}).json()["no_fee_owed"] is True
    assert client.post(f"/conversations/{cid}/quick-actions/fee-waiver", json={"reason": "   "}).status_code == 400


def test_the_closure_certificate_is_requested_only_for_a_repaid_loan(reseed_accounts):
    cid = _open("BF-1001")
    early = client.post(f"/conversations/{cid}/quick-actions/closure-certificate").json()
    assert early["eligible"] is False and early["months_remaining"] > 0

    get_connection().execute("update accounts set months_remaining = 0 where account_id = 'BF-1001'")
    done = client.post(f"/conversations/{cid}/quick-actions/closure-certificate").json()
    assert done["eligible"] is True and done["escalation_id"].startswith("ESC-")
    assert store.get_case("BF-1001", done["escalation_id"])["open"]


def test_the_buttons_need_a_verified_conversation():
    assert client.post("/conversations/nope/quick-actions/fee-waiver", json={"reason": "x"}).status_code == 404
    assert client.post("/conversations/nope/quick-actions/closure-certificate").status_code == 404
