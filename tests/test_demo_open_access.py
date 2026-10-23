"""The demo's one-click sign-in: every account is listed (new ones included) and opens without a key -- and a real
deployment turns it off with DEMO_OPEN_ACCESS=0, after which every account needs its access key again.
"""

import pytest
from fastapi.testclient import TestClient

from businessflow.accounts import store


@pytest.fixture
def client(reseed_accounts):
    from businessflow.channels.browser_api import app

    return TestClient(app)


def test_every_account_is_listed_including_a_new_one(client):
    account, _key = store.create_account("Test Person", "Test Shop", "+919800000001", "en", "Working Capital Loan",
                                         100_000, 5_000, 24, store.current_date(), True, "low")
    try:
        ids = [a["account_id"] for a in client.get("/demo/accounts").json()]
        assert "BF-1001" in ids and account.account_id in ids
    finally:
        store.get_connection().execute("delete from accounts where account_id = %s", (account.account_id,))


def test_an_account_opens_in_one_click_without_a_key(client):
    r = client.post("/conversations", json={"language": "en", "account_id": "BF-1002", "demo_open": True})
    assert r.status_code == 200 and r.json()["account_id"] == "BF-1002"
    dash = client.get(f"/conversations/{r.json()['conversation_id']}/dashboard")
    assert dash.status_code == 200 and dash.json()["account"]["account_id"] == "BF-1002"
    assert client.post("/conversations", json={"language": "en", "account_id": "BF-9999", "demo_open": True}).status_code == 404


def test_without_demo_open_a_key_is_still_required(client):
    assert client.post("/conversations", json={"language": "en", "account_id": "BF-1002"}).status_code == 401


def test_a_real_deployment_turns_it_off(client, monkeypatch):
    monkeypatch.setenv("DEMO_OPEN_ACCESS", "0")
    assert client.get("/demo/accounts").status_code == 404
    assert client.post("/conversations", json={"language": "en", "account_id": "BF-1002", "demo_open": True}).status_code == 401
    assert client.post("/conversations", json={"language": "en", "account_id": "BF-1002", "access_key": "716044"}).status_code == 200
