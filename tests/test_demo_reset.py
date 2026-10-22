"""The living demo's reset (POST /demo/reset): puts the demo borrowers back to their starting story, at most once per
cooldown, only when typed out, and not at all when the deployment turns it off.
"""

import os

import pytest

from businessflow.accounts import store


@pytest.fixture
def client(reseed_accounts):
    from fastapi.testclient import TestClient

    from businessflow.ops.api import app

    store.get_connection().execute("delete from events where event_type = 'demo_reset'")
    yield TestClient(app)
    store.get_connection().execute("delete from events where event_type = 'demo_reset'")


def _headers():
    return {"X-API-Key": os.environ["OPS_API_KEY"]}


def test_a_reset_undoes_what_was_done_on_a_demo_account_and_then_waits(client):
    before = len(store.get_account_or_raise("BF-1001").payment_history)
    store.record_payment("BF-1001", 12_500, apply_extra_to_next=False)
    assert len(store.get_account_or_raise("BF-1001").payment_history) == before + 1

    r = client.post("/demo/reset", headers=_headers(), json={"confirm": "reset"})

    assert r.status_code == 200, r.text
    assert r.json()["last_reset_at"] is not None
    assert len(store.get_account_or_raise("BF-1001").payment_history) == before
    again = client.post("/demo/reset", headers=_headers(), json={"confirm": "reset"})
    assert again.status_code == 429 and "minute" in again.json()["detail"]


def test_a_reset_must_be_typed_out(client):
    assert client.post("/demo/reset", headers=_headers(), json={"confirm": "yes"}).status_code == 422


def test_a_deployment_can_turn_the_reset_off(client, monkeypatch):
    monkeypatch.setenv("DEMO_RESET_ENABLED", "0")
    assert client.get("/demo", headers=_headers()).json()["enabled"] is False
    assert client.post("/demo/reset", headers=_headers(), json={"confirm": "reset"}).status_code == 404
