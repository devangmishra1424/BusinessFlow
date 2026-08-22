"""Tests for the ops dashboard's HTTP API. Same convention as
test_browser_api.py: FastAPI's TestClient against the real app object,
real Postgres underneath via reseed_accounts -- no mocking of the store
or the flag logic, since the whole point is that these numbers match
what a human on the other end of a dashboard would actually see.

Every data endpoint requires OPS_API_KEY (an X-API-Key header) --
_auth() below is the one place that reads the real key, so a wrong-key
test can deliberately NOT use it.
"""

import os

import pytest
from fastapi.testclient import TestClient

from businessflow.ops.api import app

client = TestClient(app)

_pg_skip = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)
_ops_key_skip = pytest.mark.skipif(
    not os.environ.get("OPS_API_KEY"),
    reason="OPS_API_KEY not set -- copy .env.example to .env and fill it in",
)


def _auth() -> dict[str, str]:
    return {"X-API-Key": os.environ["OPS_API_KEY"]}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@_ops_key_skip
def test_accounts_endpoint_rejects_a_missing_api_key():
    response = client.get("/accounts")
    assert response.status_code == 401


@_ops_key_skip
def test_accounts_endpoint_rejects_a_wrong_api_key():
    response = client.get("/accounts", headers={"X-API-Key": "definitely-not-the-real-key"})
    assert response.status_code == 401


@_pg_skip
@_ops_key_skip
def test_list_accounts_returns_all_seeded_accounts_with_correct_flags(reseed_accounts):
    response = client.get("/accounts", headers=_auth())

    assert response.status_code == 200
    by_id = {a["account_id"]: a for a in response.json()}
    assert set(by_id) == {"BF-1001", "BF-1002", "BF-1003", "BF-1004"}

    assert [f["label"] for f in by_id["BF-1001"]["flags"]] == []
    assert {f["label"] for f in by_id["BF-1003"]["flags"]} == {"overdue", "disputed", "broken_promises"}


@_pg_skip
@_ops_key_skip
def test_list_accounts_filtered_by_flag_returns_only_matching_accounts(reseed_accounts):
    response = client.get("/accounts", params={"flag": "disputed"}, headers=_auth())

    assert response.status_code == 200
    account_ids = {a["account_id"] for a in response.json()}
    assert account_ids == {"BF-1003"}


@_pg_skip
@_ops_key_skip
def test_get_account_detail_includes_flags_promises_and_payment_history(reseed_accounts):
    response = client.get("/accounts/BF-1003", headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == "BF-1003"
    assert body["dispute_open"] is True
    assert {f["label"] for f in body["flags"]} == {"overdue", "disputed", "broken_promises"}
    assert len(body["promises"]) >= 2
    assert isinstance(body["payment_history"], list)


@_pg_skip
@_ops_key_skip
def test_get_account_detail_404s_for_unknown_account():
    response = client.get("/accounts/BF-9999", headers=_auth())

    assert response.status_code == 404


@_pg_skip
@_ops_key_skip
def test_list_open_escalations_reflects_a_real_new_escalation(reseed_accounts):
    from businessflow.accounts import store

    before = {e["escalation_id"] for e in client.get("/escalations", headers=_auth()).json()}

    escalation_id = store.create_escalation("BF-1003", "customer asked for a human directly")

    response = client.get("/escalations", headers=_auth())
    assert response.status_code == 200
    after = {e["escalation_id"] for e in response.json()}
    assert after - before == {escalation_id}
    new_entry = next(e for e in response.json() if e["escalation_id"] == escalation_id)
    assert new_entry["status"] == "queued_for_human"
    assert new_entry["reason"] == "customer asked for a human directly"


@_pg_skip
@_ops_key_skip
def test_metrics_endpoint_reflects_a_real_new_event(reseed_accounts):
    from businessflow.accounts import store

    store.log_event("BF-1001", "tool_called", {"tool": "get_payment_status"})

    response = client.get("/metrics", params={"since_hours": 1}, headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["event_counts"].get("tool_called", 0) >= 1
    assert 0.0 <= body["escalation_rate"] <= 1.0


@_ops_key_skip
def test_metrics_endpoint_requires_api_key():
    response = client.get("/metrics")
    assert response.status_code == 401
