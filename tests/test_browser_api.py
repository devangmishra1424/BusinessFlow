"""Tests for the browser channel's HTTP API. Uses FastAPI's TestClient
(in-process ASGI transport, not a mock of the app or its business logic)
against the real app object -- real Postgres underneath via
start_conversation_with_recap, same as everywhere else in this project.

The one thing that genuinely needs a live Groq call (an actual chat
reply) is GROQ_API_KEY-gated, same convention as test_agent_loop.py --
everything else here (health, conversation creation, error paths) needs
no LLM at all.
"""

import os

import pytest
from fastapi.testclient import TestClient

from businessflow.channels.browser_api import app

client = TestClient(app)

_pg_skip = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@_pg_skip
def test_start_conversation_returns_a_real_id_and_echoes_request(reseed_accounts):
    response = client.post("/conversations", json={"account_id": "BF-1001", "access_key": "482913", "language": "en"})

    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == "BF-1001"
    assert body["language"] == "en"
    assert body["conversation_id"]  # non-empty


@_pg_skip
def test_start_conversation_works_with_no_account_id():
    response = client.post("/conversations", json={"language": "en"})

    assert response.status_code == 200
    assert response.json()["account_id"] is None


@_pg_skip
def test_start_conversation_requires_access_key_for_an_account(reseed_accounts):
    response = client.post("/conversations", json={"account_id": "BF-1001", "language": "en"})

    assert response.status_code == 401


@_pg_skip
def test_start_conversation_rejects_wrong_access_key(reseed_accounts):
    response = client.post("/conversations", json={"account_id": "BF-1001", "access_key": "000000", "language": "en"})

    assert response.status_code == 401


def test_start_conversation_rejects_invalid_language():
    response = client.post("/conversations", json={"language": "fr"})

    assert response.status_code == 400
    assert "language" in response.json()["detail"]


def test_send_message_to_nonexistent_conversation_returns_404():
    response = client.post("/conversations/does-not-exist/messages", json={"message": "hello"})

    assert response.status_code == 404


@_pg_skip
def test_send_empty_message_returns_400(reseed_accounts):
    start = client.post("/conversations", json={"account_id": "BF-1001", "access_key": "482913", "language": "en"})
    conversation_id = start.json()["conversation_id"]

    response = client.post(f"/conversations/{conversation_id}/messages", json={"message": "   "})

    assert response.status_code == 400
