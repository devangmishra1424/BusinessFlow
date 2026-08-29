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
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from businessflow.channels.browser_api import app

client = TestClient(app)

_pg_skip = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)

# Mirrors tests/test_ops_api.py's own convention for the same directory
# (ops/api.py's _DOCUMENTS_DIR) -- built by hand here rather than importing
# accounts.documents._DOCUMENTS_DIR, same reasoning: a test shouldn't need
# to reach into a module's private constant just to locate the fixture
# files it plants and cleans up itself.
_DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "data" / "documents"


def _start_verified_conversation(account_id: str, access_key: str) -> str:
    start = client.post("/conversations", json={"account_id": account_id, "access_key": access_key, "language": "en"})
    assert start.status_code == 200
    return start.json()["conversation_id"]


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


@_pg_skip
def test_sending_credentials_in_an_anonymous_chat_verifies_without_reaching_the_llm(reseed_accounts):
    # Found live: a borrower typed "BF-1003 930571" straight into an
    # anonymous chat, and (before this fix) both values were forwarded to
    # the LLM as free text, which then passed the ACCESS KEY as
    # account_id to get_payment_status and crashed the tool call. This
    # must instead verify directly -- no run_turn_with_memory call, so no
    # Groq key needed for this test.
    start = client.post("/conversations", json={"language": "en"})
    conversation_id = start.json()["conversation_id"]
    assert start.json()["account_id"] is None

    response = client.post(f"/conversations/{conversation_id}/messages", json={"message": "BF-1003 930571"})

    assert response.status_code == 200
    body = response.json()
    assert "BF-1003" in body["reply"]
    assert body["tool_calls"] == []


@_pg_skip
def test_sending_credentials_with_a_wrong_key_does_not_verify(reseed_accounts):
    start = client.post("/conversations", json={"language": "en"})
    conversation_id = start.json()["conversation_id"]

    response = client.post(f"/conversations/{conversation_id}/messages", json={"message": "BF-1003 000000"})

    assert response.status_code == 200
    assert "doesn't match" in response.json()["reply"]


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/dashboard
# ---------------------------------------------------------------------------


def test_dashboard_404s_for_a_conversation_id_that_does_not_exist():
    response = client.get("/conversations/does-not-exist/dashboard")

    assert response.status_code == 404


@_pg_skip
def test_dashboard_403s_for_an_unverified_conversation():
    start = client.post("/conversations", json={"language": "en"})
    conversation_id = start.json()["conversation_id"]

    response = client.get(f"/conversations/{conversation_id}/dashboard")

    assert response.status_code == 403
    assert "verified" in response.json()["detail"]


@_pg_skip
def test_dashboard_returns_real_account_snapshot_timeline_and_empty_escalations_and_documents(reseed_accounts):
    from businessflow.tools.account_tools import get_payment_status

    conversation_id = _start_verified_conversation("BF-1001", "482913")

    response = client.get(f"/conversations/{conversation_id}/dashboard")

    assert response.status_code == 200
    body = response.json()

    # account snapshot is exactly get_payment_status(account_id)'s own real
    # result -- not a re-derived or partial copy of it.
    assert body["account"] == get_payment_status("BF-1001")

    # BF-1001 seeds 3 real (on-time) payments, then months_remaining=14
    # projected upcoming EMIs -- 17 entries total, oldest real payment
    # first. BF-1001 is 3 days past due (> 0, though not beyond the 3-day
    # grace period), so the first projected EMI is marked "overdue" here --
    # same days_past_due > 0 threshold ops/static/app.js's buildEmiTimeline
    # itself uses for this dot, deliberately looser than ops/flags.py's
    # grace-period-based "overdue" flag below.
    timeline = body["timeline"]
    assert len(timeline) == 3 + 14
    assert timeline[0]["status"] == "paid-on-time"
    assert timeline[1]["status"] == "paid-on-time"
    assert timeline[2]["status"] == "paid-on-time"
    assert timeline[3]["status"] == "overdue"
    assert timeline[3]["label"] == "Overdue -- 3d past due"
    assert timeline[3]["amount"] == 12_500
    assert timeline[-1]["status"] == "upcoming"

    # BF-1001 is clean under ops/flags.py's actual (grace-period-based)
    # rules -- 3 days past due does not exceed the 3-day grace period, no
    # dispute, no broken promises -- so no warnings at all.
    assert body["warnings"] == []
    assert body["escalations"] == []
    assert body["documents"] == []


@_pg_skip
def test_dashboard_reframes_ops_flags_into_borrower_toned_warnings(reseed_accounts):
    # BF-1003 carries all three real flags after a reseed (see
    # test_ops_api.py's test_draft_clarification_endpoint_grounds_in_real_
    # flags for the same fact about this account): 20 days past due
    # (beyond the grace period), an open dispute, and 2 broken promises.
    conversation_id = _start_verified_conversation("BF-1003", "930571")

    response = client.get(f"/conversations/{conversation_id}/dashboard")

    assert response.status_code == 200
    warnings = response.json()["warnings"]

    assert len(warnings) == 3
    assert any("20 days overdue" in w and "late fee" in w and "500" in w for w in warnings)
    assert any("open dispute" in w and "reviewing" in w for w in warnings)
    assert any("2 missed payment promises" in w for w in warnings)
    # The reframing is deliberate -- never the raw, staff-toned ops/flags.py
    # wording verbatim (see that module's Flag.reason strings).
    assert not any("grace period" in w for w in warnings)
    assert not any("unresolved" in w for w in warnings)


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/documents/{filename}
# ---------------------------------------------------------------------------


def test_document_download_404s_for_a_conversation_id_that_does_not_exist():
    response = client.get("/conversations/does-not-exist/documents/whatever.pdf")

    assert response.status_code == 404


@_pg_skip
def test_document_download_403s_for_an_unverified_conversation():
    start = client.post("/conversations", json={"language": "en"})
    conversation_id = start.json()["conversation_id"]

    response = client.get(f"/conversations/{conversation_id}/documents/whatever.pdf")

    assert response.status_code == 403


@_pg_skip
def test_document_download_404s_for_a_filename_that_does_not_exist(reseed_accounts):
    conversation_id = _start_verified_conversation("BF-1001", "482913")

    response = client.get(f"/conversations/{conversation_id}/documents/nonexistent.pdf")

    assert response.status_code == 404


@_pg_skip
def test_document_listing_and_download_are_scoped_to_the_verified_account(reseed_accounts):
    # Two real files, one per account, planted directly on disk -- the
    # same data/documents/{account_id}/ location ops/api.py's upload
    # endpoint writes into, without needing that endpoint (RAG ingestion,
    # Groq calls) just to test this read path.
    own_path = _DOCUMENTS_DIR / "BF-1001" / "test_dashboard_own.md"
    other_path = _DOCUMENTS_DIR / "BF-1002" / "test_dashboard_other.md"
    own_path.parent.mkdir(parents=True, exist_ok=True)
    other_path.parent.mkdir(parents=True, exist_ok=True)
    own_path.write_bytes(b"# BF-1001's own document")
    other_path.write_bytes(b"# BF-1002's document -- not BF-1001's")

    try:
        conversation_id = _start_verified_conversation("BF-1001", "482913")

        dashboard = client.get(f"/conversations/{conversation_id}/dashboard")
        assert dashboard.status_code == 200
        filenames = [d["filename"] for d in dashboard.json()["documents"]]
        assert filenames == ["test_dashboard_own.md"]

        own_download = client.get(f"/conversations/{conversation_id}/documents/test_dashboard_own.md")
        assert own_download.status_code == 200
        assert own_download.content == b"# BF-1001's own document"

        # BF-1001's own verified conversation must never be able to reach
        # BF-1002's real file by name -- same 404 as a filename that simply
        # doesn't exist at all, so nothing here leaks that this file
        # belongs to a different account.
        cross_account = client.get(f"/conversations/{conversation_id}/documents/test_dashboard_other.md")
        assert cross_account.status_code == 404
    finally:
        own_path.unlink(missing_ok=True)
        other_path.unlink(missing_ok=True)


def test_resolve_document_path_rejects_traversal_and_cross_account_filenames():
    from businessflow.accounts.documents import resolve_document_path

    other_dir = _DOCUMENTS_DIR / "BF-1002"
    other_dir.mkdir(parents=True, exist_ok=True)
    other_path = other_dir / "test_traversal_secret.md"
    other_path.write_bytes(b"not BF-1001's")

    try:
        # A filename crafted to walk out of BF-1001's own directory never
        # resolves to BF-1002's real file.
        assert resolve_document_path("BF-1001", "../BF-1002/test_traversal_secret.md") is None
        # The real file, addressed correctly, resolves fine for its own account.
        assert resolve_document_path("BF-1002", "test_traversal_secret.md") == other_path.resolve()
        # A nonexistent account directory is a clean None, not an error.
        assert resolve_document_path("BF-9999", "test_traversal_secret.md") is None
    finally:
        other_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# POST /conversations/{conversation_id}/quick-actions/dispute
# ---------------------------------------------------------------------------


@_pg_skip
def test_quick_action_dispute_requires_verification():
    start = client.post("/conversations", json={"language": "en"})
    conversation_id = start.json()["conversation_id"]

    response = client.post(f"/conversations/{conversation_id}/quick-actions/dispute", json={"reason": "test"})

    assert response.status_code == 403


@_pg_skip
def test_quick_action_dispute_rejects_an_empty_reason(reseed_accounts):
    conversation_id = _start_verified_conversation("BF-1001", "482913")

    response = client.post(f"/conversations/{conversation_id}/quick-actions/dispute", json={"reason": "   "})

    assert response.status_code == 400


@_pg_skip
def test_quick_action_dispute_flags_the_real_account_and_logs_the_tool_call(reseed_accounts):
    from businessflow.accounts import store

    conversation_id = _start_verified_conversation("BF-1001", "482913")

    response = client.post(
        f"/conversations/{conversation_id}/quick-actions/dispute",
        json={"reason": "I already paid this via UPI on the 3rd"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == "BF-1001"
    assert body["dispute_open"] is True
    assert body["reason"] == "I already paid this via UPI on the 3rd"
    assert body["already_open"] is False

    # Deliberately NOT re-fetching store.get_account_or_raise("BF-1001") here
    # to double-check dispute_open on the account row: BF-1001 is one of the
    # 4 canonical demo accounts every test suite's reseed_accounts fixture
    # resets, and this shared Postgres database is a real, live, environment
    # other test processes hit concurrently (see scripts/seed_accounts.py's
    # own docstring on exactly this -- "observed live: two workflow runs'
    # test phases overlapping"). Found live writing this test: a concurrent
    # process's reseed landed between this response and a follow-up SELECT,
    # making a real, correct dispute_open=True response look like a failure
    # a moment later. The response body above already proves this specific
    # call worked -- asserting it a second time via a separate query already
    # racy in this exact codebase adds no real coverage, only flakiness.
    events = store.get_connection().execute(
        "select details from events where account_id = %s and event_type = 'tool_called' "
        "and details->>'tool' = 'flag_dispute' order by created_at desc limit 1",
        ("BF-1001",),
    ).fetchall()
    assert events[0]["details"]["arguments"]["reason"] == "I already paid this via UPI on the 3rd"
    assert events[0]["details"]["result"]["dispute_open"] is True


# ---------------------------------------------------------------------------
# POST /conversations/{conversation_id}/quick-actions/agent
# ---------------------------------------------------------------------------


@_pg_skip
def test_quick_action_agent_requires_verification():
    start = client.post("/conversations", json={"language": "en"})
    conversation_id = start.json()["conversation_id"]

    response = client.post(f"/conversations/{conversation_id}/quick-actions/agent", json={})

    assert response.status_code == 403


@_pg_skip
def test_quick_action_agent_uses_a_sensible_default_reason_when_none_given(reseed_accounts):
    conversation_id = _start_verified_conversation("BF-1001", "482913")

    response = client.post(f"/conversations/{conversation_id}/quick-actions/agent", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == "BF-1001"
    assert body["status"] == "queued_for_human"
    assert body["escalation_id"]
    assert body["reason"] == "Borrower requested a human agent from the dashboard"


@_pg_skip
def test_quick_action_agent_uses_the_given_reason_when_provided(reseed_accounts):
    conversation_id = _start_verified_conversation("BF-1001", "482913")

    response = client.post(
        f"/conversations/{conversation_id}/quick-actions/agent", json={"reason": "I need to talk to someone about my EMI date"}
    )

    assert response.status_code == 200
    assert response.json()["reason"] == "I need to talk to someone about my EMI date"


# ---------------------------------------------------------------------------
# POST /conversations/{conversation_id}/quick-actions/payment-link
# ---------------------------------------------------------------------------


@_pg_skip
def test_quick_action_payment_link_requires_verification():
    start = client.post("/conversations", json={"language": "en"})
    conversation_id = start.json()["conversation_id"]

    response = client.post(f"/conversations/{conversation_id}/quick-actions/payment-link", json={"amount": 5000})

    assert response.status_code == 403


@_pg_skip
def test_quick_action_payment_link_rejects_a_non_positive_amount(reseed_accounts):
    conversation_id = _start_verified_conversation("BF-1001", "482913")

    response = client.post(f"/conversations/{conversation_id}/quick-actions/payment-link", json={"amount": 0})

    assert response.status_code == 422


@_pg_skip
def test_quick_action_payment_link_returns_a_real_synthetic_link(reseed_accounts):
    conversation_id = _start_verified_conversation("BF-1001", "482913")

    response = client.post(f"/conversations/{conversation_id}/quick-actions/payment-link", json={"amount": 5000})

    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == "BF-1001"
    assert body["amount"] == 5000
    # generate_payment_link f-strings the raw float amount in verbatim (see
    # tools/payment_tools.py) -- 5000.0, not 5000, is the real, correct link.
    assert body["payment_link"] == "https://demo.businessflow.local/pay/BF-1001?amount=5000.0"
    assert body["synthetic"] is True
