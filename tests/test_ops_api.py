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
_groq_skip = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set -- copy .env.example to .env and fill it in to run this",
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
def test_create_account_endpoint_opens_a_real_new_account(reseed_accounts):
    # New accounts aren't part of reseed_accounts' BF-1001..1004 reset, so
    # this cleans up the row it creates itself rather than leaking a
    # permanent extra account into the real demo database.
    from businessflow.accounts import store

    payload = {
        "borrower_name": "Test Borrower",
        "business_name": "Test Business",
        "phone_number": "+919800011122",
        "language_preference": "en",
        "loan_type": "Working Capital Loan",
        "principal_amount": 100_000,
        "emi_amount": 8_500,
        "tenure_months": 12,
        "emi_due_date": "2026-09-15",
        "nach_mandate_active": True,
        "risk_tier": "low",
    }
    response = client.post("/accounts", json=payload, headers=_auth())
    account_id = response.json()["account"]["account_id"] if response.status_code == 201 else None
    try:
        assert response.status_code == 201
        body = response.json()
        assert body["account"]["borrower_name"] == "Test Borrower"
        assert body["account"]["flags"] == []
        assert len(body["access_key"]) == 6 and body["access_key"].isdigit()

        detail = client.get(f"/accounts/{account_id}", headers=_auth())
        assert detail.status_code == 200
        assert detail.json()["months_remaining"] == 12  # nothing paid down yet
        assert detail.json()["payment_history"] == []
    finally:
        if account_id:
            store.get_connection().execute("delete from accounts where account_id = %s", (account_id,))


@_ops_key_skip
def test_create_account_endpoint_rejects_an_invalid_phone_number():
    payload = {
        "borrower_name": "Test Borrower", "business_name": "Test Business", "phone_number": "9800011122",
        "language_preference": "en", "loan_type": "Working Capital Loan", "principal_amount": 100_000,
        "emi_amount": 8_500, "tenure_months": 12, "emi_due_date": "2026-09-15",
    }
    response = client.post("/accounts", json=payload, headers=_auth())
    assert response.status_code == 422


@_ops_key_skip
def test_create_account_endpoint_rejects_a_missing_api_key():
    response = client.post("/accounts", json={})
    assert response.status_code == 401


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

    # The "disputed" flag's reason is the real dispute's own text, not a
    # generic placeholder -- and that same real reason is independently
    # visible via the new disputes list, not just embedded in the flag.
    assert len(body["disputes"]) >= 1
    real_reason = body["disputes"][0]["reason"]
    disputed_flag_reason = next(f["reason"] for f in body["flags"] if f["label"] == "disputed")
    assert disputed_flag_reason == real_reason
    assert disputed_flag_reason != "has an open, unresolved dispute"


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
def test_approve_escalation_endpoint_applies_real_changes_and_notifies(reseed_accounts):
    # BF-1001 has no linked telegram_chat_id after a reseed -- so this
    # exercises the real logged-fallback path in outbound/send.py, not a
    # real network call to Telegram (there's nothing to send to).
    from businessflow.accounts import store
    from businessflow.tools.escalation_tools import propose_restructuring

    proposal = propose_restructuring(account_id="BF-1001", extra_months=3)

    response = client.post(f"/escalations/{proposal['escalation_id']}/approve", headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["resolved_at"] is not None

    account = store.get_account_or_raise("BF-1001")
    assert account.months_remaining == 17
    assert account.emi_amount == 10294.12

    events = store.get_connection().execute(
        "select details from events where account_id = %s and event_type = 'restructuring_decision_notified' "
        "order by created_at desc limit 1",
        ("BF-1001",),
    ).fetchone()
    assert events["details"]["approved"] is True
    assert events["details"]["delivered_via_telegram"] is False


@_pg_skip
@_ops_key_skip
def test_approve_escalation_endpoint_closes_a_plain_escalation_instead_of_500ing(reseed_accounts):
    # Regression test for a real bug: escalate_to_human (the vast majority
    # of real escalations -- an open dispute, repeated broken promises, or
    # the agent just being unsure) creates one with no proposed_changes.
    # Clicking Approve on one of these in the ops dashboard used to raise
    # an unhandled ValueError -> a plain 500 with no useful detail, on
    # ordinary use, not an edge case.
    from businessflow.accounts import store
    from businessflow.tools.escalation_tools import escalate_to_human

    escalation = escalate_to_human(account_id="BF-1001", reason="Borrower has a general question, unsure how to help.")
    before = store.get_account_or_raise("BF-1001")

    response = client.post(f"/escalations/{escalation['escalation_id']}/approve", headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["resolved_at"] is not None

    after = store.get_account_or_raise("BF-1001")
    assert after.months_remaining == before.months_remaining
    assert after.emi_amount == before.emi_amount


@_pg_skip
@_ops_key_skip
def test_reject_escalation_endpoint_with_an_optional_reason(reseed_accounts):
    from businessflow.accounts import store
    from businessflow.tools.escalation_tools import propose_restructuring

    proposal = propose_restructuring(account_id="BF-1001", extra_months=3)

    response = client.post(
        f"/escalations/{proposal['escalation_id']}/reject",
        headers=_auth(),
        json={"reason": "Borrower already 2 EMIs behind"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["resolution_reason"] == "Borrower already 2 EMIs behind"

    # Nothing was ever applied to the real account.
    account = store.get_account_or_raise("BF-1001")
    assert account.months_remaining == 14
    assert account.emi_amount == 12500


@_pg_skip
@_ops_key_skip
def test_reject_escalation_endpoint_reason_is_optional(reseed_accounts):
    from businessflow.tools.escalation_tools import propose_restructuring

    proposal = propose_restructuring(account_id="BF-1001", extra_months=3)

    response = client.post(f"/escalations/{proposal['escalation_id']}/reject", headers=_auth(), json={})

    assert response.status_code == 200
    assert response.json()["resolution_reason"] is None


@_pg_skip
@_ops_key_skip
def test_approve_escalation_endpoint_404s_for_an_unknown_escalation(reseed_accounts):
    response = client.post("/escalations/ESC-9999999/approve", headers=_auth())
    assert response.status_code == 404


@_pg_skip
@_ops_key_skip
def test_approve_escalation_endpoint_409s_for_an_already_resolved_escalation(reseed_accounts):
    from businessflow.tools.escalation_tools import propose_restructuring

    proposal = propose_restructuring(account_id="BF-1001", extra_months=3)
    client.post(f"/escalations/{proposal['escalation_id']}/approve", headers=_auth())

    response = client.post(f"/escalations/{proposal['escalation_id']}/approve", headers=_auth())
    assert response.status_code == 409


@_ops_key_skip
def test_approve_escalation_endpoint_requires_api_key():
    response = client.post("/escalations/ESC-0001/approve")
    assert response.status_code == 401


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


@_ops_key_skip
def test_upload_document_endpoint_rejects_a_missing_api_key():
    response = client.post(
        "/accounts/BF-1001/documents",
        files={"file": ("agreement.md", b"# loan agreement", "text/markdown")},
        data={"document_type": "loan_agreement"},
    )
    assert response.status_code == 401


@_pg_skip
@_ops_key_skip
@_groq_skip
def test_draft_clarification_endpoint_grounds_in_real_flags(reseed_accounts):
    # BF-1003 carries real overdue/disputed/broken_promises flags after a
    # reseed -- the draft should reflect that context, not the operator's
    # note alone, and never invent an amount/date beyond what's given.
    response = client.post(
        "/accounts/BF-1003/clarification-requests/draft",
        json={"operator_note": "Third missed promise this quarter, dispute still unresolved."},
        headers=_auth(),
    )

    assert response.status_code == 200
    draft = response.json()["draft"]
    assert isinstance(draft, str)
    assert len(draft.strip()) > 0


@_ops_key_skip
def test_draft_clarification_endpoint_404s_for_unknown_account():
    response = client.post(
        "/accounts/BF-9999/clarification-requests/draft",
        json={"operator_note": "test"},
        headers=_auth(),
    )
    assert response.status_code == 404


@_pg_skip
@_ops_key_skip
def test_send_clarification_request_endpoint_logs_and_reports_delivery(reseed_accounts):
    # BF-1001 has no linked telegram_chat_id after a reseed -- same real
    # logged-fallback path as test_approve_escalation_endpoint_applies_
    # real_changes_and_notifies, not a real network call to Telegram.
    from businessflow.accounts import store

    response = client.post(
        "/accounts/BF-1001/clarification-requests",
        json={"message": "Please contact us about your overdue payment at your earliest convenience."},
        headers=_auth(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["delivered_via_telegram"] is False
    assert body["message"] == "Please contact us about your overdue payment at your earliest convenience."
    assert body["created_at"] is not None

    events = store.get_connection().execute(
        "select details from events where account_id = %s and event_type = 'clarification_request_sent' "
        "order by created_at desc limit 1",
        ("BF-1001",),
    ).fetchall()
    assert len(events) == 1
    assert events[0]["details"]["delivered_via_telegram"] is False


@_ops_key_skip
def test_send_clarification_request_endpoint_404s_for_unknown_account():
    response = client.post(
        "/accounts/BF-9999/clarification-requests",
        json={"message": "test"},
        headers=_auth(),
    )
    assert response.status_code == 404


@_ops_key_skip
def test_send_clarification_request_endpoint_requires_api_key():
    response = client.post("/accounts/BF-1001/clarification-requests", json={"message": "test"})
    assert response.status_code == 401


@_pg_skip
@_ops_key_skip
def test_account_detail_includes_clarification_request_history(reseed_accounts):
    send_response = client.post(
        "/accounts/BF-1002/clarification-requests",
        json={"message": "Your last two EMIs were late -- please reach out."},
        headers=_auth(),
    )
    assert send_response.status_code == 200

    detail = client.get("/accounts/BF-1002", headers=_auth()).json()

    assert len(detail["clarification_requests"]) == 1
    assert detail["clarification_requests"][0]["message"] == "Your last two EMIs were late -- please reach out."
    assert detail["clarification_requests"][0]["delivered_via_telegram"] is False


@_pg_skip
@_ops_key_skip
def test_upload_document_404s_for_unknown_account():
    response = client.post(
        "/accounts/BF-9999/documents",
        files={"file": ("agreement.md", b"# loan agreement", "text/markdown")},
        data={"document_type": "loan_agreement"},
        headers=_auth(),
    )
    assert response.status_code == 404


@_pg_skip
@_ops_key_skip
def test_upload_document_rejects_an_unsupported_extension(reseed_accounts):
    response = client.post(
        "/accounts/BF-1001/documents",
        files={"file": ("agreement.exe", b"whatever", "application/octet-stream")},
        data={"document_type": "loan_agreement"},
        headers=_auth(),
    )
    assert response.status_code == 400


@_pg_skip
@_ops_key_skip
def test_upload_document_rejects_a_file_over_the_size_cap(reseed_accounts):
    from pathlib import Path

    from businessflow.ops.api import _MAX_UPLOAD_BYTES

    oversized_body = b"x" * (_MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/accounts/BF-1001/documents",
        files={"file": ("big.md", oversized_body, "text/markdown")},
        data={"document_type": "loan_agreement"},
        headers=_auth(),
    )
    assert response.status_code == 413

    saved_path = Path(__file__).resolve().parents[1] / "data" / "documents" / "BF-1001" / "big.md"
    assert not saved_path.exists()
    # A rejected upload shouldn't leave an empty account directory behind
    # either -- only true when this was the account's only upload attempt,
    # which it is here since reseed_accounts doesn't touch the filesystem.
    assert not saved_path.parent.exists()


@_pg_skip
@_ops_key_skip
def test_upload_document_rejects_an_unparseable_file_with_a_valid_extension(reseed_accounts):
    # A ".pdf" extension that passes the allow-list check but isn't
    # actually a parseable PDF (corrupt upload, wrong bytes under a
    # trusted-looking name) must surface as a clean 422, not an unhandled
    # 500 -- docling.exceptions.ConversionError is what DocumentConverter
    # actually raises for this (verified live against the installed
    # docling package by feeding it exactly this kind of garbage file).
    from pathlib import Path

    garbage_body = b"this is not a real pdf file, just garbage bytes 1234567890"
    response = client.post(
        "/accounts/BF-1001/documents",
        files={"file": ("garbage.pdf", garbage_body, "application/pdf")},
        data={"document_type": "loan_agreement"},
        headers=_auth(),
    )
    assert response.status_code == 422

    saved_path = Path(__file__).resolve().parents[1] / "data" / "documents" / "BF-1001" / "garbage.pdf"
    assert not saved_path.exists()
    # A rejected upload shouldn't leave an empty account directory behind
    # either -- only true when this was the account's only upload attempt,
    # which it is here since reseed_accounts doesn't touch the filesystem.
    assert not saved_path.parent.exists()


@_pg_skip
@_ops_key_skip
def test_upload_document_ingests_into_the_account_scoped_rag_store(reseed_accounts):
    from pathlib import Path

    from businessflow.rag.retriever import DocumentRetriever
    from businessflow.rag.store import delete_chunks_for_document

    saved_path = Path(__file__).resolve().parents[1] / "data" / "documents" / "BF-1001" / "test_agreement.md"
    body = (
        b"# BF-1001 loan agreement\n\n"
        b"A special one-time relocation waiver clause applies to this specific loan."
    )
    try:
        response = client.post(
            "/accounts/BF-1001/documents",
            files={"file": ("test_agreement.md", body, "text/markdown")},
            data={"document_type": "loan_agreement"},
            headers=_auth(),
        )

        assert response.status_code == 200
        out = response.json()
        assert out["account_id"] == "BF-1001"
        assert out["document_type"] == "loan_agreement"
        assert out["filename"] == "test_agreement.md"
        assert out["chunks_stored"] >= 1

        assert saved_path.read_bytes() == body

        own_results = DocumentRetriever().retrieve("relocation waiver clause", top_k=1, account_id="BF-1001")
        assert own_results and "relocation waiver" in own_results[0]["text"].lower()

        other_results = DocumentRetriever().retrieve("relocation waiver clause", top_k=1, account_id="BF-1002")
        assert not any("relocation waiver" in r["text"].lower() for r in other_results)
    finally:
        delete_chunks_for_document(str(saved_path))
        saved_path.unlink(missing_ok=True)


@_pg_skip
@_ops_key_skip
def test_upload_non_loan_agreement_document_never_attempts_rate_extraction(reseed_accounts):
    # A KYC (or any non-loan_agreement) upload must not trigger a Groq
    # call at all -- interest_rate_extracted stays False and the
    # account's interest_rate_pct is untouched, deterministically, with
    # no GROQ_API_KEY needed to verify it.
    from pathlib import Path

    from businessflow.accounts import store

    saved_path = Path(__file__).resolve().parents[1] / "data" / "documents" / "BF-1002" / "test_kyc.md"
    try:
        response = client.post(
            "/accounts/BF-1002/documents",
            files={"file": ("test_kyc.md", b"# KYC document\n\nPAN and Aadhaar on file.", "text/markdown")},
            data={"document_type": "kyc"},
            headers=_auth(),
        )

        assert response.status_code == 200
        assert response.json()["interest_rate_extracted"] is False
        assert store.get_account_or_raise("BF-1002").interest_rate_pct is None
    finally:
        from businessflow.rag.store import delete_chunks_for_document

        delete_chunks_for_document(str(saved_path))
        saved_path.unlink(missing_ok=True)


@_pg_skip
@_ops_key_skip
def test_list_documents_returns_uploaded_files_newest_first(reseed_accounts):
    from pathlib import Path

    from businessflow.rag.store import delete_chunks_for_document

    account_dir = Path(__file__).resolve().parents[1] / "data" / "documents" / "BF-1001"
    older_path = account_dir / "test_older.md"
    newer_path = account_dir / "test_newer.md"
    try:
        older = client.post(
            "/accounts/BF-1001/documents",
            files={"file": ("test_older.md", b"# older doc", "text/markdown")},
            data={"document_type": "kyc"},
            headers=_auth(),
        )
        assert older.status_code == 200
        newer = client.post(
            "/accounts/BF-1001/documents",
            files={"file": ("test_newer.md", b"# newer doc", "text/markdown")},
            data={"document_type": "kyc"},
            headers=_auth(),
        )
        assert newer.status_code == 200

        response = client.get("/accounts/BF-1001/documents", headers=_auth())

        assert response.status_code == 200
        body = response.json()
        names = [d["filename"] for d in body]
        assert "test_older.md" in names and "test_newer.md" in names
        # Newest first -- the just-uploaded newer file must sort ahead of
        # the one uploaded just before it.
        assert names.index("test_newer.md") < names.index("test_older.md")
        newer_entry = next(d for d in body if d["filename"] == "test_newer.md")
        assert newer_entry["size_bytes"] == len(b"# newer doc")
        assert newer_entry["uploaded_at"] is not None
    finally:
        for path in (older_path, newer_path):
            delete_chunks_for_document(str(path))
            path.unlink(missing_ok=True)


@_pg_skip
@_ops_key_skip
def test_list_documents_returns_empty_list_for_account_with_no_uploads(reseed_accounts):
    response = client.get("/accounts/BF-1004/documents", headers=_auth())
    assert response.status_code == 200
    assert response.json() == []


@_ops_key_skip
def test_list_documents_404s_for_unknown_account():
    response = client.get("/accounts/BF-9999/documents", headers=_auth())
    assert response.status_code == 404


@_ops_key_skip
def test_list_documents_rejects_a_missing_api_key():
    response = client.get("/accounts/BF-1001/documents")
    assert response.status_code == 401


@_pg_skip
@_ops_key_skip
def test_download_document_serves_the_real_uploaded_bytes(reseed_accounts):
    from pathlib import Path

    from businessflow.rag.store import delete_chunks_for_document

    saved_path = Path(__file__).resolve().parents[1] / "data" / "documents" / "BF-1001" / "test_download.md"
    body = b"# a document ops uploaded and should be able to download back"
    try:
        upload = client.post(
            "/accounts/BF-1001/documents",
            files={"file": ("test_download.md", body, "text/markdown")},
            data={"document_type": "kyc"},
            headers=_auth(),
        )
        assert upload.status_code == 200

        response = client.get("/accounts/BF-1001/documents/test_download.md", headers=_auth())

        assert response.status_code == 200
        assert response.content == body
        assert "test_download.md" in response.headers.get("content-disposition", "")
    finally:
        delete_chunks_for_document(str(saved_path))
        saved_path.unlink(missing_ok=True)


@_pg_skip
@_ops_key_skip
def test_download_document_404s_for_a_filename_that_was_never_uploaded(reseed_accounts):
    response = client.get("/accounts/BF-1001/documents/never_uploaded.md", headers=_auth())
    assert response.status_code == 404


@_ops_key_skip
def test_download_document_404s_for_unknown_account():
    response = client.get("/accounts/BF-9999/documents/whatever.md", headers=_auth())
    assert response.status_code == 404


@_pg_skip
@_ops_key_skip
def test_download_document_rejects_path_traversal_outside_the_account_directory(reseed_accounts):
    # A literal ".." survives Path(...).name (unlike "../../x", whose
    # .name is "x") -- this is exactly the case the resolve()+
    # is_relative_to() confinement check exists to catch, not the plain
    # .name strip alone. %2e%2e is used (rather than a bare "..") because
    # a bare ".." segment gets collapsed by URL normalization before the
    # request is even sent, never reaching the route at all -- the
    # percent-encoded form is what a real client sends and is what
    # arrives at the handler as the literal string "..".
    response = client.get("/accounts/BF-1001/documents/%2e%2e", headers=_auth())
    assert response.status_code == 404


@_ops_key_skip
def test_download_document_rejects_a_missing_api_key():
    response = client.get("/accounts/BF-1001/documents/whatever.md")
    assert response.status_code == 401


@_pg_skip
@_ops_key_skip
@_groq_skip
def test_upload_loan_agreement_with_a_stated_rate_extracts_and_persists_interest_rate_pct(reseed_accounts):
    from pathlib import Path

    from businessflow.accounts import store
    from businessflow.rag.store import delete_chunks_for_document

    # BF-1002, not BF-1001 -- test_account_tools.py separately asserts
    # BF-1001's interest_rate_pct is null, and reseed_accounts' upsert
    # (scripts/seed_accounts.py) does NOT reset this column on conflict,
    # so a value this test writes to an account would otherwise outlive
    # reseeding. Using a different account plus the explicit reset in
    # `finally` below keeps this test from leaking state into that one
    # regardless of file/test run order.
    saved_path = Path(__file__).resolve().parents[1] / "data" / "documents" / "BF-1002" / "test_rate_agreement.md"
    body = (
        b"# BF-1002 loan agreement\n\n"
        b"This term loan carries an interest rate of 14.5% per annum, "
        b"charged on the outstanding principal alongside the monthly EMI."
    )
    try:
        response = client.post(
            "/accounts/BF-1002/documents",
            files={"file": ("test_rate_agreement.md", body, "text/markdown")},
            data={"document_type": "loan_agreement"},
            headers=_auth(),
        )

        assert response.status_code == 200
        out = response.json()
        assert out["interest_rate_extracted"] is True

        account = store.get_account_or_raise("BF-1002")
        assert account.interest_rate_pct is not None
        assert abs(account.interest_rate_pct - 14.5) < 0.5
    finally:
        delete_chunks_for_document(str(saved_path))
        saved_path.unlink(missing_ok=True)
        store.get_connection().execute(
            "update accounts set interest_rate_pct = null where account_id = %s", ("BF-1002",)
        )
