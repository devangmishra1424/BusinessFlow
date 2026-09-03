"""Tests for the A/B prompt-routing logic. The pure hashing/bucketing
behavior needs no DB (get_active_variant is monkeypatched); the real
Postgres read path (an actual row in prompt_versions) is _pg_skip-gated,
same convention as every other real-DB test in this project.
"""

import os
from datetime import datetime, timezone

import psycopg
import pytest

from businessflow.accounts import store
from businessflow.agent import prompt_versions
from businessflow.agent.loop import start_conversation
from businessflow.agent.prompt_versions import BASELINE_VERSION_ID, choose_prompt_version, get_active_variant

_pg_skip = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_choose_prompt_version_falls_back_to_baseline_with_no_active_variant(monkeypatch):
    monkeypatch.setattr(prompt_versions, "get_active_variant", lambda: None)

    version_id, template = choose_prompt_version("BF-1001")

    assert version_id == BASELINE_VERSION_ID
    assert template is None


def test_choose_prompt_version_at_100_percent_rollout_always_picks_the_variant(monkeypatch):
    monkeypatch.setattr(prompt_versions, "get_active_variant", lambda: {
        "version_id": "v2", "system_prompt_template": "alt template {account_context}", "rollout_percent": 100,
    })

    for bucket_key in ("BF-1001", "BF-1002", "some-random-uuid"):
        version_id, template = choose_prompt_version(bucket_key)
        assert version_id == "v2"
        assert template == "alt template {account_context}"


def test_choose_prompt_version_at_0_percent_rollout_always_picks_baseline(monkeypatch):
    monkeypatch.setattr(prompt_versions, "get_active_variant", lambda: {
        "version_id": "v2", "system_prompt_template": "alt template", "rollout_percent": 0,
    })

    version_id, template = choose_prompt_version("BF-1001")

    assert version_id == BASELINE_VERSION_ID
    assert template is None


def test_choose_prompt_version_is_deterministic_for_the_same_bucket_key(monkeypatch):
    # The same borrower must land in the same arm every time, not get
    # bounced between variants across separate conversations.
    monkeypatch.setattr(prompt_versions, "get_active_variant", lambda: {
        "version_id": "v2", "system_prompt_template": "alt", "rollout_percent": 50,
    })

    first = choose_prompt_version("BF-1001")
    second = choose_prompt_version("BF-1001")

    assert first == second


def test_get_active_variant_returns_none_when_database_url_is_not_configured(monkeypatch):
    def _raise_no_database_url():
        raise RuntimeError("DATABASE_URL is not set -- copy .env.example to .env and fill it in")

    monkeypatch.setattr(prompt_versions, "get_connection", _raise_no_database_url)

    assert get_active_variant() is None


def test_get_active_variant_returns_none_when_the_table_is_not_migrated_yet(monkeypatch):
    class _FakeConnection:
        def execute(self, *a, **kw):
            raise psycopg.errors.UndefinedTable("relation \"prompt_versions\" does not exist")

    monkeypatch.setattr(prompt_versions, "get_connection", lambda: _FakeConnection())

    assert get_active_variant() is None


@pytest.fixture
def active_variant_row():
    """A real, temporary prompt_versions row -- inserted and torn down
    around one test, never left behind for a real deployment to pick up
    by accident."""
    version_id = "test-variant-v1"
    store.get_connection().execute(
        "insert into prompt_versions (version_id, description, system_prompt_template, rollout_percent, active) "
        "values (%s, %s, %s, %s, %s)",
        (version_id, "test variant", "alt template {account_context}{language_instruction}", 100, True),
    )
    try:
        yield version_id
    finally:
        store.get_connection().execute("delete from prompt_versions where version_id = %s", (version_id,))


@_pg_skip
def test_get_active_variant_reads_a_real_active_row(active_variant_row):
    variant = get_active_variant()

    assert variant is not None
    assert variant["version_id"] == active_variant_row
    assert variant["rollout_percent"] == 100


@_pg_skip
def test_get_active_variant_ignores_an_inactive_row(active_variant_row):
    store.get_connection().execute(
        "update prompt_versions set active = false where version_id = %s", (active_variant_row,)
    )

    assert get_active_variant() is None


@_pg_skip
def test_start_conversation_uses_the_variant_template_and_logs_it_at_100_percent_rollout(active_variant_row, reseed_accounts):
    since = datetime.now(timezone.utc)

    conversation = start_conversation(language="en", account_id="BF-1001")

    assert conversation[0]["content"].startswith("alt template")
    row = store.get_connection().execute(
        "select details from events where account_id = %s and event_type = 'prompt_version_assigned' "
        "and created_at >= %s order by created_at desc limit 1",
        ("BF-1001", since),
    ).fetchone()
    assert row is not None
    assert row["details"]["version_id"] == active_variant_row


@_pg_skip
def test_start_conversation_uses_the_baseline_and_logs_nothing_with_no_active_variant(reseed_accounts):
    since = datetime.now(timezone.utc)

    conversation = start_conversation(language="en", account_id="BF-1001")

    assert not conversation[0]["content"].startswith("alt template")
    row = store.get_connection().execute(
        "select details from events where account_id = %s and event_type = 'prompt_version_assigned' "
        "and created_at >= %s order by created_at desc limit 1",
        ("BF-1001", since),
    ).fetchone()
    assert row is None
