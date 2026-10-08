"""/health must tell the truth about the database.

Found live: a free-tier Supabase project paused while the VM was off. Every
database-backed request then hung ~36 s and returned a 500, while /health --
which only proved the process was alive -- kept answering 200 "ok". These
tests pin the fix on both services (chat API and ops API) and on the check
itself, with no database needed: the unreachable-database case is exercised
against a real closed local port, not a fake.
"""

import time

import psycopg
import pytest
from fastapi.testclient import TestClient

from businessflow.accounts.db import check_database
from businessflow.channels import browser_api
from businessflow.ops import api as ops_api

_SERVICES = [("chat API", browser_api), ("ops API", ops_api)]


def _unreachable(*_args, **_kwargs):
    raise psycopg.OperationalError("connection to server at db.secret-host.example failed: password for user postgres.secretref")


@pytest.mark.parametrize("name, module", _SERVICES)
def test_health_is_503_not_200_when_the_database_is_unreachable(name, module, monkeypatch):
    monkeypatch.setattr(module, "check_database", _unreachable)

    response = TestClient(module.app).get("/health")

    assert response.status_code == 503, name
    assert response.json() == {"status": "degraded", "database": "unreachable"}


@pytest.mark.parametrize("name, module", _SERVICES)
def test_health_never_leaks_the_error_text_host_or_user(name, module, monkeypatch):
    monkeypatch.setattr(module, "check_database", _unreachable)

    body = TestClient(module.app).get("/health").text

    assert "secret" not in body and "postgres." not in body and "password" not in body, name


@pytest.mark.parametrize("name, module", _SERVICES)
def test_health_is_ok_when_the_check_passes(name, module, monkeypatch):
    monkeypatch.setattr(module, "check_database", lambda *a, **k: None)

    response = TestClient(module.app).get("/health")

    assert response.status_code == 200, name
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_reports_a_missing_database_url_as_degraded_not_a_crash(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    response = TestClient(browser_api.app).get("/health")

    assert response.status_code == 503


def test_check_database_fails_fast_against_a_real_closed_port(monkeypatch):
    # Port 1 on localhost: nothing listens, the connection is refused at once.
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/db")

    started = time.monotonic()
    with pytest.raises(psycopg.OperationalError):
        check_database(timeout_seconds=2)

    assert time.monotonic() - started < 4  # seconds, not the pool's 30 s


def test_check_database_raises_a_clear_error_when_the_url_is_not_configured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        check_database()
