"""The living demo's reset (POST /demo/reset): starts scripts/build_demo.py in the background, at most one at a time
and once per cooldown, only when typed out, and not at all when the deployment turns it off. GET /demo says when the
rebuild is done, or why it failed.

The rebuild itself enters history through the running apps over HTTP, so here the process is a fake; what it does is
covered by tests/test_demo_build.py and by running it against real servers.
"""

import os
import sys

import pytest

from businessflow.accounts import store
from businessflow.ops import api as ops_api


class FakeBuild:
    """Stands in for the subprocess: running until finish() is called, then exits with the given code."""

    started: list["FakeBuild"] = []

    def __init__(self, args, cwd, env, stdout, stderr):
        self.args, self.env, self.pid, self.code = args, env, 4242, None
        FakeBuild.started.append(self)

    def poll(self):
        return self.code

    def finish(self, code, last_line=""):
        self.code = code
        ops_api._DEMO_BUILD_LOG.write_text(f"starting\n{last_line}\n", encoding="utf-8")


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    FakeBuild.started.clear()
    monkeypatch.setattr(ops_api.subprocess, "Popen", FakeBuild)
    monkeypatch.setattr(ops_api, "_DEMO_BUILD_LOG", tmp_path / "build.log")
    monkeypatch.setattr(ops_api, "_demo_build", None)
    store.get_connection().execute("delete from events where event_type = 'demo_reset'")
    yield TestClient(ops_api.app)
    store.get_connection().execute("delete from events where event_type = 'demo_reset'")


def _headers():
    return {"X-API-Key": os.environ["OPS_API_KEY"]}


def _reset(client, confirm="reset"):
    return client.post("/demo/reset", headers=_headers(), json={"confirm": confirm})


def test_a_reset_starts_the_rebuild_and_a_second_one_waits_for_it(client):
    r = _reset(client)
    assert r.status_code == 202 and r.json()["rebuilding"] is True
    [build] = FakeBuild.started
    assert build.args == [sys.executable, "-m", "scripts.build_demo"]
    assert build.env["DEMO_OPS_URL"] and build.env["DEMO_CHAT_URL"]  # configured, never from the request

    assert _reset(client).status_code == 409
    build.finish(0)
    assert client.get("/demo", headers=_headers()).json()["rebuilding"] is False


def test_a_failed_rebuild_says_why_until_the_next_one(client):
    _reset(client)
    FakeBuild.started[0].finish(1, "build_demo the demo build stopped: upload BFF-WC: 422 unreadable")
    status = client.get("/demo", headers=_headers()).json()
    assert status["rebuilding"] is False and "422 unreadable" in status["last_failure"]


def test_a_finished_reset_waits_out_the_cooldown(client):
    store.log_event(None, "demo_reset", {})
    r = _reset(client)
    assert r.status_code == 429 and "minute" in r.json()["detail"]
    assert FakeBuild.started == []


def test_a_reset_must_be_typed_out(client):
    assert _reset(client, confirm="yes").status_code == 422


def test_a_deployment_can_turn_the_reset_off(client, monkeypatch):
    monkeypatch.setenv("DEMO_RESET_ENABLED", "0")
    assert client.get("/demo", headers=_headers()).json()["enabled"] is False
    assert _reset(client).status_code == 404
