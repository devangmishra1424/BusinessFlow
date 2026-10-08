"""Tests for _create_completion's retry loop in agent/loop.py -- the
actual mechanism this whole session's Groq-quota debugging exercised
live. Mocks only the true external boundary (groq.Groq itself, the
network-touching class) so the real fallback-key state machine in
client.py and the real retry loop in loop.py both run unmodified; that's
different from mocking away the agent's own reasoning, which the rest
of this project deliberately never does.
"""

import types

import groq
import httpx
import pytest

import businessflow.agent.client as client_module
import businessflow.agent.loop as loop_module
from groq import RateLimitError


class _FakeChat:
    def __init__(self, api_key: str, fails_for_keys: set[str]):
        self.completions = _FakeCompletionsWithKey(api_key, fails_for_keys)


class _FakeCompletionsWithKey:
    def __init__(self, api_key: str, fails_for_keys: set[str]):
        self._api_key = api_key
        self._fails_for_keys = fails_for_keys

    def create(self, **kwargs):
        if self._api_key in self._fails_for_keys:
            response = httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.groq.com/x"))
            raise RateLimitError("rate limited", response=response, body=None)
        return f"success with {self._api_key}"


class _FakeGroq:
    """Stands in for groq.Groq -- the real external boundary. Its
    behavior depends only on the api_key it was constructed with, same
    as the real thing would (a rate-limited key fails regardless of
    which logical "slot" client.py currently thinks is active)."""

    def __init__(self, api_key: str, _fails_for_keys: set[str] = frozenset()):
        self.api_key = api_key
        self.chat = _FakeChat(api_key, _fails_for_keys)


@pytest.fixture
def reset_fallback_state(monkeypatch):
    # Clear every real ALTERNATE_GROQ_KEY{N} this process's actual .env
    # may have loaded first -- otherwise a test expecting "no more
    # fallbacks configured" would see whatever real ones happen to be
    # set right now instead of the clean slate it sets up explicitly.
    for n in range(2, client_module._MAX_FALLBACK_KEY_SUFFIX + 1):
        monkeypatch.delenv(f"ALTERNATE_GROQ_KEY{n}", raising=False)
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)

    original_index = client_module._current_key_index
    original_switched_at = client_module._switched_at
    yield
    client_module._current_key_index = original_index
    client_module._switched_at = original_switched_at


def test_create_completion_advances_through_two_rate_limited_keys_to_a_working_third(
    reset_fallback_state, monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "key1")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY2", "key2")
    client_module._current_key_index = 0

    fails_for = {"key0", "key1"}  # key2 is the only one that actually works
    monkeypatch.setattr(client_module, "Groq", lambda api_key, **kwargs: _FakeGroq(api_key, fails_for))

    result = loop_module._create_completion()

    assert result == "success with key2"
    assert client_module._current_key_index == 2  # advanced through both failing keys


def test_create_completion_raises_once_every_configured_key_is_rate_limited(reset_fallback_state, monkeypatch, recorded_sleeps):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "key1")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY2", raising=False)
    client_module._current_key_index = 0

    fails_for = {"key0", "key1"}  # every configured key fails, permanently -- nowhere left to go
    monkeypatch.setattr(client_module, "Groq", lambda api_key, **kwargs: _FakeGroq(api_key, fails_for))

    with pytest.raises(RateLimitError):
        loop_module._create_completion()

    # It waited the bounded number of rounds before giving up, not forever.
    assert len(recorded_sleeps) == loop_module._RATE_LIMIT_WAIT_ROUNDS


def test_create_completion_succeeds_immediately_when_the_primary_key_works(reset_fallback_state, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    client_module._current_key_index = 0

    monkeypatch.setattr(client_module, "Groq", lambda api_key, **kwargs: _FakeGroq(api_key, frozenset()))

    result = loop_module._create_completion()

    assert result == "success with key0"
    assert client_module._current_key_index == 0  # never had to switch


# --- fail-fast rate-limit rotation, and bounded transient retries ----------
#
# Root cause of the ~51 s tool-calling turns: the SDK's default retries
# SLEEP for the server's Retry-After on a 429 before the rotation above ever
# runs. The agent loop now builds its client with max_retries=0, so a 429
# rotates to the next key at once; these tests pin that and the bounded
# retry that replaces the SDK's handling of genuinely transient errors.


class _ScriptedGroq:
    """A Groq stand-in whose completions.create raises the next scripted
    exception (or returns "ok" when the script is exhausted)."""

    def __init__(self, api_key: str, script: list, constructor_kwargs: dict):
        self.api_key = api_key
        self.constructor_kwargs = constructor_kwargs
        self.chat = type("Chat", (), {"completions": self})()
        self._script = script

    def create(self, **kwargs):
        if self._script:
            raise self._script.pop(0)
        return "ok"


def _transient_error():
    return groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))


def _rate_limit_error():
    response = httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.groq.com/x"))
    return RateLimitError("rate limited", response=response, body=None)


@pytest.fixture
def recorded_sleeps(monkeypatch):
    """Replaces loop's `time` with one whose sleep records instead of
    sleeping, so retry backoff is observable without slowing the suite."""
    sleeps: list[float] = []
    monkeypatch.setattr(loop_module, "time", types.SimpleNamespace(sleep=sleeps.append))
    return sleeps


def test_agent_loop_builds_its_client_fail_fast(reset_fallback_state, monkeypatch, recorded_sleeps):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    client_module._current_key_index = 0
    seen = []
    monkeypatch.setattr(
        client_module, "Groq",
        lambda api_key, **kwargs: seen.append(kwargs) or _ScriptedGroq(api_key, [], kwargs),
    )

    loop_module._create_completion()

    assert seen[0]["max_retries"] == 0
    assert seen[0]["timeout"] == client_module._FAIL_FAST_TIMEOUT_SECONDS


def test_a_rate_limit_rotates_to_the_next_key_without_sleeping(reset_fallback_state, monkeypatch, recorded_sleeps):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "key1")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY2", raising=False)
    client_module._current_key_index = 0
    scripts = {"key0": [_rate_limit_error()], "key1": []}
    monkeypatch.setattr(
        client_module, "Groq",
        lambda api_key, **kwargs: _ScriptedGroq(api_key, scripts[api_key], kwargs),
    )

    result = loop_module._create_completion()

    assert result == "ok"
    assert client_module._current_key_index == 1  # rotated to key1 straight away
    assert recorded_sleeps == []  # the whole point: no waiting out a Retry-After


def _rate_limit_error_with_retry_after(seconds: str):
    response = httpx.Response(
        status_code=429, headers={"retry-after": seconds}, request=httpx.Request("POST", "https://api.groq.com/x"),
    )
    return RateLimitError("rate limited", response=response, body=None)


def test_when_every_key_is_momentarily_empty_it_waits_then_succeeds_instead_of_erroring(
    reset_fallback_state, monkeypatch, recorded_sleeps,
):
    # The saturated case that fail-fast alone got wrong: all keys out of
    # tokens at the same moment. Rotating through them all costs nothing;
    # then it waits out the server's own Retry-After and tries again.
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "key1")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY2", raising=False)
    client_module._current_key_index = 0
    scripts = {"key0": [_rate_limit_error_with_retry_after("7")], "key1": [_rate_limit_error_with_retry_after("7")]}
    monkeypatch.setattr(
        client_module, "Groq",
        lambda api_key, **kwargs: _ScriptedGroq(api_key, scripts[api_key], kwargs),
    )

    assert loop_module._create_completion() == "ok"
    assert recorded_sleeps == [7.0]  # exactly the server's Retry-After, once
    assert client_module._current_key_index == 0  # back on the primary after the wait


def test_the_wait_is_capped_so_a_huge_retry_after_cannot_hang_a_caller(reset_fallback_state, monkeypatch, recorded_sleeps):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    client_module._current_key_index = 0
    script = [_rate_limit_error_with_retry_after("3600")]  # a daily-limit style hint
    monkeypatch.setattr(client_module, "Groq", lambda api_key, **kwargs: _ScriptedGroq(api_key, script, kwargs))

    assert loop_module._create_completion() == "ok"
    assert recorded_sleeps == [loop_module._RATE_LIMIT_MAX_WAIT_SECONDS]


def test_a_missing_or_garbled_retry_after_falls_back_to_a_default_wait(reset_fallback_state, monkeypatch, recorded_sleeps):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    client_module._current_key_index = 0
    script = [_rate_limit_error(), _rate_limit_error_with_retry_after("soon")]  # no header, then a non-number
    monkeypatch.setattr(client_module, "Groq", lambda api_key, **kwargs: _ScriptedGroq(api_key, script, kwargs))

    assert loop_module._create_completion() == "ok"
    assert recorded_sleeps == [loop_module._RATE_LIMIT_DEFAULT_WAIT_SECONDS] * 2


def test_a_transient_error_is_retried_a_bounded_number_of_times_then_succeeds(
    reset_fallback_state, monkeypatch, recorded_sleeps,
):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    client_module._current_key_index = 0
    script = [_transient_error(), _transient_error()]  # two failures, then success
    monkeypatch.setattr(client_module, "Groq", lambda api_key, **kwargs: _ScriptedGroq(api_key, script, kwargs))

    assert loop_module._create_completion() == "ok"
    assert recorded_sleeps == [0.5, 1.0]  # short, growing backoff -- never a long wait


def test_a_persistent_transient_error_is_raised_after_the_retry_budget(reset_fallback_state, monkeypatch, recorded_sleeps):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    client_module._current_key_index = 0
    script = [_transient_error() for _ in range(10)]  # never recovers
    monkeypatch.setattr(client_module, "Groq", lambda api_key, **kwargs: _ScriptedGroq(api_key, script, kwargs))

    with pytest.raises(groq.APIConnectionError):
        loop_module._create_completion()

    assert len(recorded_sleeps) == loop_module._TRANSIENT_ERROR_RETRIES  # bounded, not infinite
    assert client_module._current_key_index == 0  # a connection error is not a reason to burn through keys
