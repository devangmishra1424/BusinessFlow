"""Tests for _create_completion's retry loop in agent/loop.py -- the
actual mechanism this whole session's Groq-quota debugging exercised
live. Mocks only the true external boundary (groq.Groq itself, the
network-touching class) so the real fallback-key state machine in
client.py and the real retry loop in loop.py both run unmodified; that's
different from mocking away the agent's own reasoning, which the rest
of this project deliberately never does.
"""

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
    monkeypatch.setattr(client_module, "Groq", lambda api_key: _FakeGroq(api_key, fails_for))

    result = loop_module._create_completion()

    assert result == "success with key2"
    assert client_module._current_key_index == 2  # advanced through both failing keys


def test_create_completion_raises_once_every_configured_key_is_rate_limited(reset_fallback_state, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "key1")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY2", raising=False)
    client_module._current_key_index = 0

    fails_for = {"key0", "key1"}  # every configured key fails -- nowhere left to go
    monkeypatch.setattr(client_module, "Groq", lambda api_key: _FakeGroq(api_key, fails_for))

    with pytest.raises(RateLimitError):
        loop_module._create_completion()


def test_create_completion_succeeds_immediately_when_the_primary_key_works(reset_fallback_state, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key0")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    client_module._current_key_index = 0

    monkeypatch.setattr(client_module, "Groq", lambda api_key: _FakeGroq(api_key, frozenset()))

    result = loop_module._create_completion()

    assert result == "success with key0"
    assert client_module._current_key_index == 0  # never had to switch
