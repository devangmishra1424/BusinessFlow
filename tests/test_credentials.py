"""Tests for the Telegram deep-link payload encode/decode pair in
channels/credentials.py -- pure string logic, no DB/network needed.
(looks_like_credentials/parse_credentials are exercised indirectly
throughout test_telegram_channel.py/test_browser_api.py already.)
"""

from businessflow.channels.credentials import build_telegram_start_payload, parse_telegram_start_payload


def test_build_then_parse_round_trips():
    payload = build_telegram_start_payload("BF-1010", "482913")

    assert payload == "BF-1010_482913"
    assert parse_telegram_start_payload(payload) == ("BF-1010", "482913")


def test_parse_rejects_a_malformed_payload():
    assert parse_telegram_start_payload("not-a-real-payload") is None


def test_parse_rejects_a_short_access_key():
    # Payload charset alone can't tell a genuine 6-digit key from a typo --
    # the regex enforces exactly 6 digits, same as CREDENTIALS_PATTERN does
    # for typed credentials.
    assert parse_telegram_start_payload("BF-1010_1234") is None


def test_parse_rejects_empty_string():
    assert parse_telegram_start_payload("") is None


def test_parse_strips_surrounding_whitespace():
    assert parse_telegram_start_payload("  BF-1010_482913  ") == ("BF-1010", "482913")
