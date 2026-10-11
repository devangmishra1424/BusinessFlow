"""The shared "what I can help with" buttons (channels/abilities.py): who sees which entry, that both languages are complete,
and that the wording stays modest. No database, model or Telegram needed."""

import re

import pytest

from businessflow.channels.abilities import LANGUAGES, abilities_for, find_ability, intro
from businessflow.channels.credentials import looks_like_credentials

_ACCOUNT_ENTRIES = {"status", "pay", "history", "more_time", "dispute", "closure", "person"}
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Words that promise more than the assistant can: the list must read as "a few things", not "everything".
_OVERPROMISING = re.compile(r"\b(everything|anything|all your|any question|every question|24/7|instantly|immediately|guarantee|always)\b", re.I)


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_verified_chat_gets_the_account_entries_and_the_general_ones_and_no_how_to_verify(language):
    ids = [a["id"] for a in abilities_for(language, verified=True)]

    assert _ACCOUNT_ENTRIES <= set(ids)
    assert "policy" in ids and "more" in ids
    assert "verify" not in ids  # already verified


@pytest.mark.parametrize("language", LANGUAGES)
def test_an_unverified_chat_never_gets_a_button_that_would_end_in_a_refusal(language):
    # The assistant refuses account tools for an unverified chat (PR #6), so offering them would be offering a dead end.
    ids = [a["id"] for a in abilities_for(language, verified=False)]

    assert ids == ["policy", "verify", "more"]
    assert not (_ACCOUNT_ENTRIES & set(ids))


@pytest.mark.parametrize("verified", [True, False])
@pytest.mark.parametrize("language", LANGUAGES)
def test_more_questions_is_always_the_last_button(language, verified):
    entries = abilities_for(language, verified)

    assert entries[-1]["id"] == "more"
    assert entries[-1]["reply"] and not entries[-1]["prompt"]  # a nudge to type, not a question put to the assistant


@pytest.mark.parametrize("verified", [True, False])
@pytest.mark.parametrize("language", LANGUAGES)
def test_every_entry_is_complete_exactly_one_of_prompt_or_reply_and_ids_are_unique(language, verified):
    entries = abilities_for(language, verified)

    assert len({a["id"] for a in entries}) == len(entries)
    for a in entries:
        assert a["label"].strip()
        assert bool(a["prompt"]) != bool(a["reply"]), a["id"]
        assert len(a["label"]) <= 60  # fits a button on a phone


def test_hindi_is_really_hindi_and_english_is_not():
    for a in abilities_for("hi", verified=True) + abilities_for("hi", verified=False):
        assert _DEVANAGARI.search(a["label"]), a["id"]
        assert _DEVANAGARI.search(a["prompt"] or a["reply"]), a["id"]
    assert _DEVANAGARI.search(intro("hi"))
    for a in abilities_for("en", verified=True):
        assert not _DEVANAGARI.search(a["label"]), a["id"]


def test_the_wording_does_not_overpromise():
    for language in LANGUAGES:
        for verified in (True, False):
            texts = [intro(language)] + [t for a in abilities_for(language, verified) for t in (a["label"], a["prompt"], a["reply"]) if t]
            for text in texts:
                assert not _OVERPROMISING.search(text), text
    # and the way it says the list is not the limit is a promise to say so, not to be able to
    assert "I'll say so" in find_ability("en", True, "more")["reply"]


def test_a_prompt_is_never_mistaken_for_credentials():
    # A tapped prompt goes through the same path as typed text, where anything shaped like "BF-1001 482913" is treated as a login.
    for language in LANGUAGES:
        for a in abilities_for(language, verified=True):
            if a["prompt"]:
                assert not looks_like_credentials(a["prompt"]), a["prompt"]
                assert len(a["prompt"]) <= 200


def test_find_ability_only_finds_what_that_chat_may_see():
    assert find_ability("en", True, "status")["prompt"]
    assert find_ability("en", False, "status") is None  # an account button left over from before a reset
    assert find_ability("en", True, "verify") is None  # already verified
    assert find_ability("en", False, "verify")["reply"]
    assert find_ability("en", True, "no-such-entry") is None
    assert find_ability("en", True, "") is None


def test_an_unknown_language_falls_back_to_english_rather_than_failing():
    assert abilities_for("fr", True) == abilities_for("en", True)
    assert intro("fr") == intro("en")
