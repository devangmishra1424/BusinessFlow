"""Tests for the tappable actions under a proactive Telegram reminder.

channels/reminder_actions.py is deliberately free of torch / telegram-bot / database imports, so the
logic runs here with plain stand-ins for the few things it touches (ActionDeps): no network, no
Postgres, no mocking framework. The real tools it calls (log_promise_to_pay, escalate_to_human) have
their own tests; what is checked here is the routing, the trust check and the wording.
"""

from dataclasses import dataclass
from datetime import date

import pytest

from businessflow.channels import reminder_actions as ra
from businessflow.outbound.send import _build_reply_markup

TODAY = date(2026, 10, 10)


@dataclass
class FakeAccount:
    account_id: str = "BF-1001"
    telegram_chat_id: int | None = 555
    language_preference: str = "en"


class Recorder:
    """Stand-ins for the tools, recording what the action did."""

    def __init__(self, account=None, amount=12500.0, already_logged=False):
        self.account = account
        self.amount = amount
        self.already_logged = already_logged
        self.promises, self.escalations, self.events = [], [], []

    def deps(self) -> ra.ActionDeps:
        return ra.ActionDeps(
            get_account=lambda account_id: self.account if self.account and self.account.account_id == account_id else None,
            today=lambda: TODAY,
            amount_due=lambda account, today: self.amount,
            log_promise=self._log_promise,
            escalate=self._escalate,
            log_event=lambda account_id, kind, details: self.events.append((account_id, kind, details)),
        )

    def _log_promise(self, account_id, promised_date, amount):
        self.promises.append((account_id, promised_date, amount))
        return {"logged": True, "already_logged": self.already_logged}

    def _escalate(self, account_id, reason):
        self.escalations.append((account_id, reason))
        return {"escalation_id": "ESC-42"}


def test_callback_data_round_trips_and_stays_inside_telegrams_limit():
    for days in ra.PROMISE_DAYS:
        data = ra.encode("ptp", "BF-1001", days)
        assert ra.decode(data) == ra.ActionRequest("ptp", "BF-1001", days)
        assert len(data.encode()) <= 64
    assert ra.decode(ra.encode("human", "BF-1001")) == ra.ActionRequest("human", "BF-1001")
    assert ra.decode(ra.encode("dispute", "BF-1001")) == ra.ActionRequest("dispute", "BF-1001")


def test_decode_refuses_anything_it_did_not_issue():
    for bad in ["", "menu:status", "lang:hi", "rem", "rem:ptp", "rem:ptp:BF-1001", "rem:ptp:BF-1001:2", "rem:ptp:BF-1001:abc",
                "rem:ptp:BF-1001:3:9", "rem:human:BF-1001:3", "rem:refund:BF-1001", "rem:ptp:../etc:3", "rem:ptp:" + "A" * 40 + ":3"]:
        assert ra.decode(bad) is None, bad


def test_an_oversized_callback_is_refused_at_build_time_not_by_telegram():
    with pytest.raises(ValueError):
        ra.encode("ptp", "A" * 80, 3)


def test_action_rows_cover_the_three_promises_then_person_and_dispute():
    rows = ra.action_rows("BF-1001", "en")
    assert [len(r) for r in rows] == [3, 2]
    assert [data for _, data in rows[0]] == ["rem:ptp:BF-1001:1", "rem:ptp:BF-1001:3", "rem:ptp:BF-1001:7"]
    assert [data for _, data in rows[1]] == ["rem:human:BF-1001", "rem:dispute:BF-1001"]


def test_labels_follow_the_borrowers_language_and_hinglish_runs_in_english():
    assert ra.action_rows("BF-1001", "hi")[0][0][0] == "📅 कल भुगतान होगा"
    assert ra.action_rows("BF-1001", "hinglish") == ra.action_rows("BF-1001", "en")
    assert ra.action_rows("BF-1001", None) == ra.action_rows("BF-1001", "en")


def test_the_reminder_keyboard_puts_the_pay_link_first_then_the_actions():
    markup = _build_reply_markup("https://example.test/pay/abc", 12500.0, ra.action_rows("BF-1001", "en"))
    rows = markup.inline_keyboard
    assert len(rows) == 3
    assert rows[0][0].url == "https://example.test/pay/abc" and "12,500" in rows[0][0].text
    assert all(b.callback_data and not b.url for row in rows[1:] for b in row)
    # the rows the bot keeps after a tap are exactly the URL ones
    assert [row for row in rows if all(b.url for b in row)] == [rows[0]]


def test_the_keyboard_without_a_pay_link_or_actions_is_none():
    assert _build_reply_markup(None, None, None) is None
    only_actions = _build_reply_markup(None, None, ra.action_rows("BF-1001", "en"))
    assert len(only_actions.inline_keyboard) == 2


def test_promise_to_pay_logs_the_amount_due_for_the_tapped_date():
    r = Recorder(FakeAccount())
    result = ra.run_action(555, "rem:ptp:BF-1001:3", r.deps())
    assert r.promises == [("BF-1001", "2026-10-13", 12500.0)]
    assert result.done is True
    assert "₹12,500.00" in result.text and "13 Oct" in result.text
    assert r.events[0][1] == "tool_called" and r.events[0][2]["tool"] == "log_promise_to_pay"


def test_each_button_means_its_own_number_of_days():
    for days, expected in [(1, "2026-10-11"), (3, "2026-10-13"), (7, "2026-10-17")]:
        r = Recorder(FakeAccount())
        ra.run_action(555, f"rem:ptp:BF-1001:{days}", r.deps())
        assert r.promises[0][1] == expected


def test_a_second_tap_on_the_same_promise_says_it_is_already_on_record():
    r = Recorder(FakeAccount(), already_logged=True)
    result = ra.run_action(555, "rem:ptp:BF-1001:1", r.deps())
    assert result.text.startswith("That is already on record")


def test_nothing_is_logged_when_nothing_is_due():
    r = Recorder(FakeAccount(), amount=0.0)
    result = ra.run_action(555, "rem:ptp:BF-1001:1", r.deps())
    assert r.promises == [] and "Nothing is due" in result.text


def test_a_tap_from_any_chat_other_than_the_linked_one_does_nothing():
    for chat in (556, 0):
        r = Recorder(FakeAccount(telegram_chat_id=555))
        result = ra.run_action(chat, "rem:ptp:BF-1001:3", r.deps())
        assert (r.promises, r.escalations, r.events) == ([], [], [])
        assert result.done is False and "isn't linked to your account" in result.text


def test_a_button_for_an_unknown_account_or_an_unlinked_one_does_nothing():
    r = Recorder(None)
    assert "isn't linked" in ra.run_action(555, "rem:human:BF-9999", r.deps()).text
    r = Recorder(FakeAccount(telegram_chat_id=None))
    assert "isn't linked" in ra.run_action(555, "rem:human:BF-1001", r.deps()).text
    assert r.escalations == []


def test_talk_to_a_person_opens_one_escalation_with_its_reference():
    r = Recorder(FakeAccount())
    result = ra.run_action(555, "rem:human:BF-1001", r.deps())
    assert len(r.escalations) == 1 and "reminder" in r.escalations[0][1]
    assert "ESC-42" in result.text and result.done is True


def test_dispute_only_explains_how_to_describe_it_and_changes_nothing():
    r = Recorder(FakeAccount())
    result = ra.run_action(555, "rem:dispute:BF-1001", r.deps())
    assert "/dispute" in result.text and result.done is False
    assert (r.promises, r.escalations, r.events) == ([], [], [])


def test_replies_use_the_borrowers_language():
    r = Recorder(FakeAccount(language_preference="hi"))
    result = ra.run_action(555, "rem:ptp:BF-1001:1", r.deps())
    assert "दर्ज" in result.text and "₹12,500.00" in result.text


def test_data_that_is_not_a_reminder_action_is_left_for_the_caller():
    r = Recorder(FakeAccount())
    assert ra.run_action(555, "menu:status", r.deps()) is None
    assert ra.run_action(555, "lang:hi", r.deps()) is None
    assert ra.run_action(555, "rem:ptp:BF-1001:2", r.deps()) is None
