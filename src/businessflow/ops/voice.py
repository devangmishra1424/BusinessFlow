"""Understanding a spoken (or typed) request to the ops dashboard: one language-model call turns the operator's words
into ONE call from a fixed menu of dashboard abilities (MENU below) -- a question to answer, a page to show, or an
action to propose -- with its details filled in (which borrower, how much, which date).

That is all the model does. It never sees account data, and nothing it writes reaches the screen as an answer:
ops/voice_reply.py looks the answer up in the database and writes the sentence, and turns an action into a
confirmation card. The one place the model's own figures matter -- an amount or phone number to act on, or a figure
in a message it drafted -- is checked against what the operator actually said (ops/spoken_numbers.py) before
anything is proposed. And nothing here or in voice_reply.py writes to the database: an action only happens when the
operator clicks Confirm, which calls the same endpoint the dashboard's own form for it does.

Model: OPS_VOICE_MODEL, default openai/gpt-oss-120b. Groq's daily token allowance is per model, and the borrower
chat runs on openai/gpt-oss-20b (agent/client.py), so a busy morning of ops questions cannot use up the allowance the
borrowers' chat depends on. The call goes through agent/loop.py's _create_completion, so it gets the same key
rotation, bounded rate-limit waits and malformed-tool-call retries as the borrower chat.
"""

import json
import logging
import os
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from businessflow.agent.loop import _create_completion

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_REQUEST_CHARS = 600  # 30 s of speech is roughly 450 characters; a typed question is shorter still

AccountRef = str  # an account ID, a name as spoken, or "current" (resolved by voice_reply.resolve_account)
Group = Literal["all", "overdue", "disputed", "broken_promises", "not_on_telegram"]
CaseKind = Literal["any", "payment_claim", "dispute", "fraud", "request", "restructuring", "callback", "closure", "system"]


class _Args(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def _blank_means_not_given(cls, data):
        # A model often sends "" or null for a detail the operator didn't give; both mean "not said", which each
        # ability then asks about, rather than failing validation as a bad date or amount.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None and v != ""}
        return data


class NoArgs(_Args):
    pass


class ShowAccountsArgs(_Args):
    group: Group = "all"
    sort: Literal["most_overdue", "highest_risk", "account_id"] = "most_overdue"
    search: str | None = Field(default=None, max_length=80)


class ShowInboxArgs(_Args):
    kind: CaseKind = "any"


class EscalationRateArgs(_Args):
    window: Literal["24h", "3d", "7d", "30d"] = "24h"


class AccountDetailsArgs(_Args):
    account: AccountRef = "current"
    topic: Literal["summary", "dues", "payments", "promises", "cases", "calls", "contact", "loan", "documents", "conversation"] = "summary"


class OpenPageArgs(_Args):
    page: Literal["portfolio", "inbox", "new_account"]


class CannotHelpArgs(_Args):
    reason: str | None = None  # read by the model as part of choosing; never shown or logged (it can quote the request)


class _AmountArgs(_Args):
    account: AccountRef = "current"
    amount: float | None = Field(default=None, gt=0)
    full_amount_due: bool = False


class RecordPaymentArgs(_AmountArgs):
    paid_on: date | None = None
    credit_toward_next: bool | None = None


class ConfirmPaymentClaimArgs(RecordPaymentArgs):
    note: str | None = Field(default=None, max_length=500)
    case: str | None = None  # set by a "which claim?" choice, not by the model


class CloseCaseArgs(_Args):
    account: AccountRef = "current"
    note: str | None = Field(default=None, max_length=500)
    case: str | None = None


class LogPromiseArgs(_AmountArgs):
    promised_date: date | None = None


class LogCallArgs(_Args):
    account: AccountRef = "current"
    outcome: Literal["reached", "no_answer", "voicemail", "wrong_number"]
    note: str | None = Field(default=None, max_length=500)
    # "Reached, she'll pay 8,000 by Friday" is one call and one promise: both go on one card (voice_reply._log_call).
    promised_amount: float | None = Field(default=None, gt=0)
    promised_full_amount: bool = False
    promised_date: date | None = None


class MessageBorrowerArgs(_Args):
    account: AccountRef = "current"
    message: str = Field(min_length=1, max_length=1000)


class MessageGroupArgs(_Args):
    group: Literal["overdue", "disputed", "broken_promises", "not_on_telegram"]
    message: str = Field(min_length=1, max_length=1000)


class ReplyToCaseArgs(MessageBorrowerArgs):
    case: str | None = None


class ApproveRequestArgs(_Args):
    account: AccountRef = "current"
    case: str | None = None


class RejectRequestArgs(ApproveRequestArgs):
    reason: str | None = Field(default=None, max_length=500)


class HandToPersonArgs(_Args):
    account: AccountRef = "current"
    reason: str = Field(min_length=1, max_length=500)


class SendRemindersArgs(_Args):
    account: AccountRef | None = None
    group: Literal["all", "overdue", "disputed", "broken_promises"] | None = None


class UpdateAccountArgs(_Args):
    account: AccountRef = "current"
    phone_number: str | None = Field(default=None, max_length=20)
    language: Literal["hi", "en", "hinglish"] | None = None
    risk_tier: Literal["low", "medium", "high"] | None = None


class AccountOnlyArgs(_Args):
    account: AccountRef = "current"


class StartCallListArgs(_Args):
    group: Literal["overdue", "broken_promises"] = "overdue"


class StartInboxArgs(_Args):
    kind: CaseKind = "any"


class WorkflowControlArgs(_Args):
    command: Literal["next", "skip", "previous", "stop", "repeat"]


class CaseStepArgs(_Args):
    account: AccountRef
    case: str


# Read-only: answered at once. Everything else is an action and becomes a confirmation card. A workflow
# (voice_workflows.py) only reads too: it walks a queue, and each action in it is a card like any other.
QUESTION_TOOLS = {
    "portfolio_overview", "show_accounts", "show_inbox", "escalation_rate", "account_details", "open_page", "help",
    "cannot_help", "daily_briefing", "start_call_list", "start_inbox", "workflow_control", "call_step", "case_step",
}
# Steps of a workflow the page asks for itself, one queue item at a time -- never chosen by the model, so not on MENU.
PAGE_ONLY_TOOLS = {"call_step", "case_step"}

ARGS: dict[str, type[_Args]] = {
    "portfolio_overview": NoArgs,
    "show_accounts": ShowAccountsArgs,
    "show_inbox": ShowInboxArgs,
    "escalation_rate": EscalationRateArgs,
    "account_details": AccountDetailsArgs,
    "open_page": OpenPageArgs,
    "help": NoArgs,
    "cannot_help": CannotHelpArgs,
    "record_payment": RecordPaymentArgs,
    "confirm_payment_claim": ConfirmPaymentClaimArgs,
    "close_case": CloseCaseArgs,
    "log_promise": LogPromiseArgs,
    "log_call": LogCallArgs,
    "message_borrower": MessageBorrowerArgs,
    "message_group": MessageGroupArgs,
    "reply_to_case": ReplyToCaseArgs,
    "approve_request": ApproveRequestArgs,
    "reject_request": RejectRequestArgs,
    "hand_to_person": HandToPersonArgs,
    "send_reminders": SendRemindersArgs,
    "update_account": UpdateAccountArgs,
    "mark_messages_handled": AccountOnlyArgs,
    "daily_briefing": NoArgs,
    "start_call_list": StartCallListArgs,
    "start_inbox": StartInboxArgs,
    "workflow_control": WorkflowControlArgs,
    "call_step": AccountOnlyArgs,
    "case_step": CaseStepArgs,
}

_ACCOUNT = {
    "type": "string",
    "description": "Account ID (e.g. BF-1003) or the borrower's or business's name as the operator said it, in "
    "English letters even if said in Hindi; 'current' for the account open on screen (this one / her / him / them).",
}
_AMOUNT = {"type": "number", "description": "Rupees, only if the operator said a figure. Never guess one."}
_FULL = {"type": "boolean", "description": "True if they said the full amount / what's due / their EMI instead of a figure."}
_GROUP = ["all", "overdue", "disputed", "broken_promises", "not_on_telegram"]
_TEXT_FOR_BORROWER = (
    "Written to the borrower in plain, polite English ('we', not 'I'), saying only what the operator asked to say. "
    "No figure, date or promise the operator did not say."
)


def _tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
        },
    }


MENU = [
    _tool("portfolio_overview", "Headline numbers for the whole portfolio: accounts, how many need attention, overdue "
          "amount, open cases, risk mix."),
    _tool("show_accounts", "Count or list accounts, optionally one group, sorted, or searched by name or business. "
          "E.g. who is overdue, who to call first, high-risk accounts, who isn't on Telegram, find Sharma.", {
              "group": {"type": "string", "enum": _GROUP},
              "sort": {"type": "string", "enum": ["most_overdue", "highest_risk", "account_id"]},
              "search": {"type": "string", "description": "Name, business or ID text to filter by."},
          }),
    _tool("show_inbox", "Open cases waiting on staff -- payment claims ('I already paid'), disputes, requests for "
          "more time, call-backs, closure certificates, fraud claims, system notices: counts, oldest, stale ones.", {
              "kind": {"type": "string", "enum": list(CaseKind.__args__)},
          }),
    _tool("escalation_rate", "Share of the AI assistant's actions handed to a person, over a time window.", {
        "window": {"type": "string", "enum": ["24h", "3d", "7d", "30d"]},
    }),
    _tool("account_details", "Facts about ONE account (opens its profile).", {
        "account": _ACCOUNT,
        "topic": {"type": "string", "enum": list(AccountDetailsArgs.model_fields["topic"].annotation.__args__),
                  "description": "summary = overall status; dues = what is owed now; loan = principal, EMI, tenure."},
    }, ["account"]),
    _tool("open_page", "Just go to a page.", {"page": {"type": "string", "enum": ["portfolio", "inbox", "new_account"]}},
          ["page"]),
    _tool("help", "What this assistant can do."),
    _tool("cannot_help", "Use when no other function fits: unclear, outside this dashboard, or something these "
          "functions can't do (e.g. opening an account, resetting an access key, uploading a document).",
          {"reason": {"type": "string"}}),
    _tool("record_payment", "Record a payment the borrower made (cash, UPI, NEFT, cheque...).", {
        "account": _ACCOUNT, "amount": _AMOUNT, "full_amount_due": _FULL,
        "paid_on": {"type": "string", "description": "YYYY-MM-DD; omit for today."},
        "credit_toward_next": {"type": "boolean",
                               "description": "Only if they said whether a short payment counts toward the next EMI."},
    }, ["account"]),
    _tool("confirm_payment_claim", "Confirm the borrower's open 'I already paid' claim: records the payment and closes "
          "the claim.", {
              "account": _ACCOUNT, "amount": _AMOUNT, "full_amount_due": _FULL,
              "paid_on": {"type": "string", "description": "YYYY-MM-DD; omit for today."},
              "note": {"type": "string", "description": "Optional note to the borrower. " + _TEXT_FOR_BORROWER},
          }, ["account"]),
    _tool("close_case", "Close the account's open dispute or payment claim WITHOUT recording a payment.", {
        "account": _ACCOUNT,
        "note": {"type": "string", "description": "Why / how it was resolved; sent to the borrower. " + _TEXT_FOR_BORROWER},
    }, ["account"]),
    _tool("log_promise", "Log a promise to pay the borrower made (e.g. on a call).", {
        "account": _ACCOUNT, "amount": _AMOUNT, "full_amount_due": _FULL,
        "promised_date": {"type": "string", "description": "YYYY-MM-DD they promised to pay by."},
    }, ["account"]),
    _tool("log_call", "Log a phone call attempt and its outcome -- in the call list, the outcome of the call on "
          "screen ('no answer', 'reached, she'll pay 8000 by Friday').", {
        "account": _ACCOUNT,
        "outcome": {"type": "string", "enum": ["reached", "no_answer", "voicemail", "wrong_number"]},
        "note": {"type": "string", "description": "Short staff note, if they gave one."},
        "promised_amount": {"type": "number", "description": "If they promised to pay on the call: rupees, only as said."},
        "promised_full_amount": {"type": "boolean", "description": "They promised the full amount due."},
        "promised_date": {"type": "string", "description": "YYYY-MM-DD they promised to pay by."},
    }, ["account", "outcome"]),
    _tool("message_borrower", "Send one borrower a general message (Telegram if linked; always on their dashboard).", {
        "account": _ACCOUNT, "message": {"type": "string", "description": _TEXT_FOR_BORROWER},
    }, ["account", "message"]),
    _tool("message_group", "Send the same message to every account in a group.", {
        "group": {"type": "string", "enum": ["overdue", "disputed", "broken_promises", "not_on_telegram"]},
        "message": {"type": "string", "description": _TEXT_FOR_BORROWER},
    }, ["group", "message"]),
    _tool("reply_to_case", "Reply on the borrower's open case thread (their dispute, payment claim or request).", {
        "account": _ACCOUNT, "message": {"type": "string", "description": _TEXT_FOR_BORROWER},
    }, ["account", "message"]),
    _tool("approve_request", "Approve / close the borrower's open request (more time, closure certificate, call-back, "
          "other request) or a system notice.", {"account": _ACCOUNT}, ["account"]),
    _tool("reject_request", "Reject the borrower's open request.", {
        "account": _ACCOUNT, "reason": {"type": "string", "description": "Optional reason. " + _TEXT_FOR_BORROWER},
    }, ["account"]),
    _tool("hand_to_person", "Open a case for a person on the team to look at this account.", {
        "account": _ACCOUNT, "reason": {"type": "string", "description": "Why, for staff, in the operator's words."},
    }, ["account", "reason"]),
    _tool("send_reminders", "Send today's automated payment reminders, to one account or a group.", {
        "account": _ACCOUNT, "group": {"type": "string", "enum": ["all", "overdue", "disputed", "broken_promises"]},
    }),
    _tool("update_account", "Correct an account's phone number, language or risk tier.", {
        "account": _ACCOUNT, "phone_number": {"type": "string", "description": "Digits as said, with +91 if Indian."},
        "language": {"type": "string", "enum": ["hi", "en", "hinglish"]},
        "risk_tier": {"type": "string", "enum": ["low", "medium", "high"]},
    }, ["account"]),
    _tool("mark_messages_handled", "Mark this borrower's messages from the team as dealt with.", {"account": _ACCOUNT},
          ["account"]),
    _tool("daily_briefing", "Start of the day: what needs doing today -- calls, inbox, payment claims, stale cases.",),
    _tool("start_call_list", "Start a guided call list: borrowers to phone, one at a time, most overdue first.", {
        "group": {"type": "string", "enum": ["overdue", "broken_promises"]},
    }),
    _tool("start_inbox", "Start working through the open cases one at a time (or only one kind, e.g. payment claims).", {
        "kind": {"type": "string", "enum": list(CaseKind.__args__)},
    }),
    _tool("workflow_control", "Inside a running call list or inbox: go to the next item, skip this one, go back, "
          "repeat it, or stop.", {"command": {"type": "string", "enum": ["next", "skip", "previous", "stop", "repeat"]}},
          ["command"]),
]

_SYSTEM_PROMPT = (
    "You turn what an operator says to a loan-collections ops dashboard into exactly one function call. The "
    "operator is staff, speaking English, Hindi or Hinglish; their words may be a speech-recognition transcript, "
    "possibly in Devanagari, with small errors -- read through them. Always call exactly one function; if none fits, call cannot_help. Never invent an "
    "amount, date, phone number, name or reason the operator did not say: leave it out and the dashboard will ask. "
    "Questions about numbers or people are answered by the dashboard from its own data -- only pick the function; "
    "you have no data and must not answer yourself. Dates are YYYY-MM-DD, worked out from today's date below."
)


class Plan(BaseModel):
    tool: str
    args: _Args


class NotUnderstood(Exception):
    """The model's choice could not be used (no function, an unknown one, or arguments that don't validate)."""


def parse_plan(tool: str, arguments: dict) -> Plan:
    """A plan from a tool name and its arguments, checked against the menu. Used on the model's answer and on a plan
    the page sends back (a "did you mean" choice) -- both go through the same validation."""
    model = ARGS.get(tool)
    if model is None:
        raise NotUnderstood(f"unknown ability {tool!r}")
    try:
        return Plan(tool=tool, args=model.model_validate(arguments or {}))
    except ValidationError as e:
        raise NotUnderstood(f"{tool}: {e.error_count()} invalid argument(s)") from e


def understand(request_text: str, today: date, on_screen: str) -> Plan:
    """One model call. on_screen describes what the operator is looking at (so 'this account' and 'her' resolve).
    Raises NotUnderstood; lets groq's own errors (rate limit after every key was tried, connection) propagate."""
    completion = _create_completion(
        model=os.environ.get("OPS_VOICE_MODEL") or _DEFAULT_MODEL,  # set but empty, as .env.example leaves it, means the default
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Today is {today:%A} {today.isoformat()}.\nOn screen: {on_screen}.\n"
                f"Operator: {request_text[:MAX_REQUEST_CHARS]}"
            )},
        ],
        tools=MENU,
        tool_choice="required",
        reasoning_effort="low",
    )
    calls = completion.choices[0].message.tool_calls or []
    if not calls:
        raise NotUnderstood("the model called no function")
    call = calls[0]
    if call.function.name in PAGE_ONLY_TOOLS:
        raise NotUnderstood(f"{call.function.name} is asked for by the page, not chosen by the model")
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        raise NotUnderstood(f"{call.function.name}: arguments were not JSON") from e
    if not isinstance(arguments, dict):
        raise NotUnderstood(f"{call.function.name}: arguments were not an object")
    return parse_plan(call.function.name, arguments)
