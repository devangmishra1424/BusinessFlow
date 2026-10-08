"""Stage 2 of proactive outbound: turns an OutboundReminder (decide.py's
pure "who needs a nudge" output) into the actual message text, plus a
second, unrelated one-off composer for the ops dashboard's clarification-
request feature (draft_clarification_message) -- both make a real Groq
call and both are strictly grounded in real account facts, never an
invented amount/date/claim, but they're otherwise independent: one drafts
a proactive reminder to a borrower who hasn't asked for anything, the
other drafts a reply to an ops-initiated concern about an account's flags.
"""

from businessflow.accounts.dues import reminder_amounts
from businessflow.accounts.models import Account
from businessflow.agent.client import MODEL, groq_client
from businessflow.outbound.decide import OutboundReminder

# Kept local rather than reusing agent/client.py's _LANGUAGE_INSTRUCTIONS --
# that dict only has "en"/"hi" (a direct KeyError on "hinglish", since the
# live conversational agent handles Hinglish differently), but every
# seeded demo account's language_preference is one of "hi"/"en"/"hinglish"
# and a reminder has to be composable for all three without crashing.
_LANGUAGE_INSTRUCTIONS = {
    "en": "Reply in plain English.",
    "hi": "Reply in Hindi, written entirely in Devanagari script (e.g. आपके), never Romanized.",
    "hinglish": "Reply in Hinglish -- natural Hindi-English code-switching, Latin script.",
}

_TONE_INSTRUCTIONS = {
    "empathetic": "Tone: Warm, empathetic, and supportive. Emphasize that we are here to help if they face any difficulty.",
    "neutral_reminder": "Tone: Direct, professional, polite, and neutral. Focus on clear payment instructions.",
    "firm_urgent": "Tone: Firm, clear, and urgent. Highlight the importance of immediate payment to avoid further delinquency.",
    "final_notice": (
        "Tone: Very firm, formal, and urgent. State clearly that this is a critical notice requiring immediate "
        "payment or direct contact -- but never threaten, and never mention any consequence (legal action, "
        "reporting, visits) that isn't in the facts below."
    ),
}

_REMINDER_SYSTEM_PROMPT = (
    "You draft a short, professional payment reminder from a loan servicer "
    "to a borrower. Be direct and respectful, never threatening and never "
    "abusive, whatever the tone below asks for. Reference "
    "ONLY the account facts given below -- never invent an amount, date, or "
    "claim that wasn't given. Keep it to 1-3 short sentences, suitable for "
    "an SMS/Telegram message. {tone_instruction} {language_instruction} Output only the "
    "message text itself: no preamble, no quotation marks, no signature block."
)


def compose_message(account: Account, reminder: OutboundReminder, risk_profile=None) -> str:
    """One real Groq call per reminder. reminder.kind is "heads_up" (EMI
    due in reminder.days days), "due_now" (due today, or within the
    grace period -- reminder.days days past due, no late fee yet), or
    "follow_up" (reminder.days days past due, already beyond the grace
    period -- see decide.py). Grounded strictly in the account's real
    fields; the caller (outbound/run.py) is responsible for send.py's own
    real-vs-logged delivery split, not this function.

    Optionally accepts a risk_profile from outbound/risk_engine.py to dynamically
    adjust tone (empathetic, neutral, firm, final notice)."""
    # The same amounts the payment link attached to this reminder is minted
    # for (outbound/run.py): what's still due this cycle (EMI less any
    # credit), plus the late fee once a follow_up says it applies -- never
    # the bare emi_amount, which ignored credit and the fee.
    emi, late_fee, total = reminder_amounts(account, reminder.kind)
    emi_str = f"{emi:,.0f}"
    if reminder.kind == "heads_up":
        situation = f"Their EMI of {emi_str} rupees is due in {reminder.days} day(s), on {account.emi_due_date.isoformat()}."
    elif reminder.kind == "due_now":
        situation = (
            f"Their EMI of {emi_str} rupees is due now (day {reminder.days} of the grace period, no late fee yet)."
        )
    else:
        situation = (
            f"Their EMI of {emi_str} rupees is now {reminder.days} day(s) past due, and a late fee of "
            f"{late_fee:,.0f} rupees applies -- {total:,.0f} rupees in total is due now."
        )

    facts = (
        f"Borrower: {account.borrower_name}.\n"
        f"{situation}\n"
        f"Ask them to pay, or contact us if there's a problem."
    )
    tone_str = _TONE_INSTRUCTIONS.get(getattr(risk_profile, "recommended_tone", "neutral_reminder"), "")
    system_prompt = _REMINDER_SYSTEM_PROMPT.format(
        tone_instruction=tone_str,
        language_instruction=_LANGUAGE_INSTRUCTIONS.get(account.language_preference, _LANGUAGE_INSTRUCTIONS["en"])
    )
    completion = groq_client().chat.completions.create(
        model=MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": facts},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


# ============================================================
# Ops dashboard: clarification-request wording assist (unrelated to the
# reminder composer above -- see module docstring).
# ============================================================

# Grounded strictly in the real flag reasons and the operator's own note --
# the same "never invent a fact that wasn't given" discipline
# rag/extraction.py's interest-rate extraction already applies to a
# different one-off Groq call, applied here to message drafting instead.
_CLARIFICATION_SYSTEM_PROMPT = (
    "You draft a short, professional message from a loan officer to a "
    "borrower, for a loan-servicing collections context. Be direct, "
    "respectful, and firm -- never threatening, never informal. Reference "
    "ONLY the account facts and the loan officer's note given to you below "
    "-- never invent an amount, date, or claim that wasn't given. Keep it "
    "to 2-4 short sentences, suitable for a Telegram message. End with a "
    "clear, concrete ask (e.g. contact us, make a payment, explain the "
    "situation) -- never a vague sign-off. Output only the message text "
    "itself: no preamble, no quotation marks, no signature block."
)


def draft_clarification_message(
    borrower_name: str, business_name: str, flag_reasons: list[str], operator_note: str
) -> str:
    """One real Groq call. Returns the raw draft text for the caller to
    hand back to the operator for review/editing -- never sent or logged
    from here (see ops/api.py's draft endpoint and outbound/send.py's
    notify_clarification_request for the real send step)."""
    facts = (
        f"Borrower: {borrower_name} ({business_name}).\n"
        f"Current account flags: {'; '.join(flag_reasons) if flag_reasons else 'none currently.'}\n"
        f"Loan officer's note: {operator_note}"
    )
    completion = groq_client().chat.completions.create(
        model=MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": _CLARIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": facts},
        ],
    )
    return (completion.choices[0].message.content or "").strip()
