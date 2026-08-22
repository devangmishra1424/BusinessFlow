"""Stage 2: generates the actual outbound message text for one
reminder, reusing the same Groq model as the live agent (agent/
client.py) rather than a separate templating system -- grounded in the
real account facts handed to it, same "never invent a fact" discipline
as the rest of this project.
"""

from businessflow.accounts.models import Account
from businessflow.agent.client import MODEL, client
from businessflow.outbound.decide import OutboundReminder

_SYSTEM_PROMPT = """You write a short, warm, single outbound reminder message (SMS-length, 2-3 sentences \
max) from an Indian SMB lender's collections agent to a borrower. Use ONLY the real fact given -- never \
invent an amount, date, or anything else. Write in {language_name}. Don't open with "Dear customer" -- \
write like a real short text message, not a formal letter."""

_LANGUAGE_NAMES = {"en": "English", "hi": "Hindi, written entirely in Devanagari script"}


def compose_message(account: Account, reminder: OutboundReminder) -> str:
    if reminder.kind == "heads_up":
        fact = f"Their EMI of ₹{account.emi_amount:,.0f} is due in {reminder.days} day(s), on {account.emi_due_date.isoformat()}."
    else:
        fact = f"Their EMI of ₹{account.emi_amount:,.0f} was due on {account.emi_due_date.isoformat()}, now {reminder.days} day(s) past due."

    language_name = _LANGUAGE_NAMES.get(account.language_preference, "English")
    completion = client().chat.completions.create(
        model=MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT.format(language_name=language_name)},
            {"role": "user", "content": f"Borrower's name: {account.borrower_name}. Fact: {fact} Write the message now."},
        ],
    )
    return (completion.choices[0].message.content or "").strip()
