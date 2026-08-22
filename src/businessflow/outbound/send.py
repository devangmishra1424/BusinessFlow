"""Stage 3: the "send" step -- a clearly-labeled stub, same convention
as payment_tools.generate_payment_link's synthetic link. This project
has zero real outbound-channel credentials or SDK wired in anywhere
(no Telegram/SMS/WhatsApp/email client, confirmed by a full repo audit)
-- so this logs what WOULD be sent rather than pretending to actually
deliver it. Wiring a real channel is deferred alongside Telegram/voice/
hosting; this is the seam a real implementation would replace.
"""

from businessflow.accounts import store


def send_reminder(account_id: str, kind: str, message: str) -> None:
    store.log_event(account_id, "reminder_sent", {"kind": kind, "message": message})
