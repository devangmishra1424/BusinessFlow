"""The tool that hands a conversation off to a human."""

import logging

from businessflow.accounts import store
from businessflow.tools.server import mcp

logger = logging.getLogger(__name__)


@mcp.tool
def escalate_to_human(account_id: str, reason: str) -> dict:
    """Hand this account off to a human agent, with the reason so they don't
    start cold. Used whenever policy requires a human (open dispute,
    repeated broken promises) or the agent is otherwise unsure."""
    store.get_account_or_raise(account_id)  # raises if the account doesn't exist
    escalation_id = store.create_escalation(account_id, reason)

    logger.info("escalation created id=%s account_id=%s reason=%s", escalation_id, account_id, reason)

    return {
        "escalation_id": escalation_id,
        "account_id": account_id,
        "reason": reason,
        "status": "queued_for_human",
    }
