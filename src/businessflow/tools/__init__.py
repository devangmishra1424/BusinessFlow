"""Importing this package registers every tool onto the shared FastMCP
server instance. Import `mcp` from here, not from tools.server directly,
unless you specifically want the registration side effects skipped."""

from businessflow.tools.server import mcp
from businessflow.tools import (  # noqa: F401 -- imported for registration side effects
    account_tools,
    payment_tools,
    policy_tools,
    escalation_tools,
)

__all__ = ["mcp"]
