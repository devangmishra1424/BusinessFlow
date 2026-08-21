"""The single FastMCP server instance every tool module registers against.

Importing businessflow.tools triggers registration of all tool modules onto
this instance (see tools/__init__.py) -- import that, not this file directly,
unless you specifically need the bare, empty server.
"""

from fastmcp import FastMCP

mcp = FastMCP(
    name="businessflow-tools",
    instructions=(
        "Tools for a Hindi-English EMI/loan collections agent. Every tool "
        "operates on synthetic demo accounts only -- no real payment ever "
        "moves. Ground every spoken or acted-on value in a tool result; "
        "never state a balance, date, or amount from memory."
    ),
)
