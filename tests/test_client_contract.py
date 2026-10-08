"""Contract test: nothing may use the return value of agent.client.client()
as if it were a bare Groq client.

client() returns a (client, key_index) PAIR -- the agent loop needs the index
to rotate keys safely. When that changed, `client().chat.completions...` kept
working in the five other modules that call it ... until it didn't: they
raised AttributeError ('tuple' object has no attribute 'chat') on first real
use. Nothing noticed, because the tests for those modules replaced `client`
with a one-line fake (so the real return shape was never exercised) and the
tests that do hit a real Groq are skipped in CI, which has no key.

This scans the source instead: a purely static check, so it needs no key, no
network and no fake. Everything that just wants a client must call
groq_client().
"""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCANNED_DIRS = ("src", "eval", "scripts")


def _client_call_attribute_uses(source: str) -> list[int]:
    """Line numbers where the result of a call to something named `client`
    is immediately used via attribute access, e.g. client().chat or
    client_module.client().api_key."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call)):
            continue
        func = node.value.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name == "client":
            lines.append(node.lineno)
    return lines


def test_the_detector_flags_the_bad_shape_and_allows_the_good_ones():
    # Guards against this test passing vacuously.
    assert _client_call_attribute_uses("x = client().chat.completions") == [1]
    assert _client_call_attribute_uses("x = client_module.client().api_key") == [1]
    assert _client_call_attribute_uses("x, i = client()") == []
    assert _client_call_attribute_uses("x = client()[0].chat") == []
    assert _client_call_attribute_uses("x = groq_client().chat") == []
    assert _client_call_attribute_uses("x = client(fail_fast=True)") == []


def test_no_module_uses_client_result_as_a_bare_groq_client():
    offenders = []
    for directory in _SCANNED_DIRS:
        for path in (_ROOT / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for line in _client_call_attribute_uses(path.read_text(encoding="utf-8")):
                offenders.append(f"{path.relative_to(_ROOT)}:{line}")

    assert not offenders, (
        "client() returns a (client, key_index) pair. Use groq_client() when you only need the "
        "client, or unpack the pair. Offending uses: " + ", ".join(offenders)
    )
