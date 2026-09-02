"""Unit tests for payment_tools.compute -- pure logic, no DB or LLM,
deliberately not gated behind DATABASE_URL (see test_grounding.py for the
same convention). Exists so the model never has to do arithmetic in its
own reply text -- see the real guardrail_failed events that motivated it,
documented in payment_tools.py's own comment above compute().
"""

import pytest

from businessflow.tools.payment_tools import compute


def test_computes_a_percentage_of_a_real_balance():
    # The exact real case found live: a 5% settlement discount on a real,
    # grounded outstanding balance.
    result = compute("1084741.92 * 0.95")

    assert result == {"expression": "1084741.92 * 0.95", "result": 1030504.82}


def test_computes_a_difference():
    result = compute("1084741.92 - 1030504.82")

    assert result["result"] == 54237.1


def test_computes_a_sum():
    result = compute("12500 + 12500 + 12500")

    assert result["result"] == 37500.0


def test_supports_parentheses_and_division():
    result = compute("(250000 - 12500 * 3) / 12")

    assert result["result"] == round((250000 - 12500 * 3) / 12, 2)


def test_rounds_the_result_to_2_decimal_places():
    result = compute("100 / 3")

    assert result["result"] == 33.33


def test_rejects_a_name_lookup():
    # Not a code execution tool -- only plain arithmetic on literal
    # numbers is ever a reachable branch in the AST walk.
    with pytest.raises(ValueError):
        compute("os.environ")


def test_rejects_a_function_call():
    with pytest.raises(ValueError):
        compute("__import__('os').system('echo hi')")


def test_rejects_division_by_zero():
    with pytest.raises(ValueError):
        compute("100 / 0")


def test_rejects_a_non_arithmetic_string():
    with pytest.raises(ValueError):
        compute("please compute 5 percent of my balance")
