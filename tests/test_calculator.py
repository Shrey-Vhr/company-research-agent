import pytest

from research_agent.tools.calculator import calculate, safe_eval


def test_valid_expression():
    # TCS revenue growth FY2023-24 -> FY2024-25
    assert safe_eval("(255324 - 240893) / 240893 * 100") == pytest.approx(
        5.99063, rel=1e-5
    )


def test_decimals():
    assert safe_eval("1.5 * (2 + 0.25)") == 3.375
    assert safe_eval(".5 + 1") == 1.5


def test_divide_by_zero_is_a_clear_error():
    with pytest.raises(ValueError, match="Division by zero"):
        safe_eval("1 / 0")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "__import__('os').system('echo hacked')",
        "2**100000",
        "2 ** 100000",
        "7 // 2",
        "open('secrets.txt')",
        "().__class__",
        "abs(-3)",
    ],
)
def test_malicious_or_unsupported_input_is_rejected(expression, monkeypatch):
    # If anything slipped through and ran a shell command, this would record it.
    calls = []
    monkeypatch.setattr("os.system", lambda cmd: calls.append(cmd))

    with pytest.raises(ValueError):
        safe_eval(expression)
    assert calls == []


def test_power_is_rejected_before_it_is_computed():
    # The tree walk stops at the ** node, so 2**100000 is never calculated.
    with pytest.raises(ValueError, match="Pow"):
        safe_eval("2**100000")


def test_calculate_tool_returns_error_text_instead_of_raising():
    assert calculate.invoke({"expression": "62293 / 255324 * 100"}).startswith("24.39")
    assert calculate.invoke({"expression": "__import__('os')"}).startswith("Error:")


def test_commas_in_numbers_are_rejected():
    with pytest.raises(ValueError):
        safe_eval("2,55,324")
