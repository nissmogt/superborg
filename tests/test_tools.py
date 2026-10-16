"""Tests for superborg.tools: the subprocess Python runner and the safe calculator."""
from __future__ import annotations

import math
import sys
import time

import pytest

from superborg.tools import CalculatorError, calculator, extract_code, run_python

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="resource limits need POSIX")


# ---- run_python -----------------------------------------------------------------------

def test_answer_variable():
    r = run_python("x = 16 - 3 - 4\nanswer = x * 2\n")
    assert r.ok and r.error is None
    assert r.value == 18
    assert r.extra["value_kind"] == "answer"


def test_solution_function_and_print():
    r = run_python("def solution():\n    print('working')\n    return 7 / 2\n")
    assert r.ok and r.value == 3.5
    assert "working" in r.stdout


def test_trailing_expression_value():
    r = run_python("a = 6\na * 7")
    assert r.ok and r.value == 42


def test_last_assignment_fallback():
    r = run_python("pay = 10 * 40\novertime = 12 * 5\ntotal_pay = pay + overtime\n")
    assert r.ok and r.value == 460 and r.extra["value_kind"] == "last_assignment"
    r = run_python("t = 1\nt += 4")
    assert r.value == 5
    r = run_python("def f():\n    return 1\ng = f")  # functions are not results
    assert r.value is None


def test_answer_takes_priority_over_solution():
    r = run_python("answer = 1\ndef solution():\n    return 2\n")
    assert r.value == 1


def test_stdout_only():
    r = run_python("for i in range(3):\n    print(i)")
    assert r.ok and r.value is None
    assert r.stdout.split() == ["0", "1", "2"]


def test_non_json_values_come_back_as_repr():
    r = run_python("from fractions import Fraction\nanswer = Fraction(3, 4)")
    assert r.value == 0.75
    r = run_python("answer = {1, 2}")
    assert r.value == "{1, 2}"
    r = run_python("answer = (1, 'a')")
    assert r.value == [1, "a"]


def test_exception_reports_line_and_type():
    r = run_python("a = 1\nb = 0\nc = a / b\n")
    assert not r.ok
    assert "ZeroDivisionError" in r.error and "line 3" in r.error


def test_syntax_error():
    r = run_python("def f(:\n  pass")
    assert not r.ok and "SyntaxError" in r.error


def test_no_stdin():
    r = run_python("x = input()")
    assert not r.ok and "EOFError" in r.error


def test_sys_exit():
    assert run_python("import sys; sys.exit(0)").ok
    r = run_python("import sys; sys.exit(3)")
    assert not r.ok and "SystemExit" in r.error


def test_timeout_on_infinite_loop():
    t = time.perf_counter()
    r = run_python("while True:\n    pass", timeout=1.5)
    assert not r.ok and "timeout" in r.error
    assert time.perf_counter() - t < 10


def test_timeout_on_sleep():
    r = run_python("import time\ntime.sleep(60)", timeout=1.0)
    assert not r.ok and "timeout" in r.error


@posix_only
def test_memory_limit():
    r = run_python("x = bytearray(2 * 1024 ** 3)\nanswer = len(x)", memory_mb=256)
    assert not r.ok
    assert "MemoryError" in (r.error or "") or "limit" in (r.error or "") or "killed" in (r.error or "")


def test_output_flood_is_truncated():
    r = run_python("while True:\n    print('x' * 1000)", timeout=1.5, max_output=2000)
    assert not r.ok
    assert len(r.stdout) < 2200
    assert "truncated" in r.stdout


def test_large_output_finishing_normally_is_truncated():
    r = run_python("print('y' * 100000)\nanswer = 5", max_output=1000)
    assert r.ok and r.value == 5
    assert len(r.stdout) < 1100 and "truncated" in r.stdout


def test_runs_isolated_in_temp_dir():
    import os
    r = run_python("import os, sys\nanswer = [os.getcwd(), sys.flags.isolated, 'PYTHONPATH' in os.environ]")
    cwd, isolated, has_pp = r.value
    assert isolated == 1 and not has_pp
    assert cwd != os.getcwd()
    assert not os.path.exists(cwd)  # cleaned up afterwards


def test_files_written_stay_in_temp_dir():
    r = run_python("open('out.txt', 'w').write('hi')\nimport os\nanswer = os.path.abspath('out.txt')")
    import os
    assert r.ok and not os.path.exists(r.value)


def test_observation_text():
    r = run_python("print('hello')\nanswer = 3")
    obs = r.observation()
    assert "hello" in obs and "3" in obs
    r = run_python("1/0")
    assert "ZeroDivisionError" in r.observation()


def test_extract_code():
    assert extract_code("Here:\n```python\nx = 1\n```\nDone") == "x = 1"
    assert extract_code("x = 2\nanswer = x\n") == "x = 2\nanswer = x"
    assert extract_code("```py\na=1\n") == "a=1"  # unclosed fence (cut off by a stop string)
    two = "```python\na = 1\n```\ntext\n```python\nb = 2\nc = 3\n```"
    assert extract_code(two) == "b = 2\nc = 3"


# ---- calculator -----------------------------------------------------------------------

@pytest.mark.parametrize("expr,want", [
    ("2+3*4", 14), ("(2+3)*4", 20), ("7/2", 3.5), ("8/2", 4), ("17 // 5", 3), ("17 % 5", 2),
    ("2**10", 1024), ("2^10", 1024), ("-3 + +2", -1), ("1/3 + 1/6", 0.5), ("sqrt(16)", 4),
    ("max(3, 9, 4)", 9), ("abs(-2.5)", 2.5), ("factorial(5)", 120), ("gcd(12, 18)", 6),
    ("round(2.567, 2)", 2.57), ("floor(2.7) + ceil(2.1)", 5), ("3 × 4 ÷ 2", 6), ("10 =", 10),
])
def test_calculator_values(expr, want):
    got = calculator(expr)
    assert got == want and type(got) is type(want)


def test_calculator_exact_fractions():
    assert calculator("0.1 + 0.2") == pytest.approx(0.3)
    assert calculator("(1/3) * 3") == 1
    assert math.isclose(calculator("pi"), math.pi)
    assert calculator("2 ** -2") == 0.25


@pytest.mark.parametrize("expr", [
    "__import__('os').system('true')", "os.getcwd()", "(1).__class__", "[1, 2]", "'a' * 3", "x + 1",
    "lambda: 1", "open('f')", "1 if 1 else 2", "True + 1",
])
def test_calculator_rejects_non_arithmetic(expr):
    with pytest.raises(CalculatorError):
        calculator(expr)


@pytest.mark.parametrize("expr", ["10**10**10", "9**9**9", "2**100000", "factorial(100000)", "1e308 * 10",
                                  "comb(10**9, 5)"])
def test_calculator_bounds_huge_results(expr):
    t = time.perf_counter()
    with pytest.raises(CalculatorError):
        calculator(expr)
    assert time.perf_counter() - t < 1.0


def test_calculator_errors():
    for bad in ("1/0", "5 // 0", "5 % 0", "(1+2", "", "sqrt(-1)", "a" * 3000):
        with pytest.raises(CalculatorError):
            calculator(bad)
