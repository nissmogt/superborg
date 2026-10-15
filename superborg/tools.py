"""Tools the reasoning layer can call: a Python runner and an exact calculator.

run_python() executes code in a separate interpreter process (``sys.executable -I``)
with CPU-time, address-space and file-size limits (where the OS supports them), a
throw-away working directory, no stdin, a scrubbed environment and a wall-clock timeout.

SECURITY NOTE: this is *isolation-light*. The child process runs as the current user,
can read the user's files and can open network connections; the limits only stop
runaway loops, memory blow-ups and output floods. It exists so a local model can
check its own arithmetic on the user's own machine, by the user's choice. Never feed
it code from untrusted third parties, and do not expose it as a network service.
"""
from __future__ import annotations

import ast
import json
import math
import operator
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

__all__ = ["ToolResult", "run_python", "calculator", "CalculatorError", "extract_code"]

MAX_OUTPUT = 4096  # characters of stdout kept (the head; a marker notes the cut)


@dataclass
class ToolResult:
    ok: bool                      # ran to completion without an exception or limit hit
    stdout: str = ""              # printed output (truncated to MAX_OUTPUT chars)
    error: str | None = None      # short error text (last traceback line, "timeout", ...)
    value: Any = None             # `answer`, solution(), or the last expression's value
    elapsed_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def observation(self, limit: int = 1500) -> str:
        """Compact text for feeding the result back to a model."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip()[:limit])
        if self.value is not None and (not parts or str(self.value) not in parts[-1]):
            parts.append(f"answer = {self.value!r}"[:limit])
        if self.error:
            parts.append(f"Error: {self.error}"[:limit])
        return "\n".join(parts) if parts else "(no output)"


# The runner executes inside the child. It keeps the user's prints on the real stdout
# (the parent truncates) and reports the result value through a side file, so code
# that prints a lot cannot corrupt the protocol.
_RUNNER = r'''
import ast, json, math, sys, traceback
src_path, out_path, mem_mb, cpu_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
try:  # limits are set here, not in a preexec_fn, because forking a threaded parent is unsafe
    import resource
    def _setl(name, soft, hard):
        lim = getattr(resource, name, None)
        if lim is None:
            return
        try:
            cur_hard = resource.getrlimit(lim)[1]
            if cur_hard != resource.RLIM_INFINITY:
                soft, hard = min(soft, cur_hard), min(hard, cur_hard)
            resource.setrlimit(lim, (soft, hard))
        except (ValueError, OSError):
            pass
    if mem_mb > 0:
        _setl("RLIMIT_AS", mem_mb << 20, mem_mb << 20)
    _setl("RLIMIT_CPU", cpu_s, cpu_s + 1)
    _setl("RLIMIT_FSIZE", 16 << 20, 16 << 20)  # no filling the disk
    _setl("RLIMIT_CORE", 0, 0)
except ImportError:  # not POSIX: only the parent's wall-clock timeout applies
    pass
with open(src_path, encoding="utf-8") as f:
    src = f.read()
res = {"ok": False, "error": None, "value": None, "kind": None}

def _jsonable(v):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return v if abs(v) < 10**300 else repr(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else repr(v)
    try:
        import fractions, decimal
        if isinstance(v, fractions.Fraction):
            return v.numerator if v.denominator == 1 else float(v)
        if isinstance(v, decimal.Decimal):
            return float(v)
    except Exception:
        pass
    if hasattr(v, "item") and callable(v.item):  # numpy scalars
        try:
            return _jsonable(v.item())
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v][:100]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in list(v.items())[:100]}
    return repr(v)[:1000]

def _short(e):
    line = traceback.format_exception_only(type(e), e)[-1].strip()
    tb = traceback.extract_tb(e.__traceback__)
    lines = [fr.lineno for fr in tb if fr.filename == "<program>"]
    return (f"line {lines[-1]}: " if lines else "") + line

g = {"__name__": "__main__", "__builtins__": __builtins__}
try:
    tree = ast.parse(src, "<program>", "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<program>", "exec"), g)
    if "answer" in g and not callable(g["answer"]):
        res["value"], res["kind"] = _jsonable(g["answer"]), "answer"
    elif callable(g.get("solution")):
        res["value"], res["kind"] = _jsonable(g["solution"]()), "solution"
    if last is not None:
        v = eval(compile(last, "<program>", "eval"), g)
        if res["kind"] is None and v is not None:
            res["value"], res["kind"] = _jsonable(v), "expression"
            print(repr(v))
    res["ok"] = True
except SystemExit as e:
    res["ok"] = e.code in (None, 0)
    if not res["ok"]:
        res["error"] = f"SystemExit({e.code!r})"
except BaseException as e:  # noqa: BLE001 - report everything, including MemoryError
    res["error"] = _short(e)
try:
    sys.stdout.flush()
except Exception:
    pass
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(res, f)
'''


def _child_env(workdir: str) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": workdir, "TMPDIR": workdir,
           "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
           # BLAS libraries reserve big per-thread buffers that trip RLIMIT_AS
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if sys.platform == "win32":  # pragma: no cover
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    return env


def _kill(proc: subprocess.Popen) -> None:
    try:
        if sys.platform != "win32":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def run_python(code: str, timeout: float = 10.0, memory_mb: int = 512, max_output: int = MAX_OUTPUT) -> ToolResult:
    """Run `code` in a fresh, resource-limited Python process and collect its result.

    The value is the variable ``answer`` if the code defines one, else the return value of
    ``solution()`` if defined, else the value of a trailing expression (notebook style).
    Values come back as JSON-compatible Python objects (or their repr).
    """
    t0 = time.perf_counter()
    workdir = tempfile.mkdtemp(prefix="borg-py-")
    try:
        src = os.path.join(workdir, "program.py")
        out = os.path.join(workdir, ".result.json")
        runner = os.path.join(workdir, ".runner.py")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_RUNNER)
        posix = sys.platform != "win32"
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", runner, src, out, str(int(memory_mb or 0)),
                 str(max(1, int(math.ceil(timeout))))],
                cwd=workdir, env=_child_env(workdir), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=posix)
        except OSError as e:
            return ToolResult(ok=False, error=f"cannot start python: {e}", elapsed_s=time.perf_counter() - t0)

        # Drain the pipe in a thread, keeping only the head, so a print loop can neither
        # block the child nor eat the parent's memory.
        chunks: list[bytes] = []
        state = {"kept": 0, "dropped": 0}
        cap = max_output * 4 + 16  # bytes; utf-8 is at most 4 per char

        def drain():
            assert proc.stdout is not None
            while True:
                b = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(65536)
                if not b:
                    break
                room = cap - state["kept"]
                if room > 0:
                    chunks.append(b[:room])
                    state["kept"] += min(room, len(b))
                state["dropped"] += max(0, len(b) - max(room, 0))

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            proc.wait()
        if posix:
            _kill(proc)  # reap anything the program forked into its session
        reader.join(timeout=2.0)
        if not reader.is_alive() and proc.stdout is not None:
            proc.stdout.close()
        text = b"".join(chunks).decode("utf-8", errors="replace")
        truncated = state["dropped"] > 0 or len(text) > max_output
        if len(text) > max_output:
            text = text[:max_output]
        if truncated:
            text += "\n...[output truncated]"

        res: dict = {}
        try:
            with open(out, encoding="utf-8") as f:
                res = json.load(f)
        except (OSError, ValueError):
            res = {}
        elapsed = time.perf_counter() - t0
        rc = proc.returncode
        if timed_out:
            return ToolResult(ok=False, stdout=text, error=f"timeout after {timeout:g}s", elapsed_s=elapsed)
        if not res:
            # killed before reporting: a resource limit (CPU -> SIGXCPU/SIGKILL, memory, ...)
            sig = -rc if rc is not None and rc < 0 else None
            reason = {getattr(signal, "SIGXCPU", -1): "CPU time limit exceeded",
                      getattr(signal, "SIGKILL", -1): "killed (resource limit)",
                      getattr(signal, "SIGXFSZ", -1): "file size limit exceeded",
                      getattr(signal, "SIGSEGV", -1): "crashed (segmentation fault)"}.get(sig)
            tail = text.strip().splitlines()[-1] if text.strip() else ""
            if "MemoryError" in text:
                reason = "MemoryError (memory limit)"
            return ToolResult(ok=False, stdout=text, error=reason or tail or f"exited with status {rc}",
                              elapsed_s=elapsed, extra={"returncode": rc})
        return ToolResult(ok=bool(res.get("ok")), stdout=text, error=res.get("error"), value=res.get("value"),
                          elapsed_s=elapsed, extra={"value_kind": res.get("kind"), "returncode": rc})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def extract_code(text: str) -> str:
    """The Python code in a model reply: the last fenced block that looks like code,
    or the whole text when there is no fence (e.g. when the fence was in the prompt)."""
    import re
    blocks = re.findall(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)(?:```|$)", text, flags=re.S | re.I)
    blocks = [b for b in blocks if b.strip()]
    if blocks:
        return blocks[0].strip("\n") if len(blocks) == 1 else max(blocks, key=len).strip("\n")
    return text.strip("\n")


# ---------------------------------------------------------------------------------------
# Calculator: a whitelist AST evaluator (no names, attributes, calls except math functions)

class CalculatorError(ValueError):
    pass


_MAX_INT_BITS = 4096       # refuse integers bigger than ~1200 decimal digits
_MAX_FACTORIAL = 500


def _check(v):
    if isinstance(v, int) and not isinstance(v, bool) and v.bit_length() > _MAX_INT_BITS:
        raise CalculatorError("result too large")
    if isinstance(v, Fraction) and (v.numerator.bit_length() > _MAX_INT_BITS or
                                    v.denominator.bit_length() > _MAX_INT_BITS):
        raise CalculatorError("result too large")
    if isinstance(v, float) and not math.isfinite(v):
        raise CalculatorError("overflow" if math.isinf(v) else "not a number")
    return v


def _pow(a, b):
    if isinstance(b, Fraction) and b.denominator == 1:
        b = b.numerator
    if isinstance(a, (int, Fraction)) and isinstance(b, int):
        mag = abs(a) if isinstance(a, int) else max(abs(a.numerator), a.denominator)
        if mag > 1 and abs(b) * math.log2(mag) > _MAX_INT_BITS:
            raise CalculatorError("exponent too large")
        return Fraction(a) ** b if b < 0 else a ** b
    a, b = float(a), float(b)
    if abs(b) > 10000 and abs(a) > 1:
        raise CalculatorError("exponent too large")
    try:
        r = a ** b
    except OverflowError as e:
        raise CalculatorError("overflow") from e
    if isinstance(r, complex):
        raise CalculatorError("complex result")
    return r


def _div(a, b):
    if b == 0:
        raise CalculatorError("division by zero")
    if isinstance(a, (int, Fraction)) and isinstance(b, (int, Fraction)):
        return Fraction(a) / Fraction(b)
    return float(a) / float(b)


def _factorial(n):
    if isinstance(n, Fraction) and n.denominator == 1:
        n = n.numerator
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if not isinstance(n, int) or n < 0:
        raise CalculatorError("factorial needs a non-negative integer")
    if n > _MAX_FACTORIAL:
        raise CalculatorError("factorial argument too large")
    return math.factorial(n)


def _intargs(f):
    def g(*args):
        conv = []
        for a in args:
            if isinstance(a, Fraction) and a.denominator == 1:
                a = a.numerator
            if isinstance(a, float) and a.is_integer():
                a = int(a)
            if not isinstance(a, int):
                raise CalculatorError(f"{f.__name__} needs integers")
            if a.bit_length() > _MAX_INT_BITS:
                raise CalculatorError("argument too large")
            conv.append(a)
        if f in (math.comb, math.perm) and conv and conv[0] > 100000:
            raise CalculatorError("argument too large")
        return f(*conv)
    g.__name__ = f.__name__
    return g


def _float_fn(f):
    def g(*args):
        try:
            return f(*[float(a) for a in args])
        except (ValueError, OverflowError) as e:
            raise CalculatorError(f"{f.__name__}: {e}") from e
    return g


_FUNCS = {
    "sqrt": _float_fn(math.sqrt), "exp": _float_fn(math.exp), "log": _float_fn(math.log),
    "ln": _float_fn(math.log), "log10": _float_fn(math.log10), "log2": _float_fn(math.log2),
    "sin": _float_fn(math.sin), "cos": _float_fn(math.cos), "tan": _float_fn(math.tan),
    "asin": _float_fn(math.asin), "acos": _float_fn(math.acos), "atan": _float_fn(math.atan),
    "atan2": _float_fn(math.atan2), "sinh": _float_fn(math.sinh), "cosh": _float_fn(math.cosh),
    "tanh": _float_fn(math.tanh), "degrees": _float_fn(math.degrees), "radians": _float_fn(math.radians),
    "hypot": _float_fn(math.hypot),
    "floor": lambda x: math.floor(x), "ceil": lambda x: math.ceil(x), "trunc": lambda x: math.trunc(x),
    "abs": abs, "round": lambda x, n=0: round(float(x), int(n)) if n else round(x),
    "min": min, "max": max, "pow": lambda a, b: _pow(a, b),
    "factorial": _factorial, "gcd": _intargs(math.gcd), "lcm": _intargs(math.lcm),
    "comb": _intargs(math.comb), "perm": _intargs(math.perm), "isqrt": _intargs(math.isqrt),
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: _div,
           ast.FloorDiv: None, ast.Mod: None, ast.Pow: _pow}


def _eval(node, depth=0):
    if depth > 100:
        raise CalculatorError("expression too deeply nested")
    if isinstance(node, ast.Expression):
        return _eval(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise CalculatorError(f"unsupported constant {v!r}")
        return _check(v)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand, depth + 1)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _eval(node.left, depth + 1), _eval(node.right, depth + 1)
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            if b == 0:
                raise CalculatorError("division by zero")
            return _check(a // b if isinstance(node.op, ast.FloorDiv) else a % b)
        try:
            return _check(_BINOPS[type(node.op)](a, b))
        except OverflowError as e:
            raise CalculatorError("overflow") from e
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise CalculatorError(f"unknown name {node.id!r}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        f = _FUNCS.get(node.func.id)
        if f is None:
            raise CalculatorError(f"unknown function {node.func.id!r}")
        args = [_eval(a, depth + 1) for a in node.args]
        try:
            return _check(f(*args))
        except CalculatorError:
            raise
        except (TypeError, ValueError, OverflowError) as e:
            raise CalculatorError(f"{node.func.id}: {e}") from e
    raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def calculator(expr: str) -> int | float:
    """Evaluate an arithmetic expression exactly where possible.

    Supports numbers, + - * / // % ** (also ^ for power), parentheses, unary minus,
    constants pi/e/tau and common math functions. Integer arithmetic and division are
    exact (fractions); the result is an int when integral, else a float. Exponents and
    factorials are bounded so a single call cannot hang or exhaust memory.
    """
    if not isinstance(expr, str) or len(expr) > 2000:
        raise CalculatorError("expression must be a string of at most 2000 characters")
    s = expr.strip().replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.rstrip("=").strip()
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"syntax error: {e.msg}") from e
    v = _eval(tree)
    if isinstance(v, Fraction):
        v = v.numerator if v.denominator == 1 else float(v)
    if isinstance(v, float) and v.is_integer() and abs(v) < 2 ** 53:
        v = int(v)
    return v
