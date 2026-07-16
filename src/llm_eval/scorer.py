"""Scores model-generated code against a task's test cases.

The generated code and a small JSON-reporting test driver are combined
into one script and executed in the sandbox. The driver prints one JSON
line per test case so the parent process can grade without importing
untrusted code into its own interpreter.
"""

from __future__ import annotations

import ast
import json

from .models import EvalResult, FailureMode, Task
from .sandbox import run_python

_DRIVER = """
import json as _json
import traceback as _tb

_results = []
for _call, _expected in _CASES:
    try:
        _actual = eval(_call)
        _results.append({
            "ok": _actual == _expected,
            "call": _call,
            "expected": repr(_expected),
            "actual": repr(_actual),
            "trace": "",
        })
    except Exception:
        _results.append({
            "ok": False,
            "call": _call,
            "expected": repr(_expected),
            "actual": None,
            "trace": _tb.format_exc(),
        })
print("__RESULTS__" + _json.dumps(_results))
"""


def _build_script(code: str, task: Task) -> str:
    cases = [(tc.call, tc.expected) for tc in task.test_cases]
    return f"{code}\n\n_CASES = {cases!r}\n{_DRIVER}"


def _syntax_check(code: str) -> str | None:
    """Return a formatted SyntaxError trace, or None if the code parses."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"


async def score(
    code: str,
    task: Task,
    model_name: str,
    timeout_s: float = 10.0,
    memory_limit_mb: int = 256,
) -> EvalResult:
    """Run one generated solution against a task and classify the outcome."""
    total = len(task.test_cases)

    def fail(mode: FailureMode, trace: str = "", duration: float = 0.0,
             passed_n: int = 0) -> EvalResult:
        return EvalResult(
            task_id=task.task_id, model=model_name, passed=False,
            failure_mode=mode, tests_passed=passed_n, tests_total=total,
            duration_s=duration, error_trace=trace, generated_code=code,
        )

    if not code.strip():
        return fail(FailureMode.EMPTY_RESPONSE)

    if (syntax_trace := _syntax_check(code)) is not None:
        return fail(FailureMode.SYNTAX_ERROR, syntax_trace)

    result = await run_python(
        _build_script(code, task), timeout_s=timeout_s,
        memory_limit_mb=memory_limit_mb,
    )

    if result.timed_out:
        return fail(FailureMode.TIMEOUT, result.stderr, result.duration_s)
    if "MemoryError" in result.stderr:
        return fail(FailureMode.MEMORY_EXCEEDED, result.stderr, result.duration_s)
    if result.exit_code != 0:
        return fail(FailureMode.RUNTIME_ERROR, result.stderr, result.duration_s)

    marker = "__RESULTS__"
    line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith(marker)), None
    )
    if line is None:
        return fail(FailureMode.SANDBOX_ERROR,
                    f"driver produced no results.\nstdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}", result.duration_s)

    cases = json.loads(line[len(marker):])
    passed_n = sum(1 for c in cases if c["ok"])
    first_bad = next((c for c in cases if not c["ok"]), None)

    if passed_n == total:
        return EvalResult(
            task_id=task.task_id, model=model_name, passed=True,
            failure_mode=FailureMode.NONE, tests_passed=passed_n,
            tests_total=total, duration_s=result.duration_s,
            generated_code=code,
        )

    if first_bad and first_bad["trace"]:
        mode, trace = FailureMode.RUNTIME_ERROR, first_bad["trace"]
    else:
        mode = FailureMode.WRONG_ANSWER
        trace = (f"call: {first_bad['call']}\nexpected: {first_bad['expected']}\n"
                 f"actual: {first_bad['actual']}") if first_bad else ""
    return fail(mode, trace, result.duration_s, passed_n)
