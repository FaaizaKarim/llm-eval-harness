"""Core data models for the evaluation harness.

Every evaluation run produces structured, reproducible records so that
model failure modes can be documented and compared across runs.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field, asdict
from typing import Any


class FailureMode(str, enum.Enum):
    """Taxonomy of ways an LLM-generated solution can fail.

    Keeping failures categorized (rather than a single pass/fail bit)
    is what makes the harness useful for hardening model reasoning.
    """

    NONE = "none"                        # passed all tests
    SYNTAX_ERROR = "syntax_error"        # code does not parse
    RUNTIME_ERROR = "runtime_error"      # raised during execution
    WRONG_ANSWER = "wrong_answer"        # ran fine, output incorrect
    TIMEOUT = "timeout"                  # exceeded the time budget
    MEMORY_EXCEEDED = "memory_exceeded"  # exceeded the memory budget
    EMPTY_RESPONSE = "empty_response"    # model returned no usable code
    SANDBOX_ERROR = "sandbox_error"      # harness-side infrastructure error


@dataclass(frozen=True)
class TestCase:
    """A single input/expected-output check for a task."""

    __test__ = False     # tell pytest this is not a test class

    call: str            # expression to evaluate, e.g. "two_sum([2,7,11,15], 9)"
    expected: Any        # expected return value (JSON-serializable)


@dataclass(frozen=True)
class Task:
    """A coding task presented to the model."""

    task_id: str
    category: str                     # e.g. "algorithms", "async", "debugging"
    prompt: str                       # natural-language problem statement
    entry_point: str                  # function name the model must define
    test_cases: tuple[TestCase, ...]
    difficulty: str = "medium"

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Task":
        return Task(
            task_id=raw["task_id"],
            category=raw["category"],
            prompt=raw["prompt"],
            entry_point=raw["entry_point"],
            difficulty=raw.get("difficulty", "medium"),
            test_cases=tuple(
                TestCase(call=tc["call"], expected=tc["expected"])
                for tc in raw["test_cases"]
            ),
        )


@dataclass
class ExecutionResult:
    """Raw outcome of running code in the sandbox."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    duration_s: float = 0.0


@dataclass
class EvalResult:
    """Structured result of evaluating one task against one model."""

    task_id: str
    model: str
    passed: bool
    failure_mode: FailureMode
    tests_passed: int
    tests_total: int
    duration_s: float
    error_trace: str = ""            # reproducible traceback, if any
    generated_code: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def pass_rate(self) -> float:
        return self.tests_passed / self.tests_total if self.tests_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure_mode"] = self.failure_mode.value
        d["pass_rate"] = round(self.pass_rate, 4)
        return d
