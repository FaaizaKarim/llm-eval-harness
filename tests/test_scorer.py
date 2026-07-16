"""Tests for failure-mode classification in the scorer."""

from llm_eval.models import FailureMode, Task, TestCase
from llm_eval.scorer import score

TASK = Task(
    task_id="t-add",
    category="algorithms",
    prompt="Write add(a, b).",
    entry_point="add",
    test_cases=(
        TestCase(call="add(2, 3)", expected=5),
        TestCase(call="add(-1, 1)", expected=0),
    ),
)


async def test_correct_solution_passes():
    result = await score("def add(a, b):\n    return a + b", TASK, "test-model")
    assert result.passed
    assert result.failure_mode is FailureMode.NONE
    assert result.tests_passed == 2


async def test_wrong_answer_is_classified():
    result = await score("def add(a, b):\n    return a - b", TASK, "test-model")
    assert not result.passed
    assert result.failure_mode is FailureMode.WRONG_ANSWER
    assert "expected" in result.error_trace


async def test_syntax_error_is_classified():
    result = await score("def add(a, b:\n    return", TASK, "test-model")
    assert result.failure_mode is FailureMode.SYNTAX_ERROR


async def test_runtime_error_is_classified_with_trace():
    result = await score(
        "def add(a, b):\n    return a + b + undefined_name", TASK, "test-model"
    )
    assert result.failure_mode is FailureMode.RUNTIME_ERROR
    assert "NameError" in result.error_trace


async def test_empty_response_is_classified():
    result = await score("", TASK, "test-model")
    assert result.failure_mode is FailureMode.EMPTY_RESPONSE


async def test_timeout_is_classified():
    result = await score(
        "def add(a, b):\n    while True: pass", TASK, "test-model", timeout_s=1.0
    )
    assert result.failure_mode is FailureMode.TIMEOUT
