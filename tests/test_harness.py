"""End-to-end tests: mock model -> harness -> report."""

from pathlib import Path

from llm_eval.client import MockClient, extract_code
from llm_eval.harness import Harness, load_tasks
from llm_eval.models import FailureMode
from llm_eval.report import summarize, to_markdown

TASKS_FILE = Path(__file__).parent.parent / "tasks" / "tasks.json"


def test_extract_code_strips_markdown_fences():
    reply = "Here you go:\n```python\ndef f():\n    return 1\n```\nEnjoy!"
    assert extract_code(reply) == "def f():\n    return 1"


def test_extract_code_passes_through_bare_code():
    assert extract_code("def f(): return 1") == "def f(): return 1"


def test_load_tasks_parses_suite():
    tasks = load_tasks(TASKS_FILE)
    assert len(tasks) >= 5
    assert all(t.test_cases for t in tasks)


async def test_full_run_with_mock_model():
    tasks = load_tasks(TASKS_FILE)[:2]  # add + palindrome
    client = MockClient(responses={
        "sum of two integers": "```python\ndef add(a, b):\n    return a + b\n```",
        "palindrome": (
            "```python\ndef is_palindrome(s):\n"
            "    s = ''.join(c.lower() for c in s if c.isalnum())\n"
            "    return s == s[::-1]\n```"
        ),
    })
    harness = Harness(client, max_concurrency=2)
    results = await harness.run(tasks)

    assert [r.passed for r in results] == [True, True]
    assert len(client.calls) == 2

    summary = summarize(results)
    assert summary["pass_rate"] == 1.0

    md = to_markdown(results)
    assert "Evaluation report" in md


async def test_generation_failure_becomes_result_not_exception():
    class ExplodingClient(MockClient):
        async def generate(self, prompt: str) -> str:
            raise ConnectionError("provider down")

    tasks = load_tasks(TASKS_FILE)[:1]
    results = await Harness(ExplodingClient()).run(tasks)
    assert results[0].failure_mode is FailureMode.SANDBOX_ERROR
    assert "provider down" in results[0].error_trace
