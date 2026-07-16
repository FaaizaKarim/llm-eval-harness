"""Tests for process-level sandboxed execution."""

from llm_eval.sandbox import run_python


async def test_runs_simple_code():
    result = await run_python("print('hello')")
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert not result.timed_out


async def test_captures_runtime_error():
    result = await run_python("raise ValueError('boom')")
    assert result.exit_code != 0
    assert "ValueError: boom" in result.stderr


async def test_enforces_timeout():
    result = await run_python("while True: pass", timeout_s=1.0)
    assert result.timed_out
    assert result.duration_s >= 1.0


async def test_environment_is_not_leaked():
    result = await run_python(
        "import os; print(len(os.environ.get('LLM_API_KEY', '')))"
    )
    assert result.stdout.strip() == "0"
