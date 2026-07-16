# LLM Code Evaluation Harness

An **async Python framework** that benchmarks LLM-generated code: it sends coding prompts to any OpenAI-compatible model, executes the generated Python in a **sandboxed subprocess**, scores correctness against unit-test cases, and **classifies and documents failure modes** with reproducible error traces.

Built to answer a practical question: *when a language model writes code, exactly how and why does it fail?*

## Why this exists

Model quality work needs more than a pass rate. This harness produces a structured record per task — failure taxonomy, error trace, generated code, timing — so that prompt-engineering changes and model comparisons can be evaluated with real metrics instead of vibes.

## Architecture

```
tasks.json ──► Harness (asyncio, bounded concurrency)
                 │  prompt
                 ▼
             LLMClient  (OpenAI-compatible HTTP / offline Mock)
                 │  reply ──► extract_code()
                 ▼
             Sandbox (subprocess: -I isolated mode, wall-clock timeout,
                      RLIMIT_AS memory cap, empty env — no key leakage)
                 │  JSON test-driver output
                 ▼
             Scorer ──► EvalResult {failure_mode, error_trace, pass_rate}
                 ▼
             Report (JSON + Markdown, failure-mode histogram)
```

### Failure-mode taxonomy

`syntax_error` · `runtime_error` · `wrong_answer` · `timeout` · `memory_exceeded` · `empty_response` · `sandbox_error`

## Quick start

```bash
pip install -e ".[dev]"

# offline demo (no API key needed) — uses the built-in mock model
llm-eval --tasks tasks/tasks.json --mock

# against a real model
export LLM_API_KEY=sk-...
llm-eval --tasks tasks/tasks.json --model gpt-4o-mini

# against a local model (Ollama, vLLM, LM Studio)
llm-eval --tasks tasks/tasks.json --model llama3 --base-url http://localhost:11434/v1
```

Output: a Markdown report on stdout and full JSON results (including reproducible error traces) in `results.json`. Exit code is non-zero if any task failed, so it slots into CI.

## Writing your own task suite

Tasks are plain JSON — prompt, entry point, and test cases:

```json
{
  "task_id": "ds-001-two-sum",
  "category": "data-structures",
  "prompt": "Write a Python function `two_sum(nums, target)` ...",
  "entry_point": "two_sum",
  "test_cases": [
    { "call": "two_sum([2,7,11,15], 9)", "expected": [0, 1] }
  ]
}
```

The bundled suite covers algorithms, data structures (hash maps, LRU cache), string manipulation, and debugging tasks (e.g., the mutable-default-argument bug).

## Security model

Generated code is untrusted. It runs in a **separate process** with Python's isolated mode (`-I`), a hard timeout, an address-space limit, an empty environment (API keys never reach the sandbox), and a throwaway working directory. For hostile inputs, run the harness inside a container for defense in depth.

## Tests

```bash
pytest -v
```

15 tests cover the sandbox (timeouts, env isolation, error capture), the scorer (every failure mode), and the full pipeline with a mock model — the suite runs offline, no API key required.

## Tech

Python 3.10+ · asyncio · httpx · dataclasses · pytest / pytest-asyncio · GitHub Actions CI
